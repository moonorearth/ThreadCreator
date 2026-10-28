import os
import cv2
import numpy as np

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")

CANVAS_SIZE = 1024        # Final output square size (1024x1024)
DETECTION_SIZE = 1024     # Standardized size for robust face detection
BLUR_KERNEL_SIZE = (99, 99) # Background blur intensity

def get_face_detector(width, height):
    """Initializes OpenCV's built-in YuNet Face Detector."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file missing at: {MODEL_PATH}")
    return cv2.FaceDetectorYN.create(
        model=MODEL_PATH,
        config="",
        input_size=(width, height),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )

def process_image_bytes(image_bytes: bytes) -> bytes:
    """Advanced pipeline: Scale-invariant detection -> Landmark mapping ->

    Proportional framing -> Alpha-aware background blur fill -> 1024x1024 canvas.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Invalid image file or decoding failed.")

    # Standardize to 4-channel BGRA for transparency support
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    orig_h, orig_w = image.shape[:2]

    # --- STEP 1: SCALE-INVARIANT DETECTION ON RESIZED PREVIEW ---
    scale_factor = DETECTION_SIZE / max(orig_h, orig_w)
    det_w = int(orig_w * scale_factor)
    det_h = int(orig_h * scale_factor)

    bgr_for_det = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    det_resized = cv2.resize(bgr_for_det, (det_w, det_h), interpolation=cv2.INTER_AREA)

    detector = get_face_detector(det_w, det_h)
    detector.setInputSize((det_w, det_h))
    
    _, faces = detector.detect(det_resized)

    if faces is None or len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Select face with highest confidence score
    best_face = max(faces, key=lambda f: f[14])

    # --- STEP 2: MAP LANDMARKS BACK TO ORIGINAL FULL-RESOLUTION ---
    inv_scale = 1.0 / scale_factor
    rx, ry = float(best_face[4]) * inv_scale, float(best_face[5]) * inv_scale
    lx, ly = float(best_face[6]) * inv_scale, float(best_face[7]) * inv_scale

    eye_center_x = (rx + lx) / 2.0
    eye_center_y = (ry + ly) / 2.0
    eye_distance = np.hypot(lx - rx, ly - ry)

    # --- STEP 3: PROPORTIONAL HEAD FRAMING ---
    crop_dim = int(eye_distance * 5.0)
    
    x1 = int(eye_center_x - (crop_dim / 2))
    y1 = int(eye_center_y - (crop_dim * 0.42))
    x2 = int(x1 + crop_dim)
    y2 = int(y1 + crop_dim)

    # --- STEP 4: LOSSLESS CROP & ALPHA-AWARE BACKGROUND BLUR FILL ---
    blurred_bg = cv2.GaussianBlur(bgr_for_det, BLUR_KERNEL_SIZE, 30)
    crop_canvas = np.zeros((crop_dim, crop_dim, 4), dtype=np.uint8)
    
    pad_margin = crop_dim
    padded_blurred_bg = cv2.copyMakeBorder(
        blurred_bg, 
        top=pad_margin, bottom=pad_margin, left=pad_margin, right=pad_margin, 
        borderType=cv2.BORDER_REPLICATE
    )

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(orig_w, x2)
    src_y2 = min(orig_h, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        crop_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    bg_slice_x1 = x1 + pad_margin
    bg_slice_y1 = y1 + pad_margin
    bg_slice_x2 = bg_slice_x1 + crop_dim
    bg_slice_y2 = bg_slice_y1 + crop_dim

    bg_crop_canvas = padded_blurred_bg[bg_slice_y1:bg_slice_y2, bg_slice_x1:bg_slice_x2]

    # --- STEP 5: BLENDING & FINAL RESIZE ---
    fg_rgb = crop_canvas[:, :, :3].astype(float)
    fg_alpha = (crop_canvas[:, :, 3].astype(float) / 255.0)[:, :, np.newaxis]

    composite_2k = (fg_rgb * fg_alpha) + (bg_crop_canvas.astype(float) * (1.0 - fg_alpha))
    final_canvas = cv2.resize(composite_2k.astype(np.uint8), (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_AREA)

    success, encoded_image = cv2.imencode('.png', final_canvas)
    if not success:
        raise RuntimeError("Image encoding failed.")
        
    return encoded_image.tobytes()
