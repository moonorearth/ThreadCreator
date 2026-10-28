import os
import io
import discord
from discord import app_commands
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Response
from fastapi.responses import FileResponse
import uvicorn
from threading import Thread
import aiohttp

# Import our advanced image processor from thumb_processor.py
from thumb_processor import process_image_bytes

# --- 1. FASTAPI IMAGE PROCESSING API & WEB UI ---
app = FastAPI()

API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "your-secret-key")

def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    if token != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/", response_class=FileResponse)
def serve_ui():
    """Serves the web user interface index.html at root url"""
    return FileResponse("index.html")

@app.post("/api/thumb")
async def generate_thumbnail(file: UploadFile = File(...), authorization: str = Header(None)):
    verify_token(authorization)
    try:
        contents = await file.read()
        result_bytes = process_image_bytes(contents)
        return Response(content=result_bytes, media_type="image/png")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

# --- 2. DISCORD BOT CONFIGURATION ---
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp')
DEFAULT_FORUM_ID = 1542780525205389372

class UnifiedBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()
        print(f"Logged in as {self.user} (ID: {self.user.id})")

client = UnifiedBot()

@client.tree.command(name="ping", description="Replies with Pong!")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("🏓 Pong! Bot is working.")

@client.tree.command(name="thumb", description="Upload an image to generate your advanced 1024x1024 face thumbnail.")
@app_commands.describe(image="Upload your image (PNG, JPG, WEBP, etc.)")
async def thumb(interaction: discord.Interaction, image: discord.Attachment):
    filename_lower = image.filename.lower()
    if not filename_lower.endswith(ALLOWED_EXTENSIONS):
        await interaction.response.send_message("❌ Invalid file type! Please upload a valid image file.", ephemeral=True)
        return

    await interaction.response.defer(thinking=True)

    try:
        last_dot = image.filename.rfind('.')
        base_name = image.filename[:last_dot] if last_dot != -1 else image.filename
        base_name = base_name.replace('_', ' ')  # Replaces underscores with spaces
        output_filename = f"{base_name}.png"

        image_bytes = await image.read()
        port = int(os.environ.get("PORT", 8080))
        local_api_url = f"http://127.0.0.1:{port}/api/thumb"
        headers = {"Authorization": f"Bearer {API_SECRET_KEY}"}

        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field('file', image_bytes, filename=image.filename, content_type=image.content_type or 'application/octet-stream')
            
            async with session.post(local_api_url, data=form, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    await interaction.followup.send(f"❌ Processing Error ({response.status}): {error_text}")
                    return

                result_bytes = await response.read()
                discord_file = discord.File(fp=io.BytesIO(result_bytes), filename=output_filename)
                await interaction.followup.send(content=f"Thumbnail for `{output_filename}`", file=discord_file)

    except Exception as e:
        await interaction.followup.send(f"❌ An unexpected error occurred: `{str(e)}`")


# --- 3. AUTOCOMPLETE FUNCTIONS ---

async def title_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    matching_titles = []
    search_query = current.lower()

    # Scan server channels & forums for existing thread titles
    if interaction.guild:
        for guild_channel in interaction.guild.channels:
            if isinstance(guild_channel, (discord.ForumChannel, discord.TextChannel)):
                for t in guild_channel.threads:
                    if search_query in t.name.lower():
                        choice_name = f"⚠️ ALREADY EXISTS: {t.name}"
                        matching_titles.append(app_commands.Choice(name=choice_name, value=t.name))

    # If no existing matches or empty input, show hint
    if not matching_titles and current:
        matching_titles.append(app_commands.Choice(name=f"✨ Create new thread: {current}", value=current))

    return matching_titles[:25]


async def tag_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    channel = interaction.channel
    if not isinstance(channel, discord.ForumChannel) and interaction.guild:
        channel = interaction.guild.get_channel(DEFAULT_FORUM_ID)

    if isinstance(channel, discord.ForumChannel) and channel.available_tags:
        tags = [t.name for t in channel.available_tags]
    else:
        tags = [
            "Instagram", "Actress", "Actor", "Singer", "Twitter", "Youtuber",
            "Influencer", "Athlete", "Streamer", "Model", "Cosplayer", "Comedian",
            "Asian", "Adult-ContentCreator", "Podcast-ASMR", "TV-Host", "Activist", "Other"
        ]

    matching_tags = [
        app_commands.Choice(name=tag, value=tag)
        for tag in tags if current.lower() in tag.lower()
    ]
    return matching_tags[:25]


# --- 4. CREATETHREAD COMMAND ---

@client.tree.command(name="createthread", description="Create a media thread with an auto-cropped face thumbnail.")
@app_commands.describe(
    image="Upload the face photo",
    title="Thread title (e.g., Tom Holland)",
    tag="Select or type a tag",
    forum="Optional: Choose a specific forum channel (defaults to main forum)"
)
@app_commands.autocomplete(title=title_autocomplete)
@app_commands.autocomplete(tag=tag_autocomplete)
async def createthread(
    interaction: discord.Interaction, 
    image: discord.Attachment, 
    title: str, 
    tag: str = None, 
    forum: discord.ForumChannel = None
):
    await interaction.response.defer(thinking=True)

    try:
        # 1. DETERMINE TARGET FORUM CHANNEL
        if forum:
            target_channel = forum
        else:
            target_channel = interaction.guild.get_channel(DEFAULT_FORUM_ID) if interaction.guild else None

        if not target_channel or not isinstance(target_channel, discord.ForumChannel):
            await interaction.followup.send("❌ Error: Target forum channel is invalid or not a Forum channel!")
            return

        # 2. DUPLICATE CHECK ACROSS THE SERVER
        existing_thread = None
        for guild_channel in interaction.guild.channels:
            if isinstance(guild_channel, (discord.ForumChannel, discord.TextChannel)):
                for t in guild_channel.threads:
                    if t.name.lower() == title.lower():
                        existing_thread = t
                        break
            if existing_thread:
                break

        if existing_thread:
            await interaction.followup.send(
                f"⚠️ Thread already exists for **{title}**. Here is the thread link: {existing_thread.jump_url}"
            )
            return

        # 3. VALIDATE EXTENSION
        filename_lower = image.filename.lower()
        if not filename_lower.endswith(ALLOWED_EXTENSIONS):
            await interaction.followup.send("❌ Invalid file type! Please upload a valid image file.")
            return

        # 4. PROCESS THUMBNAIL VIA LOCAL API
        image_bytes = await image.read()
        port = int(os.environ.get("PORT", 8080))
        local_api_url = f"http://127.0.0.1:{port}/api/thumb"
        headers = {"Authorization": f"Bearer {API_SECRET_KEY}"}

        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field('file', image_bytes, filename=image.filename, content_type=image.content_type or 'application/octet-stream')
            
            async with session.post(local_api_url, data=form, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    await interaction.followup.send(f"❌ Processing Error ({response.status}): {error_text}")
                    return

                result_bytes = await response.read()

        # 5. PREPARE RENAMED FILE
        output_filename = f"{title}.png"
        discord_file = discord.File(fp=io.BytesIO(result_bytes), filename=output_filename)

        # 6. CREATE THREAD IN THE TARGET FORUM
        description_text = f"Media for {title}..."
        selected_tag = None
        if tag:
            selected_tag = discord.utils.get(target_channel.available_tags, name=tag)
        tags_to_use = [selected_tag] if selected_tag else []
        
        thread_message = await target_channel.create_thread(
            name=title,
            content=description_text,
            file=discord_file,
            applied_tags=tags_to_use
        )
        created_thread = thread_message.thread

        await interaction.followup.send(f"✅ Successfully created thread for **{title}** in {target_channel.mention}! Link: {created_thread.jump_url}")

    except Exception as e:
        await interaction.followup.send(f"❌ An unexpected error occurred: `{str(e)}`")


# --- 5. RUN BOTH CONCURRENTLY ---
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        print("❌ Error: DISCORD_BOT_TOKEN is missing!")
        exit(1)

    web_thread = Thread(target=run_web_server)
    web_thread.daemon = True
    web_thread.start()

    client.run(DISCORD_BOT_TOKEN)
