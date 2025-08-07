import os
import time
import uuid
import secrets  # <-- NEW: Import secrets for secure comparison
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from typing import Optional

# --- Configuration ---
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)

    # --- NEW: Get our own API key from environment variables ---
    API_KEY = os.getenv("MY_API_KEY")
    if not API_KEY:
        raise ValueError("MY_API_KEY environment variable for securing the API is not set.")

except Exception as e:
    print(f"Error on startup: {e}")

# --- NEW: Security Dependency ---
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """A dependency to verify the X-API-Key header against our secret."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header is missing")
    # Use a constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API Key")

# --- App Initialization ---
app = FastAPI(
    title="Video Analysis API",
    description="An API to analyze video content using Gemini 1.5 Flash.",
    # Apply the security dependency to all routes in the app
    dependencies=[Depends(verify_api_key)]
)

# Define File Size Limit & Model
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    """
    Accepts a video file, uploads it to Gemini for processing,
    and returns a textual analysis.
    This endpoint is now protected by an API key.
    """
    # The validation logic below only runs if verify_api_key succeeds.
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the limit of {MAX_FILE_SIZE_MB} MB."
        )
    # ... (rest of the function is the same as before)
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    temp_file_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    gemini_file = None
    try:
        print(f"Uploading file '{file.filename}' to Gemini API...")
        gemini_file = genai.upload_file(path=temp_file_path)
        while gemini_file.state.name == "PROCESSING":
            time.sleep(10)
            gemini_file = genai.get_file(gemini_file.name)
        if gemini_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail=f"Gemini API video processing failed.")
        prompt = "Transcribe the audio from this video, giving timestamps for salient events. Also provide detailed visual descriptions of what is happening throughout the video."
        response = model.generate_content([prompt, gemini_file], request_options={'timeout': 600})
        return JSONResponse(content={"analysis": response.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if gemini_file:
            genai.delete_file(gemini_file.name)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)