import os
import time
import uuid
import secrets
import re
import google.generativeai as genai  # <-- THIS LINE WAS MISSING
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from typing import Optional

# --- Configuration ---
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)

    API_KEY = os.getenv("MY_API_KEY")
    if not API_KEY:
        raise ValueError("MY_API_KEY environment variable for securing the API is not set.")

except Exception as e:
    print(f"Error on startup: {e}")
    # In case of a startup error, we should exit so the container restart loop is clear.
    raise e

# --- Security Dependency ---
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header is missing")
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API Key")

# --- App Initialization ---
app = FastAPI(
    title="Video Analysis API",
    description="An API to analyze video content using Gemini 1.5 Flash.",
    dependencies=[Depends(verify_api_key)]
)

# Define File Size Limit & Model
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# --- Endpoint to Stream Large Files ---
@app.post("/analyze/")
async def analyze_video(request: Request):
    """
    Accepts a video file as a stream to handle large uploads,
    and returns a textual analysis.
    """
    content_type = request.headers.get('content-type')
    if not content_type or 'multipart/form-data' not in content_type:
        raise HTTPException(status_code=400, detail="Invalid content type, please upload a form-data file.")

    # Manually stream the file to disk
    file_name = f"{uuid.uuid4()}.mp4" # Default filename
    temp_file_path = f"/tmp/{file_name}"
    current_size = 0

    try:
        with open(temp_file_path, "wb") as buffer:
            async for chunk in request.stream():
                current_size += len(chunk)
                if current_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size exceeds the application limit of {MAX_FILE_SIZE_MB} MB."
                    )
                buffer.write(chunk)
    except HTTPException as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise e
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error streaming file: {str(e)}")

    # The rest of the logic is the same
    gemini_file = None
    try:
        print(f"Uploading file '{file_name}' to Gemini API...")
        gemini_file = genai.upload_file(path=temp_file_path)
        
        while gemini_file.state.name == "PROCESSING":
            print("Waiting for video to be processed...")
            time.sleep(10)
            gemini_file = genai.get_file(gemini_file.name)

        if gemini_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail=f"Gemini API video processing failed.")
        
        print("Video processing complete.")
        prompt = "Transcribe the audio from this video, giving timestamps for salient events. Also provide detailed visual descriptions of what is happening throughout the video."
        response = model.generate_content([prompt, gemini_file], request_options={'timeout': 600})
        return JSONResponse(content={"analysis": response.text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up files
        if gemini_file:
            genai.delete_file(gemini_file.name)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)