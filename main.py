import os
import time
import uuid
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# --- Configuration ---
# Load the API key from environment variables for security
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring GenerativeAI: {e}")

# Define File Size Limit
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # 50 MB in bytes

# Initialize the FastAPI app
app = FastAPI(
    title="Video Analysis API",
    description="An API to analyze video content using Gemini 1.5 Flash."
)

# Initialize the Gemini Model
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    """
    Accepts a video file, uploads it to Gemini for processing,
    and returns a textual analysis.
    """
    # Add validation checks at the beginning
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, # HTTP 413: Payload Too Large
            detail=f"File size exceeds the limit of {MAX_FILE_SIZE_MB} MB."
        )

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")

    # Save the uploaded file temporarily to the server's disk
    temp_file_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())

    gemini_file = None
    try:
        # 1. Upload the file to the Gemini API
        print(f"Uploading file '{file.filename}' to Gemini API...")
        gemini_file = genai.upload_file(path=temp_file_path)
        print(f"Completed upload: {gemini_file.uri}")

        # 2. Wait for the file to be processed
        while gemini_file.state.name == "PROCESSING":
            print("Waiting for video to be processed...")
            time.sleep(10)
            gemini_file = genai.get_file(gemini_file.name)

        if gemini_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail=f"Gemini API video processing failed.")
        
        print("Video processing complete.")

        # 3. Define the prompt and generate content
        prompt = "Transcribe the audio from this video, giving timestamps for salient events. Also provide detailed visual descriptions of what is happening throughout the video."
        
        print("Generating analysis...")
        response = model.generate_content(
            [prompt, gemini_file],
            request_options={'timeout': 600} # 10-minute timeout
        )

        return JSONResponse(content={"analysis": response.text})

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 4. Clean up files regardless of success or failure
        if gemini_file:
            print(f"Deleting file '{gemini_file.name}' from Gemini API.")
            genai.delete_file(gemini_file.name)
        
        if os.path.exists(temp_file_path):
            print(f"Deleting temporary file: {temp_file_path}")
            os.remove(temp_file_path)