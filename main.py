import os
import time
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Video Description API",
    description="API for transcribing videos and providing visual descriptions with timestamps using Google's Gemini AI",
    version="1.0.0"
)

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_VIDEO_TYPES = [
    "video/mp4", "video/mpeg", "video/mov", "video/avi", 
    "video/x-flv", "video/mpg", "video/webm", "video/wmv", "video/3gpp"
]

# Default prompt for video description
DEFAULT_PROMPT = "Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions."

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MY_API_KEY = os.getenv("MY_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

if not MY_API_KEY:
    raise ValueError("MY_API_KEY environment variable is required")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Pydantic models
class YouTubeRequest(BaseModel):
    youtube_url: str

class VideoDescriptionResponse(BaseModel):
    success: bool
    description: str
    processing_time: float
    file_name: str

# Dependency for API key verification
async def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != MY_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Please provide a valid X-API-Key header."
        )
    return x_api_key

@app.get("/")
async def root():
    return {
        "message": "Video Description API",
        "version": "1.0.0",
        "status": "running",
        "description": "Transcribe videos and provide visual descriptions with timestamps"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/describe-video", response_model=VideoDescriptionResponse)
async def describe_video(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Transcribe video audio and provide visual descriptions with timestamps.
    
    This endpoint processes videos to:
    - Transcribe the audio content
    - Provide visual descriptions of what's happening
    - Include timestamps for salient events
    - Sample video at 1 frame per second
    
    - **file**: Video file to describe (max 50MB)
    """
    
    start_time = time.time()
    
    # Validate file type
    if file.content_type not in SUPPORTED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_VIDEO_TYPES)}"
        )
    
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing video for description: {file.filename} ({file_size} bytes)")
        
        # Check if file is small enough for inline processing (< 20MB)
        if file_size < 20 * 1024 * 1024:
            # Use inline processing for smaller files
            logger.info("Using inline video processing for description")
            
            # Create content parts for inline processing
            parts = [
                {
                    "inline_data": {
                        "data": content,
                        "mime_type": file.content_type
                    }
                },
                {
                    "text": DEFAULT_PROMPT
                }
            ]
            
            # Generate content
            model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
            response = model.generate_content(
                parts,
                request_options={'timeout': 600}
            )
            
            description_result = response.text
            
        else:
            # Use File API for larger files
            logger.info("Using File API for video description processing")
            
            # Upload file to Gemini API
            video_file = genai.upload_file(path=temp_file_path)
            logger.info(f"Uploaded file: {video_file.uri}")
            
            # Wait for processing
            while video_file.state.name == "PROCESSING":
                logger.info("Waiting for video to be processed...")
                time.sleep(10)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                raise HTTPException(
                    status_code=500,
                    detail=f"Video processing failed: {video_file.state}"
                )
            
            logger.info("Video processing complete!")
            
            # Generate description
            model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
            response = model.generate_content(
                [DEFAULT_PROMPT, video_file],
                request_options={'timeout': 600}
            )
            
            description_result = response.text
            
            # Clean up uploaded file
            genai.delete_file(video_file.name)
            logger.info(f"Deleted file '{video_file.name}' from API")
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        processing_time = time.time() - start_time
        
        return VideoDescriptionResponse(
            success=True,
            description=description_result,
            processing_time=processing_time,
            file_name=file.filename
        )
        
    except Exception as e:
        logger.error(f"Error processing video description: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video description: {str(e)}"
        )

@app.post("/describe-youtube")
async def describe_youtube(
    request: YouTubeRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Transcribe YouTube video audio and provide visual descriptions with timestamps.
    
    - **youtube_url**: YouTube video URL (sent in request body as JSON)
    """
    
    start_time = time.time()
    
    try:
        # Create content parts for YouTube processing
        parts = [
            {
                "file_data": {
                    "file_uri": request.youtube_url
                }
            },
            {
                "text": DEFAULT_PROMPT
            }
        ]
        
        # Generate content
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        response = model.generate_content(
            parts,
            request_options={'timeout': 600}
        )
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "description": response.text,
            "processing_time": processing_time,
            "youtube_url": request.youtube_url
        }
        
    except Exception as e:
        logger.error(f"Error processing YouTube video description: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing YouTube video description: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
