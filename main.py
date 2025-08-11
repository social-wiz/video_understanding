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
import httpx
import json
import asyncio

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

# Get Cloudflare credentials from environment
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_CUSTOMER_CODE = os.getenv("CLOUDFLARE_CUSTOMER_CODE")  # e.g., "loq77cmw0xduem6o"

# Note: Cloudflare credentials are optional for basic functionality
# but required for the /describe-cloudflare-video endpoint
if not CLOUDLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_CUSTOMER_CODE:
    logger.warning("Cloudflare credentials not fully configured. /describe-cloudflare-video endpoint will not work.")
    logger.warning("Set CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, and CLOUDFLARE_CUSTOMER_CODE for Cloudflare Stream support.")

# Pydantic models
class YouTubeRequest(BaseModel):
    youtube_url: str

class CloudflareVideoRequest(BaseModel):
    video_id: str

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

@app.post("/describe-cloudflare-video")
async def describe_cloudflare_video(
    request: CloudflareVideoRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Fetch video from Cloudflare Stream using video ID and provide description.
    
    - **video_id**: Cloudflare Stream video ID (sent in request body as JSON)
    """
    
    start_time = time.time()
    
    try:
        # Get Cloudflare API token from environment
        cloudflare_token = CLOUDFLARE_API_TOKEN
        if not cloudflare_token:
            raise HTTPException(
                status_code=500,
                detail="CLOUDFLARE_API_TOKEN environment variable not configured"
            )
        
        # Get account ID from environment
        cloudflare_account_id = CLOUDFLARE_ACCOUNT_ID
        if not cloudflare_account_id:
            raise HTTPException(
                status_code=500,
                detail="CLOUDFLARE_ACCOUNT_ID environment variable not configured"
            )
        
        logger.info(f"Processing Cloudflare video {request.video_id}")
        
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Step 1: Get video info to check if it's ready
            video_info_url = f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/stream/{request.video_id}"
            headers = {
                "Authorization": f"Bearer {cloudflare_token}",
                "Content-Type": "application/json"
            }
            
            video_info_response = await client.get(video_info_url, headers=headers)
            if not video_info_response.is_success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to get video info from Cloudflare: {video_info_response.text}"
                )
            
            video_info = video_info_response.json()
            if not video_info.get("success"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Cloudflare API error: {video_info.get('errors', [])}"
                )
            
            video_data = video_info["result"]
            video_status = video_data.get("status", {})
            
            # Check if video is ready for processing
            if video_status.get("state") != "ready":
                logger.info(f"Video not ready yet (state: {video_status.get('state')}), waiting for it to become ready...")
                
                # Wait for video to become ready
                max_video_wait_attempts = 60  # Maximum 10 minutes (60 * 10 seconds)
                video_wait_attempt = 0
                
                # Map of video states to user-friendly descriptions
                state_descriptions = {
                    "pendingupload": "Video upload is pending",
                    "downloading": "Video is being downloaded from source",
                    "encoding": "Video is being encoded and processed",
                    "ready": "Video is ready for processing",
                    "error": "Video processing failed",
                    "uploading": "Video is being uploaded",
                    "queued": "Video is queued for processing"
                }
                
                while video_status.get("state") != "ready" and video_wait_attempt < max_video_wait_attempts:
                    video_wait_attempt += 1
                    current_state = video_status.get("state", "unknown")
                    state_desc = state_descriptions.get(current_state, f"Unknown state: {current_state}")
                    
                    logger.info(f"Waiting for video to be ready (attempt {video_wait_attempt}/{max_video_wait_attempts}) - {state_desc}")
                    
                    # Wait 10 seconds before checking again
                    await asyncio.sleep(10)
                    
                    # Check video status again
                    video_info_response = await client.get(video_info_url, headers=headers)
                    if video_info_response.is_success:
                        video_info = video_info_response.json()
                        if video_info.get("success"):
                            video_data = video_info["result"]
                            video_status = video_data.get("status", {})
                            current_state = video_status.get("state", "unknown")
                            
                            # Check if video has required fields
                            if current_state == "ready":
                                if video_data.get("duration") and video_data.get("playback"):
                                    logger.info("Video is now ready and has all required fields!")
                                    break
                                else:
                                    logger.info("Video state is ready but missing duration or playback URLs, continuing to wait...")
                            elif current_state == "error":
                                error_details = video_status.get("error", "Unknown error")
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Video processing failed with state: {current_state} - {error_details}"
                                )
                            elif current_state in ["pendingupload", "downloading", "encoding", "uploading", "queued"]:
                                # These are normal processing states, continue waiting
                                logger.info(f"Video is still processing: {state_descriptions.get(current_state, current_state)}")
                            else:
                                logger.warning(f"Unexpected video state: {current_state}")
                        else:
                            logger.warning(f"Failed to get video info: {video_info.get('errors', [])}")
                    else:
                        logger.warning(f"Failed to get video info: {video_info_response.status_code}")
                
                if video_status.get("state") != "ready":
                    final_state = video_status.get("state", "unknown")
                    final_desc = state_descriptions.get(final_state, f"Unknown state: {final_state}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Video not ready after {max_video_wait_attempts} attempts. Final state: {final_state} ({final_desc})"
                    )
                
                logger.info("Video is now ready for processing!")
            
            # Check if video has required fields
            if not video_data.get("duration") or not video_data.get("playback"):
                raise HTTPException(
                    status_code=400,
                    detail="Video metadata incomplete. Missing duration or playback URLs."
                )
            
            logger.info(f"Video ready: {video_data.get('meta', {}).get('name', 'Unnamed')} - Duration: {video_data.get('duration')}s")
            
            # Step 2: Enable downloads for the video using the Downloads API
            download_create_url = f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/stream/{request.video_id}/downloads"
            
            logger.info(f"Enabling downloads at: {download_create_url}")
            # Send POST request with empty JSON body to enable downloads
            download_response = await client.post(download_create_url, headers=headers, json={})
            if not download_response.is_success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to enable downloads: {download_response.text}"
                )
            
            download_data = download_response.json()
            logger.info(f"Downloads API response: {download_data}")
            
            if not download_data.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to enable downloads: {download_data.get('errors', [])}"
                )
            
            # Step 3: Get the download URL from the response and wait for it to be ready
            download_url = None
            result = download_data.get("result", {})
            
            # Try to get download URL from various possible locations in the response
            if isinstance(result, dict):
                # Check if there's a default download URL
                if result.get("default") and result["default"].get("url"):
                    download_url = result["default"]["url"]
                    download_status = result["default"].get("status", "unknown")
                    logger.info(f"Found download URL in result.default: {download_url} (status: {download_status})")
                    
                    # If the download is not ready, we need to wait for it
                    if download_status != "ready":
                        logger.info(f"Download not ready yet (status: {download_status}), waiting...")
                        # Poll the Downloads API until the status is ready
                        max_attempts = 30  # Maximum 5 minutes (30 * 10 seconds)
                        attempt = 0
                        
                        while download_status != "ready" and attempt < max_attempts:
                            attempt += 1
                            logger.info(f"Polling download status (attempt {attempt}/{max_attempts})...")
                            
                            # Wait 10 seconds before checking again
                            await asyncio.sleep(10)
                            
                            # Check the download status again
                            status_response = await client.get(download_create_url, headers=headers)
                            if status_response.is_success:
                                status_data = status_response.json()
                                if status_data.get("success"):
                                    new_result = status_data.get("result", {})
                                    if new_result.get("default"):
                                        download_status = new_result["default"].get("status", "unknown")
                                        logger.info(f"Download status: {download_status}")
                                        
                                        # Update the download URL if it changed
                                        if new_result["default"].get("url"):
                                            download_url = new_result["default"]["url"]
                                            logger.info(f"Updated download URL: {download_url}")
                                    else:
                                        logger.warning("No default download found in status response")
                                else:
                                    logger.warning(f"Failed to get download status: {status_data.get('errors', [])}")
                            else:
                                logger.warning(f"Failed to get download status: {status_response.status_code}")
                        
                        if download_status != "ready":
                            raise HTTPException(
                                status_code=500,
                                detail=f"Download not ready after {max_attempts} attempts. Final status: {download_status}"
                            )
                        
                        logger.info("Download is now ready!")
                        
                elif result.get("url"):
                    download_url = result["url"]
                    logger.info(f"Found download URL in result: {download_url}")
            
            # If still no URL found, try to construct it using the customer code
            if not download_url:
                if CLOUDFLARE_CUSTOMER_CODE:
                    download_url = f"https://customer-{CLOUDFLARE_CUSTOMER_CODE}.cloudflarestream.com/{request.video_id}/downloads/default.mp4"
                    logger.info(f"Constructed download URL: {download_url}")
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="CLOUDFLARE_CUSTOMER_CODE environment variable not set. Cannot construct download URL."
                    )
            
            logger.info(f"Using download URL: {download_url}")
            
            # Step 4: Download the video content
            logger.info("Downloading video content...")
            video_response = await client.get(download_url)
            if not video_response.is_success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to download video: {video_response.status_code} - {video_response.text}"
                )
            
            video_content = video_response.content
            video_size = len(video_content)
            
            # Check file size
            if video_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Video too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            logger.info(f"Downloaded video: {video_size} bytes")
            
            # Step 5: Process the video with Gemini AI
            try:
                # Check if file is small enough for inline processing (< 20MB)
                if video_size < 20 * 1024 * 1024:
                    # Use inline processing for smaller files
                    logger.info("Using inline video processing for description")
                    
                    # Create content parts for inline processing
                    parts = [
                        {
                            "inline_data": {
                                "data": video_content,
                                "mime_type": "video/mp4"
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
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                        temp_file.write(video_content)
                        temp_file_path = temp_file.name
                    
                    try:
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
                        
                    finally:
                        # Clean up temporary file
                        os.unlink(temp_file_path)
                
                processing_time = time.time() - start_time
                
                return VideoDescriptionResponse(
                    success=True,
                    description=description_result,
                    processing_time=processing_time,
                    file_name=f"cloudflare_video_{request.video_id}.mp4"
                )
                
            except Exception as e:
                logger.error(f"Error processing video with Gemini AI: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing video with Gemini AI: {str(e)}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing Cloudflare video description: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing Cloudflare video description: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
