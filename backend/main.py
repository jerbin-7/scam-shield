"""
ScamShield API - Main FastAPI Application
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import asyncio
from typing import Optional
import traceback

app = FastAPI(
    title="ScamShield API",
    description="ML-powered scam and phishing detection API",
    version="1.0.0"
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy-loaded models ───────────────────────────────────────────────────────

_url_detector = None
_email_detector = None
_login_detector = None
_qr_detector = None
_voice_detector = None


def get_url_model():
    global _url_detector
    if _url_detector is None:
        from models.url_detector import get_url_detector
        _url_detector = get_url_detector()
    return _url_detector


def get_email_model():
    global _email_detector
    if _email_detector is None:
        from models.email_detector import get_email_detector
        _email_detector = get_email_detector()
    return _email_detector


def get_login_model():
    global _login_detector
    if _login_detector is None:
        from models.login_page_detector import get_login_detector
        _login_detector = get_login_detector()
    return _login_detector


def get_qr_model():
    global _qr_detector
    if _qr_detector is None:
        from models.qr_detector import get_qr_detector
        _qr_detector = get_qr_detector()
    return _qr_detector


def get_voice_model():
    global _voice_detector
    if _voice_detector is None:
        from models.voice_detector import get_voice_detector
        _voice_detector = get_voice_detector()
    return _voice_detector


# ─── Request/Response Models ──────────────────────────────────────────────────

class URLRequest(BaseModel):
    url: str


class EmailRequest(BaseModel):
    subject: str
    body: str
    sender: str = ""
    headers: str = ""


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "ScamShield API",
        "version": "1.0.0",
        "endpoints": [
            "POST /detect/url",
            "POST /detect/email",
            "POST /detect/login-page",
            "POST /detect/qr-code",
            "POST /detect/voice"
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/detect/url")
async def detect_url(request: URLRequest):
    """Detect phishing URLs using ML model"""
    try:
        if not request.url or len(request.url.strip()) < 4:
            raise HTTPException(status_code=400, detail="Please provide a valid URL")
        
        detector = get_url_model()
        result = detector.predict(request.url.strip())
        
        return {
            "success": True,
            "input": request.url,
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/email")
async def detect_email(request: EmailRequest):
    """Detect phishing emails using NLP + ML"""
    try:
        if not request.body and not request.subject:
            raise HTTPException(status_code=400, detail="Email body or subject is required")
        
        detector = get_email_model()
        email_data = {
            "subject": request.subject,
            "body": request.body,
            "sender": request.sender,
            "headers": request.headers
        }
        result = detector.predict(email_data)
        
        return {
            "success": True,
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/login-page")
async def detect_login_page(file: UploadFile = File(...)):
    """Detect fake login pages from screenshots using computer vision"""
    try:
        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 
                        'image/webp', 'image/bmp']
        
        if file.content_type and file.content_type not in allowed_types:
            # Try by extension
            fname = file.filename or ""
            if not any(fname.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Please upload an image file (JPG, PNG, etc.)"
                )
        
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        detector = get_login_model()
        result = detector.predict(image_bytes)
        
        return {
            "success": True,
            "filename": file.filename,
            "file_size": len(image_bytes),
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/qr-code")
async def detect_qr_code(file: UploadFile = File(...)):
    """Detect phishing QR codes from images"""
    try:
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        detector = get_qr_model()
        result = detector.predict(image_bytes)
        
        return {
            "success": True,
            "filename": file.filename,
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/voice")
async def detect_voice(file: UploadFile = File(...)):
    """Detect deepfake/AI-generated voice using audio ML"""
    try:
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(audio_bytes) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")
        
        detector = get_voice_model()
        result = detector.predict(audio_bytes)
        
        return {
            "success": True,
            "filename": file.filename,
            "file_size": len(audio_bytes),
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )