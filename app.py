from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import os
import uuid

from utils.model import load_brain_model, load_chest_model
from utils.inference import brain_inference, chest_inference, spine_inference
from utils.report_generator import generate_medical_report

# ------------------------
# App Initialization
# ------------------------
app = FastAPI(
    title="Automated Medical Report Generator",
    description="AI-based medical report generation using MRI, X-ray, and Spine images",
    version="1.0"
)

# ------------------------
# CORS (Frontend Support)
# ------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # frontend allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------
# Load Models
# ------------------------
brain_model = load_brain_model(os.path.join(MODEL_DIR, "brain_mri_model.pth"))
chest_model = load_chest_model(os.path.join(MODEL_DIR, "chest_xray_model.pth"))

# ------------------------
# Serve Frontend (AFTER BUILD)
# ------------------------
if os.path.exists(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
else:
    @app.get("/")
    def root():
        return {
            "status": "running",
            "message": "Backend running. Frontend not built yet."
        }

# ------------------------
# Brain MRI Endpoint
# ------------------------
@app.post("/predict/brain-mri")
async def predict_brain_mri(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid image format")

    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = brain_inference(brain_model, file_path)

    return {
        "success": True,
        "report": generate_medical_report(
            scan_type="Brain MRI",
            prediction=prediction,
            confidence=confidence
        )
    }

# ------------------------
# Chest X-ray Endpoint
# ------------------------
@app.post("/predict/chest-xray")
async def predict_chest_xray(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid image format")

    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = chest_inference(chest_model, file_path)

    return {
        "success": True,
        "report": generate_medical_report(
            scan_type="Chest X-ray",
            prediction=prediction,
            confidence=confidence
        )
    }

# ------------------------
# Spine MRI Endpoint
# ------------------------
@app.post("/predict/spine-mri")
async def predict_spine_mri(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid image format")

    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = spine_inference(file_path)

    return {
        "success": True,
        "report": generate_medical_report(
            scan_type="Spine MRI",
            prediction=prediction,
            confidence=confidence / 100
        )
    }
