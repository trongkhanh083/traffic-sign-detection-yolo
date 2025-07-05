from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import os
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi import Depends, HTTPException

app = FastAPI()
model = YOLO("best.onnx", task='detect')  # Load your exported model

api_key_header = APIKeyHeader(name="X-API-Key")
API_KEY = os.getenv("API_KEY")

@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    # Read image
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    h, w = image.shape[:2]
    scale = 640 / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w*scale), int(h*scale)))
    
    # Run inference
    result = model.predict(image, conf=0.5)
    
    # Format response
    detections = []
    for box in result[0].boxes:
        detections.append({
            "class": int(box.cls),
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        })

    # Return bounding boxes
    return JSONResponse({
        "detections": detections,
        "image_size": result[0].orig_shape
    })

@app.get("/")
async def root():
    return {"message": "YOLOv5 FastAPI"}