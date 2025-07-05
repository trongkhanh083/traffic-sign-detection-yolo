from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import os
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi import Depends, HTTPException
from fastapi_limiter.depends import RateLimiter

app = FastAPI()
model = YOLO("best.onnx", task='detect')  # Load your exported model

api_key_header = APIKeyHeader(name="X-API-Key")
API_KEY = os.getenv("API_KEY")

@app.post("/predict", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def predict(file: UploadFile = File(...), api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    # Read image
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
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