from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi.responses import JSONResponse

app = FastAPI()
model = YOLO("best.onnx", task='detect')  # Load your exported model

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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