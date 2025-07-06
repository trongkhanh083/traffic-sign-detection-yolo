from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import gc
from fastapi.responses import JSONResponse
import base64
from io import BytesIO

app = FastAPI()
model = YOLO("best.onnx", task='detect')  # Load your exported model

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if max(img.shape) > 1280:
        img = cv2.resize(img, (1280, 1280))

    result = model(img, imgsz=640)
    gc.collect()
    
    # Format response
    detections = []
    for box in result[0].boxes:
        detections.append({
            "class": int(box.cls),
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        })

    buffered = BytesIO()
    result[0].save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Return bounding boxes
    return JSONResponse({
        "detections": detections,
        "image_size": result[0].orig_shape,
        "image_with_boxes": img_str
    })

@app.get("/")
async def root():
    return {"message": "YOLOv5 FastAPI"}