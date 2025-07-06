import streamlit as st
import requests
import numpy as np
from PIL import Image
import cv2

st.title("YOLOv5 Detector")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Convert to OpenCV format (BGR)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Send to API
    response = requests.post(
        "https://yolov5-api-ldqh.onrender.com/predict",
        files={"file": uploaded_file.getvalue()}
    )

    if response.status_code == 200:
        # Display JSON results
        st.subheader("Detection Results")
        st.json(response.json())

        # Draw bounding boxes locally
        detections = response.json().get("detections", [])
        
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]  # Unpack first 6 values
            label = f"Class {int(cls_id)} ({conf:.2f})"
            
            # Draw rectangle
            cv2.rectangle(img_cv, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(img_cv, label, 
                       (int(x1), int(y1 - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)

        # Convert back to RGB for Streamlit
        img_with_boxes = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        st.image(img_with_boxes, caption="Detected Objects", use_column_width=True)

    else:
        st.error(f"Error {response.status_code}: {response.text}")