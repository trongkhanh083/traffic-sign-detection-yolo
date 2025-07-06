import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import base64
import cv2

st.title("YOLOv5 Detector")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send to API
    response = requests.post(
        "https://yolov5-api-ldqh.onrender.com/predict",
        files={"file": uploaded_file.getvalue()}
    )

    if response.status_code == 200:
        # Display JSON results
        st.subheader("Detection Result")
        st.json(response.json())

        # --- NEW: Display image with boxes ---
        # Option 1: If your API returns base64 image
        if "image_with_boxes" in response.json():
            img_bytes = base64.b64decode(response.json()["image_with_boxes"])
            st.image(Image.open(io.BytesIO(img_bytes)), caption="Detected Objects")

        # Option 2: Draw boxes locally (requires OpenCV)
        else:
            detections = response.json().get("detections", [])
            
            # Convert PIL to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Draw boxes
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det  # Adjust based on your API response
                cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(img_cv, f"{cls_id}: {conf:.2f}", (int(x1), int(y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            
            # Convert back to PIL and display
            st.image(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), 
                   caption="Detected Objects")

    else:
        st.error(f"Error {response.status_code}: {response.text}")