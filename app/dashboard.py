import streamlit as st
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont

st.title("Traffic Sign Detector")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# Load a basic font (size 15)
try:
    font = ImageFont.truetype("arial.ttf", 15)
except:
    font = ImageFont.load_default()

CLASS_NAMES = {
    0: "Green Light",
    1: "Red Light",
    2: "Speed Limit 10",
    3: "Speed Limit 100",
    4: "Speed Limit 110",
    5: "Speed Limit 120",
    6: "Speed Limit 20",
    7: "Speed Limit 30",
    8: "Speed Limit 40",
    9: "Speed Limit 50",
    10: "Speed Limit 60",
    11: "Speed Limit 70",
    12: "Speed Limit 80",
    13: "Speed Limit 90",
    14: "Stop"
}

if uploaded_file:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Create a copy for drawing
    detected_image = image.copy()
    draw = ImageDraw.Draw(detected_image)

    # Send to API
    response = requests.post(
        "https://yolov5-api-ldqh.onrender.com/predict",
        files={"file": uploaded_file.getvalue()}
    )

    if response.status_code == 200:
        response_data = response.json()
        st.subheader("Detection Results")
        st.json(response_data)

        # Draw bounding boxes
        if "detections" in response_data:
            for detection in response_data["detections"]:
                try:
                    # Get detection info
                    cls_id = int(detection["class"])
                    conf = float(detection["confidence"])
                    bbox = detection["bbox"]
                    
                    # Convert coordinates
                    width, height = image.size
                    x1 = int(bbox[0] * (width / response_data["image_size"][0]))
                    y1 = int(bbox[1] * (height / response_data["image_size"][1]))
                    x2 = int(bbox[2] * (width / response_data["image_size"][0]))
                    y2 = int(bbox[3] * (height / response_data["image_size"][1]))

                    # Draw rectangle
                    box_color = "green"
                    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

                    # Draw label
                    label = f"{CLASS_NAMES.get(cls_id)} ({conf:.2f})"
                    draw.text((x1, y1 - 20), label, fill=box_color, font=font)
                
                except Exception as e:
                    st.warning(f"Error processing detection: {e}")
                    continue

            # Display processed image
            st.image(detected_image,
                     caption="Detected Objects",
                     use_column_width=True)
        else:
            st.warning("No detections found in response")
    else:
        st.error(f"API Error {response.status_code}: {response.text}")