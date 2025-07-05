import streamlit as st
import requests

st.title("YOLOv5 Detector")
uploaded_file = st.file_uploader("Choose an image")

if uploaded_file:
    response = requests.post(
        "https://yolov5-api-ldqh.onrender.com/predict",
        files={"file": uploaded_file.getvalue()},
        headers={"X-API-Key": "trongkhanh083"}
    )
    
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error(f"Error {response.status_code}: {response.text}")