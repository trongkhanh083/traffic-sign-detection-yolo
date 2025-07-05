# dashboard.py
import streamlit as st
import requests

st.title("Traffic Sign Detector")
uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    response = requests.post(
        "https://yolov5-api-ldqh.onrender.com/predict",
        files={"file": uploaded_file}
    )
    st.json(response.json())
    st.image(uploaded_file)