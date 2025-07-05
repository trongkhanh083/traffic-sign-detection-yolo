import requests

url = "https://yolov5-api-ldqh.onrender.com/predict"
files = {"file": open("test.jpg", "rb")}  # Replace with your image

response = requests.post(url, files=files)
print(response.json())