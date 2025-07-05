import requests

url = "https://yolov5-api-ldqh.onrender.com/predict"
headers = {"X-API-Key": "trongkhanh083"}
files = {"file": open("test.jpg", "rb")}  # Replace with your image

response = requests.post(url, files=files, headers=headers)
print(response.json())