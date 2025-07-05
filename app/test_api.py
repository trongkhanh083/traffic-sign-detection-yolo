import requests

url = "http://localhost:8000/predict"
files = {"file": open("data/images/test/000883_JPG_jpg.rf.bcfbe7999ec1b4e660a155e0e145e234.jpg", "rb")}  # Replace with your image
response = requests.post(url, files=files)
print(response.json())