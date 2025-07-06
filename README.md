# Traffic Sign Detection System With YOLOv5 🚦

A complete solution for detecting traffic signs using YOLOv5, with a FastAPI backend and Streamlit dashboard.

![Demo GIF](test_result/detect/predict/road43_png.rf.c576096fc5f0b00379321f94ddffcbc6.jpg)
![Demo GIF](test_result/detect/predict/000722_jpg.rf.9bee48a5dc22112fb5a705192fd54d45.jpg)

## 🚀 Features
- **Real-time detection** of 15 traffic sign classes
- **Web API** for integration with other apps
- **Interactive dashboard** with visualization

## 🛠️ Installation 
  ```bash
  git clone https://github.com/trongkhanh083/traffic-sign-detection-yolo.git
  cd traffic-sign-detection-yolo
  pip install -r requirements.txt
  ```

## 📡 Usage 
### 1. Web API with Render
https://yolov5-api-ldqh.onrender.com/

#### Check API endpoint
  ``` bash
  curl -X POST -F "file=@test.jpg" https://yolov5-api-ldqh.onrender.com/predict
  ```
### 2. Demo with Streamlit
https://traffic-sign-detection-yolo-x6wavcgbmji9eucets2yjy.streamlit.app/

#### Run on local
  ``` bash
  streamlit run app/dashboard.py
  ```

## 🎨 Customization 
- **Add new signs:** Retrain model with additional classes
- **Change colors:** Modify box_color in dashboard.py
- **Adjust sensitivity:** Edit conf_threshold in API