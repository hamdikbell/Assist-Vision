
# 🦯 Assist Vision

A real-time assistive system for visually impaired pedestrians that detects obstacles, reads signs, and describes surroundings using Computer Vision and Deep Learning — with spoken feedback.

## 📌 Features

- 🚗 **Obstacle detection** — cars, bicycles, people, animals, and more
- 📏 **Distance estimation** — approximate distance in meters/centimeters
- 🧭 **Position awareness** — left, right, or ahead
- 🔤 **Sign reading** — detects and reads street signs / text
- 🗣️ **Scene description** — natural-language description of the surroundings
- 🔊 **Voice feedback** — spoken alerts, prioritized by danger level

## 🛠️ Tech Stack

- Python 3.11
- Ultralytics YOLOv8 (obstacle detection)
- EasyOCR (sign / text reading)
- Transformers — BLIP (scene description)
- OpenCV (webcam & frame processing)
- Streamlit + streamlit-webrtc (web interface)

## 📁 Project Structure
```
assist-vision/
├── data/                  # collected images/data (not included in repo)
├── models/                # saved / fine-tuned model weights (not included in repo)
├── app.py                 # Streamlit web interface
├── assist_vision.ipynb    # notebook prototype (Jupyter version)
├── requirements.txt
├── packages.txt            # system dependencies
└── README.md
```
## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/hamdikbell/assist-vision.git
cd assist-vision
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Click **Start** and allow camera access in your browser.

## 🎯 Model Details

| Module | Model | Role |
|---|---|---|
| Obstacle detection | YOLOv8n | Detects cars, people, bicycles, animals... |
| Sign reading | EasyOCR | Reads visible text / street signs |
| Scene description | BLIP | Generates a natural-language scene caption |

## 🔮 Next Steps

- [ ] Replace distance approximation with MiDaS depth estimation
- [ ] Fine-tune YOLOv8 on custom obstacle data (curbs, stairs, construction signs)
- [ ] Add French/Arabic sign reading support 🇹🇳
- [ ] Optimize for embedded deployment (Raspberry Pi / Jetson Nano)

## 👩‍💻 Author

**Ikbel Hamdi** — Data Science & AI Engineering Student

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](linkedin.com/in/ikbel-hamdi-706427286)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/hamdikbell)
