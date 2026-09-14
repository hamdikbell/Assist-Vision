"""
Assist Vision - Streamlit app
Pedestrian assistance app for visually impaired users:
- Real-time obstacle detection (YOLOv8) with approximate distance
- Sign / text reading (EasyOCR)
- Scene description (BLIP)
- Voice output via the BROWSER's speech synthesis (works once deployed online)
"""

import time
import threading
import queue

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


# --------------------------------------------------------------------------
# Config / constants
# --------------------------------------------------------------------------

st.set_page_config(page_title="Assist Vision", page_icon="🦯", layout="centered")

FOCAL_LENGTH = 700  # recalibrate if needed (see calibration note at bottom)

REAL_HEIGHTS = {
    "person": 1.70, "bicycle": 1.10, "car": 1.50, "motorcycle": 1.30,
    "bus": 3.00, "truck": 2.80, "animal": 0.50, "bench": 0.45,
    "chair": 0.90, "obstacle on the ground": 0.30, "obstacle": 0.80,
    "stop sign": 2.10, "traffic light": 2.50,
}

CATEGORY_MAP = {
    "person": "person", "bicycle": "bicycle", "car": "car",
    "motorcycle": "motorcycle", "bus": "bus", "truck": "truck",
    "dog": "animal", "cat": "animal", "bench": "bench", "chair": "chair",
    "backpack": "obstacle on the ground", "suitcase": "obstacle on the ground",
    "fire hydrant": "obstacle", "stop sign": "stop sign",
    "traffic light": "traffic light",
}

DANGER_PRIORITY = {
    "car": 1, "bus": 1, "truck": 1, "motorcycle": 1,
    "bicycle": 2, "person": 2, "animal": 2,
    "obstacle": 3, "obstacle on the ground": 3, "bench": 3, "chair": 3,
    "stop sign": 4, "traffic light": 4,
}

OCR_INTERVAL_SEC = 5
DESCRIBE_INTERVAL_SEC = 15


def format_distance(distance_m):
    if distance_m is None:
        return "unknown distance"
    if distance_m < 1:
        return f"{int(distance_m * 100)} centimeters"
    return f"{distance_m:.1f} meters"


# --------------------------------------------------------------------------
# Cached model loaders (loaded once per session, not on every rerun)
# --------------------------------------------------------------------------

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")


@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)


@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


# --------------------------------------------------------------------------
# Detection logic
# --------------------------------------------------------------------------

def estimate_distance_m(category, box_height):
    real_height = REAL_HEIGHTS.get(category, 1.0)
    if box_height <= 0:
        return None
    return round((real_height * FOCAL_LENGTH) / box_height, 1)


def detect_obstacles(model, frame):
    results = model(frame, verbose=False)[0]
    h, w = frame.shape[:2]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label_en = model.names[cls_id]
        if label_en not in CATEGORY_MAP:
            continue

        category = CATEGORY_MAP[label_en]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        box_height = y2 - y1

        if cx < w / 3:
            position = "on your left"
        elif cx > 2 * w / 3:
            position = "on your right"
        else:
            position = "ahead"

        distance_m = estimate_distance_m(category, box_height)

        detections.append({
            "category": category,
            "position": position,
            "distance_m": distance_m,
            "confidence": round(conf, 2),
            "priority": DANGER_PRIORITY.get(category, 5),
            "box": (int(x1), int(y1), int(x2), int(y2)),
        })

    detections.sort(key=lambda d: (d["priority"], d["distance_m"] or 999))
    return detections


def format_obstacle_message(detections):
    if not detections:
        return None
    top = detections[:2]
    parts = [f"{d['category']} {format_distance(d['distance_m'])}, {d['position']}" for d in top]
    return "Warning: " + ", ".join(parts)


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        dist = d["distance_m"] or 99
        color = (0, 0, 255) if dist < 1.5 else (0, 165, 255) if dist < 4 else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d['category']} - {format_distance(d['distance_m'])}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


# --------------------------------------------------------------------------
# Browser-based text-to-speech (works when deployed, unlike pyttsx3)
# --------------------------------------------------------------------------

def speak_in_browser(text: str, key: str):
    """Injects JS that uses the browser's SpeechSynthesis API.
    `key` must change for the speech to re-trigger (Streamlit dedups identical components)."""
    safe_text = text.replace('"', '\\"')
    components.html(
        f"""
        <script>
        var msg = new SpeechSynthesisUtterance("{safe_text}");
        msg.lang = "en-US";
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
        </script>
        <div style="display:none">{key}</div>
        """,
        height=0,
    )


# --------------------------------------------------------------------------
# Video processor (runs in a background thread per streamlit-webrtc)
# --------------------------------------------------------------------------

class AssistVisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.yolo = load_yolo()
        self.ocr = load_ocr()
        self.blip_processor, self.blip_model = load_blip()
        self.message_queue = queue.Queue(maxsize=1)
        self.last_ocr = 0
        self.last_describe = 0
        self.lock = threading.Lock()

    def _push_message(self, msg):
        # Keep only the latest message (drop old ones, non-blocking)
        try:
            self.message_queue.get_nowait()
        except queue.Empty:
            pass
        self.message_queue.put_nowait(msg)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        detections = detect_obstacles(self.yolo, img)
        msg = format_obstacle_message(detections)
        if msg:
            self._push_message(msg)

        now = time.time()
        if now - self.last_ocr > OCR_INTERVAL_SEC:
            texts = self.ocr.readtext(img)
            for (_, text, conf) in texts:
                if conf >= 0.5 and text.strip():
                    self._push_message(f"Sign says: {text.strip()}")
                    break
            self.last_ocr = now

        if now - self.last_describe > DESCRIBE_INTERVAL_SEC:
            pil_img = Image.fromarray(img[:, :, ::-1])
            inputs = self.blip_processor(pil_img, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_new_tokens=40)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            self._push_message(caption)
            self.last_describe = now

        display_img = draw_detections(img.copy(), detections)
        return av.VideoFrame.from_ndarray(display_img, format="bgr24")


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.title("🦯 Assist Vision")
st.write("Real-time obstacle detection, sign reading, and scene description for visually impaired pedestrians.")

ctx = webrtc_streamer(
    key="assist-vision",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=AssistVisionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

status_placeholder = st.empty()

if ctx.state.playing and ctx.video_processor:
    # Poll the processor's message queue and speak new messages via the browser
    try:
        msg = ctx.video_processor.message_queue.get_nowait()
        status_placeholder.info(msg)
        speak_in_browser(msg, key=str(time.time()))
    except queue.Empty:
        pass
    time.sleep(0.5)
    st.rerun()
else:
    status_placeholder.write("Click **Start** above and allow camera access.")

st.markdown("---")
st.caption(
    "Distance is an approximation based on object size in frame, not a precise laser measurement. "
    "Recalibrate FOCAL_LENGTH in the code for your specific camera if distances look off."
)
