"""
AI Employee Monitoring - Streamlit App
======================================
Run:  streamlit run app.py

Source options:
  - Webcam (local PC)
  - Video file upload
  - RTSP / IP camera URL
"""

import os
import time
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from engine import MonitorEngine, STATUS_COLORS

st.set_page_config(page_title="AI Employee Monitoring", layout="wide")

# ------------------------------------------------------------------ session state
if "running" not in st.session_state:
    st.session_state.running = False

# ------------------------------------------------------------------ sidebar
st.sidebar.title("AI EMPLOYEE MONITORING")

model_path = st.sidebar.text_input(
    "Model file (.pt)",
    "models/best.pt",
    help="Kaggle par train kiya hua best.pt yahan rakhein. "
         "File na ho to default yolov8n.pt download ho ga.",
)

source_type = st.sidebar.radio("Source", ["Webcam", "Upload Video", "RTSP / IP Camera"])

uploaded_file = None
rtsp_url = ""
if source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Video upload", type=["mp4", "avi", "mov", "mkv"])
elif source_type == "RTSP / IP Camera":
    rtsp_url = st.sidebar.text_input("RTSP URL", "rtsp://user:pass@192.168.1.10:554/stream")

col1, col2 = st.sidebar.columns(2)
if col1.button("▶ Start", use_container_width=True):
    st.session_state.running = True
if col2.button("⏸ Stop", use_container_width=True):
    st.session_state.running = False

# status placeholder (sidebar list)
status_ph = st.sidebar.empty()
metric_ph = st.sidebar.empty()

# ------------------------------------------------------------------ main frame
frame_ph = st.empty()

# ------------------------------------------------------------------ helpers
@st.cache_resource
def load_model(path: str):
    if path and os.path.exists(path):
        return YOLO(path)
    return YOLO("yolov8n.pt")  # auto-download, detects person + cell phone (COCO)


def get_source():
    if source_type == "Webcam":
        return cv2.VideoCapture(0)
    if source_type == "Upload Video" and uploaded_file is not None:
        tpath = os.path.join("temp_video.mp4")
        with open(tpath, "wb") as f:
            f.write(uploaded_file.read())
        return cv2.VideoCapture(tpath)
    if source_type == "RTSP / IP Camera" and rtsp_url:
        return cv2.VideoCapture(rtsp_url)
    return None


def status_html(statuses):
    lines = []
    for tid in sorted(statuses.keys()):
        s = statuses[tid]
        color = "#32d74b" if s == "WORKING" else "#ff453a" if s in ("PHONE USAGE", "ABSENT") else "#ff9f0a"
        lines.append(
            f"<div style='margin:2px 0;font-size:15px;'>"
            f"<span style='color:{color};'>&#9679;</span> "
            f"ID {tid} - {s}</div>"
        )
    return "<div style='font-family:monospace;'>" + "".join(lines) + "</div>"


# ------------------------------------------------------------------ loop
if st.session_state.running:
    model = load_model(model_path)
    engine = MonitorEngine(model, "config.json")
    cap = get_source()

    if cap is None or not cap.isOpened():
        st.sidebar.error("Source open nahi hua. Check karein.")
        st.session_state.running = False
    else:
        prev = time.time()
        fps_smooth = None
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                st.sidebar.warning("Video khatam / feed band ho gaya.")
                break

            annotated, statuses, alerts = engine.process(frame)
            working = sum(1 for s in statuses.values() if s == "WORKING")

            # fps
            now = time.time()
            inst = 1.0 / max(now - prev, 1e-6)
            fps_smooth = inst if fps_smooth is None else 0.9 * fps_smooth + 0.1 * inst
            prev = now
            engine.set_fps(fps_smooth)

            frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            status_ph.markdown(status_html(statuses), unsafe_allow_html=True)
            metric_ph.markdown(
                f"<span style='font-family:monospace;'>Working: <b>{working}</b> | "
                f"Alerts: <b style='color:{'#32d74b' if alerts == 0 else '#ff453a'};'>{alerts}</b></span>",
                unsafe_allow_html=True,
            )

        cap.release()

        if source_type == "Upload Video" and os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
else:
    st.info("Sidebar se source select karein aur **▶ Start** dabayen.\n\n"
            "Green box = WORKING, Red = PHONE USAGE / ABSENT, Yellow = phone detection.")
