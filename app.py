"""
AI Employee Monitoring - Streamlit App
======================================
Run:  streamlit run app.py

Sources:
  - Camera (Browser)  -> browser webcam, Streamlit Cloud par bhi chalta hai
  - Upload Video      -> mp4/avi/mkv upload kar ke analyze karein
  - RTSP / IP Camera  -> sirf LOCAL PC par (cloud server RTSP tak nahi pahunch sakta)
"""

import os
import time
import cv2
import av
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

from engine import MonitorEngine

st.set_page_config(page_title="AI Employee Monitoring", page_icon="🛡", layout="wide")

# ------------------------------------------------------------------ theme / CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background: #0b0f0e;}
    section[data-testid="stSidebar"] {
        background: #0e1412; border-right: 1px solid #1e2a24;
    }
    section[data-testid="stSidebar"] .stRadio label {color: #cfe3d8 !important;}
    .block-container {padding-top: 1.4rem; max-width: 1250px;}
    .app-title {
        font-weight: 800; font-size: 1.35rem; letter-spacing: 1px;
        color: #eafff2; margin: 0;
    }
    .app-sub {font-size: 0.8rem; color: #7fa290; margin-top: 2px;}
    .panel {
        background: #0d0d0d; border: 1px solid #24312b; border-radius: 12px;
        padding: 14px 16px; font-family: 'Consolas', 'Monaco', monospace; color: #eafff2;
    }
    .panel .p-title {font-weight: 700; font-size: 0.95rem; letter-spacing: 0.5px;}
    .panel .p-live {font-size: 0.7rem; color: #8aa79a; margin: 2px 0 10px;}
    .panel .p-row {margin: 4px 0; font-size: 0.82rem;}
    .panel .p-foot {
        margin-top: 10px; padding-top: 8px; border-top: 1px solid #24312b;
        font-size: 0.8rem;
    }
    .dot {
        display: inline-block; width: 9px; height: 9px; border-radius: 50%;
        margin-right: 7px; vertical-align: middle;
    }
    .card {
        background: #101614; border: 1px solid #24312b; border-radius: 12px;
        padding: 14px 10px; text-align: center; flex: 1;
    }
    .card .c-title {font-size: 0.72rem; color: #9fb8ab; font-weight: 600;
        letter-spacing: 0.6px; margin-bottom: 6px;}
    .card .c-val {font-size: 1.7rem; font-weight: 800;}
    .stButton > button {
        width: 100%; border-radius: 10px; border: 1px solid #2c3a33;
        background: #14201b; color: #eafff2; font-weight: 600;
    }
    .stButton > button:hover {border-color: #32d74b;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ session
if "running" not in st.session_state:
    st.session_state.running = False


@st.cache_resource
def load_model():
    """models/best.pt ho to wo, warna default yolov8n.pt (person + cell phone)."""
    path = "models/best.pt"
    if os.path.exists(path):
        return YOLO(path)
    return YOLO("yolov8n.pt")


if "engine" not in st.session_state:
    st.session_state.engine = MonitorEngine(load_model(), "config.json")

engine = st.session_state.engine
engine.reload_config("config.json")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("""
        <div style='margin-bottom:4px;'>
            <div style='font-weight:800;font-size:1.15rem;color:#eafff2;letter-spacing:1px;'>
                AI EMPLOYEE MONITORING
            </div>
            <div style='font-size:0.78rem;color:#7fa290;'>YOLOv8 • Person & Phone Detection</div>
        </div>
    """, unsafe_allow_html=True)

    source = st.radio(
        "Source",
        ["Camera (Browser)", "Upload Video", "RTSP / IP Camera"],
        help="RTSP sirf local PC par chalega — Streamlit Cloud server aapke "
             "office ke camera tak nahi pahunch sakta.",
    )

    uploaded_file = None
    rtsp_url = ""
    if source == "Upload Video":
        uploaded_file = st.file_uploader("Video upload", type=["mp4", "avi", "mov", "mkv"])
    elif source == "RTSP / IP Camera":
        rtsp_url = st.text_input("RTSP URL", "rtsp://user:pass@192.168.1.10:554/stream")
        st.warning("⚠ Ye option sirf local PC par kaam karega (cloud nahi).", icon="⚠")

    if st.button("▶ Start", use_container_width=True, type="primary"):
        st.session_state.running = True
        st.rerun()
    if st.button("⏹ Stop", use_container_width=True):
        st.session_state.running = False
        st.rerun()

    side_ph = st.empty()

# ------------------------------------------------------------------ header
head_l, head_r = st.columns([5, 1])
head_l.markdown(
    "<div class='app-title'>🛡 AI Employee Monitoring</div>"
    "<div class='app-sub'>Real-time worker tracking — WORKING / PHONE USAGE / ABSENT</div>",
    unsafe_allow_html=True,
)

dash_ph = st.empty()   # stat cards
frame_ph = st.empty()  # video

# ------------------------------------------------------------------ render helpers
GREEN = "#32d74b"
RED = "#ff453a"
ORANGE = "#ff9f0a"


def _status_color(s):
    return GREEN if s == "WORKING" else RED if s in ("PHONE USAGE", "ABSENT") else ORANGE


def render_dashboard(statuses):
    ids = [k for k in statuses if k >= 0]
    working = sum(1 for k in ids if statuses[k] == "WORKING")
    phone = sum(1 for k in ids if statuses[k] == "PHONE USAGE")
    absent = sum(1 for k in ids if statuses[k] == "ABSENT")
    alerts = phone + absent

    def card(title, val, color):
        return (f"<div class='card'><div class='c-title'>{title}</div>"
                f"<div class='c-val' style='color:{color};'>{val}</div></div>")

    cards = (
        "<div style='display:flex;gap:12px;margin:10px 0 14px;'>"
        + card("TOTAL IDS", len(ids), "#eafff2")
        + card("WORKING", working, GREEN)
        + card("PHONE USAGE", phone, RED)
        + card("ABSENT", absent, ORANGE)
        + card("ALERTS", alerts, RED if alerts else GREEN)
        + "</div>"
    )
    dash_ph.markdown(cards, unsafe_allow_html=True)

    rows = "".join(
        f"<div class='p-row'><span class='dot' style='background:{_status_color(s)};'></span>"
        f"ID {k} - {s}</div>"
        for k, s in sorted(statuses.items()) if k >= 0
    ) or "<div class='p-row' style='color:#7fa290;'>No IDs detected yet…</div>"

    side_ph.markdown(
        f"<div class='panel'>"
        f"<div class='p-title'>AI EMPLOYEE MONITORING</div>"
        f"<div class='p-live'>● LIVE FEED</div>"
        f"{rows}"
        f"<div class='p-foot'>Working: <b style='color:{GREEN};'>{working}</b> "
        f"&nbsp; Alerts: <b style='color:{RED if alerts else GREEN};'>{alerts}</b></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def show_frame(img_bgr):
    frame_ph.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)


# ------------------------------------------------------------------ camera (browser)
def run_camera():
    holder = {"statuses": {}, "alerts": 0}

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            annotated, statuses, alerts = engine.process(img)
            holder["statuses"] = statuses
            holder["alerts"] = alerts
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception:
            return frame

    ctx = webrtc_streamer(
        key="employee-cam",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"width": {"ideal": 1280}}, "audio": False},
        async_processing=True,
    )

    while ctx.state.playing:
        render_dashboard(holder["statuses"])
        time.sleep(0.5)


# ------------------------------------------------------------------ video file / rtsp
def run_stream(cap, cleanup=None):
    if not cap.isOpened():
        st.error("Source open nahi hua. URL / file check karein.")
        st.session_state.running = False
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, statuses, _ = engine.process(frame)
        show_frame(annotated)
        render_dashboard(statuses)

    cap.release()
    if cleanup and os.path.exists(cleanup):
        os.remove(cleanup)
    st.session_state.running = False
    frame_ph.info("Feed khatam ho gaya. Dobara **Start** dabayen.")


# ------------------------------------------------------------------ main flow
if st.session_state.running:
    if source == "Camera (Browser)":
        run_camera()
    elif source == "Upload Video":
        if uploaded_file is None:
            st.warning("Pehle sidebar se video upload karein.", icon="⚠")
            st.session_state.running = False
        else:
            tpath = "temp_video.mp4"
            with open(tpath, "wb") as f:
                f.write(uploaded_file.read())
            run_stream(cv2.VideoCapture(tpath), cleanup=tpath)
    else:
        run_stream(cv2.VideoCapture(rtsp_url))
else:
    st.markdown(
        "<div class='panel' style='text-align:center;padding:28px;'>"
        "<div style='font-size:2.2rem;'>🎥</div>"
        "<div style='margin-top:8px;font-size:0.95rem;'>"
        "Sidebar se source select karein aur <b>▶ Start</b> dabayen.</div>"
        "<div style='margin-top:6px;font-size:0.8rem;color:#7fa290;'>"
        "🟢 WORKING &nbsp; 🔴 PHONE USAGE / ABSENT &nbsp; 🟡 phone</div>"
        "</div>",
        unsafe_allow_html=True,
    )
