import streamlit as st
import base64
import cv2
import numpy as np
import time
from PIL import Image

st.set_page_config(page_title="Pallathulli Peruvalam", layout="wide")

def add_bg_video():
    video_path = "static/bg.mp4"
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
    b64_video = base64.b64encode(video_bytes).decode()

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;500;700&display=swap');
        video {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
            object-fit: cover;
            opacity: 0.9;
        }}
        .stApp {{
            background: transparent;
            font-family: 'Montserrat', sans-serif;
        }}
        .centered {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            z-index: 10;
        }}
        .title {{
            color: #ffffff;
            font-size: 2.7rem;
            text-shadow: 2px 2px 10px #000;
            margin-bottom: 2rem;
        }}
        .counter-display {{
            display: block;
            color: #ffffff;
            font-size: 1.5rem;
            margin-top: 25px;
            text-shadow: 1px 1px 3px #000;
            font-weight: 400;
        }}
        .upload-wrapper {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 2rem;
        }}
        div[data-testid="stFileUploader"] > label {{
            font-size: 14px;
            padding: 6px 14px;
            background-color: rgba(255, 255, 255, 0.2);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.4);
            border-radius: 8px;
            backdrop-filter: blur(6px);
            font-weight: 500;
            cursor: pointer;
        }}
        .camera-btn {{
            font-size: 18px;
            padding: 8px 14px;
            background-color: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.4);
            border-radius: 8px;
            backdrop-filter: blur(6px);
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }}
        .camera-btn:hover {{
            background-color: rgba(255, 255, 255, 0.35);
            color: black;
        }}
        </style>
        <video autoplay loop muted>
            <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)

add_bg_video()

st.markdown('<div class="centered">', unsafe_allow_html=True)
st.markdown('<div class="title">Pallathulli Peruvalam 💧</div>', unsafe_allow_html=True)
st.markdown('<p class="counter-display">Upload a video and watch each drop get counted.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="upload-wrapper">
    <div id="file-upload-holder"></div>
    <button class="camera-btn">📷</button>
</div>
<script>
const uploaderContainer = parent.document.querySelector('div[data-testid="stFileUploader"]');
const placeholder = parent.document.getElementById('file-upload-holder');
if (uploaderContainer && placeholder) {
    placeholder.appendChild(uploaderContainer);
}
</script>
""", unsafe_allow_html=True)

video_file = st.file_uploader("", type=["mp4", "avi", "mov"], label_visibility="collapsed")

if video_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    drop_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())

    cap.release()

    if not frames:
        st.error("Failed to load frames")
    else:
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        last_drop_time = time.time()
        slow_repeat = 2  # repeat each frame 2x to slow video to 50%

        col1, col2 = st.columns([3, 1])
        with col1:
            frame_placeholder = st.empty()
        with col2:
            drop_placeholder = st.empty()

        desired_fps = 30
        frame_interval = 1.0 / desired_fps

        for frame in frames:
            for _ in range(slow_repeat):
                start_time = time.time()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                current_time = time.time()

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 100:
                        continue

                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area != 0 else 0
                    rect_area = w * h
                    extent = float(area) / rect_area if rect_area != 0 else 0

                    if current_time - last_drop_time >= 0.3:
                        if 0.3 < aspect_ratio < 1.2 and 0.6 < solidity < 0.95 and 0.4 < extent < 0.9:
                            drop_count += 1
                            last_drop_time = current_time
                            break

                prev_gray = gray.copy()
                thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                frame_rgb = cv2.cvtColor(thresh_bgr, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(frame_rgb, (800, 450))
                frame_pil = Image.fromarray(resized_frame)

                frame_placeholder.image(frame_pil, caption="Threshold View")
                drop_placeholder.markdown(
                    f'<div class="counter-display">💧 Drop Count: {drop_count}</div>',
                    unsafe_allow_html=True
                )

                elapsed = time.time() - start_time
                remaining = frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
