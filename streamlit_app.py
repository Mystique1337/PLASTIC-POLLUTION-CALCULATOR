import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
import tempfile
import time

# Load model
model = YOLO("best.pt")

# Streamlit UI Setup
st.set_page_config(page_title="Plastic Pollution Detector", layout="wide")
st.title("♻️ Real-Time Plastic Pollution Detector")

# Sidebar: Input Source
st.sidebar.header("🎥 Input Source")
source_type = st.sidebar.radio("Choose input source", ["Webcam", "Upload Video"])

# Sidebar: PPI Threshold
threshold = st.sidebar.slider("🚨 Alert Threshold (% PPI)", min_value=1, max_value=100, value=50)

# Main layout
status_placeholder = st.empty()
frame_placeholder = st.empty()
ppi_placeholder = st.empty()
chart_placeholder = st.empty()

# Trend Data
ppi_history = []

def process_frame(frame, frame_area):
    """Detect plastics and calculate accurate PPI using non-overlapping mask"""
    results = model.predict(frame, verbose=False)[0]
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # Fill the mask with white in bounding box region (handle overlaps)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
            # Draw on actual frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            cv2.putText(frame, "Plastic", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # Count non-zero pixels in mask (i.e., all unique plastic area)
    total_plastic_area = np.count_nonzero(mask)
    ppi = total_plastic_area / frame_area
    ppi_percent = round(ppi * 100, 2)

    # Draw PPI bar
    bar_length = 300
    filled = int(min(ppi, 1.0) * bar_length)
    cv2.rectangle(frame, (20, 20), (20 + bar_length, 50), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 20), (20 + filled, 50), (0, 165, 255), -1)
    cv2.rectangle(frame, (20, 20), (20 + bar_length, 50), (255, 255, 255), 2)
    cv2.putText(frame, f"PPI: {ppi_percent}%", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, ppi_percent

def display_output(frame, ppi):
    """Display frame, metric, alert and trend chart"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    ppi_placeholder.metric("📊 Plastic Pollution Index (PPI)", f"{ppi:.2f}%")
    if ppi >= threshold:
        status_placeholder.error(f"🚨 ALERT: High Plastic Pollution ({ppi:.2f}%)")
    else:
        status_placeholder.success(f"✅ Normal Pollution Level ({ppi:.2f}%)")

    ppi_history.append({"time": time.time(), "ppi": ppi})
    if len(ppi_history) > 30:  # Keep last 30 readings
        ppi_history.pop(0)
    chart_data = {"Time": [p["time"] for p in ppi_history],
                  "PPI (%)": [p["ppi"] for p in ppi_history]}
    chart_placeholder.line_chart(chart_data, y="PPI (%)")

# Webcam Mode
if source_type == "Webcam":
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_area = frame_width * frame_height

    stop_btn = st.sidebar.button("⛔ Stop")

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break
        frame, ppi = process_frame(frame, frame_area)
        display_output(frame, ppi)

    cap.release()
    status_placeholder.info("Webcam stream ended.")

# Video Upload Mode
else:
    uploaded_file = st.sidebar.file_uploader("📤 Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_area = frame_width * frame_height

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, ppi = process_frame(frame, frame_area)
            display_output(frame, ppi)
        cap.release()
        status_placeholder.info("Video processing completed.")
