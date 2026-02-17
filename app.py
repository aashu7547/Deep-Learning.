import cv2
import tempfile
import streamlit as st
import pandas as pd
from ultralytics import YOLO

st.set_page_config(page_title="YOLO11 Roadside Detection", layout="wide")
st.title("🚦 YOLO11 Roadside Detection + Counting")

# Sidebar controls
st.sidebar.header("Options")
mode = st.sidebar.radio("Input Mode", ["Webcam", "Video File"])
model_choice = st.sidebar.selectbox("YOLO11 Model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Buttons
start_button = st.sidebar.button("▶️ Start Detection")
stop_button = st.sidebar.button("⏹️ Stop Detection")
refresh_button = st.sidebar.button("🔄 Refresh")

if "running" not in st.session_state:
    st.session_state.running = False
if "frame_data" not in st.session_state:
    st.session_state.frame_data = []
if "video_file" not in st.session_state:
    st.session_state.video_file = None

if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False
if refresh_button:
    st.session_state.running = False
    st.session_state.frame_data = []
    st.session_state.video_file = None
    st.experimental_rerun()

# Load model
model = YOLO(model_choice)

# Video source
source = 0 if mode == "Webcam" else None
if mode == "Video File":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        source = temp_file.name

# Display areas
stframe = st.empty()
chart_placeholder = st.empty()

# Detection loop
if st.session_state.running and source is not None:
    cap = cv2.VideoCapture(source)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(out_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence_threshold)

        car_count = person_count = bus_count = truck_count = bike_count = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = model.names.get(cls, str(cls))

                if label == "car":
                    car_count += 1
                elif label == "person":
                    person_count += 1
                elif label == "bus":
                    bus_count += 1
                elif label == "truck":
                    truck_count += 1
                elif label in ["bicycle", "motorcycle"]:
                    bike_count += 1

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), max(15, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Overlay counts
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        cv2.putText(frame, f"Cars: {car_count}", (20, 40), font, scale, (0, 255, 0), thickness)
        cv2.putText(frame, f"Persons: {person_count}", (20, 80), font, scale, (255, 0, 0), thickness)
        cv2.putText(frame, f"Buses: {bus_count}", (20, 120), font, scale, (0, 255, 255), thickness)
        cv2.putText(frame, f"Trucks: {truck_count}", (20, 160), font, scale, (255, 255, 0), thickness)
        cv2.putText(frame, f"Bikes: {bike_count}", (20, 200), font, scale, (255, 0, 255), thickness)

        # Save frame to video
        out.write(frame)

        # Save frame data
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        st.session_state.frame_data.append({
            "frame": frame_id,
            "cars": car_count,
            "persons": person_count,
            "buses": bus_count,
            "trucks": truck_count,
            "bikes": bike_count
        })

        # Show frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        # Live chart
        df_live = pd.DataFrame(st.session_state.frame_data)
        if not df_live.empty:
            chart_placeholder.line_chart(df_live.set_index("frame"))

    cap.release()
    out.release()
    st.session_state.video_file = out_file.name
    st.session_state.running = False

# Download buttons and summary after detection
if st.session_state.frame_data and st.session_state.video_file:
    st.success("✅ Detection finished. Download your results below:")
    with open(st.session_state.video_file, "rb") as f:
        st.download_button("📥 Download Processed Video", f, file_name="processed_output.mp4")
    df = pd.DataFrame(st.session_state.frame_data)
    csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(csv_file.name, index=False)
    with open(csv_file.name, "rb") as f:
        st.download_button("📊 Download CSV Counts", f, file_name="counts.csv")

    # Summary dashboard
    st.subheader("📈 Summary Dashboard")
    st.bar_chart(df.drop(columns="frame").sum())