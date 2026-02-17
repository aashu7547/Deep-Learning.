import cv2
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import time
from ultralytics import YOLO

st.set_page_config(page_title="YOLO11 Roadside Detection", layout="wide")
st.title("🚦 YOLO11 Roadside Detection + Counting")

# Sidebar controls
st.sidebar.header("Options")
mode = st.sidebar.radio("Input Mode", ["Webcam", "Video File", "Image File"])
model_choice = st.sidebar.selectbox("YOLO11 Model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Buttons
start_button = st.sidebar.button("▶️ Start Detection")
stop_button = st.sidebar.button("⏹️ Stop Detection")
refresh_button = st.sidebar.button("🔄 Refresh")

# Session state
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
    st.rerun()

# Load model
model = YOLO(model_choice)

# Display areas
stframe = st.empty()
chart_placeholder = st.empty()
summary_placeholder = st.empty()
download_placeholder = st.empty()
csv_table_placeholder = st.empty()

# IMAGE MODE
if mode == "Image File":
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = uploaded_image.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(frame, conf=confidence_threshold)

        counts = {"cars":0,"persons":0,"buses":0,"trucks":0,"bikes":0}
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = model.names.get(cls, str(cls))

                if label == "car": counts["cars"]+=1
                elif label == "person": counts["persons"]+=1
                elif label == "bus": counts["buses"]+=1
                elif label == "truck": counts["trucks"]+=1
                elif label in ["bicycle","motorcycle"]: counts["bikes"]+=1

                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                cv2.putText(frame,f"{label} {conf:.2f}",(int(x1),max(15,int(y1)-10)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", caption="Detected Objects")

        # Show counts
        st.write("### Object Counts")
        st.write(counts)

        # Save CSV
        df_img = pd.DataFrame([counts])
        csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_img.to_csv(csv_file.name, index=False)
        with open(csv_file.name, "rb") as f:
            st.download_button("📊 Download Image Detection CSV", f, file_name="image_counts.csv")

# VIDEO/WEBCAM MODE
if st.session_state.running and mode in ["Webcam","Video File"]:
    source = 0 if mode == "Webcam" else None
    if mode == "Video File":
        uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            temp_file.flush()
            source = temp_file.name

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("❌ Unable to open video source.")
    else:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(out_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            if not st.session_state.running:
                break

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

                    if label == "car": car_count+=1
                    elif label == "person": person_count+=1
                    elif label == "bus": bus_count+=1
                    elif label == "truck": truck_count+=1
                    elif label in ["bicycle","motorcycle"]: bike_count+=1

                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                    cv2.putText(frame,f"{label} {conf:.2f}",(int(x1),max(15,int(y1)-10)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            # Overlay counts
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            cv2.putText(frame,f"Cars: {car_count}",(20,40),font,scale,(0,255,0),thickness)
            cv2.putText(frame,f"Persons: {person_count}",(20,80),font,scale,(255,0,0),thickness)
            cv2.putText(frame,f"Buses: {bus_count}",(20,120),font,scale,(0,255,255),thickness)
            cv2.putText(frame,f"Trucks: {truck_count}",(20,160),font,scale,(255,255,0),thickness)
            cv2.putText(frame,f"Bikes: {bike_count}",(20,200),font,scale,(255,0,255),thickness)

            out.write(frame)

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            st.session_state.frame_data.append({
                "frame": frame_id,
                "cars": car_count,
                "persons": person_count,
                "buses": bus_count,
                "trucks": truck_count,
                "bikes": bike_count
            })

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

            df_live = pd.DataFrame(st.session_state.frame_data)
            if not df_live.empty:
                chart_placeholder.line_chart(df_live.set_index("frame"))
                summary_placeholder.bar_chart(df_live.drop(columns="frame").sum())
                csv_table_placeholder.dataframe(df_live.tail(20))
                df_live.to_csv("counts.csv", index=False)
                with open(out_file.name, "rb") as f:
                    download_placeholder.download_button("📥 Download Processed Video", f, file_name="processed_output.mp4")
                with open("counts.csv", "rb") as f:
                    download_placeholder.download_button("📊 Download CSV Counts", f, file_name="counts.csv")

            time.sleep(0.01)  # yield control back to Streamlit

        cap.release()
        out.release()
        st.session_state.video_file = out_file.name
        st.session_state.running = False