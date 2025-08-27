import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from mtcnn import MTCNN

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Initialize MTCNN detector
detector = MTCNN()

def preprocess_face(face_img):
    """Preprocess face: resize and normalize for DeepFace"""
    face_img = cv2.resize(face_img, (224, 224))  # DeepFace expects 224x224
    return face_img

def detect_emotions(frame):
    """Detect emotions on a single frame/image using MTCNN + DeepFace."""
    if frame is None or frame.size == 0:
        return frame, []

    frame_resized = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    results = []
    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        w, h = max(10, w), max(10, h)

        face_img = rgb_frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        face_img = preprocess_face(face_img)

        try:
            analysis = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='mtcnn'
            )
            if isinstance(analysis, list):
                analysis = analysis[0]
            emotion = analysis['dominant_emotion']
            results.append((x, y, w, h, emotion))

            # Draw results
            cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame_resized, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except Exception as e:
            print("⚠️ DeepFace could not analyze:", e)

    return frame_resized, results

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Mood Mirror 🎭", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF6347;'>🎭 Mood Mirror: Real-Time Emotion Detection</h1>", unsafe_allow_html=True)

mode = st.sidebar.radio("Choose Mode", ["Upload Image", "Webcam"])

# Upload Image
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        output, results = detect_emotions(img)
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        if results:
            st.subheader("Detected Emotions:")
            for (_, _, _, _, emotion) in results:
                st.write(f"- {emotion}")
        else:
            st.warning("⚠️ No face detected.")


# Webcam with streamlit-webrtc
elif mode == "Webcam":
    st.warning("⚠️ Webcam feature may not work on all cloud deployments due to network constraints.")
    st.info("For best results, run this app locally for webcam functionality.")
    
    # Alternative: Use Streamlit's native camera input instead of WebRTC
    picture = st.camera_input("Take a picture")
    
    if picture:
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        output, results = detect_emotions(img)
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        
        if results:
            st.subheader("Detected Emotions:")
            for (_, _, _, _, emotion) in results:
                st.write(f"- {emotion}")
        else:
            st.warning("⚠️ No face detected.")
