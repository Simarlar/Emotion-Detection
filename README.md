# 😊 Emotion Detection App

This is a **real-time emotion detection app** using DeepFace and MTCNN.  
You can **upload an image** or use your **webcam** to detect emotions on faces.

---

## Features

- Detects faces in images and webcam frames.
- Recognizes emotions such as happy, sad, angry, surprised, etc.
- Real-time webcam support.
- Clean and user-friendly Streamlit interface.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Simarlar/emotion-detection.git
cd emotion-detection
Create a virtual environment (recommended):

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt

Usage

Run the Streamlit app:

streamlit run app.py


Choose Upload Image to detect emotions in an image.

Choose Webcam to detect emotions in real-time from your camera.

Press Stop Webcam to end the webcam session.

Requirements

Python 3.10+

Streamlit

OpenCV

DeepFace

MTCNN

Screenshots




License

This project is open-source and available under the MIT License.