import streamlit as st
import cv2
from PIL import Image
import numpy as np

st.title("Face Detection App (No XML Needed)")

# Use OpenCV's built-in Haar cascade path
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert("RGB"))

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display result
    st.image(img_array, caption=f"Detected {len(faces)} face(s)", use_column_width=True)
