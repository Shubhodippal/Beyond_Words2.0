import streamlit as st
import cv2
from PIL import Image
import numpy as np
from keras.models import model_from_json

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

@st.cache
def load_model():
    return model

@st.cache
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
        return roi_gray

def main():
    st.title('Emotion Detection with Streamlit')

    run_detection = st.checkbox('Run Emotion Detection')
    if run_detection:
        cap = cv2.VideoCapture("EMOTION480.mp4")
        model = load_model()

        # Create an empty placeholder for the video frame
        video_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces and emotions
            face = detect_faces(frame)
            if face is not None:
                pred = model.predict(face)
                emotion_label = labels[np.argmax(pred)]
                cv2.putText(frame, emotion_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the processed frame
            video_placeholder.image(frame, channels="BGR", use_column_width=True)
    
        cap.release()

if __name__ == "__main__":
    main()
