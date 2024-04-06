import streamlit as st
import cv2
from PIL import Image
import numpy as np
from streamlit_webrtc import VideoTransformerBase, WebRTCState
from keras.models import model_from_json
import matplotlib.pyplot as plt
from collections import Counter

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

emotion=[]

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
@st.cache(allow_output_mutation=True)
def load_model():
    return model

@st.cache(allow_output_mutation=True)
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detected_labels = [] #empty list
    for (x, y, w, h) in faces:
        # Increase bounding box size by a factor of 1.5
        new_w = int(w * 1.5)
        new_h = int(h * 1.5)
        x_offset = int((new_w - w) / 2)  # Center the expanded box
        y_offset = int((new_h - h) / 2)
        #bounding box
        cv2.rectangle(image, (x - x_offset, y - y_offset), (x + new_w - x_offset, y + new_h - y_offset), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
        #pred = model.predict
        return roi_gray

def main():
    st.title('Beyond Words')

    cam_detect = st.checkbox('Detect from webcam')
    vid_detect = st.checkbox('Detect from video file')
    run_detection = st.button('Run')

    if (run_detection and (cam_detect or vid_detect)):
        if  cam_detect:
            cap = cv2.VideoCapture(0)
        elif vid_detect:
            cap = cv2.VideoCapture('EMOTION480.mp4')

        model = load_model()
        # Create an empty placeholder for the video frame
        video_placeholder = st.empty()
        #brk = st.button('Break')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                # Detect faces and emotions
                face = detect_faces(frame)
                if face is not None:
                    pred = model.predict(face)
                    emotion_label = labels[np.argmax(pred)]
                    emotion.append(labels[np.argmax(pred)])
                    cv2.putText(frame, emotion_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                # Display the processed frame
                video_placeholder.image(frame, channels="BGR", use_column_width=True)
                #if(brk):
                #    return emotion
                #    break
        cap.release()
        #analyze_list(emotion)
        return emotion

def analyze_list(input_list):
    # Count the elements in the list
    element_counts = Counter(input_list)
    total_count = len(input_list)
    
    # Calculate percentages
    percentages = {element: (count / total_count) * 100 for element, count in element_counts.items()}
    
    # Plot the percentages
    labels = list(percentages.keys())
    values = list(percentages.values())

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Elements')
    plt.ylabel('Percentage')
    plt.title('Percentage of Different Types of Elements')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot()

if __name__ == "__main__":
    emotion=main()
    if(emotion):
        #print(emotion)
        analyze_list(emotion)
        #print(emotion)