import tkinter as tk
import cv2
from PIL import Image, ImageTk
from keras.models import model_from_json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import subprocess

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        # self.master.geometry("500x500")
        self.cap = cv2.VideoCapture("EMOTION480.mp4")
        self.out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

        self.canvas = tk.Canvas(window, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.button = tk.Button(root, text="Analyze File", command=self.analyze_file)
        self.button.pack()
        self.file = open("output.txt", "w")
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.mainloop()

    def analyze_file(self):
        root.destroy()
        subprocess.run(["python", "analyze.py"])
        sys.exit()

    def update(self):
        with open("output.txt", "a") as file:
            ret, im = self.cap.read()
            if ret:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (p, q, r, s) in faces:
                    image = gray[q:q + s, p:p + r]
                    cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
                    image = cv2.resize(image, (48, 48))
                    img = extract_features(image)
                    pred = model.predict(img)
                    prediction_label = labels[pred.argmax()]
                    print("Predicted Output:", prediction_label)
                    print(type(prediction_label))
                    cv2.putText(im, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                                (0, 0, 255))
                    file.write(f"{str(prediction_label)}\n")
                self.out.write(im)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)

    def next_question(self):
        self.current_question_index = (self.current_question_index + 1) % len(self.questions)
        self.lbl_question.config(text=self.questions[self.current_question_index])

    def on_close(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


root = tk.Tk()
app = CameraApp(root, "Camera App", )
