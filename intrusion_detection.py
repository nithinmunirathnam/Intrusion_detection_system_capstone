import cv2
import numpy as np
import pyttsx3
import pickle

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Load trained recognizer and label map
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")

with open("labels.pickle", "rb") as f:
    label_map = pickle.load(f)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)

        if confidence < 70:
            name = label_map[label]
            cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            speak(f"Welcome {name}")
        else:
            cv2.putText(frame, "Intruder!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            speak("Intruder detected!")

    cv2.imshow("Intrusion Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
