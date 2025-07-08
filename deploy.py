import cv2
import urllib.request
import numpy as np
import pickle
import csv
import os
from datetime import datetime
from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load face detection model
classifier = cv2.CascadeClassifier(
    "C:/Users/prade/Machine Learning/Python CTTC/smart Attendance system/smart Attendance system/haarcascade_frontalface_default.xml"
)

# Load your trained CNN model
model = load_model(
    "C:/Users/prade/Machine Learning/Python CTTC/smart Attendance system/smart Attendance system/final_modell.h5"
    #"C:/Users/prade/Machine Learning/Python CTTC/smart Attendance system/smart Attendance system/efficientnet_face_attendance.h5"
)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load student names and regd. numbers from CSV
student_df = pd.read_csv(
    "C:/Users/prade/Machine Learning/Python CTTC/smart Attendance system/smart Attendance system/students.csv"
) 

# Attendance tracker
marked_students = set()

# Prepare daily attendance file
today = datetime.now().strftime("%Y-%m-%d")
filename = f"attendance_{today}.csv"
if not os.path.exists(filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Regd_No", "Date", "Time"])

# Attendance marking function
def mark_attendance(name):
    if name not in marked_students:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        regd = student_df.loc[student_df["Name"] == name, "Regd_No"].values
        if len(regd) > 0:
            with open(filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, regd[0], date, time])
            marked_students.add(name)
            print(f"[INFO] Attendance marked: {name} - {regd[0]}")

# Image preprocessing function

def preprocess(img):
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = img.reshape(1, 100, 100, 1)
    return img
'''
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = preprocess_input(img)  # ✅ scales to [-1, 1]
    img = np.expand_dims(img, axis=0)  # shape = (1, 128, 128, 3)
    return img
'''
# Set your mobile IP camera URL
#url = "http://192.0.0.4:8080/shot.jpg"  # Replace with actual IP
cap=cv2.VideoCapture(0)
print("📸 Starting attendance via mobile camera... Press ESC to stop.")

# Start attendance loop
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture from webcam.")
            break
        
        '''
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        '''
        faces = classifier.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            processed_face = preprocess(face)

            prediction = model.predict(processed_face)
            confidence = np.max(prediction)
            predicted_label = np.argmax(prediction)

            if confidence >= 0.90:
                name = label_encoder.inverse_transform([predicted_label])[0]

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{name} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                )

                # Mark attendance
                mark_attendance(name)
            else:
                # Draw unknown face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    frame, f"Unknown ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
                )

        cv2.imshow("Smart Attendance - Mobile Camera", frame)
        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print(f"[ERROR] {e}")
        continue

cv2.destroyAllWindows()
