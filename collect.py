import cv2
import urllib
import numpy as np
import os

# Load Haar cascade classifier
classifier = cv2.CascadeClassifier(
    r"C:/Users/prade/Machine Learning/Python CTTC/smart Attendance system/smart Attendance system/haarcascade_frontalface_default.xml"
)

url = "http://192.0.0.4:8080/shot.jpg"
data = []

# Input user name
name = input("Enter user name: ").strip()

# Create user-specific folder inside 'images'
user_folder = os.path.join("images", name)
os.makedirs(user_folder, exist_ok=True)

while len(data) < 150:
    try:
        print('Connecting to camera...')
        image_from_url = urllib.request.urlopen(url)
        frame = np.array(bytearray(image_from_url.read()), np.uint8)
        frame = cv2.imdecode(frame, -1)
    except:
        print("⚠️ Could not read from camera URL.")
        continue

    faces = classifier.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        face_frame = frame[y:y+h, x:x+w]
        face_frame = cv2.resize(face_frame, (128, 128))
        cv2.imshow("Face", face_frame)
        
        data.append(face_frame)
        print(f"{len(data)} / 100 captured")
        
        if len(data) >= 150:
            break

    cv2.imshow("Preview Window", frame)
    if cv2.waitKey(25) == 27:
        break

cv2.destroyAllWindows()

# Save captured images to user folder
if len(data) == 150:
    for i in range(150):
        img_path = os.path.join(user_folder, f"{name}_{i+1}.jpg")
        cv2.imwrite(img_path, data[i])
    print(f"✅ Completed. 100 images saved to '{user_folder}'")
else:
    print("⚠️ Insufficient data collected.")
