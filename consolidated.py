import os
import cv2
import numpy as np
import pickle

# Define paths
project_root = "C:/Users/prade/Machine Learning/Python CTTC/smart Attendance system/smart Attendance system"
data_dir = os.path.join(project_root, "C:/Users/prade/Machine Learning/Python CTTC/smart Attendance system/smart Attendance system/clean_data")
img_dir = os.path.join(project_root, "C:/Users/prade/Machine Learning/Python CTTC/smart Attendance system/smart Attendance system/images")

# Create the data directory if not exist
os.makedirs(data_dir, exist_ok=True)

# Preprocessing function
def preprocess(image):
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

images = []
labels = []

# Walk through each subfolder (person)
for person_name in os.listdir(img_dir):
    person_folder = os.path.join(img_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    for file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Couldn't read image: {img_path}")
            continue
        image = preprocess(image)
        images.append(image)
        labels.append(person_name)

# Convert to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Save with pickle
with open(os.path.join(data_dir, 'images.p'), 'wb') as f:
    pickle.dump(images, f)

with open(os.path.join(data_dir, 'labels.p'), 'wb') as f:
    pickle.dump(labels, f)

print(f"[✅ INFO] Saved {len(images)} preprocessed images and labels to '{data_dir}'")
