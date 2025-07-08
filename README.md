# Smart Attendance System using Deep Learning

An AI-based face recognition system that marks student attendance in real-time using a webcam or mobile camera. It uses CNN for face recognition, OpenCV for face detection, and stores attendance with timestamps in CSV files.

---

## 📌 Project Features

- 🔍 Face detection using Haar Cascade
- 🧠 CNN-based face recognition model
- 📷 Real-time face capture from webcam/mobile
- 🗓️ Attendance logging in CSV with date & time
- 🧾 Linked with student registration database (CSV)
- 🗂️ Organized student image dataset (by name)
- 📊 Overfitting reduced with data augmentation, dropout
- 🔁 Model training with validation & early stopping

---

## 🧠 Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- Haarcascade Classifier
- Pandas & CSV
- LabelEncoder & Pickle
- ImageDataGenerator

---

## 🗂️ Project Structure

smart-attendance-system/
├── data/ # Student face images
│ └── student_name/
├── haarcascades/
│ └── haarcascade_frontalface_default.xml
├── model/
│ └── trained_cnn_model.h5
├── attendance.csv
├── students.csv
├── capture_and_train.py
├── mark_attendance.py
├── README.md
└── .gitignore

---

## 🚀 How It Works

1. Student images are collected and organized in folders.
2. Haar cascade detects faces; CNN model is trained.
3. On recognition, attendance is saved with name, reg. number, and time.
4. Attendance is logged in a daily CSV file.

---

## 📝 Dataset

- Real-time webcam face capture
- Kaggle face datasets (augmented)
- Images stored per student folder (100+ per student)

---

## 📈 Model Performance

- Custom CNN trained with `categorical_crossentropy`
- Image size: 100x100 grayscale
- Training epochs: 20+
- Achieved accuracy ~65%+
- Overfitting minimized using:
  - `ImageDataGenerator`
  - `Dropout`
  - `EarlyStopping`

---

## 📚 References

- [OpenCV Docs](https://docs.opencv.org/)
- [Keras ImageDataGenerator](https://keras.io/api/preprocessing/image/)
- [Haar Cascade GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- [Kaggle Face Datasets](https://www.kaggle.com/)

---

## 📬 Contact

📧 mr.pradeepkumarsingha@gmail.com
🔗 [GitHub Profile](https://github.com/pradeepkumarsingha)

---

