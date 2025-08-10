#92
import cv2
import face_recognition
import pickle
import os
import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

# ====== Configurations ======
ENCODING_FILE = "dataset/encodings.pkl"
SNAPSHOT_DIR = "snapshots"
ATTENDANCE_CSV = "attendance.csv"
ATTENDANCE_XLSX = "attendance.xlsx"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ====== Load face encodings ======
with open(ENCODING_FILE, "rb") as f:
    raw_data = pickle.load(f)

known_face_encodings = []
known_face_names = []

for name, encodings in raw_data.items():
    for enc in encodings:
        if isinstance(enc, np.ndarray) and enc.shape == (128,):
            known_face_encodings.append(enc)
            known_face_names.append(name)

print(f"[INFO] Loaded {len(known_face_encodings)} face encodings.")

# ====== Setup tracking ======
presence_log = defaultdict(set)
snapshot_saved = {}
session_date = datetime.datetime.now().strftime("%Y-%m-%d")

# ====== Start webcam ======
print("[INFO] Starting video stream. Press 'q' to quit...")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    now = datetime.datetime.now()
    minute_str = now.strftime("%Y-%m-%d %H:%M")

    for face_encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        name = "Unknown"

        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        if name != "Unknown":
            presence_log[name].add(minute_str)

            if name not in snapshot_saved:
                top, right, bottom, left = location
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                face_crop = frame[top:bottom, left:right]
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.jpg"
                snapshot_path = os.path.join(SNAPSHOT_DIR, filename)
                cv2.imwrite(snapshot_path, face_crop)
                snapshot_saved[name] = snapshot_path

        # Draw bounding box
        top, right, bottom, left = location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

    #cv2.imshow('Attendance Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# ====== Save attendance ======
print("[INFO] Saving attendance...")
with open(ATTENDANCE_CSV, "w") as f:
    f.write("Name,Minutes Present,Status,Date,Snapshot Path\n")
    for name, minutes in presence_log.items():
        total_minutes = len(minutes)
        status = "Present" if total_minutes >= 3 else "Absent"
        snapshot = snapshot_saved.get(name, "None")
        f.write(f"{name},{total_minutes},{status},{session_date},{snapshot}\n")

# Convert CSV to Excel
df = pd.read_csv(ATTENDANCE_CSV)
df.to_excel(ATTENDANCE_XLSX, index=False)
print("[INFO] Attendance saved to CSV and Excel.")
