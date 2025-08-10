import cv2
import face_recognition
import pickle
import datetime

# ✅ Load face encodings
with open("dataset/encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_face_names = list(data.keys())
known_face_encodings = list(data.values())


# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Attendance record
attendance = {}

print("[INFO] Starting video stream. Press 'q' to quit...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to capture frame")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin() if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        if name != "Unknown" and name not in attendance:
            attendance[name] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[✅] {name} marked present at {attendance[name]}")

    # Show the frame
    cv2.imshow("Attendance System", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()

# Save attendance
with open("attendance.csv", "w") as f:
    f.write("Name,Time\n")
    for name, time in attendance.items():
        f.write(f"{name},{time}\n")

print("[INFO] Attendance saved to attendance.csv")
