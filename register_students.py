import cv2
import face_recognition
import pickle
import os

ENCODINGS_PATH = 'dataset/encodings.pkl'
URL = 'http://10.229.237.92:8080/video'  # Replace with your actual phone IP
STUDENT_NAME = input("Enter student name: ").strip()

# Load existing encodings if they exist
if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, 'rb') as f:
        known_encodings = pickle.load(f)
else:
    known_encodings = {}

# Access phone camera stream
cap = cv2.VideoCapture(URL)

print("[INFO] Press 'c' to capture face, 'q' to quit")

captured_encodings = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame from phone camera.")
        break

    # Display frame
    cv2.imshow("Registration - Phone Cam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, boxes)

        if encodings:
            captured_encodings.extend(encodings)
            print(f"[INFO] Captured {len(encodings)} face(s) for {STUDENT_NAME}")
        else:
            print("[WARNING] No face detected.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to encodings file
if captured_encodings:
    if STUDENT_NAME in known_encodings:
        known_encodings[STUDENT_NAME].extend(captured_encodings)
    else:
        known_encodings[STUDENT_NAME] = captured_encodings

    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump(known_encodings, f)
    print(f"[INFO] Registered {STUDENT_NAME} with {len(captured_encodings)} encodings.")
else:
    print("[INFO] No face encodings captured.")
