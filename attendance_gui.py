#107
import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import face_recognition
import pickle
import os
import threading
import datetime
import pandas as pd
import numpy as np

# Constants
ENCODINGS_PATH = 'dataset/encodings.pkl'
ATTENDANCE_CSV = 'attendance.csv'
ATTENDANCE_XLSX = 'attendance.xlsx'
SNAPSHOT_DIR = 'snapshots'
THRESHOLD_MINUTES = 3  # Set 30 for real session
CAMERA_INDEX = 0  #  = default laptop camera

# Globals
video_capture = None
known_face_encodings = []
known_face_names = []
presence_log = {}
snapshot_saved = {}
session_date = datetime.date.today().isoformat()
running = False

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs("dataset", exist_ok=True)


def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(ENCODINGS_PATH):
        return

    with open(ENCODINGS_PATH, 'rb') as f:
        data = pickle.load(f)
        for name, encodings in data.items():
            for enc in encodings:
                if isinstance(enc, np.ndarray) and enc.shape == (128,):
                    known_face_encodings.append(enc)
                    known_face_names.append(name)


def mark_attendance(name, frame):
    now = datetime.datetime.now().strftime("%H:%M")
    if name not in presence_log:
        presence_log[name] = set()
        filename = os.path.join(SNAPSHOT_DIR, f"{name}_{session_date}.jpg")
        cv2.imwrite(filename, frame)
        snapshot_saved[name] = filename
    presence_log[name].add(now)


def save_attendance():
    print("[INFO] Saving attendance...")
    rows = []
    for name, minutes in presence_log.items():
        count = len(minutes)
        status = "Present" if count >= THRESHOLD_MINUTES else "Absent"
        path = snapshot_saved.get(name, "None")
        rows.append([name, count, status, session_date, path])

    df = pd.DataFrame(rows, columns=["Name", "Minutes Present", "Status", "Date", "Snapshot Path"])
    df.to_csv(ATTENDANCE_CSV, index=False)
    df.to_excel(ATTENDANCE_XLSX, index=False)
    print(f"[INFO] Attendance saved to {ATTENDANCE_CSV} and {ATTENDANCE_XLSX}")


def process_video(label):
    global video_capture, running
    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    print("[INFO] Starting video stream. Press 'q' to quit...")

    while running:
        ret, frame = video_capture.read()
        if not ret:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    print(f"[DEBUG] Matched: {name}")
                    mark_attendance(name, frame)

                    top, right, bottom, left = [v * 4 for v in face_location]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        #Scv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if video_capture:
        video_capture.release()
    cv2.destroyAllWindows()


def start_attendance(label):
    global running
    if running:
        return
    running = True
    threading.Thread(target=process_video, args=(label,), daemon=True).start()


def stop_attendance():
    global running
    if running:
        running = False
        print("[INFO] Stopping attendance and saving file...")
        save_attendance()
        messagebox.showinfo("Attendance", "Attendance saved to CSV and Excel.")


def register_new_user():
    name = simpledialog.askstring("New Registration", "Enter student name:")
    if not name:
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        return

    encodings = []
    count = 0
    messagebox.showinfo("Info", "Capturing 5 face samples. Please look at the camera.")

    while count < 5:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            encodings.append(face_encodings[0])
            count += 1
            cv2.putText(frame, f"Captured {count}/5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Registering Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count < 5:
        messagebox.showerror("Error", "Not enough samples captured.")
        return

    # Save to file
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}

    data[name] = encodings

    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump(data, f)

    messagebox.showinfo("Success", f"User '{name}' registered successfully!")
    load_known_faces()


class AttendanceApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition Attendance System")
        master.geometry("400x300")

        self.label = tk.Label(master, text="Face Recognition Attendance", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.start_btn = tk.Button(master, text="Start Attendance", command=lambda: start_attendance(self.label))
        self.start_btn.pack(pady=10)

        self.stop_btn = tk.Button(master, text="Stop & Save", command=stop_attendance)
        self.stop_btn.pack(pady=10)

        self.delete_btn = tk.Button(master, text="Delete Registered User", command=self.delete_user_popup)
        self.delete_btn.pack(pady=10)

        self.quit_btn = tk.Button(master, text="Quit", command=master.quit)
        self.quit_btn.pack(pady=10)

        load_known_faces()
        print("[INFO] GUI Ready.")

    def delete_user_popup(self):
        popup = tk.Toplevel(self.master)
        popup.title("Delete User")

        tk.Label(popup, text="Enter user name to delete:").pack(pady=5)
        name_entry = tk.Entry(popup)
        name_entry.pack(pady=5)

        def confirm_delete():
            name = name_entry.get()
            if not name:
                messagebox.showwarning("Input Error", "Please enter a name.")
                return
            success = self.delete_user(name)
            if success:
                messagebox.showinfo("Success", f"User '{name}' deleted.")
                popup.destroy()
            else:
                messagebox.showerror("Error", f"User '{name}' not found.")

        tk.Button(popup, text="Delete", command=confirm_delete).pack(pady=10)

    def delete_user(self, user_name):
        if not os.path.exists(ENCODINGS_PATH):
            print("[ERROR] No encodings file found.")
            return False

        with open(ENCODINGS_PATH, 'rb') as f:
            data = pickle.load(f)

        if user_name not in data:
            return False

        # Delete from encodings
        del data[user_name]

        with open(ENCODINGS_PATH, 'wb') as f:
            pickle.dump(data, f)

        # Delete snapshots (if any)
        for file in os.listdir(SNAPSHOT_DIR):
            if file.startswith(f"{user_name}_"):
                os.remove(os.path.join(SNAPSHOT_DIR, file))

        print(f"[INFO] User '{user_name}' deleted.")
        return True


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()