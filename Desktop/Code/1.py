import cv2
import face_recognition
import sqlite3
import datetime
import numpy as np
import pickle

# ========== STEP 1: Set Up Database ==========
conn = sqlite3.connect("face_recognition.db")
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS known_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    encoding BLOB,
    time_added TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    time_identified TEXT
)
''')

conn.commit()

# ========== STEP 2: Load Encodings from DB ==========
def load_known_faces():
    cursor.execute("SELECT name, encoding FROM known_faces")
    rows = cursor.fetchall()
    names = []
    encodings = []
    for name, encoding_blob in rows:
        encoding = pickle.loads(encoding_blob)
        names.append(name)
        encodings.append(encoding)
    return names, encodings

# ========== STEP 3: Main Loop ==========
video_capture = cv2.VideoCapture(0)
print("📷 Starting live feed. Press 'Q' to quit.")

known_names, known_encodings = load_known_faces()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
        else:
            # Prompt user to label the face
            cv2.imshow("Unknown Face - Label It", frame)
            print("\n🆕 New face detected. Enter name for the new person (or leave blank to skip):")
            new_name = input("Name: ").strip()

            if new_name:
                name = new_name
                # Store in DB
                encoded_blob = pickle.dumps(face_encoding)
                time_added = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("INSERT INTO known_faces (name, encoding, time_added) VALUES (?, ?, ?)",
                               (name, encoded_blob, time_added))
                conn.commit()

                known_names.append(name)
                known_encodings.append(face_encoding)

        # Log recognition time
        time_identified = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO logs (name, time_identified) VALUES (?, ?)", (name, time_identified))
        conn.commit()

        # Draw label on screen
        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
conn.close()
