import cv2
import face_recognition
import sqlite3
import datetime
import numpy as np
import pickle
import time

# ========== DB SETUP ==========
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

# ========== LOAD ENCODINGS FROM DB ==========
def load_known_faces():
    cursor.execute("SELECT name, encoding FROM known_faces")
    rows = cursor.fetchall()
    names = []
    encodings = []
    for name, encoding_blob in rows:
        try:
            encoding = pickle.loads(encoding_blob)
            names.append(name)
            encodings.append(encoding)
        except Exception as e:
            print("Error loading face:", e)
            continue
    return names, encodings

known_names, known_encodings = load_known_faces()

# ========== CAMERA SETUP ==========
rtsp_url = "rtsp://user:user%40123@202.53.92.62:8081/cam/realmonitor?channel=10&subtype=0"
video_capture = cv2.VideoCapture(rtsp_url)
print("📡 RTSP feed connected. Press Q to quit.")

seen_face_ids = set()

# ========== MAIN LOOP ==========
while True:
    ret, frame = video_capture.read()
    if not ret or frame is None or frame.shape[0] < 50:
        print("❌ Failed to grab frame or frame too small.")
        time.sleep(1)
        continue

    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

    # Detect faces using CNN
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
        else:
            face_id = tuple(np.round(face_encoding, 2))
            if face_id in seen_face_ids:
                continue
            seen_face_ids.add(face_id)

            # Show prompt only once per face
            print("🆕 Unknown face detected. Enter name or leave blank to skip:")
            cv2.imshow("Unknown Face", frame)
            new_name = input("Name: ").strip()
            cv2.destroyWindow("Unknown Face")

            if new_name:
                name = new_name
                encoded_blob = pickle.dumps(face_encoding)
                time_added = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("INSERT INTO known_faces (name, encoding, time_added) VALUES (?, ?, ?)",
                               (name, encoded_blob, time_added))
                conn.commit()
                known_names.append(name)
                known_encodings.append(face_encoding)

        # Log recognition
        time_identified = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO logs (name, time_identified) VALUES (?, ?)", (name, time_identified))
        conn.commit()

        # Draw rectangle
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display video
    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
conn.close()
