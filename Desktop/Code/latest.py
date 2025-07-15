import cv2
import face_recognition
import sqlite3
import datetime
import numpy as np
import pickle

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
        except:
            continue
    return names, encodings

known_names, known_encodings = load_known_faces()

# ========== CAMERA SETUP ==========
rtsp_url="rtsp://user:user%40123@202.53.92.62:8081/cam/realmonitor?channel=10&subtype=0"
video_capture = cv2.VideoCapture(0)  # Change to RTSP if needed
print("📡 RTSP feed connected. Press Q to quit.")

seen_face_ids = set()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Resize frame to 50% for performance and accuracy
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Face recognition
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
        else:
            # Prevent repeat unknowns in same session
            face_id = tuple(np.round(face_encoding, 2))
            if face_id in seen_face_ids:
                continue
            seen_face_ids.add(face_id)

            # Prompt to label
            cv2.imshow("Unknown Face", frame)
            print("🆕 Unknown face detected. Enter name or leave blank to skip:")
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

        # Draw box (scale coords)
        top, right, bottom, left = [v * 2 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
conn.close()
