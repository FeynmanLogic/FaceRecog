import cv2
import face_recognition
import sqlite3
import pickle
import datetime

# ========== Setup DB ==========
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
            print(f"❌ Failed to load encoding for {name}: {e}")
    return names, encodings


# ========== Start Webcam ==========
video = cv2.VideoCapture(0)
print("📷 Webcam started. Press Q to quit.")
known_names, known_encodings = load_known_faces()

while True:
    ret, frame = video.read()
    if not ret:
        continue

    # Resize + Convert to RGB
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # Detect and Encode
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for encoding, loc in zip(encodings, locations):
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            name = known_names[matches.index(True)]
        else:
            cv2.imshow("Label Unknown Face", frame)
            print("🆕 New face. Enter name (or leave blank):")
            new_name = input("Name: ").strip()
            if new_name:
                name = new_name
                blob = pickle.dumps(encoding)
                time_added = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("INSERT INTO known_faces (name, encoding, time_added) VALUES (?, ?, ?)",
                               (name, blob, time_added))
                conn.commit()
                known_names.append(name)
                known_encodings.append(encoding)

        # Log Recognition
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO logs (name, time_identified) VALUES (?, ?)", (name, timestamp))
        conn.commit()

        # Draw Label
        top, right, bottom, left = [v * 4 for v in loc]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Webcam Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
conn.close()
