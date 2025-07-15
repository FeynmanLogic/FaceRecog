import cv2
import face_recognition
import numpy as np

# Start webcam
video = cv2.VideoCapture(0)
print("📷 Webcam started. Press Q to quit.")

while True:
    ret, frame = video.read()
    if not ret or frame is None or frame.size == 0:
        print("⚠️ Skipped empty frame.")
        continue

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR to RGB (VERY IMPORTANT)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Validate shape and dtype
    if rgb_frame.dtype != np.uint8 or rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        print("⚠️ Invalid frame format. Skipping...")
        continue

    # Face Detection
    face_locations = face_recognition.face_locations(rgb_frame)

    # Draw boxes
    for top, right, bottom, left in face_locations:
        # Scale back to original size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Webcam Face Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
