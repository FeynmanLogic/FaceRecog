import cv2
import face_recognition

video_capture = cv2.VideoCapture(0)
print("📷 Starting webcam. Press Q to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Failed to grab frame.")
        continue

    # Slightly larger resizing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    print(f"✅ Frame shape: {rgb_small_frame.shape}, dtype: {rgb_small_frame.dtype}")

    try:
        face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
        print(f"🔍 Faces found: {len(face_locations)}")
    except Exception as e:
        print("❌ Face detection error:", e)
        continue

    # Draw rectangles
    for (top, right, bottom, left) in face_locations:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
