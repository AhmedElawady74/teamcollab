import cv2
import time
import os
import sys
from datetime import datetime

# ✅ Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # adjust path to reach manage.py
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
import django
django.setup()

# ✅ Import the AbsenceLog model
from api.models import AbsenceLog

print("✅ Monitoring script started...")

# Load Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

# Create folder to save snapshots
os.makedirs("snapshots", exist_ok=True)

absence_start = None
absence_threshold = 15  # in seconds

def log_absence(timestamp, image_path):
    # Save to database
    AbsenceLog.objects.create(timestamp=timestamp, image_path=image_path)
    print(f"📝 Absence logged at {timestamp}, image: {image_path}")

print("⏳ Monitoring started... Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture image from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangle and display coordinates
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        label = f"Face: ({x}, {y}, {w},{h})"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if len(faces) == 0:
        if absence_start is None:
            absence_start = time.time()
        elif time.time() - absence_start >= absence_threshold:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            img_name = f"snapshots/absence_{timestamp}.jpg"
            cv2.imwrite(img_name, frame)
            log_absence(timestamp, img_name)
            print(f"📸 Snapshot saved during absence: {img_name}")
            absence_start = None
    else:
        absence_start = None

    cv2.imshow("AI Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Monitoring stopped.")