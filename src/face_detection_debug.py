import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===== Callback =====
latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# ===== Load model =====
model_path = "../models/face_landmarker.task"

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_face_blendshapes=True,
    result_callback=result_callback
)

landmarker = FaceLandmarker.create_from_options(options)

# ===== Webcam =====
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    landmarker.detect_async(mp_image, timestamp)
    timestamp += 1

    # ===== Draw face landmarks =====
    if latest_result and latest_result.face_landmarks:
        for face in latest_result.face_landmarks:

            for lm in face:
                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Face Mesh (Modern MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()