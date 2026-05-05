import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

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

cap = cv2.VideoCapture(0)
timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    landmarker.detect_async(mp_image, timestamp)
    timestamp += 1

    if latest_result and latest_result.face_blendshapes:
        blendshapes = latest_result.face_blendshapes[0]

        smile_left = 0
        smile_right = 0

        for b in blendshapes:
            if b.category_name == "mouthSmileLeft":
                smile_left = b.score
            elif b.category_name == "mouthSmileRight":
                smile_right = b.score

        smile_score = (smile_left + smile_right) / 2

        if smile_score > 0.5:
            text = f"Smiling ({smile_score:.2f})"
            color = (0, 255, 0)
        else:
            text = f"Not Smiling ({smile_score:.2f})"
            color = (0, 0, 255)

        cv2.putText(frame, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2, cv2.LINE_AA)

    cv2.imshow("Smile Detection (MediaPipe Tasks)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()