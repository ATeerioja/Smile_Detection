import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

model_path = "../models/face_landmarker.task"

#Tuodaan valmiit asetukset mallille
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

#Asetetaan oletusasetukset
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_face_blendshapes=True,
    result_callback=result_callback
)

#Luodaan FaceLandmarker olio
landmarker = FaceLandmarker.create_from_options(options)

#Alustetaan live-video OpenCV kirjastolla
cap = cv2.VideoCapture(0)
timestamp = 0

while True:
    ret, frame = cap.read()

    #Luetaan yksi ruutu videosta
    frame = cv2.flip(frame, 1)

    #Käsitellään video MediaPipen formaattiin
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #Käytetään asynchronista metodia livevideon tunnistukseen
    landmarker.detect_async(mp_image, timestamp)
    timestamp += 1

    #Käytetään kasvojen blendshapeja hymyn tunnistukseen
    if latest_result and latest_result.face_blendshapes:
        blendshapes = latest_result.face_blendshapes[0]

        #Käytetään valmiita blendshapeja hymyn tunnistukseen
        smile_left = 0
        smile_right = 0
        shrug_lower = 0
        shrug_upper = 0
        pucker = 0


        for b in blendshapes:
            if b.category_name == "mouthSmileLeft":
                smile_left = b.score
            elif b.category_name == "mouthSmileRight":
                smile_right = b.score
            elif b.category_name == "mouthShrugUpper":
                shrug_upper = b.score
            elif b.category_name == "mouthShrugLower":
                shrug_lower = b.score
            elif b.category_name == "mouthPucker":
                pucker = b.score

        #Hymyarvon ja mutruarvon luonti ja testaaminen
        smile_score = (smile_left + smile_right) / 2
        frown_score = (shrug_upper + shrug_lower + pucker) / 3

        if smile_score > 0.3:
            smile = f"Smiling ({smile_score:.2f})"
            smile_color = (0, 255, 0)
        else:
            smile = f"Not Smiling ({smile_score:.2f})"
            smile_color = (0, 0, 255)

        if frown_score > 0.3:
            frown = f"Frowning ({frown_score:.2f})"
            frown_color = (0, 255, 0)
        else:
            frown = f"Not Frowning ({frown_score:.2f})"
            frown_color = (0, 0, 255)

        #Näytetään arvauksen tulos videossa
        cv2.putText(frame, smile, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, smile_color, 2, cv2.LINE_AA)

        cv2.putText(frame, frown, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, frown_color, 2, cv2.LINE_AA)

    cv2.imshow("Smile Detection (MediaPipe Tasks)", frame)

    #Painettaessa ESC suljetaan ikkuna
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()