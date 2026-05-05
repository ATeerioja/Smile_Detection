import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

model_path = "face_landmarker.task"

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
    h, w, _ = frame.shape

    #Käsitellään video MediaPipen formaattiin
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #Käytetään asynchronista metodia livevideon tunnistukseen
    landmarker.detect_async(mp_image, timestamp)
    timestamp += 1

    #Käytetään kasvojen blendshapeja debuggaugkseen
    if latest_result and latest_result.face_blendshapes:
        blendshapes = latest_result.face_blendshapes[0]

        #Järjestetään kaikki scoret suurimmasta pienimpään
        sorted_bs = sorted(blendshapes, key=lambda x: x.score, reverse=True)

        y = 30

        #Näytetään 10 parasta, jotka liittyvät suuhun tai hymyyn
        for i, b in enumerate(sorted_bs[:10]):
            name = b.category_name
            score = b.score

            #Vain suu ja hymy lasketaan
            if "Smile" in name or "mouth" in name:
                color = (0, 255, 0)
                text = f"{name}: {score:.2f}"
                cv2.putText(frame, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1, cv2.LINE_AA)
                y += 20

        #Piirretään kasvojen pisteet
        if latest_result and latest_result.face_landmarks:
            for face in latest_result.face_landmarks:

                for lm in face:
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Blendshape Debug View", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()