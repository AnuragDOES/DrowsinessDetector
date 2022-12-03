import cv2 as cv
from functions import aspectRatio, sleepyLevel, draw_landmarks
import mediapipe as mp

LEFT_EYE_LEFT_RIGHT = [263, 362]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

capture = cv.VideoCapture(2)
frame = 0
level = 0
time = 0
normalEAR = 0.0
normalarr = []
warning = ""
mpDrawing = mp.solutions.drawing_utils
mpMesh = mp.solutions.face_mesh
with mpMesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as faceModel:
    while True:
        _, image = capture.read()
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = faceModel.process(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image.flags.writeable = True
        if results.multi_face_landmarks:
            draw_landmarks(
                image,
                results,
                LEFT_EYE,
                (0, 255, 0)
            )
            draw_landmarks(
                image,
                results,
                RIGHT_EYE,
                (0, 255, 0)
            )
            draw_landmarks(
                image,
                results,
                LIPS,
                (0, 255, 0)
            )
            EAR = min(aspectRatio(image, results, RIGHT_EYE),
                      aspectRatio(image, results, LEFT_EYE))
            LAR = aspectRatio(image, results, LIPS)
            frame += 1
            if frame < 50:
                normalEAR += EAR
            elif frame == 50:
                normalEAR += EAR
                normalEAR = (normalEAR/50)
            else:
                level = sleepyLevel(EAR, normalEAR)
            image = cv.flip(image, 1)
            cv.putText(image, f'EAR: {round(EAR,3)},normal:{round(normalEAR,5)},{time}', (100, 100),
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
            if warning:
                cv.putText(image, f'{level}!!', (20, 120),
                           cv.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 3)
            cv.putText(image, f'LAR: {round(LAR,3)},{time}', (300, 70),
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
        cv.imshow('MediaPipe Face Mesh', image)
        if cv.waitKey(5) & 0xFF == ord('q'):
            break
