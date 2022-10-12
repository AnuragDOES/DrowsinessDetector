import cv2 as cv
import mediapipe as mp
from scipy.spatial import distance
LEFT_EYE_LEFT_RIGHT = [263, 362]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

capture = cv.VideoCapture(0)
frame = 0
level = 0
time = 0
normalEAR = 0.0
warning = False

mpDrawing = mp.solutions.drawing_utils
mpMesh = mp.solutions.face_mesh


def aspectRatio(image, outputs, part):
    landmark = outputs.multi_face_landmarks[0]
    numerator, count = 0, 0
    height, width = image.shape[:2]
    for i in range(1, int(len(part)/2)):
        top = landmark.landmark[part[i]]
        bottom = landmark.landmark[part[-i]]
        topPt = int(top.x * width), int(top.y * height)
        bottomPt = int(bottom.x * width), int(bottom.y * height)
        numerator += distance.euclidean(topPt, bottomPt)
        count += 1
    left = landmark.landmark[part[0]]
    right = landmark.landmark[part[int(len(part)/2)]]
    leftPt = int(left.x*width), int(left.y*height)
    rightPt = int(right.x*width), int(right.y*height)
    denominator = distance.euclidean(leftPt, rightPt)
    return numerator/(count*denominator)


def sleepyLevel(ratio, normalRatio, preLevel):
    global time
    global warning
    if 0.50*normalRatio <= ratio and ratio <= 0.8*normalRatio:
        level = 1
    elif 0.15*normalRatio <= ratio and ratio <= 0.50*normalRatio:
        level = 2
    elif ratio <= 0.15*normalRatio:
        level = 3
    else:
        level = 0
    if level > 0 and preLevel <= level:
        time += 1
        if time > 10:
            warning = True
    else:
        time = 0
        warning = False
    return level


def draw_landmarks(image, outputs, land_mark, color):
    height, width = image.shape[:2]
    for lms in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[lms]
        point_scale = ((int)(point.x * width), (int)(point.y*height))
        cv.circle(image, point_scale, 2, color, 1)


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
            if frame == 50:
                normalEAR = EAR
            if frame > 50:
                level = sleepyLevel(EAR, normalEAR, level)
            image = cv.flip(image, 1)
            cv.putText(image, f'EAR: {round(EAR,3)},{time}', (20, 70),
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
            if warning:
                cv.putText(image, f'{level}!!', (20, 120),
                           cv.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 3)
            cv.putText(image, f'LAR: {round(LAR,3)},{time}', (300, 70),
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
        cv.imshow('MediaPipe Face Mesh', image)
        if cv.waitKey(5) & 0xFF == ord('q'):
            break
