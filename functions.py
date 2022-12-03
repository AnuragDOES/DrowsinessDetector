
from scipy.spatial import distance
import cv2 as cv


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

def sleepyLevel(ratio, normalRatio):
    global time
    global warning
    if ratio < normalRatio and time > 50:
        warning = "Medium Warning"
    level = 0

    return level

def draw_landmarks(image, outputs, land_mark, color):
    height, width = image.shape[:2]
    for lms in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[lms]
        point_scale = ((int)(point.x * width), (int)(point.y*height))
        cv.circle(image, point_scale, 2, color, 1)