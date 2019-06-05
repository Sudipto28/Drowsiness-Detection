import cv2
import dlib
import numpy as np
from math import hypot

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Learning/Projects/GazeControlledKeyboard/shape_predictor_68_face_landmarks.dat')
font = cv2.FONT_HERSHEY_DUPLEX


def get_midpoint(p1, p2):
    x = int((p1.x + p2.x) / 2)
    y = int((p1.y + p2.y) / 2)
    return x, y


def get_blinking_ratio(eye_points, facial_landmarks):
    left = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    top = get_midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    bottom = get_midpoint(facial_landmarks.part(eye_points[4]), facial_landmarks.part(eye_points[5]))

    ver_line_length = hypot((top[0] - bottom[0]), (top[1] - bottom[1]))
    hor_line_length = hypot((left[0] - right[0]), (left[1] - right[1]))

    ratio = hor_line_length / ver_line_length

    return ratio


# Counters
blinking_frames = 0

while True:
    _, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(gray, face)

        # Eye Blink Detection
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio >= 3.50:
            # cv2.putText(frame, 'BLINK...', (50, 150), font, 4, (0, 0, 255), cv2.LINE_4)
            blinking_frames += 1

            if blinking_frames >= 4:
                cv2.putText(frame, 'DROWSING!!!', (10, 50), font, 2, (0, 0, 255), 2)
        else:
            blinking_frames = 0

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()