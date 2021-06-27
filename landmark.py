# Inspirado no projeto de Adrian Rosenbroke

# importar pacotes
from os.path import join, dirname
import cv2
import dlib
import time
import imutils
from imutils.video import VideoStream
from imutils import face_utils

SHAPE_PREDICTOR = join(dirname(__file__), 'shape_predictor_68_face_landmarks.dat')

# dlib detector para 68 landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
vs = VideoStream(src=1).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# clean
cv2.destroyAllWindows()
vs.stop()