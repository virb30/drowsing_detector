# importar pacotes necessários
from os.path import join, dirname
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt
import os

# definir constantes
ALARM_SOUND = join(dirname(__file__), "buzina.wav")
WEBCAM = os.environ.get('WEBCAM', 1)
EYE_THRESHOLD = 0.25
FRAMES_SEQ = 40
COUNTER = 0
ALARM_TRIGGERED = False
SHAPE_PREDICTOR = join(dirname(__file__), 'shape_predictor_68_face_landmarks.dat')


def trigger_alarm(path=ALARM_SOUND):
    """
    Trigger an sound alarm from .wav file

    :param path: .wav file path
    :return: None
    """
    playsound.playsound(path)
    return None


def calculate_eye_aspect_ratio(eye):
    # calcular a distância euclidiana entre os conjuntos das
    # landmarks verticais do olho coordenadas-(x, y)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmarks (x, y)-coordinates
    # calcular a distância euclidiana entre as
    # landmarks horizontais do olho coordenadas-(x, y)
    C = dist.euclidean(eye[0], eye[3])

    # calcular o EAR
    ear = (A + B) / (2.0 * C)
    return ear


# carregar o dlib para detectar rostos
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

# pegar os indices do previdor, para olhos esquero e direito
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# inicializar video
print("[INFO] inicializando streaming de video")
vs = VideoStream(src=WEBCAM).start()
time.sleep(2.0)

# # desenhar um objeto do tipo figure
y = [None] * 100
x = np.arange(0, 100)
fig = plt.figure(figsize=(5, 3))
ax = fig.add_subplot(111)
li, = ax.plot(x, y)


def draw_graph(ear):
    # salvar histórico para plot
    y.pop(0)
    y.append(ear)

    # update canvas
    plt.xlim([0, 100])
    plt.ylim([0, 0.4])
    ax.relim()
    ax.autoscale_view(True, True, True)
    fig.canvas.draw()
    plt.show(block=False)
    li.set_ydata(y)
    fig.canvas.draw()
    plt.pause(0.01)


# loop sobre os frames do video
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecter faces
    rects = detector(gray, 0)

    # loop nas faces detectadas
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extrair coordenadas dos olhos e calcular a proporção de abertura
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = calculate_eye_aspect_ratio(leftEye)
        rightEAR = calculate_eye_aspect_ratio(rightEye)

        # ratio medio para os dois olhos
        ear = (leftEAR + rightEAR) / 2.0

        # convex hull para os olhos
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # exibe gráfico
        draw_graph(ear)

        # checar ratio do olho x threshold
        if ear < EYE_THRESHOLD:
            COUNTER += 1

            # verificar critério para soar o alarme
            if COUNTER >= FRAMES_SEQ:
                # ligar alarme
                if not ALARM_TRIGGERED:
                    ALARM_TRIGGERED = True
                    t = Thread(target=trigger_alarm)
                    t.daemon = True
                    t.start()

                cv2.putText(frame, "[ALERTA] FADIGA!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # se acima do threshold, desliga alarme e reseta contador
        else:
            ALARM_TRIGGERED = False
            COUNTER = 0

        # desenhar a proporção de abertura dos olhos
        cv2.putText(frame, "EAR {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # mostrar frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # tecla para sair do script "q"
    if key == ord("q"):
        break

# clean
cv2.destroyAllWindows()
vs.stop()
