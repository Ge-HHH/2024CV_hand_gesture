import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
import mediapipe as mp
import pyautogui
from Hands import Hand, HandSequence

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mouse_control = True

    def set_handseq(self, handseq):
        self.handseq = handseq

    def run(self):
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        self._run_flag = True
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            
            no_hand_cnt = 0
            while self._run_flag:
                ret, frame = cap.read()
                if not ret:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        self.handseq.append(Hand(hand_landmarks))
                        if self.mouse_control:self.handseq.mouse_control()
                else:
                    no_hand_cnt += 1
                    if no_hand_cnt > 10:
                        self.handseq.clear()

                self.change_pixmap_signal.emit(image)

        cap.release()
    
    def set_mouse_control(self, flag):
        self.mouse_control = flag

    def stop(self):
        self._run_flag = False
        self.wait()