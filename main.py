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
from CameraThread import CameraThread
from Canvas import MplCanvas
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Hand Gesture Recognition')
        self.resize(1920, 500)
        self.move(320,750)
        # self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.correct_mode=None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)

        self.top_layout = QHBoxLayout()
        self.main_layout.addLayout(self.top_layout)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumWidth(self.width() // 3)
        self.top_layout.addWidget(self.image_label)

        self.plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.top_layout.addWidget(self.plot_canvas)

        # 提示词标签
        self.hint_label = QLabel("")
        self.hint_label.setFont(QFont("Arial", 16))
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.hint_label)

        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)

        self.start_button = QPushButton("开始/暂停")
        self.start_button.setFont(QFont("Arial", 12))
        self.start_button.clicked.connect(self.toggle_camera)
        self.button_layout.addWidget(self.start_button)

        self.palm_correct_button = QPushButton("手掌姿态修正")
        
        self.palm_correct_button.setFont(QFont("Arial", 16))
        self.palm_correct_button.clicked.connect(lambda: self.start_countdown("手掌姿态修正"))
        self.button_layout.addWidget(self.palm_correct_button)

        self.fist_correct_button = QPushButton("握拳姿态修正")
        self.fist_correct_button.setFont(QFont("Arial", 16))
        self.fist_correct_button.clicked.connect(lambda: self.start_countdown("握拳姿态修正"))
        self.button_layout.addWidget(self.fist_correct_button)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(100)

        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)

        self.correct_fist_timer = QTimer()
        self.correct_fist_timer.timeout.connect(self.correct_fist)

        self.correct_palm_timer = QTimer()
        self.correct_palm_timer.timeout.connect(self.correct_palm)

        self.camera_thread = CameraThread()
        self.camera_thread.set_handseq(handseq)
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.controling = False
        self.toggle_camera()  # 默认开启鼠标操控
        self.camera_thread.start()

    def start_countdown(self, message):
        self.countdown = 5
        self.hint_label.setText(f"{message}开始倒计时: {self.countdown}秒")
        self.correct_mode = message
        self.countdown_timer.start(1000)  # 每1000毫秒更新一次
        # if self.controling:
        #     self.toggle_camera()

    def update_countdown(self):
        self.countdown -= 1
        if self.countdown > 0:
            self.hint_label.setText(f"{self.hint_label.text().split()[0]} {self.countdown}秒")
        else:
            self.hint_label.setText("")
            self.countdown_timer.stop()
            if self.correct_mode == "手掌姿态修正":
                handseq.clear()
                self.countdown=None
                self.correct_palm_timer.start(100)
            elif self.correct_mode == "握拳姿态修正":
                handseq.clear()
                self.countdown=None
                self.correct_fist_timer.start(100)
    
    def correct_fist(self):
        if len(handseq.hands) < 100 and self.countdown is None:
            self.hint_label.setText(f"请保持握拳姿态, 校准已完成:{len(handseq.hands)}")
        else:
            if self.countdown is None:
                self.countdown = 1.5
                for finger in Hand.fingers:
                    mean=np.mean(handseq.fingers_dis[finger])
                    std=np.std(handseq.fingers_dis[finger])
                    Hand.thresholds[finger][2]=mean+0.1
                    print("%s %.4lf±%.4lf"%(finger,mean,std))
                finger="index_fist"
                mean=np.mean(handseq.fingers_dis[finger])
                std=np.std(handseq.fingers_dis[finger])
                Hand.thresholds[finger][2]=mean-0.1
                print("%s %.4lf±%.4lf"%(finger,mean,std))
                self.hint_label.setText(f"校准已完成，可放松手掌")
            else:
                self.countdown -= 0.1
                # self.hint_label.setText(f"请保持手掌姿态, 校准已完成:{len(handseq.hands)} {self.countdown}秒")
                if self.countdown <= 0:
                    self.correct_fist_timer.stop()
                    self.hint_label.setText("")
    
    def correct_palm(self):
        if len(handseq.hands) < 100 and self.countdown is None:
            self.countdown=None
            self.hint_label.setText(f"请保持手掌张开姿态, 校准已完成:{len(handseq.hands)}")
        else:
            if self.countdown is None:
                self.countdown = 1.5
                for finger in Hand.fingers:
                    mean=np.mean(handseq.fingers_dis[finger])
                    std=np.std(handseq.fingers_dis[finger])
                    Hand.thresholds[finger][3]=mean-0.1
                    print("%s %.4lf±%.4lf"%(finger,mean,std))
                finger="index_fist"
                mean=np.mean(handseq.fingers_dis[finger])
                std=np.std(handseq.fingers_dis[finger])
                # Hand.thresholds[finger][2]=mean-0.05
                print("%s %.4lf±%.4lf"%(finger,mean,std))
                self.hint_label.setText(f"校准已完成，可放松手掌")
            else:
                self.countdown -= 0.1
                # self.hint_label.setText(f"请保持手掌姿态, 校准已完成:{len(handseq.hands)} {self.countdown}秒")
                if self.countdown <= 0:
                    self.correct_palm_timer.stop()
                    self.hint_label.setText("")
            

    def toggle_camera(self):
        if self.controling:
            # self.camera_thread.stop()
            self.camera_thread.set_mouse_control(False)
            self.start_button.setText("开始")
        else:
            # self.camera_thread.start()
            self.camera_thread.set_mouse_control(True)
            self.start_button.setText("暂停")
        self.controling = not self.controling

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    def update_plot(self):
        self.plot_canvas.update_plot(handseq)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.image_label.setMaximumWidth(self.width() // 3)
        self.start_button.setMaximumWidth(self.width() // 7)
        self.palm_correct_button.setMaximumWidth(self.width() // 7)
        self.fist_correct_button.setMaximumWidth(self.width() // 7)
        self.start_button.setMinimumHeight(40)
        self.palm_correct_button.setMinimumHeight(40)
        self.fist_correct_button.setMinimumHeight(40)
        self.hint_label.setMinimumHeight(40)

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    handseq = HandSequence(Figure(figsize=(10, 10)), 640, 480)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
