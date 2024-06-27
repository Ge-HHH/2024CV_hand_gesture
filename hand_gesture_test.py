import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from Hands import Hand
from Hands import HandSequence

if __name__ == '__main__':
    # 初始化Mediapipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    
    fig=plt.figure(figsize=(10,7))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=1)

    handseq=HandSequence(fig,width,height)

    ani = animation.FuncAnimation(fig, handseq.update_vis, blit=False, interval=24)

    def process_frame():
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            no_hand_cnt=0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    no_hand_cnt=0
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks_3d = []
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            h, w, _ = image.shape
                            cx, cy = int(landmark.x * w), int(landmark.y * h)
                            cz = landmark.z
                            size = int(5 * (1 - cz * 10))
                            cv2.circle(image, (cx, cy), size, (0, 255, 0), -1)
                        handseq.append(Hand(hand_landmarks))
                        # handseq.mouse_control()
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        continue
                else:
                    no_hand_cnt+=1
                    # print(no_hand_cnt)
                    if no_hand_cnt>11:
                        # handseq.clear()
                        no_hand_cnt=0
                cv2.imshow('Hand Gesture Recognition', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

    thread = threading.Thread(target=process_frame)
    thread.start()

    plt.show()

    thread.join()
