# 手势识别

该项目是一个使用电脑前置摄像头的简单手势识别系统。它利用 MediaPipe 进行手部追踪，通过 OpenCV 捕获视频帧，并使用 PyQt 创建图形用户界面。该系统可以基于识别的手势控制鼠标实现一些简单的操作，同时对手部关键点和部分判别指标进行可视化。

## 依赖

运行该项目需要安装以下依赖：

- Python 3.8
- OpenCV
- MediaPipe
- Matplotlib
- PyQt5
- PyAutoGUI

可以使用以下命令安装这些依赖：

```bash
pip install opencv-python mediapipe matplotlib pyqt5 pyautogui
```

## Quick Start

1. **克隆仓库**：

    ```bash
    git clone https://github.com/yourusername/2024CV_hand_gesture.git
    cd 2024CV_hand_gesture
    ```


2. **运行应用程序**：

    要启动应用程序，运行 `main.py` 文件：

    ```bash
    python main.py
    ```

3. **使用方法**：

    - 应用程序将打开一个窗口，显示摄像头的实时视频和手势识别结果。
    - 使用“开始/暂停”按钮启动或暂停基于手势的鼠标控制。
    - 使用“手掌姿态修正”和“握拳姿态修正”按钮开始手掌和握拳姿态修正的5秒倒计时。
    - 竖起食指可以控制鼠标移动，在此状态下握拳将按下鼠标左键（可进行拖拽），食指弯曲将使用右键点击。
    - 竖起食指和中指并移动将操作鼠标滚轮。

## 文件结构

- `main.py`：运行应用程序的主脚本。
- `Hands.py`：包含处理手部关键点和手势识别的 `Hand` 和 `HandSequence` 类。
- `CameraThread.py`：包含捕获摄像头视频和调用Mediapipe进行关键点识别的 `CameraThread` 类。
- `Canvas.py`：包含用于绘制三维手部关键点和部分识别指标的 `MplCanvas` 类。

