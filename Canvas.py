import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from Hands import Hand, HandSequence
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=1)
        self.ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_index_fist = fig.add_subplot(3, 2, 2)
        self.ax_index_roll = fig.add_subplot(3, 2, 4)
        self.ax_mid_rool = fig.add_subplot(3, 2, 6)

        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)

    def update_plot(self, handseq):
        # 3D hand landmarks
        self.ax_3d.clear()
        self.ax_3d.set_xlim(-1.5, 1.5)
        self.ax_3d.set_ylim(-1.5, 1.5)
        self.ax_3d.set_zlim(0, 2)
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        if handseq.hands:
            handseq.hands[-1].draw(self.ax_3d)

        # index finger
        def draw_line(ax, dis, title='Index Finger'):
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Distance')
            if len(dis) > 0:
                ax.set_xlim(max(0, len(dis) - 100), len(dis))
                ax.set_ylim(min(dis) - 0.1, max(dis) + 0.1)
                ax.plot(range(len(dis)), dis)

        draw_line(self.ax_index_fist, handseq.fingers_dis['index_fist'], title='Index Fist')
        draw_line(self.ax_index_roll, handseq.fingers_dis['index'], title='Index Roll')
        draw_line(self.ax_mid_rool, handseq.fingers_dis['middle'], title='Mid Roll')
        self.draw()