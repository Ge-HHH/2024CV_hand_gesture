import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyautogui
class Hand:

    fingers = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20]
    }

    thresholds = {
        'thumb': [0.7,0.8,0,0.4],
        'index': [0.7,0.8,-0.7,0.85],
        'middle': [0.7,0.8,-0.5,0.85],
        'ring': [0.6,0.75,0,0.4],
        'pinky': [0.5,0.55,0,0.4],
        'three_cluster':[2,0.55,0,0.4],
        'index_fist':[0.4,0.4,0,0.4],
    }

    STRAIGHT={
        'thumb':2**0,
        'index':2**1,
        'middle':2**2,
        'ring':2**3,
        'pinky':2**4
    }
    ROLL={
        'thumb':2**5,
        'index':2**6,
        'middle':2**7,
        'ring':2**8,
        'pinky':2**9
    }

    THREE_CLUSTER=2**10
    

    FIST=ROLL['thumb']|ROLL['index']|ROLL['middle']|ROLL['ring']|ROLL['pinky']
    ONE_FINGER=2**11
    TWO_FINGER=2**12
    INDEX_FIST=2**13

    def __init__(self,landmarks=None,hands=None):
        if landmarks is not None:
            self.landmarks=np.array([(landmark.x,landmark.y,landmark.z) for landmark in landmarks.landmark])
            
        elif hands is not None:
            if len(hands)>=2:
                self.landmarks=np.zeros((21,3))
                for hand in hands:
                    self.landmarks+=hand.landmarks
                self.landmarks/=len(hands)
        self.points=self.landmarks-self.landmarks[0]
        self.points/=np.linalg.norm(self.points[5] - self.points[0])
        self.points[:,1]=-self.points[:,1]
        self.points=self.points[:,(0,2,1)]
        self.plam_nor_vec=np.cross(self.points[5]-self.points[0],self.points[9]-self.points[0])
        self.plam_nor_vec/=np.linalg.norm(self.plam_nor_vec)
        self.gesture=self.get_gesture_vec()

    
    def get_gesture_eur(self):
        status=0
        def distance(finger):
            return np.linalg.norm(self.points[finger[0]]-self.points[finger[3]])
        for finger in self.fingers:
            dis=distance(self.fingers[finger])
            if dis<self.thresholds[finger][0]:
                status|=self.ROLL[finger]
            elif dis>self.thresholds[finger][1]:
                status|=self.STRAIGHT[finger]
        return status
    
    def get_gesture_vec(self):
        status=0
        def distance(finger):
            a=(self.points[finger[3]]-self.points[finger[2]])
            b=(self.points[finger[1]]-self.points[finger[0]])
            return np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
        def distance1(finger):
            a=(self.points[finger[1]]-self.points[finger[0]])
            # print(self.points[finger[0]],self.points[finger[1]],self.plam_nor_vec)
            b=self.plam_nor_vec
            return np.dot(a,b)/np.linalg.norm(a)
        # def distance(finger):
        #     return np.linalg.norm(self.points[finger[0]]-self.points[finger[3]])
        for finger in self.fingers:
            dis=distance(self.fingers[finger])
            
            if dis<self.thresholds[finger][2]:
                status|=self.ROLL[finger]
            elif dis>self.thresholds[finger][3]:
                status|=self.STRAIGHT[finger]
            # if finger=='index':
            #     print(finger,dis,status&self.STRAIGHT[finger],self.thresholds[finger])
        eur_dis=np.linalg.norm(self.points[4]-self.points[16])+np.linalg.norm(self.points[4]-self.points[12])
        if eur_dis<self.thresholds['three_cluster'][0]:
            status|=self.THREE_CLUSTER
        nor_dis=distance1(self.fingers['index'])
        # print(nor_dis)
        if nor_dis>self.thresholds['index_fist'][0]:
            status|=self.INDEX_FIST
        if not all([(not status&self.ROLL[finger]) for finger in ['thumb','ring','pinky']]):
            if  status&self.STRAIGHT['index'] and (not status&self.STRAIGHT['middle']):
                status|=self.ONE_FINGER
            if status&self.STRAIGHT['index'] and status&self.STRAIGHT['middle']:
                status|=self.TWO_FINGER

        return status
        
    def draw(self,ax):
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='g', marker='o')
        def draw_edges(route,color='r'):
            ax.plot(self.points[route, 0], self.points[route, 1], self.points[route, 2], c=color)
        for finger in self.fingers:
            color='blue'
            if self.gesture&self.STRAIGHT[finger]:
                color='yellow'
            elif self.gesture&self.ROLL[finger]:
                color='red'
            draw_edges(self.fingers[finger],color)
        #手掌
        color='blue' if not self.gesture&self.FIST else 'red'
        draw_edges([0,5,9,13,17,0],color)
        draw_edges([0,1],color)
            

class HandSequence:
    def __init__(self,fig,width=640,height=480):
        self.hands=[]
        self.fig=fig
        self.last_one_finger=None
        self.fingers_ax={
            'thumb': self.fig.add_subplot(5,2,1),
            'index': self.fig.add_subplot(5,2,3),
            'middle': self.fig.add_subplot(5,2,5),
            'ring': self.fig.add_subplot(5,2,7),
            'pinky': self.fig.add_subplot(5,2,9)
        }
        self.fingers_dis={
            'thumb': [],
            'index': [],
            'middle': [],
            'ring': [],
            'pinky': [],
            'index_fist':[],
        }
        self.hand_ax=self.fig.add_subplot(1,2,2,projection='3d')
        self.release_tag=True
        self.action_cnt=0
    def append(self,hand):
        self.hands.append(hand)
        if len(self.hands)>100:
            self.hands.pop(0)
        def distance(finger):
            return np.linalg.norm(finger[0]-finger[3])
        def distance1(finger):
            a=(finger[3]-finger[2])
            b=(finger[1]-finger[0])
            # return np.linalg.norm(np.cross(a,b))/np.linalg.norm(a)/np.linalg.norm(b)
            return np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
        def distance2(finger,plam_nor_vec):
            a=(finger[1]-finger[0])
            return np.dot(a,plam_nor_vec)/np.linalg.norm(a)


        for finger in self.fingers_dis:
            if finger=='index_fist':
                dis=distance2(hand.points[hand.fingers['index']],hand.plam_nor_vec)
            else: 
                dis=distance1(hand.points[hand.fingers[finger]])
            self.fingers_dis[finger].append(dis)
            if len(self.fingers_dis[finger])>100:
                self.fingers_dis[finger].pop(0)
    
    def update_vis(self,frame):
        for finger in self.fingers_ax:
            ax=self.fingers_ax[finger]
            ax.clear()
            ax.set_title(finger)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Distance')
            if len(self.fingers_dis[finger])>0:
                ax.set_xlim(max(0, len(self.fingers_dis[finger]) - 100), len(self.fingers_dis[finger]))
                ax.set_ylim(min(self.fingers_dis[finger])-0.1,max(self.fingers_dis[finger])+0.1)
                ax.plot(range(len(self.fingers_dis[finger])),self.fingers_dis[finger])

        if self.hands:
            ax=self.hand_ax
            ax.clear()
            ax.set_title('Hand Landmarks in 3D Space')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_zlim(0, 2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            self.hands[-1].draw(ax)
    def clear(self):
        self.hands.clear()
        for finger in self.fingers_dis:
            self.fingers_dis[finger].clear()
        self.last_one_finger=None
        self.action_cnt=0
    def mouse_move(self):
        dx=-(self.hands[-1].landmarks[5,0]-self.hands[-2].landmarks[5,0])
        dy=self.hands[-1].landmarks[5,1]-self.hands[-2].landmarks[5,1]
        speed=np.sqrt(dx**2+dy**2)
        coe=np.tan(speed*np.pi/2)
        # print(dx,dy)
        theta=np.arctan2(dy,dx)
        dx=640*coe*np.cos(theta)*1.5
        dy=480*coe*np.sin(theta)*1.5
        dx=dx if abs(dx)>3 else 0
        dy=dy if abs(dy)>3 else 0
        # print(dx,dy)
        pyautogui.moveRel(dx,dy,duration=1/15,_pause=True)
    def sroll(self):
        dx=-(self.hands[-1].landmarks[5,0]-self.hands[-2].landmarks[5,0])
        dy=self.hands[-1].landmarks[5,1]-self.hands[-2].landmarks[5,1]
        speed=np.sqrt(dx**2+dy**2)
        coe=np.tan(speed*np.pi/2)
        # print(dx,dy)
        theta=np.arctan2(dy,dx)
        dx=320*coe*np.cos(theta)
        dy=240*coe*np.sin(theta)
        pyautogui.vscroll(int(dy),_pause=True)
        pyautogui.hscroll(int(dx),_pause=True)
    def mouse_control(self):
        if len(self.hands)>2:
            last_hand=self.hands[-2]
            hand=self.hands[-1]
            if hand.gesture&Hand.TWO_FINGER:
                if last_hand.gesture&Hand.TWO_FINGER:
                    print('srool')
                    self.sroll()
            elif hand.gesture&Hand.ONE_FINGER:
                if last_hand.gesture&Hand.ONE_FINGER:
                    self.mouse_move()
                elif last_hand.gesture&Hand.INDEX_FIST:
                    print('mouseUp: left')
                    pyautogui.mouseUp(button='left')
                    self.release_tag=True
                elif last_hand.gesture&Hand.ROLL['index']:
                    print('mouseUp: right')
                    pyautogui.mouseUp(button='right')
                    self.release_tag=True
            elif hand.gesture&Hand.INDEX_FIST:
                if last_hand.gesture&Hand.ONE_FINGER:
                    print('mouseDown: left')
                    pyautogui.mouseDown(button='left')
                    self.release_tag=False
                elif last_hand.gesture&Hand.INDEX_FIST:
                    self.mouse_move()
            elif hand.gesture&Hand.ROLL['index']:
                if last_hand.gesture&Hand.ONE_FINGER:
                    print('mouseClick: right')
                    pyautogui.click(button='right')
                    self.release_tag=True
            else:
                if (not self.release_tag) :
                    pyautogui.mouseUp(button='left')
                    self.release_tag=True
        else:
            if (not self.release_tag): 
                pyautogui.mouseUp(button='left')
                self.release_tag=True
        
        # if len(self.hands)>2:
        #     if self.action_cnt<2:
        #         self.action_cnt+=1
        #         return
        #     else :
        #         self.action_cnt=0
        #     last_hand=Hand(hands=[self.hands[-4],self.hands[-3]])
        #     hand=Hand(hands=[self.hands[-2],self.hands[-1]])
        #     # last_hand=Hand(hands=[self.hands[-6],self.hands[-5],self.hands[-4]])
        #     # hand=Hand(hands=[self.hands[-3],self.hands[-2],self.hands[-1]])
        #     if hand.gesture&Hand.TWO_FINGER:
        #         if last_hand.gesture&Hand.TWO_FINGER:
        #             print('srool')
        #             self.sroll()
        #     elif hand.gesture&Hand.ONE_FINGER:
        #         if last_hand.gesture&Hand.ONE_FINGER:
        #             self.mouse_move()
        #         elif last_hand.gesture&Hand.INDEX_FIST:
        #             print('mouseUp: left')
        #             pyautogui.mouseUp(button='left')
        #             self.release_tag=True
        #         elif last_hand.gesture&Hand.ROLL['index']:
        #             print('mouseUp: right')
        #             pyautogui.mouseUp(button='right')
        #             self.release_tag=True
        #     elif hand.gesture&Hand.INDEX_FIST:
        #         if last_hand.gesture&Hand.ONE_FINGER:
        #             print('mouseDown: left')
        #             pyautogui.mouseDown(button='left')
        #             self.release_tag=False
        #         elif last_hand.gesture&Hand.INDEX_FIST:
        #             self.mouse_move()
        #     elif hand.gesture&Hand.ROLL['index']:
        #         if last_hand.gesture&Hand.ONE_FINGER:
        #             print('mouseClick: right')
        #             pyautogui.click(button='right')
        #             # self.release_tag=True
        #     else:
        #         if (not self.release_tag) :
        #             pyautogui.mouseUp(button='left')
        #             self.release_tag=True
        # else:
        #     if (not self.release_tag): 
        #         pyautogui.mouseUp(button='left')
        #         self.release_tag=True
            
                

                

        



