import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# pyautogui configuration (reducing default delay)
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0

# mediapipe inicialization
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 1, 0.75, 0.5)

# webcam inicialization
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30.0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
prevTime = 0
currTime = 0

# Getting screen properties
screenw, screenh = list(pyautogui.size())[0], list(pyautogui.size())[1]
reduction = 100
smoothing = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Setting audio controls
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
minVol, maxVol, _ = volume.GetVolumeRange()

tips = [4, 8, 12, 16, 20]

def getLandmarks(results):
    """Converts the landmarks provided by mediapipe to a list with the pixel coordinates of each landmark
    
    Args:
        results (list): List with landmarks especificaton of the hand, provided by mediapipe
    
    Returns:
        list: List with the pixel coordinates of each landmark
    """
    
    if results.multi_hand_landmarks:
        # Getting hand landmarks in a list
        landmarks = []
        for hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
            
            for landmarkid, lm in enumerate(hand.landmark):
                # The x and y coordinates are normalized, so we need to multiply them by the width and height of the frame to get the pixel values
                pixelx, pixely = int(lm.x * width), int(lm.y * height)
                
                landmarks.append([landmarkid, pixelx, pixely])
    
    return landmarks
    
def getFingers(landmarks):
    """Creates a list with the state of each finger (0 = closed, 1 = open)
    
    Args:
        landmarks (list): List with the pixel coordinates of the landmarks of the hand, provided by mediapipe
        
    Returns:
        list: List with the state of each finger (0 = closed, 1 = open) 
    """
    
    global tips 
    
    fingers = []  # 0 = closed, 1 = open
    if landmarks[tips[0]][1] < landmarks[tips[0] - 1][1]:  # Thumb
        fingers.append(1)
    else:
        fingers.append(0)
        
    for tip in tips[1:]:
        # Getting if each finger is up or down (except the thumb)
        if landmarks[tip][2] < landmarks[tip - 1][2]:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers


while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    if ret:  # if frame is read correctly (ret is True)
        frame = cv2.flip(frame, 1)  # flip the frame horizontally (mirror view)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert frame to RGB
        
        results = hands.process(frameRGB)
        if results.multi_hand_landmarks:
            landmarks = getLandmarks(results)
            fingers = getFingers(landmarks)
            
            # Mode: Mouse control
            if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 0:
                cv2.rectangle(frame, (reduction, reduction), (width - reduction, height - reduction), (255, 0, 255), 2)
                
                x1, y1 = landmarks[tips[1]][1:]  # Index finger
                
                # Convert coordinates (cam resolution to screen resolution)
                x_scaled = np.interp(x1, (reduction, 640 - reduction), (0, screenw))
                y_scaled = np.interp(y1, (reduction, 480 - reduction), (0, screenh))
                
                # Smoothen moviment
                curr_x = prev_x + (x_scaled - prev_x) / smoothing
                curr_y = prev_y + (y_scaled - prev_y) / smoothing
                prev_x, prev_y = curr_x, curr_y
                
                pyautogui.moveTo(curr_x, curr_y)
                cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                
            # Mode: Left / Right click
            if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0:
                cv2.rectangle(frame, (reduction, reduction), (width - reduction, height - reduction), (255, 0, 255), 2)
                
                x1, y1 = landmarks[tips[1]][1:]  # Index finger
                x2, y2 = landmarks[tips[2]][1:]  # Middle finger
            
                length = math.hypot(x2 - x1, y2 - y1)  # Distance between fingers
                
                cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                #print(length)
                if length < 25:
                    cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click(interval=0.25)
                    
                if length > 70:
                    cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.rightClick(interval=0.25)
                
            # Mode: Volume control
            if fingers[1] == 1 and fingers[0] == 1 and fingers[2] == 0:
                x1, y1 = landmarks[tips[1]][1:]  # Index finger
                x2, y2 = landmarks[tips[0]][1:]  # Thumb
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
                length = math.hypot(x2 - x1, y2 - y1)  # Distance between fingers (20 <-> 150)
                
                cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                vol = np.interp(length, [20, 150], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)
                
                percent = np.interp(length, [20, 150], [0, 100])
                cv2.putText(frame, f'{int(percent)} %', (center_x, center_y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        
        # FPS calculation
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(frame, str(int(fps)), (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("SysControl", frame)
        if cv2.waitKey(1) == 27:  # ESC key to quit
            break

cv2.destroyAllWindows()
