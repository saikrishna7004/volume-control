import cv2
import numpy as np
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



# Function to detect a fist (with fingers in vertical direction and thumb in horizontal direction_
def isFist(lmList):
    gesture = [False]*5    # True = Finger open, False = Finger closed
    fist = [False]*5

    pseudoFixKeyPt = lmList[2][1]
    # For right hand thumb
    if lmList[0][1] > pseudoFixKeyPt and lmList[3][1] < pseudoFixKeyPt and lmList[4][1] < pseudoFixKeyPt:
        gesture[0] = True
    # For left hand thumb
    elif lmList[0][1] < pseudoFixKeyPt and lmList[3][1] > pseudoFixKeyPt and lmList[4][1] > pseudoFixKeyPt:
        gesture[0] = True

    FingerPts = [2, 6, 10, 14, 18]
    for i in range(1,5):
        pseudoFixKeyPt = lmList[FingerPts[i]][2]
        if lmList[FingerPts[i]+1][2] < pseudoFixKeyPt and lmList[FingerPts[i]+2][2] < pseudoFixKeyPt:
            gesture[i] = True

    #print(gesture)

    if gesture == fist:
        return True
    return False



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:         # Atleast one hand in the image
        vol = 0
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id,lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w),  int(lm.y*h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        if lmList:
            if isFist(lmList):
                print("Exited Successfully!!")
                break

            x1, y1 = lmList[4][1], lmList[4][2]       # (x1, y1) => Coordinate of Thumb Tip
            x2, y2 = lmList[8][1], lmList[8][2]       # (x2, y2) => Coordinate of Index Finger Tip

            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            length = math.dist((x1,y1), (x2,y2))
            #print(length)

            #x_mid, y_mid = (x1+x2)//2, (y1+y2)//2     # (x_mid, y_mid) => Coordinate of Index Finger Tip
            #cv2.circle(img, (x_mid, y_mid), 10, (255, 0, 9), cv2.FILLED)

        volRange = volume.GetVolumeRange()
        minVol = volRange[0]
        maxVol = volRange[1]
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)

        volBar = np.interp(length, [50, 300], [400, 150])
        cv2.rectangle(img, (50,150), (85,400), (0, 233, 23), 3)
        cv2.rectangle(img, (50,int(volBar)), (85,400), (0,233,43), cv2.FILLED)

        volPercent = np.interp(length, [50,300], [0,100])
        cv2.putText(img, str(int(volPercent)), (40,450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,2,234), 2)


    cv2.imshow("Image", img)
    cv2.waitKey(1)