import math
from time import sleep

import cv2
import cvzone
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

keys=[["Q","W","E","R","T","Y","U","I","O","P"],
      ["A","S","D","F","G","H","J","K","L",";"],
      ["Z","X","C","V","B","N","M",",",".","/"]]



cv2.namedWindow("Virtual Keyboard", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Keyboard", 1200, 900)

def drawAll(img, buttonList):

    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h),(255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text,(x + 5, y + 38),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)
    return img

# def drawALL(img,buttonList):
#
#     imgNew=np.zeros_like(img,np.unit8)
#
#     for button in buttonList:
#         x,y=button.pos
#         cvzone.cornerRect(imgNew,(button.pos[0],button.pos[1],button.size[0],button.size[1]),20,rt=0)
#         cv2.rectangle(imgNew,button.pos,(x+button.size[0],y+button.size[1]),(255,0,255),cv2.FILLED)
#         cv2.putText(imgNew,button.text,(x + 5, y + 38),
#                             cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)
class Button():
    def __init__(self,pos,text,size=[40,40]):
        self.pos=pos
        self.size=size
        self.text=text


buttonList = []
for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
          buttonList.append(Button([60 * j + 1, 100*i+50], key))
    # img=myButton.draw(img)
finalText=""
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=drawAll(img,buttonList)

    result = hands.process(imgRGB)

    lmList = []
    if result.multi_hand_landmarks:
        for handLMs in result.multi_hand_landmarks:
            for lm in handLMs.landmark:
                lm_x, lm_y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                lmList.append([lm_x, lm_y])



    if len(lmList) >= 2:
        # Calculate the Euclidean distance between two landmarks (e.g., thumb and index finger)
        finger1 = lmList[12]  # Change to the index finger or other finger you want to measure
        finger2 = lmList[8]  # Change to the thumb or other finger you want to measure
        distance = math.dist(finger1, finger2)

        for button in buttonList:
                x, y = button.pos
                w, h = button.size
                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.rectangle(img, button.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 5, y + 38),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)

        if distance < 30:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size
                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    finalText += button.text
                    sleep(0.25)




    cv2.rectangle(img, (0, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (20, 400),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)


    cv2.imshow("Virtual Keyboard", img)
    # if cv2.waitKey(1) & 0xFF == 27:
    #     break
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

