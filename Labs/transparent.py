import cv2
import platform
from imutils import resize

import numpy as np

print(cv2.__version__)
print(platform.architecture())

img = cv2.imread('opencv.png',-1)
resized = resize(img, width=300, height=300)
logo = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening device camera")
    exit(-1)
while(True):
    ret,frame=cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    frame_h, frame_w, frame_c = frame.shape

    logo_h, logo_w, logo_c = logo.shape
    overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')

    for i in range (0, logo_h):
        for j in range(0, logo_w):
            if logo[i,j][3] !=0:
                overlay[i+10,j+10] = logo[i,j]

    cv2.addWeighted(overlay,1,frame,1,0,frame)
    cv2.imshow("dasndaisd", frame)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()