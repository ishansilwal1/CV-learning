import cv2 as cv 
import numpy as np 


# video = cv.VideoCapture(0)

# if not video.isOpened():
#     print("Cannot find camera")
#     exit()

# while True:
#     ret,frame = video.read()
#     if not ret:
#         print("Cannot receive frame")
#         break 
#     gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#     cv.imshow('Webcam Gray',gray)   

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# video.release()
# cv.destroyAllWindows()

cap = cv.VideoCapture('video.mp4')
if not cap.isOpened():
    print("Error opening video file")
    exit()

while True:
    ret,frame = cap.read()
    if not ret:
        print("End of video stream")
        break
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow('Video Gray',gray)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break