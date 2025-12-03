import cv2 as cv 
import matplotlib.pyplot as plt 

import numpy as np 


img = cv .imread("../images/spiderman.jpg")
cv.imshow('img',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray,125,175)
cv.imshow('canny',canny)

ret,thres = cv.threshold(gray,125,255,cv.THRESH_BINARY)
# contours,hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

# print(contours)

# contours1,hierarchy = cv.findContours(thres,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img,contours,-1,(255,0,0),-1)

# cv.imshow('img',img)
# # print(contours)

# # area = cv.contourArea(contours)
# # print(area)

# # print(len(contours))
# # print(len(contours1))
 

def check(image):
    contours1, hierarchy = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # import json
    # contours_serializable = [c.tolist() for c in contours1]
    # return json.dumps(contours_serializable)
    

print(check('../images/spiderman.jpg'))

# print("Implementing contour functions: ")
# max_area = 0
# largest_contour = None

# for i, c in enumerate(contours):
#     area = cv.contourArea(c)

#     if area > max_area:
#         max_area  = area
#         largest_contour = c

# if area ==max_area:
#        x,y,h,w= cv.boundingRect(contours)
#        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),-1)
#        cv.imshow('img',img)
    
# print(f"largest area is {max_area}")



cv.waitKey(0)
