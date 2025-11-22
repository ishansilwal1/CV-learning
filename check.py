import cv2 as cv
import numpy as np 
# practice 

image = cv.imread('dots.jpg')
# cv.imshow('image',image)

# Resizing the image 
# def rescaleFrame(frame,scale=0.5):
#     width = int(frame.shape[1]*scale)
#     height = int(frame.shape[0]*scale)
#     dimensions = (width,height)
#     return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

# resized_image = rescaleFrame(image,scale=0.2)
# cv.imshow('resized image',resized_image)




# blank image now 
blank = np.zeros((500,500,3),dtype='uint8')
# cv.imshow('blank',blank)

# print the image red 
# blank[:]= 0,0,255
# cv.imshow('red',blank)


# print some pixels
# blank[200:300 ,300:400]= 0,0,255
# cv.imshow('check',blank)



# DRaw a shape
cv.rectangle(blank,(0,0),(250,250),(255,0,0),thickness=-1)
cv.circle(blank,(250,250),40,(0,0,255),thickness = 3)
cv.line(blank,(255,255),(500,500),(255,0,0),thickness = 3)

cv.imshow('shapes',blank)
cv.waitKey(0)