import cv2 as cv
import numpy as np
def rescaleFrame(frame,scale=0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA )



img = cv.imread('dots.jpg')
resized_img = rescaleFrame(img,scale=0.5)
# cv.imshow('resized dots',resized_img)
# cv.waitKey(0)

# blank = np.zeros((# height width and color chanel),dtype='uint8')
blank = np.zeros((500,500,3),dtype='uint8')
# cv.imshow('blank',blank)
# cv.waitKey(0)

# # let's try to paint the image 
# blank [:] = 0,0,255 # color code blank [:] means select all the pixels
# cv.imshow('red',blank)

# to print some pixels only 

# blank[200:300, 300:400] = 0,255,0
# cv.imshow('green',blank)








# draw a rectangle 
cv.rectangle(blank,(0,0),(250,250),(255,0,0),thickness=-1)
cv.imshow('rectangle',blank)





# draw a circle 
cv.circle(blank,(250,250),40,(0,0,255),thickness=-1)
cv.imshow('circle',blank)

# Draw a line 

cv.line(blank,(0,0),(500,500),(255,0,0),thickness=3)

cv.imshow('line',blank)





# write a text 
cv.putText(blank,'Hello ISHAN',(225,225),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),thickness=2)
cv.imshow('text',blank)
cv.waitKey(0)
