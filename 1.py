import cv2 as cv
import numpy as np

# img = cv.imread('dots.jpg')
# cv.imshow('image',img)
img2 = cv.imread('image2.jpg')



# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)

# print(img,0)

# cv.waitKey(0)
cv.flip(img2,0,img2)
cv.imshow('image2',img2)

cv.waitKey(0)