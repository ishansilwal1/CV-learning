import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

img = cv.imread("../images/spiderman.jpg")

cv.imshow('img',img)



# Averaging : 

# average = cv.blur(img,(7,7))
# cv.imshow("blur",average)

# gaussian = cv.GaussianBlur(img,(3,3),0)
# cv.imshow('gaussian',gaussian)



median = cv.medianBlur(img,7)
cv.imshow('median',median)

cv.waitKey(0)

