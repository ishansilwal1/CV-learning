import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 


## drawing circles lines rectangles , adding lines and texts

img = cv.imread('image2.jpg')
cv.imshow('original',img)


plt.imshow(img[:,:,::-1])
cv.waitKey(0)