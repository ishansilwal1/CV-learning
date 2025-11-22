import cv2 as cv
import numpy as np 


img = cv.imread('image2.jpg')
cv.imshow('original',img)


gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)



blur = cv.GaussianBlur(gray,(9,9),cv.BORDER_DEFAULT)
cv.imshow('blur',blur)
canny = cv.Canny(img,125,175)
cv.imshow('canny',canny)

# Thresholding looks to an image and binarize it based on a threshold value
# pixels with intensity values above the threshold are set to the maximum value (usually 255),
#while those below the threshold are set to 0.
# Here we use binary thresholding
# ret, thres = cv.threshold(blur,175,255,cv.THRESH_BINARY)

# contours is a list of all the contours in the image
# hierarchy is the optional output vector containing information about the image topology
contours,hierarchy = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

print (len(contours))

blank = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
cv.imshow('blank',blank)

cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('contours drawn',blank)





cv.waitKey(0)