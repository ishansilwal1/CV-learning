import cv2 as cv
import numpy as np 

img = cv.imread('image2.jpg')
cv.imshow('original',img)

# Translation of image 
# imageshape 0 - width image shape 1 - height 
def translate (img,x,y):
    transmat = np.float32([[1,0,x],[0,1,y]])
    dimensions = img.shape[1],img.shape[0]
    return cv.warpAffine(img,transmat,dimensions)

# -x --> left
# -y --> up
#+x --> right
#+y --> down
translated = translate(img,100,100)
cv.imshow('translated',translated)
 

# roTATION 
#ANGLE IN DEGREES
def rotation (img,angle,rotpoint=None):
    (height,width) = img.shape[:2]
    if rotpoint is None:
        rotpoint = (width//2,height//2)
    rotmat = cv.getRotationMatrix2D(rotpoint,angle,1.0)
    dimensions = (width,height)
    return cv.warpAffine(img,rotmat,dimensions)

rotated = rotation(img,45)
cv.imshow('rotated',rotated)











cv.waitKey(0)