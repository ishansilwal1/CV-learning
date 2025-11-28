import cv2 as cv 
import numpy as np 


img = cv.imread('../images/spiderman.jpg')

blank = np.zeros (img.shape[:2],dtype = "uint8")

# cv.imshow('blank',blank)

masking = cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),150,255,-1)

masked_image = cv.bitwise_and(img, img, mask=masking)
    
    # Display the results
cv.imshow('Original Image', img) 
cv.imshow('Mask', masking) 
cv.imshow('Masked Image', masked_image) 

cv.waitKey(0)
cv.destroyAllWindows()
# cv.imshow('img',img)



cv.waitKey(0)