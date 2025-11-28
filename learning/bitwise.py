import cv2 as cv 
import numpy as np 


img = cv.imread('../images/spiderman.jpg')


cv.imshow('img',img)

blank = np.zeros((512,512), dtype = "uint8")


rectangle = cv.rectangle(blank.copy(), (30,30),(400,400),255,-1)

circle = cv.circle(blank.copy(), (256,256), 200, 255,-1)
cv.imshow('blank',rectangle)
cv.imshow('crcle',circle)


bitwise_and = cv.bitwise_and(rectangle,circle)

bitwise_or = cv.bitwise_or(rectangle,circle)


# bitwise XOR -- finds non intersecting regions 

bitwise_xor = cv.bitwise_xor(rectangle,circle)

cv.imshow('bitwise_xor',bitwise_xor)



# bitwise not -- > returns nothing inverts binary color 


bitwise_not = cv.bitwise_not(rectangle,circle)
cv.imshow('bitwise_or',bitwise_or)
cv.imshow('bitwise_and', bitwise_and)
cv.imshow('bitwise_not',bitwise_not)
cv.waitKey(0)