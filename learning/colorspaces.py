import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 

img = cv.imread("../images/spiderman.jpg")

cv.imshow('img',img)

blank = np.zeros(img.shape[:2], dtype = "uint8")

# # converting to gray 

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)


# # BGR to hsv 

# hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# cv.imshow('hsv',hsv)



# #lab 
# lab = cv.cvtColor(img,cv.COLOR_BGR2LAB)
# cv.imshow('lab',lab)

# # bgr to rgb 

# rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)



# plt.imshow(rgb)
# plt.show()
# cv.waitKey(0)


b,g,r = cv.split(img)

cv.imshow('blue',b)
cv.imshow('green',g)
cv.imshow('red',r)

print(img.shape)
print(r.shape)
print(b.shape)
print(g.shape)


# # merged = cv.merge([b,g,r])
# cv.imshow('merged_img',merged)


blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])

cv.imshow('blue',blue)
cv.imshow('green',green)
cv.imshow('red',red)



cv.waitKey(0)
