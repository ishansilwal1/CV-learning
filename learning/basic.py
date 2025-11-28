import cv2 as cv

img = cv.imread('image2.jpg')


# cv.imshow('dots',img)
# converting to grayscale does what ?

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)


# blurring a image  remove noise in the image  

#kernel size must be odd numbers 

blur = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
# cv.imshow('blur',blur)


# Edge cascade is ued to detect the edges in the image
canny = cv.Canny(blur,125,175)
cv.imshow('canny',canny)



# dilating the image using structuring element
dilated = cv.dilate(canny,(7,7),iterations=3)
cv.imshow('dilated',dilated)

#Eroding the image 
eroding = cv.erode(dilated,(7,7),iterations=3)
cv.imshow('eroded',eroding)




#REsize 
resized = cv.resize(img,(500,500),interpolation=cv.INTER_LINEAR)
cv.imshow('resized',resized)


# cropping
cropped = img[200:400,300:400]
cv.imshow('cropped',cropped)





cv.waitKey(0)
 