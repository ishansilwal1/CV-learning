import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
img = np.zeros((512, 512, 3), dtype="uint8")

h, w, c = img.shape
rows = 3 
cols = 4

# Calculate the height of each row and the width of each column
cell_h = h // rows
cell_w = w // cols

# Draw lines now 
for i in range(1, cols):
    x_pos = i * cell_w
    cv.line(img, (x_pos, 0), (x_pos, h), (255,0,0),1)
for i in range(1, rows):
    y_pos = i * cell_h
    cv.line(img, (0, y_pos), (w, y_pos), (255,0,0),1)

# cv.imshow('image', img)


a = [11,7,3,2,1,0,4,8,9,10,11] # sequence of index to display images in order 

img2 = cv.imread("D:\CV practice\learning\image2.jpg")
resized = cv.resize(img2,(cell_w,cell_h)) 

for index in a:
    row_index = index//cols # to get which row the index is referring 
    col_index = index%cols # to get which column the index is referring 
    temp_img = img.copy()
    # coordinates of the points now 
    y1 = row_index*cell_h
    y2 = y1+170
    x1 = col_index*cell_w
    x2 = x1+128
    # temporary image = resized image to display in the blink 
    temp_img[y1:y2, x1:x2] = resized
    cv.imshow("check",temp_img)
    cv.waitKey(1000)

plt.imshow(img)

cv.waitKey(0)
cv.destroyAllWindows()