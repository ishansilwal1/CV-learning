# # import cv2 as cv
# # import numpy as np 


# # img = cv.imread('image2.jpg')
# # cv.imshow('original',img)


# # gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # cv.imshow('gray',gray)



# # blur = cv.GaussianBlur(gray,(9,9),cv.BORDER_DEFAULT)
# # cv.imshow('blur',blur)
# # canny = cv.Canny(img,125,175)
# # cv.imshow('canny',canny)

# # # Thresholding looks to an image and binarize it based on a threshold value
# # # pixels with intensity values above the threshold are set to the maximum value (usually 255),
# # #while those below the threshold are set to 0.
# # # Here we use binary thresholding
# # # ret, thres = cv.threshold(blur,175,255,cv.THRESH_BINARY)

# # # contours is a list of all the contours in the image
# # # hierarchy is the optional output vector containing information about the image topology
# # contours,hierarchy = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

# # print (len(contours))

# # blank = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
# # cv.imshow('blank',blank)

# # cv.drawContours(blank, contours, -1, (0,0,255), 1)
# # cv.imshow('contours drawn',blank)





# # cv.waitKey(0)
# import cv2 as cv 
# import matplotlib.pyplot as plt 
# import numpy as np 


# img = cv.imread("../images/image2.jpg")
# # Check if image loaded correctly
# if img is None:
#     print("Error: Could not read image. Check the file path.")
#     exit()

# # cv.imshow('img',img) # Can comment out extra imshow calls

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# canny = cv.Canny(gray,125,175)
# # cv.imshow('canny',canny)

# ret,thres = cv.threshold(gray,125,255,cv.THRESH_BINARY)
# contours,hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)


# contours1,hierarchy = cv.findContours(thres,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

# # Draw ALL contours first in blue
# cv.drawContours(img,contours,-1,(255,0,0),1)

# # cv.imshow('img',img) # Can comment out extra imshow calls
 
# print("Implementing contour functions: ")
# max_area = 0
# largest_contour = None

# # Loop to FIND the largest contour
# for i, c in enumerate(contours):
#     area = cv.contourArea(c)

#     if area > max_area:
#         max_area  = area
#         largest_contour = c

# # --- FIXES START HERE ---

# # 1. We are now OUTSIDE the 'for' loop.
# # 2. Check if we actually found a contour (prevents errors if list was empty)
# if largest_contour is not None:
#     # 3. Calculate bounding rectangle (Corrected order: x, y, w, h)
#     x, y, w, h = cv.boundingRect(largest_contour)
    
#     # 4. Draw the rectangle ONLY on the FINAL largest contour, in GREEN for clarity
#     # You can remove the 'cv.imshow("imh", img)' from inside the loop completely.
#     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# # --- FIXES END HERE ---

# print(f"Largest area is {max_area} pixels")

# # Show the final result with blue contours AND the green rectangle
# cv.imshow('Final Image with Bounding Box',img)
   
# cv.waitKey(0)
# cv.destroyAllWindows()


#imports
import cv2
from matplotlib import pyplot as plt
import numpy as np

#loading img
file = "../images/spiderman.jpg"
img = cv2.imread(file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

#conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

#Func to find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
detected_contours = img.copy()
cv2.drawContours(detected_contours, contours, -1, (0, 255, 0), -1)
plt.imshow(detected_contours)
plt.title('Detected contours')
plt.show()

contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#masking img
highlight = np.ones_like(img)
cv2.drawContours(highlight, contours, -1, (0, 200, 175), cv2.FILLED)
plt.imshow(highlight)
plt.title('Highlight contour with color')
plt.show()

#masking again
mask = np.zeros_like(img)
cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)
foreground = cv2.bitwise_and(img, mask)

#edges
plt.imshow(foreground)
plt.title('Extract contours')
plt.show()

print("\t")

#--main-plot--
contours = {"Original": img, "Detected contours": detected_contours,
            "Color contours": highlight, "Extract contours": foreground}
plt.subplots_adjust(wspace=.2, hspace=.2)
plt.tight_layout()

for i, (key, value) in enumerate(contours.items()):
    plt.subplot(2, 2, i + 1)
    plt.tick_params(labelbottom=False)
    plt.tick_params(labelleft=False)
    plt.title("{}".format(key))
    plt.imshow(value)

plt.show()