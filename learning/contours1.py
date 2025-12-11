import cv2 as cv 
import matplotlib.pyplot as plt 

import numpy as np 


img = cv .imread("../images/spiderman.jpg")
cv.imshow('img',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray,125,175)
cv.imshow('canny',canny)

ret,thres = cv.threshold(gray,125,255,cv.THRESH_BINARY)
# contours,hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

# print(contours)

# contours1,hierarchy = cv.findContours(thres,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img,contours,-1,(255,0,0),-1)

# cv.imshow('img',img)
# # print(contours)

# # area = cv.contourArea(contours)
# # print(area)

# # print(len(contours))
# # print(len(contours1))
 

def check(image):
    contours1, hierarchy = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # import json
    # contours_serializable = [c.tolist() for c in contours1]
    # return json.dumps(contours_serializable)
    

print(check('../images/spiderman.jpg'))

# print("Implementing contour functions: ")
# max_area = 0
# largest_contour = None

# for i, c in enumerate(contours):
#     area = cv.contourArea(c)

#     if area > max_area:
#         max_area  = area
#         largest_contour = c

# if area ==max_area:
#        x,y,h,w= cv.boundingRect(contours)
#        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),-1)
#        cv.imshow('img',img)
    
# print(f"largest area is {max_area}")



cv.waitKey(0)


def get_parallel_line(self,line: tuple[float, float, float], offset: float) -> tuple[float, float, float]:
        """
        Author: Pranav Subedi
        Function: get_parallel_line (Normalized)
        Description:
            Given a normalized line in standard form:
            this function returns another normalized line that is
            offset by 'px' pixels downward (in the positive Y direction).

            Since the line is normalized, the offset simply shifts C by 'px':
                C' = C + px

            This ensures the output line remains normalized, i.e.,
            sqrt(A² + B²) = 1, and the parallel distance between
            the two lines equals exactly 'px' pixels.
        Inputs:
            line_coeffs (tuple[float, float, float]): Coefficients (A, B, C) of a normalized line such that sqrt(A² + B²) = 1.
            px (float): Offset distance in pixels.
                Positive = downward (in image coordinates).
                Negative = upward.
        Output:
            new_line (tuple[float, float, float]): Coefficients (A, B, C') of the new normalized parallel line.
        """
        A, B, C = map(float, line)

        # --- Verify normalization ---
        norm = np.hypot(A, B)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(
                f"Line must be normalized (√(A²+B²)=1), but got {norm:.6f}. "
                "Normalize it first using to_standard_line(..., normalize=True)."
            )

         # --- Offset line ---
        C_new = C + offset
        new_line=(A, B, C_new)
        return new_line