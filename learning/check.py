import cv2 
import numpy as np 
# practice 

image = cv.imread('dots.jpg')
# cv.imshow('image',image)

# Resizing the image 
# def rescaleFrame(frame,scale=0.5):
#     width = int(frame.shape[1]*scale)
#     height = int(frame.shape[0]*scale)
#     dimensions = (width,height)
#     return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

# resized_image = rescaleFrame(image,scale=0.2)
# cv.imshow('resized image',resized_image)




# blank image now 
blank = np.zeros((500,500,3),dtype='uint8')
# cv.imshow('blank',blank)

# print the image red 
# blank[:]= 0,0,255
# cv.imshow('red',blank)


# print some pixels
# blank[200:300 ,300:400]= 0,0,255
# cv.imshow('check',blank)



# DRaw a shape
cv.rectangle(blank,(0,0),(250,250),(255,0,0),thickness=-1)
cv.circle(blank,(250,250),40,(0,0,255),thickness = 3)
cv.line(blank,(255,255),(500,500),(255,0,0),thickness = 3)

cv.imshow('shapes',blank)
cv.waitKey(0)





# # Syntax being USED 
# lab_image = cv2.cvtColor(image, cv2.color_bgr2lab)

# #Syntax to use : 
# lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)


# def crop_by_region_of_interest(self, image: np.ndarray, roi: list) -> np.ndarray:
#         """
#         Author: Crimson Tech
#         Function: crop_by_region_of_interest
#         Description: Crop the image based on a single ROI.
#         Inputs:
#             image (np.ndarray): Input image
#             roi (list): A list containing one dict with keys 'x', 'y', 'width', 'height'
#         Output:
#             cropped_image (np.ndarray): Cropped image
#         """
#         x = int(roi[0]["x"])
#         y = int(roi[0]["y"])
#         w = int(roi[0]["width"])
#         h = int(roi[0]["height"])

#         x_end = min(x + w, image.shape[1])
#         y_end = min(y + h, image.shape[0])
#         x = max(x, 0)
#         y = max(y, 0)
#         cropped_image = image[y:y_end, x:x_end]
#         return cropped_image

#   def text_detection(self, image, model_path, use_server_net=True):
#         """
#         Author: Shreejit Gautam
#         Function: Text Detection
#         Description: Detects text regions in the image using the provided model path.
#         Inputs:
#             image (ndarray): Input image
#             model_path (str): Detection model path if '/' in model, else model name
#             use_server_net (bool): Server model if true else mobile model

#         Outputs:
#             bounding_boxes (list): Detected text region coordinates.
#             annotated_image (np.ndarray): Image with detection results visualized.
#         """
        
#         print("Loading detection model...")
#         annotated_image = image.copy()
#         if "/" in model_path :
#             last_slash_index = model_path.rfind("/")  
#             model_name = model_path[last_slash_index + 1:]
#             model_dir = model_path[:last_slash_index]
#         else:
#             model_name = model_path
#             model_dir = None
        
#         if use_server_net:
#             if 'server_det' not in model_name:
#                 model_name = f'{model_name}_server_det'
#         else:
#             if 'mobile_det' not in model_name:
#                 model_name = f'{model_name}_mobile_det'

#         text_detection = TextDetection(model_name=model_name, model_dir=model_dir)
#         output = text_detection.predict(image)
#         for item in output:
#             rec_polys = item.get("dt_polys", [])
#         bounding_boxes = [p for p in rec_polys if Polygon(p).area >= 10]
        
#         # Draw bounding_boxes 
#         for box in bounding_boxes:
#             box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
#             annotated_image = cv2.polylines(np.array(annotated_image), [box], True, (255, 0, 0), 2)
#         return bounding_boxes, annotated_image


