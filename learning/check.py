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



class FeatureDetectionClassical(Matching):
    def _init_(self):
        super()._init_()
    
    def detect_harris( self,image: np.ndarray,
                    block_size: int = 2,
                    ksize: int = 3,
                    k: float = 0.04,
                    thresh: float = 0.01) -> np.ndarray:
        """
        Author: Crimson Tech
        Function: detect_harris
        Description: Detects corners in an image using the Harris corner detection algorithm.
        Inputs:
            image (np.ndarray): Input BGR image.
            block_size (int, optional): Neighborhood size for corner detection. Defaults to 2.
            ksize (int, optional): Aperture parameter for the Sobel operator. Defaults to 3.
            k (float, optional): Harris detector free parameter. Defaults to 0.04.
            thresh (float, optional): Threshold for detecting strong corners. Defaults to 0.01.
        Output:
            put (np.ndarray): Output image with detected corners marked in red.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        dst = cv2.dilate(dst, None)
        out = image.copy()
        out[dst > thresh * dst.max()] = [0, 0, 255]
        return out

    def detect_shi_tomasi(image: np.ndarray,
                        max_corners: int = 100,
                        quality_level: float = 0.01,
                        min_distance: float = 10) -> np.ndarray:
        """
        Author: Crimson Tech
        Function: detect_shi_tomasi
        Description: Detects corners in an image using the Shi-Tomasi corner detection algorithm.
        Inputs:
            image (np.ndarray): Input BGR image.
            max_corners (int, optional): Maximum number of corners to return. Defaults to 100.
            quality_level (float, optional): Minimum accepted quality of image corners. Defaults to 0.01.
            min_distance (float, optional): Minimum possible Euclidean distance between the returned corners. Defaults to 10.
        Output:
            out (np.ndarray): Output image with detected corners marked in green.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
        out = image.copy()
        if corners is not None:
            for x, y in corners.reshape(-1, 2).astype(int):
                cv2.circle(out, (x, y), 4, (0, 255, 0), -1)
        return out

    def detect_fast(image: np.ndarray,
                    threshold: int = 10,
                    nonmax_suppression: bool = True) -> np.ndarray:
        
        """
        Author: Crimson Tech
        Function: detect_fast
        Description: Detects FAST features in an image.
        Inputs:
            image (np.ndarray): Input BGR image.
            threshold (int, optional): Detection threshold. Defaults to 10.
            nonmax_suppression (bool, optional): If true, non-maximum suppression is applied. Defaults to True.
        Output:
            image (np.ndarray): Output image with detected FAST features marked in red.
        """

        fast = cv2.FastFeatureDetector_create(threshold, nonmax_suppression)
        keypoints = fast.detect(image, None)
        image = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0))
        return image

    def detect_orb(image: np.ndarray,
                n_features: int = 500) -> np.ndarray:
        
        """
        Author: Crimson Tech
        Function: detect_orb
        Description: Detects ORB features in an image.
        Inputs:
            image (np.ndarray): Input BGR image.
            n_features (int, optional): Maximum number of features to detect. Defaults to 500.
        Output:
            image (np.ndarray): Output image with detected ORB features marked in yellow.
        """
        orb = cv2.ORB_create(nfeatures=n_features)
        keypoints = orb.detect(image, None)
        image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 255))
        return image

    def detect_mser(image: np.ndarray) -> np.ndarray:
        """
        Author: Crimson Tech
        Function: detect_mser
        Description:            
            Detects Maximally Stable Extremal Regions (MSERs) in an image.
        Input:
            image (np.ndarray): Input BGR image.
        Output:
            out (np.ndarray): Output image with detected MSERs outlined in magenta.
        """
        mser = cv2.MSER_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        regions, _ = mser.detectRegions(gray)
        out = image.copy()
        for pts in regions:
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))
            cv2.polylines(out, [hull], True, (255, 0, 255), 1)
        return out
