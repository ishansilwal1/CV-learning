import cv2 as cv
import numpy as np

def detect_circles(image_path, dp=1.2, minDist=100, param1=50, param2=30, minRadius=30, maxRadius=400):
    img = cv.imread(image_path)
    if img is None:
        print('Image not found:', image_path)
        return []
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9, 9), 2)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Find the largest circle (outer washer)
        largest_circle = max(circles[0, :], key=lambda x: x[2])
        # Filter circles that are close to the center of the largest circle
        center_x, center_y = largest_circle[0], largest_circle[1]
        centered_circles = [c for c in circles[0, :] if np.sqrt((c[0]-center_x)**2 + (c[1]-center_y)**2) < 50]
        # Sort by radius and select smallest and largest
        sorted_circles = sorted(centered_circles, key=lambda x: x[2])
        main_circles = [sorted_circles[0], sorted_circles[-1]] if len(sorted_circles) > 1 else sorted_circles
        for i in main_circles:
            radius = i[2]
            diameter = 2 * radius
            print(f"Radius: {radius}, Diameter: {diameter}")
            cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv.imshow('Main Detected Circles', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return [list(map(int, c)) for c in main_circles]
    else:
        print('No circles detected.')
        return []

# Example usage for testing:
if __name__ == "__main__":
    # Use the correct image path
    circles = detect_circles(r'D:\CV practice\images\circle.jpg')
    print('Detected circles:', circles)
