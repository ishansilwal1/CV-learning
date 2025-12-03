import cv2 as cv
import numpy as np

def detect_minarea_rect(image_path):
    """Detect contours and draw minimum area rectangles around them."""
    img = cv.imread(image_path)
    if img is None:
        print('Image not found:', image_path)
        return []
    
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Apply threshold or edge detection
    _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    for contour in contours:
        # Get minimum area rectangle
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.intp(box)
        
        # Draw the rectangle
        cv.drawContours(img, [box], 0, (0, 255, 0), 2)
        
        # Extract rectangle parameters
        center, size, angle = rect
        rectangles.append({
            'center': center,
            'width': size[0],
            'height': size[1],
            'angle': angle,
            'area': size[0] * size[1]
        })
        
        # Print rectangle info
        print(f"Center: {center}, Size: {size}, Angle: {angle:.2f}°, Area: {size[0]*size[1]:.2f}")
    
    # Display result
    cv.imshow('Minimum Area Rectangles', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return rectangles

# Example usage
if __name__ == "__main__":
    rectangles = detect_minarea_rect(r'D:\CV practice\images\circle.jpg')
    print(f"\nTotal rectangles detected: {len(rectangles)}")
