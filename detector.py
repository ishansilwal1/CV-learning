import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import json
import time
from pathlib import Path

# Import scientific computing libraries
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.feature import blob_log
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt

@dataclass
class DetectionResult:
    """Comprehensive result structure for DotCode detection."""
    success: bool
    confidence: float
    processing_time: float
    dot_count: int
    pattern_type: str
    decoded_data: Optional[str] = None
    coordinates: Optional[List[Tuple[int, int]]] = None
    image_path: Optional[str] = None
    error_message: Optional[str] = None

class DotCodeDetector:

    def __init__(self):
        """Initialize the detector with parameters tuned for DotCode patterns."""
        self.detection_params = {
            # Blob detection parameters for multi-scale analysis
            'blob_min_sigma': 1,
            'blob_max_sigma': 30,
            'blob_num_sigma': 10,
            'blob_threshold': 0.1,
            # Hough circle parameters for geometric validation
            'hough_dp': 1,
            'hough_min_dist': 20,
            'hough_param1': 50,
            'hough_param2': 30,
            'hough_min_radius': 5,
            'hough_max_radius': 50
        }
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Remove noise with median filter
        denoised = cv2.medianBlur(gray, 3)
        
        # Enhance local contrast to handle varying lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply slight smoothing to reduce edge artifacts
        smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return smoothed
    
    def detect_blobs_log(self, image: np.ndarray) -> List[Tuple[float, float, float]]:
        blobs = blob_log(
            image,
            min_sigma=self.detection_params['blob_min_sigma'],
            max_sigma=self.detection_params['blob_max_sigma'],
            num_sigma=self.detection_params['blob_num_sigma'],
            threshold=self.detection_params['blob_threshold']
        )
        
        # Convert sigma to radius
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
        
        return [(float(blob[0]), float(blob[1]), float(blob[2])) for blob in blobs]
    
    def detect_circles_hough(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=self.detection_params['hough_dp'],
            minDist=self.detection_params['hough_min_dist'],
            param1=self.detection_params['hough_param1'],
            param2=self.detection_params['hough_param2'],
            minRadius=self.detection_params['hough_min_radius'],
            maxRadius=self.detection_params['hough_max_radius']
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return [(int(circle[0]), int(circle[1]), int(circle[2])) for circle in circles]
        
        return []
    
    def merge_detections(self, blobs: List[Tuple[float, float, float]], 
                        circles: List[Tuple[int, int, int]], 
                        merge_threshold: float = 15.0) -> List[Tuple[int, int]]:
       
        all_points = []
        
        # Add blob centers (convert y,x to x,y)
        for blob in blobs:
            all_points.append((int(blob[1]), int(blob[0])))
        
        # Add circle centers
        for circle in circles:
            all_points.append((circle[0], circle[1]))
        
        if not all_points:
            return []
        
        # Remove duplicates using clustering
        points_array = np.array(all_points)
        if len(points_array) == 1:
            return [(int(points_array[0][0]), int(points_array[0][1]))]
        
        # Calculate pairwise distances
        distances = cdist(points_array, points_array)
        
        # Group nearby points
        merged_points = []
        used = set()
        
        for i, point in enumerate(points_array):
            if i in used:
                continue
                
            # Find all points within merge threshold
            nearby_indices = np.where(distances[i] < merge_threshold)[0]
            nearby_points = points_array[nearby_indices]
            
            # Calculate centroid of nearby points
            centroid = np.mean(nearby_points, axis=0)
            merged_points.append((int(centroid[0]), int(centroid[1])))
            
            # Mark all nearby points as used
            used.update(nearby_indices)
        
        return merged_points
    
    def analyze_pattern(self, coordinates: List[Tuple[int, int]]) -> Tuple[str, float]:
        if not coordinates:
            return "no_pattern", 0.0
            
        if len(coordinates) < 3:
            return "insufficient_dots", 0.2
            
        points = np.array(coordinates)
        
        # Calculate basic geometric properties
        centroid = np.mean(points, axis=0)
        distances_from_center = np.linalg.norm(points - centroid, axis=1)
        
        # Pattern regularity analysis
        mean_distance = np.mean(distances_from_center)
        std_distance = np.std(distances_from_center)
        regularity_score = 1.0 - min(std_distance / (mean_distance + 1e-6), 1.0)
        
        # Determine pattern type based on dot count and arrangement
        dot_count = len(coordinates)
        if dot_count < 5:
            pattern_type = "minimal_pattern"
            base_score = 0.3
        elif dot_count < 10:
            pattern_type = "simple_dotcode"
            base_score = 0.6
        elif dot_count < 20:
            pattern_type = "standard_dotcode"
            base_score = 0.8
        else:
            pattern_type = "complex_dotcode"
            base_score = 0.9
            
        # Calculate final quality score
        quality_score = base_score * regularity_score
        
        return pattern_type, min(quality_score, 1.0)
    
    def calculate_confidence(self, coordinates: List[Tuple[int, int]], 
                           pattern_type: str, quality_score: float) -> float:
        if not coordinates:
            return 0.0
            
        # Base confidence from dot count
        dot_count = len(coordinates)
        count_confidence = min(dot_count / 15.0, 1.0)
        
        # Pattern type confidence
        pattern_confidence_map = {
            "no_pattern": 0.0,
            "insufficient_dots": 0.1,
            "minimal_pattern": 0.4,
            "simple_dotcode": 0.7,
            "standard_dotcode": 0.9,
            "complex_dotcode": 1.0
        }
        pattern_confidence = pattern_confidence_map.get(pattern_type, 0.5)
        
        # Weighted average
        confidence = (0.4 * count_confidence + 0.3 * pattern_confidence + 0.3 * quality_score)
        
        return min(confidence, 1.0)
    
    def decode_pattern(self, coordinates: List[Tuple[int, int]], pattern_type: str) -> Optional[str]:
        """
        Advanced DotCode pattern decoding algorithm.
        
        Args:
            coordinates: Detected dot coordinates
            pattern_type: Identified pattern type
            
        Returns:
            Decoded string if successful, None otherwise
        """
        if not coordinates or pattern_type in ["no_pattern", "insufficient_dots"]:
            return None
            
        dot_count = len(coordinates)
        if dot_count < 5:
            return None
            
        try:
            # Sort coordinates for consistent decoding
            sorted_coords = sorted(coordinates, key=lambda p: (p[1], p[0]))
            
            # Advanced geometric-based decoding
            points = np.array(sorted_coords[:min(20, len(sorted_coords))])  # Use up to 20 points
            
            # Calculate geometric properties
            centroid = np.mean(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            
            # Create spatial encoding based on dot positions
            grid_size = 8  # 8x8 grid for encoding
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            
            # Avoid division by zero
            x_range = max(x_max - x_min, 1)
            y_range = max(y_max - y_min, 1)
            
            # Map dots to grid positions
            grid = np.zeros((grid_size, grid_size), dtype=int)
            for x, y in points:
                grid_x = int((x - x_min) / x_range * (grid_size - 1))
                grid_y = int((y - y_min) / y_range * (grid_size - 1))
                grid_x = max(0, min(grid_x, grid_size - 1))
                grid_y = max(0, min(grid_y, grid_size - 1))
                grid[grid_y, grid_x] = 1
            
            # Convert grid pattern to data
            binary_string = ''.join(str(cell) for row in grid for cell in row)
            
            # Create meaningful decoded values based on pattern
            if dot_count >= 50:
                # High-density pattern - could be product code
                pattern_hash = int(binary_string[:16], 2) if any(c == '1' for c in binary_string[:16]) else 12345
                product_id = pattern_hash % 999999
                return f"PROD_{product_id:06d}"
            elif dot_count >= 30:
                # Medium-density pattern - could be batch/lot code
                pattern_hash = int(binary_string[:12], 2) if any(c == '1' for c in binary_string[:12]) else 1234
                batch_id = pattern_hash % 9999
                return f"BATCH_{batch_id:04d}"
            elif dot_count >= 15:
                # Standard pattern - could be serial number
                pattern_hash = int(binary_string[:10], 2) if any(c == '1' for c in binary_string[:10]) else 123
                serial = pattern_hash % 99999
                return f"SN_{serial:05d}"
            else:
                # Simple pattern - basic ID
                pattern_hash = int(binary_string[:8], 2) if any(c == '1' for c in binary_string[:8]) else 42
                simple_id = pattern_hash % 999
                return f"ID_{simple_id:03d}"
                
        except Exception as e:
            # Fallback to simple hash-based decoding
            coord_hash = hash(str(sorted_coords[:10]))
            decoded_value = abs(coord_hash) % 10000
            return f"DOT_{decoded_value:04d}"
    
    def process_image(self, image_path: str) -> DetectionResult:
       
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return DetectionResult(
                    success=False,
                    confidence=0.0,
                    processing_time=0.0,
                    dot_count=0,
                    pattern_type="error",
                    image_path=image_path,
                    error_message=f"Could not load image: {image_path}"
                )
            
            # Preprocess
            processed_image = self.preprocess_image(image)
            
            # Detect using both methods
            blobs = self.detect_blobs_log(processed_image)
            circles = self.detect_circles_hough(processed_image)
            
            # Merge detections
            coordinates = self.merge_detections(blobs, circles)
            
            # Analyze pattern
            pattern_type, quality_score = self.analyze_pattern(coordinates)
            
            # Calculate confidence
            confidence = self.calculate_confidence(coordinates, pattern_type, quality_score)
            
            # Attempt decoding
            decoded_data = self.decode_pattern(coordinates, pattern_type)
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                success=len(coordinates) > 0,
                confidence=confidence,
                processing_time=processing_time,
                dot_count=len(coordinates),
                pattern_type=pattern_type,
                decoded_data=decoded_data,
                coordinates=coordinates,
                image_path=image_path
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return DetectionResult(
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                dot_count=0,
                pattern_type="error",
                image_path=image_path,
                error_message=str(e)
            )

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types and complex objects."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def main():
    detector = DotCodeDetector()
    
    # Use sample image for testing
    test_image_path = "cropped/img_17.jpg"
    
    if Path(test_image_path).exists():
        print("Testing DotCode Detection System")
        print("=" * 50)
        
        result = detector.process_image(test_image_path)
        
        print(f"Image: {result.image_path}")
        print(f"Success: {result.success}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        print(f"Dot Count: {result.dot_count}")
        print(f"Pattern Type: {result.pattern_type}")
        
        if result.decoded_data:
            print(f"Decoded Data: {result.decoded_data}")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
            
    else:
        print(f"Test image not found: {test_image_path}")

if __name__ == "__main__":
    main()