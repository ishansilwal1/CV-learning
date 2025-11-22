import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from dotcode_detector_main import DotCodeDetector, DetectionResult
class DotCodeVisualizer:  
    def __init__(self):
        """Initialize the visualizer."""
        self.colors = {
            'dot': (0, 255, 0),        # Green for detected dots
            'boundary': (255, 0, 0),   # Red for boundary
            'text': (255, 255, 255),   # White for text
            'background': (0, 0, 0)    # Black for background
        }
        
    def create_overlay(self, image_path: str, result: DetectionResult, 
                      output_path: Optional[str] = None) -> str:
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Create a copy for overlay
        overlay = image.copy()
        
        if result.success and result.coordinates:
            # Draw detected dots with green overlay
            for x, y in result.coordinates:
                cv2.circle(overlay, (int(x), int(y)), 4, self.colors['dot'], -1)  # Filled green circle
        else:
            # Draw failure message
            cv2.putText(overlay, "DETECTION FAILED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if result.error_message:
                cv2.putText(overlay, f"Error: {result.error_message}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Determine output path
        if output_path is None:
            input_path = Path(image_path)
            output_path = f"visualization_{input_path.stem}.jpg"
        
        # Save overlay image
        cv2.imwrite(output_path, overlay)
        return output_path
    
    def create_comparison_view(self, image_path: str, result: DetectionResult, 
                              output_path: Optional[str] = None) -> str:
        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Create overlay version
        overlay_path = self.create_overlay(image_path, result, "temp_overlay.jpg")
        overlay = cv2.imread(overlay_path)
        
        # Resize images to same height if needed
        height = min(original.shape[0], overlay.shape[0], 600)  # Max height 600px
        aspect_ratio_orig = original.shape[1] / original.shape[0]
        aspect_ratio_over = overlay.shape[1] / overlay.shape[0]
        
        width_orig = int(height * aspect_ratio_orig)
        width_over = int(height * aspect_ratio_over)
        
        original_resized = cv2.resize(original, (width_orig, height))
        overlay_resized = cv2.resize(overlay, (width_over, height))
        
        # Create side-by-side comparison
        comparison = np.hstack([original_resized, overlay_resized])
        
        # Add titles
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Detection Result", (width_orig + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Determine output path
        if output_path is None:
            input_path = Path(image_path)
            output_path = f"comparison_{input_path.stem}.jpg"
        
        # Save comparison image
        cv2.imwrite(output_path, comparison)
        
        # Clean up temporary file
        Path("temp_overlay.jpg").unlink(missing_ok=True)
        
        return output_path

def main():
    print("DotCode Visualization Demo")
    print("=" * 40)
    
    # Initialize detector and visualizer
    detector = DotCodeDetector()
    visualizer = DotCodeVisualizer()
    
    # Test with sample image
    test_image = "cropped/img_17.jpg"
    
    if Path(test_image).exists():
        print(f"Processing: {test_image}")
        
        # Detect DotCode
        result = detector.process_image(test_image)
        
        # Create visualizations
        overlay_path = visualizer.create_overlay(test_image, result)
        comparison_path = visualizer.create_comparison_view(test_image, result)
        
        print(f"Overlay created: {overlay_path}")
        print(f"Comparison created: {comparison_path}")
        print(f"Detection confidence: {result.confidence:.3f}")
        print(f"Dots detected: {result.dot_count}")
        
    else:
        print(f"Test image not found: {test_image}")

if __name__ == "__main__":
    main()