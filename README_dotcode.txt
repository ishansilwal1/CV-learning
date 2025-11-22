DotCode Detector — Beginner README

Purpose
This README explains, in simple steps, what the DotCode detector and visualizer do and how to run them on Windows (PowerShell). It also includes quick troubleshooting and tuning tips.

Files of interest (in your workspace)
- dotcode_detector_main.py  (detector code)
- dotcode_visualizer.py    (creates overlay & comparison images)
- detector.py              (duplicate/alternate detector file location)
- cropped/                 (place sample images here, e.g. img_17.jpg)

Requirements
- Python 3.8+ (recommended)
- A virtual environment (optional but recommended)
- Packages: opencv-python, numpy, scipy, scikit-image, matplotlib

Quick install (PowerShell)
# from the workspace root (D:\CV practice)
# 1) (optional) activate your virtual environment if you have one
env\Scripts\Activate.ps1

# 2) Install required packages (use pip)
python -m pip install --upgrade pip
python -m pip install opencv-python numpy scipy scikit-image matplotlib

Running the detector (simple)
- The detector scripts expect a sample image at a relative path like "cropped/img_17.jpg" by default.
- You can run the detector script that exists in the repository. Example PowerShell commands:

# Run the main detector (prints results to console)
python "d:\Computer vision\dotcode_detector_main.py"

# Or if your detector is in this folder as "detector.py"
python "d:\CV practice\detector.py"

Notes:
- If the script prints "Test image not found: cropped/img_17.jpg", put your sample image at that path or edit the script to point to an existing image.
- The detector will print a DetectionResult summary with fields like success, confidence, dot_count, and any decoded_data.

Running the visualizer (creates images)
- The visualizer uses the detector to produce overlay and comparison images.

# Example (PowerShell)
python "d:\Computer vision\dotcode_visualizer.py"

Expected outputs:
- Files named like "visualization_img_17.jpg" and "comparison_img_17.jpg" will be saved in the current working directory (or the directory where the script runs).

How it works (very short)
1. Preprocess image: grayscale -> median blur -> CLAHE -> small Gaussian blur.
2. Detect candidate dots using two methods:
   - blob_log (Laplacian of Gaussian across scales) to find bright blob centers
   - HoughCircles to geometrically detect circle edges
3. Merge both detection lists (cluster nearby detections).
4. Analyze dot positions to classify pattern type and compute a confidence score.
5. Attempt a heuristic decoding of the pattern into a short ID string.
6. Visualizer draws green dots at detected positions and saves images for review.

Tips for beginners (tuning & troubleshooting)
- If you get no detections:
  - Ensure the image path is correct and file is readable.
  - Check image contrast: try opening the image to verify brightness/contrast.
  - Loosen thresholds: increase blob detection range (min_sigma/max_sigma) or lower blob_log threshold.
- If you get too many false detections:
  - Increase the blob_log threshold parameter.
  - Add stronger denoising (median blur radius) before detection.
  - Filter detected blobs by expected radius (min/max) after detection.
- Visualization:
  - Open the saved visualization image to confirm detected dot positions.
  - If dots are slightly off, tune merge_detections merge_threshold or adjust preprocessing.

Developer tips
- Use the provided preprocessing as a starting point. Lighting can make a big difference; CLAHE helps when lighting varies.
- For parameter tuning, add a small script to draw detections on images (visual feedback speeds tuning).
- If detection must run fast on many images, consider reducing image resolution before processing or using OpenCV's C++ SimpleBlobDetector for speed.

Where to go next (suggested exercises)
- Draw and label each detected dot’s area or index on the overlay image (helpful when debugging).
- Add CLI flags to choose image path, thresholds, and output folder.
- Create a small folder of labeled images and write a simple evaluation script to compute precision/recall for your detector.

If you want, I can now:
- Add inline beginner-friendly comments to the detector and visualizer files.
- Create a small run script that copies sample images to `cropped/` and runs the detector + visualizer, saving outputs to an `outputs/` folder.
- Add a requirements.txt file with pinned package versions.

