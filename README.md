# 3D Object Detection with YOLOv8 and MiDaS

This project demonstrates 3D object detection using YOLOv8 for object detection and MiDaS for depth estimation in real-time using a webcam.

## Features:
- **Real-time Object Detection:** Uses the YOLOv8 model to detect objects in the video stream.
- **Depth Estimation:** Uses the MiDaS model to estimate depth from each frame and calculate the distance to the detected objects.
- **Webcam Integration:** Captures live video from your webcam to perform object detection and depth estimation on the fly.
- **Depth Map Overlay:** Displays depth information as an overlay on the video feed.

## Requirements:
1. Python 3.6+
2. Required Libraries:
   - `torch`
   - `opencv-python`
   - `numpy`
   - `ultralytics` (for YOLOv8)
   - `Pillow`
   - `torchvision`

You can install the required libraries by running:

```bash
pip install torch opencv-python numpy ultralytics Pillow torchvision
```

## Usage:

1. Clone the repository:

```bash
git clone https://github.com/your-username/3D-Object-Detection.git
cd 3D-Object-Detection
```

2. Download the pre-trained YOLOv8 model weights (`yolov8n.pt`) and place them in the project folder or specify the correct path in the `load_yolo_model()` function.

3. Run the Python script:

```bash
python main.py
```

4. The webcam feed will open, and you will see the detected objects with their distances displayed on the screen. Press `'q'` to exit the program.

## Model Details:

- **YOLOv8:** A state-of-the-art object detection model for identifying objects in the webcam stream.
- **MiDaS:** A model for monocular depth estimation that predicts depth maps from RGB images.

### Functions:
- `load_yolo_model`: Loads the pre-trained YOLOv8 model.
- `load_midas_model`: Loads the MiDaS model for depth estimation.
- `estimate_depth`: Estimates the depth map from a single frame.
- `calculate_distance`: Calculates the distance to an object based on its center point in the depth map.
- `detect_objects_and_depth`: Combines object detection and depth estimation on each frame.
- `main`: The main function that opens the webcam, performs detection and depth estimation, and displays the results.

## Notes:
- The depth calculation is an approximation and may require calibration for better accuracy.
- Make sure to have a webcam connected and accessible for the application to run successfully.
- The YOLOv8 model and MiDaS model can be adjusted for different use cases by switching model weights or tuning the depth calculation parameters.

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
