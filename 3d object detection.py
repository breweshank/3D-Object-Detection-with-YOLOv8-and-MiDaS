import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as T
from PIL import Image


def load_yolo_model(model_path="yolov8n.pt"):
    """Load the pre-trained YOLOv8 model."""
    model = YOLO(model_path)
    print("✅ YOLOv8 model loaded successfully!")
    return model


def load_midas_model():
    """Load MiDaS model for depth estimation."""
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    print("✅ MiDaS model loaded successfully!")

    # Define MiDaS transformation
    transform = T.Compose([
        T.Resize(384),  # Input size for MiDaS
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return midas, transform


def estimate_depth(frame, midas, transform):
    """Estimate depth from a single frame using MiDaS."""
    # Convert frame from BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL format
    img_pil = Image.fromarray(img_rgb)
    # Apply MiDaS transformation
    input_tensor = transform(img_pil).unsqueeze(0)

    # Move to GPU if available
    if torch.cuda.is_available():
        midas.cuda()
        input_tensor = input_tensor.cuda()

    # Predict depth
    with torch.no_grad():
        depth_map = midas(input_tensor)

    # Normalize and convert depth map to 8-bit for display
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # ✅ Resize depth map to match the original frame size
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    return depth_map


def calculate_distance(depth_map, x1, y1, x2, y2):
    """Calculate distance based on the depth map."""
    # Get the center point of the detected object
    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

    # Ensure the center point is within bounds
    if 0 <= center_x < depth_map.shape[1] and 0 <= center_y < depth_map.shape[0]:
        distance = depth_map[center_y, center_x]
    else:
        distance = -1  # Invalid distance if out of bounds

    # Convert depth to real-world distance (approximation, tune for accuracy)
    Z = 1000 / (distance + 1) if distance > 0 else -1
    return Z


def detect_objects_and_depth(model, midas, transform, frame):
    """Detect objects and estimate depth for each detected object."""
    # Estimate depth map from the frame
    depth_map = estimate_depth(frame, midas, transform)

    # Detect objects using YOLOv8
    results = model(frame)

    # Draw bounding boxes and calculate depth
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]}: {confidence:.2f}"

            # Estimate distance to object
            distance = calculate_distance(depth_map, x1, y1, x2, y2)
            depth_label = f"Distance: {distance:.2f}m" if distance > 0 else "Distance: N/A"

            # Draw bounding box and depth label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, depth_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Show depth map as an overlay
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    depth_colormap_resized = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]))
    combined_frame = cv2.addWeighted(frame, 0.7, depth_colormap_resized, 0.3, 0)

    return combined_frame


def main():
    """Main function for 3D object detection."""
    # Load YOLOv8 and MiDaS models
    model = load_yolo_model("yolov8n.pt")
    midas, transform = load_midas_model()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Unable to access the webcam.")
        return

    print("✅ Press 'q' to exit...")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read from webcam.")
            break

        # Detect objects and estimate depth
        detected_frame = detect_objects_and_depth(model, midas, transform, frame)

        # Display results
        cv2.imshow("3D Object Detection with Depth", detected_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
