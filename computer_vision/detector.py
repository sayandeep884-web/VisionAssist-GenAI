from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model
model = YOLO("models/yolo11n.pt")


def detect_objects(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image_width = image.shape[1]

    results = model(
        image_path,
        verbose=False,
        conf=0.50
    )

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]

            box_coordinates = box.xyxy[0].tolist()

            # Find object center
            x1, y1, x2, y2 = box_coordinates
            center_x = (x1 + x2) / 2

            # Determine position
            if center_x < image_width / 3:
                position = "left"

            elif center_x < (2 * image_width / 3):
                position = "center"

            else:
                position = "right"

            detections.append({
                "object": class_name,
                "confidence": round(confidence, 2),
                "box": box_coordinates,
                "position": position
            })

    return detections