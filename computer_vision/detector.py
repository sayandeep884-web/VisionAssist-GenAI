from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("models/yolo11n.pt")


def detect_objects(image_path):
    results = model(image_path, verbose=False , conf=0.50)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]

            detections.append({
                "object": class_name,
                "confidence": round(confidence, 2),
                "box": box.xyxy[0].tolist()
            })

    return detections