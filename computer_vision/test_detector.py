from detector import detect_objects

image_path = "test_image.jpg"

detections = detect_objects(image_path)

print("\nDetected Objects:")

for detection in detections:
    print(
        f"Object: {detection['object']}, "
        f"Confidence: {detection['confidence']}, "
        f"Box: {detection['box']}"
    )