from computer_vision.detector import detect_objects
from computer_vision.midas_depth import estimate_depth
from computer_vision.navigation import get_object_depth


image_path = "camera_frame.jpg"

detections = detect_objects(image_path)
depth_map = estimate_depth(image_path)

print("\nObject Depth Information")
print("------------------------")

for detection in detections:

    depth = get_object_depth(
        depth_map,
        detection["box"]
    )

    print(
        f"Object: {detection['object']} | "
        f"Position: {detection['position']} | "
        f"Confidence: {detection['confidence']} | "
        f"MiDaS depth: {depth:.2f}"
    )