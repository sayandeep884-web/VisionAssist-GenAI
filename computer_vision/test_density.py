from midas_depth import estimate_depth
from density_checker import classify_scene


image_path = "test_image.jpg"

# Generate MiDaS depth map
depth_map = estimate_depth(image_path)

# Analyze scene density
result = classify_scene(depth_map)

print("\nScene Analysis")
print("-----------------------")
print("Scene type:", result["scene_type"])
print("Near-field ratio:", result["near_ratio"])