import cv2
from midas_depth import estimate_depth


image_path = "test_image.jpg"

depth_map = estimate_depth(image_path)

print("\nMiDaS depth estimation successful!")

print("Depth map shape:", depth_map.shape)

print("Minimum depth value:", depth_map.min())

print("Maximum depth value:", depth_map.max())


# Normalize depth map for visualization
depth_normalized = cv2.normalize(
    depth_map,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

depth_normalized = depth_normalized.astype("uint8")


# Save depth map
cv2.imwrite(
    "midas_depth_result.jpg",
    depth_normalized
)

print("Depth map saved as: midas_depth_result.jpg")