import numpy as np


def calculate_near_field_ratio(depth_map, near_threshold=0.35):
    """
    Estimate how much of the scene belongs to the near field.

    MiDaS produces relative depth values, not centimeters.
    The depth map is normalized before applying the threshold.
    """

    # Normalize depth map between 0 and 1
    depth_min = depth_map.min()
    depth_max = depth_map.max()

    normalized_depth = (
        (depth_map - depth_min)
        / (depth_max - depth_min)
    )

    # Pixels below this value are considered near-field
    near_pixels = normalized_depth <= near_threshold

    near_ratio = np.mean(near_pixels)

    return near_ratio


def classify_scene(depth_map, crowded_threshold=0.45):
    """
    Classify the scene as normal or crowded.

    This is a prototype heuristic.
    The threshold must later be tuned using real test scenes.
    """

    near_ratio = calculate_near_field_ratio(depth_map)

    if near_ratio >= crowded_threshold:
        scene_type = "crowded"
    else:
        scene_type = "normal"

    return {
        "scene_type": scene_type,
        "near_ratio": round(float(near_ratio), 3)
    }