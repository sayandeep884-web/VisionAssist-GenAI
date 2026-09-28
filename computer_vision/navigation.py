import numpy as np


def get_object_depth(depth_map, box):
    x1, y1, x2, y2 = map(int, box)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    region = depth_map[
        max(0, center_y - 10):center_y + 10,
        max(0, center_x - 10):center_x + 10
    ]

    if region.size == 0:
        return None

    return float(np.mean(region))


def find_closest_object(detections, depth_map):
    closest_object = None
    highest_depth = -float("inf")

    for detection in detections:
        depth = get_object_depth(
            depth_map,
            detection["box"]
        )

        if depth is not None and depth > highest_depth:
            highest_depth = depth

            # Keep the COMPLETE detection
            closest_object = detection.copy()

            # Add depth to that exact detection
            closest_object["depth"] = depth

    return closest_object


def get_direction(closest_object):
    if closest_object is None:
        return "forward"

    position = closest_object["position"]

    if position == "left":
        return "right"

    elif position == "right":
        return "left"

    else:
        return "stop"


def get_urgency(closest_object):
    if closest_object is None:
        return "low"

    depth = closest_object["depth"]

    if depth > 700:
        return "high"

    elif depth > 400:
        return "medium"

    else:
        return "low"


def make_navigation_decision(detections, depth_map):

    if not detections:
        return {
            "direction": "forward",
            "urgency": "low",
            "reason": "no obstacle detected",
            "closest_object": None
        }

    # Find the exact closest detection
    closest_object = find_closest_object(
        detections,
        depth_map
    )

    direction = get_direction(
        closest_object
    )

    urgency = get_urgency(
        closest_object
    )

    if direction == "stop":

        reason = (
            f"{closest_object['object']} "
            "detected ahead"
        )

    elif direction == "right":

        reason = (
            f"{closest_object['object']} "
            "detected on the left"
        )

    else:

        reason = (
            f"{closest_object['object']} "
            "detected on the right"
        )

    return {
        "direction": direction,
        "urgency": urgency,
        "reason": reason,

        # Object name
        "closest_object": closest_object["object"],

        # IMPORTANT:
        # Return the exact detection selected
        "closest_detection": closest_object,

        "depth": round(
            closest_object["depth"],
            2
        )
    }