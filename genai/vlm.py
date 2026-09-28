import subprocess
import cv2
import re

from genai.tts import speak

from computer_vision.detector import detect_objects
from computer_vision.midas_depth import estimate_depth
from computer_vision.density_checker import classify_scene
from computer_vision.navigation import make_navigation_decision


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_PATH = (
    r"C:\Users\Sayandeep\AppData\Local\Programs\Ollama\ollama.exe"
)

MODEL_NAME = "moondream:latest"


# ============================================================
# OBJECT POSITION
# ============================================================

def get_object_position(box, image_width):

    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2

    if center_x < image_width / 3:

        return "left"

    elif center_x < (2 * image_width / 3):

        return "center"

    else:

        return "right"


# ============================================================
# BUILD NAVIGATION CONTEXT
# ============================================================

def build_navigation_context(image_path, detections):

    image = cv2.imread(image_path)

    if image is None:

        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    image_height, image_width = image.shape[:2]

    objects = []

    for detection in detections:

        position = get_object_position(
            detection["box"],
            image_width
        )

        objects.append(
            f'{detection["object"]} '
            f'(confidence {detection["confidence"]}, '
            f'{position})'
        )

    if not objects:

        return (
            "No objects were confidently detected "
            "by the object detector."
        )

    return "\n".join(
        f"- {obj}"
        for obj in objects
    )


# ============================================================
# ASK VLM
# ============================================================

def ask_vlm(image_path, prompt):

    command = [
        OLLAMA_PATH,
        "run",
        MODEL_NAME,
        prompt,
        image_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Ollama error:\n{result.stderr}"
        )

    return result.stdout.strip()


# ============================================================
# DETERMINISTIC SAFETY RESPONSE
# ============================================================

def get_safety_response(navigation):

    direction = navigation["direction"]

    if direction == "stop":

        return "Obstacle ahead. Stop."

    elif direction == "right":

        return (
            "Obstacle on the left. "
            "Move right carefully."
        )

    elif direction == "left":

        return (
            "Obstacle on the right. "
            "Move left carefully."
        )

    elif direction == "forward":

        return (
            "Path ahead is clear. "
            "Continue forward carefully."
        )

    return "Proceed carefully."


# ============================================================
# VALIDATE VLM RESPONSE
# ============================================================

def validate_vlm_response(
    response,
    target_object,
    target_position,
    target_direction
):

    response_lower = response.lower().strip()


    # --------------------------------------------------------
    # OBJECT CHECK
    # --------------------------------------------------------

    object_found = (
        target_object.lower()
        in response_lower
    )


    # --------------------------------------------------------
    # POSITION CHECK
    #
    # We specifically look for phrases describing
    # WHERE THE OBJECT IS.
    #
    # Example:
    # "chair is on the left"
    #
    # We do NOT count:
    # "move left"
    # --------------------------------------------------------

    if target_position == "left":

        position_pattern = (
            r"(?:on|to|at)\s+the\s+left"
            r"|(?:on|to|at)\s+left"
            r"|object\s+is\s+left"
            r"|object\s+on\s+left"
        )

        opposite_position_pattern = (
            r"(?:on|to|at)\s+the\s+right"
            r"|(?:on|to|at)\s+right"
            r"|object\s+is\s+right"
            r"|object\s+on\s+right"
        )


    elif target_position == "right":

        position_pattern = (
            r"(?:on|to|at)\s+the\s+right"
            r"|(?:on|to|at)\s+right"
            r"|object\s+is\s+right"
            r"|object\s+on\s+right"
        )

        opposite_position_pattern = (
            r"(?:on|to|at)\s+the\s+left"
            r"|(?:on|to|at)\s+left"
            r"|object\s+is\s+left"
            r"|object\s+on\s+left"
        )


    else:

        # Center is slightly different.
        position_pattern = (
            r"(?:in|at)\s+the\s+center"
            r"|(?:in|at)\s+center"
            r"|ahead"
            r"|in\s+front"
        )

        opposite_position_pattern = (
            r"on\s+the\s+left"
            r"|on\s+the\s+right"
        )


    position_found = bool(
        re.search(
            position_pattern,
            response_lower
        )
    )


    # --------------------------------------------------------
    # Check whether VLM explicitly gives the WRONG
    # object position.
    # --------------------------------------------------------

    opposite_position_found = bool(
        re.search(
            opposite_position_pattern,
            response_lower
        )
    )


    # --------------------------------------------------------
    # MOVEMENT DIRECTION
    #
    # IMPORTANT:
    #
    # "move left" means navigation direction.
    #
    # "chair is on the left" means object position.
    # --------------------------------------------------------

    if target_direction == "left":

        direction_pattern = (
            r"move\s+left"
            r"|go\s+left"
            r"|turn\s+left"
            r"|keep\s+left"
        )

    elif target_direction == "right":

        direction_pattern = (
            r"move\s+right"
            r"|go\s+right"
            r"|turn\s+right"
            r"|keep\s+right"
        )

    else:

        direction_pattern = (
            r"continue"
            r"|go\s+forward"
            r"|move\s+forward"
            r"|proceed"
        )


    direction_found = bool(
        re.search(
            direction_pattern,
            response_lower
        )
    )


    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    valid = (
        object_found
        and position_found
        and not opposite_position_found
        and direction_found
    )


    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    print("\nVLM Validation")
    print("-----------------------")

    print(
        "Object found:",
        object_found
    )

    print(
        "Correct position found:",
        position_found
    )

    print(
        "Wrong position found:",
        opposite_position_found
    )

    print(
        "Correct direction found:",
        direction_found
    )

    print(
        "Final validation:",
        valid
    )


    return valid


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    image_path = "test_image.jpg"


    print(
        "Testing VisionAssist VLM..."
    )


    detections = detect_objects(
        image_path
    )


    depth_map = estimate_depth(
        image_path
    )


    scene_info = classify_scene(
        depth_map,
        detections
    )


    navigation = make_navigation_decision(
        detections,
        depth_map
    )


    print("\nDetections:")
    print(detections)


    print("\nScene:")
    print(scene_info)


    print("\nNavigation:")
    print(navigation)


    # ========================================================
    # NO OBJECT
    # ========================================================

    if not detections:

        response = get_safety_response(
            navigation
        )

        print("\nResponse:")
        print(response)


    # ========================================================
    # STOP
    # ========================================================

    elif navigation["direction"] == "stop":

        response = get_safety_response(
            navigation
        )

        print("\nResponse:")
        print(response)


    # ========================================================
    # VLM
    # ========================================================

    else:

        target_object = navigation[
            "closest_object"
        ]


        target_detection = None


        for detection in detections:

            if detection["object"] == target_object:

                target_detection = detection

                break


        if target_detection is None:

            response = get_safety_response(
                navigation
            )


        else:

            target_position = target_detection[
                "position"
            ]


            target_direction = navigation[
                "direction"
            ]


            # =================================================
            # PROMPT
            # =================================================

            prompt = f"""
You are VisionAssist, an assistive navigation
assistant for a visually impaired person.

The computer vision system has already verified
the following information.

TARGET OBJECT:
{target_object}

TARGET POSITION:
{target_position}

VERIFIED MOVEMENT DIRECTION:
{target_direction}

VERIFIED URGENCY:
{navigation["urgency"]}

VERIFIED REASON:
{navigation["reason"]}

Generate ONE short spoken navigation sentence.

STRICT RULES:

1. Mention ONLY the target object.
2. State the object's position correctly.
3. The object's position is {target_position}.
4. The required movement direction is {target_direction}.
5. Do not confuse object position with movement direction.
6. Never say the object is on the opposite side.
7. Do not invent distance.
8. Do not mention confidence.
9. Do not mention other objects.
10. Output only one short sentence.

The expected structure is:

"The {target_object} is on the {target_position}. Move {target_direction} carefully."

Follow the verified information exactly.
"""


            print("\nTarget Object")
            print("-----------------------")

            print(
                "Object:",
                target_object
            )

            print(
                "Position:",
                target_position
            )

            print(
                "Direction:",
                target_direction
            )


            print("\nPrompt sent to VLM")
            print("-----------------------")

            print(prompt)


            # =================================================
            # CALL VLM
            # =================================================

            response = ask_vlm(
                image_path,
                prompt
            )


            # =================================================
            # VALIDATE
            # =================================================

            valid_response = validate_vlm_response(
                response,
                target_object,
                target_position,
                target_direction
            )


            if not valid_response:

                print(
                    "\nVLM validation failed."
                )

                print(
                    "Using deterministic safety response."
                )

                response = get_safety_response(
                    navigation
                )


            print("\nFinal Response:")
            print(response)


            speak(response)