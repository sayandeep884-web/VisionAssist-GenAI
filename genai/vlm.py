import subprocess
import cv2
from genai.tts import speak

from computer_vision.detector import detect_objects
from computer_vision.midas_depth import estimate_depth
from computer_vision.density_checker import classify_scene


OLLAMA_PATH = r"C:\Users\Sayandeep\AppData\Local\Programs\Ollama\ollama.exe"
MODEL_NAME = "moondream:latest"


def get_object_position(box, image_width):
    """
    Estimate whether an object is on the left, center, or right.
    Uses the actual image width.
    """

    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2

    if center_x < image_width / 3:
        return "left"

    elif center_x < (2 * image_width / 3):
        return "center"

    else:
        return "right"

def build_navigation_context(image_path , detections):
    """
    Run YOLO and convert detections into structured information
    for the VLM.
    """

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
        return "No objects were confidently detected by the object detector."

    return "\n".join(
        f"- {obj}"
        for obj in objects
    )


def ask_vlm(image_path, prompt):
    """
    Send image and grounded prompt to Moondream.
    """

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

def validate_vlm_response(response, detections, image_width):
    """
    Validate the VLM response against verified YOLO objects
    and their positions.
    """

    response_lower = response.lower()

    for detection in detections:

        object_name = detection["object"].lower()

        position = get_object_position(
            detection["box"],
            image_width
        )

        # Check whether the verified object is mentioned
        object_found = object_name in response_lower

        # Check whether the verified position is mentioned
        position_found = position in response_lower

        if object_found and position_found:
            return True

    return False


if __name__ == "__main__":

    image_path = "test_image2.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    image_width = image.shape[1]

    # Estimate scene depth using MiDaS
    depth_map = estimate_depth(image_path)

    # Get verified object information from YOLO
    detections = detect_objects(image_path)

    # Classify scene using both depth and YOLO
    scene_info = classify_scene(
        depth_map,
        detections
    )

    print("\nScene Information")
    print("-----------------------")
    print("Scene type:", scene_info["scene_type"])
    print("Near-field ratio:", scene_info["near_ratio"])

    # Get verified object information from YOLO
    detected_objects = build_navigation_context(
        image_path,
        detections
    )

    if scene_info["scene_type"] == "crowded":

        prompt = f"""
You are VisionAssist, an assistive navigation assistant for a
visually impaired person.

The scene has been classified as CROWDED by the computer vision system.

Verified information:
{detected_objects}

Give ONE short general safety instruction for a crowded environment.

Rules:
1. Do not mention specific objects.
2. Do not invent objects.
3. Do not invent distances.
4. Do not mention confidence scores.
5. Use simple language suitable for speech.
6. Output ONLY one short sentence.

Example:
"The area ahead is crowded, so proceed carefully."
"""

    else:

        prompt = f"""
You are VisionAssist, an assistive navigation assistant for a
visually impaired person.

Verified information:
{detected_objects}

Give ONE short navigation instruction.

Rules:
1. Do not invent objects.
2. Do not invent exact distances.
3. Do not mention confidence scores.
4. Mention the most important detected object and its position naturally.
5. For left, say "on the left".
6. For center, say "ahead in the center".
7. For right, say "on the right".
8. Use simple language suitable for speech.
9. Output ONLY one short sentence.

Example:
"A car is ahead on the right, so proceed carefully."
"""

    print("\nPrompt sent to VLM")
    print("-----------------------")
    print(prompt)

    if not detections:
        response = (
            "No verified obstacle detected ahead. "
            "Proceed carefully."
        )
    else:
        response = ask_vlm(
        image_path,
        prompt
        )

        if scene_info["scene_type"] == "normal":
            if not validate_vlm_response(
                response,
                detections,
                image_width
            ):
                position = get_object_position(
                    detections[0]["box"],
                    image_width
                )

                if position == "left":
                    position_text = "on the left"
                elif position == "center":
                    position_text = "ahead in the center"
                else:
                    position_text = "on the right"

                response = (
                    f"A {detections[0]['object']} is "
                    f"{position_text}. Proceed carefully."
                )

    print("\nVerified YOLO Information")
    print("-----------------------")
    print(detected_objects)

    print("\nVisionAssist VLM Response")
    print("-----------------------")
    print(response)

    speak(response)