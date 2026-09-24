import subprocess

from computer_vision.detector import detect_objects


OLLAMA_PATH = r"C:\Users\Sayandeep\AppData\Local\Programs\Ollama\ollama.exe"
MODEL_NAME = "moondream:latest"


def get_object_position(box, image_width=1280):
    """
    Estimate whether an object is on the left, center, or right.
    """

    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2

    if center_x < image_width / 3:
        return "left"

    elif center_x < (2 * image_width / 3):
        return "center"

    else:
        return "right"


def build_navigation_context(image_path):
    """
    Run YOLO and convert detections into structured information
    for the VLM.
    """

    detections = detect_objects(image_path)

    objects = []

    for detection in detections:

        position = get_object_position(
            detection["box"]
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


if __name__ == "__main__":

    image_path = "test_image2.jpg"

    # Get verified object information from YOLO
    detected_objects = build_navigation_context(
        image_path
    )

    prompt = f"""
You are VisionAssist, an assistive navigation assistant for a
visually impaired person.

The computer vision system has already detected these objects:

{detected_objects}

Convert this information into ONE natural spoken instruction.

Rules:
1. Do not repeat confidence scores.
2. Do not output technical computer-vision information.
3. Do not list the detections in brackets.
4. Do not invent objects.
5. Do not invent exact distances.
6. Mention the most important object and its position.
7. If multiple objects are present, mention them naturally.
8. Use simple language suitable for speech.
9. Give practical caution when appropriate.
10. Output ONLY one short sentence.

Example style:
"A car is ahead on the right, so proceed carefully."
"""

    response = ask_vlm(
        image_path,
        prompt
    )

    print("\nVerified YOLO Information")
    print("-----------------------")
    print(detected_objects)

    print("\nVisionAssist VLM Response")
    print("-----------------------")
    print(response)