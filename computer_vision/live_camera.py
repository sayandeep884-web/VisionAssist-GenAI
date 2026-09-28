import cv2

from computer_vision.detector import detect_objects
from computer_vision.midas_depth import estimate_depth
from computer_vision.density_checker import classify_scene
from computer_vision.navigation import make_navigation_decision

from genai.vlm import (
    build_navigation_context,
    ask_vlm,
    validate_vlm_response
)

from genai.tts import speak


def analyze_scene(image_path):

    print("\nAnalyzing scene...\n")

    # -----------------------------------
    # 1. Estimate depth
    # -----------------------------------

    depth_map = estimate_depth(image_path)

    # -----------------------------------
    # 2. Detect objects
    # -----------------------------------

    detections = detect_objects(image_path)

    # -----------------------------------
    # 3. Get image width
    # -----------------------------------

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    image_width = image.shape[1]

    # -----------------------------------
    # 4. Classify scene
    # -----------------------------------

    scene_info = classify_scene(
        depth_map,
        detections
    )

    # -----------------------------------
    # 5. Navigation decision
    # -----------------------------------

    navigation = make_navigation_decision(
        detections,
        depth_map
    )

    print("\nNavigation Decision")
    print("-----------------------")
    print("Direction:", navigation["direction"])
    print("Urgency:", navigation["urgency"])
    print("Reason:", navigation["reason"])

    # -----------------------------------
    # 6. Scene information
    # -----------------------------------

    print("\nScene Information")
    print("-----------------------")
    print("Scene type:", scene_info["scene_type"])
    print("Near-field ratio:", scene_info["near_ratio"])

    # -----------------------------------
    # 7. Build verified information
    # -----------------------------------

    detected_objects = build_navigation_context(
        image_path,
        detections
    )

    print("\nVerified YOLO Information")
    print("-----------------------")
    print(detected_objects)

    # -----------------------------------
    # 8. Create grounded prompt
    # -----------------------------------

    if scene_info["scene_type"] == "crowded":

        prompt = f"""
You are VisionAssist, an assistive navigation assistant for a
visually impaired person.

The computer vision safety system has analyzed the scene.

Verified navigation decision:
Direction: {navigation["direction"]}
Urgency: {navigation["urgency"]}
Reason: {navigation["reason"]}

The scene is classified as CROWDED.

Give ONE short spoken navigation instruction.

Rules:
1. Follow the verified navigation decision.
2. Do not invent objects.
3. Do not invent distances.
4. Do not mention confidence scores.
5. Do not contradict the navigation decision.
6. Use simple language suitable for speech.
7. Output ONLY one short sentence.

Example:
"A person is on the left. Move right carefully."
"""

    else:

        prompt = f"""
You are VisionAssist, an assistive navigation assistant for a
visually impaired person.

The computer vision safety system has analyzed the scene.

Verified information:
{detected_objects}

Verified navigation decision:
Direction: {navigation["direction"]}
Urgency: {navigation["urgency"]}
Reason: {navigation["reason"]}

Give ONE short spoken navigation instruction.

Rules:
1. Follow the verified navigation decision.
2. Do not invent objects.
3. Do not invent distances.
4. Do not mention confidence scores.
5. Do not contradict the navigation decision.
6. Use simple language suitable for speech.
7. Output ONLY one short sentence.

Examples:
"A person is on the left. Move right carefully."
"A person is on the right. Move left carefully."
"An obstacle is ahead. Stop and proceed carefully."
"The path ahead is clear. Continue forward."
"""

    # -----------------------------------
    # 9. Generate VLM response
    # -----------------------------------

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

        # -----------------------------------
        # 10. Validate VLM response
        # -----------------------------------

        if scene_info["scene_type"] == "normal":

            if not validate_vlm_response(
                response,
                detections,
                image_width
            ):

                position = detections[0]["position"]

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

    # -----------------------------------
    # 11. Final response
    # -----------------------------------

    print("\nVisionAssist Response")
    print("-----------------------")
    print(response)

    # -----------------------------------
    # 12. Text-to-speech
    # -----------------------------------

    speak(response)

    print("\nAnalysis complete.")


def start_live_camera():

    camera = cv2.VideoCapture(1)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    print("VisionAssist camera started.")
    print("Press SPACE to analyze the current scene.")
    print("Press Q to quit.")

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Failed to read camera frame.")
            break

        cv2.imshow("VisionAssist Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        # -----------------------------------
        # Analyze current frame
        # -----------------------------------

        if key == ord(" "):

            image_path = "camera_frame.jpg"

            cv2.imwrite(
                image_path,
                frame
            )

            print("\n================================")
            print("Image captured.")
            print("================================")

            try:

                analyze_scene(
                    image_path
                )

            except Exception as e:

                print("\nError while analyzing scene:")
                print(e)

        # -----------------------------------
        # Quit
        # -----------------------------------

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera released.")


if __name__ == "__main__":
    start_live_camera()