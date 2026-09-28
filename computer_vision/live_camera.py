import cv2
import time

from computer_vision.detector import detect_objects
from computer_vision.midas_depth import estimate_depth
from computer_vision.density_checker import classify_scene
from computer_vision.navigation import make_navigation_decision

from genai.vlm import (
    ask_vlm,
    build_navigation_context,
    get_safety_response,
    validate_vlm_response
)

from genai.tts import speak


# ============================================================
# ANALYZE ONE CAPTURED FRAME
# ============================================================

def analyze_frame(image_path):

    # ========================================================
    # 1. YOLO OBJECT DETECTION
    # ========================================================

    start = time.perf_counter()

    detections = detect_objects(image_path)

    yolo_time = time.perf_counter() - start


    # ========================================================
    # 2. MiDaS DEPTH ESTIMATION
    # ========================================================

    start = time.perf_counter()

    depth_map = estimate_depth(image_path)

    midas_time = time.perf_counter() - start


    # ========================================================
    # 3. SCENE CLASSIFICATION
    # ========================================================

    start = time.perf_counter()

    scene_info = classify_scene(
        depth_map,
        detections
    )

    scene_time = time.perf_counter() - start


    # ========================================================
    # 4. NAVIGATION DECISION
    # ========================================================

    start = time.perf_counter()

    navigation = make_navigation_decision(
        detections,
        depth_map
    )

    navigation_time = time.perf_counter() - start


    # ========================================================
    # 5. READ IMAGE
    # ========================================================

    image = cv2.imread(image_path)

    if image is None:

        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    image_width = image.shape[1]


    # ========================================================
    # 6. BUILD VERIFIED YOLO CONTEXT
    # ========================================================

    detected_objects = build_navigation_context(
        image_path,
        detections
    )


    # ========================================================
    # 7. VLM PROCESSING
    # ========================================================

    vlm_time = 0


    # --------------------------------------------------------
    # CASE A: NO OBJECT DETECTED
    # --------------------------------------------------------

    if not detections:

        response = (
            "No verified obstacle detected ahead. "
            "Proceed carefully."
        )


    # --------------------------------------------------------
    # CASE B: STOP
    #
    # Safety-critical decision.
    # We don't need VLM here.
    # --------------------------------------------------------

    elif navigation["direction"] == "stop":

        response = get_safety_response(
            navigation
        )


    # --------------------------------------------------------
    # CASE C: LEFT / RIGHT / FORWARD
    # --------------------------------------------------------

    else:

        # ----------------------------------------------------
        # Find target object
        # ----------------------------------------------------

        target_object = navigation["closest_object"]
        target_detection = navigation["closest_detection"]


        # ----------------------------------------------------
        # If target cannot be found
        # ----------------------------------------------------

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
            # VLM PROMPT
            # =================================================

            prompt = f"""
You are VisionAssist, an assistive navigation
assistant for a visually impaired person.

The computer vision safety system has already
selected ONE target object.

TARGET OBJECT:
{target_object}

TARGET POSITION:
{target_position}

VERIFIED NAVIGATION DIRECTION:
{target_direction}

VERIFIED URGENCY:
{navigation["urgency"]}

VERIFIED REASON:
{navigation["reason"]}

Your task is to produce ONE short spoken
navigation instruction.

IMPORTANT RULES:

1. Talk ONLY about the TARGET OBJECT.
2. Do NOT mention any other detected object.
3. Do NOT replace the TARGET OBJECT with another object.
4. Use the TARGET POSITION exactly.
5. Follow the VERIFIED NAVIGATION DIRECTION exactly.
6. Do not invent distances.
7. Do not mention confidence scores.
8. Do not contradict the navigation decision.
9. Use simple language suitable for speech.
10. Output ONLY one short sentence.

Example:

Target object: chair
Target position: right
Verified direction: left

Correct:
"A chair is on the right. Move left carefully."

Incorrect:
"A chair is on the right. Move right."

Incorrect:
"A person is on the right."

Incorrect:
"A chair is on the left."
"""


            # =================================================
            # PRINT TARGET INFORMATION
            # =================================================

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

            start = time.perf_counter()

            response = ask_vlm(
                image_path,
                prompt
            )

            vlm_time = (
                time.perf_counter()
                - start
            )


            # =================================================
            # VALIDATE VLM RESPONSE
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


    # ========================================================
    # NAVIGATION RESULT
    # ========================================================

    print("\nNavigation Decision")
    print("-----------------------")

    print(
        "Direction:",
        navigation["direction"]
    )

    print(
        "Urgency:",
        navigation["urgency"]
    )

    print(
        "Reason:",
        navigation["reason"]
    )


    # ========================================================
    # SCENE INFORMATION
    # ========================================================

    print("\nScene Information")
    print("-----------------------")

    print(
        "Scene type:",
        scene_info["scene_type"]
    )

    print(
        "Near-field ratio:",
        scene_info["near_ratio"]
    )

    print(
        "Person count:",
        scene_info["person_count"]
    )


    # ========================================================
    # YOLO INFORMATION
    # ========================================================

    print("\nVerified YOLO Information")
    print("-----------------------")

    print(detected_objects)


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    print("\nVisionAssist Response")
    print("-----------------------")

    print(response)


    # ========================================================
    # LATENCY CALCULATION
    # ========================================================

    total_ai_time = (
        yolo_time
        + midas_time
        + scene_time
        + navigation_time
        + vlm_time
    )


    print("\nPerformance / Latency")
    print("-----------------------")

    print(
        f"YOLO                 : "
        f"{yolo_time:.3f} s"
    )

    print(
        f"MiDaS                : "
        f"{midas_time:.3f} s"
    )

    print(
        f"Scene Classification : "
        f"{scene_time:.3f} s"
    )

    print(
        f"Navigation           : "
        f"{navigation_time:.3f} s"
    )

    print(
        f"VLM                  : "
        f"{vlm_time:.3f} s"
    )

    print("-----------------------")

    print(
        f"Total AI Processing  : "
        f"{total_ai_time:.3f} s"
    )


    # ========================================================
    # TEXT TO SPEECH
    # ========================================================

    start = time.perf_counter()

    speak(response)

    tts_time = (
        time.perf_counter()
        - start
    )


    print(
        f"TTS                  : "
        f"{tts_time:.3f} s"
    )

    print("-----------------------")

    print(
        f"Total End-to-End     : "
        f"{total_ai_time + tts_time:.3f} s"
    )


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

def main():

    # ========================================================
    # CAMERA
    # ========================================================

    # 1 = DroidCam / phone camera
    camera = cv2.VideoCapture(1)


    if not camera.isOpened():

        print(
            "Could not open phone camera."
        )

        return


    print(
        "VisionAssist camera started."
    )

    print(
        "Press SPACE to capture and analyze."
    )

    print(
        "Press Q to quit."
    )


    # ========================================================
    # LIVE CAMERA LOOP
    # ========================================================

    while True:

        ret, frame = camera.read()


        if not ret:

            print(
                "Could not read camera frame."
            )

            break


        # ----------------------------------------------------
        # Show live camera
        # ----------------------------------------------------

        cv2.imshow(
            "VisionAssist Camera",
            frame
        )


        # ----------------------------------------------------
        # Read keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # Q = Quit
        # ----------------------------------------------------

        if key == ord("q"):

            break


        # ----------------------------------------------------
        # SPACE = Capture
        # ----------------------------------------------------

        if key == 32:

            print(
                "\nImage captured."
            )


            image_path = "camera_frame.jpg"


            # Save current frame
            cv2.imwrite(
                image_path,
                frame
            )


            # Analyze frame
            analyze_frame(
                image_path
            )


            print(
                "\nReady for next capture."
            )


    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()