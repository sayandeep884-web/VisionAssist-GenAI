import cv2
import time
import pyttsx3

from computer_vision.detector import detect_objects
from computer_vision.midas_depth import estimate_depth
from computer_vision.density_checker import classify_scene
from computer_vision.navigation import make_navigation_decision
from computer_vision.alert_manager import AlertManager

from genai.vlm import (
    ask_vlm,
    build_navigation_context,
    get_safety_response,
    validate_vlm_response
)


# ---------------------------------------------------------
# TEXT TO SPEECH
# ---------------------------------------------------------

def speak(text):
    engine = pyttsx3.init()

    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)

    engine.say(text)
    engine.runAndWait()

    engine.stop()


# ---------------------------------------------------------
# ANALYZE IMAGE
# ---------------------------------------------------------

def analyze_frame(image_path, alert_manager=None):

    # =====================================================
    # 1. OBJECT DETECTION
    # =====================================================

    start = time.perf_counter()

    detections = detect_objects(image_path)

    yolo_time = time.perf_counter() - start


    # =====================================================
    # 2. DEPTH ESTIMATION
    # =====================================================

    start = time.perf_counter()

    depth_map = estimate_depth(image_path)

    midas_time = time.perf_counter() - start


    # =====================================================
    # 3. SCENE CLASSIFICATION
    # =====================================================

    start = time.perf_counter()

    scene_info = classify_scene(
        depth_map,
        detections
    )

    scene_time = time.perf_counter() - start


    # =====================================================
    # 4. NAVIGATION DECISION
    # =====================================================

    start = time.perf_counter()

    navigation = make_navigation_decision(
        detections,
        depth_map
    )

    navigation_time = time.perf_counter() - start


    # =====================================================
    # 5. GENERATE RESPONSE
    # =====================================================

    vlm_time = 0


    # -----------------------------------------------------
    # CASE 1: NO OBJECT DETECTED
    # -----------------------------------------------------

    if not detections:

        response = get_safety_response(
            navigation
        )


    # -----------------------------------------------------
    # CASE 2: OBJECT DIRECTLY AHEAD
    # -----------------------------------------------------

    elif navigation["direction"] == "stop":

        # Deterministic safety response.
        # VLM is not used for a direct obstacle ahead.

        response = get_safety_response(
            navigation
        )


    # -----------------------------------------------------
    # CASE 3: OBJECT ON LEFT / RIGHT
    # -----------------------------------------------------

    else:

        target_object = navigation[
            "closest_object"
        ]

        target_detection = navigation[
            "closest_detection"
        ]

        target_position = target_detection[
            "position"
        ]

        verified_direction = navigation[
            "direction"
        ]


        # -------------------------------------------------
        # VLM PROMPT
        # -------------------------------------------------

        prompt = f"""
You are a navigation assistant.

Write exactly ONE short sentence for a visually impaired user.

Verified information from the computer vision system:

Object: {target_object}
Position: {target_position}
Direction: {verified_direction}

Rules:
1. Mention the detected object.
2. Mention its position: left, right, or center.
3. Tell the user to move in the verified direction.
4. Use ONLY the verified direction.
5. Do not invent any object.
6. Do not change the direction.
7. Do not explain your answer.
8. Do not make a list.
9. Do not repeat these instructions.
10. Output ONLY the final navigation sentence.

Example:
"A chair is on the left. Move right carefully."

Now generate the navigation sentence.
"""


        # -------------------------------------------------
        # ASK LOCAL VLM
        # -------------------------------------------------

        start = time.perf_counter()

        vlm_response = ask_vlm(
            image_path,
            prompt
        )

        vlm_time = time.perf_counter() - start


        # -------------------------------------------------
        # PRINT RAW VLM RESPONSE
        # -------------------------------------------------

        print("\nRaw VLM Response")
        print("-----------------------")
        print(vlm_response)


        # -------------------------------------------------
        # VALIDATE VLM RESPONSE
        # -------------------------------------------------

        valid = validate_vlm_response(
            vlm_response,
            target_object,
            target_position,
            verified_direction
        )


        # -------------------------------------------------
        # VLM VALIDATION RESULT
        # -------------------------------------------------

        if valid:

            response = vlm_response

            print(
                "\nVLM validation successful."
            )

        else:

            print(
                "\nVLM validation failed."
            )

            print(
                "Using deterministic safety response."
            )

            response = get_safety_response(
                navigation
            )


    # =====================================================
    # 6. PRINT NAVIGATION INFORMATION
    # =====================================================

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


    # =====================================================
    # 7. PRINT SCENE INFORMATION
    # =====================================================

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


    # =====================================================
    # 8. PRINT YOLO INFORMATION
    # =====================================================

    print("\nVerified YOLO Information")
    print("-----------------------")

    if detections:

        for detection in detections:

            print(
                f'- {detection["object"]} '
                f'(confidence {detection["confidence"]}, '
                f'{detection["position"]})'
            )

    else:

        print("- No objects detected")


    # =====================================================
    # 9. PRINT FINAL RESPONSE
    # =====================================================

    print("\nVisionAssist Response")
    print("-----------------------")

    print(response)


    # =====================================================
    # 10. ALERT MANAGER + TTS
    # =====================================================

    if (
        alert_manager is None
        or alert_manager.should_alert(navigation)
    ):

        start = time.perf_counter()

        speak(response)

        tts_time = time.perf_counter() - start

    else:

        print(
            "\nAlert Manager: speech suppressed."
        )

        tts_time = 0


    # =====================================================
    # 11. PERFORMANCE
    # =====================================================

    total_ai_time = (
        yolo_time
        + midas_time
        + scene_time
        + navigation_time
        + vlm_time
    )

    total_time = (
        total_ai_time
        + tts_time
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

    print(
        f"TTS                  : "
        f"{tts_time:.3f} s"
    )

    print("-----------------------")

    print(
        f"Total End-to-End     : "
        f"{total_time:.3f} s"
    )

# ---------------------------------------------------------
# MAIN CAMERA LOOP
# ---------------------------------------------------------

def main():

    camera = cv2.VideoCapture(1)

    if not camera.isOpened():

        print(
            "Could not open phone camera."
        )

        return


    # -----------------------------------------------------
    # ALERT MANAGER
    # -----------------------------------------------------

    alert_manager = AlertManager(
        cooldown=3.0
    )


    print(
        "VisionAssist camera started."
    )

    print(
        "Press SPACE to capture and analyze."
    )

    print(
        "Press Q to quit."
    )


    # -----------------------------------------------------
    # CAMERA LOOP
    # -----------------------------------------------------

    while True:

        ret, frame = camera.read()


        if not ret:

            print(
                "Could not read camera frame."
            )

            break


        cv2.imshow(
            "VisionAssist Camera",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        # -------------------------------------------------
        # SPACE = CAPTURE IMAGE
        # -------------------------------------------------

        if key == 32:

            image_path = "camera_frame.jpg"


            cv2.imwrite(
                image_path,
                frame
            )


            print(
                "\nImage captured."
            )


            analyze_frame(
                image_path,
                alert_manager
            )


            print(
                "\nReady for next capture."
            )


        # -------------------------------------------------
        # Q = QUIT
        # -------------------------------------------------

        elif key == ord("q"):

            break


    # -----------------------------------------------------
    # RELEASE CAMERA
    # -----------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()


# ---------------------------------------------------------
# PROGRAM ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":

    main()