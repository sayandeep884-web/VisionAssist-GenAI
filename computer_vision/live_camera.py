import cv2
import subprocess
import sys


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

        # Capture and analyze
        if key == ord(" "):

            image_path = "camera_frame.jpg"

            cv2.imwrite(image_path, frame)

            print("\nImage captured.")
            print("Analyzing scene...\n")

            subprocess.run(
                [sys.executable, "-m", "genai.vlm"],
                check=True
            )

            print("\nReady for next capture.")

        # Quit
        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera released.")


if __name__ == "__main__":
    start_live_camera()