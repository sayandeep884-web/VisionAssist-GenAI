import cv2


def start_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Camera started.")
    print("Press SPACE to capture.")
    print("Press Q to quit.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Failed to read frame.")
            break

        cv2.imshow("VisionAssist Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            image_path = "camera_frame.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image captured: {image_path}")

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()