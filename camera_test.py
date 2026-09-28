import cv2

for index in [0, 1, 2]:

    camera = cv2.VideoCapture(index)

    if not camera.isOpened():
        print(f"Could not open camera {index}")
        continue

    ret, frame = camera.read()

    if ret:
        cv2.imshow(f"Camera Index {index}", frame)
        print(f"Camera {index} opened successfully.")

    camera.release()

print("Press any key to close the windows.")
cv2.waitKey(0)
cv2.destroyAllWindows()