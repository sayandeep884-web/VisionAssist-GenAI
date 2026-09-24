import torch
import cv2


# Use CPU because our current PyTorch installation is CPU-only
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

midas.to(device)
midas.eval()


# Load MiDaS transformation
midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = midas_transforms.small_transform


def estimate_depth(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    # OpenCV uses BGR, MiDaS expects RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare image for MiDaS
    input_batch = transform(image_rgb).to(device)

    # Run depth estimation
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    return depth_map