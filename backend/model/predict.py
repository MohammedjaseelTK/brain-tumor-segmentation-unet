import torch
import cv2
import numpy as np

from .unet import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "model/unet_model.pth",
        map_location=device
    )
)

model.eval()


def predict_mask(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (256,256))

    original = image.copy()

    image = image / 255.0

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    image = torch.tensor(image, dtype=torch.float32).to(device)

    with torch.no_grad():

        prediction = model(image)

    prediction = prediction.squeeze().cpu().numpy()

    prediction = (prediction > 0.3).astype(np.uint8) * 255

    return prediction