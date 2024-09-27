import os
import torch
import models
import dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

class ToTensor:
    def __call__(self, img1, img2):
        img1 = torch.tensor(img1, dtype=torch.float).unsqueeze(0)  # Add channel dimension
        img2 = torch.tensor(img2, dtype=torch.float).unsqueeze(0)  # Add channel dimension
        return img1, img2

def load_model(model_path, device):
    """Load the trained model from the specified path."""
    model = models.UNet(n_channels=2, n_classes=1, kernel_size=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess the input image for the model."""
    to_tensor = ToTensor()

    files = sorted(os.listdir(image_path))
    img_name1, img_name2 = files[0], files[1]
    img_path1 = os.path.join(image_path, img_name1)
    img_path2 = os.path.join(image_path, img_name2)

    image1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    image1, image2 = to_tensor(image1, image2)
    image1 = (image1 / 255).float()
    image2 = (image2 / 255).float()

    return torch.cat((image1, image2), dim=0)

def predict(model, image_tensor, device):
    """Run inference and get the model prediction."""
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return output

def postprocess_output(output):
    """Convert the model output to a binary mask."""
    mask = output > 0.5
    return mask.squeeze().cpu().numpy()

def save_mask(mask, f_path):
    """Save the binary mask as an image."""
    mask_image = (255 * mask).astype(np.uint8)
    plt.imsave(os.path.join(f_path, "predicted_mask.png"), mask_image, cmap='gray')    

def run_inference(model, f_path, device):
    """Run inference on single manufactured part."""
    image_tensor = preprocess_image(f_path)

    # Predict mask
    output = predict(model, image_tensor, device)

    # Postprocess and save the mask
    mask = postprocess_output(output)
    save_mask(mask, f_path)


def main():
    # Paths and hyperparameters
    model_path = 'best_model144.pth'
    test_dir = "../test_data"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = load_model(model_path, device)

    test_images = [f.path for f in os.scandir(test_dir) if f.is_dir()]

    for i in tqdm(range(len(test_images))):
        folder = test_images[i]
        f_path = os.path.join(test_dir, folder)
        run_inference(model, f_path, device)


if __name__ == '__main__':
    main()
