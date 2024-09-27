import os
import torch
import models
import dataset
import numpy as np
from torchvision.transforms import v2
from PIL import Image

def load_model(model_path, device):
    """Load the trained model from the specified path."""
    model = models.UNet(n_channels=2, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path, transform):
    """Preprocess the input image for the model."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale if needed
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image_tensor, device):
    """Run inference and get the model prediction."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return output

def postprocess_output(output):
    """Convert the model output to a binary mask."""
    mask = output > 0.5  # Apply threshold
    return mask.squeeze().cpu().numpy()  # Convert to NumPy array, remove batch dimension

def save_mask(mask, output_path):
    """Save the binary mask as an image."""
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # Convert to uint8 for saving
    mask_image.save(output_path)

def run_inference(model, image_paths, output_dir, transform, device):
    """Run inference on multiple images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"mask_{image_name}")

        # Preprocess the image
        image_tensor = preprocess_image(image_path, transform)

        # Predict mask
        output = predict(model, image_tensor, device)

        # Postprocess and save the mask
        mask = postprocess_output(output)
        save_mask(mask, output_path)

        print(f"Saved mask for {image_name} at {output_path}")

def main():
    # Paths and hyperparameters
    model_path = 'best_model.pth'
    image_dir = '../data/test/images'  # Directory containing test images
    output_dir = '../data/test/predicted_masks'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transformation (resize as necessary and convert to tensor)
    transform = v2.Compose([
        v2.Resize((100, 120)),
        v2.ToTensor()
    ])

    # Load the model
    model = load_model(model_path, device)

    # List of image paths to process
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]

    # Run inference
    run_inference(model, image_paths, output_dir, transform, device)

if __name__ == '__main__':
    main()
