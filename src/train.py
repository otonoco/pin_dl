import os
import numpy as np
import dataset
import models
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import random


def log_images(writer, images, masks, predictions, tag, step):
    # Randomly select an image from batch
    index = random.randint(0, images.shape[0] - 1)
    img = images[index, :2].cpu()  # Only the two input channels, not the mask
    mask = masks[index].cpu()
    pred = (predictions[index] > 0.5).float().cpu()  # Threshold predictions

    # Create a grid of images
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    
    axs[0].imshow(255 * img[0].numpy(), cmap='gray')  # First input channel
    axs[0].set_title('Input 1')
    axs[0].axis('off')
    
    axs[1].imshow(255 * img[1].numpy(), cmap='gray')  # Second input channel
    axs[1].set_title('Input 2')
    axs[1].axis('off')
    
    axs[2].imshow(255 * mask[0].numpy(), cmap='gray')  # True mask
    axs[2].set_title('Mask')
    axs[2].axis('off')
    
    axs[3].imshow(255 * pred[0].numpy(), cmap='gray')  # Predicted mask
    axs[3].set_title('Prediction')
    axs[3].axis('off')

    fig.savefig(f'val_144_model_{step}.png')

    plt.close(fig)  # Close the figure to free memory


def train_model(model, train_loader, val_loader, num_epochs, device):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", current_time)
    writer = SummaryWriter(log_dir)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.2)

    best_val_loss = float('inf')

    loss_f = models.CustomLoss(rect_weight=1, center_weight=0.2, bce_weight=1, dice_weight=2)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)            
            optimizer.zero_grad()
            outputs = model(images)
            rect_loss, center_loss, bce_loss, dice_loss = loss_f(outputs, masks)
            loss = dice_loss + bce_loss + center_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        lr_scheduler.step()

        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        rect_l = 0
        cent_l = 0
        bce_l = 0
        dice_l = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                rect_loss, center_loss, bce_loss, dice_loss = loss_f(outputs, masks)
                loss = dice_loss + bce_loss + center_loss
                val_loss += loss.item()
                # rect_l += rect_loss.item()
                cent_l += center_loss.item()
                bce_l += bce_loss.item()
                dice_l += dice_loss.item()
        
        val_loss /= len(val_loader)
        rect_l /= len(val_loader)
        cent_l /= len(val_loader)
        bce_l /= len(val_loader)
        dice_l /= len(val_loader)
        
        # Log epoch-level metrics
        
        writer.add_scalar('Epoch Training Loss', train_loss, epoch)
        writer.add_scalar('Epoch Validation Loss', val_loss, epoch)
        writer.add_scalar('Epoch Validation Rect Loss', rect_l, epoch)
        writer.add_scalar('Epoch Validation Cent Loss', cent_l, epoch)
        writer.add_scalar('Epoch Validation BCE Loss', bce_l, epoch)
        writer.add_scalar('Epoch Validation Dice Loss', dice_l, epoch)

        
        # Log example predictions
        if epoch % 20 == 0:  # Log every 20 epochs
            log_images(writer, images, masks, torch.sigmoid(outputs), 'val', epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Rect Loss: {rect_l:.4f}, Center Loss: {cent_l:.4f}, BCE Loss: {bce_l:.4f}, Dice Loss: {dice_l:.4f}")
        
        if (val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model144.pth')
            print("Saved best model")
    
    writer.close()

def train_val_split(dataset, val_split=0.25):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def main():
    # Hyperparameters
    batch_size = 100
    num_epochs = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    full_dataset = dataset.TPXDataset('../data')
    train_dataset, val_dataset = train_val_split(full_dataset, val_split=0.25)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # Initialize the model
    model = models.UNet(n_channels=2, n_classes=1, kernel_size=5).to(device)

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, device)

if __name__ == '__main__':
    main()
