from torch.utils.data import Dataset
import os
import cv2
import torch, torchvision
from torchvision import transforms
import numpy as np

class RandomFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, img1, img2, mask):
        if (np.random.rand() < self.flip_prob):
            # Apply horizontal flip
            img1 = np.fliplr(img1)
            img2 = np.fliplr(img2)
            mask = np.fliplr(mask)
        
        if (np.random.rand() < self.flip_prob):
            # Apply vertical flip
            img1 = np.flipud(img1)
            img2 = np.flipud(img2)
            mask = np.flipud(mask)
        
        if (np.random.rand() < self.flip_prob):
            # Apply vertical flip
            img1 = np.rot90(img1)
            img2 = np.rot90(img2)
            mask = np.rot90(mask)

        return img1.copy(), img2.copy(), mask.copy()


class RandomShifter():
    def __init__(self, shift_prob=0.25, max_shift=10):
        self.max_shift = max_shift
        self.shift_prob = shift_prob
    
    def __call__(self, img1, img2, mask):
        if (np.random.rand() < self.shift_prob):
            # Randomly determine the amount of horizontal and vertical shift
            h_shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()  # Horizontal shift
            v_shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()  # Vertical shift

            # Apply the same transformation to both image and mask
            shifted_img1 = transforms.functional.affine(img1, angle=0, translate=(h_shift, v_shift), scale=1.0, shear=0)
            shifted_img2 = transforms.functional.affine(img2, angle=0, translate=(h_shift, v_shift), scale=1.0, shear=0)
            shifted_mask = transforms.functional.affine(mask, angle=0, translate=(h_shift, v_shift), scale=1.0, shear=0)
            
            return shifted_img1, shifted_img2, shifted_mask
        return img1, img2, mask


class ToTensor:
    def __call__(self, img1, img2, mask):
        img1 = torch.tensor(img1, dtype=torch.float).unsqueeze(0)  # Add channel dimension
        img2 = torch.tensor(img2, dtype=torch.float).unsqueeze(0)  # Add channel dimension
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)  # Add channel dimension
        return img1, img2, mask


class TPXDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.folders = [f.path for f in os.scandir(root) if f.is_dir()]
        self.fliper = RandomFlip(0.25)
        self.shifter = RandomShifter(0.25)
        self.toTensor = ToTensor()

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        files = sorted(os.listdir(os.path.join(self.root, folder)))
        img_name1, img_name2, mask_name = files[0], files[1], files[2]
        
        img_path1 = os.path.join(self.root, folder, img_name1)
        img_path2 = os.path.join(self.root, folder, img_name2)
        mask_path = os.path.join(self.root, folder, mask_name)

        image1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image1, image2, mask = self._apply_transforms(image1, image2, mask)
        # Combine the two input images and the mask
        image = torch.cat((image1, image2), dim=0)

        return image, mask
    
    def _apply_transforms(self, img1, img2, mask):
        img1, img2, mask = self.fliper(img1, img2, mask)
        img1, img2, mask = self.toTensor(img1, img2, mask)
        img1, img2, mask = self.shifter(img1, img2, mask)
        
        img1 = (img1 / 255).float()
        img2 = (img2 / 255).float()
        mask = (mask > 0).float()
        return img1, img2, mask
