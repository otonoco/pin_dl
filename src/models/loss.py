import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class RectangularityLoss(nn.Module):
    """
    A loss function that penalizes masks that deviate from a rectangular shape.
    """
    def __init__(self, weight=1.0):
        """
        Args:
            weight (float): Weight of the regularization term.
        """
        super(RectangularityLoss, self).__init__()
        self.weight = weight

    def bounding_box(self, mask):
        """
        Calculate the bounding box coordinates (top, bottom, left, right) of the binary mask.
        Expects mask to be of shape [height, width].
        """
        # Apply torch.any along height and width dimensions to find non-zero rows and columns
        # mask = mask.float()
        rows = torch.any(mask, dim=0).float()  # Find columns containing non-zero elements
        cols = torch.any(mask, dim=1).float()  # Find rows containing non-zero elements

        # Get the indices of the non-zero rows and columns
        top = torch.argmax(cols)  # First row containing non-zero element
        bottom = mask.size(0) - 1 - torch.argmax(torch.flip(cols, dims=[0]))  # Last row
        left = torch.argmax(rows)  # First column
        right = mask.size(1) - 1 - torch.argmax(torch.flip(rows, dims=[0]))  # Last column

        return top.item(), bottom.item(), left.item(), right.item()

    def forward(self, pred_mask):
        """
        Calculates the rectangularity loss by comparing the predicted mask with its bounding box.

        Args:
            pred_mask (tensor): Predicted binary mask of shape (batch_size, 1, height, width).

        Returns:
            loss (tensor): Rectangularity loss value.
        """
        batch_size, _, height, width = pred_mask.size()
        loss = 0.0

        for i in range(batch_size):
            # Get the binary prediction for this sample
            pred_bin = (pred_mask[i] > 0.5).float()
            # Calculate bounding box of the predicted mask
            y_min, y_max, x_min, x_max = self.bounding_box(pred_bin[0])

            # Create the rectangular bounding box as a binary mask
            bbox_mask = torch.zeros_like(pred_bin)
            bbox_mask[:, y_min:y_max+1, x_min:x_max+1] = 1.0

            # Calculate loss as the non-overlapping area between the predicted mask and its bounding box
            loss += torch.sum(torch.abs(bbox_mask - pred_bin))

        # Normalize loss over batch size
        return self.weight * loss / batch_size



class CenterDistanceLoss(nn.Module):
    def __init__(self):
        super(CenterDistanceLoss, self).__init__()

    
    def center_of_mass(self, mask):
        """Calculate the center of mass of a binary mask."""
        # Convert mask to boolean and get indices of non-zero values
        mask = mask.bool()
        if mask.sum() == 0:
            # Handle case where mask is completely empty
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.tensor([0.0, 0.0, 0.0, 0.0]).to(device)
        
        indices = torch.nonzero(mask, as_tuple=False).float()
        
        # Calculate the mean of the indices (center of mass)
        center = indices.mean(dim=0)
        return center 
    
    def forward(self, pred_mask, true_mask):
        """Calculate the Euclidean distance between the centers of the predicted and true masks."""
        # Threshold the predicted mask at 0.5
        pred_mask = (pred_mask > 0.5).float()
        
        # Calculate centers of mass for predicted and true masks
        pred_center = self.center_of_mass(pred_mask)
        true_center = self.center_of_mass(true_mask)

        # Compute Euclidean distance between the two centers
        distance = torch.norm(pred_center - true_center)
        return distance


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        Compute Dice loss between predicted and ground truth masks.
        
        Args:
            pred (Tensor): Predicted mask (batch_size, height, width).
            target (Tensor): Ground truth mask (batch_size, height, width).
        
        Returns:
            Tensor: Dice loss.
        """
        # Flatten the tensors to (batch_size, num_pixels)
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        # Compute intersection and union
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        # Dice coefficient
        dice_coeff = (2. * intersection + self.eps) / (union + self.eps)
        
        # Dice loss
        dice_loss = 1 - dice_coeff.mean()
        
        return dice_loss


class CustomLoss(nn.Module):
    def __init__(self, rect_weight=0.1, center_weight=0.1, bce_weight=0.1, dice_weight=1):
        super(CustomLoss, self).__init__()
        self.rect_weight = rect_weight
        self.center_weight = center_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce_f = nn.BCEWithLogitsLoss()
        self.dice_f = DiceLoss()
        self.center_f = CenterDistanceLoss()
        self.rect_f = RectangularityLoss()
        

    def forward(self, pred, target):
        dice = self.dice_f(torch.sigmoid(pred), target)
        # rect = self.rect_f(torch.sigmoid(pred))
        center = self.center_f(torch.sigmoid(pred), target)
        bce = self.bce_f(pred, target)

        return 0, self.center_weight * center, self.bce_weight * bce, self.dice_weight * dice
