import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, apply_sigmoid=True):
        """
        Args:
            smooth: A small constant to avoid division by zero.
            apply_sigmoid: Set to True if your U-Net outputs raw logits. 
                           Set to False if you already apply Sigmoid at the end of your model.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(self, inputs, targets):
        if self.apply_sigmoid:
            inputs = torch.sigmoid(inputs)
        
        # Compute Dice per image in the batch, then average
        # Flatten each image independently: (N, *) -> keep batch dim
        batch_size = inputs.size(0)
        inputs_flat = inputs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # Per-image intersection and sums
        intersection = (inputs_flat * targets_flat).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)
        
        # We want to minimize the loss, so we return 1 - dice (averaged over batch)
        return (1 - dice).mean()