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
        
        # Flatten the tensors so we compute the loss over the whole batch/image
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection
        intersection = (inputs * targets).sum()
        
        # Calculate Dice Coefficient: (2 * intersection) / (sum of inputs + sum of targets)
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # We want to minimize the loss, so we return 1 - dice
        return 1 - dice