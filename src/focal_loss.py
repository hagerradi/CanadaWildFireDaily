import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor to balance positive/negative classes. 
                   Typically 0.25 to 0.75.
            gamma: Focusing parameter. Higher values (e.g., 2.0 or 3.0) 
                   penalize easy examples more heavily.
            reduction: 'none', 'mean', or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure targets are the same float type as inputs
        targets = targets.type_as(inputs)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt: the probability of the true class
        pt = torch.exp(-bce_loss) 
        
        # Create class-dependent alpha_t
        # If target is 1, use alpha. If target is 0, use (1 - alpha).
        alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        
        # Calculate Focal Loss using the dynamic alpha_t
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss