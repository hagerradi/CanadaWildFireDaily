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
        # We use BCEWithLogitsLoss because it is numerically more stable
        # than doing Sigmoid + BCELoss separately.
        # Ensure targets are the same float type as inputs
        targets = targets.type_as(inputs)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt: the probability of the true class
        # pt = exp(-BCE) mathematically translates to the predicted probability
        pt = torch.exp(-bce_loss) 
        
        # Calculate Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss