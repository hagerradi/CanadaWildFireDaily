import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetSegFormer(nn.Module):
    def __init__(self, in_channels=20, out_classes=1):
        super().__init__()
        
        # For a single day, the input channels are what the dataloader sends
        self.in_channels = in_channels
        
        # Initialize the built-in SegFormer (MixVisionTransformer + UNet Decoder)
        self.model = smp.Unet(
            encoder_name="mit_b3",        # The Transformer encoder architecture
            encoder_weights="imagenet",   # Pre-trained knowledge
            in_channels=self.in_channels, 
            classes=out_classes,
        )

    def forward(self, x):
        # x arrives exactly as the simple dataloader sends it: [Batch, Channels, Height, Width]
        
        # Pass it directly through the built-in model
        out = self.model(x)
        
        return out