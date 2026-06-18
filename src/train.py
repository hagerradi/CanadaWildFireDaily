from __future__ import annotations

import torch
import torch.nn as nn
import time

from src.config import Config

# Dataloders
from src.dataloader import get_dataloaders
from src.dataloader_timeseries import get_timeseries_dataloaders

# Models
from src.models import unet
from src.models import unet_age
from src.models import unet_convlstm
from src.models import unet_attention
from src.models import unet_segformer
from src.models import utae

from src.trainer import Trainer
from src.utils import seed_everything
from src.focal_loss import FocalLoss 
from src.dice_loss import DiceLoss
from src.logger import CometLogger

class CombinedLoss(nn.Module):
    """Calculates a weighted combination of Focal Loss and Dice Loss for segmentation."""
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        
        # Add smooth=1.0 back to prevent gradient death on empty masks
        self.dice = DiceLoss(smooth=1.0, apply_sigmoid=True)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        """Computes the combined Focal and Dice loss.

        Args:
            inputs: The predicted logits from the model.
            targets: The ground truth target masks.

        Returns:
            The scalar tensor containing the final weighted loss.
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
            
        focal_l = self.focal(inputs, targets)
        dice_l = self.dice(inputs, targets)
        
        return (self.focal_weight * focal_l) + (self.dice_weight * dice_l)

def train(config: Config) -> None:
    """Sets up all components and executes the model training loop.

    Initializes the dataloaders, model architecture, optimizer, schedulers, 
    and loss functions based on the provided configuration. Starts the 
    training process using the Trainer class and handles optional CometML logging.

    Args:
        config (Config): The global configuration object containing all hyperparameters 
            for the model, training loop, and logging.

    Returns:
        tuple: A tuple containing:
            - trainer (Trainer): The trainer instance after the training loop completes.
            - test_loader (DataLoader): The dataloader containing the test set split.
    """
    # Fix all random seeds (CPU, CUDA, CuDNN, Python)
    seed_everything(config.seed, benchmark=config.training.cudnn_benchmark)

    # Model
    mc = config.model
    
    if mc.architecture == 'unet':
        train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                            is_sat_age=False)
        model = unet.UNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
        )
        
    elif mc.architecture == 'unet_age':
        train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                            is_sat_age=True)
        model = unet_age.UNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
        )

    elif mc.architecture == 'unet_convlstm':
        train_loader, val_loader, test_loader = get_timeseries_dataloaders(config)
        model = unet_convlstm.SpatiotemporalUNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections
        )

    elif mc.architecture == 'utae':
        train_loader, val_loader, test_loader = get_timeseries_dataloaders(
            config,
            return_positions=True,
        )
        model = utae.UTAE(
            input_dim=mc.input_channels,
            num_classes=mc.num_classes,
            encoder_widths=mc.hidden_features,
            decoder_widths=mc.utae_decoder_widths,
            out_conv_channels=mc.utae_out_conv_channels,
            agg_mode=mc.utae_agg_mode,
            encoder_norm=mc.utae_encoder_norm,
            n_head=mc.utae_n_head,
            d_model=mc.utae_d_model,
            d_k=mc.utae_d_k,
            pad_value=mc.utae_pad_value,
        )

    elif mc.architecture == 'unet_attention':
        train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                            is_sat_age=False)
        model = unet_attention.AttentionUNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
            use_attention=True
        )
    
    elif mc.architecture == 'unet_segformer':
        train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                            is_sat_age=False)
        model = unet_segformer.UNetSegFormer(
            in_channels=mc.input_channels,
            out_classes=mc.num_classes
        )
    
    else:
        raise ValueError(f"Unknown architecture specified in config: '{mc.architecture}'")

    # Optionally compile the model for faster execution (PyTorch >= 2.0, CUDA recommended)
    if config.training.use_compile:
        print("Compiling model with torch.compile (first batch will be slower)...")
        model = torch.compile(model)

    # Optimizer & loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    warmup_epochs = min(1, config.training.num_epochs - 1)
    steps_per_epoch = len(train_loader)
    total_steps = config.training.num_epochs * steps_per_epoch
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    device = Trainer._resolve_device(config.training.device)

    if config.training.use_cumuarea:
        # --- 3-CLASS SETUP ---
        print("Initializing 3-Class CrossEntropy Loss...")
        weights = torch.tensor([1.0, 10.0, 50.0], dtype=torch.float32).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        # --- 2-CLASS SETUP (Using Custom Focal + Dice) ---
        # 0 = Background + Old Fire, 1 = New Fire
        print("Initializing 2-Class Combo Loss (Focal + Dice)...")
        # Higher gamma = harder focus on difficult pixels.
        loss_fn = CombinedLoss(alpha=0.85, gamma=2.0, focal_weight=0.5, dice_weight=0.5).to(device)

    # Optional Comet logger
    logger: CometLogger | None = None
    if config.comet.enabled:
        cc = config.comet

        # Auto-generates names like: unet-baseline-1704124800
        run_name = f"{cc.experiment_name}-{int(time.time())}"

        logger = CometLogger(
            project_name=cc.project_name,
            workspace=cc.workspace,
            experiment_name=run_name,
            experiment_tags=cc.experiment_tags or None,
        )

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer, 
        scheduler=scheduler,
        loss_fn=loss_fn, 
        config=config.training, 
        logger=logger
    )

    print(f"Training on device: {trainer.device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    trainer.fit(train_loader, val_loader)

    return trainer, test_loader
