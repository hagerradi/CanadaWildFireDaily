"""
Loads a trained model checkpoint and evaluates it on the held-out test split.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from src.trainer import Trainer


def test(trainer: Trainer, checkpoint_path: str, test_loader: DataLoader) -> float:
    """Load a checkpoint and evaluate on the test split.

    Args:
        Trainer: Trained model class.
        checkpoint_path: Path to the model checkpoint file.
        test_loader: DataLoader containing the test dataset.

    Returns:
        Average test loss.
    """
    
    epoch = trainer.load_checkpoint(checkpoint_path)

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Evaluating on device: {trainer.device}")

    test_loss, test_iou_metrics = trainer.validate(test_loader, epoch=None)
    
    print(f"Test loss: {test_loss:.4f}")

    for metric_name, value in test_iou_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    if trainer.logger is not None:
        
        print("\nGetting test images for Comet visualization...")
        trainer.model.eval()
        
        images_logged = 0
        max_images = 300
        min_pixels = 50 # Minimum pixels
        
        # ==========================================
        # DYNAMIC SETUP
        # Check if we are in 3-class or 2-class mode
        # ==========================================
        is_3_class = trainer.config.use_cumuarea
        vmax_val = 2 if is_3_class else 1
        
        with torch.no_grad():
            for batch in test_loader:
                if images_logged >= max_images:
                    break # Stop if we hit max
                    
                inputs, masks = batch[0].to(trainer.device), batch[1].to(trainer.device)
                predictions = trainer.model(inputs)

                # ==========================================
                # DYNAMIC PREVIOUS FIRE MASK EXTRACTION
                # ==========================================
                if inputs.ndim == 5:
                    # Time-Series: (Batch, Time, Channel, H, W)
                    prev_fire_masks = inputs[:, -1, -2, :, :]
                elif inputs.ndim == 4:
                    # Spatial: (Batch, Channel, H, W)
                    prev_fire_masks = inputs[:, -2, :, :]
                else:
                    raise ValueError(f"Unexpected input tensor dimensions: {inputs.shape}")

                if predictions.shape[1] == 1:
                    # Convert raw logits to probabilities
                    pred_probs = torch.sigmoid(predictions).squeeze(1)
                    # Apply the new aggressive threshold
                    pred_classes = (pred_probs > trainer.binary_classification_threshold).long()
                    prob_maps = pred_probs
                else:
                    pred_classes = torch.argmax(predictions, dim=1)
                    
                    # Convert logits to probabilities
                    pred_probs_all = torch.softmax(predictions, dim=1)
                    
                    # Extract just the "New Fire" probability layer
                    target_class_idx = 2 if is_3_class else 1
                    prob_maps = pred_probs_all[:, target_class_idx, :, :]
                
                # Check each individual image in this batch
                for i in range(masks.size(0)):
                    if images_logged >= max_images:
                        break
                        
                    true_mask = masks[i]
                    
                    # ==========================================
                    # DYNAMIC FILTERING LOGIC
                    # ==========================================
                    if is_3_class:
                        # 3-Class: Need both Old Fire (1) and New Fire (2)
                        c1_count = (true_mask == 1).sum().item()
                        c2_count = (true_mask == 2).sum().item()
                        is_interesting = (c1_count >= min_pixels) and (c2_count >= min_pixels)
                    else:
                        # 2-Class: Just need enough New Fire (1)
                        c1_count = (true_mask == 1).sum().item()
                        is_interesting = (c1_count >= min_pixels)
                    
                    # If it meets our criteria!
                    if is_interesting:
                        true_np = true_mask.cpu().numpy()
                        pred_np = pred_classes[i].cpu().numpy()
                        prob_np = prob_maps[i].cpu().numpy()
                        
                        # Get numpy array for the previous fire mask
                        prev_fire_np = prev_fire_masks[i].cpu().numpy()
                        
                        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                        
                        # 1. Previous Fire Mask
                        axes[0].imshow(prev_fire_np, cmap='gray', vmin=0, vmax=1)
                        axes[0].set_title("Previous Fire Mask")
                        axes[0].axis('off')
                        
                        # 2. Ground Truth
                        axes[1].imshow(true_np, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[1].set_title("Ground Truth")
                        axes[1].axis('off')
                        
                        # 3. Test Prediction
                        axes[2].imshow(pred_np, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[2].set_title("Test Prediction (Best Checkpoint)")
                        axes[2].axis('off')

                        # 4. Probability Map
                        im = axes[3].imshow(prob_np, cmap='magma', vmin=0.0, vmax=1.0)
                        axes[3].set_title("New Fire Probability")
                        axes[3].axis('off')
                        
                        # The legend/colorbar on the right side
                        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
                        
                        # Save, log to Comet, and clean up
                        temp_img = f"test_vis_{images_logged}.png"
                        plt.savefig(temp_img, bbox_inches='tight')
                        plt.close(fig)
                        
                        trainer.logger.log_image(temp_img, name=f"Test_Results/Image_{images_logged}")
                        os.remove(temp_img)
                        
                        images_logged += 1
                        
        print(f"Successfully logged {images_logged} test visualizations to Comet!")

    return test_loss