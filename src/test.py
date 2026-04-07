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
    
    # --- 20-IMAGE VISUALIZATION (DYNAMIC) ---
    if trainer.logger is not None:
        
        print("\nHunting for 20 prime test images for Comet visualization...")
        trainer.model.eval()
        
        images_logged = 0
        max_images = 20
        min_pixels = 30 # Minimum pixels
        
        # ==========================================
        # DYNAMIC SETUP
        # Check if we are in 3-class or 2-class mode
        # ==========================================
        is_3_class = trainer.config.use_cumuarea
        vmax_val = 2 if is_3_class else 1
        
        with torch.no_grad():
            for batch in test_loader:
                if images_logged >= max_images:
                    break # Stop if we hit 20
                    

                inputs, masks = batch[0].to(trainer.device), batch[1].to(trainer.device)
                predictions = trainer.model(inputs)
                
                # --- Handle 1-channel binary logits ---
                if predictions.shape[1] == 1:
                    # Logits > 0.0 is the exact equivalent of Sigmoid > 0.5
                    pred_classes = (predictions > 0.0).squeeze(1).long()
                else:
                    pred_classes = torch.argmax(predictions, dim=1)
                
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
                        
                        # Draw the plot (vmax is now dynamic)
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        axes[0].imshow(true_np, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[0].set_title("Ground Truth")
                        axes[0].axis('off')
                        
                        axes[1].imshow(pred_np, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[1].set_title("Test Prediction (Last Checkpoint)")
                        axes[1].axis('off')
                        
                        # Save, log to Comet, and clean up
                        temp_img = f"test_vis_{images_logged}.png"
                        plt.savefig(temp_img, bbox_inches='tight')
                        plt.close(fig)
                        
                        trainer.logger.log_image(temp_img, name=f"Test_Results/Image_{images_logged}")
                        os.remove(temp_img)
                        
                        images_logged += 1
                        
        print(f"Successfully logged {images_logged} test visualizations to Comet!")
    # -------------------------------------------

    return test_loss