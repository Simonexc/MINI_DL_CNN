import lightning.pytorch as pl
import wandb
import torch
from settings import CLASS_NAMES


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{CLASS_NAMES[pred]}, Label:{CLASS_NAMES[y]}")
                            for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "epoch": trainer.current_epoch
            })