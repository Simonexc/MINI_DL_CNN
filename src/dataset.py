from torch.utils.data import TensorDataset, DataLoader, default_collate
import os
import torchvision.transforms.v2 as transforms
from lightning.pytorch.loggers import WandbLogger
import torch
import lightning.pytorch as pl
import torch.nn.functional as F

from settings import CINIC_STD, CINIC_MEAN


class CINICDataModule(pl.LightningDataModule):

    def __init__(self, wandb_logger: WandbLogger, config):
        super().__init__()
        self.logger = wandb_logger
        self.config = config
        self.transform = transforms.Compose([
            transforms.Normalize(CINIC_MEAN, CINIC_STD),
        ])
        train_transform_list = []

        if config.random_crop_add:
            train_transform_list.append(
                transforms.RandomResizedCrop(
                    config.input_size[1:],
                    (config.random_crop_scale, 1),
                    (1 - config.random_crop_ratio, 1 + config.random_crop_ratio),
                )
            )

        if config.random_flip_add:
            train_transform_list.append(
                transforms.RandomHorizontalFlip(config.random_flip_prob)
            )
            train_transform_list.append(
                transforms.RandomVerticalFlip(config.random_flip_prob)
            )
        if config.random_rotation_add:
            train_transform_list.append(
                transforms.RandomRotation(config.random_rotation_degrees)
            )

        if config.random_color_jitter_add:
            train_transform_list.append(
                transforms.ColorJitter(
                    config.random_color_jitter_brightness,
                    config.random_color_jitter_contrast,
                    config.random_color_jitter_saturation,
                    config.random_color_jitter_hue,
                )
            )

        if config.random_gaussian_blur_add:
            train_transform_list.append(
                transforms.GaussianBlur(
                    config.random_gaussian_blur_kernel_size,
                    config.random_gaussian_blur_sigma,
                )
            )

        train_transform_list.append(transforms.Normalize(CINIC_MEAN, CINIC_STD))
        self.train_transform = transforms.Compose(train_transform_list)

        self.data_dir = ""

    def _read(self, split, is_train: bool = False):
        filename = split + ".pt"
        x, y = torch.load(os.path.join(self.data_dir, filename))
        if is_train:
            x = self.train_transform(x)
        else:
            x = self.transform(x)

        return TensorDataset(x, y)

    def prepare_data(self):
        # download data, train then test
        data_artifact = self.logger.use_artifact('cinic-data:latest')
        self.data_dir = data_artifact.download()

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            self.cinic_train = self._read("train", is_train=True)
            self.cinic_val = self._read("valid")
        if stage == 'test' or stage is None:
            self.cinic_test = self._read("test")

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        def collate_fn(batch):
            if self.config.cutmix_add == "cut":
                return transforms.CutMix(
                    num_classes=self.config.num_classes
                )(*default_collate(batch))
            if self.config.cutmix_add == "mix":
                return transforms.MixUp(
                    num_classes=self.config.num_classes
                )(*default_collate(batch))
            return default_collate(batch)

        cinic_train = DataLoader(
            self.cinic_train,
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle=True,
            collate_fn=collate_fn,
        )
        return cinic_train

    def val_dataloader(self):
        def collate_fn(batch):
            if self.config.cutmix_add == "none":
                return default_collate(batch)
            input, output = default_collate(batch)
            return input, F.one_hot(output, num_classes=self.config.num_classes).float()

        cinic_val = DataLoader(
            self.cinic_val,
            batch_size=10 * self.config.batch_size,
            num_workers=4,
            #collate_fn=collate_fn,
        )
        return cinic_val

    def test_dataloader(self):
        def collate_fn(batch):
            if self.config.cutmix_add == "none":
                return default_collate(batch)
            input, output = default_collate(batch)
            return input, F.one_hot(output, num_classes=self.config.num_classes).float()

        cinic_test = DataLoader(
            self.cinic_test,
            batch_size=10 * self.config.batch_size,
            num_workers=4,
            #collate_fn=collate_fn,
        )
        return cinic_test
