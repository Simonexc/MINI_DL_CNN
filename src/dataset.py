from torch.utils.data import TensorDataset, DataLoader
import os
import torchvision.transforms.v2 as transforms
from lightning.pytorch.loggers import WandbLogger
import torch
import lightning.pytorch as pl

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
        if config.augmentation.random_crop.add:
            train_transform_list.append(
                transforms.RandomResizedCrop(
                    32,
                    config.augmentation.random_crop.scale,
                    config.augmentation.random_crop.ratio,
                )
            )

        train_transform_list.append(
            transforms.RandomHorizontalFlip(config.augmentation.random_flip)
        )
        train_transform_list.append(
            transforms.RandomVerticalFlip(config.augmentation.random_flip)
        )
        if config.augmentation.random_rotation.add:
            train_transform_list.append(
                transforms.RandomRotation(config.augmentation.random_rotation.degrees)
            )

        if config.augmentation.random_color_jitter.add:
            train_transform_list.append(
                transforms.ColorJitter(
                    config.augmentation.random_color_jitter.brightness,
                    config.augmentation.random_color_jitter.contrast,
                    config.augmentation.random_color_jitter.saturation,
                    config.augmentation.random_color_jitter.hue,
                )
            )

        if config.augmentation.gaussian_blur.add:
            train_transform_list.append(
                transforms.GaussianBlur(
                    config.augmentation.gaussian_blur.kernel_size,
                    config.augmentation.gaussian_blur.sigma,
                )
            )

        self.train_transform = transforms.Compose(train_transform_list)

        self.data_dir = ""

    def _read(self, split, is_train: bool = False):
        filename = split + ".pt"
        x, y = torch.load(os.path.join(self.data_dir, filename))
        if is_train:
            x = self.train_transform(x)

        return TensorDataset(self.transform(x), y)

    def prepare_data(self):
        # download data, train then test
        data_artifact = self.logger.use_artifact('cinic-data:latest')
        self.data_dir = data_artifact.download()

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            self.mnist_train = self._read("training", is_train=True)
            self.mnist_val = self._read("validation")
        if stage == 'test' or stage is None:
            self.mnist_test = self._read("test")

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        mnist_train = DataLoader(
            self.mnist_train, batch_size=self.config.batch_size, num_workers=4
        )
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(
            self.mnist_val, batch_size=10 * self.config.batch_size, num_workers=4
        )
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(
            self.mnist_test, batch_size=10 * self.config.batch_size, num_workers=4
        )
        return mnist_test
