import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import os
import wandb

from settings import PROJECT

DATA_DIR = "../data"


def load_data():
    with wandb.init(project=PROJECT, job_type="load-data") as run:
        artifact = wandb.Artifact(
            "cinic-data",
            type="dataset",
            description="Raw CINIC dataset split into train/valid/test",
        )

        for split in ["train", "valid", "test"]:
            with artifact.new_file(split + ".pt", mode="wb") as file:
                dataset = ImageFolder(
                    os.path.join(DATA_DIR, split),
                    transform=transforms.ToTensor(),
                )
                dataloader = DataLoader(
                    dataset, batch_size=len(dataset), shuffle=False, num_workers=4
                )
                images, labels = next(iter(dataloader))

                # Save the tensors to a file
                torch.save((images, labels), file)

        # Upload to W&B
        run.log_artifact(artifact)


if __name__ == "__main__":
    load_data()
