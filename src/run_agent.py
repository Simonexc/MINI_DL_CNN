import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl

from settings import PROJECT
from model import Net
from dataset import CINICDataModule


def train():
    wandb_logger = WandbLogger(project=PROJECT)
    config = wandb_logger.experiment.config

    cinic = CINICDataModule(wandb_logger, config)
    cinic.prepare_data()
    cinic.setup()

    trainer = pl.Trainer(
        logger=wandb_logger,  # W&B integration
        log_every_n_steps=50,  # set the logging frequency
        max_epochs=config.epochs,  # number of epochs
    )

    model = Net(config)
    wandb_logger.watch(model, log="all", log_freq=50)

    trainer.fit(model, cinic)

    trainer.test(datamodule=cinic, ckpt_path=None)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a wandb agent to execute an experiment."
    )

    parser.add_argument(
        "sweep_id",
        type=str,
        help="Sweep ID provided by sweep.py",
    )
    parser.add_argument(
        "count",
        type=int,
        default=None,
        help="Number of experiments to perform. Can be None to run indefinitely."
    )

    # Parse the arguments
    args = parser.parse_args()

    wandb.agent(args.sweep_id, train, count=args.count, project=PROJECT)
