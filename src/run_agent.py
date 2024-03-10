import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger

from settings import PROJECT
from model import Net


def train():
    wandb_logger = WandbLogger(project=PROJECT)
    config = wandb_logger.experiment.config
    model = Net(config)
    print(model)


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
