import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl

from settings import PROJECT, ENTITY
import model as pl_model
from dataset import CINICDataModule
from metrics import ImagePredictionLogger
import yaml
import os
from dataclasses import dataclass


def get_config(file_name):
    with open(os.path.join("configs", f"{file_name}.yaml"), "r") as file:
        a = {k: v["value"] if "value" in v else 0 for k, v in yaml.safe_load(file)["parameters"].items()}
        a["cutmix_add"] = False
        return a


@dataclass
class Config:
    batch_norm: bool
    batch_size: int
    dropout: float
    activation: str
    num_classes: int
    input_size: list[int]
    model_name: str
    cutmix_add: str
    optimizer: str
    beta1: float
    beta2: float
    learning_rate: float
    l2_penalty: float
    model: dict
    epochs: int
    random_crop_add: bool
    random_crop_scale: float
    random_crop_ratio: float
    random_flip_add: bool
    random_flip_prob: float
    random_rotation_add: bool
    random_rotation_degrees: bool
    random_color_jitter_add: bool
    random_gaussian_blur_add: bool
    classes: list[str] | None = None


def train():
    wandb_logger = WandbLogger(project=PROJECT)
    config = wandb_logger.experiment.config

    cinic = CINICDataModule(wandb_logger, config)
    cinic.prepare_data()
    cinic.setup()
    #samples = next(iter(cinic.val_dataloader()))

    trainer = pl.Trainer(
        logger=wandb_logger,  # W&B integration
        log_every_n_steps=5,  # set the logging frequency
        max_epochs=config.epochs,  # number of epochs
        #callbacks=[ImagePredictionLogger(samples, 10)]
    )
    print(get_config("ensemble_mammals_hyperparams"))
    models = [
        pl_model.Net(Config(**get_config("vgg_deep_cnn_fin"))),
        pl_model.Net(Config(**get_config("ensemble_mammals"))),
        pl_model.Net(Config(**get_config("ensemble_vehicles_hyperparams"))),
    ]
    for model, name in zip(models, ["final_vgg_deep_cnn:v308", "final_ensemble_mammals:v8", "final_ensemble_vehicles:v5"]):
        print(name, model)
        artifact = wandb_logger.use_artifact(name)
        model_file_name = name[:name.rfind(":")] + ".pth"
        model_path = artifact.download(path_prefix=model_file_name)
        model.load_local(os.path.join(model_path, model_file_name))

    for model in models:
        for param in model.parameters():
            param.requires_grad = False
        model.to("cuda")

    model = getattr(pl_model, getattr(config, "model_class", "Net"))(config, models)
    wandb_logger.watch(model, log="all", log_freq=5)

    trainer.fit(model, cinic)

    model.load_best_model()
    model.is_validating_best_model = True
    trainer.validate(model, cinic)

    trainer.test(model, cinic)

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
        "--count",
        type=int,
        default=None,
        help="Number of experiments to perform. Can be None to run indefinitely."
    )

    # Parse the arguments
    args = parser.parse_args()

    wandb.agent(args.sweep_id, train, count=args.count, project=PROJECT,
                entity=ENTITY)
