import wandb
import yaml
import argparse
import os
from settings import PROJECT, ENTITY


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a wandb sweep using a YAML configuration file."
    )

    # Add an argument for the YAML file path
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the YAML configuration file for the sweep "
             "(assume that parent directory is configs).",
    )

    # Parse the arguments
    args = parser.parse_args()

    with open(os.path.join("configs", f"{args.yaml_file}.yaml"), "r") as file:
        sweep_config = yaml.safe_load(file)

    print(wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT))
