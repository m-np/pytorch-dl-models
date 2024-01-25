# torch packages
import os

import argparse

from utils.common import count_parameters
from src.model_registry import (
                                MLModel,
                                get_model
                                )


def inspect_model(model_name):
    loader = get_model(model_name)
    model = loader.model
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--MLModel", "-m", 
        choices=[el.name for el in MLModel], 
        type=str.lower,
        help='which ML Model to view', 
        default=MLModel.transformer.name)

    args = parser.parse_args()
    # Wrapping training configuration into a dictionary
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    print(config)
    print()
    inspect_model(config["MLModel"])
