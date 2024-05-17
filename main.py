import argparse

from plot import plot
from test import test
from train import train


def check_args(valid_args: set) -> None:
    args.pop("mode", None)
    assert set(args.keys()).issubset(valid_args), f"Invalid argument: {set(args.keys()).difference(valid_args)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Classifier", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=64))
    parser.add_argument("mode", choices=("run", "train", "test", "plot"), help="Select mode: run, train, test, plot")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset (modes: run, train, test, plot)")
    parser.add_argument("--model-path", type=str, help="Path to load / save the model (modes: run, train, test)")
    parser.add_argument("--epochs", type=int, help="Number of epochs (modes: run, train)")
    parser.add_argument("--index", type=int, help="Index in the dataset to plot (modes: plot)")

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    match args["mode"]:
        case "run":
            check_args(("dataset_path", "model_path", "epochs"))
            train(**args)
            args.pop("epochs", None)
            test(**args)
        case "train":
            check_args(("dataset_path", "model_path", "epochs"))
            train(**args)
        case "test":
            check_args(("dataset_path", "model_path"))
            test(**args)
        case "plot":
            check_args(("dataset_path", "index"))
            plot(**args)
