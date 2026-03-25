import argparse

from src.config import Config
from src.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Wildfire segmentation — entry point.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    train(config)


if __name__ == "__main__":
    main()
