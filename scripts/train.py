"""Command-line entry point for training.

Examples:
    python scripts/train.py --config config_Trees_SimpleMultirotor_GPIDE_FOCOPS
    python scripts/train.py --config configs/config_Trees_SimpleMultirotor_GPIDE_FOCOPS.ini
"""

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
GYM_ENV_DIR = os.path.join(PROJECT_ROOT, "gym_env")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if GYM_ENV_DIR not in sys.path:
    sys.path.insert(0, GYM_ENV_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
os.chdir(PROJECT_ROOT)

from utils.thread_train import TrainingThread  # noqa: E402


def resolve_config(config_arg: str) -> str:
    if config_arg.endswith(".ini"):
        return config_arg
    if os.path.sep in config_arg or config_arg.startswith("configs"):
        return config_arg + ".ini"
    return os.path.join("configs", config_arg + ".ini")


def main():
    parser = argparse.ArgumentParser(description="Train UAV navigation model")
    parser.add_argument(
        "--config", "-c",
        default="config_Trees_SimpleMultirotor_GPIDE_FOCOPS",
        help="Config name under configs/ or a path to a .ini file.",
    )
    args = parser.parse_args()
    config_file = resolve_config(args.config)
    print("Using config:", config_file)
    trainer = TrainingThread(config_file)
    trainer.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("system exit")
