"""Command-line entry point for non-UI evaluation."""

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

from utils.thread_evaluation import EvaluateThread  # noqa: E402
from start_evaluate_with_plot import default_model_file, resolve_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained UAV navigation model without the PyQt UI")
    parser.add_argument("--eval_path", required=True, help="Training log folder")
    parser.add_argument("--config", default=None, help="Defaults to eval_path/config/config.ini")
    parser.add_argument("--model_file", default=None, help="Defaults based on config/model directory")
    parser.add_argument("--eval_eps", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--eval_env", default=None, help="Optional override of environment name")
    parser.add_argument("--eval_dynamics", default=None, help="Optional override of dynamics name")
    args = parser.parse_args()

    eval_path = resolve_path(args.eval_path)
    config_file = resolve_path(args.config) if args.config else os.path.join(eval_path, "config", "config.ini")
    model_file = resolve_path(args.model_file) if args.model_file else default_model_file(eval_path, config_file)
    evaluator = EvaluateThread(eval_path, config_file, model_file, args.eval_eps, args.eval_env, args.eval_dynamics)
    evaluator.run()


if __name__ == "__main__":
    main()
