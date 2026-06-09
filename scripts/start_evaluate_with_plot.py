import argparse
import os
import sys
from configparser import ConfigParser

from PyQt5 import QtWidgets

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
os.chdir(PROJECT_ROOT)

from utils.config_loader import normalize_ablation_config


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model with PyQt trajectory plot")
    parser.add_argument(
        "--eval_path",
        "-model_path",
        default=None,
        help="Training log folder. Example: logs/SimpleAvoid/2026_05_03_05_57_05_Multirotor_CNNGID_SAC",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.ini. Defaults to eval_path/config/config.ini.",
    )
    parser.add_argument(
        "--model_file",
        "--model",
        default=None,
        help="Path to model file. Defaults to best_model.pt/model_sequence_gpide.pt for GPIDE, model_sb3.zip for SB3.",
    )
    parser.add_argument(
        "--eval_eps",
        "-eval_eps",
        type=int,
        default=10,
        help="Evaluation episode number.",
    )
    return parser


def resolve_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def is_sequence_gpide_config(config_file):
    cfg = ConfigParser()
    cfg.read(config_file)
    normalize_ablation_config(cfg)
    if not cfg.has_section("options"):
        return False
    if not cfg.has_option("options", "temporal_encoder"):
        return False
    return cfg.get("options", "temporal_encoder").lower() == "gpide"


def default_model_file(eval_path, config_file):
    model_dir = os.path.join(eval_path, "models")
    if is_sequence_gpide_config(config_file):
        candidates = [
            "best_model.pt",
            "model_sequence_gpide.pt",
            "gpide_ckpt_300000.pt",
        ]
    else:
        candidates = [
            "model_sb3.zip",
            "model.zip",
        ]
    for name in candidates:
        candidate = os.path.join(model_dir, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Cannot find a default model file in {}. Pass --model_file explicitly.".format(model_dir))


def latest_simpleavoid_run():
    logs_dir = os.path.join(PROJECT_ROOT, "logs", "SimpleAvoid")
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(
            "Cannot find logs/SimpleAvoid. Pass --eval_path explicitly.")
    candidates = []
    for name in os.listdir(logs_dir):
        path = os.path.join(logs_dir, name)
        config_file = os.path.join(path, "config", "config.ini")
        if os.path.isdir(path) and os.path.exists(config_file):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No run with config/config.ini found under logs/SimpleAvoid. Pass --eval_path explicitly.")
    return max(candidates, key=os.path.getmtime)


def main():
    parser = get_parser()
    args = parser.parse_args()

    from utils.thread_evaluation import EvaluateThread
    from utils.ui_train import TrainingUi

    eval_path = resolve_path(args.eval_path)
    if eval_path is None:
        eval_path = latest_simpleavoid_run()
    config_file = resolve_path(args.config) if args.config else os.path.join(
        eval_path, "config", "config.ini")
    model_file = resolve_path(args.model_file) if args.model_file else default_model_file(
        eval_path, config_file)

    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config=config_file)
    gui.show()

    evaluate_thread = EvaluateThread(
        eval_path, config_file, model_file, args.eval_eps)
    evaluate_thread.env.action_signal.connect(gui.action_cb)
    evaluate_thread.env.state_signal.connect(gui.state_cb)
    evaluate_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    evaluate_thread.env.reward_signal.connect(gui.reward_plot_cb)
    evaluate_thread.env.pose_signal.connect(gui.traj_plot_cb)

    cfg = ConfigParser()
    cfg.read(config_file)
    if cfg.has_option("options", "perception") and cfg.get("options", "perception") == "lgmd":
        evaluate_thread.env.lgmd_signal.connect(gui.lgmd_plot_cb)

    evaluate_thread.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
