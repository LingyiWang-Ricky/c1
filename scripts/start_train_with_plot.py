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


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train navigation model with PyQt trajectory plot")
    parser.add_argument(
        "--config",
        "-config",
        default=os.path.join("configs", "config_GPIDE_Sequence_SimpleAvoid_Multirotor_2D.ini"),
        help="Path to config ini file.",
    )
    parser.add_argument(
        "--objective",
        "-objective",
        default="plot_training",
        help="Training objective note kept for compatibility.",
    )
    return parser


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def main():
    parser = get_parser()
    args = parser.parse_args()
    config_file = resolve_path(args.config)

    from utils.thread_train import TrainingThread
    from utils.ui_train import TrainingUi

    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config_file)
    gui.show()

    training_thread = TrainingThread(config_file)
    training_thread.env.action_signal.connect(gui.action_cb)
    training_thread.env.state_signal.connect(gui.state_cb)
    training_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    training_thread.env.reward_signal.connect(gui.reward_plot_cb)
    training_thread.env.pose_signal.connect(gui.traj_plot_cb)

    cfg = ConfigParser()
    cfg.read(config_file)
    if cfg.has_option("options", "perception") and cfg.get("options", "perception") == "lgmd":
        training_thread.env.lgmd_signal.connect(gui.lgmd_plot_cb)

    training_thread.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("system exit")
