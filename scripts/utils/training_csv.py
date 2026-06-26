import csv
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


TRAINING_CSV_COLUMNS = [
    "total_step", "episode", "episode_step", "reward", "episode_reward",
    "cost", "episode_cost", "done", "done_reason", "is_success",
    "is_crash", "is_timeout", "is_max_steps", "is_stuck",
    "max_episode_steps", "is_outside", "lagrange",
]


def ensure_training_csv_header(csv_path):
    if not csv_path or os.path.exists(csv_path):
        return
    dirname = os.path.dirname(csv_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(TRAINING_CSV_COLUMNS)


def info_bool(info, key):
    if not isinstance(info, dict):
        return 0
    if key == "is_outside":
        return int(bool(info.get("is_outside", info.get("is_not_in_workspace", False))))
    return int(bool(info.get(key, False)))


def append_training_csv_row(csv_path, step, episode, episode_step, reward, episode_reward,
                            cost, episode_cost, done, info, lagrange=""):
    if not csv_path:
        return
    ensure_training_csv_header(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            step,
            episode,
            episode_step,
            float(reward),
            float(episode_reward),
            float(cost),
            float(episode_cost),
            int(done),
            info.get("done_reason", "") if isinstance(info, dict) else "",
            info_bool(info, "is_success"),
            info_bool(info, "is_crash"),
            info_bool(info, "is_timeout"),
            info_bool(info, "is_max_steps"),
            info_bool(info, "is_stuck"),
            info.get("max_episode_steps", "") if isinstance(info, dict) else "",
            info_bool(info, "is_outside"),
            lagrange,
        ])


class AppendTrainingCsvCallback(BaseCallback):
    """Append one training CSV row on every SB3 environment step."""

    def __init__(self, csv_path, verbose=0):
        super().__init__(verbose=verbose)
        self.csv_path = csv_path
        self.episode = 1
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_cost = 0.0

    def _init_callback(self):
        ensure_training_csv_header(self.csv_path)

    @staticmethod
    def _first_value(value, default=None):
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        if isinstance(value, np.ndarray):
            return value.flat[0].item() if value.size else default
        return value

    def _on_step(self):
        if not self.csv_path:
            return True
        reward = float(self._first_value(self.locals.get("rewards"), 0.0))
        done = bool(self._first_value(self.locals.get("dones"), False))
        infos = self.locals.get("infos") or [{}]
        info = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
        cost = float(info.get("constraint_cost", 0.0)) if isinstance(info, dict) else 0.0

        self.episode_step += 1
        self.episode_reward += reward
        self.episode_cost += cost
        append_training_csv_row(
            self.csv_path, self.num_timesteps, self.episode, self.episode_step,
            reward, self.episode_reward, cost, self.episode_cost, done, info)
        if done:
            self.episode += 1
            self.episode_step = 0
            self.episode_reward = 0.0
            self.episode_cost = 0.0
        return True
