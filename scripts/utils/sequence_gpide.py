"""Self-contained GPIDE + FOCOPS-inspired SAC agent for UAV navigation.

This module replaces the previous optional adapter.  It keeps the public helper
functions used by ``thread_train.py`` and ``thread_evaluation.py``:

* ``is_sequence_gpide_enabled(cfg)``
* ``build_sequence_agent(cfg, env)``
* ``load_sequence_agent(model_file, cfg, env)``

The implementation is intentionally local to this project so a config with
``temporal_encoder = gpide`` can run without installing the upstream GPIDE or
FOCOPS repositories.  The GPIDE encoder follows the idea of accumulating history
with observation differences, summation heads, exponential-smoothing heads and
optional attention heads.  The safety constraint part is an off-policy adaptation
of FOCOPS ideas for this SAC codebase: a cost critic estimates obstacle/action
cost, a Lagrange multiplier penalizes costly actions, and a delayed policy
snapshot provides a KL trust-region penalty in policy space.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def _cfg_get(cfg, section: str, option: str, default, cast=None):
    """Read an ini option with a fallback and optional type conversion."""
    try:
        if cast is bool:
            return cfg.getboolean(section, option)
        if cast is int:
            return cfg.getint(section, option)
        if cast is float:
            return cfg.getfloat(section, option)
        value = cfg.get(section, option)
        return cast(value) if cast is not None else value
    except Exception:
        return default


def _parse_float_list(value: str, default: Sequence[float]) -> List[float]:
    if value is None:
        return list(default)
    try:
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        if not value:
            return []
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except Exception:
        return list(default)


def _parse_int_list(value: str, default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    try:
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        if not value:
            return []
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except Exception:
        return list(default)


def is_sequence_gpide_enabled(cfg) -> bool:
    """Return whether a config requests the GPIDE sequence encoder."""
    if not cfg.has_section("options") or not cfg.has_option("options", "temporal_encoder"):
        return False
    return cfg.get("options", "temporal_encoder").strip().lower() == "gpide"


class ObservationVectorizer:
    """Convert this project's image/vector observations into compact vectors.

    SB3 can consume image observations directly, but the sequence agent needs a
    fixed low-dimensional vector at each time step.  For depth observations we
    use the same obstacle-summary idea as the project's ``vector`` perception:
    split the depth image into columns, take the maximum near-obstacle response
    from each split, and concatenate the normalized UAV state features stored in
    the second channel.  This keeps the temporal model light-weight and fast.
    """

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.perception_type = _cfg_get(cfg, "options", "perception", "vector")
        self.state_dim = int(getattr(env, "state_feature_length", 0))
        self.depth_splits = _cfg_get(cfg, "GPIDE", "depth_splits", 5, int)
        self.depth_splits = max(1, self.depth_splits)
        if self.perception_type in ("vector", "lgmd"):
            self.obs_dim = int(np.prod(env.observation_space.shape))
        else:
            self.obs_dim = self.depth_splits + self.state_dim

    def __call__(self, obs) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if self.perception_type in ("vector", "lgmd"):
            out = arr.reshape(-1)
            # Vector observations in this project are already in [0, 1].  Keep a
            # fallback for legacy logs that may store 0-255 values.
            if out.size > 0 and np.nanmax(np.abs(out)) > 2.0:
                out = out / 255.0
            return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        if arr.ndim != 3 or arr.shape[-1] < 2:
            out = arr.reshape(-1)
            if out.size > 0 and np.nanmax(np.abs(out)) > 2.0:
                out = out / 255.0
            return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        depth = arr[:, :, 0] / 255.0  # 0 far, 1 near in the original env.
        col_splits = np.array_split(depth, self.depth_splits, axis=1)
        depth_features = np.array([split.max() for split in col_splits], dtype=np.float32)
        state_features = arr[0, 0:self.state_dim, 1] / 255.0
        out = np.concatenate([depth_features, state_features.astype(np.float32)], axis=0)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)


class SequenceReplayBuffer:
    """Episode-aware replay buffer that samples fixed-length history windows."""

    def __init__(self, capacity: int, seq_len: int, obs_dim: int, act_dim: int, reward_history_scale: float = 100.0):
        self.capacity = int(capacity)
        self.seq_len = int(seq_len)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.reward_history_scale = max(float(reward_history_scale), EPS)
        self.episodes: List[Dict[str, np.ndarray]] = []
        self.current: Dict[str, List[np.ndarray]] = self._empty_episode_lists()
        self.size = 0

    @staticmethod
    def _empty_episode_lists() -> Dict[str, List[np.ndarray]]:
        return {
            "obs": [],
            "action": [],
            "reward": [],
            "next_obs": [],
            "done": [],
            "cost": [],
        }

    def __len__(self):
        return self.size + len(self.current["obs"])

    def add(self, obs, action, reward: float, next_obs, done: bool, cost: float = 0.0):
        self.current["obs"].append(np.asarray(obs, dtype=np.float32).reshape(self.obs_dim))
        self.current["action"].append(np.asarray(action, dtype=np.float32).reshape(self.act_dim))
        self.current["reward"].append(np.asarray([reward], dtype=np.float32))
        self.current["next_obs"].append(np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim))
        self.current["done"].append(np.asarray([float(done)], dtype=np.float32))
        self.current["cost"].append(np.asarray([cost], dtype=np.float32))
        if done:
            self.finish_episode()

    def finish_episode(self):
        if not self.current["obs"]:
            self.current = self._empty_episode_lists()
            return
        ep = {key: np.asarray(value, dtype=np.float32) for key, value in self.current.items()}
        self.episodes.append(ep)
        self.size += len(ep["obs"])
        self.current = self._empty_episode_lists()
        self._trim_to_capacity()

    def _trim_to_capacity(self):
        while self.episodes and self.size > self.capacity:
            removed = self.episodes.pop(0)
            self.size -= len(removed["obs"])

    def _candidate_episodes(self) -> List[Dict[str, np.ndarray]]:
        candidates = list(self.episodes)
        if self.current["obs"]:
            candidates.append({key: np.asarray(value, dtype=np.float32) for key, value in self.current.items()})
        return [ep for ep in candidates if len(ep["obs"]) > 0]

    def can_sample(self, batch_size: int) -> bool:
        return len(self) >= batch_size and bool(self._candidate_episodes())

    def _reward_history_feature(self, reward: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(reward, dtype=np.float32) / self.reward_history_scale, -1.0, 1.0)

    def _make_window(self, ep: Dict[str, np.ndarray], t: int):
        start = max(0, t - self.seq_len + 1)
        obs_seq = ep["obs"][start:t + 1]
        prev_actions = []
        prev_rewards = []
        for j in range(start, t + 1):
            if j == 0:
                prev_actions.append(np.zeros(self.act_dim, dtype=np.float32))
                prev_rewards.append(np.zeros(1, dtype=np.float32))
            else:
                prev_actions.append(ep["action"][j - 1])
                prev_rewards.append(self._reward_history_feature(ep["reward"][j - 1]))
        prev_actions = np.asarray(prev_actions, dtype=np.float32)
        prev_rewards = np.asarray(prev_rewards, dtype=np.float32)

        pad = self.seq_len - len(obs_seq)
        if pad > 0:
            pad_obs = np.repeat(obs_seq[[0]], pad, axis=0)
            obs_seq = np.concatenate([pad_obs, obs_seq], axis=0)
            prev_actions = np.concatenate([np.zeros((pad, self.act_dim), dtype=np.float32), prev_actions], axis=0)
            prev_rewards = np.concatenate([np.zeros((pad, 1), dtype=np.float32), prev_rewards], axis=0)

        action = ep["action"][t]
        reward = ep["reward"][t]
        next_obs = ep["next_obs"][t]
        done = ep["done"][t]
        cost = ep["cost"][t]

        next_obs_seq = np.concatenate([obs_seq[1:], next_obs.reshape(1, self.obs_dim)], axis=0)
        next_prev_actions = np.concatenate([prev_actions[1:], action.reshape(1, self.act_dim)], axis=0)
        next_prev_rewards = np.concatenate([prev_rewards[1:], self._reward_history_feature(reward).reshape(1, 1)], axis=0)
        return obs_seq, prev_actions, prev_rewards, action, reward, next_obs_seq, next_prev_actions, next_prev_rewards, done, cost

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        candidates = self._candidate_episodes()
        lengths = np.array([len(ep["obs"]) for ep in candidates], dtype=np.float64)
        probs = lengths / lengths.sum()
        batch = []
        episode_indices = np.random.choice(len(candidates), size=batch_size, p=probs)
        for ep_idx in episode_indices:
            ep = candidates[int(ep_idx)]
            t = int(np.random.randint(0, len(ep["obs"])))
            batch.append(self._make_window(ep, t))

        keys = [
            "obs_seq", "act_seq", "rew_seq", "actions", "rewards",
            "next_obs_seq", "next_act_seq", "next_rew_seq", "dones", "costs"
        ]
        arrays = {key: np.asarray([item[i] for item in batch], dtype=np.float32) for i, key in enumerate(keys)}
        return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in arrays.items()}


class GPIDEEncoder(nn.Module):
    """Generalized PID Encoder for fixed-length histories."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        out_dim: int = 64,
        seq_len: int = 16,
        n_attention_heads: int = 1,
        n_integral_heads: int = 1,
        exp_smoothing_alphas: Optional[Sequence[float]] = None,
        embed_dim_per_head: int = 16,
        decoder_hidden_size: int = 64,
        activation: str = "tanh",
    ):
        super().__init__()
        if exp_smoothing_alphas is None:
            exp_smoothing_alphas = [0.25, 0.5, 1.0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.n_attention_heads = max(0, int(n_attention_heads))
        self.n_integral_heads = max(0, int(n_integral_heads))
        self.exp_smoothing_alphas = [float(np.clip(a, 0.0, 1.0)) for a in exp_smoothing_alphas]
        if self.n_attention_heads + self.n_integral_heads + len(self.exp_smoothing_alphas) <= 0:
            self.n_integral_heads = 1
        self.embed_dim_per_head = int(embed_dim_per_head)
        self.activation = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.input_dim = obs_dim + act_dim + 1 + obs_dim

        if self.n_attention_heads > 0:
            self.attn_proj = nn.Linear(self.input_dim, 3 * self.n_attention_heads * self.embed_dim_per_head)
        else:
            self.attn_proj = None
        if self.n_integral_heads > 0:
            self.integral_proj = nn.Linear(self.input_dim, self.n_integral_heads * self.embed_dim_per_head)
        else:
            self.integral_proj = None
        if self.exp_smoothing_alphas:
            self.exp_proj = nn.Linear(self.input_dim, len(self.exp_smoothing_alphas) * self.embed_dim_per_head)
            self.register_buffer("exp_weights", self._make_exp_weights(self.exp_smoothing_alphas, seq_len))
        else:
            self.exp_proj = None
            self.register_buffer("exp_weights", torch.empty(0))

        n_heads = self.n_attention_heads + self.n_integral_heads + len(self.exp_smoothing_alphas)
        decoder_in = n_heads * self.embed_dim_per_head
        if decoder_hidden_size > 0:
            self.decoder = nn.Sequential(
                nn.LayerNorm(decoder_in),
                nn.Linear(decoder_in, decoder_hidden_size),
                self.activation,
                nn.Linear(decoder_hidden_size, out_dim),
                nn.Tanh(),
            )
        else:
            self.decoder = nn.Sequential(nn.LayerNorm(decoder_in), nn.Linear(decoder_in, out_dim), nn.Tanh())
        self.apply(self._init_weights)

    @staticmethod
    def _make_exp_weights(alphas: Sequence[float], seq_len: int) -> torch.Tensor:
        rows = []
        ages = torch.arange(seq_len - 1, -1, -1, dtype=torch.float32)  # oldest -> newest age
        for alpha in alphas:
            if alpha >= 1.0:
                w = torch.zeros(seq_len, dtype=torch.float32)
                w[-1] = 1.0
            elif alpha <= 0.0:
                w = torch.ones(seq_len, dtype=torch.float32) / float(seq_len)
            else:
                w = alpha * torch.pow(torch.tensor(1.0 - alpha), ages)
                w = w / (w.sum() + EPS)
            rows.append(w)
        return torch.stack(rows, dim=0)  # H, L

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, obs_seq: torch.Tensor, act_seq: torch.Tensor, rew_seq: torch.Tensor) -> torch.Tensor:
        # obs_seq: B,L,obs_dim; act_seq: B,L,act_dim; rew_seq: B,L,1
        prev_obs = torch.cat([obs_seq[:, :1], obs_seq[:, :-1]], dim=1)
        obs_diff = obs_seq - prev_obs
        base = torch.cat([prev_obs, act_seq, rew_seq, obs_diff], dim=-1)
        base = torch.nan_to_num(base, nan=0.0, posinf=1.0, neginf=-1.0)
        B, L, _ = base.shape
        outputs = []

        if self.attn_proj is not None:
            qkv = self.attn_proj(base).view(B, L, self.n_attention_heads, 3, self.embed_dim_per_head)
            q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
            q_last = q[:, -1].unsqueeze(2)  # B,H,1,E
            k = k.transpose(1, 2)           # B,H,L,E
            v = v.transpose(1, 2)           # B,H,L,E
            weights = torch.softmax((q_last * k).sum(-1) / math.sqrt(self.embed_dim_per_head), dim=-1)
            attn_out = (weights.unsqueeze(-2) @ v).squeeze(-2)  # B,H,E
            outputs.append(attn_out.reshape(B, -1))

        if self.integral_proj is not None:
            integ = self.integral_proj(base).view(B, L, self.n_integral_heads, self.embed_dim_per_head)
            integ = integ.sum(dim=1) / math.sqrt(float(max(L, 1)))
            outputs.append(integ.reshape(B, -1))

        if self.exp_proj is not None:
            exp_values = self.exp_proj(base).view(B, L, len(self.exp_smoothing_alphas), self.embed_dim_per_head)
            weights = self.exp_weights[:, -L:].to(base.device)  # H,L
            weights = weights / (weights.sum(dim=1, keepdim=True) + EPS)
            exp_out = torch.einsum("b l h e, h l -> b h e", exp_values, weights)
            outputs.append(exp_out.reshape(B, -1))

        z = torch.cat(outputs, dim=-1)
        return self.decoder(z)


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], activation: str = "tanh") -> nn.Sequential:
    act_cls = nn.Tanh if activation == "tanh" else nn.ReLU
    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(hidden)))
        layers.append(act_cls())
        last_dim = int(hidden)
    return nn.Sequential(*layers)


class SequenceActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg):
        super().__init__()
        activation = _cfg_get(cfg, "options", "activation_function", "tanh")
        seq_len = _cfg_get(cfg, "GPIDE", "seq_len", 16, int)
        encoder_dim = _cfg_get(cfg, "GPIDE", "encoder_dim", 64, int)
        embed_dim = _cfg_get(cfg, "GPIDE", "embed_dim_per_head", 16, int)
        n_attn = _cfg_get(cfg, "GPIDE", "attention_heads", 1, int)
        n_int = _cfg_get(cfg, "GPIDE", "integral_heads", 1, int)
        alphas = _parse_float_list(_cfg_get(cfg, "GPIDE", "exp_smoothing_alphas", "0.25,0.5,1.0"), [0.25, 0.5, 1.0])
        decoder_hidden = _cfg_get(cfg, "GPIDE", "decoder_hidden_size", 64, int)
        obs_encode_dim = _cfg_get(cfg, "GPIDE", "obs_encode_dim", 32, int)
        hidden_sizes = _parse_int_list(_cfg_get(cfg, "GPIDE", "actor_hidden_sizes", "128,128"), [128, 128])

        self.encoder = GPIDEEncoder(obs_dim, act_dim, encoder_dim, seq_len, n_attn, n_int,
                                    alphas, embed_dim, decoder_hidden, activation)
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, obs_encode_dim), nn.Tanh() if activation == "tanh" else nn.ReLU())
        self.net = build_mlp(encoder_dim + obs_encode_dim, hidden_sizes, activation)
        last_dim = hidden_sizes[-1] if hidden_sizes else encoder_dim + obs_encode_dim
        self.mean = nn.Linear(last_dim, act_dim)
        self.log_std = nn.Linear(last_dim, act_dim)
        nn.init.uniform_(self.mean.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.mean.bias, -1e-3, 1e-3)

    def distribution(self, obs_seq, act_seq, rew_seq):
        z = self.encoder(obs_seq, act_seq, rew_seq)
        obs_z = self.obs_encoder(obs_seq[:, -1])
        h = self.net(torch.cat([z, obs_z], dim=-1))
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs_seq, act_seq, rew_seq, deterministic: bool = False):
        mean, log_std = self.distribution(obs_seq, act_seq, rew_seq)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = normal.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1.0 - action.pow(2) + EPS).sum(dim=-1, keepdim=True)
        return action, log_prob, mean, log_std


class SequenceCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg):
        super().__init__()
        activation = _cfg_get(cfg, "options", "activation_function", "tanh")
        seq_len = _cfg_get(cfg, "GPIDE", "seq_len", 16, int)
        encoder_dim = _cfg_get(cfg, "GPIDE", "encoder_dim", 64, int)
        embed_dim = _cfg_get(cfg, "GPIDE", "embed_dim_per_head", 16, int)
        n_attn = _cfg_get(cfg, "GPIDE", "attention_heads", 1, int)
        n_int = _cfg_get(cfg, "GPIDE", "integral_heads", 1, int)
        alphas = _parse_float_list(_cfg_get(cfg, "GPIDE", "exp_smoothing_alphas", "0.25,0.5,1.0"), [0.25, 0.5, 1.0])
        decoder_hidden = _cfg_get(cfg, "GPIDE", "decoder_hidden_size", 64, int)
        obs_encode_dim = _cfg_get(cfg, "GPIDE", "obs_encode_dim", 32, int)
        hidden_sizes = _parse_int_list(_cfg_get(cfg, "GPIDE", "critic_hidden_sizes", "256,256"), [256, 256])

        self.encoder = GPIDEEncoder(obs_dim, act_dim, encoder_dim, seq_len, n_attn, n_int,
                                    alphas, embed_dim, decoder_hidden, activation)
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, obs_encode_dim), nn.Tanh() if activation == "tanh" else nn.ReLU())
        self.net = build_mlp(encoder_dim + obs_encode_dim + act_dim, hidden_sizes, activation)
        last_dim = hidden_sizes[-1] if hidden_sizes else encoder_dim + obs_encode_dim + act_dim
        self.value = nn.Linear(last_dim, 1)
        nn.init.uniform_(self.value.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.value.bias, -1e-3, 1e-3)

    def forward(self, obs_seq, act_seq, rew_seq, action):
        z = self.encoder(obs_seq, act_seq, rew_seq)
        obs_z = self.obs_encoder(obs_seq[:, -1])
        h = self.net(torch.cat([z, obs_z, action], dim=-1))
        return self.value(h)


@dataclass
class TrainStats:
    q1_loss: float = 0.0
    q2_loss: float = 0.0
    actor_loss: float = 0.0
    alpha_loss: float = 0.0
    alpha: float = 0.0
    cost_loss: float = 0.0
    nu: float = 0.0
    kl: float = 0.0


class SequenceGPIDESACAgent:
    """Off-policy sequence SAC with GPIDE and FOCOPS-inspired constraints."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() and _cfg_get(cfg, "GPIDE", "use_cuda", True, bool)
                                   else "cpu")
        # Small MLP/sequence updates are often faster with a modest CPU thread count;
        # this is configurable and does not affect CUDA execution.
        torch_threads = _cfg_get(cfg, "GPIDE", "torch_num_threads", 1, int)
        if self.device.type == "cpu" and torch_threads > 0:
            try:
                torch.set_num_threads(torch_threads)
            except RuntimeError:
                pass
        self.vectorizer = ObservationVectorizer(env, cfg)
        self.obs_dim = self.vectorizer.obs_dim
        self.act_dim = int(np.prod(env.action_space.shape))
        self.action_low_np = np.asarray(env.action_space.low, dtype=np.float32).reshape(self.act_dim)
        self.action_high_np = np.asarray(env.action_space.high, dtype=np.float32).reshape(self.act_dim)
        self.action_low = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=self.device)
        self.seq_len = _cfg_get(cfg, "GPIDE", "seq_len", 16, int)
        self.gamma = _cfg_get(cfg, "DRL", "gamma", 0.99, float)
        self.tau = _cfg_get(cfg, "DRL", "tau", 0.005, float)
        self.batch_size = _cfg_get(cfg, "DRL", "batch_size", 256, int)
        self.learning_starts = _cfg_get(cfg, "DRL", "learning_starts", 1000, int)
        self.train_freq = _cfg_get(cfg, "DRL", "train_freq", 1, int)
        self.gradient_steps = _cfg_get(cfg, "DRL", "gradient_steps", 1, int)
        self.reward_scale = _cfg_get(cfg, "DRL", "reward_scale", 1.0, float)
        self.max_grad_norm = _cfg_get(cfg, "GPIDE", "max_grad_norm", 10.0, float)

        lr = _cfg_get(cfg, "DRL", "learning_rate", 3e-4, float)
        self.actor = SequenceActor(self.obs_dim, self.act_dim, cfg).to(self.device)
        self.q1 = SequenceCritic(self.obs_dim, self.act_dim, cfg).to(self.device)
        self.q2 = SequenceCritic(self.obs_dim, self.act_dim, cfg).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).to(self.device)
        self.target_q2 = copy.deepcopy(self.q2).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)

        self.target_entropy = -float(self.act_dim)
        self.log_alpha = torch.tensor(math.log(_cfg_get(cfg, "DRL", "init_alpha", 0.2, float)),
                                      dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.focops_enabled = _cfg_get(cfg, "FOCOPS", "enabled", True, bool)
        self.c_gamma = _cfg_get(cfg, "FOCOPS", "cost_gamma", self.gamma, float)
        self.cost_limit = _cfg_get(cfg, "FOCOPS", "cost_limit", 0.12, float)
        self.nu = _cfg_get(cfg, "FOCOPS", "nu", 0.0, float)
        self.nu_lr = _cfg_get(cfg, "FOCOPS", "nu_lr", 0.01, float)
        self.nu_max = _cfg_get(cfg, "FOCOPS", "nu_max", 10.0, float)
        self.kl_eta = _cfg_get(cfg, "FOCOPS", "eta", 0.02, float)
        self.kl_coef = _cfg_get(cfg, "FOCOPS", "kl_coef", 1.0, float)
        self.cost_penalty_coef = _cfg_get(cfg, "FOCOPS", "cost_penalty_coef", 1.0, float)
        self.policy_snapshot_interval = _cfg_get(cfg, "FOCOPS", "policy_snapshot_interval", 25, int)
        self.update_count = 0
        self.old_actor = copy.deepcopy(self.actor).to(self.device)
        for p in self.old_actor.parameters():
            p.requires_grad_(False)

        if self.focops_enabled:
            self.c1 = SequenceCritic(self.obs_dim, self.act_dim, cfg).to(self.device)
            self.c2 = SequenceCritic(self.obs_dim, self.act_dim, cfg).to(self.device)
            self.target_c1 = copy.deepcopy(self.c1).to(self.device)
            self.target_c2 = copy.deepcopy(self.c2).to(self.device)
            self.c1_opt = torch.optim.Adam(self.c1.parameters(), lr=lr)
            self.c2_opt = torch.optim.Adam(self.c2.parameters(), lr=lr)
        else:
            self.c1 = self.c2 = self.target_c1 = self.target_c2 = None
            self.c1_opt = self.c2_opt = None

        buffer_size = _cfg_get(cfg, "DRL", "buffer_size", 50000, int)
        self.reward_history_scale = _cfg_get(cfg, "GPIDE", "reward_history_scale", 100.0, float)
        self.replay_buffer = SequenceReplayBuffer(
            buffer_size, self.seq_len, self.obs_dim, self.act_dim, self.reward_history_scale)
        self.history_obs: List[np.ndarray] = []
        self.history_act: List[np.ndarray] = []
        self.history_rew: List[np.ndarray] = []
        self.total_steps = 0
        self.episode_reward = 0.0
        self.episode_cost = 0.0
        self.episode_len = 0
        self.episode_num = 0
        self.completed_episodes = 0
        self.success_episodes = 0
        self.crash_episodes = 0
        self.timeout_episodes = 0
        self.outside_episodes = 0
        self.latest_episode_summary: Optional[Dict[str, object]] = None
        self.last_stats = TrainStats()

        self.safety_shield = _cfg_get(cfg, "safety", "enabled", False, bool)
        self.safety_threshold = _cfg_get(cfg, "safety", "front_obstacle_threshold", 0.65, float)
        self.safety_yaw_bias = _cfg_get(cfg, "safety", "yaw_rate_bias", 0.6, float)
        self.safety_speed_scale = _cfg_get(cfg, "safety", "speed_scale", 0.45, float)
        self.safety_min_forward_speed = _cfg_get(cfg, "safety", "min_forward_speed", float(self.action_low_np[0]), float)
        self.safety_goal_turn_bias = _cfg_get(cfg, "safety", "goal_turn_bias", 0.35, float)
        self.safety_vertical_goal_bias = _cfg_get(cfg, "safety", "vertical_goal_bias", 0.25, float)
        self.navigation_3d = bool(getattr(getattr(env, "dynamic_model", None), "navigation_3d",
                                          _cfg_get(cfg, "options", "navigation_3d", False, bool)))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * (np.asarray(action, dtype=np.float32) - self.action_low_np) /
                       (self.action_high_np - self.action_low_np + EPS) - 1.0, -1.0, 1.0).astype(np.float32)

    def scale_action_tensor(self, action_norm: torch.Tensor) -> torch.Tensor:
        return self.action_low + (action_norm + 1.0) * 0.5 * (self.action_high - self.action_low)

    def scale_action_np(self, action_norm: np.ndarray) -> np.ndarray:
        return (self.action_low_np + (action_norm + 1.0) * 0.5 * (self.action_high_np - self.action_low_np)).astype(np.float32)

    def reset_history(self, obs):
        obs_vec = self.vectorizer(obs)
        self.history_obs = [obs_vec]
        self.history_act = [np.zeros(self.act_dim, dtype=np.float32)]
        self.history_rew = [np.zeros(1, dtype=np.float32)]

    def observe(self, action, reward, new_obs):
        action_norm = self.normalize_action(action)
        self.history_obs.append(self.vectorizer(new_obs))
        self.history_act.append(action_norm)
        reward_feature = np.clip(float(reward) / self.reward_history_scale, -1.0, 1.0)
        self.history_rew.append(np.asarray([reward_feature], dtype=np.float32))
        self.history_obs = self.history_obs[-self.seq_len:]
        self.history_act = self.history_act[-self.seq_len:]
        self.history_rew = self.history_rew[-self.seq_len:]

    def _history_batch(self):
        obs_seq = np.asarray(self.history_obs[-self.seq_len:], dtype=np.float32)
        act_seq = np.asarray(self.history_act[-len(obs_seq):], dtype=np.float32)
        rew_seq = np.asarray(self.history_rew[-len(obs_seq):], dtype=np.float32)
        pad = self.seq_len - len(obs_seq)
        if pad > 0:
            obs_seq = np.concatenate([np.repeat(obs_seq[[0]], pad, axis=0), obs_seq], axis=0)
            act_seq = np.concatenate([np.zeros((pad, self.act_dim), dtype=np.float32), act_seq], axis=0)
            rew_seq = np.concatenate([np.zeros((pad, 1), dtype=np.float32), rew_seq], axis=0)
        return (self._to_tensor(obs_seq[None]), self._to_tensor(act_seq[None]), self._to_tensor(rew_seq[None]))

    def select_action(self, deterministic: bool = False) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            obs_seq, act_seq, rew_seq = self._history_batch()
            action_norm, _, _, _ = self.actor.sample(obs_seq, act_seq, rew_seq, deterministic=deterministic)
        self.actor.train()
        action = self.scale_action_np(action_norm.cpu().numpy()[0])
        if self.safety_shield and self.history_obs:
            action = self.apply_safety_shield(action, self.history_obs[-1])
        return action

    def apply_safety_shield(self, action: np.ndarray, obs_vec: np.ndarray) -> np.ndarray:
        """Light-weight exploration shield for depth-vector observations.

        It only modifies actions when the compact obstacle features show a close
        object in front of the UAV.  This is disabled by default unless the config
        [safety] section enables it.
        """
        if len(obs_vec) < self.vectorizer.depth_splits or self.act_dim < 2:
            return action
        depth_feats = obs_vec[:self.vectorizer.depth_splits]
        center_idx = len(depth_feats) // 2
        center = float(depth_feats[center_idx])
        if center < self.safety_threshold:
            return action
        left = float(depth_feats[:center_idx].max()) if center_idx > 0 else 0.0
        right = float(depth_feats[center_idx + 1:].max()) if center_idx + 1 < len(depth_feats) else 0.0
        # In image coordinates, obstacle on left -> turn right.  When both sides
        # are similarly occupied, bias the shield toward the goal yaw encoded in
        # the state features so avoidance does not systematically drive the UAV
        # away from the target or out of bounds.  The state layout differs between
        # 2D ([distance, yaw]) and 3D ([distance_xy, distance_z, yaw]), so using
        # depth_splits + 1 for every task accidentally read vertical error as yaw
        # in 3D and could turn the aircraft away from the goal during avoidance.
        obstacle_turn = -1.0 if left > right else 1.0
        turn_sign = obstacle_turn
        state_start = self.vectorizer.depth_splits
        vertical_feature_idx = state_start + 1 if self.navigation_3d else None
        yaw_feature_idx = state_start + (2 if self.navigation_3d else 1)
        if len(obs_vec) > yaw_feature_idx:
            relative_yaw_norm = float(np.clip(obs_vec[yaw_feature_idx], 0.0, 1.0))
            relative_yaw = (relative_yaw_norm - 0.5) * 2.0
            goal_turn = float(np.sign(relative_yaw)) if abs(relative_yaw) > 0.05 else 0.0
            obstacle_balance = abs(left - right)
            goal_weight = self.safety_goal_turn_bias * max(0.0, 1.0 - obstacle_balance)
            blended_turn = (1.0 - goal_weight) * obstacle_turn + goal_weight * goal_turn
            if abs(blended_turn) > 1e-6:
                turn_sign = float(np.sign(blended_turn))
        protected = np.asarray(action, dtype=np.float32).copy()
        # Keep moving while turning away from obstacles; otherwise the agent can
        # discover a conservative crawl/circle behavior that avoids crashes but
        # never reaches the goal before max_episode_steps.
        protected[0] = max(self.action_low_np[0], self.safety_min_forward_speed,
                           protected[0] * self.safety_speed_scale)
        if self.navigation_3d and self.act_dim >= 3 and vertical_feature_idx is not None and len(obs_vec) > vertical_feature_idx:
            vertical_error = (float(np.clip(obs_vec[vertical_feature_idx], 0.0, 1.0)) - 0.5) * 2.0
            if abs(vertical_error) > 0.05:
                # state_raw[1] is current_z - goal_z.  Positive means the UAV is
                # above the goal, so the desired vertical velocity is downward.
                goal_vz = -np.sign(vertical_error) * float(self.action_high_np[1])
                protected[1] = (1.0 - self.safety_vertical_goal_bias) * protected[1] + \
                    self.safety_vertical_goal_bias * goal_vz
                protected[1] = np.clip(protected[1], self.action_low_np[1], self.action_high_np[1])
        protected[-1] = np.clip(protected[-1] + turn_sign * self.safety_yaw_bias *
                                float(self.action_high_np[-1]), self.action_low_np[-1], self.action_high_np[-1])
        return protected.astype(np.float32)

    def _gaussian_kl(self, mean1, log_std1, mean2, log_std2):
        var1 = torch.exp(2.0 * log_std1)
        var2 = torch.exp(2.0 * log_std2)
        return (log_std2 - log_std1 + (var1 + (mean1 - mean2).pow(2)) / (2.0 * var2 + EPS) - 0.5).sum(dim=-1, keepdim=True)

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float):
        for src, tgt in zip(source.parameters(), target.parameters()):
            tgt.data.mul_(1.0 - tau)
            tgt.data.add_(tau * src.data)

    def update_parameters(self) -> TrainStats:
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        obs_seq = batch["obs_seq"]
        act_seq = batch["act_seq"]
        rew_seq = batch["rew_seq"]
        actions = batch["actions"]
        rewards = batch["rewards"] * self.reward_scale
        next_obs_seq = batch["next_obs_seq"]
        next_act_seq = batch["next_act_seq"]
        next_rew_seq = batch["next_rew_seq"]
        dones = batch["dones"]
        costs = batch["costs"]

        with torch.no_grad():
            next_action, next_logp, _, _ = self.actor.sample(next_obs_seq, next_act_seq, next_rew_seq)
            target_q = torch.min(
                self.target_q1(next_obs_seq, next_act_seq, next_rew_seq, next_action),
                self.target_q2(next_obs_seq, next_act_seq, next_rew_seq, next_action),
            ) - self.alpha.detach() * next_logp
            q_target = rewards + (1.0 - dones) * self.gamma * target_q

        q1_pred = self.q1(obs_seq, act_seq, rew_seq, actions)
        q2_pred = self.q2(obs_seq, act_seq, rew_seq, actions)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
        self.q2_opt.step()

        cost_loss = torch.tensor(0.0, device=self.device)
        if self.focops_enabled:
            with torch.no_grad():
                next_cost_action, _, _, _ = self.actor.sample(next_obs_seq, next_act_seq, next_rew_seq)
                target_c = torch.min(
                    self.target_c1(next_obs_seq, next_act_seq, next_rew_seq, next_cost_action),
                    self.target_c2(next_obs_seq, next_act_seq, next_rew_seq, next_cost_action),
                )
                c_target = costs + (1.0 - dones) * self.c_gamma * target_c
            c1_pred = self.c1(obs_seq, act_seq, rew_seq, actions)
            c2_pred = self.c2(obs_seq, act_seq, rew_seq, actions)
            c1_loss = F.mse_loss(c1_pred, c_target)
            c2_loss = F.mse_loss(c2_pred, c_target)
            self.c1_opt.zero_grad()
            c1_loss.backward()
            nn.utils.clip_grad_norm_(self.c1.parameters(), self.max_grad_norm)
            self.c1_opt.step()
            self.c2_opt.zero_grad()
            c2_loss.backward()
            nn.utils.clip_grad_norm_(self.c2.parameters(), self.max_grad_norm)
            self.c2_opt.step()
            cost_loss = c1_loss + c2_loss
            self.nu = float(np.clip(self.nu + self.nu_lr * (float(costs.mean().item()) - self.cost_limit),
                                    0.0, self.nu_max))

        new_action, logp, mean, log_std = self.actor.sample(obs_seq, act_seq, rew_seq)
        q_new = torch.min(self.q1(obs_seq, act_seq, rew_seq, new_action),
                          self.q2(obs_seq, act_seq, rew_seq, new_action))
        actor_loss = (self.alpha.detach() * logp - q_new).mean()
        kl_mean = torch.tensor(0.0, device=self.device)
        if self.focops_enabled:
            c_new = torch.min(self.c1(obs_seq, act_seq, rew_seq, new_action),
                              self.c2(obs_seq, act_seq, rew_seq, new_action))
            # Keep a baseline cost penalty in addition to the adaptive Lagrange
            # multiplier.  Otherwise early training starts with nu=0 and the
            # actor can reinforce fast-but-unsafe trajectories before the
            # multiplier grows enough to matter.
            actor_loss = actor_loss + (self.nu + self.cost_penalty_coef) * c_new.mean()
            with torch.no_grad():
                old_mean, old_log_std = self.old_actor.distribution(obs_seq, act_seq, rew_seq)
            kl = self._gaussian_kl(mean, log_std, old_mean, old_log_std)
            kl_mean = kl.mean()
            # A first-order trust-region style penalty: no penalty below eta,
            # increasing pressure when the new policy moves too far in policy space.
            actor_loss = actor_loss + self.kl_coef * F.relu(kl - self.kl_eta).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.target_q1, self.tau)
        self._soft_update(self.q2, self.target_q2, self.tau)
        if self.focops_enabled:
            self._soft_update(self.c1, self.target_c1, self.tau)
            self._soft_update(self.c2, self.target_c2, self.tau)

        self.update_count += 1
        if self.focops_enabled and self.update_count % max(1, self.policy_snapshot_interval) == 0:
            self.old_actor.load_state_dict(self.actor.state_dict())

        stats = TrainStats(
            q1_loss=float(q1_loss.detach().cpu()),
            q2_loss=float(q2_loss.detach().cpu()),
            actor_loss=float(actor_loss.detach().cpu()),
            alpha_loss=float(alpha_loss.detach().cpu()),
            alpha=float(self.alpha.detach().cpu()),
            cost_loss=float(cost_loss.detach().cpu()),
            nu=float(self.nu),
            kl=float(kl_mean.detach().cpu()),
        )
        self.last_stats = stats
        return stats

    @staticmethod
    def _info_bool(info: Optional[Dict], key: str) -> bool:
        return bool(info.get(key, False)) if isinstance(info, dict) else False

    @staticmethod
    def _info_text(info: Optional[Dict], key: str) -> str:
        value = info.get(key, "") if isinstance(info, dict) else ""
        return "" if value is None else str(value)

    def _episode_rates(self) -> Dict[str, float]:
        denom = max(1, self.completed_episodes)
        return {
            "success_rate": self.success_episodes / denom,
            "crash_rate": self.crash_episodes / denom,
            "timeout_rate": self.timeout_episodes / denom,
            "outside_rate": self.outside_episodes / denom,
        }

    def _episode_summary(self, info: Optional[Dict]) -> Dict[str, object]:
        reason = self._info_text(info, "done_reason")
        is_success = reason == "success" or self._info_bool(info, "is_success")
        is_crash = reason == "crash" or self._info_bool(info, "is_crash")
        is_outside = reason == "outside" or self._info_bool(info, "is_not_in_workspace")
        is_timeout = reason in ("timeout", "max_steps") or (
            (self._info_bool(info, "is_timeout") or self._info_bool(info, "is_max_steps"))
            and not is_success and not is_crash and not is_outside
        )

        # Keep the rate buckets mutually exclusive using the environment's done reason
        # priority: crash/outside/success/max_steps.  Here timeout is a backward-
        # compatible metric name for the case where the episode reaches the configured max step count without reaching
        # the target or ending earlier for another terminal reason.
        if is_crash:
            reason = "crash"
            is_success = is_timeout = is_outside = False
        elif is_outside:
            reason = "outside"
            is_success = is_crash = is_timeout = False
        elif is_success:
            reason = "success"
            is_crash = is_timeout = is_outside = False
        elif is_timeout:
            reason = "max_steps"
            is_success = is_crash = is_outside = False

        self.completed_episodes += 1
        self.success_episodes += int(is_success)
        self.crash_episodes += int(is_crash)
        self.timeout_episodes += int(is_timeout)
        self.outside_episodes += int(is_outside)
        rates = self._episode_rates()
        return {
            "episode": self.episode_num,
            "total_step": self.total_steps,
            "episode_len": self.episode_len,
            "episode_reward": self.episode_reward,
            "episode_cost": self.episode_cost,
            "done_reason": reason,
            "is_success": int(is_success),
            "is_crash": int(is_crash),
            "is_timeout": int(is_timeout),
            "is_max_steps": int(is_timeout),
            "max_episode_steps": info.get("max_episode_steps", "") if isinstance(info, dict) else "",
            "is_outside": int(is_outside),
            **rates,
        }

    @staticmethod
    def _write_header(csv_path: str, columns: Sequence[str]):
        if not csv_path or os.path.exists(csv_path):
            return
        dirname = os.path.dirname(csv_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(columns)

    def _write_csv_header(self, csv_path: str):
        self._write_header(csv_path, [
            "total_step", "episode", "episode_step", "reward", "episode_reward", "cost",
            "episode_cost", "done", "done_reason", "is_success", "is_crash", "is_timeout",
            "is_max_steps", "max_episode_steps", "is_outside", "success_rate",
            "crash_rate", "timeout_rate", "outside_rate", "distance_progress",
            "progress_penalty", "reverse_progress_penalty", "low_speed_penalty",
            "heading_alignment", "heading_reward", "heading_error_penalty",
            "boundary_margin", "boundary_cost", "boundary_penalty", "path_distance", "path_penalty",
            "boundary_shield_active", "boundary_outward_score", "boundary_shield_yaw_error_deg",
            "q1_loss", "q2_loss", "critic_loss", "actor_loss", "alpha_loss", "cost_loss",
            "alpha", "nu", "kl"
        ])

    def _write_episode_csv_header(self, csv_path: str):
        self._write_header(csv_path, [
            "episode", "total_step", "episode_len", "episode_reward", "episode_cost",
            "done_reason", "is_success", "is_crash", "is_timeout", "is_max_steps",
            "max_episode_steps", "is_outside",
            "success_rate", "crash_rate", "timeout_rate", "outside_rate",
            "q1_loss", "q2_loss", "critic_loss", "actor_loss", "alpha_loss", "cost_loss",
            "alpha", "nu", "kl"
        ])

    def _append_csv(self, csv_path: str, reward: float, cost: float, done: bool, info: Optional[Dict] = None):
        if not csv_path:
            return
        self._write_csv_header(csv_path)
        rates = self._episode_rates()
        q_loss = self.last_stats.q1_loss + self.last_stats.q2_loss
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.total_steps, self.episode_num, self.episode_len, reward, self.episode_reward,
                cost, self.episode_cost, int(done), self._info_text(info, "done_reason"),
                int(self._info_bool(info, "is_success")), int(self._info_bool(info, "is_crash")),
                int(self._info_bool(info, "is_timeout")), int(self._info_bool(info, "is_max_steps")),
                info.get("max_episode_steps", "") if isinstance(info, dict) else "",
                int(self._info_bool(info, "is_not_in_workspace")),
                rates["success_rate"], rates["crash_rate"], rates["timeout_rate"], rates["outside_rate"],
                info.get("distance_progress", "") if isinstance(info, dict) else "",
                info.get("progress_penalty", "") if isinstance(info, dict) else "",
                info.get("reverse_progress_penalty", "") if isinstance(info, dict) else "",
                info.get("low_speed_penalty", "") if isinstance(info, dict) else "",
                info.get("heading_alignment", "") if isinstance(info, dict) else "",
                info.get("heading_reward", "") if isinstance(info, dict) else "",
                info.get("heading_error_penalty", "") if isinstance(info, dict) else "",
                info.get("boundary_margin", "") if isinstance(info, dict) else "",
                info.get("boundary_cost", "") if isinstance(info, dict) else "",
                info.get("boundary_penalty", "") if isinstance(info, dict) else "",
                info.get("path_distance", "") if isinstance(info, dict) else "",
                info.get("path_penalty", "") if isinstance(info, dict) else "",
                info.get("boundary_shield_active", "") if isinstance(info, dict) else "",
                info.get("boundary_outward_score", "") if isinstance(info, dict) else "",
                info.get("boundary_shield_yaw_error_deg", "") if isinstance(info, dict) else "",
                self.last_stats.q1_loss, self.last_stats.q2_loss, q_loss, self.last_stats.actor_loss,
                self.last_stats.alpha_loss, self.last_stats.cost_loss, self.last_stats.alpha,
                self.last_stats.nu, self.last_stats.kl,
            ])

    def _append_episode_csv(self, csv_path: str, summary: Dict[str, object]):
        if not csv_path:
            return
        self._write_episode_csv_header(csv_path)
        q_loss = self.last_stats.q1_loss + self.last_stats.q2_loss
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                summary["episode"], summary["total_step"], summary["episode_len"],
                summary["episode_reward"], summary["episode_cost"], summary["done_reason"],
                summary["is_success"], summary["is_crash"], summary["is_timeout"], summary["is_max_steps"],
                summary["max_episode_steps"], summary["is_outside"],
                summary["success_rate"], summary["crash_rate"], summary["timeout_rate"], summary["outside_rate"],
                self.last_stats.q1_loss, self.last_stats.q2_loss, q_loss, self.last_stats.actor_loss,
                self.last_stats.alpha_loss, self.last_stats.cost_loss, self.last_stats.alpha,
                self.last_stats.nu, self.last_stats.kl,
            ])

    def _write_metrics_json(self, json_path: str, latest_episode: Optional[Dict[str, object]] = None):
        if not json_path:
            return
        dirname = os.path.dirname(json_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        payload = {
            "total_step": self.total_steps,
            "completed_episodes": self.completed_episodes,
            "success_episodes": self.success_episodes,
            "crash_episodes": self.crash_episodes,
            "timeout_episodes": self.timeout_episodes,
            "outside_episodes": self.outside_episodes,
            **self._episode_rates(),
            "last_stats": {
                "q1_loss": self.last_stats.q1_loss,
                "q2_loss": self.last_stats.q2_loss,
                "critic_loss": self.last_stats.q1_loss + self.last_stats.q2_loss,
                "actor_loss": self.last_stats.actor_loss,
                "alpha_loss": self.last_stats.alpha_loss,
                "cost_loss": self.last_stats.cost_loss,
                "alpha": self.last_stats.alpha,
                "nu": self.last_stats.nu,
                "cost_penalty_coef": self.cost_penalty_coef,
                "kl": self.last_stats.kl,
            },
            "latest_episode": latest_episode,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def learn(
        self,
        total_timesteps: int,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 2500,
        csv_path: Optional[str] = None,
        tensorboard_log: Optional[str] = None,
        max_episodes: int = 0,
        min_ent_coef: float = 0.0,
        episode_csv_path: Optional[str] = None,
        metrics_json_path: Optional[str] = None,
    ):
        del tensorboard_log, min_ent_coef  # kept for call compatibility
        os.makedirs(checkpoint_dir, exist_ok=True) if checkpoint_dir else None
        obs = self.env.reset()
        self.reset_history(obs)
        obs_vec = self.history_obs[-1]
        self.episode_reward = 0.0
        self.episode_cost = 0.0
        self.episode_len = 0
        self.episode_num = 1
        self.completed_episodes = 0
        self.success_episodes = 0
        self.crash_episodes = 0
        self.timeout_episodes = 0
        self.outside_episodes = 0
        self.latest_episode_summary = None
        self._write_csv_header(csv_path) if csv_path else None
        self._write_episode_csv_header(episode_csv_path) if episode_csv_path else None
        self._write_metrics_json(metrics_json_path) if metrics_json_path else None

        for step in range(1, int(total_timesteps) + 1):
            self.total_steps = step
            if len(self.replay_buffer) < self.learning_starts:
                action = self.env.action_space.sample().astype(np.float32)
                if self.safety_shield:
                    action = self.apply_safety_shield(action, obs_vec)
            else:
                action = self.select_action(deterministic=False)

            next_obs, reward, done, info = self.env.step(action)
            if isinstance(info, dict):
                executed_action = np.asarray(info.get("executed_action", action), dtype=np.float32)
            else:
                executed_action = action
            next_obs_vec = self.vectorizer(next_obs)
            action_norm = self.normalize_action(executed_action)
            cost = float(info.get("constraint_cost", 0.0)) if isinstance(info, dict) else 0.0
            self.replay_buffer.add(obs_vec, action_norm, float(reward), next_obs_vec, bool(done), cost)
            self.observe(executed_action, reward, next_obs)
            self.episode_reward += float(reward)
            self.episode_cost += cost
            self.episode_len += 1

            if len(self.replay_buffer) >= self.learning_starts and step % max(1, self.train_freq) == 0:
                for _ in range(max(1, self.gradient_steps)):
                    if self.replay_buffer.can_sample(self.batch_size):
                        self.update_parameters()

            episode_summary = None
            if done:
                episode_summary = self._episode_summary(info)
                self.latest_episode_summary = episode_summary
                self._append_episode_csv(episode_csv_path, episode_summary) if episode_csv_path else None
                self._write_metrics_json(metrics_json_path, episode_summary) if metrics_json_path else None

            self._append_csv(csv_path, float(reward), cost, bool(done), info) if csv_path else None

            if checkpoint_dir and checkpoint_interval > 0 and step % checkpoint_interval == 0:
                self.save(os.path.join(checkpoint_dir, f"gpide_ckpt_{step}.pt"), total_step=step)

            if done:
                print(
                    "[GPIDE-SAC] episode={} len={} reward={:.3f} cost={:.3f} "
                    "success_rate={:.3f} crash_rate={:.3f} timeout_rate={:.3f} info={}".format(
                        self.episode_num, self.episode_len, self.episode_reward, self.episode_cost,
                        episode_summary["success_rate"], episode_summary["crash_rate"],
                        episode_summary["timeout_rate"], info
                    ),
                    flush=True,
                )
                if max_episodes and self.episode_num >= max_episodes:
                    break
                obs = self.env.reset()
                self.reset_history(obs)
                obs_vec = self.history_obs[-1]
                self.episode_reward = 0.0
                self.episode_cost = 0.0
                self.episode_len = 0
                self.episode_num += 1
            else:
                obs = next_obs
                obs_vec = next_obs_vec
        self._write_metrics_json(metrics_json_path, self.latest_episode_summary) if metrics_json_path else None
        return self

    def save(self, path: str, total_step: Optional[int] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "nu": self.nu,
            "total_step": total_step if total_step is not None else self.total_steps,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "cost_penalty_coef": self.cost_penalty_coef,
            "reward_history_scale": self.reward_history_scale,
            "training_metrics": {
                "completed_episodes": self.completed_episodes,
                "success_episodes": self.success_episodes,
                "crash_episodes": self.crash_episodes,
                "timeout_episodes": self.timeout_episodes,
                "outside_episodes": self.outside_episodes,
                **self._episode_rates(),
                "latest_episode": self.latest_episode_summary,
            },
        }
        if self.focops_enabled:
            payload.update({
                "c1": self.c1.state_dict(),
                "c2": self.c2.state_dict(),
                "target_c1": self.target_c1.state_dict(),
                "target_c2": self.target_c2.state_dict(),
                "c1_opt": self.c1_opt.state_dict(),
                "c2_opt": self.c2_opt.state_dict(),
                "old_actor": self.old_actor.state_dict(),
            })
        torch.save(payload, path)

    def load(self, path: str):
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.q1.load_state_dict(payload.get("q1", self.q1.state_dict()))
        self.q2.load_state_dict(payload.get("q2", self.q2.state_dict()))
        self.target_q1.load_state_dict(payload.get("target_q1", self.q1.state_dict()))
        self.target_q2.load_state_dict(payload.get("target_q2", self.q2.state_dict()))
        if "log_alpha" in payload:
            with torch.no_grad():
                self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        self.nu = float(payload.get("nu", self.nu))
        if self.focops_enabled and "c1" in payload:
            self.c1.load_state_dict(payload["c1"])
            self.c2.load_state_dict(payload["c2"])
            self.target_c1.load_state_dict(payload.get("target_c1", payload["c1"]))
            self.target_c2.load_state_dict(payload.get("target_c2", payload["c2"]))
            self.old_actor.load_state_dict(payload.get("old_actor", payload["actor"]))
        return self


def build_sequence_agent(cfg, env) -> SequenceGPIDESACAgent:
    return SequenceGPIDESACAgent(cfg, env)


def load_sequence_agent(model_file, cfg, env) -> SequenceGPIDESACAgent:
    agent = SequenceGPIDESACAgent(cfg, env)
    agent.load(model_file)
    return agent
