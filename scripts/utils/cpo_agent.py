"""Lightweight constrained policy optimizer for the AirSim gym wrappers.

The original jachiam/cpo repository is an rllab module.  This project uses a
Gym/SB3-style training loop, so this file provides a small PyTorch implementation
with the same high-level contract used by the local training/evaluation code:
``learn()``, ``predict()``, ``save()`` and ``load()``.

The update is a practical CPO-style primal-dual policy-gradient optimizer:
it optimizes reward advantages while penalizing cost advantages from
``info["constraint_cost"]`` with an adaptive Lagrange multiplier.  It is kept
dependency-light so ``algo = CPO`` can run on NH/City_400 without rllab.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _cfg_get(cfg, section, option, default, cast):
    try:
        if cast is bool:
            return cfg.getboolean(section, option)
        if cast is int:
            return cfg.getint(section, option)
        if cast is float:
            return cfg.getfloat(section, option)
        return cfg.get(section, option)
    except Exception:
        return default


def _mlp(sizes, activation=nn.Tanh, output_activation=None):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.pi = _mlp([obs_dim] + hidden_sizes + [act_dim], activation, None)
        self.v = _mlp([obs_dim] + hidden_sizes + [1], activation, None)
        self.vc = _mlp([obs_dim] + hidden_sizes + [1], activation, None)
        self.log_std = nn.Parameter(-0.5 * th.ones(act_dim))

    def distribution(self, obs):
        mean = self.pi(obs)
        std = th.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, obs):
        return self.v(obs).squeeze(-1), self.vc(obs).squeeze(-1)


@dataclass
class CPOConfig:
    steps_per_epoch: int = 2048
    gamma: float = 0.99
    cost_gamma: float = 0.99
    lam: float = 0.95
    cost_lam: float = 0.95
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iters: int = 10
    train_v_iters: int = 40
    target_kl: float = 0.02
    clip_ratio: float = 0.2
    cost_limit: float = 0.05
    lagrange_lr: float = 0.05
    max_lagrange: float = 50.0
    hidden_sizes: Tuple[int, ...] = (256, 256)
    device: str = "cuda" if th.cuda.is_available() else "cpu"


class CPOAgent:
    config_section = "CPO"

    def __init__(self, env, cfg, policy_kwargs=None):
        self.env = env
        self.cfg = cfg
        self.obs_shape = env.observation_space.shape
        self.act_low = env.action_space.low.astype(np.float32)
        self.act_high = env.action_space.high.astype(np.float32)
        self.act_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = int(np.prod(self.obs_shape))

        section = self.config_section
        hidden = _cfg_get(cfg, section, "hidden_sizes", None, str)
        if hidden:
            hidden_sizes = tuple(int(x.strip()) for x in hidden.split(",") if x.strip())
        elif policy_kwargs and "net_arch" in policy_kwargs:
            hidden_sizes = tuple(int(x) for x in policy_kwargs["net_arch"])
        else:
            hidden_sizes = CPOConfig.hidden_sizes

        activation = nn.Tanh
        if policy_kwargs and policy_kwargs.get("activation_fn") is not None:
            activation = policy_kwargs["activation_fn"]

        self.hps = CPOConfig(
            steps_per_epoch=_cfg_get(cfg, section, "steps_per_epoch", 2048, int),
            gamma=_cfg_get(cfg, section, "gamma", _cfg_get(cfg, "DRL", "gamma", 0.99, float), float),
            cost_gamma=_cfg_get(cfg, section, "cost_gamma", 0.99, float),
            lam=_cfg_get(cfg, section, "lam", 0.95, float),
            cost_lam=_cfg_get(cfg, section, "cost_lam", 0.95, float),
            pi_lr=_cfg_get(cfg, section, "pi_lr", _cfg_get(cfg, "DRL", "learning_rate", 3e-4, float), float),
            vf_lr=_cfg_get(cfg, section, "vf_lr", 1e-3, float),
            train_pi_iters=_cfg_get(cfg, section, "train_pi_iters", 10, int),
            train_v_iters=_cfg_get(cfg, section, "train_v_iters", 40, int),
            target_kl=_cfg_get(cfg, section, "target_kl", 0.02, float),
            clip_ratio=_cfg_get(cfg, section, "clip_ratio", 0.2, float),
            cost_limit=_cfg_get(cfg, section, "cost_limit", 0.05, float),
            lagrange_lr=_cfg_get(cfg, section, "lagrange_lr", 0.05, float),
            max_lagrange=_cfg_get(cfg, section, "max_lagrange", 50.0, float),
            hidden_sizes=hidden_sizes,
            device=_cfg_get(cfg, section, "device", "cuda" if th.cuda.is_available() else "cpu", str),
        )
        self.device = th.device(self.hps.device if th.cuda.is_available() or self.hps.device == "cpu" else "cpu")
        self.ac = ActorCritic(self.obs_dim, self.act_dim, list(self.hps.hidden_sizes), activation).to(self.device)
        self.pi_optimizer = th.optim.Adam(list(self.ac.pi.parameters()) + [self.ac.log_std], lr=self.hps.pi_lr)
        self.v_optimizer = th.optim.Adam(list(self.ac.v.parameters()) + list(self.ac.vc.parameters()), lr=self.hps.vf_lr)
        self.lagrange = 0.0
        self._last_obs = None

    def _obs_tensor(self, obs):
        arr = np.asarray(obs, dtype=np.float32)
        if arr.size and arr.max() > 1.5:
            arr = arr / 255.0
        return th.as_tensor(arr.reshape(1, -1), dtype=th.float32, device=self.device)

    def _scale_action(self, action_norm):
        return self.act_low + (action_norm + 1.0) * 0.5 * (self.act_high - self.act_low)

    def _unscale_action(self, action):
        return 2.0 * (action - self.act_low) / np.maximum(self.act_high - self.act_low, 1e-6) - 1.0

    def predict(self, obs, deterministic=True):
        with th.no_grad():
            obs_t = self._obs_tensor(obs)
            dist = self.ac.distribution(obs_t)
            raw = dist.mean if deterministic else dist.sample()
            action_norm = th.tanh(raw).cpu().numpy()[0]
        return self._scale_action(action_norm).astype(np.float32), None

    def _discount_cumsum(self, x, discount):
        y = np.zeros_like(x, dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(x))):
            running = x[t] + discount * running
            y[t] = running
        return y

    def _finish_path(self, rewards, costs, values, cost_values, dones, last_v, last_cv):
        rewards = np.asarray(rewards, dtype=np.float32)
        costs = np.asarray(costs, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        cost_values = np.asarray(cost_values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        adv = np.zeros_like(rewards, dtype=np.float32)
        ret = np.zeros_like(rewards, dtype=np.float32)
        cadv = np.zeros_like(costs, dtype=np.float32)
        cret = np.zeros_like(costs, dtype=np.float32)
        gae = cgae = 0.0
        next_v, next_cv = last_v, last_cv
        next_ret, next_cret = last_v, last_cv
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.hps.gamma * next_v * mask - values[t]
            gae = delta + self.hps.gamma * self.hps.lam * mask * gae
            adv[t] = gae
            next_ret = rewards[t] + self.hps.gamma * mask * next_ret
            ret[t] = next_ret

            cdelta = costs[t] + self.hps.cost_gamma * next_cv * mask - cost_values[t]
            cgae = cdelta + self.hps.cost_gamma * self.hps.cost_lam * mask * cgae
            cadv[t] = cgae
            next_cret = costs[t] + self.hps.cost_gamma * mask * next_cret
            cret[t] = next_cret
            next_v, next_cv = values[t], cost_values[t]
        return adv, ret, cadv, cret

    def learn(self, total_timesteps, log_interval=1, callback=None):
        obs = self.env.reset()
        self._last_obs = obs
        steps_done = 0
        epoch = 0
        ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
        while steps_done < total_timesteps:
            batch = {k: [] for k in ("obs", "actn", "logp", "rew", "cost", "val", "cval", "done")}
            path = {k: [] for k in ("rew", "cost", "val", "cval")}
            for _ in range(min(self.hps.steps_per_epoch, total_timesteps - steps_done)):
                obs_t = self._obs_tensor(obs)
                with th.no_grad():
                    dist = self.ac.distribution(obs_t)
                    raw = dist.sample()
                    action_norm_t = th.tanh(raw)
                    logp = dist.log_prob(raw).sum(axis=-1)
                    v, vc = self.ac.value(obs_t)
                action_norm = action_norm_t.cpu().numpy()[0]
                action = self._scale_action(action_norm).astype(np.float32)
                next_obs, reward, done, info = self.env.step(action)
                cost = float(info.get("constraint_cost", 0.0)) if isinstance(info, dict) else 0.0

                batch["obs"].append(np.asarray(obs, dtype=np.float32).reshape(-1))
                batch["actn"].append(action_norm)
                batch["logp"].append(float(logp.cpu().numpy()[0]))
                batch["rew"].append(float(reward))
                batch["cost"].append(cost)
                batch["val"].append(float(v.cpu().numpy()[0]))
                batch["cval"].append(float(vc.cpu().numpy()[0]))
                batch["done"].append(float(done))
                path["rew"].append(float(reward))
                path["cost"].append(cost)
                path["val"].append(float(v.cpu().numpy()[0]))
                path["cval"].append(float(vc.cpu().numpy()[0]))

                ep_ret += float(reward)
                ep_cost += cost
                ep_len += 1
                steps_done += 1
                obs = next_obs
                if done:
                    obs = self.env.reset()
                    path = {k: [] for k in path}
                    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0

            with th.no_grad():
                last_v, last_cv = self.ac.value(self._obs_tensor(obs))
            adv, ret, cadv, cret = self._finish_path(
                batch["rew"], batch["cost"], batch["val"], batch["cval"], batch["done"],
                float(last_v.cpu().numpy()[0]), float(last_cv.cpu().numpy()[0]))
            self._update(batch, adv, ret, cadv, cret)
            epoch += 1
            if log_interval and epoch % log_interval == 0:
                print("CPO epoch {} steps {} lagrange {:.3f} cost {:.3f}".format(
                    epoch, steps_done, self.lagrange, float(np.mean(batch["cost"]) if batch["cost"] else 0.0)))
        self._last_obs = obs
        return self

    def _update(self, batch: Dict, adv, ret, cadv, cret):
        obs = th.as_tensor(np.asarray(batch["obs"], dtype=np.float32), dtype=th.float32, device=self.device)
        if obs.numel() and obs.max() > 1.5:
            obs = obs / 255.0
        actn = th.as_tensor(np.asarray(batch["actn"], dtype=np.float32), dtype=th.float32, device=self.device)
        old_logp = th.as_tensor(np.asarray(batch["logp"], dtype=np.float32), dtype=th.float32, device=self.device)
        adv_t = th.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=th.float32, device=self.device)
        cadv_t = th.as_tensor((cadv - cadv.mean()) / (cadv.std() + 1e-8), dtype=th.float32, device=self.device)
        ret_t = th.as_tensor(ret, dtype=th.float32, device=self.device)
        cret_t = th.as_tensor(cret, dtype=th.float32, device=self.device)

        mean_cost = float(np.mean(batch["cost"]) if batch["cost"] else 0.0)
        self.lagrange = float(np.clip(
            self.lagrange + self.hps.lagrange_lr * (mean_cost - self.hps.cost_limit),
            0.0, self.hps.max_lagrange))

        raw_act = th.atanh(th.clamp(actn, -0.999, 0.999))
        for _ in range(self.hps.train_pi_iters):
            dist = self.ac.distribution(obs)
            logp = dist.log_prob(raw_act).sum(axis=-1)
            ratio = th.exp(logp - old_logp)
            clipped = th.clamp(ratio, 1 - self.hps.clip_ratio, 1 + self.hps.clip_ratio)
            reward_obj = th.min(ratio * adv_t, clipped * adv_t)
            cost_obj = ratio * cadv_t
            loss_pi = -(reward_obj - self.lagrange * cost_obj).mean()
            approx_kl = (old_logp - logp).mean().item()
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), 10.0)
            self.pi_optimizer.step()
            if approx_kl > 1.5 * self.hps.target_kl:
                break

        for _ in range(self.hps.train_v_iters):
            v, vc = self.ac.value(obs)
            loss_v = F.mse_loss(v, ret_t) + F.mse_loss(vc, cret_t)
            self.v_optimizer.zero_grad()
            loss_v.backward()
            self.v_optimizer.step()

    def save(self, path):
        if not path.endswith(".pt"):
            path = path + ".pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        th.save({
            "state_dict": self.ac.state_dict(),
            "lagrange": self.lagrange,
            "obs_shape": self.obs_shape,
            "act_low": self.act_low,
            "act_high": self.act_high,
            "hps": self.hps.__dict__,
        }, path)

    @classmethod
    def load(cls, path, env, cfg):
        if not path.endswith(".pt") and os.path.exists(path + ".pt"):
            path = path + ".pt"
        agent = cls(env, cfg)
        data = th.load(path, map_location=agent.device)
        agent.ac.load_state_dict(data["state_dict"])
        agent.lagrange = float(data.get("lagrange", 0.0))
        return agent


class PPOLagrangianAgent(CPOAgent):
    """PPO-Lagrangian variant from Safety Starter Agents' algorithm family.

    It shares the clipped policy-gradient/Lagrangian update with ``CPOAgent`` but
    reads hyperparameters from the ``[PPO-Lagrangian]`` config section and is
    exposed under ``algo = PPO-Lagrangian``.
    """

    config_section = "PPO-Lagrangian"
