"""Helpers for resolving and validating INI configuration files."""

import os
from configparser import ConfigParser


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))

ABLATION_ALIASES = {
    "": "",
    "none": "",
    "default": "",
    "sac": "sac",
    "base_sac": "sac",
    "baseline": "sac",
    "sac_baseline": "sac",
    "sac+innovation1": "sac_gpide",
    "sac_innovation1": "sac_gpide",
    "sac_pid": "sac_gpide",
    "sac+创新点1": "sac_gpide",
    "sac_创新点1": "sac_gpide",
    "sac_gpide": "sac_gpide",
    "gpide": "sac_gpide",
    "pid": "sac_gpide",
    "pid_inspired": "sac_gpide",
    "innovation1": "sac_gpide",
    "sac+innovation1+innovation2": "sac_gpide_focops",
    "sac_innovation1_innovation2": "sac_gpide_focops",
    "sac_pid_focops": "sac_gpide_focops",
    "sac+创新点1+创新点2": "sac_gpide_focops",
    "sac_创新点1_创新点2": "sac_gpide_focops",
    "sac_gpide_focops": "sac_gpide_focops",
    "gpide_focops": "sac_gpide_focops",
    "focops": "sac_gpide_focops",
    "innovation2": "sac_gpide_focops",
}


def resolve_project_path(path):
    """Resolve a path relative to the repository root when needed."""
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def available_config_files():
    """Return known config file paths relative to the repository root."""
    config_dir = os.path.join(PROJECT_ROOT, "configs")
    if not os.path.isdir(config_dir):
        return []
    return [
        os.path.join("configs", name)
        for name in sorted(os.listdir(config_dir))
        if name.endswith(".ini")
    ]


def canonical_ablation_mode(cfg):
    """Return the normalized SAC ablation mode requested by ``[ablation]``.

    Empty string means no explicit ablation override, preserving legacy configs.
    Supported explicit modes are:
    - ``sac``: plain SB3 SAC baseline.
    - ``sac_gpide``: SAC plus the PID/GPIDE-inspired sequence encoder.
    - ``sac_gpide_focops``: SAC plus PID/GPIDE and FOCOPS-inspired constraints.
    """
    if not cfg.has_section("ablation") or not cfg.has_option("ablation", "mode"):
        return ""
    raw_mode = cfg.get("ablation", "mode").strip().lower()
    normalized_key = raw_mode.replace(" ", "").replace("-", "_")
    normalized_key = normalized_key.replace("+", "+")
    mode = ABLATION_ALIASES.get(normalized_key)
    if mode is None:
        valid = "sac, sac_gpide, sac_gpide_focops"
        raise ValueError(
            "Unsupported [ablation] mode '{}'. Use one of: {}.".format(raw_mode, valid)
        )
    return mode


def normalize_ablation_config(cfg):
    """Apply SAC ablation mode overrides to a loaded ConfigParser in-place."""
    mode = canonical_ablation_mode(cfg)
    if not mode:
        return cfg

    if not cfg.has_section("options"):
        cfg.add_section("options")
    if not cfg.has_section("FOCOPS"):
        cfg.add_section("FOCOPS")

    requested_algo = cfg.get("options", "algo", fallback="SAC").strip().upper()
    if requested_algo != "SAC":
        # Ablation modes describe SAC/GPIDE/FOCOPS variants.  If the user selects
        # another public algorithm (for example CPO), preserve that selection so
        # changing only ``algo = CPO`` is enough to enter the requested branch.
        cfg.set("options", "temporal_encoder", "none")
        cfg.set("FOCOPS", "enabled", "False")
        cfg.set("options", "ablation_mode", "disabled_for_{}".format(requested_algo.lower()))
        return cfg

    # Keep the public algorithm as SAC for every ablation.  The temporal encoder
    # selects whether training uses SB3 SAC or the local sequence SAC variant.
    cfg.set("options", "algo", "SAC")
    cfg.set("options", "ablation_mode", mode)

    if mode == "sac":
        cfg.set("options", "temporal_encoder", "none")
        cfg.set("FOCOPS", "enabled", "False")
    elif mode == "sac_gpide":
        cfg.set("options", "temporal_encoder", "gpide")
        cfg.set("FOCOPS", "enabled", "False")
    elif mode == "sac_gpide_focops":
        cfg.set("options", "temporal_encoder", "gpide")
        cfg.set("FOCOPS", "enabled", "True")
    return cfg


def normalize_td3_config(cfg):
    """Apply TD3-specific compatibility fixes to a loaded ConfigParser in-place.

    The project SAC/GPIDE depth configs often use ``perception=depth`` together
    with ``policy_name=mlp``.  With SB3 TD3 that means flattening the whole
    60x90x2 image into an MLP while the goal direction is encoded in only a few
    state pixels.  In practice TD3 quickly collapses to the local "do not move,
    keep turning" policy because it cannot reliably read the target signal.

    Unless the user explicitly opts out, route that combination through the
    existing compact vector observation, which preserves depth-sector features
    plus distance/yaw state in a small MLP-friendly input.
    """
    if not cfg.has_section("options"):
        return cfg

    algo = cfg.get("options", "algo", fallback="").strip().upper()
    if algo != "TD3":
        return cfg

    perception = cfg.get("options", "perception", fallback="").strip().lower()
    policy_name = cfg.get("options", "policy_name", fallback="").strip().lower()
    keep_depth = cfg.getboolean("options", "td3_keep_depth_mlp", fallback=False)
    if perception == "depth" and policy_name == "mlp" and not keep_depth:
        cfg.set("options", "perception", "vector")
        cfg.set("options", "td3_auto_vectorized", "True")
        print(
            "TD3 config adjustment: perception=depth with policy_name=mlp was "
            "changed to perception=vector. Set options.td3_keep_depth_mlp=True "
            "to keep the original flattened-depth MLP behavior."
        )
    return cfg


def read_required_config(config_file):
    """Read a config file and fail early with an actionable error if invalid."""
    resolved_config = resolve_project_path(config_file)
    cfg = ConfigParser()
    read_files = cfg.read(resolved_config)
    if not read_files:
        available = available_config_files()
        available_text = ", ".join(available) if available else "<none found>"
        raise FileNotFoundError(
            "Cannot read config file '{}'. Resolved path: '{}'. "
            "Run from the project root or pass an existing config with --config. "
            "Available configs: {}".format(config_file, resolved_config, available_text)
        )
    if not cfg.has_section("options"):
        raise ValueError(
            "Config file '{}' was read but does not contain required [options] section."
            .format(resolved_config)
        )
    normalize_ablation_config(cfg)
    normalize_td3_config(cfg)
    return cfg, resolved_config
