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
    return cfg, resolved_config
