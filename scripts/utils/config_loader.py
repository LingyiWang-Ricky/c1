"""Helpers for resolving and validating INI configuration files."""

import os
from configparser import ConfigParser


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))


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
    return cfg, resolved_config
