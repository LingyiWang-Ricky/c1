"""Optional adapter for sequence GPIDE agents.

The upstream sequence GPIDE implementation lives in ``algorithms.sequence_sac``.
That package is not part of the base project checkout, so importing it at module
load time breaks standard SB3 training/evaluation workflows.  This adapter keeps
GPIDE support available when the optional package exists while allowing regular
configs to run without it.
"""

import importlib
import importlib.util


_SEQUENCE_SAC_MODULE = None


def _find_sequence_sac_spec():
    """Return the optional sequence SAC module spec when it is installed."""
    if importlib.util.find_spec("algorithms") is None:
        return None
    return importlib.util.find_spec("algorithms.sequence_sac")


def _load_sequence_sac_module():
    """Load the optional sequence SAC implementation or explain how to proceed."""
    global _SEQUENCE_SAC_MODULE
    if _SEQUENCE_SAC_MODULE is not None:
        return _SEQUENCE_SAC_MODULE

    if _find_sequence_sac_spec() is None:
        raise ModuleNotFoundError(
            "GPIDE sequence training requires the optional "
            "'algorithms.sequence_sac' module, but it is not available in this "
            "checkout. Use a standard config without 'temporal_encoder = gpide' "
            "or add the optional algorithms package to the project/PYTHONPATH."
        )

    _SEQUENCE_SAC_MODULE = importlib.import_module("algorithms.sequence_sac")
    return _SEQUENCE_SAC_MODULE


def is_sequence_gpide_enabled(cfg):
    """Return whether a config requests the optional GPIDE sequence encoder."""
    if not cfg.has_section("options"):
        return False
    if not cfg.has_option("options", "temporal_encoder"):
        return False
    return cfg.get("options", "temporal_encoder").lower() == "gpide"


def build_sequence_agent(cfg, env):
    """Build a GPIDE sequence agent using the optional implementation."""
    return _load_sequence_sac_module().build_sequence_agent(cfg, env)


def load_sequence_agent(model_file, cfg, env):
    """Load a GPIDE sequence agent using the optional implementation."""
    return _load_sequence_sac_module().load_sequence_agent(model_file, cfg, env)
