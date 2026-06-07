"""AirSim client endpoint helpers."""

import socket


DEFAULT_AIRSIM_IP = "127.0.0.1"
DEFAULT_AIRSIM_PORT = 41451
DEFAULT_AIRSIM_TIMEOUT = 10.0


def get_airsim_endpoint(cfg):
    """Return AirSim RPC endpoint settings from config with safe defaults."""
    ip = cfg.get('airsim', 'ip', fallback=DEFAULT_AIRSIM_IP)
    port = cfg.getint('airsim', 'port', fallback=DEFAULT_AIRSIM_PORT)
    timeout = cfg.getfloat('airsim', 'timeout_value', fallback=DEFAULT_AIRSIM_TIMEOUT)
    return ip, port, timeout


def ensure_airsim_port_open(ip, port, timeout):
    """Fail fast if the configured AirSim RPC TCP endpoint is not reachable."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return
    except OSError as exc:
        raise RuntimeError(
            "Cannot connect to AirSim RPC endpoint {ip}:{port} within {timeout}s. "
            "Start the matching AirSim/Unreal scene first and make sure its "
            "settings.json uses the same ApiServerPort as this config's "
            "[airsim] port. If you are running NH and City_400 together, each "
            "simulator instance must have a different ApiServerPort and each "
            "training config must point to the matching port.".format(
                ip=ip, port=port, timeout=timeout)
        ) from exc
