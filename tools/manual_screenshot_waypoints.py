"""Interactive AirSim waypoint helper for manual NH/City_400 screenshots.

This script keeps one fixed target point and moves the drone through four
predefined screenshot poses.  At each pose it draws a target marker and a line
from the drone to the target, then waits until you type ``yes`` before moving to
the next pose.  You can manually take a screenshot from the Unreal/AirSim window
while the script is paused.

Example:
    python tools/manual_screenshot_waypoints.py --env NH_center
    python tools/manual_screenshot_waypoints.py --env City_400 --goal 160 -120 50

Coordinates use the project convention: x/y in AirSim local frame and positive
z means height above ground.  The script converts positive z to AirSim's NED
negative-z coordinate when calling AirSim.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import airsim


@dataclass(frozen=True)
class PoseSpec:
    """A screenshot pose using project coordinates (positive z is altitude)."""

    x: float
    y: float
    z: float
    yaw_deg: float | None = None


DEFAULT_SCENES = {
    "NH_center": {
        "config": "configs/config_NH_center_SimpleMultirotor_3D.ini",
        "goal": (68.0, -54.0, 5.0),
        "poses": (
            PoseSpec(-15.0, 12.0, 7.0),
            PoseSpec(10.0, -4.0, 8.0),
            PoseSpec(34.0, -25.0, 8.0),
            PoseSpec(56.0, -44.0, 7.0),
        ),
    },
    "City_400": {
        "config": "configs/config_City_400_Multirotor_2D.ini",
        "goal": (160.0, -160.0, 50.0),
        "poses": (
            PoseSpec(0.0, 0.0, 50.0),
            PoseSpec(45.0, -35.0, 50.0),
            PoseSpec(90.0, -80.0, 50.0),
            PoseSpec(130.0, -125.0, 50.0),
        ),
    },
}


def parse_xyz(values: Sequence[str]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected exactly three numbers: X Y Z")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def yaw_toward_goal(pose: PoseSpec, goal: Sequence[float]) -> float:
    return math.atan2(goal[1] - pose.y, goal[0] - pose.x)


def to_vector3r_xyz(point: Sequence[float]) -> airsim.Vector3r:
    return airsim.Vector3r(float(point[0]), float(point[1]), -float(point[2]))


def set_drone_pose(client: airsim.VehicleClient, pose_spec: PoseSpec, goal: Sequence[float]) -> None:
    yaw = math.radians(pose_spec.yaw_deg) if pose_spec.yaw_deg is not None else yaw_toward_goal(pose_spec, goal)
    pose = client.simGetVehiclePose()
    pose.position.x_val = pose_spec.x
    pose.position.y_val = pose_spec.y
    pose.position.z_val = -pose_spec.z
    pose.orientation = airsim.to_quaternion(0.0, 0.0, yaw)
    client.simSetVehiclePose(pose, True)


def draw_scene_guides(
    client: airsim.VehicleClient,
    pose_spec: PoseSpec,
    goal: Sequence[float],
    persistent_seconds: float,
) -> None:
    """Draw visible target/drone helpers in AirSim when debug-plot APIs exist."""
    try:
        client.simFlushPersistentMarkers()
    except Exception:
        pass

    drone_point = to_vector3r_xyz((pose_spec.x, pose_spec.y, pose_spec.z))
    target_point = to_vector3r_xyz(goal)
    line_points = [drone_point, target_point]

    try:
        client.simPlotPoints(
            [target_point],
            color_rgba=[1.0, 0.05, 0.05, 1.0],
            size=35.0,
            duration=persistent_seconds,
            is_persistent=True,
        )
        client.simPlotPoints(
            [drone_point],
            color_rgba=[0.05, 0.3, 1.0, 1.0],
            size=25.0,
            duration=persistent_seconds,
            is_persistent=True,
        )
        client.simPlotLineStrip(
            line_points,
            color_rgba=[1.0, 1.0, 0.0, 1.0],
            thickness=4.0,
            duration=persistent_seconds,
            is_persistent=True,
        )
        client.simPlotStrings(
            ["TARGET", "DRONE"],
            [target_point, drone_point],
            scale=2.0,
            color_rgba=[1.0, 1.0, 1.0, 1.0],
            duration=persistent_seconds,
        )
    except Exception as exc:
        print(f"[warn] AirSim debug marker drawing failed: {exc}")
        print("       The drone pose is still set; use the AirSim vehicle model/scene for screenshots.")


def load_endpoint_from_config(config_path: Path) -> tuple[str, int, float]:
    import configparser

    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return (
        cfg.get("airsim", "ip", fallback="127.0.0.1"),
        cfg.getint("airsim", "port", fallback=41451),
        cfg.getfloat("airsim", "timeout_value", fallback=10.0),
    )


def iter_yes(prompt: str) -> bool:
    answer = input(prompt).strip().lower()
    return answer in {"y", "yes", "是", "好", "继续"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move AirSim drone through four manual screenshot poses.")
    parser.add_argument("--env", choices=sorted(DEFAULT_SCENES), default="NH_center", help="Scene preset to use.")
    parser.add_argument("--config", type=Path, help="Optional project config; [airsim] endpoint is read from it.")
    parser.add_argument("--ip", help="Override AirSim RPC IP.")
    parser.add_argument("--port", type=int, help="Override AirSim RPC port.")
    parser.add_argument("--timeout", type=float, help="Override AirSim RPC timeout seconds.")
    parser.add_argument("--goal", nargs=3, metavar=("X", "Y", "Z"), type=float, help="Fixed target point.")
    parser.add_argument(
        "--pose",
        action="append",
        nargs=3,
        metavar=("X", "Y", "Z"),
        type=float,
        help="Screenshot pose. Repeat exactly four times to override the preset poses.",
    )
    parser.add_argument("--fov", type=float, default=90.0, help="Camera 0 field of view to set before pausing.")
    parser.add_argument("--marker-duration", type=float, default=3600.0, help="Debug marker lifetime in seconds.")
    parser.add_argument("--settle", type=float, default=0.5, help="Seconds to wait after each pose change.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scene = DEFAULT_SCENES[args.env]
    config_path = args.config or Path(scene["config"])
    ip, port, timeout = load_endpoint_from_config(config_path)
    ip = args.ip or ip
    port = args.port or port
    timeout = args.timeout or timeout

    goal = tuple(args.goal) if args.goal is not None else scene["goal"]
    poses: Iterable[PoseSpec]
    if args.pose is None:
        poses = scene["poses"]
    else:
        if len(args.pose) != 4:
            raise SystemExit("Please provide exactly four --pose X Y Z arguments, or omit --pose to use defaults.")
        poses = tuple(PoseSpec(*pose) for pose in args.pose)

    print(f"Connecting to AirSim {ip}:{port} (timeout={timeout}s) ...")
    client = airsim.VehicleClient(ip=ip, port=port, timeout_value=timeout)
    client.confirmConnection()

    try:
        client.simSetCameraFov("0", args.fov)
    except Exception as exc:
        print(f"[warn] Could not set camera FOV: {exc}")

    print(f"Fixed target: x={goal[0]:.2f}, y={goal[1]:.2f}, z={goal[2]:.2f}")
    print("At each stop, take your screenshot in AirSim/Unreal, then type 'yes' here.")

    pose_list = list(poses)
    for index, pose_spec in enumerate(pose_list, start=1):
        set_drone_pose(client, pose_spec, goal)
        time.sleep(args.settle)
        draw_scene_guides(client, pose_spec, goal, args.marker_duration)
        print(
            f"\n[{index}/{len(pose_list)}] Screenshot pose: "
            f"x={pose_spec.x:.2f}, y={pose_spec.y:.2f}, z={pose_spec.z:.2f}; "
            "drone yaw points at target."
        )
        if index < len(pose_list):
            while not iter_yes("截图完成后输入 yes 继续到下一个点: "):
                print("未继续。请输入 yes/y/是/好/继续，或按 Ctrl+C 退出。")

    print("\nAll four screenshot poses are complete.")


if __name__ == "__main__":
    main()
