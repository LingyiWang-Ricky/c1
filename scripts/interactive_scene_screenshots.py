"""Interactively place the UAV at fixed screenshot poses for NH/City_400 scenes.

This helper is intended for manual figure capture in AirSim/Unreal.  It fixes a
single target point, draws a visible target marker, moves the UAV to four preset
poses one by one, and waits for you to type ``yes`` before continuing to the next
pose.

Examples
--------
NH_center default four views::

    python scripts/interactive_scene_screenshots.py --scene NH

City_400 default four views::

    python scripts/interactive_scene_screenshots.py --scene City_400

Override target and UAV positions::

    python scripts/interactive_scene_screenshots.py \
        --scene City_400 \
        --goal 160 -160 50 \
        --positions "-160,160,50; -80,80,50; 40,-20,50; 120,-120,50"

Coordinate convention: x/y/z arguments use the same positive-up coordinates used
by this project configs.  The script converts z to AirSim NED internally.
"""

import argparse
import math
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
GYM_ENV_DIR = os.path.join(PROJECT_ROOT, "gym_env")
if GYM_ENV_DIR not in sys.path:
    sys.path.insert(0, GYM_ENV_DIR)

import airsim


SCENE_PRESETS = {
    "NH": {
        "goal": [90.0, 90.0, 10.0],
        "positions": [
            [-90.0, -90.0, 10.0],
            [-45.0, -45.0, 10.0],
            [0.0, 0.0, 10.0],
            [45.0, 45.0, 10.0],
        ],
    },
    "NH_center": {
        "goal": [90.0, 90.0, 10.0],
        "positions": [
            [-90.0, -90.0, 10.0],
            [-45.0, -45.0, 10.0],
            [0.0, 0.0, 10.0],
            [45.0, 45.0, 10.0],
        ],
    },
    "City_400": {
        "goal": [160.0, -160.0, 50.0],
        "positions": [
            [-160.0, 160.0, 50.0],
            [-80.0, 80.0, 50.0],
            [20.0, -20.0, 50.0],
            [100.0, -100.0, 50.0],
        ],
    },
    "City_400_400": {
        "goal": [160.0, -160.0, 50.0],
        "positions": [
            [-160.0, 160.0, 50.0],
            [-80.0, 80.0, 50.0],
            [20.0, -20.0, 50.0],
            [100.0, -100.0, 50.0],
        ],
    },
}


def parse_xyz(value):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Expected an xyz value formatted as 'x,y,z'.")
    return [float(part) for part in parts]


def parse_positions(value):
    positions = [parse_xyz(chunk) for chunk in value.split(";") if chunk.strip()]
    if len(positions) != 4:
        raise argparse.ArgumentTypeError(
            "Expected exactly four positions, e.g. 'x1,y1,z1; x2,y2,z2; x3,y3,z3; x4,y4,z4'.")
    return positions


def get_parser():
    parser = argparse.ArgumentParser(
        description="Move a UAV through four manual screenshot poses with a fixed target marker.")
    parser.add_argument(
        "--scene",
        choices=sorted(SCENE_PRESETS.keys()),
        default="NH_center",
        help="Preset scene coordinate set to use.",
    )
    parser.add_argument(
        "--goal",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Fixed target point in project positive-up coordinates.",
    )
    parser.add_argument(
        "--positions",
        type=parse_positions,
        default=None,
        help="Four UAV positions: 'x1,y1,z1; x2,y2,z2; x3,y3,z3; x4,y4,z4'.",
    )
    parser.add_argument(
        "--vehicle-name",
        default="",
        help="AirSim vehicle name. Leave empty for the default vehicle.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=30.0,
        help="Size of the red target debug marker.",
    )
    parser.add_argument(
        "--line-thickness",
        type=float,
        default=6.0,
        help="Thickness of the red vertical target guide line.",
    )
    parser.add_argument(
        "--no-flush",
        action="store_true",
        help="Do not clear existing persistent AirSim debug markers on start/exit.",
    )
    return parser


def to_vector3r(point):
    """Convert project positive-up xyz to AirSim NED Vector3r."""
    return airsim.Vector3r(point[0], point[1], -point[2])


def yaw_to_goal(position, goal):
    return math.atan2(goal[1] - position[1], goal[0] - position[0])


def set_vehicle_pose(client, position, goal, vehicle_name):
    yaw = yaw_to_goal(position, goal)
    pose = airsim.Pose(to_vector3r(position), airsim.to_quaternion(0.0, 0.0, yaw))
    client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)
    return math.degrees(yaw)


def draw_goal_marker(client, goal, marker_size, line_thickness):
    goal_point = to_vector3r(goal)
    line_bottom = to_vector3r([goal[0], goal[1], max(goal[2] - 12.0, 0.0)])
    line_top = to_vector3r([goal[0], goal[1], goal[2] + 12.0])
    client.simPlotPoints(
        [goal_point],
        color_rgba=[1.0, 0.0, 0.0, 1.0],
        size=marker_size,
        duration=0.0,
        is_persistent=True,
    )
    client.simPlotLineList(
        [line_bottom, line_top],
        color_rgba=[1.0, 0.0, 0.0, 1.0],
        thickness=line_thickness,
        duration=0.0,
        is_persistent=True,
    )


def prompt_next(index, total):
    while True:
        answer = input(
            "Screenshot pose {}/{} is ready. Type 'yes' to move to the next pose, "
            "or 'quit' to stop: ".format(index, total)
        ).strip().lower()
        if answer in {"yes", "y"}:
            return True
        if answer in {"quit", "q", "exit"}:
            return False
        print("Please type 'yes' to continue or 'quit' to stop.")


def main():
    parser = get_parser()
    args = parser.parse_args()
    preset = SCENE_PRESETS[args.scene]
    goal = args.goal if args.goal is not None else preset["goal"]
    positions = args.positions if args.positions is not None else preset["positions"]

    client = airsim.VehicleClient()
    client.confirmConnection()
    if not args.no_flush:
        client.simFlushPersistentMarkers()

    print("Scene: {}".format(args.scene))
    print("Fixed target point: {}".format(goal))
    print("UAV screenshot positions:")
    for i, position in enumerate(positions, start=1):
        print("  {}: {}".format(i, position))
    print("A red marker and vertical guide line will be drawn at the target point.")

    try:
        for index, position in enumerate(positions, start=1):
            if not args.no_flush:
                client.simFlushPersistentMarkers()
            draw_goal_marker(client, goal, args.marker_size, args.line_thickness)
            yaw_deg = set_vehicle_pose(client, position, goal, args.vehicle_name)
            print(
                "Pose {}/{}: UAV={}, target={}, yaw_to_target={:.1f} deg".format(
                    index, len(positions), position, goal, yaw_deg)
            )
            if index < len(positions):
                if not prompt_next(index, len(positions)):
                    break
            else:
                input("Final screenshot pose is ready. Press Enter to finish: ")
    finally:
        if not args.no_flush:
            client.simFlushPersistentMarkers()


if __name__ == "__main__":
    main()
