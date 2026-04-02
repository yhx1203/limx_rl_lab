import logging
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import limxsdk.datatypes as datatypes
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType


logger = logging.getLogger("Main")


def run_ability_cli(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "limxsdk.ability.cli", *args], check=False).returncode


def switch_ability(stop_list: str, start_name: str) -> None:
    run_ability_cli("switch", stop_list, start_name)


def sensor_joy_callback(sensor_joy: datatypes.SensorJoy) -> None:
    if sensor_joy.buttons[4] == 1 and sensor_joy.buttons[3] == 1:
        switch_ability("walk damping", "stand")

    if sensor_joy.buttons[7] == 1 and sensor_joy.buttons[2] == 1:
        switch_ability("stand damping", "walk")

    if sensor_joy.buttons[4] == 1 and sensor_joy.buttons[0] == 1:
        switch_ability("stand walk", "damping")

    if sensor_joy.buttons[4] == 1 and sensor_joy.buttons[2] == 1:
        switch_ability("stand walk damping", "")


if __name__ == "__main__":
    robot_type = os.getenv("ROBOT_TYPE")
    if not robot_type:
        print("Error: Please set ROBOT_TYPE, for example `export ROBOT_TYPE=HU_D04_01`.")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    controller_path = script_dir / "controllers" / robot_type / "controllers.yaml"
    if not controller_path.exists():
        print(f"Error: controller config not found: {controller_path}")
        sys.exit(1)

    robot = Robot(RobotType.Humanoid)
    robot_ip = os.getenv("ROBOT_IP", "127.0.0.1")
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]

    if not robot.init(robot_ip):
        print(f"Error: failed to initialize robot at {robot_ip}")
        sys.exit(1)

    os.environ["ROBOT_IP"] = robot_ip

    sensor_joy_callback_partial = partial(sensor_joy_callback)
    robot.subscribeSensorJoy(sensor_joy_callback_partial)

    print(f"Loading controllers from {controller_path}")
    rc = run_ability_cli("load", "--config", str(controller_path))
    sys.exit(rc)
