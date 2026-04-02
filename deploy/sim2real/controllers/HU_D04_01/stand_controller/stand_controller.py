import time
from pathlib import Path

import yaml

import limxsdk.datatypes as datatypes
import limxsdk.robot.Rate as Rate
from limxsdk.ability.base_ability import BaseAbility
from limxsdk.ability.registry import register_ability


@register_ability("stand/controller")
class StandController(BaseAbility):
    def initialize(self, config):
        self.robot = self.get_robot_instance()
        self.update_rate = int(config.get("update_rate", 1000))

        param_path = Path(__file__).resolve().parent / "joint_params.yaml"
        try:
            self.joint_params = yaml.safe_load(param_path.read_text())
        except Exception as exc:
            self.logger.error(f"Failed to load joint params from {param_path}: {exc}")
            return False

        self.target_pos = self.joint_params.get("stand_pos", {})
        self.target_kp = self.joint_params.get("stand_kp", {})
        self.target_kd = self.joint_params.get("stand_kd", {})
        self.transition_steps = int(self.joint_params.get("transition_steps", 2000))
        self.robot_cmd = datatypes.RobotCmd()
        self.current_step = 0
        self.init_joint_angles = []
        return True

    def on_start(self):
        while self.running and self.get_robot_state() is None:
            self.logger.warning("Waiting for robot state data")
            time.sleep(0.1)

        if not self.running:
            return

        robot_state = self.get_robot_state()
        motor_names = list(robot_state.motor_names)
        self.init_joint_angles = list(robot_state.q)

        self.robot_cmd.motor_names = motor_names
        self.robot_cmd.mode = [0] * len(motor_names)
        self.robot_cmd.q = list(robot_state.q)
        self.robot_cmd.dq = [0.0] * len(motor_names)
        self.robot_cmd.tau = [0.0] * len(motor_names)
        self.robot_cmd.Kp = [float(self.target_kp.get(name, 0.0)) for name in motor_names]
        self.robot_cmd.Kd = [float(self.target_kd.get(name, 0.0)) for name in motor_names]
        self.current_step = 0
        self.logger.info("StandController started")

    def on_main(self):
        rate = Rate(self.update_rate)
        while self.running:
            progress = min(1.0, self.current_step / max(1, self.transition_steps))
            for idx, name in enumerate(self.robot_cmd.motor_names):
                target = float(self.target_pos.get(name, self.init_joint_angles[idx]))
                self.robot_cmd.q[idx] = (1.0 - progress) * self.init_joint_angles[idx] + progress * target

            self.robot.publishRobotCmd(self.robot_cmd)
            self.current_step += 1
            rate.sleep()

    def on_stop(self):
        self.logger.info("StandController stopped")
