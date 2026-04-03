import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

import limxsdk.datatypes as datatypes
import limxsdk.robot.Rate as Rate
from limxsdk.ability.base_ability import BaseAbility
from limxsdk.ability.registry import register_ability


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def find_project_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "source" / "limx_rl_lab").exists() and (candidate / "scripts" / "rsl_rl").exists():
            return candidate
    return start_path.parent


def resolve_policy_path(config_path: Path, policy_root: str | None, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    project_root = find_project_root(config_path.resolve())

    if policy_root:
        root_path = Path(policy_root)
        if not root_path.is_absolute():
            root_path = (project_root / root_path).resolve()
        else:
            root_path = root_path.resolve()
        return (root_path / path).resolve()

    return resolve_path(config_path.parent, path_str)


def as_float_array(value, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        return np.full(length, float(array.item()), dtype=np.float32)
    if array.size != length:
        raise ValueError(f"{name} must have length {length}, got {array.size}.")
    return array


def find_training_deploy_path(policy_path: Path) -> Path | None:
    candidates = [policy_path.parent / "deploy.yaml"]
    if len(policy_path.parents) >= 2:
        candidates.append(policy_path.parents[1] / "params" / "deploy.yaml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def max_abs_range(value, fallback: float) -> float:
    if value is None:
        return fallback
    if len(value) != 2:
        return fallback
    return float(max(abs(float(value[0])), abs(float(value[1]))))


def load_training_deploy_overrides(policy_path: Path, num_actions: int, update_rate: int) -> dict:
    deploy_path = find_training_deploy_path(policy_path)
    if deploy_path is None:
        return {}

    deploy_cfg = yaml.safe_load(deploy_path.read_text())
    overrides = {}

    action_cfg = deploy_cfg.get("actions", {}).get("JointPositionAction")
    if action_cfg is not None:
        if "offset" in action_cfg:
            overrides["default_angles"] = as_float_array(action_cfg["offset"], num_actions, "action offset")
        elif "default_joint_pos" in deploy_cfg:
            overrides["default_angles"] = as_float_array(
                deploy_cfg["default_joint_pos"], num_actions, "default_joint_pos"
            )

        if "scale" in action_cfg:
            overrides["action_scale"] = as_float_array(action_cfg["scale"], num_actions, "action scale")

    joint_ids_map = deploy_cfg.get("joint_ids_map")
    if joint_ids_map is not None:
        joint_ids_map = np.asarray(joint_ids_map, dtype=np.int64).reshape(-1)
        if joint_ids_map.size != num_actions:
            raise ValueError(f"joint_ids_map must have length {num_actions}, got {joint_ids_map.size}.")

        if "stiffness" in deploy_cfg:
            stiffness_sdk = as_float_array(deploy_cfg["stiffness"], num_actions, "stiffness")
            overrides["kps"] = stiffness_sdk[joint_ids_map]
        if "damping" in deploy_cfg:
            damping_sdk = as_float_array(deploy_cfg["damping"], num_actions, "damping")
            overrides["kds"] = damping_sdk[joint_ids_map]

    step_dt = deploy_cfg.get("step_dt")
    if step_dt is not None:
        overrides["control_decimation"] = max(1, int(round(float(step_dt) * update_rate)))

    gait_phase_cfg = deploy_cfg.get("observations", {}).get("gait_phase")
    if gait_phase_cfg is not None:
        gait_phase_params = gait_phase_cfg.get("params", {})
        overrides["gait_period"] = float(gait_phase_params.get("period", 0.0))
        if "command_threshold" in gait_phase_params:
            overrides["command_threshold"] = float(gait_phase_params["command_threshold"])

    command_ranges = deploy_cfg.get("commands", {}).get("base_velocity", {}).get("ranges")
    if command_ranges is not None:
        overrides["command_ranges"] = command_ranges

    overrides["deploy_path"] = deploy_path
    return overrides


def get_projected_gravity(quat_wxyz) -> np.ndarray:
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    rotation = R.from_quat(quat_xyzw)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return rotation.inv().apply(gravity_world).astype(np.float32)


def get_gait_phase(
    sim_time: float, period: float, cmd: np.ndarray | None = None, command_threshold: float = 0.1
) -> np.ndarray:
    if cmd is not None and np.linalg.norm(cmd) < command_threshold:
        return np.array([0.0, 1.0], dtype=np.float32)
    phase = 2.0 * np.pi * (sim_time % period) / max(period, 1e-6)
    return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)


def apply_deadband(value: float, deadband: float) -> float:
    if abs(value) <= deadband:
        return 0.0
    return float(value)


class SensorJoyInput:
    def __init__(self):
        self.axes = None

    def callback(self, sensor_joy: datatypes.SensorJoy) -> None:
        self.axes = copy.deepcopy(sensor_joy.axes)


@register_ability("walk/controller")
class WalkController(BaseAbility):
    def initialize(self, config):
        self.robot = self.get_robot_instance()
        self.script_dir = Path(__file__).resolve().parent
        self.param_path = resolve_path(self.script_dir, config.get("param_path", "walk_param.yaml"))

        try:
            self.walking_param = yaml.safe_load(self.param_path.read_text()) or {}
        except Exception as exc:
            self.logger.error(f"Failed to load walking parameters from {self.param_path}: {exc}")
            return False

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        humanoid_cfg = self.walking_param.get("HumanoidRobotCfg", {})
        control_cfg = humanoid_cfg.get("control", {})
        normalization_cfg = humanoid_cfg.get("normalization", {})
        size_cfg = humanoid_cfg.get("size", {})
        obs_scale_cfg = normalization_cfg.get("obs_scales", {})
        clip_cfg = normalization_cfg.get("clip_scales", {})

        self.update_rate = int(config.get("update_rate", self.walking_param.get("update_rate", 1000)))
        self.num_actions = int(size_cfg.get("actions_size", self.walking_param.get("num_actions", 31)))
        self.num_obs = int(size_cfg.get("observations_size", self.walking_param.get("num_obs", 104)))
        self.policy_joints = list(self.walking_param["policy_joints"])

        if len(self.policy_joints) != self.num_actions:
            self.logger.error(
                f"policy_joints length {len(self.policy_joints)} does not match num_actions {self.num_actions}"
            )
            return False

        policy_path_str = os.getenv("LIMX_WALK_POLICY") or config.get("policy_path") or self.walking_param.get("policy_path")
        policy_root = os.getenv("LIMX_POLICY_ROOT") or config.get("policy_root") or self.walking_param.get("policy_root")
        if not policy_path_str:
            self.logger.error("policy_path is not configured")
            return False

        policy_path = resolve_policy_path(self.param_path, policy_root, policy_path_str)
        if not policy_path.exists():
            self.logger.error(f"Policy file not found: {policy_path}")
            return False

        try:
            self.policy = torch.jit.load(str(policy_path), map_location="cpu")
            self.policy.eval()
        except Exception as exc:
            self.logger.error(f"Failed to load TorchScript policy {policy_path}: {exc}")
            return False

        deploy_overrides = load_training_deploy_overrides(
            policy_path=policy_path, num_actions=self.num_actions, update_rate=self.update_rate
        )
        if "deploy_path" in deploy_overrides:
            self.logger.info(f"Loaded training deploy parameters from {deploy_overrides['deploy_path']}")

        self.default_angles = as_float_array(
            deploy_overrides.get("default_angles", control_cfg.get("default_angle")), self.num_actions, "default_angle"
        )
        self.kps = as_float_array(deploy_overrides.get("kps", control_cfg.get("kp")), self.num_actions, "kp")
        self.kds = as_float_array(deploy_overrides.get("kds", control_cfg.get("kd")), self.num_actions, "kd")
        self.action_scale = as_float_array(
            deploy_overrides.get("action_scale", control_cfg.get("action_scale")), self.num_actions, "action_scale"
        )
        self.user_torque_limit = as_float_array(
            control_cfg.get("user_torque_limit"), self.num_actions, "user_torque_limit"
        )

        self.ang_vel_scale = float(self.walking_param.get("ang_vel_scale", obs_scale_cfg.get("ang_vel", 0.25)))
        self.dof_pos_scale = float(self.walking_param.get("dof_pos_scale", obs_scale_cfg.get("dof_pos", 1.0)))
        self.dof_vel_scale = float(self.walking_param.get("dof_vel_scale", obs_scale_cfg.get("dof_vel", 0.05)))
        self.cmd_scale = as_float_array(self.walking_param.get("cmd_scale", [1.0, 1.0, 1.0]), 3, "cmd_scale")
        self.clip_observations = float(clip_cfg.get("clip_observations", 100.0))
        self.clip_actions = float(clip_cfg.get("clip_actions", 100.0))

        self.control_decimation = int(deploy_overrides.get("control_decimation", control_cfg.get("decimation", 20)))
        self.gait_period = float(
            deploy_overrides.get("gait_period", size_cfg.get("gait_period", self.walking_param.get("gait_period", 0.72)))
        )
        self.command_threshold = float(
            deploy_overrides.get("command_threshold", config.get("command_threshold", control_cfg.get("command_threshold", 0.1)))
        )
        self.parallel_solve_required = bool(
            config.get("parallel_solve_required", control_cfg.get("parallel_solve_required", True))
        )
        self.command_warmup_s = float(config.get("command_warmup_s", control_cfg.get("command_warmup_s", 0.0)))
        self.joy_deadband = float(control_cfg.get("joy_deadband", 0.05))

        command_ranges = deploy_overrides.get("command_ranges", {})
        self.max_vx = float(config.get("max_vx", control_cfg.get("max_vx", max_abs_range(command_ranges.get("lin_vel_x"), 0.8))))
        self.max_vy = float(config.get("max_vy", control_cfg.get("max_vy", max_abs_range(command_ranges.get("lin_vel_y"), 0.2))))
        self.max_vz = float(config.get("max_vz", control_cfg.get("max_vz", max_abs_range(command_ranges.get("ang_vel_z"), 0.5))))

        self.cmd_init = np.asarray(self.walking_param.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.current_cmd = self.cmd_init.copy()

        self.joy_input = SensorJoyInput()
        self.robot.subscribeSensorJoy(self.joy_input.callback)

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.loop_count = 0
        self.robot_order_index = None

        min_obs_dim = 9 + 3 * self.num_actions + 2
        if self.num_obs < min_obs_dim:
            self.logger.error(f"num_obs={self.num_obs} is too small, expected at least {min_obs_dim}")
            return False

        self.logger.info(f"Walking policy loaded from {policy_path}")
        return True

    def wait_for_first_state(self):
        self.logger.info("Waiting for robot state and IMU data...")
        while self.running and (self.get_robot_state() is None or self.get_imu_data() is None):
            time.sleep(0.01)

        if not self.running:
            return False

        motor_names = list(self.get_robot_state().motor_names)
        missing = [name for name in self.policy_joints if name not in motor_names]
        if missing:
            raise RuntimeError(f"Robot state is missing joints required by the policy: {missing}")

        self.robot_order_index = [motor_names.index(name) for name in self.policy_joints]
        return True

    def read_command(self, sim_time: float) -> np.ndarray:
        axes = self.joy_input.axes
        if axes is None or len(axes) < 4:
            raw_cmd = self.cmd_init.copy()
        else:
            raw_cmd = np.array(
                [
                    apply_deadband(float(axes[1]), self.joy_deadband) * self.max_vx,
                    apply_deadband(float(axes[0]), self.joy_deadband) * self.max_vy,
                    apply_deadband(float(axes[3]), self.joy_deadband) * self.max_vz,
                ],
                dtype=np.float32,
            )

        if self.command_warmup_s > 0.0:
            raw_cmd *= min(1.0, sim_time / self.command_warmup_s)
        return raw_cmd

    def build_observation(self, robot_state, imu_data, sim_time: float) -> np.ndarray:
        joint_pos = np.asarray(robot_state.q, dtype=np.float32)[self.robot_order_index]
        joint_vel = np.asarray(robot_state.dq, dtype=np.float32)[self.robot_order_index]

        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[:3] = np.asarray(imu_data.gyro, dtype=np.float32) * self.ang_vel_scale
        obs[3:6] = get_projected_gravity(imu_data.quat)
        obs[6:9] = self.current_cmd * self.cmd_scale
        obs[9 : 9 + self.num_actions] = (joint_pos - self.default_angles) * self.dof_pos_scale
        obs[9 + self.num_actions : 9 + 2 * self.num_actions] = joint_vel * self.dof_vel_scale
        obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.last_action
        obs[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = get_gait_phase(
            sim_time, self.gait_period, self.current_cmd, self.command_threshold
        )
        return np.clip(obs, -self.clip_observations, self.clip_observations)

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = self.policy(obs_tensor)
            if isinstance(action, (tuple, list)):
                action = action[0]
            action = action.detach().cpu().numpy().squeeze().astype(np.float32)
        return np.clip(action, -self.clip_actions, self.clip_actions)

    def publish_command(self, robot_state) -> None:
        motor_names = list(robot_state.motor_names)
        motor_count = len(motor_names)

        joint_pos = np.asarray(robot_state.q, dtype=np.float32)[self.robot_order_index]
        joint_vel = np.asarray(robot_state.dq, dtype=np.float32)[self.robot_order_index]
        safe_scale = np.maximum(self.action_scale, 1e-6)
        safe_kp = np.maximum(self.kps, 1e-6)
        soft_torque_limit = 0.95

        action_min = joint_pos - self.default_angles + (
            self.kds * joint_vel - self.user_torque_limit * soft_torque_limit
        ) / safe_kp
        action_max = joint_pos - self.default_angles + (
            self.kds * joint_vel + self.user_torque_limit * soft_torque_limit
        ) / safe_kp
        limited_action = np.clip(self.last_action, action_min / safe_scale, action_max / safe_scale)
        desired_q_policy = limited_action * self.action_scale + self.default_angles

        cmd_msg = datatypes.RobotCmd()
        cmd_msg.stamp = time.time_ns()
        cmd_msg.mode = [0 for _ in range(motor_count)]
        cmd_msg.q = [float(x) for x in robot_state.q]
        cmd_msg.dq = [0.0 for _ in range(motor_count)]
        cmd_msg.tau = [0.0 for _ in range(motor_count)]
        cmd_msg.Kp = [0.0 for _ in range(motor_count)]
        cmd_msg.Kd = [0.0 for _ in range(motor_count)]
        cmd_msg.motor_names = motor_names
        if hasattr(cmd_msg, "parallel_solve_required"):
            cmd_msg.parallel_solve_required = [self.parallel_solve_required for _ in range(motor_count)]

        for policy_idx, _joint_name in enumerate(self.policy_joints):
            robot_idx = self.robot_order_index[policy_idx]
            cmd_msg.q[robot_idx] = float(desired_q_policy[policy_idx])
            cmd_msg.Kp[robot_idx] = float(self.kps[policy_idx])
            cmd_msg.Kd[robot_idx] = float(self.kds[policy_idx])

        self.robot.publishRobotCmd(cmd_msg)

    def on_start(self):
        if not self.wait_for_first_state():
            return
        self.start_t = time.perf_counter()
        self.loop_count = 0
        self.last_action.fill(0.0)
        self.current_cmd = self.cmd_init.copy()
        self.logger.info("WalkController started")

    def on_main(self):
        rate = Rate(self.update_rate)
        while self.running:
            robot_state = copy.deepcopy(self.get_robot_state())
            imu_data = copy.deepcopy(self.get_imu_data())
            if robot_state is None or imu_data is None:
                rate.sleep()
                continue

            sim_time = self.loop_count / float(self.update_rate)
            self.current_cmd = self.read_command(sim_time)

            if self.loop_count % self.control_decimation == 0:
                obs = self.build_observation(robot_state, imu_data, sim_time)
                self.last_action = self.compute_action(obs)

            self.publish_command(robot_state)

            self.loop_count += 1
            if self.loop_count % self.update_rate == 0:
                fps = self.loop_count / max(1e-6, (time.perf_counter() - self.start_t))
                print(
                    f"[sim2real-walk] fps={fps:.1f} "
                    f"cmd=({self.current_cmd[0]:+.2f}, {self.current_cmd[1]:+.2f}, {self.current_cmd[2]:+.2f})",
                    end="\r",
                )
            rate.sleep()

    def on_stop(self):
        print()
        self.logger.info("WalkController stopped")
