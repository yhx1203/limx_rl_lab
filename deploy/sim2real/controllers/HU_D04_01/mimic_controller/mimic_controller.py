import copy
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

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


def as_float_array(value, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        return np.full(length, float(array.item()), dtype=np.float32)
    if array.size != length:
        raise ValueError(f"{name} must have length {length}, got {array.size}.")
    return array


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
SIM2SIM_DIR = PROJECT_ROOT / "deploy" / "sim2sim"
SIM2SIM_GAMEPAD = load_module_from_path(
    "limx_sim2sim_mimic_ability_shared",
    SIM2SIM_DIR / "gamepad_policy_controller.py",
)


@register_ability("mimic/controller")
class MimicController(BaseAbility):
    def initialize(self, config):
        self.robot = self.get_robot_instance()
        self.script_dir = Path(__file__).resolve().parent
        self.param_path = resolve_path(self.script_dir, config.get("param_path", "mimic_param.yaml"))
        self.walk_param_path = resolve_path(
            self.script_dir,
            config.get("walk_param_path", "../walk_controller/walk_param.yaml"),
        )

        try:
            self.mimic_param = yaml.safe_load(self.param_path.read_text()) or {}
        except Exception as exc:
            self.logger.error(f"Failed to load mimic parameters from {self.param_path}: {exc}")
            return False

        try:
            self.walk_param = yaml.safe_load(self.walk_param_path.read_text()) or {}
        except Exception as exc:
            self.logger.error(f"Failed to load walk parameters from {self.walk_param_path}: {exc}")
            return False

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        humanoid_cfg = self.walk_param.get("HumanoidRobotCfg", {})
        control_cfg = humanoid_cfg.get("control", {})

        self.update_rate = int(config.get("update_rate", self.mimic_param.get("update_rate", 1000)))
        self.parallel_solve_required = bool(
            config.get("parallel_solve_required", control_cfg.get("parallel_solve_required", True))
        )

        walk_policy_joints = list(self.walk_param.get("policy_joints", []))
        if not walk_policy_joints:
            self.logger.error("walk_param.yaml is missing policy_joints, which are required for mimic torque limits.")
            return False

        try:
            walk_user_torque_limit = as_float_array(
                control_cfg.get("user_torque_limit"),
                len(walk_policy_joints),
                "user_torque_limit",
            )
            walk_default_angles = as_float_array(
                control_cfg.get("default_angle"),
                len(walk_policy_joints),
                "default_angle",
            )
            walk_kps = as_float_array(control_cfg.get("kp"), len(walk_policy_joints), "kp")
            walk_kds = as_float_array(control_cfg.get("kd"), len(walk_policy_joints), "kd")
        except Exception as exc:
            self.logger.error(f"Failed to parse walk controller arrays: {exc}")
            return False

        walk_torque_limit_by_name = {
            joint_name: float(limit)
            for joint_name, limit in zip(walk_policy_joints, walk_user_torque_limit, strict=False)
        }
        walk_joint_index = {name: idx for idx, name in enumerate(walk_policy_joints)}
        self.head_hold_targets = {}
        for joint_name in SIM2SIM_GAMEPAD.HEAD_JOINT_NAMES:
            if joint_name in walk_joint_index:
                joint_idx = walk_joint_index[joint_name]
                self.head_hold_targets[joint_name] = (
                    float(walk_default_angles[joint_idx]),
                    float(walk_kps[joint_idx]),
                    float(walk_kds[joint_idx]),
                )

        mimic_policy = os.getenv("LIMX_MIMIC_POLICY") or config.get("policy_path") or self.mimic_param.get("policy_path")
        mimic_policy_root = (
            os.getenv("LIMX_MIMIC_POLICY_ROOT")
            or config.get("policy_root")
            or self.mimic_param.get("policy_root")
            or self.walk_param.get("policy_root")
        )
        mimic_motion_file = (
            os.getenv("LIMX_MIMIC_MOTION_FILE") or config.get("motion_file") or self.mimic_param.get("motion_file")
        )
        mimic_robot_xml = os.getenv("LIMX_MIMIC_ROBOT_XML") or config.get("robot_xml") or self.mimic_param.get(
            "robot_xml"
        )

        if not mimic_policy:
            self.logger.error("mimic policy_path is not configured")
            return False

        mimic_args = SimpleNamespace(
            mimic_policy=mimic_policy,
            mimic_policy_root=mimic_policy_root,
            mimic_motion_file=mimic_motion_file,
            mimic_robot_xml=mimic_robot_xml,
        )
        mimic_config = {
            "policy_root": mimic_policy_root,
            "mimic_policy_path": mimic_policy,
            "mimic_policy_root": mimic_policy_root,
            "mimic_motion_file": mimic_motion_file,
            "mimic_robot_xml": mimic_robot_xml,
            "mimic_reference_yaw_offset_deg": float(
                config.get(
                    "reference_yaw_offset_deg",
                    self.mimic_param.get("reference_yaw_offset_deg", 0.0),
                )
            ),
            "mimic_align_heading_on_switch": bool(
                config.get(
                    "align_heading_on_switch",
                    self.mimic_param.get("align_heading_on_switch", True),
                )
            ),
            "mimic_clip_observations": float(
                config.get("clip_observations", self.mimic_param.get("clip_observations", 100.0))
            ),
            "mimic_clip_actions": float(config.get("clip_actions", self.mimic_param.get("clip_actions", 100.0))),
        }
        shared_config_path = PROJECT_ROOT / "deploy" / "sim2sim" / "configs" / "limx_flat_velocity_sdk.yaml"

        try:
            self.mimic = SIM2SIM_GAMEPAD.build_mimic_bundle(
                args=mimic_args,
                config_path=shared_config_path,
                config=mimic_config,
                update_rate=self.update_rate,
                torque_limit_by_name=walk_torque_limit_by_name,
            )
        except Exception as exc:
            self.logger.error(f"Failed to build mimic policy bundle: {exc}")
            return False

        if self.mimic is None:
            self.logger.error("Mimic bundle is unavailable. Please configure mimic policy and motion paths.")
            return False

        try:
            self.anchor_kinematics = SIM2SIM_GAMEPAD.RobotAnchorKinematics(
                self.mimic.robot_xml_path,
                self.mimic.anchor_body_name,
            )
        except Exception as exc:
            self.logger.error(f"Failed to build mimic anchor kinematics: {exc}")
            return False

        self.loop_count = 0
        self.playback_time_s = 0.0
        self.start_t = 0.0

        self.logger.info(f"Mimic policy loaded from {self.mimic.policy_path}")
        self.logger.info(f"Mimic motion loaded from {self.mimic.motion.path}")
        return True

    def wait_for_first_state(self):
        self.logger.info("Waiting for robot state and IMU data...")
        while self.running and (self.get_robot_state() is None or self.get_imu_data() is None):
            time.sleep(0.01)

        if not self.running:
            return False

        motor_names = list(self.get_robot_state().motor_names)
        missing = [name for name in self.mimic.joint_names if name not in motor_names]
        if missing:
            raise RuntimeError(f"Robot state is missing joints required by the mimic policy: {missing}")

        self.mimic.robot_order_index = [motor_names.index(name) for name in self.mimic.joint_names]
        return True

    def _build_cmd_message(self, motor_names: list[str], q_init: np.ndarray) -> datatypes.RobotCmd:
        cmd_msg = datatypes.RobotCmd()
        cmd_msg.stamp = time.time_ns()
        cmd_msg.mode = [0 for _ in motor_names]
        cmd_msg.q = [float(x) for x in q_init]
        cmd_msg.dq = [0.0 for _ in motor_names]
        cmd_msg.tau = [0.0 for _ in motor_names]
        cmd_msg.Kp = [0.0 for _ in motor_names]
        cmd_msg.Kd = [0.0 for _ in motor_names]
        cmd_msg.motor_names = list(motor_names)
        if hasattr(cmd_msg, "parallel_solve_required"):
            cmd_msg.parallel_solve_required = [self.parallel_solve_required for _ in motor_names]
        return cmd_msg

    def _align_reference_heading(self, robot_state, imu_data):
        if not self.mimic.align_heading_on_switch:
            self.mimic.active_reference_yaw_offset_quat_w = self.mimic.reference_yaw_offset_quat_w.copy()
            return

        motor_names = list(robot_state.motor_names)
        all_joint_pos = np.asarray(robot_state.q, dtype=np.float32)
        robot_anchor_quat_w = self.anchor_kinematics.compute_anchor_quat(
            base_quat_wxyz=np.asarray(imu_data.quat, dtype=np.float32),
            joint_names=motor_names,
            joint_pos=all_joint_pos,
        )
        _, _, _, ref_anchor_quat_w = self.mimic.motion.sample(0.0)
        static_ref_anchor_quat_w = SIM2SIM_GAMEPAD.quat_multiply_wxyz(
            self.mimic.reference_yaw_offset_quat_w,
            ref_anchor_quat_w,
        )

        robot_heading_quat_w = SIM2SIM_GAMEPAD.yaw_quat_from_quat_wxyz(robot_anchor_quat_w)
        ref_heading_quat_w = SIM2SIM_GAMEPAD.yaw_quat_from_quat_wxyz(static_ref_anchor_quat_w)
        dynamic_heading_offset_w = SIM2SIM_GAMEPAD.quat_multiply_wxyz(
            robot_heading_quat_w,
            SIM2SIM_GAMEPAD.quat_conjugate_wxyz(ref_heading_quat_w),
        )
        active_offset_w = SIM2SIM_GAMEPAD.quat_multiply_wxyz(
            dynamic_heading_offset_w,
            self.mimic.reference_yaw_offset_quat_w,
        )
        self.mimic.active_reference_yaw_offset_quat_w = SIM2SIM_GAMEPAD.normalize_quat_wxyz(active_offset_w).astype(
            np.float32
        )

    def _build_mimic_observation(self, robot_state, imu_data) -> np.ndarray:
        joint_pos = np.asarray(robot_state.q, dtype=np.float32)[self.mimic.robot_order_index]
        joint_vel = np.asarray(robot_state.dq, dtype=np.float32)[self.mimic.robot_order_index]
        motor_names = list(robot_state.motor_names)
        all_joint_pos = np.asarray(robot_state.q, dtype=np.float32)

        (
            self.mimic.current_frame_index,
            ref_joint_pos,
            ref_joint_vel,
            ref_anchor_quat_w,
        ) = self.mimic.motion.sample(self.playback_time_s)
        ref_anchor_quat_w = SIM2SIM_GAMEPAD.quat_multiply_wxyz(
            self.mimic.active_reference_yaw_offset_quat_w,
            ref_anchor_quat_w,
        ).astype(np.float32)

        robot_anchor_quat_w = self.anchor_kinematics.compute_anchor_quat(
            base_quat_wxyz=np.asarray(imu_data.quat, dtype=np.float32),
            joint_names=motor_names,
            joint_pos=all_joint_pos,
        )
        relative_anchor_quat_w = SIM2SIM_GAMEPAD.quat_multiply_wxyz(
            SIM2SIM_GAMEPAD.quat_conjugate_wxyz(robot_anchor_quat_w),
            ref_anchor_quat_w,
        )

        raw_terms = {
            "command": np.concatenate([ref_joint_pos, ref_joint_vel], axis=0),
            "motion_anchor_ori_b": SIM2SIM_GAMEPAD.rotation_6d_from_quat_wxyz(relative_anchor_quat_w),
            "base_ang_vel": np.asarray(imu_data.gyro, dtype=np.float32),
            "joint_pos": joint_pos - self.mimic.default_joint_pos,
            "joint_vel": joint_vel,
            "actions": self.mimic.last_action,
        }

        obs_parts = []
        for obs_name in self.mimic.obs_terms:
            obs_value = raw_terms[obs_name].astype(np.float32, copy=False)
            obs_value = obs_value * self.mimic.obs_scales[obs_name]
            obs_clip = self.mimic.obs_clips[obs_name]
            if obs_clip is not None:
                if obs_clip.size == 1:
                    obs_value = np.clip(obs_value, -float(obs_clip[0]), float(obs_clip[0]))
                elif obs_clip.size == obs_value.size:
                    obs_value = np.clip(obs_value, -obs_clip, obs_clip)
                else:
                    raise RuntimeError(
                        f"Mimic observation `{obs_name}` clip dimension {obs_clip.size} "
                        f"does not match observation dimension {obs_value.size}."
                    )
            obs_parts.append(obs_value)

        observation = np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False)
        return np.clip(observation, -self.mimic.clip_observations, self.mimic.clip_observations)

    def _compute_mimic_action(self, observation: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            obs_tensor = torch.from_numpy(observation).unsqueeze(0)
            action = self.mimic.policy(obs_tensor)
            if isinstance(action, (tuple, list)):
                action = action[0]
            action = action.detach().cpu().numpy().squeeze().astype(np.float32)
        return np.clip(action, -self.mimic.clip_actions, self.mimic.clip_actions)

    def _publish_mimic_command(self, robot_state):
        motor_names = list(robot_state.motor_names)
        full_joint_pos = np.asarray(robot_state.q, dtype=np.float32)
        full_joint_vel = np.asarray(robot_state.dq, dtype=np.float32)
        joint_pos = full_joint_pos[self.mimic.robot_order_index]
        joint_vel = full_joint_vel[self.mimic.robot_order_index]

        safe_scale = np.maximum(self.mimic.action_scale, 1.0e-6)
        safe_kp = np.maximum(self.mimic.kps, 1.0e-6)
        soft_torque_limit = 0.95

        action_min = joint_pos - self.mimic.default_joint_pos + (
            self.mimic.kds * joint_vel - self.mimic.user_torque_limit * soft_torque_limit
        ) / safe_kp
        action_max = joint_pos - self.mimic.default_joint_pos + (
            self.mimic.kds * joint_vel + self.mimic.user_torque_limit * soft_torque_limit
        ) / safe_kp
        limited_action = np.clip(self.mimic.last_action, action_min / safe_scale, action_max / safe_scale)
        desired_q_policy = limited_action * self.mimic.action_scale + self.mimic.default_joint_pos

        cmd_msg = self._build_cmd_message(motor_names, np.asarray(robot_state.q, dtype=np.float32))
        for policy_index, joint_name in enumerate(self.mimic.joint_names):
            robot_index = self.mimic.robot_order_index[policy_index]
            cmd_msg.q[robot_index] = float(desired_q_policy[policy_index])
            cmd_msg.Kp[robot_index] = float(self.mimic.kps[policy_index])
            cmd_msg.Kd[robot_index] = float(self.mimic.kds[policy_index])

        for joint_name, (target_q, target_kp, target_kd) in self.head_hold_targets.items():
            if joint_name in motor_names:
                robot_index = motor_names.index(joint_name)
                cmd_msg.q[robot_index] = target_q
                cmd_msg.Kp[robot_index] = target_kp
                cmd_msg.Kd[robot_index] = target_kd

        self.robot.publishRobotCmd(cmd_msg)

    def on_start(self):
        if not self.wait_for_first_state():
            return

        robot_state = copy.deepcopy(self.get_robot_state())
        imu_data = copy.deepcopy(self.get_imu_data())
        self.loop_count = 0
        self.playback_time_s = 0.0
        self.mimic.last_action.fill(0.0)
        self._align_reference_heading(robot_state, imu_data)
        self.start_t = time.perf_counter()
        self.logger.info("MimicController started")

    def on_main(self):
        rate = Rate(self.update_rate)
        dt = 1.0 / float(self.update_rate)

        while self.running:
            robot_state = copy.deepcopy(self.get_robot_state())
            imu_data = copy.deepcopy(self.get_imu_data())
            if robot_state is None or imu_data is None:
                rate.sleep()
                continue

            if self.loop_count % self.mimic.control_decimation == 0:
                observation = self._build_mimic_observation(robot_state, imu_data)
                self.mimic.last_action = self._compute_mimic_action(observation)

            self._publish_mimic_command(robot_state)

            self.loop_count += 1
            self.playback_time_s += dt
            if self.loop_count % self.update_rate == 0:
                fps = self.loop_count / max(1e-6, (time.perf_counter() - self.start_t))
                print(
                    f"[sim2real-mimic] fps={fps:.1f} "
                    f"frame={self.mimic.current_frame_index:04d}/{self.mimic.motion.total_frames:04d} "
                    f"motion={self.mimic.motion.path.name}",
                    end="\r",
                )
            rate.sleep()

    def on_stop(self):
        print()
        self.logger.info("MimicController stopped")
