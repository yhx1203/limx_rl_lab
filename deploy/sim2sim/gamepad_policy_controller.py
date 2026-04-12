import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

try:
    import pygame
    import pygame._sdl2.controller as sdl2_controller
except ImportError as exc:
    raise ImportError(
        "gamepad_policy_controller.py requires pygame with SDL2 controller support. "
        "Please install `pygame` in the runtime environment."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sdk_policy_controller import (
    LimxSDKPolicyController,
    Rate,
    as_float_array,
    get_projected_gravity,
    resolve_path,
    resolve_policy_path,
)

import limxsdk.datatypes as datatypes


AXIS_FULL_SCALE = 32768.0
PR_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]
HEAD_JOINT_NAMES = ["head_yaw_joint", "head_pitch_joint"]
SDK_JOINT_NAMES = PR_JOINT_NAMES + HEAD_JOINT_NAMES
SUPPORTED_MIMIC_OBS_TERMS = {
    "command",
    "motion_anchor_ori_b",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
}
DEFAULT_MIMIC_XML = "source/limx_rl_lab/data/Robots/limx/HU_D04_description/xml/HU_D04_01.xml"


def project_root_from_config(config_path: Path) -> Path:
    return config_path.parents[3]


def normalize_axis(raw_value: int) -> float:
    return float(np.clip(raw_value / AXIS_FULL_SCALE, -1.0, 1.0))


def apply_deadzone(value: float, deadzone: float, expo: float) -> float:
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0

    scaled = (magnitude - deadzone) / max(1e-6, 1.0 - deadzone)
    if expo != 1.0:
        scaled = scaled**expo
    return float(np.sign(value) * scaled)


def slew_limit_vector(
    current: np.ndarray, target: np.ndarray, rise_rate: np.ndarray, fall_rate: np.ndarray, dt: float
) -> np.ndarray:
    delta = target - current
    same_direction = current * target >= 0.0
    growing = np.abs(target) > np.abs(current)
    use_rise = same_direction & growing
    rate = np.where(use_rise, rise_rate, fall_rate)
    max_delta = rate * max(dt, 1e-6)
    return current + np.clip(delta, -max_delta, max_delta)


def find_training_deploy_path(policy_path: Path) -> Path | None:
    candidates = [policy_path.parent / "deploy.yaml"]
    if len(policy_path.parents) >= 2:
        candidates.append(policy_path.parents[1] / "params" / "deploy.yaml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_training_env_path(policy_path: Path) -> Path | None:
    if len(policy_path.parents) >= 2:
        candidate = policy_path.parents[1] / "params" / "env.yaml"
        if candidate.exists():
            return candidate
    return None


def load_training_motion_path(policy_path: Path) -> Path | None:
    env_path = find_training_env_path(policy_path)
    if env_path is None:
        return None

    try:
        env_cfg = yaml.load(env_path.read_text(), Loader=yaml.UnsafeLoader)
    except Exception:
        return None

    motion_path = env_cfg.get("commands", {}).get("motion", {}).get("motion_file")
    if not motion_path:
        return None
    return Path(motion_path).resolve()


def quat_conjugate_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]], dtype=np.float64)


def normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return quat_wxyz / max(1e-9, np.linalg.norm(quat_wxyz))


def quat_multiply_wxyz(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(lhs_wxyz, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(rhs_wxyz, dtype=np.float64)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quat_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    yaw_rad = np.deg2rad(float(yaw_deg))
    half_yaw = 0.5 * yaw_rad
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)


def yaw_rad_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = normalize_quat_wxyz(quat_wxyz)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quat_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    yaw_rad = yaw_rad_from_quat_wxyz(quat_wxyz)
    half_yaw = 0.5 * yaw_rad
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)


def wrap_degrees(angle_deg: float) -> float:
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def rotation_6d_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = normalize_quat_wxyz(quat_wxyz)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    rotation = R.from_quat(quat_xyzw).as_matrix()
    return rotation[:, :2].reshape(-1).astype(np.float32)


class GamepadTeleop:
    def __init__(self, args):
        self.controller_index = args.controller_index
        self.max_lin_vel_x = float(args.max_lin_vel_x)
        self.max_lin_vel_y = float(args.max_lin_vel_y)
        self.max_ang_vel_z = float(args.max_ang_vel_z)
        self.deadzone = float(args.deadzone)
        self.expo = float(args.expo)
        self.slow_factor = float(args.slow_factor)
        self.reconnect_interval_s = float(args.reconnect_interval_s)

        self.enabled = bool(args.start_enabled)
        self.controller = None
        self.mode_request = None
        self._last_reconnect_attempt_s = -1e9
        self._last_waiting_log_s = -1e9

        self._init_pygame()
        self._try_open_controller(force=True)

    def _init_pygame(self):
        pygame.init()
        pygame.display.init()
        try:
            pygame.display.set_mode((1, 1))
        except pygame.error:
            pass

        sdl2_controller.init()
        sdl2_controller.set_eventstate(True)

    def _candidate_indices(self):
        count = sdl2_controller.get_count()
        if self.controller_index is not None:
            return [self.controller_index] if self.controller_index < count else []
        return [idx for idx in range(count) if sdl2_controller.is_controller(idx)]

    def _try_open_controller(self, force: bool = False):
        if self.controller is not None and self.controller.attached():
            return

        now = time.monotonic()
        if not force and (now - self._last_reconnect_attempt_s) < self.reconnect_interval_s:
            return
        self._last_reconnect_attempt_s = now

        for index in self._candidate_indices():
            if not sdl2_controller.is_controller(index):
                continue

            self.controller = sdl2_controller.Controller(index)
            print(
                f"[gamepad-sim2sim] connected controller index={index} name={self.controller.name!r}. "
                "START: enable/disable, BACK: pause, LB: slow mode, R1+X: walk, R1+A: mimic."
            )
            return

        if (now - self._last_waiting_log_s) >= 2.0:
            if sdl2_controller.get_count() == 0:
                print("[gamepad-sim2sim] waiting for a controller to be connected...")
            else:
                print(
                    "[gamepad-sim2sim] controller detected but not recognized by SDL2 game-controller API. "
                    "Try an Xbox/PS-compatible mode or pass --controller-index."
                )
            self._last_waiting_log_s = now

    def _handle_disconnect(self):
        if self.controller is not None:
            try:
                self.controller.quit()
            except Exception:
                pass
        self.controller = None
        self.enabled = False
        self.mode_request = None
        print("[gamepad-sim2sim] controller disconnected, commands set to zero.")

    def _update_mode_request(self):
        if self.controller is None:
            return

        right_shoulder = bool(self.controller.get_button(pygame.CONTROLLER_BUTTON_RIGHTSHOULDER))
        if not right_shoulder:
            return

        if self.controller.get_button(pygame.CONTROLLER_BUTTON_X):
            self.mode_request = "walk"
        elif self.controller.get_button(pygame.CONTROLLER_BUTTON_A):
            self.mode_request = "mimic"

        if self.mode_request is not None:
            print(f"[gamepad-sim2sim] requested mode switch: {self.mode_request}.")

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.CONTROLLERBUTTONDOWN:
                if event.button == pygame.CONTROLLER_BUTTON_START:
                    self.enabled = not self.enabled
                    state = "enabled" if self.enabled else "paused"
                    print(f"[gamepad-sim2sim] controller {state}.")
                elif event.button == pygame.CONTROLLER_BUTTON_BACK:
                    if self.enabled:
                        print("[gamepad-sim2sim] controller paused.")
                    self.enabled = False

                self._update_mode_request()

            if event.type == pygame.CONTROLLERDEVICEREMOVED and self.controller is not None:
                instance_id = getattr(event, "instance_id", None)
                if instance_id is None or instance_id == self.controller.id:
                    self._handle_disconnect()

            if event.type == pygame.CONTROLLERDEVICEADDED and self.controller is None:
                self._try_open_controller(force=True)

    def read_command(self) -> np.ndarray:
        self._process_events()
        self._try_open_controller()

        if self.controller is None or not self.controller.attached():
            if self.controller is not None:
                self._handle_disconnect()
            return np.zeros(3, dtype=np.float32)

        pygame.event.pump()
        sdl2_controller.update()

        if not self.enabled:
            return np.zeros(3, dtype=np.float32)

        left_y = -normalize_axis(self.controller.get_axis(pygame.CONTROLLER_AXIS_LEFTY))
        left_x = normalize_axis(self.controller.get_axis(pygame.CONTROLLER_AXIS_LEFTX))
        right_x = normalize_axis(self.controller.get_axis(pygame.CONTROLLER_AXIS_RIGHTX))

        vx = apply_deadzone(left_y, self.deadzone, self.expo) * self.max_lin_vel_x
        vy = apply_deadzone(left_x, self.deadzone, self.expo) * self.max_lin_vel_y
        wz = apply_deadzone(right_x, self.deadzone, self.expo) * self.max_ang_vel_z

        if self.controller.get_button(pygame.CONTROLLER_BUTTON_LEFTSHOULDER):
            vx *= self.slow_factor
            vy *= self.slow_factor
            wz *= self.slow_factor

        return np.array([vx, vy, wz], dtype=np.float32)

    def consume_mode_request(self) -> str | None:
        mode_request = self.mode_request
        self.mode_request = None
        return mode_request

    def close(self):
        if self.controller is not None:
            try:
                self.controller.quit()
            except Exception:
                pass
            self.controller = None
        sdl2_controller.quit()
        pygame.quit()


class RobotAnchorKinematics:
    def __init__(self, robot_xml_path: Path, anchor_body_name: str):
        self.model = mujoco.MjModel.from_xml_path(str(robot_xml_path))
        self.data = mujoco.MjData(self.model)
        floating_base_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if floating_base_joint_id < 0:
            raise RuntimeError("Could not find `floating_base_joint` in the MuJoCo XML.")
        self.floating_base_qpos_adr = self.model.jnt_qposadr[floating_base_joint_id]
        self.anchor_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name)
        if self.anchor_body_id < 0:
            raise RuntimeError(f"Could not find body `{anchor_body_name}` in the MuJoCo XML.")
        self.joint_qpos_adr = {
            joint_name: self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
            for joint_name in SDK_JOINT_NAMES
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0
        }

    def compute_anchor_quat(
        self, base_quat_wxyz: np.ndarray, joint_names: list[str], joint_pos: np.ndarray
    ) -> np.ndarray:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.floating_base_qpos_adr : self.floating_base_qpos_adr + 7] = np.array(
            [0.0, 0.0, 0.0, *np.asarray(base_quat_wxyz, dtype=np.float64)], dtype=np.float64
        )

        for name, value in zip(joint_names, joint_pos, strict=False):
            qpos_adr = self.joint_qpos_adr.get(name)
            if qpos_adr is not None:
                self.data.qpos[qpos_adr] = float(value)

        mujoco.mj_forward(self.model, self.data)
        return self.data.xquat[self.anchor_body_id].astype(np.float32).copy()


@dataclass
class MotionReference:
    path: Path
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    anchor_body_index: int

    @classmethod
    def load(cls, motion_path: Path, robot_xml_path: Path, anchor_body_name: str):
        data = np.load(str(motion_path))
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        anchor_body_index = infer_motion_anchor_body_index(
            joint_pos=joint_pos,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            robot_xml_path=robot_xml_path,
            anchor_body_name=anchor_body_name,
        )
        return cls(
            path=motion_path,
            fps=fps,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            anchor_body_index=anchor_body_index,
        )

    @property
    def total_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    def frame_index(self, playback_time_s: float) -> int:
        if self.total_frames <= 0:
            return 0
        frame = int(np.floor(max(playback_time_s, 0.0) * self.fps))
        return frame % self.total_frames

    def sample(self, playback_time_s: float) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        frame_index = self.frame_index(playback_time_s)
        return (
            frame_index,
            self.joint_pos[frame_index],
            self.joint_vel[frame_index],
            self.body_quat_w[frame_index, self.anchor_body_index],
        )


@dataclass
class MimicPolicyBundle:
    policy_path: Path
    motion_path: Path
    robot_xml_path: Path
    anchor_body_name: str
    policy: torch.jit.ScriptModule
    joint_names: list[str]
    action_scale: np.ndarray
    default_joint_pos: np.ndarray
    kps: np.ndarray
    kds: np.ndarray
    user_torque_limit: np.ndarray
    control_decimation: int
    clip_observations: float
    clip_actions: float
    obs_terms: list[str]
    obs_scales: dict[str, np.ndarray]
    obs_clips: dict[str, np.ndarray | None]
    motion: MotionReference
    reference_yaw_offset_quat_w: np.ndarray
    align_heading_on_switch: bool
    active_reference_yaw_offset_quat_w: np.ndarray | None = None
    robot_order_index: list[int] | None = None
    last_action: np.ndarray | None = None
    current_frame_index: int = 0

    def __post_init__(self):
        if self.last_action is None:
            self.last_action = np.zeros(len(self.joint_names), dtype=np.float32)
        if self.active_reference_yaw_offset_quat_w is None:
            self.active_reference_yaw_offset_quat_w = self.reference_yaw_offset_quat_w.copy()


def infer_motion_anchor_body_index(
    joint_pos: np.ndarray, body_pos_w: np.ndarray, body_quat_w: np.ndarray, robot_xml_path: Path, anchor_body_name: str
) -> int:
    model = mujoco.MjModel.from_xml_path(str(robot_xml_path))
    data = mujoco.MjData(model)
    anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name)
    floating_base_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if anchor_body_id < 0 or floating_base_joint_id < 0:
        raise RuntimeError(f"Failed to infer anchor body index for `{anchor_body_name}` from {robot_xml_path}.")

    floating_base_qpos_adr = model.jnt_qposadr[floating_base_joint_id]
    joint_qpos_adr = {
        joint_name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
        for joint_name in PR_JOINT_NAMES
    }

    frame_candidates = sorted(
        {
            0,
            min(10, joint_pos.shape[0] - 1),
            min(20, joint_pos.shape[0] - 1),
            max(0, joint_pos.shape[0] // 3),
            max(0, (2 * joint_pos.shape[0]) // 3),
            joint_pos.shape[0] - 1,
        }
    )

    total_score = np.zeros(body_pos_w.shape[1], dtype=np.float64)
    for frame_index in frame_candidates:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[floating_base_qpos_adr : floating_base_qpos_adr + 7] = np.array(
            [*body_pos_w[frame_index, 0], *body_quat_w[frame_index, 0]], dtype=np.float64
        )
        for joint_index, joint_name in enumerate(PR_JOINT_NAMES):
            data.qpos[joint_qpos_adr[joint_name]] = float(joint_pos[frame_index, joint_index])

        mujoco.mj_forward(model, data)
        target_pos = data.xpos[anchor_body_id]
        target_quat = data.xquat[anchor_body_id]
        pos_error = np.linalg.norm(body_pos_w[frame_index] - target_pos, axis=1)
        quat_alignment = np.abs(np.sum(body_quat_w[frame_index] * target_quat, axis=1))
        quat_error = 1.0 - np.clip(quat_alignment, 0.0, 1.0)
        total_score += pos_error + quat_error

    anchor_body_index = int(np.argmin(total_score))
    if total_score[anchor_body_index] > 1.0e-2:
        raise RuntimeError(
            f"Failed to match motion anchor body `{anchor_body_name}` in {robot_xml_path}. "
            f"Best matching score was {total_score[anchor_body_index]:.4f}."
        )
    return anchor_body_index


def build_mimic_bundle(
    args,
    config_path: Path,
    config: dict,
    update_rate: int,
    torque_limit_by_name: dict[str, float],
) -> MimicPolicyBundle | None:
    policy_path_str = args.mimic_policy or config.get("mimic_policy_path")
    if not policy_path_str:
        return None

    project_root = project_root_from_config(config_path)
    policy_root = args.mimic_policy_root or config.get("mimic_policy_root") or config.get("policy_root") or str(project_root)
    policy_path = resolve_policy_path(config_path, policy_root, policy_path_str)
    if not policy_path.exists():
        raise FileNotFoundError(f"Mimic policy file not found: {policy_path}")

    deploy_path = find_training_deploy_path(policy_path)
    if deploy_path is None:
        raise FileNotFoundError(f"Could not find deploy.yaml next to mimic policy: {policy_path}")
    deploy_cfg = yaml.load(deploy_path.read_text(), Loader=yaml.UnsafeLoader) or {}

    motion_path_str = args.mimic_motion_file or config.get("mimic_motion_file")
    if motion_path_str:
        motion_path = resolve_path(project_root, motion_path_str)
    else:
        motion_path = load_training_motion_path(policy_path)
    if motion_path is None:
        raise FileNotFoundError(
            "Could not infer the mimic motion file. Pass --mimic-motion-file or set mimic_motion_file in the config."
        )
    if not motion_path.exists():
        raise FileNotFoundError(f"Mimic motion file not found: {motion_path}")

    robot_xml_path = resolve_path(
        project_root, args.mimic_robot_xml or config.get("mimic_robot_xml", DEFAULT_MIMIC_XML)
    )
    if not robot_xml_path.exists():
        raise FileNotFoundError(f"Mimic robot xml not found: {robot_xml_path}")

    torch_policy = torch.jit.load(str(policy_path), map_location="cpu")
    torch_policy.eval()

    if len(deploy_cfg.get("actions", {})) != 1:
        raise RuntimeError("Mimic deploy.yaml must contain exactly one action term.")
    action_name, action_cfg = next(iter(deploy_cfg["actions"].items()))
    if action_name != "joint_pos":
        raise RuntimeError(
            f"Unsupported mimic action term `{action_name}`. Only the BeyondMimic `joint_pos` action is supported."
        )

    joint_names = list(action_cfg["joint_names"])
    joint_ids = np.asarray(action_cfg["joint_ids"], dtype=np.int64).reshape(-1)
    action_scale = as_float_array(action_cfg["scale"], len(joint_names), "mimic action scale")

    joint_ids_map = np.asarray(deploy_cfg["joint_ids_map"], dtype=np.int64).reshape(-1)
    default_joint_pos_asset = as_float_array(deploy_cfg["default_joint_pos"], joint_ids_map.size, "default_joint_pos")
    default_joint_pos = default_joint_pos_asset[joint_ids]

    stiffness_sdk = as_float_array(deploy_cfg["stiffness"], joint_ids_map.size, "stiffness")
    damping_sdk = as_float_array(deploy_cfg["damping"], joint_ids_map.size, "damping")
    action_sdk_indices = joint_ids_map[joint_ids]
    kps = stiffness_sdk[action_sdk_indices]
    kds = damping_sdk[action_sdk_indices]
    user_torque_limit = np.asarray([torque_limit_by_name[name] for name in joint_names], dtype=np.float32)

    obs_cfg = deploy_cfg.get("observations", {})
    obs_terms = list(obs_cfg.keys())
    unsupported_terms = [name for name in obs_terms if name not in SUPPORTED_MIMIC_OBS_TERMS]
    if unsupported_terms:
        raise RuntimeError(
            "The mimic sim2sim controller currently supports the "
            "`LimX-HU-D04-01-Flat-BeyondMimic-No-State-Estimation` observation set only. "
            f"Unsupported terms: {unsupported_terms}"
        )

    obs_scales = {}
    obs_clips = {}
    obs_dim = 0
    for obs_name, term_cfg in obs_cfg.items():
        history_length = int(term_cfg.get("history_length", 1))
        if history_length != 1:
            raise RuntimeError(f"Unsupported history_length={history_length} for mimic observation `{obs_name}`.")
        scale = np.asarray(term_cfg.get("scale", []), dtype=np.float32).reshape(-1)
        obs_scales[obs_name] = scale
        obs_dim += int(scale.size)
        clip_value = term_cfg.get("clip")
        if clip_value is None:
            obs_clips[obs_name] = None
        else:
            obs_clips[obs_name] = np.asarray(clip_value, dtype=np.float32).reshape(-1)

    with torch.inference_mode():
        test_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
        test_action = torch_policy(test_obs)
        if isinstance(test_action, (tuple, list)):
            test_action = test_action[0]
    if int(test_action.shape[-1]) != len(joint_names):
        raise RuntimeError(
            f"Mimic policy output dimension {int(test_action.shape[-1])} does not match action dimension {len(joint_names)}."
        )

    motion = MotionReference.load(motion_path, robot_xml_path, anchor_body_name="waist_pitch_link")
    control_decimation = max(1, int(round(float(deploy_cfg.get("step_dt", 0.02)) * update_rate)))
    reference_yaw_offset_quat_w = quat_from_yaw_deg(config.get("mimic_reference_yaw_offset_deg", 0.0)).astype(
        np.float32
    )

    return MimicPolicyBundle(
        policy_path=policy_path,
        motion_path=motion_path,
        robot_xml_path=robot_xml_path,
        anchor_body_name="waist_pitch_link",
        policy=torch_policy,
        joint_names=joint_names,
        action_scale=action_scale,
        default_joint_pos=default_joint_pos,
        kps=kps,
        kds=kds,
        user_torque_limit=user_torque_limit,
        control_decimation=control_decimation,
        clip_observations=float(config.get("mimic_clip_observations", 100.0)),
        clip_actions=float(config.get("mimic_clip_actions", 100.0)),
        obs_terms=obs_terms,
        obs_scales=obs_scales,
        obs_clips=obs_clips,
        motion=motion,
        reference_yaw_offset_quat_w=reference_yaw_offset_quat_w,
        align_heading_on_switch=bool(config.get("mimic_align_heading_on_switch", True)),
    )


class GamepadPolicyController(LimxSDKPolicyController):
    def __init__(self, args):
        super().__init__(args)
        self.walk_last_action = self.last_action.copy()
        self.walk_loop_count = 0
        self.mimic_loop_count = 0
        self.mimic_playback_time_s = 0.0
        self.active_mode = "walk"

        self.target_cmd = self.cmd.copy()
        self.teleop = GamepadTeleop(args)
        self.command_rise_rate = np.array(
            [args.command_rise_rate_x, args.command_rise_rate_y, args.command_rise_rate_z], dtype=np.float32
        )
        self.command_fall_rate = np.array(
            [args.command_fall_rate_x, args.command_fall_rate_y, args.command_fall_rate_z], dtype=np.float32
        )

        walk_torque_limit_by_name = {
            joint_name: float(limit) for joint_name, limit in zip(self.policy_joints, self.user_torque_limit, strict=False)
        }
        self.head_hold_targets = {}
        for joint_name in HEAD_JOINT_NAMES:
            if joint_name in self.policy_joint_index:
                joint_idx = self.policy_joint_index[joint_name]
                self.head_hold_targets[joint_name] = (
                    float(self.default_angles[joint_idx]),
                    float(self.kps[joint_idx]),
                    float(self.kds[joint_idx]),
                )

        self.mimic = build_mimic_bundle(
            args=args,
            config_path=self.config_path,
            config=self.config,
            update_rate=self.update_rate,
            torque_limit_by_name=walk_torque_limit_by_name,
        )
        self.anchor_kinematics = None
        if self.mimic is not None:
            self.anchor_kinematics = RobotAnchorKinematics(self.mimic.robot_xml_path, self.mimic.anchor_body_name)
            print(
                f"[gamepad-sim2sim] loaded mimic policy from {self.mimic.policy_path} "
                f"with motion {self.mimic.motion.path}."
            )

    def wait_for_first_state(self):
        super().wait_for_first_state()
        if self.mimic is None:
            return

        motor_names = list(self.io.robot_state.motor_names)
        missing = [name for name in self.mimic.joint_names if name not in motor_names]
        if missing:
            raise RuntimeError(f"Mimic policy joints are missing from the simulator state: {missing}")
        self.mimic.robot_order_index = [motor_names.index(name) for name in self.mimic.joint_names]

    def _align_mimic_reference_heading(self, robot_state, imu_data):
        assert self.mimic is not None and self.anchor_kinematics is not None
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
        static_ref_anchor_quat_w = quat_multiply_wxyz(
            self.mimic.reference_yaw_offset_quat_w, ref_anchor_quat_w
        )

        robot_heading_quat_w = yaw_quat_from_quat_wxyz(robot_anchor_quat_w)
        ref_heading_quat_w = yaw_quat_from_quat_wxyz(static_ref_anchor_quat_w)
        dynamic_heading_offset_w = quat_multiply_wxyz(robot_heading_quat_w, quat_conjugate_wxyz(ref_heading_quat_w))
        active_offset_w = quat_multiply_wxyz(dynamic_heading_offset_w, self.mimic.reference_yaw_offset_quat_w)
        self.mimic.active_reference_yaw_offset_quat_w = normalize_quat_wxyz(active_offset_w).astype(np.float32)

        robot_yaw_deg = np.rad2deg(yaw_rad_from_quat_wxyz(robot_anchor_quat_w))
        ref_yaw_deg = np.rad2deg(yaw_rad_from_quat_wxyz(static_ref_anchor_quat_w))
        dynamic_yaw_deg = wrap_degrees(robot_yaw_deg - ref_yaw_deg)
        active_yaw_deg = np.rad2deg(yaw_rad_from_quat_wxyz(self.mimic.active_reference_yaw_offset_quat_w))
        print(
            "[gamepad-sim2sim] mimic heading aligned "
            f"robot_yaw={wrap_degrees(robot_yaw_deg):+.1f}deg "
            f"ref0_yaw={wrap_degrees(ref_yaw_deg):+.1f}deg "
            f"dynamic_offset={dynamic_yaw_deg:+.1f}deg "
            f"active_offset={wrap_degrees(active_yaw_deg):+.1f}deg."
        )

    def _switch_mode(self, mode_name: str, robot_state=None, imu_data=None):
        if mode_name == self.active_mode:
            return

        if mode_name == "mimic" and self.mimic is None:
            print("[gamepad-sim2sim] mimic mode is unavailable. Set --mimic-policy and --mimic-motion-file first.")
            return

        self.active_mode = mode_name
        if mode_name == "mimic":
            self.mimic_loop_count = 0
            self.mimic_playback_time_s = 0.0
            self.mimic.last_action.fill(0.0)
            if robot_state is not None and imu_data is not None:
                self._align_mimic_reference_heading(robot_state, imu_data)
        print(f"[gamepad-sim2sim] active mode -> {self.active_mode}")

    def _run_walk_step(self, robot_state, imu_data):
        sim_time = self.walk_loop_count / float(self.update_rate)
        self.last_action = self.walk_last_action

        if self.walk_loop_count % self.control_decimation == 0:
            obs = super().build_observation(robot_state, imu_data, sim_time)
            self.last_action = super().compute_action(obs)
            self.walk_last_action = self.last_action.copy()

        super().publish_command(robot_state)
        self.walk_loop_count += 1

    def _build_mimic_observation(self, robot_state, imu_data) -> np.ndarray:
        assert self.mimic is not None and self.anchor_kinematics is not None
        joint_pos = np.asarray(robot_state.q, dtype=np.float32)[self.mimic.robot_order_index]
        joint_vel = np.asarray(robot_state.dq, dtype=np.float32)[self.mimic.robot_order_index]
        motor_names = list(robot_state.motor_names)
        all_joint_pos = np.asarray(robot_state.q, dtype=np.float32)

        (
            self.mimic.current_frame_index,
            ref_joint_pos,
            ref_joint_vel,
            ref_anchor_quat_w,
        ) = self.mimic.motion.sample(self.mimic_playback_time_s)
        ref_anchor_quat_w = quat_multiply_wxyz(self.mimic.active_reference_yaw_offset_quat_w, ref_anchor_quat_w).astype(
            np.float32
        )

        robot_anchor_quat_w = self.anchor_kinematics.compute_anchor_quat(
            base_quat_wxyz=np.asarray(imu_data.quat, dtype=np.float32),
            joint_names=motor_names,
            joint_pos=all_joint_pos,
        )
        relative_anchor_quat_w = quat_multiply_wxyz(quat_conjugate_wxyz(robot_anchor_quat_w), ref_anchor_quat_w)

        raw_terms = {
            "command": np.concatenate([ref_joint_pos, ref_joint_vel], axis=0),
            "motion_anchor_ori_b": rotation_6d_from_quat_wxyz(relative_anchor_quat_w),
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
        assert self.mimic is not None
        with torch.inference_mode():
            obs_tensor = torch.from_numpy(observation).unsqueeze(0)
            action = self.mimic.policy(obs_tensor)
            if isinstance(action, (tuple, list)):
                action = action[0]
            action = action.detach().cpu().numpy().squeeze().astype(np.float32)
        return np.clip(action, -self.mimic.clip_actions, self.mimic.clip_actions)

    def _publish_mimic_command(self, robot_state):
        assert self.mimic is not None
        motor_names = list(robot_state.motor_names)
        motor_count = len(motor_names)
        full_joint_pos = np.asarray(robot_state.q, dtype=np.float32)
        full_joint_vel = np.asarray(robot_state.dq, dtype=np.float32)
        joint_pos = full_joint_pos[self.mimic.robot_order_index]
        joint_vel = full_joint_vel[self.mimic.robot_order_index]

        safe_scale = np.maximum(self.mimic.action_scale, 1e-6)
        safe_kp = np.maximum(self.mimic.kps, 1e-6)
        soft_torque_limit = 0.95

        action_min = joint_pos - self.mimic.default_joint_pos + (
            self.mimic.kds * joint_vel - self.mimic.user_torque_limit * soft_torque_limit
        ) / safe_kp
        action_max = joint_pos - self.mimic.default_joint_pos + (
            self.mimic.kds * joint_vel + self.mimic.user_torque_limit * soft_torque_limit
        ) / safe_kp
        limited_action = np.clip(self.mimic.last_action, action_min / safe_scale, action_max / safe_scale)
        desired_q_policy = limited_action * self.mimic.action_scale + self.mimic.default_joint_pos

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

    def _run_mimic_step(self, robot_state, imu_data, dt: float):
        assert self.mimic is not None
        if self.mimic_loop_count % self.mimic.control_decimation == 0:
            observation = self._build_mimic_observation(robot_state, imu_data)
            self.mimic.last_action = self._compute_mimic_action(observation)

        self._publish_mimic_command(robot_state)
        self.mimic_loop_count += 1
        if self.teleop.enabled:
            self.mimic_playback_time_s += dt

    def run(self):
        self.wait_for_first_state()
        rate = Rate(self.update_rate)
        t0 = time.perf_counter()
        dt = 1.0 / float(self.update_rate)

        while True:
            teleop_command = self.teleop.read_command()
            robot_state = copy.deepcopy(self.io.robot_state)
            imu_data = copy.deepcopy(self.io.imu_data)

            requested_mode = self.teleop.consume_mode_request()
            if requested_mode is not None:
                self._switch_mode(requested_mode, robot_state, imu_data)

            self.target_cmd = teleop_command
            self.cmd = slew_limit_vector(
                self.cmd, self.target_cmd, self.command_rise_rate, self.command_fall_rate, dt
            )

            if self.active_mode == "walk":
                self._run_walk_step(robot_state, imu_data)
            else:
                self._run_mimic_step(robot_state, imu_data, dt)

            self.loop_count += 1
            if self.loop_count % self.update_rate == 0:
                fps = self.loop_count / max(1e-6, (time.perf_counter() - t0))
                state = "ACTIVE" if self.teleop.enabled else "PAUSED"
                controller_name = self.teleop.controller.name if self.teleop.controller is not None else "none"
                if self.active_mode == "walk":
                    mode_status = (
                        f"cmd=({self.cmd[0]:+.2f}, {self.cmd[1]:+.2f}, {self.cmd[2]:+.2f}) "
                        f"target=({self.target_cmd[0]:+.2f}, {self.target_cmd[1]:+.2f}, {self.target_cmd[2]:+.2f})"
                    )
                else:
                    mode_status = (
                        f"frame={self.mimic.current_frame_index:04d}/{self.mimic.motion.total_frames:04d} "
                        f"motion={self.mimic.motion.path.name}"
                    )
                print(
                    f"[gamepad-sim2sim] fps={fps:.1f} state={state} mode={self.active_mode.upper()} "
                    f"controller={controller_name} {mode_status}",
                    end="\r",
                )
            rate.sleep()

    def close(self):
        self.teleop.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "LimX sdk-based sim2sim controller with SDL2 gamepad teleoperation for the official MuJoCo simulator. "
            "The controller starts in walk mode and can optionally switch to a BeyondMimic no-state-estimation "
            "policy with R1 + A."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/limx_flat_velocity_sdk.yaml",
        help="Path to sdk sim2sim yaml config.",
    )
    parser.add_argument("--policy", type=str, default=None, help="Optional override path to the walk TorchScript policy.")
    parser.add_argument(
        "--mimic-policy",
        type=str,
        default=None,
        help="Optional BeyondMimic no-state-estimation TorchScript policy. Enables mimic mode switching.",
    )
    parser.add_argument(
        "--mimic-policy-root",
        type=str,
        default=None,
        help="Optional root used to resolve a relative --mimic-policy path.",
    )
    parser.add_argument(
        "--mimic-motion-file",
        type=str,
        default=None,
        help="Motion npz used by mimic mode. If omitted, it is inferred from the training env.yaml when possible.",
    )
    parser.add_argument(
        "--mimic-robot-xml",
        type=str,
        default=None,
        help="MuJoCo xml used for mimic anchor forward kinematics.",
    )
    parser.add_argument(
        "--controller-index",
        type=int,
        default=None,
        help="SDL controller index. Defaults to the first SDL2-compatible controller.",
    )
    parser.add_argument(
        "--lin-vel-x",
        type=float,
        default=None,
        help="Compatibility option inherited from sdk_policy_controller.py. Ignored after teleop starts.",
    )
    parser.add_argument(
        "--lin-vel-y",
        type=float,
        default=None,
        help="Compatibility option inherited from sdk_policy_controller.py. Ignored after teleop starts.",
    )
    parser.add_argument(
        "--ang-vel-z",
        type=float,
        default=None,
        help="Compatibility option inherited from sdk_policy_controller.py. Ignored after teleop starts.",
    )
    parser.add_argument(
        "--start-enabled",
        action="store_true",
        help="Start the controller in active state. By default it starts paused and waits for START.",
    )
    parser.add_argument(
        "--max-lin-vel-x",
        type=float,
        default=0.4,
        help="Maximum forward/backward command in m/s.",
    )
    parser.add_argument(
        "--max-lin-vel-y",
        type=float,
        default=0.2,
        help="Maximum lateral command in m/s.",
    )
    parser.add_argument(
        "--max-ang-vel-z",
        type=float,
        default=0.5,
        help="Maximum yaw-rate command in rad/s.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.12,
        help="Normalized stick deadzone in [0, 1).",
    )
    parser.add_argument(
        "--expo",
        type=float,
        default=1.5,
        help="Exponent for stick shaping. Values > 1 soften small motions.",
    )
    parser.add_argument(
        "--slow-factor",
        type=float,
        default=0.35,
        help="Scale applied while holding LB.",
    )
    parser.add_argument(
        "--reconnect-interval-s",
        type=float,
        default=1.0,
        help="How often to retry controller connection when disconnected.",
    )
    parser.add_argument(
        "--command-rise-rate-x",
        type=float,
        default=1.6,
        help="Forward command rise-rate limit in m/s^2 for policy input shaping.",
    )
    parser.add_argument(
        "--command-rise-rate-y",
        type=float,
        default=1.0,
        help="Lateral command rise-rate limit in m/s^2 for policy input shaping.",
    )
    parser.add_argument(
        "--command-rise-rate-z",
        type=float,
        default=2.0,
        help="Yaw command rise-rate limit in rad/s^2 for policy input shaping.",
    )
    parser.add_argument(
        "--command-fall-rate-x",
        type=float,
        default=0.7,
        help="Forward command fall-rate limit in m/s^2 to avoid abrupt stop-induced pitching.",
    )
    parser.add_argument(
        "--command-fall-rate-y",
        type=float,
        default=0.9,
        help="Lateral command fall-rate limit in m/s^2 for policy input shaping.",
    )
    parser.add_argument(
        "--command-fall-rate-z",
        type=float,
        default=1.5,
        help="Yaw command fall-rate limit in rad/s^2 for policy input shaping.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    controller = GamepadPolicyController(parse_args())
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n[gamepad-sim2sim] stopped by user.")
    finally:
        controller.close()
