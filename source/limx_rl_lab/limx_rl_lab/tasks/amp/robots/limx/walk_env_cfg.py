from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from isaaclab.utils import configclass

from ...amp_env_cfg import AmpEnvCfg
from ....beyondmimic.robots.limx.robot_cfg import (
    LIMX_OLI_CFG,
    OLI_ACTION_SCALE,
    OLI_ANCHOR_BODY_NAME,
    OLI_PR_JOINT_NAMES,
)


_REPO_ROOT = Path(__file__).resolve().parents[7]
_MOTION_FILE = _REPO_ROOT / "motions" / "hu_d04_walk1_subject1_beyondmimic" / "motion.npz"
_KEY_BODY_NAMES = [
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
_AMP_OBS_SIZE = len(OLI_PR_JOINT_NAMES) * 2 + 1 + 6 + 3 + 3 + len(_KEY_BODY_NAMES) * 3
_POLICY_OBS_SIZE = _AMP_OBS_SIZE + 3


@configclass
class OliWalkAmpEnvCfg(AmpEnvCfg):
    """OLI walk AMP task driven by a BeyondMimic motion clip."""

    robot = LIMX_OLI_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    motion_file = str(_MOTION_FILE)
    joint_names = OLI_PR_JOINT_NAMES
    action_scale = OLI_ACTION_SCALE

    root_body = "base_link"
    reference_body = OLI_ANCHOR_BODY_NAME
    key_body_names = _KEY_BODY_NAMES
    reset_strategy = "random"

    command_lin_vel_x_range = (0.25, 0.45)
    command_lin_vel_y_range = (0.0, 0.0)
    command_ang_vel_z_range = (0.0, 0.0)
    command_resampling_time_range_s = (10.0, 10.0)
    rel_standing_envs = 0.0

    observation_space = _POLICY_OBS_SIZE
    amp_observation_space = _AMP_OBS_SIZE
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(OLI_PR_JOINT_NAMES),), dtype=np.float32)


@configclass
class OliWalkAmpPlayEnvCfg(OliWalkAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.command_lin_vel_x_range = (0.35, 0.35)
