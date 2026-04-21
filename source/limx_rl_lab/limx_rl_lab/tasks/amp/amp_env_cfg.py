from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import SpaceType
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


@configclass
class AmpEnvCfg(DirectRLEnvCfg):
    """Base config for direct AMP tasks driven by motion npz files."""

    episode_length_s: float = 10.0
    decimation: int = 4
    is_finite_horizon: bool = False

    observation_space: SpaceType = MISSING
    action_space: SpaceType = MISSING
    state_space: SpaceType = 0
    num_amp_observations: int = 2
    amp_observation_space: int = MISSING
    command_dim: int = 3

    early_termination: bool = True
    termination_height: float = 0.45

    motion_file: str = MISSING
    joint_names: list[str] = MISSING
    action_scale: dict[str, float] = MISSING

    root_body: str = "base_link"
    reference_body: str = "base_link"
    key_body_names: list[str] = [
        "left_wrist_roll_link",
        "right_wrist_roll_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    reset_strategy: str = "random"
    root_height_offset: float = 0.0

    command_resampling_time_range_s: tuple[float, float] = (10.0, 10.0)
    command_lin_vel_x_range: tuple[float, float] = (0.25, 0.45)
    command_lin_vel_y_range: tuple[float, float] = (0.0, 0.0)
    command_ang_vel_z_range: tuple[float, float] = (0.0, 0.0)
    rel_standing_envs: float = 0.0

    lin_vel_reward_std: float = 0.35
    ang_vel_reward_std: float = 0.25
    lin_vel_reward_scale: float = 1.0
    ang_vel_reward_scale: float = 0.25
    z_vel_penalty_scale: float = -0.25
    ang_vel_xy_penalty_scale: float = -0.05
    action_rate_penalty_scale: float = -0.01

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    robot: ArticulationCfg = MISSING
