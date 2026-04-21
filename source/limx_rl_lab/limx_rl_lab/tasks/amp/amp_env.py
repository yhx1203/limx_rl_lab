from __future__ import annotations

import re

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .amp_env_cfg import AmpEnvCfg
from .motion_loader import MotionLoader


class AmpEnv(DirectRLEnv):
    """Direct AMP environment for LimX humanoid motions."""

    cfg: AmpEnvCfg

    def __init__(self, cfg: AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_ids = torch.tensor(
            self.robot.find_joints(self.cfg.joint_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.root_body_index = self.robot.body_names.index(self.cfg.root_body)
        self.reference_body_index = self.robot.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.body_names.index(name) for name in self.cfg.key_body_names]

        self.action_offset = self.robot.data.default_joint_pos[0, self.joint_ids].clone()
        self.action_scale = torch.tensor(
            [self._resolve_action_scale(name) for name in self.cfg.joint_names], dtype=torch.float32, device=self.device
        )
        self.action_lower_limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids, 0]
        self.action_upper_limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids, 1]

        self._motion_loader = MotionLoader(
            motion_file=self.cfg.motion_file,
            joint_names=self.cfg.joint_names,
            body_names=self.robot.body_names,
            device=self.device,
        )

        amp_obs_size = self.joint_ids.numel() * 2 + 1 + 6 + 3 + 3 + len(self.key_body_indexes) * 3
        policy_obs_size = amp_obs_size + self.cfg.command_dim
        if self.cfg.observation_space != policy_obs_size:
            raise ValueError(
                f"Configured observation size ({self.cfg.observation_space}) does not match {policy_obs_size}."
            )
        if self.cfg.amp_observation_space != amp_obs_size:
            raise ValueError(
                f"Configured AMP observation size ({self.cfg.amp_observation_space}) does not match {amp_obs_size}."
            )

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )
        self.actions = torch.zeros((self.num_envs, self.joint_ids.numel()), dtype=torch.float32, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros((self.num_envs, self.cfg.command_dim), dtype=torch.float32, device=self.device)
        self._command_steps_left = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _resolve_action_scale(self, joint_name: str) -> float:
        """Resolve per-joint action scale from either exact-name or regex keyed configs."""
        if joint_name in self.cfg.action_scale:
            return float(self.cfg.action_scale[joint_name])

        for pattern, value in self.cfg.action_scale.items():
            if re.fullmatch(pattern, joint_name):
                return float(value)

        available = ", ".join(list(self.cfg.action_scale.keys())[:8])
        raise KeyError(f"No action scale found for joint '{joint_name}'. Available keys include: {available}")

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions.copy_(self.actions)
        self._update_commands()
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        targets = self.action_offset + self.action_scale * self.actions
        targets = torch.clamp(targets, self.action_lower_limits, self.action_upper_limits)
        self.robot.set_joint_position_target(targets, joint_ids=self.joint_ids)

    def _get_observations(self) -> dict:
        obs = compute_obs(
            self.robot.data.joint_pos[:, self.joint_ids],
            self.robot.data.joint_vel[:, self.joint_ids],
            self.robot.data.body_pos_w[:, self.reference_body_index],
            self.robot.data.body_quat_w[:, self.reference_body_index],
            self.robot.data.body_lin_vel_w[:, self.reference_body_index],
            self.robot.data.body_ang_vel_w[:, self.reference_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = obs

        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
        policy_obs = torch.cat((obs, self.commands), dim=-1)
        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        z_vel_penalty = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        ang_vel_xy_penalty = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
        action_rate_penalty = torch.sum(torch.square(self.actions - self.previous_actions), dim=1)

        lin_vel_reward = torch.exp(-lin_vel_error / (self.cfg.lin_vel_reward_std**2)) * self.cfg.lin_vel_reward_scale
        ang_vel_reward = torch.exp(-ang_vel_error / (self.cfg.ang_vel_reward_std**2)) * self.cfg.ang_vel_reward_scale

        return (
            lin_vel_reward
            + ang_vel_reward
            + z_vel_penalty * self.cfg.z_vel_penalty_scale
            + ang_vel_xy_penalty * self.cfg.ang_vel_xy_penalty_scale
            + action_rate_penalty * self.cfg.action_rate_penalty_scale
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.reference_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel, amp_obs = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            root_state, joint_pos, joint_vel, amp_obs = self._reset_strategy_random(
                env_ids, start="start" in self.cfg.reset_strategy
            )
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        joint_pos_to_write = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel_to_write = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos_to_write[:, self.joint_ids] = joint_pos
        joint_vel_to_write[:, self.joint_ids] = joint_vel

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos_to_write, joint_vel_to_write, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos_to_write, env_ids=env_ids)

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self._sample_commands(env_ids)
        self.amp_observation_buffer[env_ids] = amp_obs.view(len(env_ids), self.cfg.num_amp_observations, -1)

    def _reset_strategy_default(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        joint_pos = self.robot.data.default_joint_pos[env_ids][:, self.joint_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids][:, self.joint_ids].clone()
        amp_obs = self.collect_reference_motions(len(env_ids), np.zeros(len(env_ids)))
        return root_state, joint_pos, joint_vel, amp_obs

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = len(env_ids)
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = self._motion_loader.sample(
            num_samples=num_samples, times=times
        )

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_pos_w[:, self.root_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += self.cfg.root_height_offset
        root_state[:, 3:7] = body_quat_w[:, self.root_body_index]
        root_state[:, 7:10] = body_lin_vel_w[:, self.root_body_index]
        root_state[:, 10:13] = body_ang_vel_w[:, self.root_body_index]

        amp_obs = self.collect_reference_motions(num_samples, times)
        return root_state, joint_pos, joint_vel, amp_obs

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)

        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()

        joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = self._motion_loader.sample(
            num_samples=num_samples, times=times
        )

        amp_obs = compute_obs(
            joint_pos,
            joint_vel,
            body_pos_w[:, self.reference_body_index],
            body_quat_w[:, self.reference_body_index],
            body_lin_vel_w[:, self.reference_body_index],
            body_ang_vel_w[:, self.reference_body_index],
            body_pos_w[:, self.key_body_indexes],
        )
        return amp_obs.view(-1, self.amp_observation_size)

    def _update_commands(self):
        self._command_steps_left -= 1
        env_ids = torch.nonzero(self._command_steps_left <= 0, as_tuple=False).flatten()
        if env_ids.numel() > 0:
            self._sample_commands(env_ids)

    def _sample_commands(self, env_ids: torch.Tensor):
        num_envs = len(env_ids)
        low = torch.tensor(
            [
                self.cfg.command_lin_vel_x_range[0],
                self.cfg.command_lin_vel_y_range[0],
                self.cfg.command_ang_vel_z_range[0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        high = torch.tensor(
            [
                self.cfg.command_lin_vel_x_range[1],
                self.cfg.command_lin_vel_y_range[1],
                self.cfg.command_ang_vel_z_range[1],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        commands = low + torch.rand((num_envs, self.cfg.command_dim), device=self.device) * (high - low)

        if self.cfg.rel_standing_envs > 0.0:
            standing_mask = torch.rand((num_envs,), device=self.device) < self.cfg.rel_standing_envs
            commands[standing_mask] = 0.0

        self.commands[env_ids] = commands
        self._command_steps_left[env_ids] = self._sample_command_durations(num_envs)

    def _sample_command_durations(self, num_envs: int) -> torch.Tensor:
        low = max(1, int(round(self.cfg.command_resampling_time_range_s[0] / self.step_dt)))
        high = max(low, int(round(self.cfg.command_resampling_time_range_s[1] / self.step_dt)))
        if low == high:
            return torch.full((num_envs,), low, dtype=torch.long, device=self.device)
        return torch.randint(low, high + 1, (num_envs,), dtype=torch.long, device=self.device)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1.0
    ref_normal[..., -1] = 1.0
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    key_body_pos_w: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            joint_pos,
            joint_vel,
            root_pos_w[:, 2:3],
            quaternion_to_tangent_and_normal(root_quat_w),
            root_lin_vel_w,
            root_ang_vel_w,
            (key_body_pos_w - root_pos_w.unsqueeze(-2)).view(key_body_pos_w.shape[0], -1),
        ),
        dim=-1,
    )
