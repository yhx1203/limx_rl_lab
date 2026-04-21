from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch


class MotionLoader:
    """Load BeyondMimic-style motion npz files and sample interpolated states."""

    def __init__(self, motion_file: str, joint_names: Sequence[str], body_names: Sequence[str], device: str) -> None:
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Invalid motion file path: {motion_file}")

        data = np.load(motion_file)

        self.device = device
        self.motion_file = motion_file
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)

        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=self.device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=self.device)
        self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=self.device)
        self.body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=self.device)
        self.body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=self.device)
        self.body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=self.device)

        fps = np.asarray(data["fps"]).reshape(-1)
        self.dt = 1.0 / float(fps[0])
        self.num_frames = self.joint_pos.shape[0]
        self.duration = self.dt * max(self.num_frames - 1, 1)

        if self.joint_pos.shape[1] != len(self.joint_names) or self.joint_vel.shape[1] != len(self.joint_names):
            raise ValueError(
                f"Motion joint dimension ({self.joint_pos.shape[1]}) does not match expected joints "
                f"({len(self.joint_names)})."
            )
        if self.body_pos_w.shape[1] != len(self.body_names):
            raise ValueError(
                f"Motion body dimension ({self.body_pos_w.shape[1]}) does not match expected bodies "
                f"({len(self.body_names)})."
            )

        print(f"[INFO]: Motion loaded: {motion_file}, duration={self.duration:.3f}s, frames={self.num_frames}")

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta).unsqueeze(-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1.0 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_qx = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_qy = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_qz = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_qw = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_qw, new_qx, new_qy, new_qz], dim=len(new_qw.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1.0, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase = np.clip(times / self.duration, 0.0, 1.0)
        index_0 = (phase * (self.num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
        return index_0, index_1, blend

    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        duration = self.duration if duration is None else duration
        if duration > self.duration:
            raise ValueError(f"Requested duration {duration} exceeds motion duration {self.duration}.")
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self, num_samples: int, times: np.ndarray | None = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.joint_pos, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.joint_vel, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_pos_w, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_quat_w, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_lin_vel_w, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_ang_vel_w, blend=blend, start=index_0, end=index_1),
        )
