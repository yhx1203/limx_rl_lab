"""Convert a BeyondMimic CSV motion into the npz format used by the tracking task."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Load Trimesh/SciPy/Numpy modules before Isaac Sim prepends its pip_prebundle path.
import numpy.lib.recfunctions  # noqa: F401
import numpy.random  # noqa: F401
import scipy.spatial  # noqa: F401
import trimesh  # noqa: F401

from isaaclab.app import AppLauncher

# Launch Isaac Sim Simulator first.
parser = argparse.ArgumentParser(description="Convert a retargeted BeyondMimic CSV motion to npz.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input motion CSV file.")
parser.add_argument("--input_fps", type=int, default=30, help="FPS of the input motion.")
parser.add_argument("--robot", choices=("oli",), default="oli", help="Robot model used by the retargeted motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Frame range: START END, both inclusive, 1-indexed. Defaults to all frames.",
)
parser.add_argument("--output_name", type=str, required=True, help="Motion artifact/name stem.")
parser.add_argument("--output_fps", type=int, default=50, help="FPS of the output motion.")
parser.add_argument("--output_dir", type=str, default="motions", help="Directory used for local npz output.")
parser.add_argument("--skip_wandb", action="store_true", help="Only save locally; do not upload to wandb.")
parser.add_argument("--wandb_project", type=str, default="csv_to_npz", help="wandb project for artifact upload.")
parser.add_argument("--wandb_artifact_type", type=str, default="motions", help="wandb artifact type.")
parser.add_argument(
    "--wandb_registry_prefix",
    type=str,
    default="wandb-registry-motions",
    help="wandb registry prefix used when linking the artifact.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from limx_rl_lab.tasks.beyondmimic.robots.limx.robot_cfg import (
    LIMX_OLI_CFG,
    OLI_JOINT_SDK_NAMES,
    OLI_PR_JOINT_NAMES,
)

ROBOT_CFG = LIMX_OLI_CFG
ROBOT_JOINT_NAMES = OLI_PR_JOINT_NAMES
ROBOT_INPUT_JOINT_NAMES = OLI_JOINT_SDK_NAMES


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Scene used to replay a motion and sample all body states from the robot."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        joint_names: list[str],
        input_joint_names: list[str],
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.joint_names = joint_names
        self.input_joint_names = input_joint_names
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Load root pose and joint positions from the CSV file."""
        if self.frame_range is None:
            motion_np = np.loadtxt(self.motion_file, delimiter=",")
        else:
            motion_np = np.loadtxt(
                self.motion_file,
                delimiter=",",
                skiprows=self.frame_range[0] - 1,
                max_rows=self.frame_range[1] - self.frame_range[0] + 1,
            )
        motion = torch.from_numpy(np.atleast_2d(motion_np)).to(torch.float32).to(self.device)
        self.motion_base_pos_input = motion[:, :3]
        self.motion_base_rot_input = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
        self.motion_dof_pos_input = motion[:, 7:]

        if self.motion_dof_pos_input.shape[1] == len(self.input_joint_names):
            joint_indexes = [self.input_joint_names.index(name) for name in self.joint_names]
            self.motion_dof_pos_input = self.motion_dof_pos_input[:, joint_indexes]
        elif self.motion_dof_pos_input.shape[1] != len(self.joint_names):
            raise ValueError(
                f"Input motion has {self.motion_dof_pos_input.shape[1]} joints, but {args_cli.robot} expects either "
                f"{len(self.joint_names)} exported joints or {len(self.input_joint_names)} full input joints."
            )

        self.input_frames = motion.shape[0]
        if self.input_frames < 3:
            raise ValueError("At least 3 input frames are required to compute velocities.")
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"[INFO]: Motion loaded: {self.motion_file}, duration={self.duration:.3f}s, frames={self.input_frames}")

    def _interpolate_motion(self):
        """Interpolate the input motion to the output FPS."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_pos = self._lerp(
            self.motion_base_pos_input[index_0],
            self.motion_base_pos_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rot = self._slerp(
            self.motion_base_rot_input[index_0],
            self.motion_base_rot_input[index_1],
            blend,
        )
        self.motion_dof_pos = self._lerp(
            self.motion_dof_pos_input[index_0],
            self.motion_dof_pos_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"[INFO]: Motion interpolated: input_fps={self.input_fps}, output_fps={self.output_fps}, "
            f"output_frames={self.output_frames}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vel = torch.gradient(self.motion_base_pos, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vel = torch.gradient(self.motion_dof_pos, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vel = self._so3_derivative(self.motion_base_rot, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (
            self.motion_base_pos[self.current_idx : self.current_idx + 1],
            self.motion_base_rot[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vel[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vel[self.current_idx : self.current_idx + 1],
            self.motion_dof_pos[self.current_idx : self.current_idx + 1],
            self.motion_dof_vel[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        done = self.current_idx >= self.output_frames
        if done:
            self.current_idx = 0
        return state, done


def _save_motion(log: dict[str, list[np.ndarray] | list[int]]) -> Path:
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[key] = np.stack(log[key], axis=0)

    output_path = Path(args_cli.output_dir).expanduser() / args_cli.output_name / "motion.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **log)
    print(f"[INFO]: Motion saved locally: {output_path}")
    return output_path


def _upload_motion(output_path: Path):
    if args_cli.skip_wandb:
        return

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install wandb or rerun with --skip_wandb to only save the local motion.npz."
        ) from exc

    run = wandb.init(project=args_cli.wandb_project, name=args_cli.output_name)
    print(f"[INFO]: Logging motion to wandb: {args_cli.output_name}")
    artifact = run.log_artifact(
        artifact_or_path=str(output_path),
        name=args_cli.output_name,
        type=args_cli.wandb_artifact_type,
    )
    target_path = f"{args_cli.wandb_registry_prefix}/{args_cli.output_name}"
    run.link_artifact(artifact=artifact, target_path=target_path)
    print(f"[INFO]: Motion linked to wandb registry: {target_path}")
    run.finish()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        joint_names=joint_names,
        input_joint_names=ROBOT_INPUT_JOINT_NAMES,
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    if motion.motion_dof_pos.shape[1] != len(robot_joint_indexes):
        raise ValueError(
            f"Input motion has {motion.motion_dof_pos.shape[1]} joints, but {args_cli.robot} expects "
            f"{len(robot_joint_indexes)} joints: {joint_names}"
        )

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            done,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        log["joint_pos"].append(robot.data.joint_pos[0, robot_joint_indexes].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, robot_joint_indexes].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if done:
            output_path = _save_motion(log)
            _upload_motion(output_path)
            break


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("[INFO]: Setup complete.")
    run_simulator(sim, scene, joint_names=ROBOT_JOINT_NAMES)


if __name__ == "__main__":
    main()
    simulation_app.close()
