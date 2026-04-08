"""Replay a BeyondMimic motion npz with the LimX OLI robot."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Load Trimesh/SciPy/Numpy modules before Isaac Sim prepends its pip_prebundle path.
import numpy.lib.recfunctions  # noqa: F401
import numpy.random  # noqa: F401
import scipy.spatial  # noqa: F401
import trimesh  # noqa: F401
import torch

from isaaclab.app import AppLauncher

# Launch Isaac Sim Simulator first.
parser = argparse.ArgumentParser(description="Replay a converted BeyondMimic motion.")
parser.add_argument("--registry_name", type=str, default=None, help="wandb motion artifact path.")
parser.add_argument("--motion_file", type=str, default=None, help="Local motion.npz path.")
parser.add_argument("--robot", choices=("oli",), default="oli", help="Robot model used by the motion npz.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of replay environments.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.registry_name is None and args_cli.motion_file is None:
    parser.error("one of --registry_name or --motion_file is required")
if args_cli.registry_name is not None and args_cli.motion_file is not None:
    parser.error("provide only one of --registry_name or --motion_file")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from limx_rl_lab.tasks.beyondmimic.mdp import MotionLoader
from limx_rl_lab.tasks.beyondmimic.robots.limx.robot_cfg import LIMX_OLI_CFG, OLI_PR_JOINT_NAMES

ROBOT_CFG = LIMX_OLI_CFG
ROBOT_JOINT_NAMES = OLI_PR_JOINT_NAMES


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Scene used to replay a converted motion."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _resolve_motion_file() -> Path:
    if args_cli.motion_file is not None:
        motion_file = Path(args_cli.motion_file).expanduser()
        if not motion_file.is_file():
            raise FileNotFoundError(f"Motion file does not exist: {motion_file}")
        return motion_file.resolve()

    registry_name = args_cli.registry_name
    if ":" not in registry_name:
        registry_name += ":latest"

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install wandb or replay a local file with --motion_file /path/to/motion.npz."
        ) from exc

    artifact = wandb.Api().artifact(registry_name)
    motion_file = Path(artifact.download()) / "motion.npz"
    if not motion_file.is_file():
        raise FileNotFoundError(f"Motion artifact does not contain motion.npz: {motion_file}")
    return motion_file.resolve()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion_file: Path):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    print(f"[INFO]: Replaying motion file: {motion_file}")

    motion = MotionLoader(str(motion_file), torch.tensor([0], dtype=torch.long, device=sim.device), sim.device)
    robot_joint_indexes = robot.find_joints(ROBOT_JOINT_NAMES, preserve_order=True)[0]
    if motion.joint_pos.shape[1] != len(robot_joint_indexes):
        raise ValueError(
            f"Motion npz has {motion.joint_pos.shape[1]} joints, but {args_cli.robot} expects "
            f"{len(robot_joint_indexes)} joints: {ROBOT_JOINT_NAMES}"
        )

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion.joint_pos[time_steps]
        joint_vel[:, robot_joint_indexes] = motion.joint_vel[time_steps]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    motion_file = _resolve_motion_file()
    motion_data = np.load(motion_file)
    fps = float(np.atleast_1d(motion_data["fps"])[0])

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayMotionsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    sim.reset()
    print("[INFO]: Setup complete.")
    run_simulator(sim, scene, motion_file)


if __name__ == "__main__":
    main()
    simulation_app.close()
