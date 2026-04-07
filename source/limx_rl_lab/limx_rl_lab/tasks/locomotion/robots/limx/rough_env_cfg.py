import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from limx_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import (
    CurriculumCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
    RobotSceneCfg,
)


@configclass
class RoughRobotSceneCfg(RobotSceneCfg):
    """Rough-terrain scene for the LimX humanoid robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )


@configclass
class RoughObservationsCfg(ObservationsCfg):
    """Asymmetric observations: terrain height scans are critic-only."""

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RoughRewardsCfg(RewardsCfg):
    """Reward terms tuned for rough-terrain velocity tracking."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class RoughCurriculumCfg(CurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class RoughRobotEnvCfg(RobotEnvCfg):
    """Rough-terrain locomotion velocity-tracking environment."""

    scene: RoughRobotSceneCfg = RoughRobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: RoughObservationsCfg = RoughObservationsCfg()
    rewards: RoughRewardsCfg = RoughRewardsCfg()
    curriculum: RoughCurriculumCfg = RoughCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.rewards.base_height = None
        self.rewards.action_rate.weight = -0.03
        self.rewards.feet_slide.weight = -0.25

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)


@configclass
class RoughRobotPlayEnvCfg(RoughRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
