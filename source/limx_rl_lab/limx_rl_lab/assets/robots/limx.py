"""Configuration for LimX robots."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass


_EXTENSION_DIR = Path(__file__).resolve().parents[3]
_LIMX_HU_D04_DESCRIPTION_DIR = _EXTENSION_DIR / "data" / "Robots" / "limx" / "HU_D04_description"


@configclass
class LimxArticulationCfg(ArticulationCfg):
    """Configuration for LimX articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


LIMX_HU_D04_01_CFG = LimxArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_LIMX_HU_D04_DESCRIPTION_DIR / "usd" / "HU_D04_01.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.93),
        joint_pos={
            # Legs
            ".*_hip_pitch_joint": -0.25,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.55,
            ".*_ankle_pitch_joint": -0.30,
            ".*_ankle_roll_joint": 0.0,
            # Waist
            "waist_.*_joint": 0.0,
            # Head
            "head_.*_joint": 0.0,
            # Arms
            ".*_shoulder_pitch_joint": 0.10,
            "left_shoulder_roll_joint": 0.10,
            "right_shoulder_roll_joint": -0.10,
            "left_shoulder_yaw_joint": -0.20,
            "right_shoulder_yaw_joint": 0.20,
            ".*_elbow_joint": -0.20,
            ".*_wrist_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=140.0,
            velocity_limit_sim=5.0,
            stiffness={
                ".*_hip_pitch_joint": 220.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_yaw_joint": 200.0,
                ".*_knee_joint": 240.0,
            },
            damping={
                ".*_hip_pitch_joint": 6.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_yaw_joint": 5.0,
                ".*_knee_joint": 6.0,
            },
            armature=0.14125,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=42.0,
            velocity_limit_sim=13.6,
            stiffness={
                ".*_ankle_pitch_joint": 60.0,
                ".*_ankle_roll_joint": 60.0,
            },
            damping={
                ".*_ankle_pitch_joint": 1.5,
                ".*_ankle_roll_joint": 1.5,
            },
            armature=0.1845504,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=42.0,
            velocity_limit_sim=13.6,
            stiffness={
                "waist_yaw_joint": 120.0,
                "waist_roll_joint": 90.0,
                "waist_pitch_joint": 90.0,
            },
            damping={
                "waist_yaw_joint": 3.0,
                "waist_roll_joint": 2.5,
                "waist_pitch_joint": 2.5,
            },
            armature=0.1845504,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=42.0,
            velocity_limit_sim=19.6,
            stiffness={
                ".*_shoulder_pitch_joint": 80.0,
                ".*_shoulder_roll_joint": 80.0,
                ".*_shoulder_yaw_joint": 70.0,
                ".*_elbow_joint": 70.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.5,
                ".*_shoulder_roll_joint": 2.5,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
            },
            armature=0.0886706,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=19.0,
            velocity_limit_sim=13.0,
            stiffness=30.0,
            damping=1.0,
            armature=0.0153218,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim=19.0,
            velocity_limit_sim=13.0,
            stiffness=20.0,
            damping=0.8,
            armature=0.0153218,
        ),
    },
    joint_sdk_names=[
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
        "head_yaw_joint",
        "head_pitch_joint",
    ],
)
