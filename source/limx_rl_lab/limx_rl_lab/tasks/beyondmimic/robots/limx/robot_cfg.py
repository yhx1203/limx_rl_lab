"""LimX OLI metadata used by the BeyondMimic tracking task."""

from limx_rl_lab.assets.robots.limx import LIMX_HU_D04_01_CFG

OLI_HEAD_JOINT_NAMES = [
    "head_yaw_joint",
    "head_pitch_joint",
]

OLI_PR_JOINT_NAMES = [
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

OLI_JOINT_SDK_NAMES = OLI_PR_JOINT_NAMES + OLI_HEAD_JOINT_NAMES

assert len(OLI_PR_JOINT_NAMES) == 29
assert len(OLI_JOINT_SDK_NAMES) == 31

OLI_ANCHOR_BODY_NAME = "waist_pitch_link"

OLI_TRACKED_BODY_NAMES = [
    "base_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    OLI_ANCHOR_BODY_NAME,
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
]

LIMX_OLI_CFG = LIMX_HU_D04_01_CFG

OLI_ACTION_SCALE = {}
for actuator in LIMX_HU_D04_01_CFG.actuators.values():
    efforts = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    joint_names = actuator.joint_names_expr
    if not isinstance(efforts, dict):
        efforts = {name: efforts for name in joint_names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in joint_names}
    for name in joint_names:
        if name in OLI_HEAD_JOINT_NAMES:
            continue
        if name in efforts and name in stiffness and stiffness[name]:
            OLI_ACTION_SCALE[name] = 0.25 * efforts[name] / stiffness[name]
