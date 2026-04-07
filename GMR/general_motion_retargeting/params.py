import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

ROBOT_XML_DICT = {
    "hu_d04": ASSET_ROOT / "HU_D04_description" / "xml" / "HU_D04_01_gmr.xml",
}

IK_CONFIG_DICT = {
    "bvh_lafan1":{
        "hu_d04": IK_CONFIG_ROOT / "bvh_lafan1_to_hu_d04.json",
    },
}


ROBOT_BASE_DICT = {
    "hu_d04": "base_link",
}

VIEWER_CAM_DISTANCE_DICT = {
    "hu_d04": 2.5,
}

ROBOT_PR_SPACE_JOINTS_DICT = {
    "hu_d04": [
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
    ],
}
