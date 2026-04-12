from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ...agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from ...beyondmimic_env_cfg import BeyondMimicEnvCfg
from .robot_cfg import (
    LIMX_OLI_CFG,
    OLI_ACTION_SCALE,
    OLI_ANCHOR_BODY_NAME,
    OLI_PR_JOINT_NAMES,
    OLI_TRACKED_BODY_NAMES,
)


@configclass
class OliFlatEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        def controlled_joint_cfg():
            return SceneEntityCfg("robot", joint_names=OLI_PR_JOINT_NAMES, preserve_order=True)

        self.scene.robot = LIMX_OLI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.joint_pos.joint_names = OLI_PR_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = OLI_ACTION_SCALE

        self.commands.motion.anchor_body_name = OLI_ANCHOR_BODY_NAME
        self.commands.motion.body_names = OLI_TRACKED_BODY_NAMES
        self.commands.motion.joint_names = OLI_PR_JOINT_NAMES

        self.observations.policy.joint_pos.params = {"asset_cfg": controlled_joint_cfg()}
        self.observations.policy.joint_vel.params = {"asset_cfg": controlled_joint_cfg()}
        self.observations.critic.joint_pos.params = {"asset_cfg": controlled_joint_cfg()}
        self.observations.critic.joint_vel.params = {"asset_cfg": controlled_joint_cfg()}

        self.events.add_joint_default_pos = None
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=OLI_ANCHOR_BODY_NAME)
        self.rewards.joint_limit.params["asset_cfg"] = controlled_joint_cfg()

        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                (
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    r"(?!left_wrist_roll_link$)(?!right_wrist_roll_link$)"
                    r"(?!left_hand_contact$)(?!right_hand_contact$).+$"
                )
            ],
        )
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
        ]


@configclass
class OliFlatWoStateEstimationEnvCfg(OliFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class OliFlatLowFreqEnvCfg(OliFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


@configclass
class OliFlatPlayEnvCfg(OliFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.motion.debug_vis = True
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class OliFlatWoStateEstimationPlayEnvCfg(OliFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.motion.debug_vis = True
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class OliFlatLowFreqPlayEnvCfg(OliFlatLowFreqEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.motion.debug_vis = True
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
