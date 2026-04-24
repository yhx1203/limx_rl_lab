import gymnasium as gym

gym.register(
    id="LimX-HU-D04-01-Flat-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "limx_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="LimX-HU-D04-01-Flat-Velocity-No-Gait-Phase",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_no_gait_env_cfg:RobotNoGaitPhaseEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_no_gait_env_cfg:RobotNoGaitPhasePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "limx_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="LimX-HU-D04-01-Rough-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RoughRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rough_env_cfg:RoughRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "limx_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)
