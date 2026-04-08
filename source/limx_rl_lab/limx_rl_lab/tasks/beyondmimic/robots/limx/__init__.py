import gymnasium as gym

_TASK_PACKAGE = "limx_rl_lab.tasks.beyondmimic.robots.limx"

gym.register(
    id="LimX-HU-D04-01-Flat-BeyondMimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_TASK_PACKAGE}.flat_env_cfg:OliFlatEnvCfg",
        "play_env_cfg_entry_point": f"{_TASK_PACKAGE}.flat_env_cfg:OliFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "limx_rl_lab.tasks.beyondmimic.agents.rsl_rl_ppo_cfg:OliFlatPPORunnerCfg",
    },
)

gym.register(
    id="LimX-HU-D04-01-Flat-BeyondMimic-Wo-State-Estimation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_TASK_PACKAGE}.flat_env_cfg:OliFlatWoStateEstimationEnvCfg",
        "play_env_cfg_entry_point": f"{_TASK_PACKAGE}.flat_env_cfg:OliFlatWoStateEstimationPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "limx_rl_lab.tasks.beyondmimic.agents.rsl_rl_ppo_cfg:OliFlatPPORunnerCfg",
    },
)

gym.register(
    id="LimX-HU-D04-01-Flat-BeyondMimic-Low-Freq",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_TASK_PACKAGE}.flat_env_cfg:OliFlatLowFreqEnvCfg",
        "play_env_cfg_entry_point": f"{_TASK_PACKAGE}.flat_env_cfg:OliFlatLowFreqPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "limx_rl_lab.tasks.beyondmimic.agents.rsl_rl_ppo_cfg:OliFlatLowFreqPPORunnerCfg",
    },
)
