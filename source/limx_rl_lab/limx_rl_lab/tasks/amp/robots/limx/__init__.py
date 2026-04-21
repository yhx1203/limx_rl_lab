import gymnasium as gym

_TASK_PACKAGE = "limx_rl_lab.tasks.amp.robots.limx"
_AGENTS_PACKAGE = "limx_rl_lab.tasks.amp.agents"


gym.register(
    id="LimX-HU-D04-01-Walk-AMP-Direct",
    entry_point="limx_rl_lab.tasks.amp.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_TASK_PACKAGE}.walk_env_cfg:OliWalkAmpEnvCfg",
        "play_env_cfg_entry_point": f"{_TASK_PACKAGE}.walk_env_cfg:OliWalkAmpPlayEnvCfg",
        "skrl_amp_cfg_entry_point": f"{_AGENTS_PACKAGE}:skrl_walk_amp_cfg.yaml",
    },
)
