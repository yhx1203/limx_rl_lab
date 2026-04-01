## 方案
如果你本地已经有 `~/limx_ws/humanoid-mujoco-sim` 和 `~/limx_ws/humanoid-rl-deploy-python`，用官方这套思路：

1. 启动官方 MuJoCo 仿真器

```bash
cd ~/limx_ws
export ROBOT_TYPE=HU_D04_01
python humanoid-mujoco-sim/simulator.py
```

2. 在另一个终端运行这个 repo 里的 PR 空间 policy controller

```bash
python delpoy/sim2sim/sdk_policy_controller.py --policy /home/edy/lab/limx_rl_lab/logs/rsl_rl/limx_hu_d04_01_flat_velocity_gaitphase/2026-03-31_16-22-00_flat-002/exported/policy.pt
```

这条路径会复用 `humanoid-mujoco-sim` 里的 `kinematic_projection`，让闭链脚踝/腰部的 PR→AB 转换交给官方机制处理，通常会比直接在 MuJoCo 里手写 `A/B mixing` 更稳。
