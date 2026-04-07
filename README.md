# Limx_rl_lab

![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-4a4a4a)
![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.2-4a4a4a)
![python](https://img.shields.io/badge/python-3.11-2f86c9)
![platform](https://img.shields.io/badge/platform-linux--64-f08a3c)

## Overview
`limx_rl_lab` is a reinforcement learning codebase built on Isaac Lab for the Limx Oli robot, developed independently from the core Isaac Lab repository.

## Installation
**Conda environment**
```bash
conda create -n limx_rl_lab python=3.11
conda activate limx_rl_lab
```

**Install dependencies**
```bash
# Upgrade pip
pip install --upgrade pip

# Install Isaac Lab and Isaac Sim
pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com
# Install PyTorch
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

**Verify**
```bash
isaacsim
```

**Install limx_rl_lab**
```bash
python -m pip install -e source/limx_rl_lab
```

## Training

**Local training**
```bash
# List available tasks
python scripts/list_envs.py

# Start training
python scripts/rsl_rl/train.py --task LimX-HU-D04-01-Flat-Velocity --headless

# Resume training from the latest checkpoint
python scripts/rsl_rl/train.py --task LimX-HU-D04-01-Flat-Velocity --headless --resume
```

**Training on a server**
```bash
# It is recommended to log in to Weights & Biases for tracking training metrics
export WANDB_API_KEY=your_key
wandb login

# It is recommended to use tmux so training keeps running after disconnection
tmux new -s mysession

# Run training inside tmux
python scripts/rsl_rl/train.py --task LimX-HU-D04-01-Flat-Velocity --headless --logger wandb --log_project_name limx-hu-d04-01 --run_name flat-001

# Press CTRL+B, then release and press D to detach the session
# Training will continue running in the background

# Reattach to the session
tmux attach -t mysession
```

## Evaluate
```bash
python scripts/rsl_rl/play.py   --task LimX-HU-D04-01-Flat-Velocity
```
The following animation shows the learned policy running on flat terrain.

![train_flat_lab](docs/train_flat_lab.gif)

## Sim2sim (mujoco)

**Preparation**

```bash
# Install the official LimX SDK in the current environment
git clone --recurse git@github.com:limxdynamics/humanoid-mujoco-sim.git

python -m pip install --force-reinstall
humanoid-mujoco-sim/limxsdk-lowlevel/python3/amd64/limxsdk-4.0.1-py3-none-any.whl
```

**Launch the MuJoCo simulator**
```bash
cd ~/limx_ws
export ROBOT_TYPE=HU_D04_01
python humanoid-mujoco-sim/simulator.py
```

**Run in another terminal**
```bash
python deploy/sim2sim/sdk_policy_controller.py --policy your_policy

python deploy/sim2sim/sdk_policy_controller.py --policy /home/edy/limx_rl_lab/logs/rsl_rl/limx_hu_d04_01_flat_velocity/2026-04-03_11-35-05_flat-004/exported/policy.pt

# Use a gamepad for control
python deploy/sim2sim/gamepad_policy_controller.py --policy your_policy

python deploy/sim2sim/gamepad_policy_controller.py --policy /home/edy/limx_rl_lab/logs/rsl_rl/limx_hu_d04_01_flat_velocity/2026-04-03_11-35-05_flat-004/exported/policy.pt
```

The following animation shows the learned policy running in MuJoCo on flat terrain.

![train_flat_mujoco](docs/train_flat_mujoco.gif)

### Default Controller Mapping

- Left stick：`vx / vy`
- Right stick：`wz`
- `START`：enable / pause 
- `BACK`：immediately pause and reset the velocity command to zero

## Sim2real 

**Preparation**

Power on the robot

Connect to the Wi-Fi network `HU_D04_xx_xxx_5G`  
Password: `12345678`

```bash
# Check whether the connection is successful
ping 10.192.1.2
```
Enter the developer low-level control mode as shown in the image below.
![control_method](docs/control_method.png)

```bash
# Replace with your own policy
export LIMX_POLICY_ROOT=/home/edy/limx_rl_lab/logs/rsl_rl
export LIMX_WALK_POLICY=limx_hu_d04_01_flat_velocity/2026-04-03_11-35-05_flat-004/exported/policy.pt
```

**Run**

```bash
cd /home/edy/limx_rl_lab
export ROBOT_TYPE=HU_D04_01
python deploy/sim2real/main.py 10.192.1.2
```
**Controller switching**

- `L1 + Y`: switch to `stand`
- `R1 + X`: switch to `walk`
- `L1 + A`: switch to `damping`

The following animation shows the learned policy running in reality.

![train_flat_real](docs/train_flat_real.gif)



## Citation

If you use this codebase or any part of it in your research or project, please cite:

```bibtex
@software{xu2026limx_rl_lab,
  author = {Yiheng Xu},
  title = {limx_rl_lab: Reinforcement Learning Codebase for the Limx Oli Robot, Based on Isaac Lab.},
  url = {https://github.com/yhx1203/limx_rl_lab},
  year = {2026}
}
```

## Acknowledgements

This project builds upon and benefits from the following open-source repositories:

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
- [limxdynamics](https://github.com/limxdynamics)


