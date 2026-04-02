# Sim2real Deployment

This directory mirrors the official LimX deployment layout, but plugs in the walking policy trained in this repo.

## Structure

- `main.py`: official-style entrypoint that initializes the robot and loads abilities
- `controllers/HU_D04_01/controllers.yaml`: ability registry for `stand`, `walk`, and `damping`
- `controllers/HU_D04_01/walk_controller/walk_controller.py`: TorchScript RL walk controller

## Preparation

Install the official `limxsdk` in the runtime environment first.

## Configure the policy

Edit `controllers/HU_D04_01/walk_controller/walk_param.yaml` and set:

- `policy_root`
- `policy_path`

You can also override them with environment variables:

```bash
export LIMX_POLICY_ROOT=/absolute/path/to/logs/rsl_rl
export LIMX_WALK_POLICY=limx_hu_d04_01_flat_velocity_gaitphase/.../exported/policy.pt
```

## Run

```bash
cd /home/edy/limx_rl_lab
export ROBOT_TYPE=HU_D04_01
python deploy/sim2real/main.py 10.192.1.2
```

If `ROBOT_IP` is already exported, the CLI argument is optional.

## Controller switching

- `L1 + Y`: switch to `stand`
- `R1 + X`: switch to `walk`
- `L1 + A`: switch to `damping`
- `L1 + X`: unload all abilities

`controllers.yaml` currently autostarts `stand`, which is the safer default for real hardware.
