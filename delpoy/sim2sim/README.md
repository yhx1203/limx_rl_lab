### 默认手柄映射

- 左摇杆：`vx / vy`
- 右摇杆左右：`wz`
- `START`：启用 / 暂停 teleop
- `BACK`：立即暂停并把速度命令归零
- `LB`：慢速模式（默认缩放到 `35%`）

### 常用参数

```bash
python delpoy/sim2sim/gamepad_policy_controller.py \
  --policy /path/to/policy.pt \
  --start-enabled \
  --max-lin-vel-x 0.4 \
  --max-lin-vel-y 0.2 \
  --max-ang-vel-z 0.5 \
  --deadzone 0.12
```

说明：

- 脚本默认是 `paused` 状态，启动后按一下 `START` 才会开始给命令。
- 如果手柄断开，脚本会自动把命令清零，并持续尝试重连。
- 目前走的是 SDL2 标准 controller 映射，Xbox/PS 兼容模式通常最稳。
- 运行环境里需要有 `pygame`。如果没有，可以先安装：`pip install pygame`
