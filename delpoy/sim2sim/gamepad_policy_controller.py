import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

try:
    import pygame
    import pygame._sdl2.controller as sdl2_controller
except ImportError as exc:
    raise ImportError(
        "gamepad_policy_controller.py requires pygame with SDL2 controller support. "
        "Please install `pygame` in the runtime environment."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sdk_policy_controller import LimxSDKPolicyController, Rate


AXIS_FULL_SCALE = 32768.0


def normalize_axis(raw_value: int) -> float:
    return float(np.clip(raw_value / AXIS_FULL_SCALE, -1.0, 1.0))


def apply_deadzone(value: float, deadzone: float, expo: float) -> float:
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0

    scaled = (magnitude - deadzone) / max(1e-6, 1.0 - deadzone)
    if expo != 1.0:
        scaled = scaled**expo
    return float(np.sign(value) * scaled)


class GamepadTeleop:
    def __init__(self, args):
        self.controller_index = args.controller_index
        self.max_lin_vel_x = float(args.max_lin_vel_x)
        self.max_lin_vel_y = float(args.max_lin_vel_y)
        self.max_ang_vel_z = float(args.max_ang_vel_z)
        self.deadzone = float(args.deadzone)
        self.expo = float(args.expo)
        self.slow_factor = float(args.slow_factor)
        self.reconnect_interval_s = float(args.reconnect_interval_s)

        self.enabled = bool(args.start_enabled)
        self.controller = None
        self._last_reconnect_attempt_s = -1e9
        self._last_waiting_log_s = -1e9

        self._init_pygame()
        self._try_open_controller(force=True)

    def _init_pygame(self):
        pygame.init()
        pygame.display.init()
        try:
            pygame.display.set_mode((1, 1))
        except pygame.error:
            pass

        sdl2_controller.init()
        sdl2_controller.set_eventstate(True)

    def _candidate_indices(self):
        count = sdl2_controller.get_count()
        if self.controller_index is not None:
            return [self.controller_index] if self.controller_index < count else []
        return [idx for idx in range(count) if sdl2_controller.is_controller(idx)]

    def _try_open_controller(self, force: bool = False):
        if self.controller is not None and self.controller.attached():
            return

        now = time.monotonic()
        if not force and (now - self._last_reconnect_attempt_s) < self.reconnect_interval_s:
            return
        self._last_reconnect_attempt_s = now

        for index in self._candidate_indices():
            if not sdl2_controller.is_controller(index):
                continue

            self.controller = sdl2_controller.Controller(index)
            print(
                f"[gamepad-sim2sim] connected controller index={index} name={self.controller.name!r}. "
                "START: enable/disable, BACK: pause, LB: slow mode."
            )
            return

        if (now - self._last_waiting_log_s) >= 2.0:
            if sdl2_controller.get_count() == 0:
                print("[gamepad-sim2sim] waiting for a controller to be connected...")
            else:
                print(
                    "[gamepad-sim2sim] controller detected but not recognized by SDL2 game-controller API. "
                    "Try an Xbox/PS-compatible mode or pass --controller-index."
                )
            self._last_waiting_log_s = now

    def _handle_disconnect(self):
        if self.controller is not None:
            try:
                self.controller.quit()
            except Exception:
                pass
        self.controller = None
        self.enabled = False
        print("[gamepad-sim2sim] controller disconnected, commands set to zero.")

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.CONTROLLERBUTTONDOWN:
                if event.button == pygame.CONTROLLER_BUTTON_START:
                    self.enabled = not self.enabled
                    state = "enabled" if self.enabled else "paused"
                    print(f"[gamepad-sim2sim] teleop {state}.")
                elif event.button == pygame.CONTROLLER_BUTTON_BACK:
                    if self.enabled:
                        print("[gamepad-sim2sim] teleop paused.")
                    self.enabled = False

            if event.type == pygame.CONTROLLERDEVICEREMOVED and self.controller is not None:
                instance_id = getattr(event, "instance_id", None)
                if instance_id is None or instance_id == self.controller.id:
                    self._handle_disconnect()

            if event.type == pygame.CONTROLLERDEVICEADDED and self.controller is None:
                self._try_open_controller(force=True)

    def read_command(self) -> np.ndarray:
        self._process_events()
        self._try_open_controller()

        if self.controller is None or not self.controller.attached():
            if self.controller is not None:
                self._handle_disconnect()
            return np.zeros(3, dtype=np.float32)

        pygame.event.pump()
        sdl2_controller.update()

        if not self.enabled:
            return np.zeros(3, dtype=np.float32)

        left_y = -normalize_axis(self.controller.get_axis(pygame.CONTROLLER_AXIS_LEFTY))
        left_x = normalize_axis(self.controller.get_axis(pygame.CONTROLLER_AXIS_LEFTX))
        right_x = normalize_axis(self.controller.get_axis(pygame.CONTROLLER_AXIS_RIGHTX))

        vx = apply_deadzone(left_y, self.deadzone, self.expo) * self.max_lin_vel_x
        vy = apply_deadzone(left_x, self.deadzone, self.expo) * self.max_lin_vel_y
        wz = apply_deadzone(right_x, self.deadzone, self.expo) * self.max_ang_vel_z

        if self.controller.get_button(pygame.CONTROLLER_BUTTON_LEFTSHOULDER):
            vx *= self.slow_factor
            vy *= self.slow_factor
            wz *= self.slow_factor

        return np.array([vx, vy, wz], dtype=np.float32)

    def close(self):
        if self.controller is not None:
            try:
                self.controller.quit()
            except Exception:
                pass
            self.controller = None
        sdl2_controller.quit()
        pygame.quit()


class GamepadPolicyController(LimxSDKPolicyController):
    def __init__(self, args):
        super().__init__(args)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.teleop = GamepadTeleop(args)

    def run(self):
        self.wait_for_first_state()
        rate = Rate(self.update_rate)
        t0 = time.perf_counter()

        while True:
            self.cmd = self.teleop.read_command()

            robot_state = copy.deepcopy(self.io.robot_state)
            imu_data = copy.deepcopy(self.io.imu_data)
            sim_time = self.loop_count / float(self.update_rate)

            if self.loop_count % self.control_decimation == 0:
                obs = self.build_observation(robot_state, imu_data, sim_time)
                self.last_action = self.compute_action(obs)

            self.publish_command(robot_state)

            self.loop_count += 1
            if self.loop_count % self.update_rate == 0:
                fps = self.loop_count / max(1e-6, (time.perf_counter() - t0))
                state = "ACTIVE" if self.teleop.enabled else "PAUSED"
                controller_name = self.teleop.controller.name if self.teleop.controller is not None else "none"
                print(
                    f"[gamepad-sim2sim] fps={fps:.1f} state={state} controller={controller_name} "
                    f"cmd=({self.cmd[0]:+.2f}, {self.cmd[1]:+.2f}, {self.cmd[2]:+.2f})",
                    end="\r",
                )
            rate.sleep()

    def close(self):
        self.teleop.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LimX sdk-based sim2sim controller with SDL2 gamepad teleoperation for the official MuJoCo simulator."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/limx_flat_velocity_sdk.yaml",
        help="Path to sdk sim2sim yaml config.",
    )
    parser.add_argument("--policy", type=str, default=None, help="Optional override path to exported TorchScript policy.")
    parser.add_argument(
        "--controller-index",
        type=int,
        default=None,
        help="SDL controller index. Defaults to the first SDL2-compatible controller.",
    )
    parser.add_argument(
        "--lin-vel-x",
        type=float,
        default=None,
        help="Compatibility option inherited from sdk_policy_controller.py. Ignored after teleop starts.",
    )
    parser.add_argument(
        "--lin-vel-y",
        type=float,
        default=None,
        help="Compatibility option inherited from sdk_policy_controller.py. Ignored after teleop starts.",
    )
    parser.add_argument(
        "--ang-vel-z",
        type=float,
        default=None,
        help="Compatibility option inherited from sdk_policy_controller.py. Ignored after teleop starts.",
    )
    parser.add_argument(
        "--start-enabled",
        action="store_true",
        help="Start teleop in enabled state. By default the script starts paused and waits for START.",
    )
    parser.add_argument(
        "--max-lin-vel-x",
        type=float,
        default=0.4,
        help="Maximum forward/backward command in m/s.",
    )
    parser.add_argument(
        "--max-lin-vel-y",
        type=float,
        default=0.2,
        help="Maximum lateral command in m/s.",
    )
    parser.add_argument(
        "--max-ang-vel-z",
        type=float,
        default=0.5,
        help="Maximum yaw-rate command in rad/s.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.12,
        help="Normalized stick deadzone in [0, 1).",
    )
    parser.add_argument(
        "--expo",
        type=float,
        default=1.5,
        help="Exponent for stick shaping. Values > 1 soften small motions.",
    )
    parser.add_argument(
        "--slow-factor",
        type=float,
        default=0.35,
        help="Scale applied while holding LB.",
    )
    parser.add_argument(
        "--reconnect-interval-s",
        type=float,
        default=1.0,
        help="How often to retry controller connection when disconnected.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    controller = GamepadPolicyController(parse_args())
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n[gamepad-sim2sim] stopped by user.")
    finally:
        controller.close()
