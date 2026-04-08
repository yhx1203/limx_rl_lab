from __future__ import annotations

import argparse
from dataclasses import MISSING
from pathlib import Path


def add_motion_args(parser: argparse.ArgumentParser):
    """Add optional BeyondMimic motion-source arguments."""
    parser.add_argument("--motion_file", type=str, default=None, help="Path to a BeyondMimic motion npz file.")
    parser.add_argument(
        "--registry_name",
        type=str,
        default=None,
        help="Weights & Biases motion artifact path. The ':latest' alias is appended when omitted.",
    )


def _is_missing(value) -> bool:
    return value is MISSING or isinstance(value, type(MISSING))


def apply_motion_source_cfg(env_cfg, args_cli: argparse.Namespace) -> str | None:
    """Apply a motion npz source to env configs that expose commands.motion.motion_file."""
    commands_cfg = getattr(env_cfg, "commands", None)
    motion_cfg = getattr(commands_cfg, "motion", None)
    if motion_cfg is None or not hasattr(motion_cfg, "motion_file"):
        return None

    if args_cli.motion_file is not None and args_cli.registry_name is not None:
        raise ValueError("Please provide only one of --motion_file or --registry_name.")

    if args_cli.motion_file is not None:
        motion_path = Path(args_cli.motion_file).expanduser()
        if not motion_path.is_file():
            raise FileNotFoundError(f"Motion file does not exist: {motion_path}")
        motion_cfg.motion_file = str(motion_path.resolve())
        return motion_cfg.motion_file

    if args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"

        import wandb

        artifact = wandb.Api().artifact(registry_name)
        motion_path = Path(artifact.download()) / "motion.npz"
        if not motion_path.is_file():
            raise FileNotFoundError(f"Motion artifact does not contain motion.npz: {motion_path}")
        motion_cfg.motion_file = str(motion_path.resolve())
        return motion_cfg.motion_file

    if _is_missing(motion_cfg.motion_file):
        raise ValueError(
            "This BeyondMimic task requires a motion source. Provide --motion_file /path/to/motion.npz, "
            "--registry_name <wandb-artifact>, or a Hydra override for commands.motion.motion_file."
        )

    motion_path = Path(str(motion_cfg.motion_file)).expanduser()
    if motion_path.is_file():
        motion_cfg.motion_file = str(motion_path.resolve())

    return str(motion_cfg.motion_file)
