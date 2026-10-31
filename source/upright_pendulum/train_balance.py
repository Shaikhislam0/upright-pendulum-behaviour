"""Separate PPO local-balancing diagnostic, not swing-up training.

Run: python -m upright_pendulum.train_balance
Implementation check: python -m upright_pendulum.train_balance --smoke
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .experiments import ExperimentConfig
from .train_pilot import PilotSettings, run_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent.parent / "configs" / "sin_cos_small_penalty.json",
                        help="experiment JSON; default retains the original local-balancing reward and inputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-action-std", type=float, default=1.0,
                        help="initial Gaussian std in normalized command units; learnable, default 1.0")
    parser.add_argument("--steps", type=int, default=50_000, help="multiple of 1000; default 50000")
    parser.add_argument("--checkpoint-every", type=int, default=10_000, help="multiple of 1000")
    parser.add_argument("--output", type=Path, help="new directory; existing directories are refused")
    parser.add_argument("--smoke", action="store_true", help="256 steps; implementation check only")
    args = parser.parse_args()
    if args.smoke and (args.steps != 50_000 or args.checkpoint_every != 10_000):
        parser.error("--smoke sets its own budget; omit --steps and --checkpoint-every")
    settings = PilotSettings(
        total_steps=256 if args.smoke else args.steps,
        checkpoint_every=128 if args.smoke else args.checkpoint_every,
        rollout_steps=64 if args.smoke else 1000,
        batch_size=64 if args.smoke else 250,
        epochs=2 if args.smoke else 10, seed=args.seed,
        training_reward_scale=1.0, normalize_observations=True,
        initial_action_std=args.initial_action_std,
    )
    config = ExperimentConfig.load(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    prefix = "balance_smoke" if args.smoke else "balance"
    output = args.output if args.output else Path("runs") / f"{prefix}_{stamp}_seed{settings.seed}"
    print(f"Starting local-balancing diagnostic: {settings.total_steps:,} steps, seed={settings.seed}, "
          f"config={config.name}, position penalty={config.reward.position_penalty:g}, "
          f"initial action std={settings.initial_action_std:g}, "
          "normalized observations, training reward scale=1", flush=True)
    print("Training starts near upright; four fixed evaluations run for the full 20 s. No swing-up training.", flush=True)
    run_pilot(config, settings, output, smoke=args.smoke, local_balance=True)


if __name__ == "__main__":
    main()
