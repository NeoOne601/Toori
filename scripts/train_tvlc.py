"""
train_tvlc.py — Entry point for TVLCConnector training.

Usage:
  python3.11 scripts/train_tvlc.py --random-init
  python3.11 scripts/train_tvlc.py --coco-dir /path/to/coco2017 --epochs 3

The full trainer lives in scripts/train_tvlc_mlx.py. This wrapper preserves the
original command while routing semantic training to the Gemma-guided trainer.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    known, remaining = parser.parse_known_args()

    if known.random_init:
        from cloud.perception.tvlc_connector import TVLCConnector

        connector = TVLCConnector.create_random_init(save_path=known.output)
        print(f"Random-init TVLC saved to: {TVLCConnector.default_weights_path() if known.output is None else known.output}")
        print(
            "NOTE: Random init is pipeline correctness only. "
            "Use --coco-dir to train a Gemma-guided semantic connector."
        )
        _ = connector
        return

    sys.argv = [
        str(Path(__file__).with_name("train_tvlc_mlx.py")),
        *([f"--output={known.output}"] if known.output else []),
        *remaining,
    ]
    runpy.run_path(str(Path(__file__).with_name("train_tvlc_mlx.py")), run_name="__main__")


if __name__ == "__main__":
    main()
