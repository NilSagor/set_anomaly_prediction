"""
Run training with multiple random seeds.
Usage: python scripts/run_sweep.py --seeds 1 2 3 4 5 --config config/trainer/default.yaml
"""


"""
Run training with multiple random seeds.
Usage: python scripts/run_sweep.py --seeds 42 43 44 --config config/trainer/default.yaml
"""

import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs='+', required=True)
    parser.add_argument("--config", type=str, default="config/trainer/default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_cmd = ["python", str(PROJECT_ROOT / "src/scripts/train.py"), "--config", args.config]
    for seed in args.seeds:
        cmd = base_cmd + ["--seed", str(seed)]
        cmd_str = " ".join(cmd)
        print(f"Running: {cmd_str}")
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

if __name__ == "__main__":
    main()