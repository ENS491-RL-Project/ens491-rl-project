"""
One-time developer setup for a fresh clone.

Installs requirements-dev.txt (runtime pins + graphify) into the active
interpreter, installs graphify's git hooks and builds the code graph in
graphify-out/. After this, the hooks rebuild the graph after every commit,
pull and branch switch.

Every step runs through `sys.executable`, so graphify's hooks pin this
interpreter and keep working when git runs without the conda env activated
(GUI clients, another terminal).

Usage (inside the ens491 conda env):
    conda activate ens491
    python scripts/setup_dev.py
    python scripts/setup_dev.py --skip-install   # hooks + graph only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ENV = "ens491"


def run(*args: str) -> None:
    cmd = [sys.executable, *args]
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="do not pip install requirements-dev.txt",
    )
    parser.add_argument(
        "--any-env",
        action="store_true",
        help=f"allow running outside the '{EXPECTED_ENV}' conda env",
    )
    args = parser.parse_args()

    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env != EXPECTED_ENV and not args.any_env:
        print(
            f"Active conda env is {env!r}, expected {EXPECTED_ENV!r}.\n"
            f"Run `conda activate {EXPECTED_ENV}` first (packages must not be "
            "installed outside it), or pass --any-env.",
            file=sys.stderr,
        )
        return 1

    if not args.skip_install:
        run("-m", "pip", "install", "-r", "requirements-dev.txt")
    run("-m", "graphify", "hook", "install")
    run("-m", "graphify", "update", ".")

    print("\nDone. graphify-out/ is built and the git hooks keep it current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
