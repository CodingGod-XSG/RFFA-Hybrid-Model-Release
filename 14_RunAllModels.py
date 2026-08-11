# -*- coding: utf-8 -*-
"""14_RunAllModels.py
Run model entry scripts in sequence.

Default run list:
  06_RF.py
  09_ANN.py
  10_ANN_Joint.py
  11_GEV_NN.py
  13_GEV_NN_NLL.py
  12_GEV_NN_MSE.py

Usage examples
--------------
python 14_RunAllModels.py
python 14_RunAllModels.py --group baseline
python 14_RunAllModels.py --continue-on-error
python 14_RunAllModels.py --only 06_RF.py 11_GEV_NN.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

SCRIPT_GROUPS: dict[str, list[str]] = {
    "all": [
        "06_RF.py",
        "07_SVM.py",
        "08_XGBoost.py",
        "09_ANN.py",
        "10_ANN_Joint.py",
        "11_GEV_NN.py",
        "13_GEV_NN_NLL.py",
        "12_GEV_NN_MSE.py",
    ],
    "baseline": [
        "06_RF.py",
        "07_SVM.py",
        "08_XGBoost.py",
        "09_ANN.py",
        "10_ANN_Joint.py",
    ],
    "gev": [
        "11_GEV_NN.py",
        "13_GEV_NN_NLL.py",
        "12_GEV_NN_MSE.py",
    ],
}


def _resolve_scripts(group: str, only: list[str] | None) -> list[Path]:
    names = only if only else SCRIPT_GROUPS[group]
    paths = []
    missing = []
    for name in names:
        p = SCRIPT_DIR / name
        if p.exists():
            paths.append(p)
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"Missing script(s): {', '.join(missing)}")
    return paths


def _run_one(script: Path, python_exec: str, pass_args: list[str]) -> tuple[int, float]:
    cmd = [python_exec, str(script), *pass_args]
    t0 = time.time()
    print(f"\n[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    dt = time.time() - t0
    return proc.returncode, dt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all model entry scripts sequentially")
    parser.add_argument(
        "--group",
        choices=sorted(SCRIPT_GROUPS.keys()),
        default="all",
        help="Predefined script group to run",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Explicit script filenames to run (overrides --group)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining scripts even if one fails",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run child scripts",
    )
    parser.add_argument(
        "--pass-arg",
        action="append",
        default=[],
        help="Extra argument to pass to every child script (repeatable)",
    )
    args = parser.parse_args()

    scripts = _resolve_scripts(args.group, args.only)

    print("=" * 72)
    print("14_RunAllModels")
    print("=" * 72)
    print(f"python: {args.python}")
    print(f"scripts: {len(scripts)}")
    for i, s in enumerate(scripts, start=1):
        print(f"  {i}. {s.name}")

    failed = []
    start_all = time.time()

    for script in scripts:
        rc, dt = _run_one(script, args.python, args.pass_arg)
        if rc == 0:
            print(f"[OK]   {script.name} ({dt / 60.0:.1f} min)")
        else:
            print(f"[FAIL] {script.name} rc={rc} ({dt / 60.0:.1f} min)")
            failed.append((script.name, rc))
            if not args.continue_on_error:
                break

    total_min = (time.time() - start_all) / 60.0
    print("\n" + "=" * 72)
    print(f"Total runtime: {total_min:.1f} min")

    if failed:
        print("Failed scripts:")
        for name, rc in failed:
            print(f"  - {name} (rc={rc})")
        return 1

    print("All scripts finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
