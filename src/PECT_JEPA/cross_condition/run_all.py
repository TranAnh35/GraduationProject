"""One-command campaign launcher, independent of PowerShell execution policy."""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plan once, then run/resume the entire PECT campaign")
    parser.add_argument("--root", default=str(package / "experiments/campaigns/pect_v4"))
    parser.add_argument("--data-dir")
    parser.add_argument("--settings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", dest="verbose", action="store_true",
                           help="show tqdm progress for campaigns and long-running tasks (default)")
    verbosity.add_argument("--no-verbose", dest="verbose", action="store_false",
                           help="disable tqdm progress display")
    parser.set_defaults(verbose=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true")
    group.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    entry = [sys.executable, str(package / "run_campaign.py")]

    def invoke(arguments):
        result = subprocess.run(entry + arguments, cwd=package)
        if result.returncode:
            raise SystemExit(result.returncode)

    if args.status:
        if not (root / "plan.json").exists():
            print(f"No campaign planned at {root}")
            return
        invoke(["status", "--root", str(root)])
        return
    if not (root / "plan.json").exists():
        data = Path(args.data_dir).resolve() if args.data_dir else package.parents[2] / "data"
        command = ["plan", "--root", str(root), "--data-dir", str(data),
            "--transform-csv", str(data / "ground_truth/raw_tdms_rotate_crop.csv")]
        command.append("--verbose" if args.verbose else "--no-verbose")
        if args.settings:
            command += ["--settings", str(Path(args.settings).resolve())]
        invoke(command)
    else:
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        if args.data_dir and str(Path(args.data_dir).resolve()) != plan["settings"]["data_dir"]:
            parser.error("Data directory differs from locked plan; choose a new --root")
        if args.settings:
            requested = json.loads(Path(args.settings).read_text(encoding="utf-8-sig"))
            if any(plan["settings"].get(k) != v for k, v in requested.items()):
                parser.error("Settings differ from locked plan; choose a new --root")
    if args.plan_only:
        print(f"Locked plan: {root / 'plan.json'}")
        return
    command = ["run", "--root", str(root), "--device", args.device]
    command.append("--verbose" if args.verbose else "--no-verbose")
    if args.keep_going:
        command.append("--keep-going")
    if args.max_jobs is not None:
        command += ["--max-jobs", str(args.max_jobs)]
    invoke(command)


if __name__ == "__main__":
    main()
