"""One isolated stage per process; memory and GPU allocations die with worker."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .common import PACKAGE, REPO, ProgressReporter, atomic_json, code_identity, exclusive_lock, read_json, sha256


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.root)
    plan = read_json(root / "plan.json")
    import torch
    torch.set_num_threads(plan["settings"].get("cpu_threads", 2))
    if code_identity() != plan["code_identity"]:
        raise RuntimeError("Implementation changed after campaign lock")
    job = next(j for j in plan["jobs"] if j["id"] == args.job)
    output = root / "jobs" / job["id"]
    progress = ProgressReporter(output)
    progress.update("starting", 0, 1, "stage", detail=job["id"], force=True)
    if job["kind"] == "train":
        from .learning import train
        train(job, plan["protocols"][job["scenario"]], output,
              plan["settings"], args.device, plan["fingerprint"], plan["prepare_dir"])
    elif job["kind"] == "prepare":
        from .stages import prepare
        prepare(plan, output)
    elif job["kind"] == "extract":
        from .stages import extract
        extract(job, plan, output, args.device)
    elif job["kind"] == "evaluate":
        from .stages import evaluate
        evaluate(job, plan, output)
        # Mandatory graph maintenance after every real eval. No smoke-based
        # hypothesis overrides and no hand-editing of the SQLite database.
        progress.update("knowledge_graph", 0, 1, "step", detail="rebuild evidence graph", force=True)
        subprocess.run([sys.executable, "-m", "src.PECT_JEPA.cross_condition.knowledge.build",
            "--db", str(PACKAGE / "knowledge/graph.sqlite")], cwd=REPO, check=True)
        progress.update("knowledge_graph", 1, 1, "step", detail="complete", force=True)
    else:
        raise ValueError(f"Unknown stage {job['kind']}")
    artifacts = {}
    files = [p for p in sorted(output.rglob("*")) if p.is_file()
             and p.name not in ("receipt.json", "progress.json")
             and not p.name.endswith((".log", ".tmp"))
             and not (job["kind"] == "extract" and
                      (p.suffix == ".npy" or p.name.endswith(".check.json")))]
    progress.update("finalize", 0, len(files), "artifact", detail="hash outputs", force=True)
    for index, path in enumerate(files, start=1):
        artifacts[str(path.relative_to(output))] = sha256(path)
        progress.update("finalize", index, len(files), "artifact", detail=path.name)
    if not (output / "result.json").exists():
        raise RuntimeError("Stage exited without result.json")
    atomic_json(output / "receipt.json", {"fingerprint": plan["fingerprint"], "artifacts": artifacts})
    progress.update("completed", 1, 1, "stage", detail=job["kind"], force=True)


def main():
    with exclusive_lock(PACKAGE / "experiments/.campaign.worker.lock"):
        execute()


if __name__ == "__main__":
    main()
