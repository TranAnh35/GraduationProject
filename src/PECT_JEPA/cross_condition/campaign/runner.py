"""Manifest DAG runner. Exactly one child, atomic state and verified receipts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import PACKAGE, REPO, SCHEMA, atomic_json, code_identity, code_files, digest, exclusive_lock, now, read_json, sha256
from .protocol import SCENARIOS, inventory, matched_roles
from tqdm import tqdm

MODULE = "src.PECT_JEPA.cross_condition.campaign.worker"


def build_plan(data_dir, transform_csv, root, profile="full", seeds=(42, 43, 44), overrides=None, verbose=True):
    inv = inventory(data_dir, transform_csv, verbose=verbose)
    scenarios = ["air_tmr"] if profile == "smoke" else list(SCENARIOS) + ["compound"]
    settings = dict(data_dir=str(Path(data_dir).resolve()), transform_csv=inv["transform_csv"],
        epochs=100, adapt_epochs=10, samples_per_epoch=16384, batch_size=64,
        grid_stride=4, feature_stride=2, feature_dtype="float16", extract_batch_size=128,
        embed_dim=64, encoder_depth=4, lr=4e-4, adapt_lr=1e-4,
        save_every_steps=25, probe_samples_per_file=4000,
        validation_fpr=0.01, localization_tolerance_px=10,
        neural_source_steps=400, neural_adapt_steps=400, cpu_threads=2)
    if profile == "smoke":
        settings.update(epochs=1, adapt_epochs=1, samples_per_epoch=256,
            batch_size=32, feature_stride=8, extract_batch_size=32, probe_samples_per_file=1000,
            neural_source_steps=4, neural_adapt_steps=4)
        seeds = (42,)
    unknown = set(overrides or {}) - set(settings)
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    if set(overrides or {}) & {"data_dir", "transform_csv"}:
        raise ValueError("Use --data-dir/--transform-csv to select input paths; settings JSON is budget-only")
    settings.update(overrides or {})
    for k in ["epochs", "adapt_epochs", "samples_per_epoch", "batch_size", "grid_stride",
              "feature_stride", "extract_batch_size", "save_every_steps", "neural_source_steps", "neural_adapt_steps", "cpu_threads"]:
        if not isinstance(settings[k], int) or settings[k] <= 0:
            raise ValueError(f"{k} must be a positive integer")
    if settings["samples_per_epoch"] % settings["batch_size"]:
        raise ValueError("samples_per_epoch must be divisible by batch_size")
    if settings["embed_dim"] % 4 or not 0 < settings["validation_fpr"] < 1:
        raise ValueError("embed_dim must be divisible by 4; validation_fpr must lie in (0,1)")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be non-empty and unique")
    if any(not isinstance(s, int) or s < 0 or s > 1000000 for s in seeds):
        raise ValueError("Seeds must be integers in [0,1000000]")
    if settings["feature_dtype"] not in ("float16", "float32"):
        raise ValueError("feature_dtype must be float16 or float32")
    root = Path(root).resolve()
    protocols = {s: matched_roles(inv["files"], s) for s in scenarios}
    if profile == "full":
        protocols["air_tmr_spatial"] = {**protocols["air_tmr"], "name": "air_tmr_spatial",
            "spatial_holdout": True, "claim": "held-out spatial locations, same physical specimens"}
        scenarios.append("air_tmr_spatial")
    plan = {"schema": SCHEMA, "profile": profile, "created": now(),
        "code_identity": code_identity(), "inventory": inv, "settings": settings,
        "seeds": list(seeds), "protocols": protocols, "prepare_dir": str(root / "jobs" / "prepare"),
        "jobs": [], "claims_excluded": ["new physical specimen generalization",
            "official TFDA/TS-JEPA/TimeSiam reproduction", "full population confidence intervals"]}

    def add(kind, name, deps=(), **params):
        job = {"id": name, "kind": kind, "dependencies": list(deps), **params}
        plan["jobs"].append(job)
        return name

    add("prepare", "prepare")
    objectives = ["paired_jepa", "self_jepa", "spatial_mae", "random"]
    for scenario in scenarios:
        dual = add("extract", scenario + "__dualview_extract", ["prepare"], scenario=scenario,
                   seed=42, representation="dualview")
        for dual_seed in seeds:
            for arm in ["tf_source", "tf_consistency_pect"]:
                add("evaluate", f"{scenario}__s{dual_seed}__{arm}", [dual], scenario=scenario,
                    seed=dual_seed, representation="dualview", arm=arm,
                    features=str(root / "jobs" / dual))
        fft_feature = None
        if profile != "smoke":
            # This representation is fixed by the input signal, not by a model
            # seed. Extract it once and reuse it for each readout seed.
            tag = f"{scenario}__spatial_fft"
            fft_feature = add("extract", tag + "__extract", ["prepare"], scenario=scenario,
                              seed=42, representation="spatial_fft")
            for fft_seed in seeds:
                for arm in ["raw_lr", "maha"]:
                    add("evaluate", f"{scenario}__s{fft_seed}__spatial_fft__{arm}", [fft_feature],
                        scenario=scenario, seed=fft_seed, representation="spatial_fft", arm=arm,
                        features=str(root / "jobs" / fft_feature))
        for seed in seeds:
            for objective in objectives:
                scopes = ["source", "transductive"] if objective == "paired_jepa" else ["source"]
                for scope in scopes:
                    tag = f"{scenario}__s{seed}__{objective}__{scope}"
                    train = add("train", tag + "__train", ["prepare"], scenario=scenario,
                        seed=seed, objective=objective, scope=scope)
                    ckpt = str(root / "jobs" / train / "best.pt")
                    feat = add("extract", tag + "__extract", [train], scenario=scenario,
                        seed=seed, representation=f"{objective}_{scope}", checkpoint=ckpt)
                    feature_dir = str(root / "jobs" / feat)
                    arms = ["raw_lr", "maha"]
                    if objective == "paired_jepa" and scope == "source":
                        if scenario == "air_tmr":
                            precision = add("extract", tag + "__float32_extract", [train],
                                scenario=scenario, seed=seed, representation="paired_jepa_source_float32",
                                checkpoint=ckpt, feature_dtype="float32")
                            for precision_arm in ["raw_lr", "maha"]:
                                add("evaluate", tag + "__float32_" + precision_arm, [precision],
                                    scenario=scenario, seed=seed, representation="paired_jepa_source_float32",
                                    arm=precision_arm, features=str(root / "jobs" / precision))
                        arms += ["coral", "whiten_lr", "neural_control", "dann"]
                    for arm in arms:
                        add("evaluate", tag + "__" + arm, [feat], scenario=scenario, seed=seed,
                            representation=f"{objective}_{scope}", arm=arm, features=feature_dir)
                    if objective == "paired_jepa" and scope == "source":
                        adapt = add("train", tag + "__adapt", [train], scenario=scenario,
                            seed=seed, objective=objective, scope="adapt", parent_checkpoint=ckpt)
                        af = add("extract", tag + "__adapt_extract", [adapt], scenario=scenario,
                            seed=seed, representation="paired_jepa_adapt",
                            checkpoint=str(root / "jobs" / adapt / "best.pt"))
                        for arm in ["raw_lr", "maha"]:
                            # Refit-on-source control is named explicitly. A
                            # fixed-head transfer arm is added by worker below.
                            add("evaluate", tag + "__adapt_" + arm, [af, feat], scenario=scenario,
                                seed=seed, representation="paired_jepa_adapt", arm=arm,
                                features=str(root / "jobs" / af), fit_features=feature_dir)
                        for radius in ([10, 30] if scenario in ("air_tmr", "z1_z3") else []):
                            add("evaluate", tag + f"__sizing_r{radius}", [feat], scenario=scenario,
                                seed=seed, representation=f"paired_jepa_source_roi{radius}", arm="raw_lr",
                                features=feature_dir, sizing_radius_px=radius)
            if scenario in ("air_tmr", "square_chirp", "z1_z3"):
                tag = f"{scenario}__s{seed}__no_condition"
                tr = add("train", tag + "__train", ["prepare"], scenario=scenario, seed=seed,
                         objective="paired_jepa", scope="source", use_condition_token=False)
                fe = add("extract", tag + "__extract", [tr], scenario=scenario, seed=seed,
                         representation="paired_jepa_no_condition",
                         checkpoint=str(root / "jobs" / tr / "best.pt"))
                for arm in ["raw_lr", "maha"]:
                    add("evaluate", tag + "__" + arm, [fe], scenario=scenario, seed=seed,
                        representation="paired_jepa_no_condition", arm=arm, features=str(root / "jobs" / fe))
    plan["fingerprint"] = digest({k: v for k, v in plan.items() if k not in ("created", "fingerprint")})
    side = len(range(0, 270, settings["feature_stride"]))
    def feature_bytes(job):
        scans = len(protocols[job["scenario"]]["source"] + protocols[job["scenario"]]["target"])
        dimensions = (646 if job["representation"] == "dualview" else
                      130 if job["representation"] == "spatial_fft" else 2 * settings["embed_dim"])
        element_bytes = 2 if job.get("feature_dtype", settings["feature_dtype"]) == "float16" else 4
        return scans * side * side * dimensions * element_bytes

    extract_jobs = {job["id"]: job for job in plan["jobs"] if job["kind"] == "extract"}
    extract_sizes = {jid: feature_bytes(job) for jid, job in extract_jobs.items()}
    last_consumer = {}
    for index, job in enumerate(plan["jobs"]):
        if job["kind"] == "evaluate":
            for dependency in job["dependencies"]:
                if dependency in extract_jobs:
                    last_consumer[dependency] = index
    live = peak = 0
    for index, job in enumerate(plan["jobs"]):
        if job["kind"] == "extract":
            live += extract_sizes[job["id"]]
            peak = max(peak, live)
        for jid, last_index in last_consumer.items():
            if last_index == index:
                live -= extract_sizes[jid]
    cumulative = sum(extract_sizes.values())
    plan["resources"] = {"estimated_feature_cache_bytes": peak,
        "cumulative_feature_cache_bytes": cumulative,
        "canonical_cache_bytes": 81 * 270 * 270 * 128 * 4,
        "note": "Peak assumes feature maps are deleted after their last dependent evaluation; checkpoints/logs excluded"}
    plan["fingerprint"] = digest({k: v for k, v in plan.items() if k not in ("created", "fingerprint")})
    return plan


def receipt_valid(output, fingerprint):
    receipt = Path(output) / "receipt.json"
    if not receipt.exists():
        return False
    try:
        result = read_json(receipt)
        return result["fingerprint"] == fingerprint and bool(result["artifacts"]) and all(
            (Path(output) / name).is_file() and sha256(Path(output) / name) == checksum
            for name, checksum in result["artifacts"].items())
    except (KeyError, OSError, ValueError):
        return False


def feature_path(output, path):
    name = Path(path).stem + "_" + digest(str(Path(path).resolve()))[:20] + ".npy"
    return Path(output) / name


def feature_cache_valid(job, plan, output):
    """Validate transient maps only while at least one downstream job needs them."""
    paths = plan["protocols"][job["scenario"]]["source"] + plan["protocols"][job["scenario"]]["target"]
    for path in paths:
        cached = feature_path(output, path)
        check = cached.with_suffix(".check.json")
        try:
            if not cached.is_file() or read_json(check)["sha256"] != sha256(cached):
                return False
        except (KeyError, OSError, TypeError, ValueError):
            return False
    return True


def cleanup_consumed_feature_caches(plan, root, state):
    """Delete an extraction cache after all direct evaluation consumers finish."""
    for job in plan["jobs"]:
        if job["kind"] != "extract":
            continue
        consumers = [candidate for candidate in plan["jobs"]
                     if candidate["kind"] == "evaluate" and job["id"] in candidate["dependencies"]]
        if not consumers or not all(state["jobs"].get(c["id"], {}).get("status") == "completed"
                                    for c in consumers):
            continue
        output = Path(root) / "jobs" / job["id"]
        paths = plan["protocols"][job["scenario"]]["source"] + plan["protocols"][job["scenario"]]["target"]
        for path in paths:
            cached = feature_path(output, path)
            cached.unlink(missing_ok=True)
            cached.with_suffix(".check.json").unlink(missing_ok=True)


def assert_gpu_idle():
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True, check=False)
    if query.returncode:
        raise RuntimeError("Cannot verify GPU slot with nvidia-smi; use --device cpu or fix NVIDIA tools")
    # WDDM reports desktop graphics apps as compute-apps too. Block Python and
    # common ML executables, without treating Explorer/Chrome as a training job.
    busy = [line for line in query.stdout.splitlines()
            if any(name in line.lower() for name in ("python", "torch", "train", "tensorflow", "jupyter", "wsl"))]
    if busy:
        raise RuntimeError("GPU already has training processes: " + "\n".join(busy))


def _progress_label(job_id, progress):
    if not progress:
        return f"{job_id[:42]} | starting"
    phase = str(progress.get("phase", "running"))
    current, total = progress.get("current"), progress.get("total")
    unit = str(progress.get("unit", ""))
    count = f" {current}/{total} {unit}" if current is not None and total is not None else ""
    detail = progress.get("detail")
    return f"{job_id[:42]} | {phase}{count}" + (f" | {detail}" if detail else "")


def run(root, device="cuda", keep_going=False, max_jobs=None, verbose=True):
    root = Path(root).resolve()
    plan = read_json(root / "plan.json")
    if plan["fingerprint"] != digest({k: v for k, v in plan.items() if k not in ("created", "fingerprint")}):
        raise ValueError("Plan contents do not match its fingerprint")
    if plan["code_identity"] != code_identity():
        raise RuntimeError("Code changed since plan lock. Create a new campaign; do not mix implementations")
    if sha256(plan["inventory"]["transform_csv"]) != plan["inventory"]["transform_sha256"]:
        raise RuntimeError("Registration CSV changed since plan lock")
    for artifact, checksum in plan.get("gt_checksums", {}).items():
        if sha256(artifact) != checksum:
            raise RuntimeError(f"Ground truth changed since plan lock: {artifact}")
    # Prevent accidentally moving a manifest while it still names the old root.
    if Path(plan["prepare_dir"]).parent.parent != root:
        raise ValueError("Campaign was moved. Create a new plan for the new output directory")
    state_path = root / "state.json"
    state = read_json(state_path) if state_path.exists() else {"fingerprint": plan["fingerprint"], "jobs": {}}
    if state["fingerprint"] != plan["fingerprint"]:
        raise ValueError("State/plan mismatch")
    # Account for already materialized cache bytes on resume. Refuse a new
    # campaign that cannot fit its declared artifacts on the actual volume.
    allocated = sum(p.stat().st_size for p in (root / "jobs").rglob("*.npy")) if (root / "jobs").exists() else 0
    required = max(0, plan.get("resources", {}).get("estimated_feature_cache_bytes", 0)
                   + plan.get("resources", {}).get("canonical_cache_bytes", 0) - allocated)
    if shutil.disk_usage(root).free < required + 2 * 1024**3:
        raise RuntimeError(f"Insufficient disk: need about {required / 2**30:.1f} GiB plus 2 GiB reserve")
    attempted = 0
    bar = tqdm(total=len(plan["jobs"]), desc="PECT campaign", unit="job",
               dynamic_ncols=True, disable=not verbose)
    with bar, exclusive_lock(PACKAGE / "experiments" / ".campaign.lock"):
        # A worker can survive an abruptly terminated parent.
        with exclusive_lock(PACKAGE / "experiments/.campaign.worker.lock"):
            pass
        for job in plan["jobs"]:
            jid = job["id"]
            output = root / "jobs" / jid
            current = state["jobs"].get(jid, {})
            bar.set_postfix_str(f"{jid[:42]} | checking", refresh=False)
            receipt_ok = receipt_valid(output, plan["fingerprint"])
            pending_consumers = [candidate for candidate in plan["jobs"]
                if candidate["kind"] == "evaluate" and jid in candidate["dependencies"]
                and state["jobs"].get(candidate["id"], {}).get("status") != "completed"]
            cache_ok = (job["kind"] != "extract" or not pending_consumers
                        or feature_cache_valid(job, plan, output))
            if receipt_ok and cache_ok:
                state["jobs"][jid] = {**current, "status": "completed", "recovered_from_receipt": True}
                cleanup_consumed_feature_caches(plan, root, state)
                bar.update(1)
                continue
            # A completed receipt that no longer validates is corruption, not a
            # reason to silently refit a model under existing downstream results.
            # Missing transient maps are different: while consumers remain,
            # recreate them from the preserved model checkpoint.
            if current.get("status") == "completed" and job["kind"] != "extract":
                raise RuntimeError(f"Corrupt/missing completed artifact: {jid}; use a new campaign")
            if current.get("status") == "completed" and not receipt_ok:
                raise RuntimeError(f"Corrupt/missing completed artifact: {jid}; use a new campaign")
            if any(state["jobs"].get(d, {}).get("status") != "completed" for d in job["dependencies"]):
                state["jobs"][jid] = {"status": "blocked", "reason": "dependency incomplete"}
                atomic_json(state_path, state)
                bar.update(1)
                continue
            if max_jobs is not None and attempted >= max_jobs:
                break
            if device.startswith("cuda") and job["kind"] in ("train", "extract"):
                assert_gpu_idle()
            output.mkdir(parents=True, exist_ok=True)
            attempt = current.get("attempts", 0) + 1
            log = output / f"attempt_{attempt:03d}.log"
            state["jobs"][jid] = {"status": "running", "attempts": attempt,
                "started": now(), "log": str(log)}
            atomic_json(state_path, state)
            atomic_json(output / "progress.json", {"phase": "starting", "current": 0,
                "total": 1, "unit": "stage", "detail": jid, "updated": now()})
            print(f"[{attempted + 1}] {jid} -> {log}", flush=True)
            env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONUNBUFFERED": "1",
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "OMP_NUM_THREADS": str(plan.get("settings", {}).get("cpu_threads", 2)),
                   "MKL_NUM_THREADS": str(plan.get("settings", {}).get("cpu_threads", 2))}
            child = None
            try:
                with log.open("w", encoding="utf-8") as stream:
                    child = subprocess.Popen([sys.executable, "-m", MODULE, "--root", str(root),
                        "--job", jid, "--device", device], cwd=REPO, env=env,
                        stdout=stream, stderr=subprocess.STDOUT)
                    state["jobs"][jid]["pid"] = child.pid
                    atomic_json(state_path, state)
                    while child.poll() is None:
                        state["jobs"][jid]["heartbeat"] = now()
                        atomic_json(state_path, state)
                        if verbose:
                            progress = read_json(output / "progress.json") if (output / "progress.json").exists() else None
                            bar.set_postfix_str(_progress_label(jid, progress))
                        time.sleep(2)
                success = child.returncode == 0 and receipt_valid(output, plan["fingerprint"])
                state["jobs"][jid].update(status="completed" if success else "failed",
                    finished=now(), exit_code=child.returncode)
                atomic_json(state_path, state)
                if success:
                    cleanup_consumed_feature_caches(plan, root, state)
                attempted += 1
                bar.update(1)
                if not success and not keep_going:
                    from .stages import aggregate
                    aggregate(plan, root, state, verbose=verbose)
                    raise RuntimeError(f"Job failed: {jid}. See {log}; rerun resumes its checkpoint")
            except KeyboardInterrupt:
                if child is not None and child.poll() is None:
                    child.terminate()
                    child.wait()
                state["jobs"][jid].update(status="interrupted", finished=now())
                atomic_json(state_path, state)
                raise
        atomic_json(state_path, state)
    from .stages import aggregate
    aggregate(plan, root, state, verbose=verbose)
    if max_jobs is None:
        failed = [jid for jid, item in state["jobs"].items() if item["status"] != "completed"]
        if failed:
            raise RuntimeError(f"Campaign incomplete: {len(failed)} jobs. See state.json and completion.json")


def cli(argv=None):
    parser = argparse.ArgumentParser(description="PECT campaign: plan, run/resume, status")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_verbosity_flags(command_parser):
        group = command_parser.add_mutually_exclusive_group()
        group.add_argument("--verbose", dest="verbose", action="store_true")
        group.add_argument("--no-verbose", dest="verbose", action="store_false")
        command_parser.set_defaults(verbose=True)

    p = sub.add_parser("plan")
    p.add_argument("--data-dir", default=str(REPO / "data"))
    p.add_argument("--transform-csv", default=str(REPO / "data/ground_truth/raw_tdms_rotate_crop.csv"))
    p.add_argument("--root", required=True)
    p.add_argument("--profile", choices=["full", "smoke"], default="full")
    p.add_argument("--settings", help="JSON overrides for budgets; locked into manifest")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    add_verbosity_flags(p)
    for cmd in ["run", "status", "aggregate", "check", "smoke"]:
        p = sub.add_parser(cmd)
        p.add_argument("--root", required=True)
        if cmd == "run":
            p.add_argument("--device", default="cuda")
            p.add_argument("--keep-going", action="store_true")
            p.add_argument("--max-jobs", type=int)
            add_verbosity_flags(p)
        if cmd == "aggregate":
            add_verbosity_flags(p)
        if cmd == "smoke":
            p.add_argument("--inventory-plan", required=True)
            p.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if not root.is_relative_to(PACKAGE):
        raise ValueError("Output must stay inside cross_condition")
    if args.command == "plan":
        if (root / "plan.json").exists():
            raise FileExistsError("Plan already locked. Use run to resume or choose a new root")
        overrides = read_json(args.settings) if args.settings else {}
        plan = build_plan(args.data_dir, args.transform_csv, root, args.profile, args.seeds,
                          overrides, verbose=args.verbose)
        gt = Path(args.data_dir).resolve() / "ground_truth"
        plan["gt_checksums"] = {str(p): sha256(p) for p in sorted(gt.rglob("*"))
                                if p.suffix in (".npy", ".json")}
        plan["fingerprint"] = digest({k: v for k, v in plan.items() if k not in ("created", "fingerprint")})
        atomic_json(root / "plan.json", plan)
        import platform
        versions = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
        for module in ["numpy", "torch", "sklearn", "scipy", "tqdm"]:
            imported = __import__(module)
            versions[module] = imported.__version__
        info = {"code_identity": plan["code_identity"], "environment": versions}
        for name, command in [("commit", ["rev-parse", "HEAD"]), ("branch", ["branch", "--show-current"]),
                              ("dirty", ["status", "--porcelain", "--", str(PACKAGE)])]:
            res = subprocess.run(["git", "-c", "safe.directory=E:/Project_On_Lab", *command],
                                 cwd=REPO, capture_output=True, text=True)
            info[name] = res.stdout.strip() if res.returncode == 0 else "unknown"
        atomic_json(root / "git_info.json", info)
        for source in code_files():
            destination = root / "code_snapshot" / source.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        counts = {kind: sum(j["kind"] == kind for j in plan["jobs"]) for kind in {j["kind"] for j in plan["jobs"]}}
        print({"jobs": len(plan["jobs"]), "stages": counts,
               "peak_features_GiB": round(plan["resources"]["estimated_feature_cache_bytes"] / 2**30, 2),
               "cumulative_features_if_kept_GiB": round(plan["resources"]["cumulative_feature_cache_bytes"] / 2**30, 2),
               "fingerprint": plan["fingerprint"]})
    elif args.command == "smoke":
        from .real_smoke import smoke
        if root.exists():
            raise FileExistsError("Use a fresh smoke directory")
        with exclusive_lock(PACKAGE / "experiments/.campaign.lock"):
            if args.device.startswith("cuda"):
                assert_gpu_idle()
            smoke(root, args.inventory_plan, args.device)
    elif args.command == "check":
        from .verification import verify
        verify(root)
    elif args.command == "run":
        run(root, args.device, args.keep_going, args.max_jobs, args.verbose)
    elif args.command == "aggregate":
        from .stages import aggregate
        aggregate(read_json(root / "plan.json"), root, read_json(root / "state.json"),
                  verbose=args.verbose)
    else:
        plan = read_json(root / "plan.json")
        state = read_json(root / "state.json") if (root / "state.json").exists() else {"jobs": {}}
        from collections import Counter
        counts = Counter(state["jobs"].get(j["id"], {}).get("status", "pending") for j in plan["jobs"])
        print(dict(counts))
        for jid, s in state["jobs"].items():
            if s["status"] in ("running", "failed", "interrupted"):
                print(jid, s)
                progress = root / "jobs" / jid / "progress.json"
                if progress.exists():
                    print("progress", read_json(progress))
