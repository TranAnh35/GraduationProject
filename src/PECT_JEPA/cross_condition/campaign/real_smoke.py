"""One-epoch real-data acceptance, including exact interrupted SSL resume.

This is infrastructure validation on a tiny subset, never a research result.
"""
from pathlib import Path
import subprocess
import sys

import torch

from .common import PACKAGE, REPO, atomic_json, read_json
from . import learning
from .stages import prepare, extract, evaluate


def smoke(root, inventory_plan, device="cuda"):
    if device == "cpu":
        torch.set_num_threads(2)
    root = Path(root).resolve()
    base = read_json(inventory_plan)
    source = [p for p in base["protocols"]["air_tmr"]["source"] if "Corrosion" in p][:2]
    target = [p for p in base["protocols"]["air_tmr"]["target"] if "Corrosion" in p][:1]
    protocol = dict(name="air_tmr", source=source, fit=source[:1], val=source[1:], target=target)
    settings = {**base["settings"], "epochs": 1, "adapt_epochs": 1, "samples_per_epoch": 256,
        "batch_size": 32, "embed_dim": 32, "encoder_depth": 1, "feature_stride": 4,
        "extract_batch_size": 64, "save_every_steps": 2,
        "probe_samples_per_file": 2000, "neural_source_steps": 4, "neural_adapt_steps": 4}
    plan = dict(settings=settings, protocols={"air_tmr": protocol}, prepare_dir=str(root / "prepare"),
        inventory={"files": [e for e in base["inventory"]["files"] if e["path"] in source + target]})
    prepare(plan, root / "prepare")
    checks = []
    for objective in ["paired_jepa", "self_jepa", "spatial_mae"]:
        learning.train(dict(seed=42, objective=objective, scope="source"), protocol,
            root / objective, settings, device, "real_smoke", plan["prepare_dir"])
        import csv
        rows = list(csv.DictReader((root / objective / "metrics_epoch.csv").open()))
        assert len(rows) == 1 and {"pred_loss", "val_auc", "val_sound_var", "effective_rank"} <= rows[0].keys()
        checks.append(objective + "_one_epoch_256_real_samples")
    saved_save = learning.atomic_torch

    def fault(path, value):
        saved_save(path, value)
        if value.get("step") == 2:
            raise InterruptedError("intentional real SSL checkpoint interruption")

    learning.atomic_torch = fault
    try:
        try:
            learning.train(dict(seed=42, objective="paired_jepa", scope="source"), protocol,
                root / "interrupted", settings, device, "real_smoke", plan["prepare_dir"])
            raise AssertionError("Interruption did not fire")
        except InterruptedError:
            pass
    finally:
        learning.atomic_torch = saved_save
    learning.train(dict(seed=42, objective="paired_jepa", scope="source"), protocol,
        root / "interrupted", settings, device, "real_smoke", plan["prepare_dir"])
    reference = torch.load(root / "paired_jepa/best.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(root / "interrupted/best.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(v, resumed["model"][k]) for k, v in reference["model"].items())
    assert reference["history"] == resumed["history"]
    checks.append("real_ssl_interruption_resume_exact_model_and_metrics")
    learning.train(dict(seed=42, objective="paired_jepa", scope="adapt",
        parent_checkpoint=str(root / "paired_jepa/best.pt")), protocol,
        root / "adapt", settings, device, "real_smoke", plan["prepare_dir"])
    checks.append("adapter_one_epoch_target_unlabelled")
    for name, representation, checkpoint in [
        ("features", "paired_jepa_source", str(root / "paired_jepa/best.pt")),
        ("adapt_features", "paired_jepa_adapt", str(root / "adapt/best.pt")),
        ("dual_features", "dualview", None), ("fft_features", "spatial_fft", None)]:
        extract(dict(scenario="air_tmr", representation=representation, checkpoint=checkpoint),
                plan, root / name, device)
    checks.append("four_representation_extraction_paths")
    for arm in ["raw_lr", "coral", "whiten_lr", "maha", "neural_control", "dann", "tf_source", "tf_consistency_pect", "adapt_fixed_head"]:
        dual = arm in ("tf_source", "tf_consistency_pect")
        job = dict(scenario="air_tmr", seed=42, arm="raw_lr" if arm == "adapt_fixed_head" else arm,
                   representation="dualview" if dual else "paired_jepa_source",
                   features=str(root / ("dual_features" if dual else "features")))
        if arm == "adapt_fixed_head":
            job.update(representation="paired_jepa_adapt", features=str(root / "adapt_features"), fit_features=str(root / "features"))
        evaluate(job, plan, root / ("eval_" + arm))
        subprocess.run([sys.executable, "-m", "src.PECT_JEPA.cross_condition.knowledge.build",
            "--db", str(PACKAGE / "knowledge/graph.sqlite")], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
        checks.append(arm + "_real_eval_and_graph_rebuild")
    result = {"status": "passed", "scope": "implementation smoke; not comparative research evidence",
              "checks": checks, "device": device}
    atomic_json(root / "verification.json", result)
    print(result)
