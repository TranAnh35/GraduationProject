"""Acceptance checks for orchestration and scientific contracts (no pytest).

Run: python run_campaign.py check --root experiments/campaigns/checks
Synthetic tests establish implementation contracts, never research performance.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import tempfile

import numpy as np
import torch

from ..configs.config import CrossConditionConfig
from ..models.cross_condition_jepa import CrossConditionJEPA
from .common import ProgressReporter, atomic_json, exclusive_lock, read_json, seed_all, sha256
from .protocol import matched_roles, spatial_masks
from .runner import cleanup_consumed_feature_caches, feature_cache_valid, feature_path, receipt_valid
from .learning import CampaignModel
from .metrics import candidates, match_candidates, object_descriptor
from .adaptation import fit_readout, predict


def verify(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="acceptance_", dir=root) as scratch:
        if not Path(scratch).resolve().is_relative_to(root):
            raise ValueError("Verification scratch outside requested output")
        _verify(scratch)
        atomic_json(root / "verification.json", read_json(Path(scratch) / "verification.json"))


def _verify(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    passed = []
    files = [dict(path=f"/{s}/{p}/{w}/{z}.tdms", sensor=s, specimen=p, waveform=w, liftoff=z)
        for s, p, w, z in product(["Hall_Air_Core", "Hall_Pot_Core", "TMR"],
            ["Corrosion", "Rivet_v1", "Rivet_v2"], ["Square", "Gaussian", "Chirp"], ["z1", "z2", "z3"])]
    for name in ["air_tmr", "pot_tmr", "square_chirp", "gaussian_chirp", "z1_z3", "z3_z1"]:
        role = matched_roles(files, name)
        assert [len(role[k]) for k in ["source", "fit", "val", "target"]] == [27, 21, 6, 27]
        assert not set(role["source"]) & set(role["target"])
        assert not set(role["fit"]) & set(role["val"])
        for block in role["blocks"]:
            a, b = (next(f for f in files if f["path"] == block[k]) for k in ["source", "target"])
            assert [k for k in ["sensor", "specimen", "waveform", "liftoff"] if a[k] != b[k]] == [role["axis"]]
    role = matched_roles(files, "compound")
    assert [len(role[k]) for k in ["source", "fit", "val", "target"]] == [24, 18, 6, 57]
    passed.append("matched_factor_roles_and_disjoint_labels")

    cfg = CrossConditionConfig(embed_dim=32, encoder_depth=1, predictor_depth=1)
    seed_all(42)
    original = CrossConditionJEPA(cfg)
    seed_all(42)
    model = CampaignModel(cfg).eval()
    assert original.state_dict().keys() == model.state_dict().keys()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.state_dict().items())
    passed.append("legacy_state_dict_unchanged")
    x = torch.randn(4, 5, 5, 128)
    ctx = model.context_idx_bank[:4]
    a, _ = model.visible_tokens(x, ctx)
    y = x.reshape(4, 25, 128).clone()
    for i in range(4):
        y[i, model.target_idx_bank[i]] *= -100
    b, _ = model.visible_tokens(y.reshape_as(x), ctx)
    assert torch.equal(a, b)
    passed.append("no_hidden_probe_tokenizer_coupling")
    for objective in ["paired_jepa", "self_jepa", "spatial_mae"]:
        results = []
        for _ in range(2):
            seed_all(7)
            m = CampaignModel(cfg, objective)
            batch = dict(x_a=x, x_b=y.reshape_as(x), ref_a=x, ref_b=x,
                         scan_id=torch.tensor([0, 0, 1, 1]))
            loss, metrics = m.ssl_loss(batch)
            loss.backward()
            results.append((loss.detach(), next(m.context_encoder.parameters()).grad.clone()))
        assert torch.equal(results[0][0], results[1][0])
        assert torch.equal(results[0][1], results[1][1])
    passed.append("same_seed_same_ssl_loss_and_gradient")

    tr, te = spatial_masks((270, 270))
    assert tr.any() and te.any() and not (tr & te).any()
    assert not candidates(np.zeros((30, 30)), 0)
    matches, fp, fn = match_candidates([dict(x=1, y=1), dict(x=50, y=50)], [dict(x=0, y=0)], 5)
    assert len(matches) == 1 and fp == 1 and fn == 0
    z, score = np.ones((30, 30, 4)), np.zeros((30, 30))
    assert np.isfinite(object_descriptor(z, score, 15, 15, 1)).all()
    passed.append("spatial_buffer_constant_null_object_matching_fixed_roi")

    with exclusive_lock(root / "lock"):
        try:
            with exclusive_lock(root / "lock"):
                raise AssertionError("Duplicate lock accepted")
        except RuntimeError:
            pass
    out = root / "receipt_test"
    atomic_json(out / "result.json", {"complete": True})
    atomic_json(out / "receipt.json", {"fingerprint": "test", "artifacts": {"result.json": sha256(out / "result.json")}})
    assert receipt_valid(out, "test")
    atomic_json(out / "result.json", {"complete": False})
    assert not receipt_valid(out, "test")
    passed.append("exclusive_lock_and_corruption_detection")

    progress_dir = root / "progress"
    reporter = ProgressReporter(progress_dir, interval=0)
    assert reporter.update("fixture", 2, 5, "step", detail="deterministic", force=True)
    saved_progress = read_json(progress_dir / "progress.json")
    assert (saved_progress["phase"], saved_progress["current"], saved_progress["total"]) == ("fixture", 2, 5)
    passed.append("atomic_task_progress_reporting")

    # A map remains available while any consumer is pending, then is evicted.
    feature_plan = {"protocols": {"fixture": {"source": ["/fixture/scan.tdms"], "target": []}},
        "jobs": [dict(id="extract", kind="extract", scenario="fixture"),
                 dict(id="score", kind="evaluate", dependencies=["extract"])]}
    feature_output = root / "jobs" / "extract"
    cached = feature_path(feature_output, "/fixture/scan.tdms")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"feature-map")
    atomic_json(cached.with_suffix(".check.json"), {"sha256": sha256(cached)})
    feature_state = {"jobs": {"score": {"status": "pending"}}}
    assert feature_cache_valid(feature_plan["jobs"][0], feature_plan, feature_output)
    cleanup_consumed_feature_caches(feature_plan, root, feature_state)
    assert cached.exists()
    feature_state["jobs"]["score"]["status"] = "completed"
    cleanup_consumed_feature_caches(feature_plan, root, feature_state)
    assert not cached.exists() and not cached.with_suffix(".check.json").exists()
    passed.append("feature_cache_retained_until_last_consumer_then_evicted")

    # Interrupt after a valid step checkpoint, then restore optimizer + RNG.
    rng = np.random.RandomState(42)
    xx, yy = rng.randn(64, 646).astype(np.float32), np.arange(64) % 2
    cfg_train = dict(neural_source_steps=2, neural_adapt_steps=3, batch_size=16, save_every_steps=1)
    from . import adaptation
    for kind in ["neural_control", "dann", "tf_source", "tf_consistency_pect"]:
        reference = fit_readout(root / (kind + "_reference"), xx, yy, xx, xx + 0.2, 42, kind, cfg_train)
        saved = adaptation.atomic_torch

        def interrupt(path, value):
            saved(path, value)
            if value["step"] == 1:
                raise InterruptedError("intentional acceptance-test interruption")

        adaptation.atomic_torch = interrupt
        try:
            try:
                fit_readout(root / (kind + "_resumed"), xx, yy, xx, xx + 0.2, 42, kind, cfg_train)
                raise AssertionError("Fault injection did not fire")
            except InterruptedError:
                pass
        finally:
            adaptation.atomic_torch = saved
        resumed = fit_readout(root / (kind + "_resumed"), xx, yy, xx, xx + 0.2, 42, kind, cfg_train)
        assert np.array_equal(predict(reference, xx), predict(resumed, xx)), kind
    passed.append("neural_checkpoint_resume_exact_all_four_arms")
    from . import runner
    from .common import digest
    demo = root / "dag"
    atomic_json(demo / "transform.csv", {"fixture": True})
    # Replace only process execution; exercise real receipt/state/dependency
    # handling, including a crash and a second user-invoked run.
    plan = dict(schema=1, code_identity=runner.code_identity(),
        inventory={"transform_csv": str(demo / "transform.csv"),
                   "transform_sha256": sha256(demo / "transform.csv")},
        prepare_dir=str(demo / "jobs/prepare"), jobs=[
            dict(id="prepare", kind="prepare", dependencies=[]),
            dict(id="failure", kind="prepare", dependencies=["prepare"]),
            dict(id="tail", kind="prepare", dependencies=["failure"])])
    plan["fingerprint"] = digest(plan)
    atomic_json(demo / "plan.json", plan)
    calls = []
    should_fail = True

    class Child:
        def __init__(self, args, **kwargs):
            jid = args[args.index("--job") + 1]
            calls.append(jid)
            self.pid = 12345
            self.returncode = 1 if should_fail and jid == "failure" else 0
            if self.returncode == 0:
                dest = demo / "jobs" / jid
                atomic_json(dest / "result.json", {"complete": True})
                atomic_json(dest / "receipt.json", {"fingerprint": plan["fingerprint"],
                    "artifacts": {"result.json": sha256(dest / "result.json")}})

        def poll(self):
            return self.returncode

    saved_popen = runner.subprocess.Popen
    runner.subprocess.Popen = Child
    try:
        try:
            runner.run(demo, device="cpu")
            raise AssertionError("Expected stage failure")
        except RuntimeError:
            assert read_json(demo / "state.json")["jobs"]["failure"]["status"] == "failed"
        should_fail = False
        runner.run(demo, device="cpu")
        assert calls == ["prepare", "failure", "failure", "tail"]
        assert read_json(demo / "completion.json")["complete"]
    finally:
        runner.subprocess.Popen = saved_popen
    passed.append("dag_failure_dependency_receipt_skip_and_resume")
    atomic_json(root / "verification.json", {"status": "passed", "checks": passed})
    print("PASS:", "\n".join(passed))
