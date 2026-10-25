"""
CLI contract tests for `train.py`.

Why this file exists: the model/loss tests build `CrossConditionConfig(...)`
directly, so they can never notice a CLI flag that was never registered in
`train.py`, or a flag that `train.py` forwards into a config field that no
longer exists (which is a hard `TypeError` before the first batch). Both
failure modes have already happened in this project:

    train.py: error: unrecognized arguments: --spectral_phase_encoding cos_sin
    TypeError: CrossConditionConfig.__init__() got an unexpected keyword argument

Covered here:
1. `parse_args()` accepts the canonical flags and the new work-order flags.
2. Every registered flag is forwarded into `CrossConditionConfig` (no dead flag).
3. Every forwarded keyword is a real dataclass field (no deleted-field crash).
4. New flags keep their OFF / legacy defaults so old checkpoints load strict.
5. Flags of the deleted legacy branches are rejected, not silently ignored.
"""

import ast
import os
import sys
import unittest

import torch  # noqa: F401  (train.py pulls torch at import time)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition import train  # noqa: E402
from src.PECT_JEPA.cross_condition.configs.config import CrossConditionConfig  # noqa: E402

TRAIN_PY = os.path.join(REPO_ROOT, "src", "PECT_JEPA", "cross_condition", "train.py")

# Flags that are intentionally NOT forwarded into CrossConditionConfig.
FLAGS_NOT_IN_CONFIG = {"device"}

# Config fields that main() derives from the run directory instead of a CLI flag.
CONFIG_FIELDS_WITHOUT_FLAG = {
    "transform_csv", "exp_dir", "checkpoint_dir", "eval_dir", "visualizations_dir",
}


def _parse_args(argv):
    """Run train.parse_args() on a synthetic argv without touching the real process argv."""
    old = sys.argv
    sys.argv = ["train.py"] + list(argv)
    try:
        return train.parse_args()
    finally:
        sys.argv = old


def _parser_flags():
    tree = ast.parse(open(TRAIN_PY, encoding="utf-8").read())
    flags = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--"):
                    flags.add(a.value)
    return flags


def _config_kwargs():
    tree = ast.parse(open(TRAIN_PY, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "CrossConditionConfig":
            return [k.arg for k in node.keywords if k.arg]
    raise AssertionError("CrossConditionConfig(...) call not found in train.py")


class TestTrainCLIContract(unittest.TestCase):
    def test_parser_is_not_empty(self):
        flags = _parser_flags()
        self.assertGreater(len(flags), 25, f"parser only exposes {len(flags)} flags")

    def test_every_flag_is_forwarded_into_config(self):
        """A flag that is registered but never forwarded is silently dead."""
        flags = {f.lstrip("-") for f in _parser_flags()}
        kwargs = set(_config_kwargs())

        dropped = sorted(flags - FLAGS_NOT_IN_CONFIG - kwargs)
        self.assertEqual(dropped, [], f"flag(s) registered but never passed to CrossConditionConfig: {dropped}")

        stray = sorted(kwargs - flags - CONFIG_FIELDS_WITHOUT_FLAG)
        self.assertEqual(stray, [], f"config arg(s) with no CLI flag or documented default: {stray}")

    def test_every_forwarded_kwarg_is_a_real_config_field(self):
        """Forwarding a deleted field raises TypeError in main() before training starts."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(CrossConditionConfig)}
        unknown = sorted(set(_config_kwargs()) - fields)
        self.assertEqual(unknown, [], f"train.py forwards non-existent config field(s): {unknown}")

    def test_new_flags_default_to_off_and_legacy(self):
        """AGENTS invariant: new flags default OFF so old checkpoints load strict."""
        a = _parse_args([])
        self.assertEqual(a.split_mode, "cross_specimen")
        self.assertIs(a.global_bg_energy_anchor, False)
        self.assertIs(a.inv_predictor, False)
        self.assertEqual(a.pretrain_scope, "base_domain")
        self.assertEqual(a.pred_loss_type, "l1")
        self.assertEqual(a.condition_modulation, "cross_attention")
        self.assertEqual(a.anti_hijack_ratio, 0.0)
        self.assertEqual(a.inv_trim_q, 0.75)
        self.assertEqual(a.inv_weight, 0.0)

    def test_work_order_flags_are_accepted_and_parsed(self):
        a = _parse_args([
            "--split_mode", "cross_condition",
            "--global_bg_energy_anchor", "true",
            "--inv_predictor", "true",
            "--inv_trim_q", "0.5",
            "--anti_hijack_ratio", "0.25",
            "--pred_loss_type", "normalized_l1",
            "--condition_modulation", "adaln",
        ])
        self.assertEqual(a.split_mode, "cross_condition")
        self.assertIs(a.global_bg_energy_anchor, True)
        self.assertIs(a.inv_predictor, True)
        self.assertEqual(a.inv_trim_q, 0.5)
        self.assertEqual(a.anti_hijack_ratio, 0.25)
        self.assertEqual(a.pred_loss_type, "normalized_l1")
        self.assertEqual(a.condition_modulation, "adaln")

    def test_unknown_split_mode_is_rejected(self):
        with self.assertRaises(SystemExit):
            _parse_args(["--split_mode", "cross_specimen_v2"])

    def test_deleted_legacy_flags_are_rejected(self):
        """The aggressive cleanup removed these branches; the CLI must not resurrect them."""
        for flag in ("--ref_patch_mode", "--unified_latent", "--spectral_condition_token",
                     "--latent_disentanglement", "--two_stream_backbone", "--tokenizer_type",
                     "--xcov_weight", "--zc_cond_weight", "--inv_oracle_mask"):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    _parse_args([flag, "true"])

    def test_config_instantiates_from_parsed_defaults(self):
        """Guard the exact main() call: parser defaults -> CrossConditionConfig."""
        a = _parse_args([])
        kwargs = {}
        for name in _config_kwargs():
            if name in ("exp_dir", "checkpoint_dir", "eval_dir", "visualizations_dir",
                        "transform_csv", "resume_checkpoint", "data_dir"):
                continue
            if hasattr(a, name):
                kwargs[name] = getattr(a, name)
        cfg = CrossConditionConfig(**kwargs)
        self.assertEqual(cfg.split_mode, "cross_specimen")


if __name__ == "__main__":
    unittest.main(verbosity=2)
