"""
CLI contract tests for `train.py`.

Why this file exists: `tests/test_exp68_features.py` exercises the model / loss / config-object
layer by constructing `CrossConditionConfig(...)` directly, so it can never notice a CLI flag
that was never registered in `train.py`. That is exactly how EXP-68B died on Colab:

    train.py: error: unrecognized arguments: --spectral_phase_encoding cos_sin --ref_patch_mode corner_median

All 14 feature tests were green at the time, because argparse was not part of the surface under
test. These tests cover that surface:

1. `parse_args()` accepts every flag the EXP-68A/68B Colab tabs pass (the real failure mode).
2. Every registered flag is forwarded into `CrossConditionConfig` (no "registered but dropped").
3. New EXP-68 flags keep their OFF / legacy defaults so old checkpoints still load strict.
"""

import ast
import json
import os
import re
import shlex
import sys
import unittest

import numpy as np
import torch  # noqa: F401  (train.py pulls torch at import time)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition import train  # noqa: E402

TRAIN_PY = os.path.join(REPO_ROOT, "src", "PECT_JEPA", "cross_condition", "train.py")
COLAB_DIR = os.path.join(REPO_ROOT, "src", "PECT_JEPA", "cross_condition", "colab")

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


def _notebook_flags(nb_path):
    """Extract the literal --flags a Colab tab passes to train.py from its config cell."""
    nb = json.load(open(nb_path, encoding="utf-8"))
    argv = []
    found = False
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for var in ("BASE_FLAGS", "EXTRA"):
            m = re.search(rf"^{var}\s*=\s*(['\"])(.*?)\1", src, re.MULTILINE | re.DOTALL)
            if m:
                argv += shlex.split(m.group(2))
                found = True
    if not found:
        raise AssertionError(f"no BASE_FLAGS/EXTRA found in {nb_path}")
    return argv


class TestTrainCLIContract(unittest.TestCase):
    def test_parser_is_not_empty(self):
        flags = _parser_flags()
        self.assertGreater(len(flags), 50, f"parser only exposes {len(flags)} flags")

    def test_every_flag_is_forwarded_into_config(self):
        """A flag that is registered but never forwarded is silently dead (the EXP-68 failure)."""
        flags = {f.lstrip("-") for f in _parser_flags()}
        kwargs = set(_config_kwargs())

        dropped = sorted(flags - FLAGS_NOT_IN_CONFIG - kwargs)
        self.assertEqual(dropped, [], f"flag(s) registered but never passed to CrossConditionConfig: {dropped}")

        stray = sorted(kwargs - flags - CONFIG_FIELDS_WITHOUT_FLAG)
        self.assertEqual(stray, [], f"config arg(s) with no CLI flag or documented default: {stray}")

    def test_exp68_flags_default_to_off_and_legacy(self):
        """AGENTS invariant: new flags default OFF so old checkpoints load strict."""
        a = _parse_args([])
        self.assertIs(a.unified_latent, False)
        self.assertIs(a.spectral_condition_token, False)
        self.assertEqual(a.spectral_phase_encoding, "cos_sin")
        self.assertEqual(a.ref_patch_mode, "legacy")
        self.assertEqual(a.pred_loss_type, "l1")
        self.assertEqual(a.anti_hijack_ratio, 0.0)
        self.assertEqual(a.self_focus_gamma, 0.0)

    def test_exp68_flags_are_accepted_and_parsed(self):
        """The exact invocation that failed on Colab."""
        a = _parse_args([
            "--spectral_condition_token", "true",
            "--spectral_phase_encoding", "cos_sin",
            "--ref_patch_mode", "corner_median",
            "--pred_loss_type", "normalized_l1",
            "--unified_latent", "true",
        ])
        self.assertIs(a.spectral_condition_token, True)
        self.assertEqual(a.spectral_phase_encoding, "cos_sin")
        self.assertEqual(a.ref_patch_mode, "corner_median")
        self.assertEqual(a.pred_loss_type, "normalized_l1")
        self.assertIs(a.unified_latent, True)

    def test_raw_phase_encoding_is_a_choice(self):
        a = _parse_args(["--spectral_phase_encoding", "raw"])
        self.assertEqual(a.spectral_phase_encoding, "raw")

    def test_unknown_phase_encoding_is_rejected(self):
        with self.assertRaises(SystemExit):
            _parse_args(["--spectral_phase_encoding", "cos_sin_v2"])

    def test_legacy_ref_patch_is_the_default_for_exp63_reproduction(self):
        a = _parse_args(["--pred_loss_type", "raw_l1"])
        self.assertEqual(a.ref_patch_mode, "legacy",
                         "68A/A-B comparison must not silently switch the background patch")


class TestColabTabsMatchParser(unittest.TestCase):
    """Replay the real argv of each Colab tab through argparse.

    This is the regression test for the reported Colab crash: the tab and the parser must agree.
    """

    TABS = {
        "train_68a.ipynb": {"pred_loss_type": "normalized_l1", "ref_patch_mode": "legacy",
                            "unified_latent": False},
        "train_68b.ipynb": {"pred_loss_type": "normalized_l1", "ref_patch_mode": "corner_median",
                            "unified_latent": True, "spectral_phase_encoding": "cos_sin"},
    }

    def test_tabs_parse(self):
        for name, expected in self.TABS.items():
            path = os.path.join(COLAB_DIR, name)
            if not os.path.exists(path):
                self.skipTest(f"{name} not present (colab/ is gitignored)")
            with self.subTest(tab=name):
                argv = _notebook_flags(path) + [
                    "--exp_name", "cli_contract_test",
                    "--data_dir", "data",
                    "--epochs", "1",
                ]
                try:
                    a = _parse_args(argv)
                except SystemExit as e:  # argparse exits 2 on an unknown flag
                    known = {f.lstrip("-") for f in _parser_flags()}
                    unknown = sorted({t.lstrip("-") for t in argv
                                      if t.startswith("--") and t.lstrip("-") not in known})
                    self.fail(f"{name} passes flag(s) train.py does not register: {unknown}")
                for attr, value in expected.items():
                    self.assertEqual(getattr(a, attr), value,
                                     f"{name}: --{attr} parsed as {getattr(a, attr)!r}, expected {value!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
