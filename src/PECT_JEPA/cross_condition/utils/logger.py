"""
Unified Experiment Tracker & Multi-Tier Logger for Cross-Condition PECT-JEPA.
Provides:
- Console and file logging with clean formatting.
- Structured CSV metric tracking (metrics_epoch.csv).
- Checkpoint directory and run metadata persistence (config.json, git_info.json).
"""

import os
import sys
import json
import csv
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional


def get_git_revision_info() -> Dict[str, str]:
    """Retrieves current git commit hash, branch name, and dirty status."""
    info = {"commit": "unknown", "branch": "unknown", "dirty": "unknown"}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        info["commit"] = commit
    except Exception:
        pass
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        info["branch"] = branch
    except Exception:
        pass
    return info


class CrossConditionLogger:
    """
    Multi-tier experiment logger for Cross-Condition PECT-JEPA.
    Creates a dedicated timestamped run folder under experiments/cross_condition/runs/<exp_name>_<timestamp>/
    """
    def __init__(self, run_dir: str, run_name: str, config: Any):
        self.run_dir = run_dir
        self.run_name = run_name
        self.config = config

        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        self.results_dir = os.path.join(run_dir, "results")
        self.visualizations_dir = os.path.join(run_dir, "visualizations")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

        # Setup python logging
        self.logger = logging.getLogger(f"CrossCondition_{run_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        c_formatter = logging.Formatter("[%(asctime)s | %(levelname)s] %(message)s", datefmt="%H:%M:%S")
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(c_formatter)
        self.logger.addHandler(ch)

        log_file = os.path.join(self.run_dir, "run.log")
        f_formatter = logging.Formatter("[%(asctime)s | %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(f_formatter)
        self.logger.addHandler(fh)

        # Save config and git info
        cfg_dict = vars(config) if hasattr(config, "__dict__") else dict(config)
        # Convert non-serializable fields if any
        serializable_cfg = {}
        for k, v in cfg_dict.items():
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                serializable_cfg[k] = v
            else:
                serializable_cfg[k] = str(v)

        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(serializable_cfg, f, indent=2)

        with open(os.path.join(self.run_dir, "git_info.json"), "w", encoding="utf-8") as f:
            json.dump(get_git_revision_info(), f, indent=2)

        self.metrics_csv_path = os.path.join(self.run_dir, "metrics_epoch.csv")
        self.csv_initialized = False

    def info(self, msg: str):
        self.logger.info(msg)

    def log_epoch(self, metrics: Dict[str, Any]):
        fieldnames = list(metrics.keys())
        write_header = not os.path.exists(self.metrics_csv_path)

        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
