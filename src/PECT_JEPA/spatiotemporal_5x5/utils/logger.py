"""
Unified Experiment Tracker & Multi-Tier Logger for 5x5 Spatiotemporal PECT-JEPA.

Inherits full feature parity with 1D temporal logger:
1. PyTorch TensorBoard SummaryWriter (live scalar curves, parameter histograms, C-Scan heatmaps)
2. Python logging module (dual console + rotating file log with timestamps)
3. Structured CSV metrics (metrics_step.csv & metrics_epoch.csv for paper/pandas analysis)
4. Optional Weights & Biases (wandb) integration
5. Experiment snapshotting (config.json, git_info.json, best/latest model checkpoint binding)
"""

import os
import sys
import glob
import json
import csv
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


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
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        info["dirty"] = "true" if len(status) > 0 else "false"
    except Exception:
        pass
    return info


class PECTExperimentLogger5x5:
    """
    Multi-tier experiment logger for 5x5 Spatiotemporal PECT-JEPA.
    """

    def __init__(self, config: Any, rank: int = 0):
        self.config = config
        self.rank = rank
        self.is_main = (rank == 0)

        # 1. Resolve Experiment Directory and Run Name
        exp_name = getattr(config, "exp_name", "pect_jepa_5x5_base")
        log_base_dir = getattr(config, "log_dir", "experiments/5x5")
        add_timestamp = getattr(config, "add_timestamp", False)

        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{exp_name}_{timestamp}"
        else:
            self.run_name = exp_name
        self.run_dir = os.path.join(log_base_dir, self.run_name)

        if self.is_main:
            os.makedirs(self.run_dir, exist_ok=True)

        # 2. Setup Python Logging Handler
        self.logger = logging.getLogger(f"PECT_JEPA_5x5_{self.run_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        if self.is_main:
            # Console Formatter
            c_formatter = logging.Formatter("[%(asctime)s | %(levelname)s] %(message)s", datefmt="%H:%M:%S")
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(c_formatter)
            self.logger.addHandler(ch)

            # File Formatter
            f_formatter = logging.Formatter(
                "[%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            log_file = os.path.join(self.run_dir, "train.log")
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(f_formatter)
            self.logger.addHandler(fh)

        # 3. Setup TensorBoard
        self.use_tb = getattr(config, "use_tensorboard", True) and TENSORBOARD_AVAILABLE and self.is_main
        self.tb_writer = None
        if self.use_tb:
            self.tb_writer = SummaryWriter(log_dir=self.run_dir)
            cfg_dict = config.to_dict() if hasattr(config, "to_dict") else {}
            self.tb_writer.add_text("Experiment/Config", json.dumps(cfg_dict, indent=2), 0)

        # 4. Setup WandB
        self.use_wandb = getattr(config, "use_wandb", False) and WANDB_AVAILABLE and self.is_main
        if self.use_wandb:
            wandb_proj = getattr(config, "wandb_project", "PECT_JEPA_5x5")
            wandb_ent = getattr(config, "wandb_entity", None)
            cfg_dict = config.to_dict() if hasattr(config, "to_dict") else {}
            wandb.init(project=wandb_proj, entity=wandb_ent, name=self.run_name, config=cfg_dict, dir=self.run_dir)

        # 5. Setup Structured CSV files
        self.step_csv_file = None
        self.epoch_csv_file = None
        self.step_csv_writer = None
        self.epoch_csv_writer = None

        if self.is_main:
            # Step CSV
            step_csv_path = os.path.join(self.run_dir, "metrics_step.csv")
            step_exists = os.path.exists(step_csv_path) and os.path.getsize(step_csv_path) > 0
            self.step_csv_file = open(step_csv_path, "a" if step_exists else "w", newline="", encoding="utf-8")
            self.step_csv_cols = ["step", "epoch", "loss", "loss_pred", "loss_var", "loss_cov", "lr", "momentum", "grad_norm"]
            self.step_csv_writer = csv.DictWriter(self.step_csv_file, fieldnames=self.step_csv_cols)
            if not step_exists:
                self.step_csv_writer.writeheader()
            self.step_csv_file.flush()

            # Epoch CSV
            epoch_csv_path = os.path.join(self.run_dir, "metrics_epoch.csv")
            epoch_exists = os.path.exists(epoch_csv_path) and os.path.getsize(epoch_csv_path) > 0
            self.epoch_csv_file = open(epoch_csv_path, "a" if epoch_exists else "w", newline="", encoding="utf-8")
            self.epoch_csv_cols = ["epoch", "train_loss", "val_loss", "effective_rank", "probe_cnr", "lr", "time_sec"]
            self.epoch_csv_writer = csv.DictWriter(self.epoch_csv_file, fieldnames=self.epoch_csv_cols)
            if not epoch_exists:
                self.epoch_csv_writer.writeheader()
            self.epoch_csv_file.flush()

            # 6. Save Config & Git Snapshot
            config_snapshot_path = os.path.join(self.run_dir, "config.json")
            if hasattr(config, "save_json"):
                config.save_json(config_snapshot_path)
            else:
                with open(config_snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(config.__dict__, f, indent=2)

            git_info = get_git_revision_info()
            git_info_path = os.path.join(self.run_dir, "git_info.json")
            with open(git_info_path, "w", encoding="utf-8") as f:
                json.dump(git_info, f, indent=2)

            self.info(f"=== 5x5 PECT-JEPA Experiment Initialized: {self.run_name} ===")
            self.info(f"Run Directory: {self.run_dir}")
            self.info(f"TensorBoard Active: {self.use_tb} | WandB Active: {self.use_wandb}")

    def info(self, msg: str):
        if self.is_main:
            self.logger.info(msg)

    def warning(self, msg: str):
        if self.is_main:
            self.logger.warning(msg)

    def error(self, msg: str):
        if self.is_main:
            self.logger.error(msg)

    def log_step(self, step: int, metrics: Dict[str, Any], epoch: int = 0):
        """Logs high-frequency step metrics (losses, lr, momentum, grad norm)."""
        if not self.is_main:
            return

        if self.tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    self.tb_writer.add_scalar(f"train_step/{k}", float(v), step)

        if self.use_wandb:
            wandb.log({f"train_step/{k}": v for k, v in metrics.items()}, step=step)

        if self.step_csv_writer:
            row = {col: metrics.get(col, "") for col in self.step_csv_cols}
            row["step"] = step
            row["epoch"] = epoch
            self.step_csv_writer.writerow(row)
            if step % 50 == 0:
                self.step_csv_file.flush()

    def log_epoch(self, epoch: int, metrics: Dict[str, Any], step: int = 0):
        """Logs epoch-level aggregated metrics."""
        if not self.is_main:
            return

        if self.tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    prefix = "val" if "val" in k else "train"
                    self.tb_writer.add_scalar(f"{prefix}_epoch/{k}", float(v), epoch)

        if self.use_wandb:
            wandb.log({f"epoch/{k}": v for k, v in metrics.items()}, step=step)

        if self.epoch_csv_writer:
            row = {col: metrics.get(col, "") for col in self.epoch_csv_cols}
            row["epoch"] = epoch
            self.epoch_csv_writer.writerow(row)
            self.epoch_csv_file.flush()

    def log_figure(self, tag: str, figure, global_step: int):
        """Logs a matplotlib figure directly to TensorBoard / WandB."""
        if not self.is_main:
            return
        if self.tb_writer:
            self.tb_writer.add_figure(tag, figure, global_step=global_step)
        if self.use_wandb:
            wandb.log({tag: wandb.Image(figure)}, step=global_step)

    def log_image(self, tag: str, image_tensor_or_np, global_step: int, dataformats: str = "HWC"):
        """Logs image array / C-scan anomaly map to TensorBoard / WandB."""
        if not self.is_main:
            return
        if self.tb_writer:
            self.tb_writer.add_image(tag, image_tensor_or_np, global_step=global_step, dataformats=dataformats)
        if self.use_wandb:
            wandb.log({tag: wandb.Image(image_tensor_or_np)}, step=global_step)

    def close(self):
        """Safely flushes and closes all handlers."""
        if self.is_main:
            if self.tb_writer:
                self.tb_writer.flush()
                self.tb_writer.close()
            if self.step_csv_file:
                self.step_csv_file.flush()
                self.step_csv_file.close()
            if self.epoch_csv_file:
                self.epoch_csv_file.flush()
                self.epoch_csv_file.close()
            if self.use_wandb and wandb.run:
                wandb.finish()
            self.info(f"=== Experiment Logger Closed for {self.run_name} ===")
