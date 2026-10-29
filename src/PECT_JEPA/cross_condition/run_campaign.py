"""Run from any working directory: python run_campaign.py plan|run|status ..."""
import os
from pathlib import Path
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.PECT_JEPA.cross_condition.campaign.runner import cli

if __name__ == "__main__":
    cli()
