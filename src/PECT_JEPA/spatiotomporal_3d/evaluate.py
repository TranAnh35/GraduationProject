"""
Downstream Evaluation Script for PECT-JEPA (Spatio-Temporal 3D).
Runs downstream evaluations using frozen pretrained encoder with tqdm progress tracking:
1. Anomaly Detection
2. Cross-Sensor evaluation
3. Cross-Wave evaluation
4. Cross-Lift-off evaluation
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.PECT_JEPA.spatiotomporal_3d.models.jepa import PECT_JEPA
from src.PECT_JEPA.spatiotomporal_3d.configs.config import get_default_config, PECTJEPAConfig
from src.PECT_JEPA.spatiotomporal_3d.data.dataset import find_all_tdms_files
from src.PECT_JEPA.spatiotomporal_3d.evaluation.cross_sensor import evaluate_cross_sensor
from src.PECT_JEPA.spatiotomporal_3d.evaluation.cross_wave import evaluate_cross_wave
from src.PECT_JEPA.spatiotomporal_3d.evaluation.cross_liftoff import evaluate_cross_liftoff


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PECT-JEPA representations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing TDMS files")
    parser.add_argument("--task", type=str, default="all", choices=["all", "sensor", "wave", "liftoff"], help="Evaluation task")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load checkpoint
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Initialize model
    config_dict = checkpoint.get("config", {})
    if config_dict:
        config = PECTJEPAConfig.from_dict(config_dict)
    else:
        config = get_default_config()

    model = PECT_JEPA(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_files = find_all_tdms_files(args.data_dir)
    print(f"Found {len(all_files)} total TDMS files in {args.data_dir}")
    if len(all_files) == 0:
        print("ERROR: No TDMS files found.")
        return

    # 1. Cross-Sensor
    if args.task in ["all", "sensor"]:
        print("\n" + "=" * 55)
        print("RUNNING CROSS-SENSOR EVALUATION (Section 18.2)")
        print("=" * 55)
        sensors = ["TMR", "Hall_Pot_Core", "Differential_Pot_Core", "Hall_Air_Core"]
        for target_sensor in tqdm(sensors, desc="Cross-Sensor Tasks", dynamic_ncols=True):
            res = evaluate_cross_sensor(model, all_files, test_sensor=target_sensor, device=args.device)
            print(f"Target Sensor: {target_sensor} | Evaluated {res.get('num_test_files', 0)} files")

    # 2. Cross-Wave
    if args.task in ["all", "wave"]:
        print("\n" + "=" * 55)
        print("RUNNING CROSS-WAVE EVALUATION (Section 18.3)")
        print("=" * 55)
        waveforms = ["Chirp", "Gaussian", "Square"]
        for target_wave in tqdm(waveforms, desc="Cross-Wave Tasks", dynamic_ncols=True):
            res = evaluate_cross_wave(model, all_files, test_waveform=target_wave, device=args.device)
            print(f"Target Waveform: {target_wave} | Evaluated {res.get('num_test_files', 0)} files")

    # 3. Cross-Lift-off
    if args.task in ["all", "liftoff"]:
        print("\n" + "=" * 55)
        print("RUNNING CROSS-LIFT-OFF EVALUATION (Section 18.4)")
        print("=" * 55)
        liftoffs = ["3mm", "2mm", "1mm"]
        for target_liftoff in tqdm(liftoffs, desc="Cross-Liftoff Tasks", dynamic_ncols=True):
            res = evaluate_cross_liftoff(model, all_files, test_liftoff=target_liftoff, device=args.device)
            print(f"Target Lift-off: {target_liftoff} | Evaluated {res.get('num_test_files', 0)} files")

    print("\n--- Evaluation Finished ---")


if __name__ == "__main__":
    main()
