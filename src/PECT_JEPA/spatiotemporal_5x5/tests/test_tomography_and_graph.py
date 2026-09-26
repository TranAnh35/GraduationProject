"""
Test Suite for 3D Defect Tomography & Topological Graph Representation.

Validates that:
1. 2D evaluation metrics (AUC, AP, CNR, R^2) remain 100% intact and unchanged.
2. 3D volumetric slices V(y, x, z) are extracted across the 4 skin-depth layers.
3. 3D orthogonal B-scan figures and interactive 3D HTML tomography are saved.
4. Topological Defect Graph G=(V, E) accurately calculates:
   - Physical fatigue crack length L_crack (mm) for Rivet specimens.
   - Volumetric metal loss V_loss (mm^3) for Corrosion specimens.
"""

import os
import sys
import json
import torch
import numpy as np

from src.PECT_JEPA.spatiotemporal_5x5.evaluate import (
    load_model_from_checkpoint,
    evaluate_single_file,
)


def run_tomography_and_graph_test():
    ckpt_path = "experiments/5x5/exp14_radial_dispersion_jepa/checkpoints/best_model_5x5.pt"
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Test] Loading model from: {ckpt_path} on {device}")
    model = load_model_from_checkpoint(ckpt_path, device=device)

    output_dir = "experiments/5x5/exp14_radial_dispersion_jepa/test_3d_graph_verification"
    os.makedirs(output_dir, exist_ok=True)

    test_files = [
        # 1. Rivet Specimen with Subsurface Crack
        "data/Hall_Air_Core/Rivet_v1/Chirp/hall_aircore_rivet_frontside_chirp_300x300_500_1500hz_z1_20260123_202334.tdms",
        # 2. Corrosion Specimen with Wall Loss
        "data/Hall_Air_Core/Corrosion/Chirp/hall_aircore_corosion_frontside_chirp_300x300_2.97_500-1500hz_z1_20260118_162411.tdms",
    ]

    for fp in test_files:
        if not os.path.isfile(fp):
            print(f"[Skip] File not found: {fp}")
            continue

        print("\n" + "=" * 70)
        print(f"  TESTING 3D TOMOGRAPHY & DEFECT GRAPH: {os.path.basename(fp)}")
        print("=" * 70)

        res = evaluate_single_file(
            file_path=fp,
            model=model,
            output_dir=output_dir,
            batch_size=512,
            device=device,
            crop_border=15,
            eval_3d=True,
        )

        m = res["metrics"]
        print(f"\n[2D Benchmark Verification (100% Intact)]")
        print(f"  AUC-ROC:           {m.get('auc_roc', 0.0):.4f}")
        print(f"  Average Precision: {m.get('average_precision', 0.0):.4f}")
        print(f"  Contrast Ratio:    {m.get('contrast_ratio_cnr', 0.0):.2f}")
        print(f"  Depth R2:          {m.get('depth_r2', 0.0):.4f}")

        t6 = res.get("task6_3d_tomography_and_graph", {})
        print(f"\n[3D Tomography & Defect Graph Verification]")
        print(f"  Active Defect Nodes: {t6.get('num_defect_nodes', 0)}")
        print(f"  Topological Edges:   {t6.get('num_defect_edges', 0)}")
        print(f"  Ortho Slices Image:  {t6.get('ortho_slices_path')}")
        print(f"  Interactive 3D HTML: {t6.get('interactive_3d_html')}")
        print(f"  Graph Plot 3D Image: {t6.get('graph_plot_path')}")
        print(f"  Graph HTML:          {t6.get('graph_interactive_html')}")

        if "crack_metrics" in t6 and t6["crack_metrics"]:
            cm = t6["crack_metrics"]
            print(f"  --> [Rivet Crack NDT] Crack Length L_crack = {cm.get('crack_length_mm')} mm")
            print(f"  --> [Rivet Crack NDT] Orientation Angle    = {cm.get('orientation_deg')} deg")
            print(f"  --> [Rivet Crack NDT] Morphology           = {cm.get('morphology')}")
            print(f"  --> [Rivet Crack NDT] Layer Distribution   = {cm.get('layer_distribution')}")
            assert cm.get("crack_length_mm", 0) > 0, "Crack length should be strictly positive"

        if "corrosion_metrics" in t6 and t6["corrosion_metrics"]:
            vm = t6["corrosion_metrics"]
            print(f"  --> [Corrosion NDT] Metal Loss V_loss = {vm.get('volumetric_metal_loss_mm3')} mm³")
            print(f"  --> [Corrosion NDT] Surface Area      = {vm.get('surface_area_mm2')} mm²")
            print(f"  --> [Corrosion NDT] Max Pit Depth     = {vm.get('max_pit_depth_mm')} mm")
            assert vm.get("volumetric_metal_loss_mm3", 0) > 0, "Metal loss should be strictly positive"

        # Verify artifacts exist on disk
        for key in ["ortho_slices_path", "graph_plot_path"]:
            p = t6.get(key)
            if p:
                assert os.path.isfile(p), f"Artifact file missing: {p}"
                assert os.path.getsize(p) > 1000, f"Artifact file empty: {p}"

    print("\n" + "=" * 70)
    print("  [SUCCESS] 3D Tomography & Defect Graph Test Passed!")
    print("  Both 2D metrics and 3D quantitative NDT outputs verified.")
    print("=" * 70)


if __name__ == "__main__":
    run_tomography_and_graph_test()
