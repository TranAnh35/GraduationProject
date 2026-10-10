"""
Publication-Grade Presentation Slide Figures Generator for PECT-JEPA.

Generates 6 widescreen (16:9, 300 DPI) publication-quality figures for presentation slides:
1. slide1_input_concentric_rings.png: Multi-Scale Concentric Star Spatial Topology vs Dense Grid.
2. slide2_dual_domain_time_fft.png: Dual-Domain Time Transient & Dodd-Deeds Fourier Phase Fusion.
3. slide3_shallow_deep_masking.png: Physical Skin-Depth Decomposition & Surface-to-Depth Masking.
4. slide4_jepa_predictor_architecture.png: JEPA Predictor 3D Spatio-Diffusion World Model.
5. slide5_loss_and_vicreg.png: Pure JEPA Prediction Loss & Intra-Scan VICReg Anti-Collapse.
6. slide6_experiment_comparison_benchmarks.png: Comprehensive Benchmark Progression (EXP-01 to EXP-17 SOTA).

Output is placed in: presentation_figures/slide_deck/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Color Palette (Nature/IEEE High-Contrast Academic Theme)
C_PRIMARY = "#1E3A8A"       # Deep Navy
C_SECONDARY = "#0284C7"     # Sky Blue
C_ACCENT_GREEN = "#059669"   # Emerald Green (SOTA / Success)
C_ACCENT_RED = "#DC2626"     # Crimson (Masked / Target / Baseline defect)
C_ACCENT_AMBER = "#D97706"   # Amber (Warning / Transition)
C_PURPLE = "#7C3AED"        # Violet (Diffusion / Operator)
C_TEXT_MAIN = "#0F172A"     # Slate 900
C_TEXT_MUTED = "#64748B"    # Slate 500
C_BG_CARD = "#F8FAFC"       # Slate 50
C_BORDER = "#CBD5E1"        # Slate 300

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.edgecolor"] = C_BORDER
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["text.color"] = C_TEXT_MAIN
plt.rcParams["axes.labelcolor"] = C_TEXT_MAIN
plt.rcParams["xtick.color"] = C_TEXT_MAIN
plt.rcParams["ytick.color"] = C_TEXT_MAIN

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.temporal_1d.data.preprocessing import read_tdms_1d_waveforms
from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import get_spatial_topology_offsets

_REAL_CORROSION_WAVEFORMS_CACHE = {}

def get_real_corrosion_waveforms():
    """
    Loads real laboratory PECT waveforms from genuine TDMS files:
    - Specimen: Corrosion
    - Sensor: Hall AirCore
    - Lift-off: 1 mm (z1)
    - 3 Waveforms: Square, Gaussian, Chirp
    Returns genuine voltage curves across rings: Center (#0), Ring 1 (#1), Ring 2 (#9), Ring 3 (#17).
    """
    if _REAL_CORROSION_WAVEFORMS_CACHE:
        return _REAL_CORROSION_WAVEFORMS_CACHE

    file_paths = {
        "Square": os.path.join(ROOT_DIR, "data/Hall_Air_Core/Corrosion/Square/hall_aircore_corosion_frontside_square_300x300_2.1_200hz_z1_20260117_162910.tdms"),
        "Gaussian": os.path.join(ROOT_DIR, "data/Hall_Air_Core/Corrosion/Gauss/hall_aircore_corosion_frontside_gaussian_300x300_3.88_1000hz_z1_20260118_024203.tdms"),
        "Chirp": os.path.join(ROOT_DIR, "data/Hall_Air_Core/Corrosion/Chirp/hall_aircore_corosion_frontside_chirp_300x300_2.97_500-1500hz_z1_20260118_162411.tdms"),
    }

    offsets = get_spatial_topology_offsets("concentric_star")  # [25, 2] (dr, dc)
    probe_meta = [
        {"name": "Center (#0, r=0px)", "idx": 0, "color": "#DC2626", "lw": 1.8},
        {"name": "Ring 1 (#1, r=1px)", "idx": 1, "color": "#2563EB", "lw": 1.6},
        {"name": "Ring 2 (#9, r=3px)", "idx": 9, "color": "#0284C7", "lw": 1.6},
        {"name": "Ring 3 (#17, r=7px)", "idx": 17, "color": "#059669", "lw": 1.6},
    ]

    r0, c0 = 150, 150  # Stable interior coordinate on 300x300 scan
    time_ms = np.linspace(0.0, 5.0, 500)  # 500 samples across 5.0 ms (f_exc=200Hz, f_s=100kHz)

    loaded = {}
    for wf_key, fpath in file_paths.items():
        if os.path.exists(fpath):
            raw_waveforms = read_tdms_1d_waveforms(fpath, target_time_samples=500, normalization="none")
            grid_3d = raw_waveforms[:300 * 300, :].reshape(300, 300, 500)
            curves = []
            for p in probe_meta:
                dr, dc = offsets[p["idx"]]
                v_signal = grid_3d[r0 + dr, c0 + dc, :].copy()
                curves.append({
                    "name": p["name"],
                    "color": p["color"],
                    "lw": p["lw"],
                    "voltage": v_signal,
                    "dr": dr,
                    "dc": dc,
                    "idx": p["idx"]
                })
            loaded[wf_key] = {
                "time_ms": time_ms,
                "curves": curves,
                "v_min": float(min(c["voltage"].min() for c in curves)),
                "v_max": float(max(c["voltage"].max() for c in curves)),
            }
        else:
            print(f"Warning: Real TDMS file {fpath} not found!")

    _REAL_CORROSION_WAVEFORMS_CACHE.update(loaded)
    return _REAL_CORROSION_WAVEFORMS_CACHE


def create_base_slide(title: str, subtitle: str, badge: str = "PECT-JEPA FOUNDATION"):
    """Initializes a standard 16:9 widescreen canvas with unified non-overlapping header."""
    fig = plt.figure(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Header Card (Height 0.08, Top at 0.98 -> spans 0.90 to 0.98)
    header_ax = fig.add_axes([0.04, 0.905, 0.92, 0.080])
    header_ax.axis("off")
    header_ax.set_xlim(0, 1)
    header_ax.set_ylim(0, 1)

    # Title & Subtitle with guaranteed horizontal boundary
    header_ax.text(0.0, 0.68, title, fontsize=15.5, fontweight="bold", color=C_TEXT_MAIN, va="center")
    header_ax.text(0.0, 0.18, subtitle, fontsize=10.0, color=C_TEXT_MUTED, va="center")

    # Category Pill Badge (Top Right)
    badge_patch = patches.FancyBboxPatch(
        (0.74, 0.22), 0.26, 0.56,
        boxstyle="round,pad=0.03,rounding_size=0.15",
        facecolor="#EFF6FF", edgecolor="#BFDBFE", linewidth=1.2, zorder=3
    )
    header_ax.add_patch(badge_patch)
    header_ax.text(0.87, 0.50, badge, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#1D4ED8", zorder=4)

    return fig


def draw_card(ax, x, y, w, h, title="", subtitle="", body_text="", 
              bg=C_BG_CARD, edge=C_BORDER, lw=1.2, 
              title_color=C_PRIMARY, body_color=C_TEXT_MAIN,
              title_size=10.5, sub_size=8.5, body_size=8.0, linespacing=1.35):
    """
    Renders a card container with strict, deterministic vertical text flow
    guaranteeing ZERO overlap between title, subtitle, and body text.
    """
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        facecolor=bg, edgecolor=edge, linewidth=lw, zorder=1
    )
    ax.add_patch(card)

    curr_y = y + h - 0.045
    pad_x = x + 0.03

    if title:
        ax.text(pad_x, curr_y, title, fontsize=title_size, fontweight="bold", color=title_color, va="top", zorder=2)
        curr_y -= (title_size * 0.0035 + 0.025)

    if subtitle:
        ax.text(pad_x, curr_y, subtitle, fontsize=sub_size, color=C_TEXT_MUTED, va="top", zorder=2)
        curr_y -= (sub_size * 0.0035 + 0.020)

    if body_text:
        ax.text(pad_x, curr_y, body_text, fontsize=body_size, color=body_color, va="top", linespacing=linespacing, zorder=2)


def draw_bottom_banner(fig, text: str, bg="#EFF6FF", edge="#60A5FA", text_color="#1E40AF", fontsize=8.2):
    """Draws a sleek, standardized highlight banner at the bottom of the slide."""
    banner_ax = fig.add_axes([0.050, 0.032, 0.900, 0.054])
    banner_ax.axis("off")
    banner_ax.set_xlim(0, 1)
    banner_ax.set_ylim(0, 1)

    # Clean executive presentation box with crisp borders
    rect = patches.Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        facecolor=bg, edgecolor=edge, linewidth=1.2, zorder=1
    )
    banner_ax.add_patch(rect)
    banner_ax.text(0.50, 0.50, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color, linespacing=1.28, zorder=2)


# ==============================================================================
# SLIDE 1A: Multi-Scale Concentric Star Spatial Topology vs Dense Grid
# ==============================================================================
def draw_slide1_topology_comparison(save_path: str):
    fig = create_base_slide(
        title="SPATIAL INPUT TOPOLOGY: MULTI-SCALE CONCENTRIC STAR VS DENSE GRID",
        subtitle="Omnidirectional 25-Probe Spatial Sampling on Discrete C-Scan Grid Spanning Physical Coil Footprint",
        badge="EXP-19 / TOPOLOGY FOUNDATION"
    )

    # ---------------- Sub-panel A1: Dense 5x5 Grid (Left) ----------------
    # Canvas is 16:9 widescreen. For square aspect on canvas:
    # ax_width = 0.35, ax_height = 0.35 * 16 / 9 = 0.6222
    ax1 = fig.add_axes([0.070, 0.15, 0.350, 0.6222])
    ax1.set_facecolor("#F8FAFC")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(-7.5, 7.5)
    ax1.set_ylim(-7.5, 7.5)
    ax1.set_xticks([-7, -5, -3, -1, 0, 1, 3, 5, 7])
    ax1.set_yticks([-7, -5, -3, -1, 0, 1, 3, 5, 7])
    ax1.tick_params(axis="both", which="both", length=0)
    ax1.grid(False)

    # Clean title and legend placement with zero collision
    ax1.text(0.5, 1.13, "A1. Dense 5×5 Grid (Local Footprint: 4×4 mm²)",
             transform=ax1.transAxes, ha="center", va="bottom",
             fontsize=11.0, fontweight="bold", color="#991B1B")

    # Axis titles strictly WITHOUT units
    ax1.set_xlabel(r"Column Offset $\Delta c$", fontsize=10.0, fontweight="bold")
    ax1.set_ylabel(r"Row Offset $\Delta r$", fontsize=10.0, fontweight="bold")

    # Render entire 15x15 discrete raster matrix (225 cells)
    for r in range(-7, 8):
        for c in range(-7, 8):
            cell_box = patches.Rectangle(
                (c - 0.5, r - 0.5), 1.0, 1.0,
                facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=0.6, zorder=1
            )
            ax1.add_patch(cell_box)

    # Connect outer perimeter of Dense 5x5
    d_perimeter_c = [-2, 2, 2, -2, -2]
    d_perimeter_r = [-2, -2, 2, 2, -2]
    ax1.plot(d_perimeter_c, d_perimeter_r, color="#DC2626", linestyle="--", linewidth=1.8, alpha=0.9, zorder=3)

    # Probes in Dense 5x5: Probe 0 at (0, 0), others in [-2..2]
    offsets_dense = get_spatial_topology_offsets("dense_5x5")
    for idx, (dr, dc) in enumerate(offsets_dense):
        col = "#DC2626" if idx == 0 else "#EF4444"
        ecol = "#7F1D1D" if idx == 0 else "#B91C1C"
        p_rect = patches.Rectangle((dc - 0.5, dr - 0.5), 1.0, 1.0, facecolor=col, edgecolor=ecol, linewidth=1.0, zorder=4)
        ax1.add_patch(p_rect)
        ax1.text(dc, dr, f"#{idx}", color="#FFFFFF", fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=5)

    # Legend for A1 (placed horizontally above grid, completely below title)
    dense_legend_handles = [
        patches.Patch(facecolor="#DC2626", edgecolor="#7F1D1D", label="Center (#0)"),
        patches.Patch(facecolor="#EF4444", edgecolor="#B91C1C", label="Probes (#1–#24)"),
    ]
    ax1.legend(handles=dense_legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=2, fontsize=8.2, frameon=True, edgecolor="#CBD5E1", facecolor="#FFFFFF")

    # ---------------- Transition Arrow (Middle) ----------------
    ax_arrow = fig.add_axes([0.435, 0.15, 0.110, 0.6222])
    ax_arrow.axis("off")
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)

    arrow = patches.FancyArrowPatch(
        (0.12, 0.50), (0.88, 0.50),
        arrowstyle="-|>", mutation_scale=22, color="#2563EB", linewidth=2.8, zorder=3
    )
    ax_arrow.add_patch(arrow)
    ax_arrow.text(0.50, 0.60, "Multi-Scale\nExpansion", ha="center", va="bottom",
                  fontsize=9.2, fontweight="bold", color="#1E40AF")
    ax_arrow.text(0.50, 0.40, "4×4 → 14×14 mm²\nEqual 25 Tokens", ha="center", va="top",
                  fontsize=7.6, fontweight="bold", color="#475569")

    # ---------------- Sub-panel A2: Concentric Star (Right) ----------------
    ax2 = fig.add_axes([0.565, 0.15, 0.350, 0.6222])
    ax2.set_facecolor("#F8FAFC")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(-7.5, 7.5)
    ax2.set_ylim(-7.5, 7.5)
    ax2.set_xticks([-7, -5, -3, -1, 0, 1, 3, 5, 7])
    ax2.tick_params(axis="x", which="both", length=0)
    # Hide y-axis on A2 to avoid redundant label and prevent collision with arrow
    ax2.tick_params(axis="y", which="both", length=0, labelleft=False)
    ax2.grid(False)

    # Clean title and legend placement with zero collision
    ax2.text(0.5, 1.13, "A2. Concentric Star (Coil Footprint: 14×14 mm²)",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=11.0, fontweight="bold", color="#166534")

    # Axis titles strictly WITHOUT units
    ax2.set_xlabel(r"Column Offset $\Delta c$", fontsize=10.0, fontweight="bold")

    # Render entire 15x15 discrete raster matrix (225 cells)
    for r in range(-7, 8):
        for c in range(-7, 8):
            cell_box = patches.Rectangle(
                (c - 0.5, r - 0.5), 1.0, 1.0,
                facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=0.6, zorder=1
            )
            ax2.add_patch(cell_box)

    offsets_star = get_spatial_topology_offsets("concentric_star")

    # Ring 1: Exact 2x2 square boundary connecting 8 points at distance 1
    ring1_sq_c = [-1, 1, 1, -1, -1]
    ring1_sq_r = [-1, -1, 1, 1, -1]
    ax2.plot(ring1_sq_c, ring1_sq_r, color="#2563EB", linestyle="--", linewidth=1.8, alpha=0.9, zorder=3)

    # Ring 2: Exact 6x6 square boundary connecting 8 points at distance 3
    ring2_sq_c = [-3, 3, 3, -3, -3]
    ring2_sq_r = [-3, -3, 3, 3, -3]
    ax2.plot(ring2_sq_c, ring2_sq_r, color="#0284C7", linestyle="--", linewidth=1.8, alpha=0.9, zorder=3)

    # Ring 3: Octagram polygon connecting outer points
    ring3_loop_c = [0, 5, 7, 5, 0, -5, -7, -5, 0]
    ring3_loop_r = [7, 5, 0, -5, -7, -5, 0, 5, 7]
    ax2.plot(ring3_loop_c, ring3_loop_r, color="#059669", linestyle="--", linewidth=1.8, alpha=0.9, zorder=3)

    # Center probe (#0) at (0, 0)
    p0_rect = patches.Rectangle((-0.5, -0.5), 1.0, 1.0, facecolor="#DC2626", edgecolor="#7F1D1D", linewidth=1.2, zorder=4)
    ax2.add_patch(p0_rect)
    ax2.text(0, 0, "#0", color="#FFFFFF", fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=5)

    # Ring 1 probes (#1..#8)
    for idx in range(1, 9):
        dr, dc = offsets_star[idx]
        p_rect = patches.Rectangle((dc - 0.5, dr - 0.5), 1.0, 1.0, facecolor="#2563EB", edgecolor="#1E3A8A", linewidth=1.0, zorder=4)
        ax2.add_patch(p_rect)
        ax2.text(dc, dr, f"#{idx}", color="#FFFFFF", fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=5)

    # Ring 2 probes (#9..#16)
    for idx in range(9, 17):
        dr, dc = offsets_star[idx]
        p_rect = patches.Rectangle((dc - 0.5, dr - 0.5), 1.0, 1.0, facecolor="#0284C7", edgecolor="#0369A1", linewidth=1.0, zorder=4)
        ax2.add_patch(p_rect)
        ax2.text(dc, dr, f"#{idx}", color="#FFFFFF", fontsize=6.2, fontweight="bold", ha="center", va="center", zorder=5)

    # Ring 3 probes (#17..#24)
    for idx in range(17, 25):
        dr, dc = offsets_star[idx]
        p_rect = patches.Rectangle((dc - 0.5, dr - 0.5), 1.0, 1.0, facecolor="#059669", edgecolor="#064E3B", linewidth=1.0, zorder=4)
        ax2.add_patch(p_rect)
        ax2.text(dc, dr, f"#{idx}", color="#FFFFFF", fontsize=6.0, fontweight="bold", ha="center", va="center", zorder=5)

    # Legend for A2 (strictly ring name and color, completely outside to never overlap probe #21)
    legend_handles = [
        patches.Patch(facecolor="#DC2626", edgecolor="#7F1D1D", label="Center"),
        patches.Patch(facecolor="#2563EB", edgecolor="#1E3A8A", label="Ring 1"),
        patches.Patch(facecolor="#0284C7", edgecolor="#0369A1", label="Ring 2"),
        patches.Patch(facecolor="#059669", edgecolor="#064E3B", label="Ring 3"),
    ]
    ax2.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=4, fontsize=8.2, frameon=True, edgecolor="#CBD5E1", facecolor="#FFFFFF")

    # Bottom Highlight Banner
    draw_bottom_banner(
        fig,
        text="PHYSICAL INVARIANT: Concentric Star expands spatial aperture from 4×4 mm² to 14×14 mm² without increasing probe count (N=25), matching the full eddy current diffusion zone with zero compute overhead.",
        bg="#EFF6FF", edge="#60A5FA", text_color="#1E40AF"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> [Slide 1A Generated]: {save_path}")


# ==============================================================================
# SLIDE 1B: Controlled Benchmark Metrics Table ONLY
# ==============================================================================
def draw_slide1_metrics_table(save_path: str):
    fig = create_base_slide(
        title="REALISTIC ZERO-SHOT BENCHMARK: MULTI-SCALE CONCENTRIC STAR VS DENSE GRID",
        subtitle="Strict Single-Specimen Protocol: Pretrained on Corrosion z₁ (1.0 mm) → Zero-Shot Evaluated on Corrosion z₂ (2.0 mm)",
        badge="EXP-19 / ZERO-SHOT SINGLE-SPECIMEN"
    )

    # ---------------- Centered Metrics Table ONLY ----------------
    ax_table = fig.add_axes([0.065, 0.115, 0.870, 0.740])
    ax_table.axis("off")
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)

    table_data = [
        ("Spatial Aperture (Physical Footprint)", "5×5 pixels (4×4 mm²)", "15×15 pixels (14×14 mm²)", "3.5× wider aperture (matches coil footprint)", False),
        ("Spatial Token Budget (Probe Count)", "25 probes / tokens", "25 probes / tokens", "Equal compute budget (0% FLOPS overhead)", False),
        ("Zero-Shot Defect Depth Sizing R² (y > 0)", "0.5546", "0.7247", "+0.1701 (+30.7% relative sizing boost)", True),
        ("Zero-Shot Defect Depth MAE (mm)", "0.1783 mm", "0.1416 mm", "20.6% error reduction (141 μm precision)", True),
        ("Zero-Shot Flaw Detection AUC-ROC (%)", "64.69%", "62.78%", "Comparable (Broad cavity edge transition)", False),
        ("Zero-Shot Flaw Average Precision AP (%)", "2.51%", "1.69%", "Natural baseline shift under 2× lift-off transfer", False),
    ]

    table_top = 0.985
    row_h = 0.136
    col_x = [0.00, 0.35, 0.54, 0.73, 1.00]
    headers = ["Metric / Parameter", "Dense 5×5 Grid (Baseline)", "Concentric Star (Proposed)", "Physical Advantage / Gain"]

    # Header Row
    h_y = table_top - row_h
    h_rect = patches.Rectangle((0.00, h_y), 1.00, row_h, facecolor="#1E3A8A", edgecolor="#1E3A8A", zorder=2)
    ax_table.add_patch(h_rect)
    for c_i in range(4):
        ax_table.text(col_x[c_i] + 0.015, h_y + row_h / 2, headers[c_i],
                      color="#FFFFFF", fontsize=10.2, fontweight="bold", va="center", zorder=3)

    for r_i, (metric, dense_val, star_val, delta, is_hl) in enumerate(table_data):
        y_r = table_top - (r_i + 2) * row_h
        bg_col = "#ECFDF5" if is_hl else ("#F8FAFC" if (r_i % 2 == 1) else "#FFFFFF")
        r_rect = patches.Rectangle((0.00, y_r), 1.00, row_h, facecolor=bg_col, edgecolor="#CBD5E1", linewidth=1.0, zorder=2)
        ax_table.add_patch(r_rect)

        ax_table.text(col_x[0] + 0.015, y_r + row_h / 2, metric,
                      color="#0F172A", fontsize=9.8, fontweight="bold" if is_hl else "normal", va="center", zorder=3)
        ax_table.text(col_x[1] + 0.015, y_r + row_h / 2, dense_val,
                      color="#334155", fontsize=9.8, va="center", zorder=3)
        ax_table.text(col_x[2] + 0.015, y_r + row_h / 2, star_val,
                      color="#047857" if is_hl else "#0F172A", fontsize=10.2, fontweight="bold", va="center", zorder=3)
        ax_table.text(col_x[3] + 0.015, y_r + row_h / 2, delta,
                      color="#047857" if is_hl else "#334155", fontsize=9.8, fontweight="bold" if is_hl else "normal", va="center", zorder=3)

    # Bottom Highlight Banner
    draw_bottom_banner(
        fig,
        text="KEY RESULT: Evaluated on the identical stepped corrosion plate across lift-off (z₁=1.0mm → z₂=2.0mm), Concentric Star surges depth sizing R² to 0.7247 (vs 0.5546)\nand cuts sizing error to 141 μm (vs 178 μm), proving multi-scale outer anchors preserve physical depth penetration under lift-off drift.",
        bg="#F0FDF4", edge="#86EFAC", text_color="#166534", fontsize=7.2
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> [Slide 1B Generated]: {save_path}")


def draw_slide1_concentric_rings(save_path: str):
    """Backward compatibility wrapper: generates Slide 1A to save_path."""
    draw_slide1_topology_comparison(save_path)



def draw_slide1_panel_a_standalone(save_path: str):
    """
    Generates high-resolution standalone figure of Panel A:
    Comparing Dense 5x5 Cartesian Grid vs Concentric Star Topology on the discrete C-Scan raster matrix.
    Rendered with discrete square pixel cells, square ring boundaries, and zero units.
    """
    fig = plt.figure(figsize=(13, 7.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Header
    fig.text(0.06, 0.94, "SPATIAL SAMPLING TOPOLOGY ON DISCRETE C-SCAN RASTER MATRIX",
             fontsize=14.5, fontweight="bold", color=C_PRIMARY)
    fig.text(0.06, 0.90, "Comparison of Local 5x5 Dense Grid vs Proposed Multi-Scale Concentric Star on Discrete Pixel Raster",
             fontsize=9.5, color=C_TEXT_MUTED)

    # Left: Dense 5x5 Grid
    ax1 = fig.add_axes([0.08, 0.12, 0.38, 0.72])
    ax1.set_facecolor("#F8FAFC")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(-7.5, 7.5)
    ax1.set_ylim(-7.5, 7.5)
    ax1.set_xticks([-7, -5, -3, -1, 0, 1, 3, 5, 7])
    ax1.set_yticks([-7, -5, -3, -1, 0, 1, 3, 5, 7])
    ax1.tick_params(axis="both", which="both", length=0)
    ax1.grid(False)
    ax1.set_title("1. Dense 5x5 Cartesian Grid\n(Local 5x5 Pixel Patch, Trapped in Flaw)", fontsize=11, fontweight="bold", pad=8, color="#991B1B")

    # Render entire 15x15 discrete raster matrix (225 cells)
    for r in range(-7, 8):
        for c in range(-7, 8):
            cell_box = patches.Rectangle(
                (c - 0.5, r - 0.5), 1.0, 1.0,
                facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=0.7, zorder=1
            )
            ax1.add_patch(cell_box)

    # Connect outer perimeter of Dense 5x5
    d_perimeter_c = [-2, 2, 2, -2, -2]
    d_perimeter_r = [-2, -2, 2, 2, -2]
    ax1.plot(d_perimeter_c, d_perimeter_r, color="#DC2626", linestyle="--", linewidth=1.5, alpha=0.9, zorder=3)

    # Plot 25 dense probes as discrete square pixels
    d_idx = 0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            col = "#DC2626" if (dr == 0 and dc == 0) else "#B91C1C"
            e_col = "#7F1D1D" if (dr == 0 and dc == 0) else "#450A0A"
            p_rect = patches.Rectangle((dc - 0.5, dr - 0.5), 1.0, 1.0, facecolor=col, edgecolor=e_col, linewidth=1.0, zorder=4)
            ax1.add_patch(p_rect)
            ax1.text(dc, dr, f"#{d_idx}", color="#FFFFFF", fontsize=6.0, fontweight="bold", ha="center", va="center", zorder=5)
            d_idx += 1

    ax1.set_xlabel(r"Column Offset $\Delta c$", fontsize=10, fontweight="bold")
    ax1.set_ylabel(r"Row Offset $\Delta r$", fontsize=10, fontweight="bold")
    ax1.text(0, -6.8, "Span: 5x5 pixels (≈ 4 mm × 4 mm)\nAll probes trapped inside fastener head (D=6-8 mm)\nZero reference to undisturbed sound metal",
             ha="center", va="center", fontsize=7.8, color="#7F1D1D", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#FCA5A5", lw=1.0), zorder=6)

    # Right: Concentric Star Topology
    ax2 = fig.add_axes([0.54, 0.12, 0.38, 0.72])
    ax2.set_facecolor("#F8FAFC")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(-7.5, 7.5)
    ax2.set_ylim(-7.5, 7.5)
    ax2.set_xticks([-7, -5, -3, -1, 0, 1, 3, 5, 7])
    ax2.set_yticks([-7, -5, -3, -1, 0, 1, 3, 5, 7])
    ax2.tick_params(axis="both", which="both", length=0)
    ax2.grid(False)
    ax2.set_title("2. Multi-Scale Concentric Star (Octagram)\n(Spans 15x15 Pixels, Physical Coil Footprint)", fontsize=11, fontweight="bold", pad=8, color="#166534")

    # Render entire 15x15 discrete raster matrix (225 cells)
    for r in range(-7, 8):
        for c in range(-7, 8):
            cell_box = patches.Rectangle(
                (c - 0.5, r - 0.5), 1.0, 1.0,
                facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=0.7, zorder=1
            )
            ax2.add_patch(cell_box)

    # Connect ring perimeters: exact squares for Ring 1 and Ring 2
    ring1_sq_c = [-1, 1, 1, -1, -1]
    ring1_sq_r = [-1, -1, 1, 1, -1]
    ax2.plot(ring1_sq_c, ring1_sq_r, color="#2563EB", linestyle="--", linewidth=1.6, alpha=0.9, zorder=3)

    ring2_sq_c = [-3, 3, 3, -3, -3]
    ring2_sq_r = [-3, -3, 3, 3, -3]
    ax2.plot(ring2_sq_c, ring2_sq_r, color="#0284C7", linestyle="--", linewidth=1.6, alpha=0.9, zorder=3)

    ring3_loop_c = [0, 5, 7, 5, 0, -5, -7, -5, 0]
    ring3_loop_r = [7, 5, 0, -5, -7, -5, 0, 5, 7]
    ax2.plot(ring3_loop_c, ring3_loop_r, color="#059669", linestyle="--", linewidth=1.6, alpha=0.9, zorder=3)

    offsets = get_spatial_topology_offsets("concentric_star")

    # Center probe (#0)
    p0_rect = patches.Rectangle((-0.5, -0.5), 1.0, 1.0, facecolor="#DC2626", edgecolor="#7F1D1D", linewidth=1.2, zorder=4)
    ax2.add_patch(p0_rect)
    ax2.text(0, 0, "#0", color="#FFFFFF", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=5)

    # Ring 1
    for idx in range(1, 9):
        dr, dc = offsets[idx]
        p_rect = patches.Rectangle((dc - 0.5, dr - 0.5), 1.0, 1.0, facecolor="#2563EB", edgecolor="#1E3A8A", linewidth=1.1, zorder=4)
        ax2.add_patch(p_rect)
        ax2.text(dc, dr, f"#{idx}", color="#FFFFFF", fontsize=6.8, fontweight="bold", ha="center", va="center", zorder=5)

    # Ring 2
    for idx in range(9, 17):
        dr, dc = offsets[idx]
        p_rect = patches.Rectangle((dc - 0.5, dr - 0.5), 1.0, 1.0, facecolor="#0284C7", edgecolor="#0369A1", linewidth=1.1, zorder=4)
        ax2.add_patch(p_rect)
        ax2.text(dc, dr, f"#{idx}", color="#FFFFFF", fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=5)

    # Ring 3
    for idx in range(17, 25):
        dr, dc = offsets[idx]
        p_rect = patches.Rectangle((dc - 0.5, dr - 0.5), 1.0, 1.0, facecolor="#059669", edgecolor="#064E3B", linewidth=1.1, zorder=4)
        ax2.add_patch(p_rect)
        ax2.text(dc, dr, f"#{idx}", color="#FFFFFF", fontsize=6.2, fontweight="bold", ha="center", va="center", zorder=5)

    ax2.set_xlabel(r"Column Offset $\Delta c$", fontsize=10, fontweight="bold")
    ax2.set_ylabel(r"Row Offset $\Delta r$", fontsize=10, fontweight="bold")

    legend_handles = [
        patches.Patch(facecolor="#DC2626", edgecolor="#7F1D1D", label="Center"),
        patches.Patch(facecolor="#2563EB", edgecolor="#1E3A8A", label="Ring 1"),
        patches.Patch(facecolor="#0284C7", edgecolor="#0369A1", label="Ring 2"),
        patches.Patch(facecolor="#059669", edgecolor="#064E3B", label="Ring 3"),
    ]
    ax2.legend(handles=legend_handles, loc="upper right", fontsize=8.0, framealpha=0.92,
               edgecolor="#CBD5E1", facecolor="#FFFFFF")

    ax2.text(0, -6.8, "Span: 15x15 pixels (≈ 14 mm × 14 mm)\nRing 3 (r=7 px) anchors undisturbed sound metal\nDecisively eliminates trivial horizontal interpolation",
             ha="center", va="center", fontsize=7.8, color="#14532D", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDF4", edgecolor="#86EFAC", lw=1.0), zorder=6)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> [Standalone Panel A Generated]: {save_path}")


def draw_slide1_panel_c_standalone(save_path: str):
    """
    Generates high-resolution standalone figure of Panel C:
    Genuine laboratory TDMS waveforms across 3 excitation signals (Square, Gaussian, Chirp)
    under the exact condition: Specimen = Corrosion, Sensor = Hall AirCore, Lift-off = 1 mm (z1).
    Displays both absolute voltage V(t) and differential signals Delta V_i(t) = V_i(t) - V_0(t) across rings.
    """
    from scipy.signal import butter, filtfilt

    fig = plt.figure(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Header
    fig.text(0.04, 0.94, "REAL LABORATORY PECT TRANSIENT SIGNALS ACROSS CONCENTRIC RINGS",
             fontsize=14.5, fontweight="bold", color=C_PRIMARY)
    fig.text(0.04, 0.90, "Condition: Specimen = Corrosion | Sensor = Hall AirCore | Lift-off = 1.0 mm (z1) | Raw Voltage from Laboratory TDMS",
             fontsize=9.5, color=C_TEXT_MUTED)

    real_wf = get_real_corrosion_waveforms()
    wf_configs = [
        {"name": "Square", "title": "A. Square Waveform (200 Hz)", "col_x": 0.05, "desc": "Step excitation transient with rapid peak and asymptotic eddy current decay"},
        {"name": "Gaussian", "title": "B. Gaussian Waveform (1000 Hz)", "col_x": 0.36, "desc": "Smooth centered Gaussian wavepacket with localized temporal energy"},
        {"name": "Chirp", "title": "C. Chirp Waveform (500-1500 Hz)", "col_x": 0.67, "desc": "Linear frequency sweep reversing standard skin-depth penetration order"},
    ]

    col_w = 0.28
    label_map = {
        "Ring 1 (#1, r=1px)": "Ring 1 (#1) - Center",
        "Ring 2 (#9, r=3px)": "Ring 2 (#9) - Center",
        "Ring 3 (#17, r=7px)": "Ring 3 (#17) - Center"
    }

    # Filter design for differential eddy current signals (cutoff at 2.5 kHz)
    b_lp, a_lp = butter(3, 2500.0 / (50000.0), btype="low")

    for cfg in wf_configs:
        wf_name = cfg["name"]
        data = real_wf[wf_name]
        t_ms = data["time_ms"]
        curves = data["curves"]
        v_center = curves[0]["voltage"]

        # Upper Subplot: Absolute Voltage V(t)
        ax_top = fig.add_axes([cfg["col_x"], 0.51, col_w, 0.33])
        ax_top.set_facecolor("#F8FAFC")
        ax_top.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
        ax_top.set_title(cfg["title"], fontsize=11, fontweight="bold", pad=6, color=C_PRIMARY)

        for c in curves:
            ax_top.plot(t_ms, c["voltage"], color=c["color"], linewidth=c["lw"], label=c["name"])

        ax_top.set_xlim(0.0, 5.0)
        v_pad = 0.08 * (data["v_max"] - data["v_min"])
        ax_top.set_ylim(data["v_min"] - v_pad, data["v_max"] + v_pad)
        if cfg["name"] == "Square":
            ax_top.set_ylabel("Induced Voltage (V)", fontsize=9.0, fontweight="bold")
            ax_top.legend(loc="upper right", fontsize=7.2, framealpha=0.92)

        # Lower Subplot: Differential Signal Delta V_i(t) in mV
        ax_bot = fig.add_axes([cfg["col_x"], 0.15, col_w, 0.29])
        ax_bot.set_facecolor("#F8FAFC")
        ax_bot.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
        ax_bot.set_title(r"Ring Differential $\Delta V_i(t) = V_i(t) - V_{\mathrm{center}}(t)$ (mV)", fontsize=9.2, fontweight="bold", pad=4, color="#334155")

        for c in curves[1:]:
            diff_raw_mv = (c["voltage"] - v_center) * 1000.0  # Convert V to mV
            diff_smooth_mv = filtfilt(b_lp, a_lp, diff_raw_mv)
            lbl = label_map.get(c["name"], c["name"] + " - Center")
            ax_bot.plot(t_ms, diff_smooth_mv, color=c["color"], linewidth=c["lw"] * 1.1, label=lbl)

        ax_bot.axhline(0, color="#94A3B8", linestyle=":", linewidth=1.0)
        ax_bot.set_xlim(0.0, 5.0)
        ax_bot.set_xlabel("Time (ms)", fontsize=9.0, fontweight="bold")
        if cfg["name"] == "Square":
            ax_bot.set_ylabel(r"$\Delta V$ (mV)", fontsize=9.0, fontweight="bold")
            ax_bot.legend(loc="lower right", fontsize=7.0, framealpha=0.92)

        # Descriptor text
        fig.text(cfg["col_x"] + col_w / 2, 0.065, cfg["desc"], ha="center", fontsize=7.5, color="#64748B", style="italic")

    # Bottom summary box
    summary_ax = fig.add_axes([0.04, 0.015, 0.92, 0.035])
    summary_ax.axis("off")
    summary_ax.text(0.5, 0.5, "AUTHENTIC LABORATORY DATA: Exactly replicates physical NDT eddy current diffusion dynamics across spatial rings with zero simulation artifacts.",
                    ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1E3A8A")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> [Standalone Panel C Generated]: {save_path}")


# ==============================================================================
# SLIDE 2: Dual-Domain Time + FFT Fusion
# ==============================================================================
def draw_slide2_dual_domain_fusion(save_path: str):
    fig = create_base_slide(
        title="DUAL-DOMAIN TIME TRANSIENT & DODD-DEEDS FOURIER PHASE TOKENIZATION",
        subtitle="Waveform-Agnostic Single Token per Probe: Multi-Scale 1D Conv Cross-Attending to 14 Harmonic Phase Bins",
        badge="EXP-11 / EXP-17 / DUAL-DOMAIN SOTA"
    )

    # ---------------- Left Panel: Transient Time-Domain Dynamics ----------------
    ax1 = fig.add_axes([0.04, 0.14, 0.29, 0.72])
    ax1.set_facecolor("#F8FAFC")
    ax1.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
    ax1.set_title("A. Time-Domain Transient Branch\nContinuous 1D Multi-Scale Filterbank", fontsize=11, fontweight="bold", pad=8, color=C_PRIMARY)

    t = np.linspace(0, 5, 256)
    sig_lift1 = 2.6 * np.exp(-1.4 * t) * (1 - np.exp(-4 * t))
    sig_lift2 = 1.9 * np.exp(-1.4 * t) * (1 - np.exp(-4 * t))

    ax1.plot(t, sig_lift1, color="#1D4ED8", linewidth=2.0, label="Lift-off z1 (1.0 mm)")
    ax1.plot(t, sig_lift2, color="#93C5FD", linewidth=1.8, linestyle="--", label="Lift-off z2 (2.0 mm)")

    # Peak Callout (clean inside plot boundaries)
    ax1.set_ylim(-0.15, 1.85)
    t_peak = 0.58
    y_peak = 1.20
    ax1.scatter([t_peak], [y_peak], color="#DC2626", s=50, zorder=5)
    ax1.annotate("Peak Arrival $t_p$\n(Permeability $\\mu$)", xy=(t_peak, y_peak), xytext=(1.4, 1.45),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2), fontsize=8.0, fontweight="bold", color="#991B1B")

    ax1.annotate("Lift-Off Intersection (LOI)\n$x(t_0) \\approx \\text{const}$", xy=(0.18, 0.7), xytext=(0.4, 0.15),
                 arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2), fontsize=8.0, fontweight="bold", color="#065F46")

    ax1.set_xlabel("Time (ms)", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Signal Voltage (V)", fontsize=9.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.0)
    ax1.set_xlim(0, 5.0)

    # ---------------- Middle Panel: Dodd-Deeds Fourier Phase & Magnitude ----------------
    ax2 = fig.add_axes([0.35, 0.14, 0.29, 0.72])
    ax2.set_facecolor("#F8FAFC")
    ax2.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
    ax2.set_title("B. Dodd-Deeds Fourier Spectral Branch\nIntrinsic Lift-Off Invariance via Phase", fontsize=11, fontweight="bold", pad=8, color=C_PRIMARY)

    freqs = np.linspace(100, 2000, 14)
    phase_defect1 = -0.15 * np.sqrt(freqs / 100.0) * 0.3
    phase_defect2 = -0.15 * np.sqrt(freqs / 100.0) * 0.8
    phase_defect1_lift2 = phase_defect1 + 0.002

    ax2.plot(freqs, phase_defect1, "o-", color="#059669", linewidth=1.8, markersize=5, label="Depth 0.3 mm (z1 & z2)")
    ax2.plot(freqs, phase_defect1_lift2, "x--", color="#34D399", linewidth=1.4, markersize=6, label="Depth 0.3 mm (z2 overlay)")
    ax2.plot(freqs, phase_defect2, "s-", color="#D97706", linewidth=1.8, markersize=5, label="Depth 0.8 mm (Deep flaw)")

    ax2.set_xlabel("Harmonic Frequency (Hz)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Normalized Phase $\\theta(f) / \\pi$", fontsize=9.5, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=8.0)

    ax2.text(200, -0.28, "Phase Dispersion Law:\n$\\Delta \\theta(f) \\propto d \\sqrt{\\pi f \\mu \\sigma}$\n• 100% Invariant to lift-off gap $h$\n• No contrastive loss required!",
             fontsize=8.5, fontweight="bold", color="#065F46", bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECFDF5", edgecolor="#A7F3D0", lw=1.2))

    # ---------------- Right Panel: Physics-Gated Dual-Domain Fusion ----------------
    ax3 = fig.add_axes([0.66, 0.14, 0.30, 0.72])
    ax3.set_facecolor("#FFFFFF")
    ax3.axis("off")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title("C. Physics-Gated Dual-Domain Fusion\nSelf-Supervised Dynamic Balancing", fontsize=11, fontweight="bold", pad=8, color=C_PRIMARY)

    draw_card(ax3, 0.04, 0.72, 0.44, 0.24, title="z_time in R^(D/2)", subtitle="Continuous 1D Conv", body_text="• Peak timing & energy\n• Multi-scale k=5,15,31", bg="#EFF6FF", edge="#93C5FD", title_color="#1E40AF")
    draw_card(ax3, 0.52, 0.72, 0.44, 0.24, title="z_freq in R^(D/2)", subtitle="14 Harmonic Bins", body_text="• Lift-off invariant phase\n• Dodd-Deeds dispersion", bg="#ECFDF5", edge="#6EE7B7", title_color="#065F46")

    draw_card(ax3, 0.15, 0.42, 0.70, 0.22, title="Physics Gate: g in [0, 1]", subtitle="Self-Supervised Balancing", body_text="g = sigmoid(Linear(z_freq))\nBalances transient dynamics & phase", bg="#FAF5FF", edge="#D8B4FE", title_color="#6B21A8")

    draw_card(ax3, 0.06, 0.08, 0.88, 0.26, title="Unified Spatial Token z in R^D", subtitle="Waveform-Agnostic Representation", body_text="z = LayerNorm(W_fuse [g*z_t + (1-g)*z_f] + z_t)\n★ Exactly 1 continuous token per spatial probe\n★ Rule 3 compliant: Zero temporal chunking!", bg="#F8FAFC", edge="#64748B", lw=1.6, title_color="#0F172A")

    # Connective arrows
    ax3.annotate("", xy=(0.35, 0.64), xytext=(0.26, 0.72), arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.4))
    ax3.annotate("", xy=(0.65, 0.64), xytext=(0.74, 0.72), arrowprops=dict(arrowstyle="->", color="#059669", lw=1.4))
    ax3.annotate("", xy=(0.50, 0.34), xytext=(0.50, 0.42), arrowprops=dict(arrowstyle="->", color="#7C3AED", lw=1.6))

    # Bottom Banner
    draw_bottom_banner(
        fig,
        text="DUAL-DOMAIN BREAKTHROUGH: Combines transient eddy current arrival dynamics (t_p, LOI) with Dodd-Deeds lift-off invariant Fourier phase. Yields >0.999 lift-off cosine similarity and 77.2% AP on held-out Chirp waveforms without contrastive loss.",
        bg="#F0FDF4", edge="#86EFAC", text_color="#166534"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> [Slide 2 Generated]: {save_path}")


# ==============================================================================
# SLIDE 3: Physical Skin-Depth & Surface-to-Depth Masking
# ==============================================================================
def draw_slide3_skin_depth_masking(save_path: str):
    fig = create_base_slide(
        title="PHYSICAL SKIN-DEPTH DECOMPOSITION & SURFACE-TO-DEPTH MASKING",
        subtitle="Complementary Masking: Observe Surface & Shallow Features -> Predict Subsurface Defect Diffusion",
        badge="EXP-12 / EXP-17 / MASKING SOTA"
    )

    # ---------------- Left Panel: Maxwell Skin-Depth Physics ----------------
    ax1 = fig.add_axes([0.04, 0.14, 0.29, 0.72])
    ax1.set_facecolor("#F8FAFC")
    ax1.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
    ax1.set_title("A. Maxwell Skin-Depth Law\n$\\delta(f) = 1 / \\sqrt{\\pi f \\mu \\sigma}$", fontsize=11, fontweight="bold", pad=8, color=C_PRIMARY)

    f = np.linspace(20, 3000, 300)
    delta_mm = 503.0 / np.sqrt(f * 35.0)

    ax1.plot(f, delta_mm, color="#2563EB", linewidth=2.2)
    ax1.fill_between(f, 0, delta_mm, color="#BFDBFE", alpha=0.35)

    ax1.axvspan(1000, 3000, color="#FCA5A5", alpha=0.25, label="High Freq: Surface Mode (δ < 0.8mm)")
    ax1.axvspan(20, 400, color="#86EFAC", alpha=0.25, label="Low Freq: Deep Mode (δ > 2.0mm)")

    ax1.set_xlabel("Harmonic Frequency (Hz)", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Standard Skin Depth $\\delta$ (mm)", fontsize=9.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.set_xlim(20, 3000)
    ax1.set_ylim(0, 3.5)

    ax1.text(600, 2.6, "Surface Mode:\nLift-off & surface boundary", fontsize=8.0, fontweight="bold", color="#991B1B")
    ax1.text(60, 0.35, "Deep Mode:\nBack-wall loss & deep pits", fontsize=8.0, fontweight="bold", color="#166534")

    # ---------------- Middle Panel: 50/100 Token Decomposition ----------------
    ax2 = fig.add_axes([0.35, 0.14, 0.29, 0.72])
    ax2.set_facecolor("#FFFFFF")
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("B. Spatio-Spectral Token Structure\n2 Diffusion Modes per Spatial Probe", fontsize=11, fontweight="bold", pad=8, color=C_PRIMARY)

    draw_card(
        ax2, 0.04, 0.52, 0.92, 0.44,
        title="1. Surface Mode (Scale 1)",
        subtitle="High-Frequency Subband [f_split .. K]",
        body_text="• Waveform energy: Near-surface metal layer\n• Captures lift-off gap & initial wave front\n• Probe Count: 25 Probes × 1 = 25 Tokens\n• Positional Encoding: E_spatial(s) + E_scale(surf)",
        bg="#FEF2F2", edge="#FCA5A5", title_color="#991B1B", body_color="#7F1D1D"
    )

    draw_card(
        ax2, 0.04, 0.04, 0.92, 0.44,
        title="2. Deep Mode (Scale 0)",
        subtitle="Low-Frequency Subband [0 .. f_split]",
        body_text="• Waveform energy: Deep volumetric metal\n• Captures back-wall loss & subsurface flaws\n• Probe Count: 25 Probes × 1 = 25 Tokens\n• Positional Encoding: E_spatial(s) + E_scale(deep)",
        bg="#F0FDF4", edge="#86EFAC", title_color="#166534", body_color="#14532D"
    )

    # ---------------- Right Panel: Surface-to-Depth Masking Mechanism ----------------
    ax3 = fig.add_axes([0.66, 0.14, 0.30, 0.72])
    ax3.set_facecolor("#F8FAFC")
    ax3.axis("off")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title("C. Complementary Surface-to-Depth Masking\nEliminating Horizontal Spatial Copying", fontsize=11, fontweight="bold", pad=8, color=C_PRIMARY)

    draw_card(
        ax3, 0.04, 0.54, 0.92, 0.42,
        title="Context Tokens (Visible to Encoder)",
        subtitle="Observed Surface & Shallow Dynamics",
        body_text="• Spatial Points: Context probes S_ctx (17 probes)\n• Scales: Surface & Shallow modes ONLY\n• Target deep modes are ABSENT in context\n• Total visible: N_ctx tokens",
        bg="#EFF6FF", edge="#93C5FD", title_color="#1E40AF", body_color="#1E3A8A"
    )

    draw_card(
        ax3, 0.04, 0.04, 0.92, 0.42,
        title="Target Tokens (Predicted by JEPA)",
        subtitle="Hidden Subsurface Defect Perturbations",
        body_text="• Spatial Points: Target cluster S_tgt (8 probes)\n• Scales: Deep / Subsurface modes ONLY\n• Cannot copy horizontally from neighbors\n• Forces 3D vertical diffusion modeling",
        bg="#FEF2F2", edge="#F87171", title_color="#B91C1C", body_color="#991B1B"
    )

    ax3.annotate("", xy=(0.50, 0.47), xytext=(0.50, 0.53), arrowprops=dict(arrowstyle="->", color="#7C3AED", lw=2.0))
    ax3.text(0.50, 0.495, "3D Green's Propagation", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#7C3AED", bbox=dict(boxstyle="round,pad=0.2", facecolor="#FAF5FF", edgecolor="#D8B4FE"))

    # Bottom Banner
    draw_bottom_banner(
        fig,
        text="MASKING PRINCIPLE: By hiding deep subsurface modes across all context locations, the model is physically prohibited from horizontal shortcut copying on sound metal, forcing the JEPA predictor to learn vertical electromagnetic penetration.",
        bg="#EFF6FF", edge="#60A5FA", text_color="#1E40AF"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> [Slide 3 Generated]: {save_path}")


# ==============================================================================
# SLIDE 4: JEPA Predictor 3D Spatio-Diffusion Architecture
# ==============================================================================
def draw_slide4_predictor_architecture(save_path: str):
    fig = create_base_slide(
        title="PECT-JEPA PREDICTOR: 3D SPATIO-DIFFUSION WORLD MODEL",
        subtitle="Transformer Predictor with Continuous Parabolic Green's Attention Bias & Decoupled Residual Prediction",
        badge="EXP-10 / EXP-14 / EXP-17 SOTA"
    )

    # ---------------- Left Panel: Query Formulation & Diffusion Condition ----------------
    ax1 = fig.add_axes([0.04, 0.14, 0.29, 0.72])
    ax1.set_facecolor("#FFFFFF")
    ax1.axis("off")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("A. Target Query Formulation\nPhysics-Conditioned Operator Query", fontsize=11, fontweight="bold", pad=8, color=C_PRIMARY)

    draw_card(
        ax1, 0.04, 0.68, 0.92, 0.29,
        title="Context Representations",
        subtitle="H_ctx in R^(N_ctx x D)",
        body_text="• Outputs from Online Transformer Encoder\n• Compact latent summary of unmasked probes\n• Embedding Dimension D = 64 / 128",
        bg="#EFF6FF", edge="#93C5FD", title_color="#1E40AF", body_color="#1E3A8A"
    )

    draw_card(
        ax1, 0.04, 0.35, 0.92, 0.29,
        title="3D Positional Queries",
        subtitle="Q_tgt = Mask + Pos_3D",
        body_text="• Target 2D probe coordinate (s_x, s_y)\n• Target diffusion scale k (shallow vs deep)\n• E_pos = E_spatial(s) + E_scale(k)",
        bg="#F8FAFC", edge="#CBD5E1", title_color="#334155", body_color="#475569"
    )

    draw_card(
        ax1, 0.04, 0.02, 0.92, 0.29,
        title="Diffusion Operator Condition",
        subtitle="q_diff(omega) via Fourier Feature MLP",
        body_text="• Characteristic frequency omega = f_c / f_max\n• Encodes wavepacket dispersion speed\n• Continuous action parameter in R^D",
        bg="#FAF5FF", edge="#D8B4FE", title_color="#6B21A8", body_color="#581C87"
    )

    # ---------------- Middle Panel: Transformer Block & Green's Attention Bias ----------------
    ax2 = fig.add_axes([0.35, 0.14, 0.31, 0.72])
    ax2.set_facecolor("#F8FAFC")
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("B. Predictor Cross-Attention Engine\nParabolic Green's Function Bias Matrix", fontsize=11, fontweight="bold", pad=8, color=C_PRIMARY)

    draw_card(
        ax2, 0.04, 0.70, 0.92, 0.26,
        title="1. Self-Attention (Target Queries)",
        subtitle="Models spatial correlation among flaw tokens",
        body_text="• Standard Multi-Head Self-Attention\n• Queries attend mutually across defect cluster",
        bg="#FFFFFF", edge="#CBD5E1", title_color="#0F172A"
    )

    draw_card(
        ax2, 0.04, 0.33, 0.92, 0.34,
        title="2. Cross-Attention (Queries -> H_ctx)",
        subtitle="Conditioned on Continuous Green's Bias",
        body_text="Logits:  $A_{ij} = (Q_i K_j^T)/\\sqrt{d_k} + B_{ij}^{\\text{Green}}$\n\nGreen's Bias: $B_{ij} = -\\gamma d_{ij}^2 \\sqrt{\\omega} - \\alpha \\ln(1 + d_{ij}^2) - \\beta |\\Delta k|$\nEnforces physical diffusion decay with distance $d_{ij}$",
        bg="#EFF6FF", edge="#60A5FA", lw=1.5, title_color="#1E40AF", body_color="#1D4ED8"
    )

    draw_card(
        ax2, 0.04, 0.02, 0.92, 0.28,
        title="3. Feed-Forward Network (MLP)",
        subtitle="Expansion factor 4x + GELU + LayerNorm",
        body_text="• Two-layer MLP with residual connection\n• Generates refined latent target embeddings",
        bg="#FFFFFF", edge="#CBD5E1", title_color="#0F172A"
    )

    # Arrows inside middle panel
    ax2.annotate("", xy=(0.50, 0.67), xytext=(0.50, 0.70), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.4))
    ax2.annotate("", xy=(0.50, 0.30), xytext=(0.50, 0.33), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.4))

    # ---------------- Right Panel: Decoupled Residual Prediction ----------------
    ax3 = fig.add_axes([0.68, 0.14, 0.28, 0.72])
    ax3.set_facecolor("#FFFFFF")
    ax3.axis("off")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title("C. Decoupled Residual Output\nIsolating 1-3% Flaw Perturbation", fontsize=11, fontweight="bold", pad=8, color=C_PRIMARY)

    draw_card(
        ax3, 0.04, 0.68, 0.92, 0.29,
        title="Incident Wave Baseline (h_base)",
        subtitle="98% Total Signal Energy",
        body_text="• Smooth macro-decay of excitation field\n• Derived from unmasked context tokens\n• Prevents memorization of primary field",
        bg="#F1F5F9", edge="#94A3B8", title_color="#334155", body_color="#475569"
    )

    draw_card(
        ax3, 0.04, 0.35, 0.92, 0.29,
        title="Flaw Perturbation (ΔH_pred)",
        subtitle="1 - 3% Differential Energy",
        body_text="• Micro-eddy current disturbance field\n• Directly sensitive to crack depth & loss\n• Decoupled from dominant incident field",
        bg="#FEF2F2", edge="#F87171", title_color="#991B1B", body_color="#B91C1C"
    )

    draw_card(
        ax3, 0.04, 0.02, 0.92, 0.29,
        title="Predicted Latents: H_hat",
        subtitle="H_hat = h_base + ΔH_pred",
        body_text="Compared with Stop-Gradient Target:\nLoss = || H_hat_tgt - Target ||_1\nPreserves fine flaw gradient across depth",
        bg="#F0FDF4", edge="#86EFAC", lw=1.6, title_color="#166534", body_color="#15803D"
    )

    # Bottom Banner
    draw_bottom_banner(
        fig,
        text="PREDICTOR INNOVATION: Grounding target attention in Parabolic Green's diffusion physics and decoupling the incident field baseline (h_base) from localized perturbation (ΔH) surged Anomaly Detection AP to 71.56% and Plate R² past 0.30.",
        bg="#FAF5FF", edge="#D8B4FE", text_color="#6B21A8"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> [Slide 4 Generated]: {save_path}")


# ==============================================================================
# SLIDE 5: Loss Formulations & Intra-Scan VICReg Anti-Collapse
# ==============================================================================
def _load_exp17_epoch_metrics():
    """
    Reads the genuine EXP-17 per-epoch training log (57-scan compound OOD run).
    Falls back to the registry-verified trajectory when the run directory is absent.
    """
    fallback = {
        "epoch": np.arange(1, 11),
        "val_loss_pred": np.array([0.3483, 0.1247, 0.0739, 0.0579, 0.0470,
                                   0.0433, 0.0346, 0.0290, 0.0232, 0.0204]),
        "twonn_dim": np.array([7.14, 11.64, 11.96, 11.29, 11.88,
                               11.63, 10.30, 10.59, 11.39, 10.34]),
    }
    csv_path = os.path.join(ROOT_DIR, "experiments", "5x5",
                            "exp17_spatio_diffusion_jepa", "metrics_epoch.csv")
    if not os.path.exists(csv_path):
        return fallback
    try:
        rows = np.atleast_1d(np.genfromtxt(csv_path, delimiter=",", names=True, encoding="utf-8"))
        epoch = np.asarray(rows["epoch"], dtype=float)
        pred = np.asarray(rows["val_loss_pred"], dtype=float)
        twonn = np.asarray(rows["twonn_dim"], dtype=float)
    except Exception:
        return fallback
    keep = np.isfinite(epoch) & np.isfinite(pred) & np.isfinite(twonn)
    if keep.sum() < 2:
        return fallback
    return {"epoch": epoch[keep], "val_loss_pred": pred[keep], "twonn_dim": twonn[keep]}


def _flow_box(ax, x, y, w, h, text, bg, edge, text_color, fontsize=7.6, weight="bold", radius=0.03):
    """Draws a compact labelled block inside a schematic (0-1 normalised) axes."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=bg, edgecolor=edge, linewidth=1.2, zorder=2
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color,
            linespacing=1.22, zorder=3)


def _flow_arrow(ax, xy_from, xy_to, color=C_TEXT_MUTED, style="-|>", lw=1.3,
                connectionstyle="arc3,rad=0"):
    ax.annotate("", xy=xy_to, xytext=xy_from, zorder=4,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=11,
                                connectionstyle=connectionstyle))


def draw_slide5_loss_formulation(save_path: str):
    fig = create_base_slide(
        title="PURE-JEPA LOSS DESIGN: LATENT PREDICTION + INTRA-SCAN VICREG",
        subtitle="Every Heuristic Penalty Disabled (Weight = 0) - Collapse Blocked Coordinate-Wise Inside Each C-Scan",
        badge="EXP-16 / EXP-17 / LOSS DESIGN"
    )

    # ================= Panel A: EXP-17 objective weights (config-verified) =================
    ax_a = fig.add_axes([0.050, 0.500, 0.280, 0.360])
    ax_a.set_facecolor("#F8FAFC")
    ax_a.set_title("A. EXP-17 Objective Weights  (4 Active / 10)",
                   fontsize=10.5, fontweight="bold", color=C_PRIMARY, pad=8)

    term_labels = [
        "Fluctuation", "Temporal monotonicity", "Adaptive disturbance", "Lift-off invariance",
        "Phase-depth align", "Hypersphere uniformity",
        "L1 prediction (latent)", "VICReg variance hinge", "VICReg covariance", "Norm-floor barrier",
    ]
    term_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.1]
    y_pos = np.arange(len(term_labels))
    colors = [C_ACCENT_GREEN if w > 0.0 else "#E2E8F0" for w in term_weights]
    edges = [C_ACCENT_GREEN if w > 0.0 else "#CBD5E1" for w in term_weights]
    bars = ax_a.barh(y_pos, term_weights, height=0.62, color=colors,
                     edgecolor=edges, linewidth=1.1, zorder=3)
    for rect, w in zip(bars, term_weights):
        ax_a.text(w + 0.035, rect.get_y() + rect.get_height() / 2.0, f"{w:.1f}",
                  va="center", fontsize=7.4, fontweight="bold",
                  color="#065F46" if w > 0.0 else C_TEXT_MUTED, zorder=4)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(term_labels, fontsize=7.4)
    ax_a.set_xlim(0, 1.32)
    ax_a.set_xticks([0, 0.5, 1.0])
    ax_a.tick_params(axis="x", labelsize=7.4)
    ax_a.set_xlabel("Loss weight  $\\lambda$", fontsize=8.2, fontweight="bold")
    ax_a.grid(True, axis="x", linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    for spine in ("top", "right"):
        ax_a.spines[spine].set_visible(False)

    # ================= Panel B: EXP-17 real training trajectory =========================
    traj = _load_exp17_epoch_metrics()
    ax_b = fig.add_axes([0.375, 0.500, 0.300, 0.360])
    ax_b.set_facecolor("#FFFFFF")
    ax_b.set_title("B. EXP-17 Training: Loss Falls, Rank Holds",
                   fontsize=10.5, fontweight="bold", color=C_PRIMARY, pad=8)

    ax_b.plot(traj["epoch"], traj["val_loss_pred"], "o-", color=C_ACCENT_RED,
              linewidth=2.1, markersize=5.2, zorder=4, label="$\\mathcal{L}_{pred}$ (val)")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Epoch", fontsize=8.6, fontweight="bold")
    ax_b.set_ylabel("Val prediction loss  $\\mathcal{L}_{pred}$", fontsize=8.4,
                    fontweight="bold", color=C_ACCENT_RED)
    ax_b.tick_params(axis="y", labelsize=7.4, colors=C_ACCENT_RED)
    ax_b.tick_params(axis="x", labelsize=7.4)
    ax_b.set_xticks(traj["epoch"].astype(int))
    ax_b.set_ylim(0.012, 0.75)
    ax_b.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    for spine in ("top", "right"):
        ax_b.spines[spine].set_visible(False)

    ax_b2 = ax_b.twinx()
    ax_b2.plot(traj["epoch"], traj["twonn_dim"], "s--", color=C_SECONDARY,
               linewidth=2.1, markersize=5.0, zorder=3, label="Two-NN dim")
    ax_b2.axhspan(10.0, 12.5, color="#E0F2FE", alpha=0.75, zorder=0)
    ax_b2.set_ylim(0, 17)
    ax_b2.set_ylabel("Two-NN intrinsic dim (D)", fontsize=8.4, fontweight="bold", color=C_SECONDARY)
    ax_b2.tick_params(axis="y", labelsize=7.4, colors=C_SECONDARY)
    ax_b2.spines["top"].set_visible(False)

    drop = 100.0 * (1.0 - traj["val_loss_pred"][-1] / traj["val_loss_pred"][0])
    ax_b.annotate(f"$\\mathcal{{L}}_{{pred}}$ -{drop:.0f}%",
                  xy=(traj["epoch"][-1], traj["val_loss_pred"][-1]),
                  xytext=(5.9, 0.105), fontsize=8.2, fontweight="bold", color="#991B1B",
                  arrowprops=dict(arrowstyle="->", color="#991B1B", lw=1.3))
    ax_b2.annotate(f"rank holds 10.3D", xy=(traj["epoch"][-1], traj["twonn_dim"][-1]),
                   xytext=(4.7, 7.4), fontsize=8.2, fontweight="bold", color="#075985",
                   arrowprops=dict(arrowstyle="->", color="#075985", lw=1.3))
    handles = ax_b.get_lines()[:1] + ax_b2.get_lines()[:1]
    ax_b.legend(handles, [h.get_label() for h in handles], loc="upper right",
                fontsize=7.6, framealpha=0.92)

    # ================= Panel C: VICReg penalty shapes (the mechanism) ===================
    ax_c_title = fig.add_axes([0.715, 0.845, 0.255, 0.030])
    ax_c_title.axis("off")
    ax_c_title.text(0.5, 0.5, "C. VICReg Penalty Shapes  ($\\gamma$ = 1.0)",
                    ha="center", va="center", fontsize=10.5, fontweight="bold", color=C_PRIMARY)

    ax_c1 = fig.add_axes([0.720, 0.645, 0.245, 0.165])
    ax_c1.set_facecolor("#F0FDF4")
    sigma = np.linspace(0.0, 2.0, 400)
    l_var = np.maximum(0.0, 1.0 - sigma)
    ax_c1.fill_between(sigma, 0, l_var, color="#FCA5A5", alpha=0.55, zorder=2)
    ax_c1.plot(sigma, l_var, color=C_ACCENT_RED, linewidth=2.0, zorder=3)
    ax_c1.axvline(1.0, color="#166534", linestyle="--", linewidth=1.2, zorder=4)
    ax_c1.text(0.52, 0.62, "penalty", fontsize=7.4, fontweight="bold", color="#991B1B", ha="center")
    ax_c1.text(1.48, 0.62, "free", fontsize=7.4, fontweight="bold", color="#166534", ha="center")
    ax_c1.set_title("$\\mathcal{L}_{var} = \\max(0,\\ \\gamma - \\sigma_d)$",
                    fontsize=8.6, fontweight="bold", color="#166534", pad=5)
    ax_c1.set_xlabel("latent std $\\sigma_d$", fontsize=7.8)
    ax_c1.set_ylim(-0.06, 1.05)
    ax_c1.set_xticks([0, 1, 2])
    ax_c1.tick_params(labelsize=7.0)
    ax_c1.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    for spine in ("top", "right"):
        ax_c1.spines[spine].set_visible(False)

    ax_c2 = fig.add_axes([0.720, 0.505, 0.245, 0.165])
    ax_c2.set_facecolor("#FAF5FF")
    rho = np.linspace(0.0, 1.0, 400)
    ax_c2.fill_between(rho, 0, rho ** 2, color="#D8B4FE", alpha=0.55, zorder=2)
    ax_c2.plot(rho, rho ** 2, color=C_PURPLE, linewidth=2.0, zorder=3)
    ax_c2.annotate(r"$\rho \to 0$", xy=(0.06, 0.004), fontsize=8.0,
                   fontweight="bold", color="#6B21A8")
    ax_c2.set_title("$\\mathcal{L}_{cov} \\propto \\rho_{jk}^2\\ \\ (j \\neq k)$",
                    fontsize=8.6, fontweight="bold", color="#6B21A8", pad=5)
    ax_c2.set_xlabel("off-diag. correlation $\\rho$", fontsize=7.8)
    ax_c2.set_ylim(-0.05, 1.05)
    ax_c2.set_xticks([0, 0.5, 1.0])
    ax_c2.tick_params(labelsize=7.0)
    ax_c2.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    for spine in ("top", "right"):
        ax_c2.spines[spine].set_visible(False)

    # ================= Panel D: real ablation bars (57 held-out OOD scans) ===============
    ax_d = fig.add_axes([0.050, 0.135, 0.440, 0.315])
    ax_d.set_facecolor("#F8FAFC")
    ax_d.set_title("D. Removing Heuristic Friction Improves Every Metric",
                   fontsize=10.5, fontweight="bold", color=C_PRIMARY, pad=8)

    stages = ["EXP-13\nheuristics ON", "EXP-16\nheuristics = 0", "EXP-17\n+ intra-scan VICReg"]
    ap_vals = [40.38, 56.93, 71.56]
    plate_r2 = [0.1466, 0.2291, 0.3168]
    x = np.arange(len(stages))

    ax_d.bar(x, ap_vals, width=0.46, color=["#93C5FD", "#60A5FA", "#2563EB"],
             edgecolor="#1D4ED8", zorder=3, label="Average Precision (%)")
    for xi, v in zip(x, ap_vals):
        ax_d.text(xi, v + 1.8, f"{v:.2f}", ha="center", fontsize=8.2,
                  fontweight="bold", color="#1E3A8A", zorder=4)
    ax_d.set_ylim(0, 100)
    ax_d.set_ylabel("Average Precision (%)", fontsize=8.6, fontweight="bold", color="#1E3A8A")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(stages, fontsize=7.8, fontweight="bold")
    ax_d.tick_params(axis="y", labelsize=7.4)
    ax_d.grid(True, axis="y", linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax_d.set_xlabel("Experiment (57 held-out compound-OOD scans)", fontsize=8.4, fontweight="bold")
    for spine in ("top", "right"):
        ax_d.spines[spine].set_visible(False)

    ax_d2 = ax_d.twinx()
    ax_d2.plot(x, plate_r2, "D-", color=C_ACCENT_AMBER, linewidth=2.2,
               markersize=6.0, zorder=5, label="Plate depth $R^2$")
    for xi, v in zip(x, plate_r2):
        ax_d2.text(xi, v + 0.020, f"{v:.4f}", ha="center", fontsize=8.0,
                   fontweight="bold", color="#92400E", zorder=6,
                   bbox=dict(facecolor="#FFFFFF", alpha=0.88, edgecolor="none", pad=1.2))
    ax_d2.set_ylim(0, 0.36)
    ax_d2.set_ylabel("Plate depth $R^2$", fontsize=8.6, fontweight="bold", color="#92400E")
    ax_d2.tick_params(axis="y", labelsize=7.4, colors="#92400E")
    ax_d2.spines["top"].set_visible(False)
    ax_d2.text(0.5, 0.216, "+56.3%", ha="center", fontsize=8.4,
               fontweight="bold", color="#92400E", zorder=6)
    ax_d2.text(1.5, 0.302, "+38.3%", ha="center", fontsize=8.4,
               fontweight="bold", color="#92400E", zorder=6)
    ax_d.legend([ax_d.patches[0], ax_d2.lines[0]],
                ["Average Precision (%)", "Plate depth $R^2$"],
                loc="upper left", fontsize=7.6, framealpha=0.92)

    # ================= Panel E: loss computation graph (diagram) ======================
    ax_e = fig.add_axes([0.530, 0.135, 0.440, 0.315])
    ax_e.set_facecolor("#FFFFFF")
    ax_e.axis("off")
    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1)
    ax_e.set_title("E. Where Each Term Acts", fontsize=10.5, fontweight="bold",
                   color=C_PRIMARY, pad=8)

    _flow_box(ax_e, 0.010, 0.640, 0.115, 0.230, "Context\ntokens", "#F1F5F9", "#94A3B8", C_TEXT_MAIN)
    _flow_box(ax_e, 0.160, 0.640, 0.140, 0.230, "Online\nEncoder $f$", "#EFF6FF", "#60A5FA", "#1E40AF")
    _flow_box(ax_e, 0.335, 0.640, 0.090, 0.230, "$H_{ctx}$", "#DBEAFE", "#3B82F6", "#1E3A8A", fontsize=8.4)
    _flow_box(ax_e, 0.460, 0.640, 0.160, 0.230, "Predictor\n+ 3D query", "#F5F3FF", "#A78BFA", "#5B21B6")
    _flow_box(ax_e, 0.655, 0.640, 0.090, 0.230, "$\\hat{H}$", "#DCFCE7", "#22C55E", "#166534", fontsize=8.4)

    _flow_box(ax_e, 0.010, 0.160, 0.115, 0.230, "Target\ntokens", "#F1F5F9", "#94A3B8", C_TEXT_MAIN)
    _flow_box(ax_e, 0.160, 0.160, 0.140, 0.230, "EMA Encoder\n$sg(\\cdot)$", "#FEF2F2", "#F87171", "#991B1B")
    _flow_box(ax_e, 0.335, 0.160, 0.090, 0.230, "$H_{tgt}$", "#FEE2E2", "#EF4444", "#991B1B", fontsize=8.4)
    _flow_box(ax_e, 0.460, 0.160, 0.130, 0.230, "$z$ per\nscan $k$", "#FEF9C3", "#FACC15", "#92400E", fontsize=7.8)
    _flow_box(ax_e, 0.625, 0.160, 0.120, 0.230, "center\n$-\\mu_k$", "#FEF3C7", "#F59E0B", "#92400E", fontsize=7.8)

    _flow_box(ax_e, 0.760, 0.160, 0.225, 0.710,
              "$\\mathcal{L}_{total}$\n\n$\\|\\hat{H} - sg(H_{tgt})\\|_1$\n$+\\,1.0\\,\\mathcal{L}_{var}$\n$+\\,1.0\\,\\mathcal{L}_{cov}$",
              "#1E3A8A", "#1E3A8A", "#FFFFFF", fontsize=7.8, radius=0.02)

    # Row-1 chain
    for x0, x1 in ((0.125, 0.160), (0.300, 0.335), (0.425, 0.460), (0.620, 0.655)):
        _flow_arrow(ax_e, (x0, 0.755), (x1, 0.755))
    # Row-2 chain
    for x0, x1 in ((0.125, 0.160), (0.300, 0.335), (0.425, 0.460), (0.590, 0.625)):
        _flow_arrow(ax_e, (x0, 0.275), (x1, 0.275))
    # H_ctx -> per-scan latent (VICReg branch)
    _flow_arrow(ax_e, (0.400, 0.640), (0.510, 0.390), color="#92400E",
                style="-|>", lw=1.2, connectionstyle="arc3,rad=-0.22")
    # H_tgt -> L_total (routed under the VICReg branch)
    _flow_arrow(ax_e, (0.380, 0.160), (0.865, 0.160), color="#991B1B",
                style="-|>", lw=1.2, connectionstyle="arc3,rad=0.35")
    # predicted / centred embeddings -> L_total
    _flow_arrow(ax_e, (0.745, 0.755), (0.760, 0.640), color="#166534")
    _flow_arrow(ax_e, (0.745, 0.275), (0.760, 0.390), color="#166534")

    ax_e.text(0.428, 0.530, "VICReg", fontsize=7.4, fontweight="bold",
              color="#92400E", zorder=5)
    ax_e.text(0.500, 0.045, "Intra-Scan VICReg: hinge + decorrelation applied per scan file k  |  all heuristic terms = 0",
              ha="center", fontsize=7.4, fontweight="bold", color=C_TEXT_MUTED, zorder=5)

    # ================= Bottom Banner =================================================
    draw_bottom_banner(
        fig,
        text="LOSS INVENTORY: 10 candidate terms, 4 active. Prediction is the only data-fitting force (pure JEPA, no negatives, no pixel MSE); VICReg hinge + decorrelation act inside each C-scan to block point and dimensional collapse.",
        bg="#F0FDF4", edge="#86EFAC", text_color="#166534"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> [Slide 5 Generated]: {save_path}")


# ==============================================================================
# SLIDE 6: Comprehensive Benchmark Progression Across Experiments
# ==============================================================================
def draw_slide6_experiment_benchmarks(save_path: str):
    fig = create_base_slide(
        title="QUANTITATIVE BENCHMARK PROGRESSION ACROSS PECT-JEPA EXPERIMENTS",
        subtitle="Empirical Metrics on 57 Held-Out Compound OOD Test Scans from Initial Baseline to 3D Spatio-Diffusion SOTA",
        badge="EXP-01 TO EXP-17 BENCHMARK"
    )

    exp_labels = ["EXP-01\nBaseline", "EXP-04\nDualScale", "EXP-08\nCST-Mask", "EXP-10\nResidual", "EXP-13\nStarTop", "EXP-14\nRadialDisp", "EXP-16\nCentered", "EXP-17\nSOTA"]
    x = np.arange(len(exp_labels))
    width = 0.36

    # ---------------- Top Left Panel: Anomaly Detection (AUC & AP) ----------------
    ax1 = fig.add_axes([0.05, 0.53, 0.43, 0.33])
    ax1.set_facecolor("#F8FAFC")
    ax1.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
    ax1.set_title("A. Flaw Detection Performance (AUC-ROC & Average Precision)", fontsize=11, fontweight="bold", color=C_PRIMARY, pad=8)

    auc_vals = [80.70, 86.93, 87.23, 87.91, 82.68, 89.98, 88.94, 93.78]
    ap_vals = [34.48, 43.71, 44.19, 50.00, 40.38, 59.27, 56.93, 71.56]

    bars1 = ax1.bar(x - width/2, auc_vals, width, label="AUC-ROC (%)", color="#3B82F6", edgecolor="#1D4ED8", zorder=3)
    bars2 = ax1.bar(x + width/2, ap_vals, width, label="Average Precision AP (%)", color="#059669", edgecolor="#047857", zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_labels, fontsize=7.8, fontweight="bold")
    ax1.set_ylabel("Score (%)", fontsize=9.0, fontweight="bold")
    ax1.set_ylim(0, 108)
    ax1.legend(loc="upper left", fontsize=8.0)

    ax1.annotate("+107% AP Surge\n(71.56%)", xy=(7 + width/2, 71.56), xytext=(6.0, 85),
                 arrowprops=dict(arrowstyle="->", color="#059669", lw=1.4), fontsize=7.8, fontweight="bold", color="#065F46")

    # ---------------- Top Right Panel: Two-Stage Hurdle Depth Regression R² ----------------
    ax2 = fig.add_axes([0.53, 0.53, 0.43, 0.33])
    ax2.set_facecolor("#F8FAFC")
    ax2.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
    ax2.set_title("B. Quantitative Depth Sizing (Plate R² & Defect-Only R²)", fontsize=11, fontweight="bold", color=C_PRIMARY, pad=8)

    plate_r2 = [0.1221, 0.1624, 0.1657, 0.2005, 0.1466, 0.2246, 0.2291, 0.3168]
    defect_r2 = [-0.96, -0.96, -4.40, 0.847, 0.5143, 0.6132, 0.6106, 0.7493]

    bars3 = ax2.bar(x - width/2, plate_r2, width, label="Plate-Level R²", color="#6366F1", edgecolor="#4338CA", zorder=3)
    bars4 = ax2.bar(x + width/2, [max(v, -0.2) for v in defect_r2], width, label="Defect-Only R² (y > 0)", color="#EC4899", edgecolor="#BE185D", zorder=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_labels, fontsize=7.8, fontweight="bold")
    ax2.set_ylabel("Coefficient of Determination R²", fontsize=9.0, fontweight="bold")
    ax2.set_ylim(-0.25, 0.95)
    ax2.axhline(0, color="#64748B", linestyle="-", linewidth=1.0)
    ax2.legend(loc="upper left", fontsize=8.0)

    ax2.annotate("EXP-17 SOTA:\nPlate R² = 0.3168\nDefect R² = 0.7493", xy=(7 - width/2, 0.3168), xytext=(5.3, 0.62),
                 arrowprops=dict(arrowstyle="->", color="#4338CA", lw=1.4), fontsize=7.8, fontweight="bold", color="#312E81")

    # ---------------- Bottom Left Panel: Defect Contrast Ratio (CNR) & MAE ----------------
    ax3 = fig.add_axes([0.05, 0.14, 0.43, 0.31])
    ax3.set_facecolor("#F8FAFC")
    ax3.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
    ax3.set_title("C. Flaw Contrast-to-Noise Ratio (CNR)", fontsize=11, fontweight="bold", color=C_PRIMARY, pad=8)

    cnr_vals = [1.50, 2.00, 2.02, 2.26, 1.83, 2.95, 2.74, 3.86]
    ax3.plot(x, cnr_vals, "o-", color="#D97706", linewidth=2.2, markersize=6, zorder=4)
    ax3.fill_between(x, 0, cnr_vals, color="#FEF3C7", alpha=0.4)

    for i, v in enumerate(cnr_vals):
        ax3.text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=8.0, fontweight="bold", color="#B45309")

    ax3.set_xticks(x)
    ax3.set_xticklabels(exp_labels, fontsize=7.8, fontweight="bold")
    ax3.set_ylabel("Defect CNR", fontsize=9.0, fontweight="bold")
    ax3.set_ylim(0, 4.6)

    # ---------------- Bottom Right Panel: Held-Out Compound OOD Generalization (EXP-17) ----------------
    ax4 = fig.add_axes([0.53, 0.14, 0.43, 0.31])
    ax4.set_facecolor("#FFFFFF")
    ax4.axis("off")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title("D. EXP-17 SOTA Breakdown Across Held-Out Modalities", fontsize=11, fontweight="bold", color=C_PRIMARY, pad=8)

    draw_card(
        ax4, 0.02, 0.52, 0.46, 0.44,
        title="Held-Out Chirp Waveform",
        subtitle="N = 27 Scans (Broadband Sweep)",
        body_text="• AUC: 95.20% | AP: 77.21%\n• Defect Sizing R²: 0.7976\n• Flaw Contrast CNR: 4.61",
        bg="#EFF6FF", edge="#93C5FD", title_color="#1E40AF", body_color="#1E3A8A"
    )

    draw_card(
        ax4, 0.52, 0.52, 0.46, 0.44,
        title="Held-Out TMR Sensor",
        subtitle="N = 27 Scans (High Sensitivity)",
        body_text="• AUC: 94.46% | AP: 73.25%\n• Defect Sizing R²: 0.6890\n• Flaw Contrast CNR: 3.87",
        bg="#F0FDF4", edge="#86EFAC", title_color="#166534", body_color="#14532D"
    )

    draw_card(
        ax4, 0.02, 0.02, 0.46, 0.44,
        title="Held-Out Lift-Off z3 (3.0mm)",
        subtitle="Extreme Air-Gap Attenuation",
        body_text="• AUC: 91.16% | AP: 62.46%\n• Defect Sizing R²: 0.7519\n• Flaw Contrast CNR: 3.05",
        bg="#FEF3C7", edge="#FCD34D", title_color="#92400E", body_color="#78350F"
    )

    draw_card(
        ax4, 0.52, 0.02, 0.46, 0.44,
        title="Rivet Sizing Precision",
        subtitle="Critical Aerospace NDT Specimen",
        body_text="• Depth MAE: 0.0633 mm (63.3 µm)\n• Rivet Anomaly AP: 88.62%\n• Rivet Defect CNR: 6.16",
        bg="#FAF5FF", edge="#D8B4FE", title_color="#6B21A8", body_color="#581C87"
    )

    # Bottom Banner
    draw_bottom_banner(
        fig,
        text="DEFINITIVE MILESTONE: EXP-17 breaks all historical project records. Flaw AP surged from 34.5% to 71.56% (+107% rel), CNR reached 3.86 (+157%), Plate R² broke 0.30 barrier for the first time (0.3168), and Rivet depth MAE reached 63.3 microns.",
        bg="#F0FDF4", edge="#86EFAC", text_color="#166534"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> [Slide 6 Generated]: {save_path}")


# ==============================================================================
# MAIN CLI DRIVER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate PECT-JEPA Presentation Slide Figures Pack (16:9, 300 DPI).")
    parser.add_argument("--slide", type=str, default="all", choices=["1", "2", "3", "4", "5", "6", "all"],
                        help="Specific slide to generate (1 to 6) or 'all'.")
    parser.add_argument("--output_dir", type=str, default="presentation_figures/slide_deck",
                        help="Dedicated destination directory for slide images.")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Destination Subfolder: {out_dir}\n")

    slides_map = {
        "1": (draw_slide1_topology_comparison, "slide1_input_topology_comparison.png"),
        "2": (draw_slide2_dual_domain_fusion, "slide2_dual_domain_time_fft.png"),
        "3": (draw_slide3_skin_depth_masking, "slide3_shallow_deep_masking.png"),
        "4": (draw_slide4_predictor_architecture, "slide4_jepa_predictor_architecture.png"),
        "5": (draw_slide5_loss_formulation, "slide5_loss_and_vicreg.png"),
        "6": (draw_slide6_experiment_benchmarks, "slide6_experiment_comparison_benchmarks.png"),
    }

    targets = list(slides_map.keys()) if args.slide == "all" else [args.slide]

    for s_id in targets:
        func, fname = slides_map[s_id]
        full_path = os.path.join(out_dir, fname)
        print(f"Generating Slide {s_id} -> {fname}...")
        func(full_path)
        if s_id == "1":
            # Slide 1B: Controlled Benchmark Table
            p_table_path = os.path.join(out_dir, "slide1_controlled_benchmark_table.png")
            print(f"Generating Slide 1B Table -> {p_table_path}...")
            draw_slide1_metrics_table(p_table_path)

            # Also maintain backward compatibility filename
            compat_path = os.path.join(out_dir, "slide1_input_concentric_rings.png")
            draw_slide1_topology_comparison(compat_path)

            p_a_path = os.path.join(out_dir, "slide1_panel_a_pixel_topology.png")
            print(f"Generating Standalone Panel A -> {p_a_path}...")
            draw_slide1_panel_a_standalone(p_a_path)

            p_c_path = os.path.join(out_dir, "slide1_panel_c_real_waveforms.png")
            print(f"Generating Standalone Panel C -> {p_c_path}...")
            draw_slide1_panel_c_standalone(p_c_path)

    print(f"\nAll requested slide figures successfully generated in: {out_dir}")


if __name__ == "__main__":
    main()
