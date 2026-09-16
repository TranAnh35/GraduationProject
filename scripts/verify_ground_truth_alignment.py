import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import read_tdms_1d_waveforms

def load_cad_features(json_path='data/ground_truth/specimen_mask_features.json'):
    with open(json_path) as f:
        data = json.load(f)
    features_by_key = {d['key']: d for d in data}
    return features_by_key

def load_transform_csv(csv_path='data/ground_truth/raw_tdms_rotate_crop.csv'):
    df = pd.read_csv(csv_path)
    global_row = df[df['source'] == '__global__'].iloc[0]
    default_crop = {
        'crop_top': int(global_row['crop_top']),
        'crop_bottom': int(global_row['crop_bottom']),
        'crop_left': int(global_row['crop_left']),
        'crop_right': int(global_row['crop_right']),
    }
    
    transforms = {}
    for _, row in df.iterrows():
        src = row['source']
        if src == '__global__':
            continue
        rot = int(row['rotation_deg']) if pd.notnull(row['rotation_deg']) else 0
        flr = int(row['flip_lr']) if pd.notnull(row['flip_lr']) else 0
        fud = int(row['flip_ud']) if pd.notnull(row['flip_ud']) else 0
        ct = int(row['crop_top']) if pd.notnull(row['crop_top']) else default_crop['crop_top']
        cb = int(row['crop_bottom']) if pd.notnull(row['crop_bottom']) else default_crop['crop_bottom']
        cl = int(row['crop_left']) if pd.notnull(row['crop_left']) else default_crop['crop_left']
        cr = int(row['crop_right']) if pd.notnull(row['crop_right']) else default_crop['crop_right']
        transforms[src] = {
            'rot': rot,
            'flip_lr': flr,
            'flip_ud': fud,
            'crop_top': ct,
            'crop_bottom': cb,
            'crop_left': cl,
            'crop_right': cr,
        }
    return default_crop, transforms

def find_transform_for_file(file_path, transforms):
    norm_path = file_path.replace('\\', '/')
    for src_key, trans in transforms.items():
        if src_key in norm_path or os.path.basename(src_key) == os.path.basename(file_path):
            return trans
    return None

def main():
    features_by_key = load_cad_features()
    default_crop, transforms = load_transform_csv()
    
    test_files = [
        ('Corrosion Frontside (rot=0)', 'data/Hall_Pot_Core/Corrosion/Square/corosion_frontside_square_300x300_200Hz_amp2.1_z1_20260105_125851.tdms', 'corrosion'),
        ('Corrosion Backside (rot=270)', 'data/Hall_Pot_Core/Corrosion/Square/corosion_backside_square_300x300_z1_20260108_154956.tdms', 'corrosion'),
        ('Rivet Frontside (rot=180)', 'data/Hall_Pot_Core/Rivet/SquareFrontside/rivet_frontside_square_300x300_z2_20260110_190533.tdms', 'rivet'),
        ('Mixed Frontside (rot=0)', 'data/Hall_Pot_Core/Mixed/SquareFrontside/mixed_frontside_square_300x300_z1_20260112_125721.tdms', 'mixed'),
    ]
    
    fig, axes = plt.subplots(len(test_files), 3, figsize=(18, 5 * len(test_files)))
    
    for row_idx, (title, fpath, spec_key) in enumerate(test_files):
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue
            
        print(f"\nProcessing {title}: {fpath}")
        trans = find_transform_for_file(fpath, transforms)
        print(f"Transform from CSV: {trans}")
        
        # Load raw waveforms and compute P2P
        raw = read_tdms_1d_waveforms(fpath, target_time_samples=50, raster_correction=True)
        raw_p2p = np.ptp(raw, axis=1).reshape(301, 300)[:300, :300]
        
        # 1. Raw unaligned P2P
        ax_raw = axes[row_idx, 0]
        im0 = ax_raw.imshow(raw_p2p, cmap="viridis", origin="lower")
        ax_raw.set_title(f"{title}\nRaw TDMS (P2P [300x300])", fontsize=11, fontweight="bold")
        fig.colorbar(im0, ax=ax_raw, fraction=0.046, pad=0.04)
        
        # Apply transformation to the raw image:
        # Step 1: Rotate (rotation_deg)
        # Note: In numpy/scipy: rot90(k=1) is 90 deg counter-clockwise.
        # Let's test standard rotations: rot=90 -> k=1 or k=3?
        # rot=270 -> 270 deg.
        rot_deg = trans['rot'] if trans else 0
        k = (rot_deg // 90) % 4
        aligned_img = np.rot90(raw_p2p, k=k) # test if counter-clockwise
        
        if trans and trans['flip_lr']:
            aligned_img = np.fliplr(aligned_img)
        if trans and trans['flip_ud']:
            aligned_img = np.flipud(aligned_img)
            
        ct = trans['crop_top'] if trans else 15
        cb = trans['crop_bottom'] if trans else 15
        cl = trans['crop_left'] if trans else 15
        cr = trans['crop_right'] if trans else 15
        
        cropped_img = aligned_img[ct: 300 - cb, cl: 300 - cr]
        
        # 2. Transformed raw scan
        ax_trans = axes[row_idx, 1]
        im1 = ax_trans.imshow(cropped_img, cmap="viridis", origin="lower")
        ax_trans.set_title(f"Transformed Scan (rot={rot_deg}, flip_lr={trans['flip_lr'] if trans else 0})\nCropped [{cropped_img.shape[0]}x{cropped_img.shape[1]}]", fontsize=11, fontweight="bold")
        fig.colorbar(im1, ax=ax_trans, fraction=0.046, pad=0.04)
        
        # Overlay CAD features on the transformed scan
        spec_data = features_by_key[spec_key]
        for feat in spec_data['features']:
            cx = feat.get('corrosionX') if feat.get('corrosionX') is not None else feat.get('x')
            cy = feat.get('corrosionY') if feat.get('corrosionY') is not None else feat.get('y')
            d = feat.get('diameter')
            if cx is not None and cy is not None and d is not None:
                # CAD coordinates (0..300 mm). In cropped coordinates (substract cl, ct):
                # Since origin='lower', CAD x is horizontal (col), y is vertical (row)
                plot_x = cx - cl
                plot_y = cy - ct
                radius = d / 2.0
                circ = Circle((plot_x, plot_y), radius, color='red', fill=False, linewidth=1.5)
                ax_trans.add_patch(circ)
                
        # 3. Generate Nominal CAD Mask directly
        mask = np.zeros((300, 300), dtype=np.int8)
        buffer_px = 2
        for feat in spec_data['features']:
            cx = feat.get('corrosionX') if feat.get('corrosionX') is not None else feat.get('x')
            cy = feat.get('corrosionY') if feat.get('corrosionY') is not None else feat.get('y')
            d = feat.get('diameter')
            if cx is not None and cy is not None and d is not None:
                r = d / 2.0
                # rasterize circle on 300x300
                Y, X = np.ogrid[:300, :300]
                dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                mask[dist <= (r + buffer_px)] = -1
                mask[dist <= r] = 1
                
        cropped_mask = mask[ct: 300 - cb, cl: 300 - cr]
        ax_mask = axes[row_idx, 2]
        im2 = ax_mask.imshow(cropped_mask, cmap="coolwarm", origin="lower", vmin=-1, vmax=1)
        n_def = np.sum(cropped_mask == 1)
        ax_mask.set_title(f"Exact CAD Mask [{cropped_mask.shape[0]}x{cropped_mask.shape[1]}]\n({n_def} Defect Px)", fontsize=11, fontweight="bold")
        fig.colorbar(im2, ax=ax_mask, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = "data/ground_truth/verification_alignment_test.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved alignment verification plot to: {out_path}")

if __name__ == '__main__':
    main()
