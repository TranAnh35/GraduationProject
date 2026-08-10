#%% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import dataloader

np.random.seed(1)

#%% Configuration ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

data_folder = r'C:\Users\tranl\Downloads'

output_folder = os.path.join(data_folder, 'heatmap_plots')
os.makedirs(output_folder, exist_ok=True)

slice_window = range(0,50)
loi_window   = range(20,30)
peak_window  = range(6,8)

lpf_cutoff = 0.04
lpf_order  = 24

#%% Crop function -----------------------------------------------------------------

def crop_steel(img):

    # chỉnh lại nếu muốn crop vùng khác
    y_min = 10  
    y_max = 110

    x_min = 20
    x_max = 150

    return img[y_min:y_max, x_min:x_max]


#%% Diff method (coil/diff) ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def coil_diff_method(coil_data, diff_data):

    # tránh chia cho 0
    eps = 1e-8

    ratio = coil_data / (diff_data + eps)

    return ratio


#%% Get TDMS files ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

tdms_files = glob.glob(os.path.join(data_folder, '*.tdms'))
tdms_files = [f for f in tdms_files if not f.endswith('_index') and not f.endswith('PECT_default.tdms')]
tdms_files.sort()

print(f"Found {len(tdms_files)} TDMS files")

plt.rc('font', size=20)
# lưu dữ liệu coil và diff để tính ratio
coil_storage = {}
diff_storage = {}


#%% Process each TDMS file ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

for tdms_file in tdms_files:

    filename = os.path.basename(tdms_file)
    print("\nProcessing:", filename)

    try:

        # Load data
        data, samples, sX, sY = dataloader.read_scan_tdms(tdms_file, raster=True)

        data = dataloader.reduce_resolution(data)

        data -= np.mean(data, axis=2, keepdims=True)

        # Low pass filter
        data = data - data[:,:,0:1]

        X_filter = dataloader.LPF_2D(data, cutoff=lpf_cutoff, order=lpf_order)

        X_filter -= np.mean(X_filter, axis=2, keepdims=True)
       
        # =====================================
        # 1. SLICE
        # =====================================

        X_feat_slice = dataloader.extract_features(
            X_filter,
            slice_window,
            method='abs',
            sub_bg_x=False,
            sub_bg_y=False,
            plot=False
        )


        # =====================================
        # 2. LOI
        # =====================================

        X_filter_grad = np.diff(X_filter, axis=2)

        X_feat_loi = dataloader.extract_features(
            X_filter_grad,
            loi_window,
            method='abs',
            sub_bg_x=False,
            sub_bg_y=False,
            plot=False
        )


        # =====================================
        # 3. PEAK
        # =====================================

        X_feat_peak = dataloader.extract_features(
            X_filter_grad,
            peak_window,
            method='max',
            sub_bg_x=False,
            sub_bg_y=False,
            plot=False
        )


       # =====================================
        # 4. COIL / DIFF METHOD (true ratio)
        # =====================================

        # tạo key để ghép coil và diff cùng sample
        name_key = filename.lower()
        name_key = name_key.replace("coil_", "")
        name_key = name_key.replace("diff_", "")
        name_key = name_key.replace("hall_", "")
        name_key = name_key.replace(".tdms", "")

        # lưu heatmap
        if "coil_" in filename.lower():
            coil_storage[name_key] = X_feat_slice

        if "diff_" in filename.lower():
            diff_storage[name_key] = X_feat_slice

        X_feat_diff = np.zeros_like(X_feat_slice)

        # nếu đã có cả coil và diff → tính ratio
        if name_key in coil_storage and name_key in diff_storage:

            coil_map = coil_storage[name_key]
            diff_map = diff_storage[name_key]

            eps = 1e-8
            X_feat_diff = coil_map / (diff_map + eps)
        # ========================================
        # Crop heatmap for Steel
        # ========================================

        if "steel" in filename.lower():

            print("Cropping Steel data")

            X_feat_slice = crop_steel(X_feat_slice)
            X_feat_loi   = crop_steel(X_feat_loi)
            X_feat_peak  = crop_steel(X_feat_peak)
            X_feat_diff  = crop_steel(X_feat_diff)


        # =====================================
        # Plot 4 Heatmaps
        # =====================================

        fig = plt.figure(figsize=(28,7))


        # Slice
        plt.subplot(1,4,1)

        plt.imshow(
            dataloader.normalize_img(X_feat_slice),
            cmap='jet',
            interpolation='bilinear'
        )

        plt.title('Slice')
        plt.colorbar()


        # LOI
        plt.subplot(1,4,2)

        plt.imshow(
            dataloader.normalize_img(X_feat_loi),
            cmap='jet',
            interpolation='bilinear'
        )

        plt.title('LOI')
        plt.colorbar()


        # Peak
        plt.subplot(1,4,3)

        plt.imshow(
            dataloader.normalize_img(X_feat_peak),
            cmap='jet',
            interpolation='bilinear'
        )

        plt.title('Peak')
        plt.colorbar()


        plt.subplot(1,4,4)

        ratio_map = X_feat_slice / (X_feat_diff + 1e-8)

        plt.imshow(
            ratio_map,
            cmap='jet',
            interpolation='bilinear'
        )

        plt.title('Coil / Diff (no norm)')
        plt.colorbar()

        # Save image
        output_filename = os.path.splitext(filename)[0] + "_4methods.png"

        output_path = os.path.join(output_folder, output_filename)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        print("Saved:", output_filename)

        plt.show()
        plt.close()


    except Exception as e:

        print("ERROR:", e)
        continue


print("\n✓ Done")

#%% ------------------------------------------------------------------
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import dataloader

np.random.seed(1)

#%% Configuration ------------------------------------------------------------------

data_folder = '/Users/ADMIN/OneDrive/Desktop/magnetic diffuion'

output_folder = os.path.join(data_folder, 'coil_diff_peak_plots')
os.makedirs(output_folder, exist_ok=True)

peak_window = range(6,8)

lpf_cutoff = 0.04
lpf_order  = 24


#%% Get TDMS files ------------------------------------------------------------------

tdms_files = glob.glob(os.path.join(data_folder, '*.tdms'))
tdms_files = [f for f in tdms_files if not f.endswith('_index') and not f.endswith('PECT_default.tdms')]
tdms_files.sort()

print(f"Found {len(tdms_files)} TDMS files")


#%% Storage ------------------------------------------------------------------

coil_storage = {}
diff_storage = {}


#%% Process files ------------------------------------------------------------------

for tdms_file in tdms_files:

    filename = os.path.basename(tdms_file)
    print("\nProcessing:", filename)

    try:

        # Load TDMS
        data, samples, sX, sY = dataloader.read_scan_tdms(tdms_file, raster=True)

        data = dataloader.reduce_resolution(data)

        data -= np.mean(data, axis=2, keepdims=True)

        data = data - data[:,:,0:1]

        X_filter = dataloader.LPF_2D(data, cutoff=lpf_cutoff, order=lpf_order)

        X_filter -= np.mean(X_filter, axis=2, keepdims=True)


        # ============================
        # PEAK FEATURE
        # ============================

        X_filter_grad = np.diff(X_filter, axis=2)

        X_feat_peak = dataloader.extract_features(
            X_filter_grad,
            peak_window,
            method='max',
            sub_bg_x=False,
            sub_bg_y=False,
            plot=False
        )


        # ============================
        # CREATE NAME KEY
        # ============================

        name_key = filename.lower()
        name_key = name_key.replace("coil_", "")
        name_key = name_key.replace("diff_", "")
        name_key = name_key.replace(".tdms", "")


        # ============================
        # STORE COIL / DIFF
        # ============================

        if "coil_" in filename.lower():

            coil_storage[name_key] = X_feat_peak

        if "diff_" in filename.lower():

            diff_storage[name_key] = X_feat_peak


        # ============================
        # IF BOTH EXIST → PLOT
        # ============================

        if name_key in coil_storage and name_key in diff_storage:

            coil_map = coil_storage[name_key]
            diff_map = diff_storage[name_key]

            ratio_map = coil_map / (diff_map + 1e-8)

            if "steel" in filename.lower():

                print("Cropping Steel data")

                coil_map = crop_steel(coil_map)
                diff_map   = crop_steel(diff_map)
                ratio_map  = crop_steel(ratio_map)

            # ============================
            # PLOT
            # ============================

            fig = plt.figure(figsize=(18,6))


            # COIL
            plt.subplot(1,3,1)

            plt.imshow(
                dataloader.normalize_img(coil_map),
                cmap='jet',
                interpolation='bilinear'
            )

            plt.title("Coil (Peak)")
            plt.colorbar()


            # DIFF
            plt.subplot(1,3,2)

            plt.imshow(
                dataloader.normalize_img(diff_map),
                cmap='jet',
                interpolation='bilinear'
            )

            plt.title("Diff (Peak)")
            plt.colorbar()
            vmin = np.percentile(ratio_map, 5)
            vmax = np.percentile(ratio_map, 95)

            # COIL / DIFF
            plt.subplot(1,3,3)

            plt.imshow(
                ratio_map,
                cmap='jet',
                interpolation='bilinear'
            )

            plt.title("Coil / Diff")
            plt.colorbar()


            # SAVE
            output_filename = name_key + "_coil_diff_peak.png"

            output_path = os.path.join(output_folder, output_filename)

            plt.savefig(output_path, dpi=150, bbox_inches='tight')

            print("Saved:", output_filename)

            plt.show()
            plt.close()


    except Exception as e:

        print("ERROR:", e)
        continue


print("\n✓ Done")
# %%
#%% ------------------------------------------------------------------
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import dataloader

np.random.seed(1)

#%% Configuration ------------------------------------------------------------------

data_folder = '/Users/ADMIN/OneDrive/Desktop/magnetic diffuion'

output_folder = os.path.join(data_folder, 'CTADF_plots')
os.makedirs(output_folder, exist_ok=True)

lpf_cutoff = 0.04
lpf_order  = 24


#%% CTADF Feature ------------------------------------------------------------------

def compute_ctadf_feature(coil, diff):

    # -------------------------
    # 1. Coil amplitude map
    # -------------------------

    coil_amp = np.max(np.abs(coil), axis=2)


    # -------------------------
    # 2. Diff temporal gradient
    # -------------------------

    diff_grad = np.diff(diff, axis=2)

    grad_map = np.max(np.abs(diff_grad), axis=2)


    # -------------------------
    # 3. Coil-Diff waveform distortion
    # -------------------------

    distortion = np.sum(np.abs(coil - diff), axis=2)


    # -------------------------
    # Feature fusion
    # -------------------------

    fusion = coil_amp * grad_map * distortion


    # normalize
    fusion = (fusion - np.min(fusion)) / (np.max(fusion) - np.min(fusion) + 1e-8)

    return fusion


#%% Load TDMS list ------------------------------------------------------------------

tdms_files = glob.glob(os.path.join(data_folder, '*.tdms'))
tdms_files = [f for f in tdms_files if not f.endswith('_index') and not f.endswith('PECT_default.tdms')]
tdms_files.sort()

print(f"Found {len(tdms_files)} TDMS files")

plt.rc('font', size=18)



#%% Storage for fusion -------------------------------------------------------------

coil_storage = {}
diff_storage = {}


#%% Process files ------------------------------------------------------------------

for tdms_file in tdms_files:

    filename = os.path.basename(tdms_file)
    print("Processing:", filename)

    try:

        # Load data
        data, samples, sX, sY = dataloader.read_scan_tdms(tdms_file, raster=True)

        data = dataloader.reduce_resolution(data)

        data -= np.mean(data, axis=2, keepdims=True)

        data = data - data[:,:,0:1]

        X_filter = dataloader.LPF_2D(data, cutoff=lpf_cutoff, order=lpf_order)

        X_filter -= np.mean(X_filter, axis=2, keepdims=True)


        # create name key (match coil/diff pair)

        name_key = filename.lower()
        name_key = name_key.replace("coil_", "")
        name_key = name_key.replace("diff_", "")
        name_key = name_key.replace(".tdms", "")


        if "coil_" in filename.lower():
            coil_storage[name_key] = X_filter

        if "diff_" in filename.lower():
            diff_storage[name_key] = X_filter


        # If both coil and diff exist → fusion

        if name_key in coil_storage and name_key in diff_storage:

            coil_data = coil_storage[name_key]
            diff_data = diff_storage[name_key]

            fusion_map = compute_ctadf_feature(coil_data, diff_data)


            # Plot result

            plt.figure(figsize=(18,6))

            plt.subplot(1,3,1)
            plt.imshow(np.max(np.abs(coil_data),axis=2), cmap='jet')
            plt.title("Coil amplitude")
            plt.colorbar()

            plt.subplot(1,3,2)
            plt.imshow(np.max(np.abs(np.diff(diff_data,axis=2)),axis=2), cmap='jet')
            plt.title("Diff gradient")
            plt.colorbar()

            plt.subplot(1,3,3)
            plt.imshow(fusion_map, cmap='jet')
            plt.title("CTADF Fusion Crack Map")
            plt.colorbar()

            output_name = name_key + "_CTADF.png"

            plt.savefig(os.path.join(output_folder, output_name), dpi=150)

            plt.show()
            plt.close()

            print("Saved:", output_name)

    except Exception as e:

        print("ERROR:", e)


print("✓ Done")
# %%
