import json
import numpy as np

with open('experiments/5x5/exp10_residual_diff_20260925_010642/evaluation_results/evaluation_summary.json') as f:
    data = json.load(f)

files = data['per_file_results']
print(f'Total test files evaluated: {len(files)}')

def get_stats(subset):
    aucs = [x['metrics']['linear_probe_auc_roc'] for x in subset if 'linear_probe_auc_roc' in x['metrics']]
    aps = [x['metrics']['linear_probe_average_precision'] for x in subset if 'linear_probe_average_precision' in x['metrics']]
    cnrs = [x['metrics']['contrast_ratio_cnr'] for x in subset if 'contrast_ratio_cnr' in x['metrics']]
    r2s = [x['metrics']['depth_r2'] for x in subset if 'depth_r2' in x['metrics']]
    maes = [x['metrics']['depth_mae_mm'] for x in subset if 'depth_mae_mm' in x['metrics']]
    f1s = [x['metrics']['severity_macro_f1'] for x in subset if 'severity_macro_f1' in x['metrics']]
    mlp_r2s = [x['task2_depth_regression']['mlp_2layer']['r2_score'] for x in subset]
    mlp_aucs = [x['task1_anomaly_detection']['mlp_2layer']['auc_roc'] for x in subset]
    mlp_aps = [x['task1_anomaly_detection']['mlp_2layer']['average_precision'] for x in subset]
    return {
        'n': len(subset),
        'auc': float(np.mean(aucs) * 100),
        'auc_std': float(np.std(aucs) * 100),
        'ap': float(np.mean(aps) * 100),
        'cnr': float(np.mean(cnrs)),
        'r2': float(np.mean(r2s)),
        'mlp_r2': float(np.mean(mlp_r2s)),
        'mlp_auc': float(np.mean(mlp_aucs) * 100),
        'mlp_ap': float(np.mean(mlp_aps) * 100),
        'mae': float(np.mean(maes)),
        'f1': float(np.mean(f1s))
    }

print('\n=== OVERALL ===')
print(json.dumps(get_stats(files), indent=2))

print('\n=== BY SPECIMEN ===')
for spec in ['corrosion', 'rivet']:
    sub = [x for x in files if spec in x['specimen'].lower()]
    print(f'{spec}:', json.dumps(get_stats(sub), indent=2))

print('\n=== BY WAVEFORM ===')
for wf in ['Chirp', 'Square', 'Gaussian']:
    sub = [x for x in files if x['metadata']['waveform'].lower() == wf.lower()]
    print(f'{wf}:', json.dumps(get_stats(sub), indent=2))

print('\n=== BY SENSOR ===')
for s in ['Hall_Air_Core', 'Hall_Pot_Core', 'TMR']:
    sub = [x for x in files if x['metadata']['sensor'].lower() == s.lower()]
    print(f'{s}:', json.dumps(get_stats(sub), indent=2))

print('\n=== BY LIFTOFF ===')
for z in ['z1', 'z2', 'z3']:
    sub = [x for x in files if x['metadata']['liftoff'].lower() == z.lower()]
    print(f'{z}:', json.dumps(get_stats(sub), indent=2))
