import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from benchmark.viz import plot_bar_multiple
from scannetpp.common.file_io import load_json


PRED_FILES = {
    'avg obj feat+scratch': Path('/cluster/eriador/cyeshwanth/caption3d/mask3d/checkpoints/2212128/instance_evaluation_2212128_ep=129_step=107510_val/val_assigned_caption_preds.json'),
    'baseline': Path('/cluster/eriador/cyeshwanth/caption3d/mask3d/checkpoints/2212117/instance_evaluation_2212117_ep=129_step=107510_val/val_assigned_caption_preds.json')
}

OUT_PATH = Path('/rhome/cyeshwanth/compare_preds/unique_pred_cap_frac.png')
OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

def main():
    labels = []

    series_data = {pred_key: load_json(pred_file) for pred_key, pred_file in PRED_FILES.items()}

    # get labels from first data
    first_key = next(iter(series_data))
    # get all the keys in data[eval][caption_scores] starting with unique_pred
    labels = [key.removeprefix('unique_pred_cap_frac_') for key in series_data[first_key]['eval']['caption_scores'] if key.startswith('unique_pred_cap_frac_')]

    plot_data = {}
    # store values
    for pred_key, data in series_data.items():
        plot_data[pred_key] = [data['eval']['caption_scores'][f'unique_pred_cap_frac_{label}'] for label in labels]

    plot_bar_multiple(labels, plot_data, 'Unique Pred Cap Frac', 'Count', 'Unique Pred Cap Frac', figsize=(20, 10))
    print(f'Saving to: {OUT_PATH}')
    plt.savefig(OUT_PATH); plt.clf(); plt.close()

    # print avg for each series
    for pred_key, data in plot_data.items():
        print(f'{pred_key}: {np.mean(data)}')


if __name__ == '__main__':
    main()