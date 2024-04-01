import numpy as np
import os
import yaml
from pathlib import Path
from tqdm import tqdm

import experiment
from superpoint.settings import EXPER_PATH


def export_descriptors(config_path, experiment_name, export_name=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  # Use safe_load instead of load
    assert 'eval_iter' in config

    export_name = export_name if export_name else experiment_name
    output_dir = Path(EXPER_PATH, f'outputs/{export_name}/')
    if not output_dir.exists():
        os.makedirs(output_dir)
    checkpoint = Path(EXPER_PATH, experiment_name)
    if 'checkpoint' in config:
        checkpoint = Path(checkpoint, config['checkpoint'])

    with experiment._init_graph(config, with_dataset=True) as (net, dataset):
        if net.trainable:
            net.load(str(checkpoint))
        test_set = dataset.get_test_set()

        pbar = tqdm(total=config['eval_iter'] if config['eval_iter'] > 0 else None)
        i = 0
        while True:
            try:
                data = next(test_set)
            except dataset.end_set:
                break
            data1 = {'image': data['image']}
            data2 = {'image': data['warped_image']}
            pred1 = net.predict(data1, keys=['prob_nms', 'descriptors'])
            pred2 = net.predict(data2, keys=['prob_nms', 'descriptors'])
            pred = {'prob': pred1['prob_nms'],
                    'warped_prob': pred2['prob_nms'],
                    'desc': pred1['descriptors'],
                    'warped_desc': pred2['descriptors'],
                    'homography': data['homography']}

            if not ('name' in data):
                pred.update(data)
            filename = data['name'].decode('utf-8') if 'name' in data else str(i)
            filepath = Path(output_dir, f'{filename}.npz')
            np.savez_compressed(filepath, **pred)
            i += 1
            pbar.update(1)
            if i == config['eval_iter']:
                break


# This allows the script to be used both as a standalone script and an importable module
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--export_name', type=str, default=None)
    args = parser.parse_args()

    export_descriptors(args.config, args.experiment_name, args.export_name)
