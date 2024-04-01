import numpy as np
import os
import yaml
from pathlib import Path
from tqdm import tqdm

import experiment
from superpoint.settings import EXPER_PATH

def export_detections(config_path, experiment_name, export_name=None, batch_size=1, pred_only=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  # safer way to load YAML files
    assert 'eval_iter' in config

    export_name = export_name if export_name else experiment_name
    output_dir = Path(EXPER_PATH, f'outputs/{export_name}/')
    if not output_dir.exists():
        os.makedirs(output_dir)
    checkpoint = Path(EXPER_PATH, experiment_name)
    if 'checkpoint' in config:
        checkpoint = Path(checkpoint, config['checkpoint'])

    # Adjust for batch processing
    config['model']['pred_batch_size'] = batch_size
    batch_size *= experiment.get_num_gpus()

    with experiment._init_graph(config, with_dataset=True) as (net, dataset):
        if net.trainable:
            net.load(str(checkpoint))
        test_set = dataset.get_test_set()

        pbar = tqdm(total=config['eval_iter'] if config['eval_iter'] > 0 else None)
        i = 0
        while True:
            data = []
            try:
                for _ in range(batch_size):
                    data.append(next(test_set))
            except (StopIteration, dataset.end_set):
                if not data:
                    break
                data += [data[-1] for _ in range(batch_size - len(data))]  # add dummy if needed
            data = dict(zip(data[0], zip(*[d.values() for d in data])))

            if pred_only:
                pred = net.predict(data, keys='pred', batch=True)
                pred = {'points': [np.array(np.where(e)).T for e in pred]}
            else:
                pred = net.predict(data, keys='*', batch=True)

            for d in data:
                filename = d.get('name', str(i)).decode('utf-8') if isinstance(d.get('name'), bytes) else d.get('name', str(i))
                filepath = Path(output_dir, f'{filename}.npz')
                np.savez_compressed(filepath, **pred)
                i += 1
                pbar.update(1)
                if config['eval_iter'] > 0 and i >= config['eval_iter']:
                    break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--export_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pred_only', action='store_true')
    args = parser.parse_args()

    export_detections(args.config, args.experiment_name, args.export_name, args.batch_size, args.pred_only)
