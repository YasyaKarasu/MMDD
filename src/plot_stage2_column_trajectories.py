#!/usr/bin/env python
"""Plot actual saved head-training trajectories without interpolation."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def plot(output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    colors = {'C0': '#666666', 'C1': '#1769aa', 'C2': '#b23b3b'}
    records = []
    for arm in ('C0', 'C1', 'C2'):
        for seed in (13, 29):
            path = output/'CHECKPOINTS'/arm/str(seed)/'history.json'
            if not path.is_file():
                continue
            history = json.loads(path.read_text())
            epochs = [r['epoch'] for r in history]
            for axis, field, label in zip(axes, ('loss', 'MRR', 'grad_norm_mean'),
                                           ('Train set-mass CE', 'Dev query-macro MRR', 'Gradient norm before clipping'), strict=True):
                values = [r['dev_query_macro'][field] if field == 'MRR' else r[field] for r in history]
                axis.plot(epochs, values, color=colors[arm], linestyle='-' if seed == 13 else '--',
                          label=f'{arm}, seed {seed}', linewidth=1.5)
                axis.set(xlabel='Complete base-sample epoch', ylabel=label)
                axis.grid(alpha=.2)
            records.extend({'arm': arm, 'seed': seed, 'epoch': r['epoch'], 'base_visits': r['base_visits'],
                            'optimizer_steps_total': r['optimizer_steps_total'], 'loss': r['loss'],
                            'dev_MRR': r['dev_query_macro']['MRR'], 'dev_ColHit1': r['dev_query_macro']['ColHit@1'],
                            'grad_norm_mean': r['grad_norm_mean']} for r in history)
    if not records:
        raise ValueError('No actual training histories to plot')
    axes[1].legend(fontsize=8)
    figure.savefig(output/'TRAJECTORIES.png', dpi=180)
    figure.savefig(output/'TRAJECTORIES.pdf')
    plt.close(figure)
    with (output/'TRAJECTORIES.csv').open('w', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    plot(parser.parse_args().output)
