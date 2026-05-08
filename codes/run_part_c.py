"""Part C: baseline / cosine LR / dropout；默认 MLP+CNN；约每 100 iter 验 dev；默认 5 epoch。"""
from __future__ import annotations

import argparse
import os
import sys


def _run_suite(arch: str, epochs: int) -> None:
    tag = arch.lower()
    if tag not in ('mlp', 'cnn'):
        raise ValueError(f'arch must be mlp or cnn, got {arch!r}')

    common = [
        '--model',
        tag,
        '--epochs',
        str(int(epochs)),
        '--batch_size',
        '64',
        '--lr',
        '0.06',
        '--seed',
        '309',
        '--eval_interval',
        '100',
        '--log_iters',
        '100',
        '--reuse_idx',
        '--no_plot',
        '--eval_test',
    ]

    runs = [
        (
            f'[{tag.upper()}] baseline (constant LR, no dropout)',
            [
                '--scheduler',
                'none',
                '--dropout_p',
                '0',
                '--save_dir',
                rf'.\best_models\part_c_{tag}_baseline',
                '--save_fig',
                rf'.\figs\part_c_{tag}_baseline_curve.png',
                '--save_metrics',
                rf'.\figs\part_c_{tag}_baseline_metrics.npz',
            ],
        ),
        (
            f'[{tag.upper()}] Direction 1: cosine LR only',
            [
                '--scheduler',
                'cosine',
                '--scheduler_step_on',
                'epoch',
                '--cosine_eta_min',
                '0',
                '--dropout_p',
                '0',
                '--save_dir',
                rf'.\best_models\part_c_{tag}_cosine',
                '--save_fig',
                rf'.\figs\part_c_{tag}_cosine_curve.png',
                '--save_metrics',
                rf'.\figs\part_c_{tag}_cosine_metrics.npz',
            ],
        ),
        (
            f'[{tag.upper()}] Direction 2: dropout 0.5 only',
            [
                '--scheduler',
                'none',
                '--dropout_p',
                '0.5',
                '--save_dir',
                rf'.\best_models\part_c_{tag}_dropout',
                '--save_fig',
                rf'.\figs\part_c_{tag}_dropout_curve.png',
                '--save_metrics',
                rf'.\figs\part_c_{tag}_dropout_metrics.npz',
            ],
        ),
    ]

    import test_train  # noqa: PLC0415

    for title, extra in runs:
        print('\n' + '=' * 72)
        print(title)
        print('=' * 72 + '\n')
        sys.argv = ['test_train.py'] + common + extra
        test_train.main()

    from draw_tools.metrics_overlay import load_metrics, plot_many  # noqa: PLC0415

    metrics_paths = [
        rf'.\figs\part_c_{tag}_baseline_metrics.npz',
        rf'.\figs\part_c_{tag}_cosine_metrics.npz',
        rf'.\figs\part_c_{tag}_dropout_metrics.npz',
    ]
    labels = ['baseline', 'cosine (opt.)', 'dropout (reg.)']
    series = [load_metrics(p) for p in metrics_paths]
    overlay = rf'.\figs\part_c_{tag}_overlay.png'
    plot_many(
        series,
        labels,
        f'Part C ({tag.upper()}): baseline vs cosine LR vs dropout',
        overlay,
    )
    print(f'\n[saved] {overlay}')

    print(f'\n[Part C summary — {tag.upper()}] (best dev / test, best-dev checkpoint)')
    for lbl, p in zip(labels, metrics_paths):
        d = load_metrics(p)
        dev = float(d['best_score'][0])
        ta = float(d['test_acc'][0]) if 'test_acc' in d else float('nan')
        tl = float(d['test_loss'][0]) if 'test_loss' in d else float('nan')
        print(f'  - {lbl}: dev={dev:.5f}, test_acc={ta:.5f}, test_loss={tl:.5f}')


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='Part C experiments.')
    parser.add_argument(
        '--model',
        choices=['mlp', 'cnn', 'both'],
        default='both',
        help='mlp | cnn | both',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Epochs per experiment.',
    )
    args = parser.parse_args()

    if args.model == 'both':
        for arch in ('mlp', 'cnn'):
            print('\n' + '#' * 72)
            print(f'# Part C suite: {arch.upper()}')
            print('#' * 72)
            _run_suite(arch, args.epochs)
    else:
        _run_suite(args.model, args.epochs)


if __name__ == '__main__':
    main()
