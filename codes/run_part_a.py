from __future__ import annotations

import sys


def main() -> None:
    sys.argv = [
        'test_train.py',
        '--model',
        'mlp',
        '--epochs',
        '8',
        '--batch_size',
        '64',
        '--lr',
        '0.06',
        '--seed',
        '309',
        '--scheduler',
        'none',
        '--dropout_p',
        '0',
        '--eval_interval',
        '100',
        '--log_iters',
        '100',
        '--no_plot',
        '--save_fig',
        r'.\figs\part_a_mlp_learning_curve.png',
        '--save_metrics',
        r'.\figs\part_a_metrics.npz',
        '--eval_test',
    ]
    import test_train  # noqa: PLC0415

    test_train.main()


if __name__ == '__main__':
    main()
