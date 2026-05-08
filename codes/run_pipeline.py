"""Part A→B→叠图→Part C；子进程 python -u。可选 --pause-between-steps。"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _cd_codes_root() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)


def _banner(step: int, total: int, title: str) -> None:
    line = '#' * 72
    print('\n' + line, flush=True)
    print(f'# 步骤 {step}/{total}: {title}', flush=True)
    print(line + '\n', flush=True)


def _maybe_pause(enabled: bool, prompt: str) -> None:
    if not enabled:
        return
    try:
        input(prompt)
    except EOFError:
        pass


def _run_script(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, '-u', script]
    if extra:
        cmd.extend(extra)
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    print('\n' + '=' * 72, flush=True)
    print(' '.join(cmd), flush=True)
    print('=' * 72 + '\n', flush=True)
    r = subprocess.run(cmd, check=False, env=env)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def _print_metrics_row(title: str, path: str) -> None:
    import numpy as np

    if not os.path.isfile(path):
        print(f'  [skip] missing file: {path}', flush=True)
        return
    d = np.load(path, allow_pickle=True)
    dev = float(d['best_score'][0])
    if 'test_acc' in d.files:
        te = float(d['test_acc'][0])
        tl = float(d['test_loss'][0])
        print(f'  {title}: best_dev={dev:.5f}  test_acc={te:.5f}  test_loss={tl:.5f}', flush=True)
    else:
        print(f'  {title}: best_dev={dev:.5f}', flush=True)


def _overlay_ab_mlp_cnn() -> None:
    from draw_tools.metrics_overlay import load_metrics, plot_many

    pa = os.path.join('figs', 'part_a_metrics.npz')
    pb = os.path.join('figs', 'part_b_metrics.npz')
    if not os.path.isfile(pa) or not os.path.isfile(pb):
        print('[warn] skip Part A vs Part B overlay (metrics npz missing)', flush=True)
        return
    series = [load_metrics(pa), load_metrics(pb)]
    plot_many(
        series,
        ['MLP (Part A)', 'CNN (Part B)'],
        'Part A vs Part B: validation curves',
        os.path.join('figs', 'part_ab_mlp_vs_cnn_overlay.png'),
    )
    print(f'[saved] {os.path.join("figs", "part_ab_mlp_vs_cnn_overlay.png")}', flush=True)


def _print_part_c_tables(tag: str) -> None:
    base = os.path.join('figs', f'part_c_{tag}_baseline_metrics.npz')
    cos = os.path.join('figs', f'part_c_{tag}_cosine_metrics.npz')
    dr = os.path.join('figs', f'part_c_{tag}_dropout_metrics.npz')
    print(f'\n--- Part C ({tag.upper()}) metrics ---', flush=True)
    _print_metrics_row('baseline', base)
    _print_metrics_row('cosine (opt.)', cos)
    _print_metrics_row('dropout (reg.)', dr)


def main() -> None:
    _cd_codes_root()

    parser = argparse.ArgumentParser(description='Part A → Part B → overlay → Part C.')
    parser.add_argument(
        '--part-c-model',
        choices=['cnn', 'mlp', 'both'],
        default='both',
        help='Part C architecture suite.',
    )
    parser.add_argument(
        '--epochs-c',
        type=int,
        default=5,
        help='Epochs per Part C experiment.',
    )
    parser.add_argument(
        '--pause-between-steps',
        action='store_true',
        help='Wait for Enter between major steps.',
    )
    parser.add_argument('--skip-a', action='store_true', help='Skip Part A.')
    parser.add_argument('--skip-b', action='store_true', help='Skip Part B.')
    parser.add_argument('--skip-c', action='store_true', help='Skip Part C.')
    parser.add_argument('--skip-ab-overlay', action='store_true', help='Skip A/B overlay figure.')
    args = parser.parse_args()

    pause = args.pause_between_steps
    total_steps = (
        (0 if args.skip_a else 1)
        + (0 if args.skip_b else 1)
        + (0 if args.skip_ab_overlay else 1)
        + 1
        + (0 if args.skip_c else 1)
    )
    si = 0

    if not args.skip_a:
        si += 1
        _banner(si, total_steps, 'Part A — 训练 MLP（约每 100 iter 验 dev；日志 log_iters=100）')
        _run_script('run_part_a.py')
        print('\n--- Part A summary（同上另有 [mlp] test accuracy / loss）---', flush=True)
        _print_metrics_row('MLP', os.path.join('figs', 'part_a_metrics.npz'))
        _maybe_pause(pause, '\n>>> Part A 已完成。按 Enter 开始 Part B...\n')

    if not args.skip_b:
        si += 1
        _banner(si, total_steps, 'Part B — 训练 CNN（同上：eval_interval=100）')
        _run_script('run_part_b.py')
        print('\n--- Part B summary（同上另有 [cnn] test accuracy / loss）---', flush=True)
        _print_metrics_row('CNN', os.path.join('figs', 'part_b_metrics.npz'))
        _maybe_pause(pause, '\n>>> Part B 已完成。按 Enter 生成 MLP vs CNN 叠图...\n')

    if not args.skip_ab_overlay:
        si += 1
        _banner(si, total_steps, '叠图 — Part A vs Part B 验证曲线')
        _overlay_ab_mlp_cnn()
        _maybe_pause(pause, '\n>>> 叠图已保存。按 Enter 打印数值对比并进入 Part C...\n')

    si += 1
    _banner(si, total_steps, '数值对比 — MLP vs CNN（best-dev → test）')
    print(flush=True)
    _print_metrics_row('MLP (Part A)', os.path.join('figs', 'part_a_metrics.npz'))
    _print_metrics_row('CNN (Part B)', os.path.join('figs', 'part_b_metrics.npz'))

    if not args.skip_c:
        _maybe_pause(pause, '\n>>> 按 Enter 开始 Part C（多组实验，耗时较长）...\n')
        si += 1
        _banner(si, total_steps, 'Part C — baseline / cosine / dropout（子脚本内含多轮训练）')
        extra = ['--model', args.part_c_model, '--epochs', str(int(args.epochs_c))]
        _run_script('run_part_c.py', extra)
        if args.part_c_model in ('cnn', 'both'):
            _print_part_c_tables('cnn')
        if args.part_c_model in ('mlp', 'both'):
            _print_part_c_tables('mlp')

    print('\n[pipeline] done.', flush=True)


if __name__ == '__main__':
    main()
