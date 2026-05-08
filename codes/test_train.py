"""MNIST 训练 CLI。"""
import argparse
import gzip
import os
import pickle
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
import mynn.runner as runner_mod
from draw_tools.plot import plot


def load_mnist(train_images_path, train_labels_path):
    with gzip.open(train_images_path, 'rb') as f:
        _, num, _, _ = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(train_labels_path, 'rb') as f:
        _, _ = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    return train_imgs, train_labs


def _parse_int_list(s):
    if not s:
        return []
    return [int(x) for x in s.split(',') if x.strip() != '']


def build_model(model_name, input_dim, num_classes, dropout_p=0.0):
    if model_name == 'mlp':
        return nn.models.Model_MLP([input_dim, 600, num_classes], 'ReLU', None, dropout_p=dropout_p)
    if model_name == 'cnn':
        return nn.models.Model_CNN(dropout_p=dropout_p)
    raise ValueError(f'Unsupported model: {model_name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.06)
    parser.add_argument('--seed', type=int, default=309)
    parser.add_argument('--log_iters', type=int, default=100)
    parser.add_argument('--no_plot', action='store_true')
    parser.add_argument('--save_fig', type=str, default='', help='保存学习曲线 PNG 路径。')
    parser.add_argument('--eval_per_iter', action='store_true', help='每 iter 验证 dev（慢）。')
    parser.add_argument('--eval_interval', type=int, default=0, help='每 N iter 验证 dev；0=每 epoch 末。')
    parser.add_argument('--dropout_p', type=float, default=0.0, help='FC Dropout 概率。')
    parser.add_argument(
        '--scheduler',
        choices=['none', 'multistep', 'step', 'exponential', 'cosine'],
        default='none',
        help='LR 调度：none / multistep / step / exponential / cosine。',
    )
    parser.add_argument(
        '--scheduler_step_on',
        choices=['epoch', 'iter'],
        default='epoch',
        help='调度器步进单位：epoch | iter。',
    )
    parser.add_argument(
        '--milestones',
        type=str,
        default='2,4',
        help='MultiStepLR milestones，逗号分隔。',
    )
    parser.add_argument('--gamma', type=float, default=0.5, help='衰减因子。')
    parser.add_argument('--cosine_t_max', type=int, default=0, help='Cosine T_max；0 自动。')
    parser.add_argument('--cosine_eta_min', type=float, default=0.0, help='Cosine 最小 LR。')
    parser.add_argument(
        '--step_size',
        type=int,
        default=2,
        help='StepLR 步长。',
    )
    parser.add_argument('--train_limit', type=int, default=0, help='仅用训练集前 N 条；0 全量。')
    parser.add_argument('--reuse_idx', action='store_true', help='复用已有 idx.pickle。')
    parser.add_argument('--idx_path', type=str, default='idx.pickle', help='划分文件路径。')
    parser.add_argument('--save_metrics', type=str, default='', help='保存曲线 .npz。')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='',
        help='checkpoint 目录；空则 ./best_models/<model>/。',
    )
    parser.add_argument(
        '--eval_test',
        action='store_true',
        help='训练后用 best dev 权重评官方 test。',
    )
    parser.add_argument(
        '--no_auto_download',
        action='store_true',
        help='禁止自动下载 MNIST。',
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    mnist_dir = os.path.normpath(os.path.join('.', 'dataset', 'MNIST'))
    if not args.no_auto_download:
        from fetch_mnist import ensure_mnist  # noqa: PLC0415

        ensure_mnist(mnist_dir)
    train_images_path = os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz')
    train_imgs, train_labs = load_mnist(train_images_path, train_labels_path)

    idx_path = args.idx_path
    if args.reuse_idx and os.path.exists(idx_path):
        with open(idx_path, 'rb') as f:
            idx = pickle.load(f)
    else:
        idx = np.random.permutation(np.arange(train_imgs.shape[0]))
        with open(idx_path, 'wb') as f:
            pickle.dump(idx, f)
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]

    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    if args.train_limit > 0:
        train_imgs = train_imgs[:args.train_limit]
        train_labs = train_labs[:args.train_limit]

    train_imgs = train_imgs / 255.0
    valid_imgs = valid_imgs / 255.0

    model = build_model(
        args.model,
        train_imgs.shape[-1],
        int(train_labs.max() + 1),
        dropout_p=args.dropout_p,
    )
    optimizer = nn.optimizer.SGD(init_lr=args.lr, model=model)
    if args.scheduler == 'none':
        scheduler = None
    elif args.scheduler == 'multistep':
        scheduler = nn.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=_parse_int_list(args.milestones),
            gamma=args.gamma,
            step_on=args.scheduler_step_on,
        )
    elif args.scheduler == 'step':
        scheduler = nn.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
            step_on=args.scheduler_step_on,
        )
    elif args.scheduler == 'exponential':
        scheduler = nn.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=args.gamma,
            step_on=args.scheduler_step_on,
        )
    elif args.scheduler == 'cosine':
        T_max = int(args.cosine_t_max) if args.cosine_t_max and args.cosine_t_max > 0 else None
        scheduler = nn.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=T_max,
            eta_min=args.cosine_eta_min,
            step_on=args.scheduler_step_on,
        )
    else:
        raise ValueError(f'Unsupported scheduler: {args.scheduler}')
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=int(train_labs.max()) + 1)

    runner = nn.runner.RunnerM(
        model,
        optimizer,
        nn.metric.accuracy,
        loss_fn,
        batch_size=args.batch_size,
        scheduler=scheduler,
        eval_interval=args.eval_interval,
        eval_per_iter=args.eval_per_iter,
    )

    save_dir = args.save_dir.strip() if args.save_dir else ''
    if not save_dir:
        save_dir = os.path.join('.', 'best_models', args.model)
    else:
        save_dir = os.path.normpath(save_dir)

    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=args.epochs, log_iters=args.log_iters, save_dir=save_dir)
    print(f'[{args.model}] best dev accuracy: {runner.best_score:.5f}', flush=True)
    num_iter_epoch = int(train_imgs.shape[0] / args.batch_size) + 1
    if runner.train_scores and num_iter_epoch > 0:
        tail = runner.train_scores[-num_iter_epoch:]
        tail_loss = runner.train_loss[-num_iter_epoch:]
        print(
            f'[{args.model}] last epoch mean train acc: {float(np.mean(tail)):.5f}, '
            f'mean train loss: {float(np.mean(tail_loss)):.5f}',
            flush=True,
        )

    test_score, test_loss = None, None
    if args.eval_test:
        test_images_path = os.path.join(mnist_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte.gz')
        test_imgs, test_labs = load_mnist(test_images_path, test_labels_path)
        test_imgs = test_imgs / 255.0
        ckpt = os.path.join(save_dir, 'best_model.pickle')
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f'Missing checkpoint for test eval: {ckpt}')
        model.load_model(ckpt)
        runner_mod._set_training(model, False)
        test_score, test_loss = runner.evaluate([test_imgs, test_labs])
        runner_mod._set_training(model, True)
        print(f'[{args.model}] test accuracy: {test_score:.5f}, test loss: {test_loss:.5f}', flush=True)

    if args.save_metrics:
        metrics_dir = os.path.dirname(args.save_metrics)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        payload = {
            'train_loss': np.asarray(runner.train_loss, dtype=np.float64),
            'dev_loss': np.asarray(runner.dev_loss, dtype=np.float64),
            'train_scores': np.asarray(runner.train_scores, dtype=np.float64),
            'dev_scores': np.asarray(runner.dev_scores, dtype=np.float64),
            'best_score': np.asarray([runner.best_score], dtype=np.float64),
        }
        if test_score is not None:
            payload['test_acc'] = np.asarray([test_score], dtype=np.float64)
            payload['test_loss'] = np.asarray([test_loss], dtype=np.float64)
        np.savez(args.save_metrics, **payload)

    if (not args.no_plot) or args.save_fig:
        fig, axes = plt.subplots(1, 2)
        fig.set_tight_layout(1)
        runner_mod._set_training(model, False)
        plot(runner, axes.reshape(-1))
        runner_mod._set_training(model, True)

        if args.save_fig:
            os.makedirs(os.path.dirname(args.save_fig), exist_ok=True)
            fig.savefig(args.save_fig, dpi=200)

        if not args.no_plot:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()