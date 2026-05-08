import argparse
import gzip
import os
from struct import unpack

import mynn as nn
import numpy as np


def load_mnist_flat(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        _, num, _, _ = unpack('>4I', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    with gzip.open(labels_path, 'rb') as f:
        _, _ = unpack('>2I', f.read(8))
        labs = np.frombuffer(f.read(), dtype=np.uint8)
    return imgs, labs


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved Model_MLP on MNIST test set.')
    parser.add_argument(
        '--ckpt',
        type=str,
        default=r'.\best_models\mlp\best_model.pickle',
        help='Path to best_model.pickle (same format as test_train.py saves).',
    )
    parser.add_argument(
        '--test_images',
        type=str,
        default=r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz',
    )
    parser.add_argument(
        '--test_labels',
        type=str,
        default=r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz',
    )
    parser.add_argument(
        '--no_auto_download',
        action='store_true',
        help='Do not download MNIST when files under dataset/MNIST are missing.',
    )
    args = parser.parse_args()

    mnist_dir = os.path.normpath(os.path.join('.', 'dataset', 'MNIST'))
    if not args.no_auto_download:
        from fetch_mnist import ensure_mnist  # noqa: PLC0415

        ensure_mnist(mnist_dir)

    model = nn.models.Model_MLP()
    model.load_model(args.ckpt)
    for layer in getattr(model, 'layers', []):
        if hasattr(layer, 'training'):
            layer.training = False

    test_imgs, test_labs = load_mnist_flat(args.test_images, args.test_labels)
    test_imgs = test_imgs / 255.0

    logits = model(test_imgs)
    acc = nn.metric.accuracy(logits, test_labs)
    print(f'test accuracy: {acc:.5f}')


if __name__ == '__main__':
    main()
