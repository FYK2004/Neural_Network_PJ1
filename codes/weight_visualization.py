"""第一层/全连接权重可视化示例。"""import gzip
import os
import pickle
from struct import unpack

import matplotlib.pyplot as plt
import mynn as nn
import numpy as np

from fetch_mnist import ensure_mnist

mnist_dir = os.path.normpath(os.path.join('.', 'dataset', 'MNIST'))
ensure_mnist(mnist_dir)

model = nn.models.Model_MLP()
model.load_model(r'.\saved_models\best_model_1.pickle')

test_images_path = os.path.join(mnist_dir, 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte.gz')

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

mats = []
mats.append(model.layers[0].params['W'])
mats.append(model.layers[2].params['W'])

plt.figure()
plt.matshow(mats[1])
plt.xticks([])
plt.yticks([])
plt.show()