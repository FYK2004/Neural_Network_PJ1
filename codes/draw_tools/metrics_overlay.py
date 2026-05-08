"""读取 save_metrics 的 .npz，叠加 dev 曲线。"""
from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def plot_many(series: List[dict], labels: List[str], title: str, out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    for label, s in zip(labels, series):
        x = np.arange(len(s["train_loss"]))
        axes[0].plot(x, s["dev_loss"], label=f"{label} dev loss")
        axes[1].plot(x, s["dev_scores"], label=f"{label} dev acc")

    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="upper right")

    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("accuracy")
    axes[1].legend(loc="lower right")

    fig.set_tight_layout(True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
