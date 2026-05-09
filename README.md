# 神经网络与深度学习PJ1

# 项目框架

基于 NumPy 实现 MNIST 上的 **MLP / CNN**；训练入口统一为 `test_train.py`，其余脚本多为封装或辅助。

```
codes/
├── mynn/                  # 核心实现
│   ├── op.py              # Linear、conv2D、Softmax+交叉熵、ReLU、Dropout 等
│   ├── models.py          # Model_MLP、Model_CNN
│   ├── optimizer.py       # SGD（及预留 MomentGD）
│   ├── lr_scheduler.py    # CosineAnnealingLR（余弦退火）
│   ├── runner.py          # 训练循环、验证、保存 best_model.pickle
│   └── metric.py          # 准确率等指标
├── draw_tools/
│   ├── plot.py              # Runner 单 Run 学习曲线
│   └── metrics_overlay.py   # 读多个 .npz 叠加 dev 曲线（Part C 末尾调用）
├── fetch_mnist.py         # 缺少 MNIST 四个 .gz 时自动下载到 dataset/MNIST/
├── test_train.py          # 主训练脚本（全部 CLI 参数）
├── run_part_a.py          # Part A 一键参数
├── run_part_b.py          # Part B 一键参数
├── run_part_c.py          # Part C 三组对比 + 叠加图
├── run_pipeline.py        # 推荐：Part A → Part B → MLP/CNN 曲线叠图 → Part C
├── test_model.py          # 加载已保存 MLP，在测试集上算准确率
├── weight_visualization.py
└── dataset_explore.ipynb
```

**数据集**：默认路径 `dataset/MNIST/`。本地若无四个 `*-ubyte.gz` 数据集文件，`test_train.py` / `test_model.py` 会先自动下载（可加 `--no_auto_download` 关闭）。

### 推荐运行顺序（项目逻辑）

与作业要求一致，建议按下面顺序跑完实验：

1. **Part A**：训练 MLP → 训练结束后在**官方测试集**上评估（脚本已开 `--eval_test`自动进行测试集评估），终端打印 test accuracy / loss；生成训练曲线与 `figs/part_a_metrics.npz`。
2. **Part B**：训练 CNN（`--reuse_idx` 保证与 Part A 同划分）→ 同样在测试集上评估 → 与 Part A 的 MLP 在 dev 曲线上可比；如需叠图进行对比见下一步。
3. **Part C**：默认对 **MLP 与 CNN** 分别跑 baseline / cosine / dropout 三组；每组实验 **10 epoch**（Part A/B 为 **8 epoch**），各自生成一张组内叠加图与终端汇总。

---

## `run_pipeline.py`（一键流水线）

在 `codes/` 下执行：

```bash
python run_pipeline.py
python run_pipeline.py --pause-between-steps   # 每一步结束后按 Enter 再继续
```

**默认行为**：依次执行 Part A → Part B → A/B dev 叠图 → 打印 MLP/CNN 数值对比 → Part C（`run_part_c.py` 使用其脚本内默认的 `--model both`、`--epochs` 与流水线传入的 `--epochs-c` 一致）。

子进程使用 `python -u` 且设置 `PYTHONUNBUFFERED=1`，日志实时打印。

| 参数 | 默认值 | 可选值 / 说明 |
| --- | --- | --- |
| `--part-c-model` | `both` | `mlp`：Part C 只跑 MLP 三组；`cnn`：只跑 CNN；`both`：先 MLP 再三组 CNN |
| `--epochs-c` | `10` | 任意正整数；传给 `run_part_c.py` 的每组 epoch 数（Part A/B 仍为脚本固定 8） |
| `--pause-between-steps` | 不传（关闭） | 传入该标志则「每完成一大步」等待 Enter |
| `--skip-a` | 不传（不跳过） | 传入则跳过 Part A |
| `--skip-b` | 不传（不跳过） | 传入则跳过 Part B |
| `--skip-c` | 不传（不跳过） | 传入则跳过 Part C |
| `--skip-ab-overlay` | 不传（会生成叠图） | 传入则不生成 `figs/part_ab_mlp_vs_cnn_overlay.png` |

流水线会在 Part B 之后生成 `figs/part_ab_mlp_vs_cnn_overlay.png`，并在进入 Part C 前打印 Part A / Part B 的 metrics 摘要；Part C 结束后按所选架构打印三组 `.npz` 对应摘要。

手动分步（等价拆流水线）可依次：`python run_part_a.py` → `python run_part_b.py` →（叠图逻辑见 `run_pipeline.py` 中 `_overlay_ab_mlp_cnn`）→ `python run_part_c.py`。

---

## Part A — MLP 基线

```bash
python run_part_a.py
```

**命令行**：无额外参数（不提供可选开关），等价于下方固定传给 `test_train.py` 的参数：

| 参数 | 取值（固定） |
| --- | --- |
| `--model` | `mlp` |
| `--epochs` | `8` |
| `--batch_size` | `64` |
| `--lr` | `0.06` |
| `--seed` | `309` |
| `--scheduler` | `none` |
| `--dropout_p` | `0` |
| `--eval_interval` | `100` |
| `--log_iters` | `100` |
| `--no_plot` | 开启 |
| `--save_fig` | `.\figs\part_a_mlp_learning_curve.png` |
| `--save_metrics` | `.\figs\part_a_metrics.npz` |
| `--eval_test` | 开启 |

---

## Part B — CNN（与 Part A 公平对比）

```bash
python run_part_b.py
```

**命令行**：无额外参数。相对 Part A 仅 `--model cnn`、输出路径不同，并**开启** `--reuse_idx`（须先有 Part A 生成的 `idx.pickle` 或同路径已有划分文件）。

| 参数 | 取值（固定） |
| --- | --- |
| `--model` | `cnn` |
| `--reuse_idx` | 开启 |
| `--save_fig` | `.\figs\part_b_cnn_learning_curve.png` |
| `--save_metrics` | `.\figs\part_b_metrics.npz` |
| 其余 | 与 Part A 表相同（`lr=0.06`、`epochs=8` 等） |

**输出**：`figs/part_b_cnn_learning_curve.png`、`figs/part_b_metrics.npz`，权重目录 `best_models/cnn/`（由 `test_train` 默认规则决定，除非改 `save_dir`）。

---

## Part C — 余弦 LR / Dropout 对比

```bash
python run_part_c.py
python run_part_c.py --model mlp
python run_part_c.py --model cnn --epochs 10
```

| 参数 | 默认值 | 可选值 / 说明 |
| --- | --- | --- |
| `--model` | `both` | `mlp`、`cnn`、`both` |
| `--epochs` | `10` | 每组实验的训练轮数（正整数） |

**每组固定**（写在 `_run_suite` 内，不经命令行修改）：`batch_size=64`，`lr=0.1`，`seed=64`，`eval_interval=100`，`log_iters=100`，`--reuse_idx`，`--eval_test`，`--no_plot`。其中 cosine 轮次另含 `--scheduler cosine`、`--scheduler_step_on epoch`、`--cosine_eta_min 0`。

**三轮实验**：

1. baseline：`scheduler=none`，`dropout_p=0`
2. 优化：`scheduler=cosine`，`dropout_p=0`
3. 正则：`scheduler=none`，`dropout_p=0.3`

产物：`figs/part_c_{mlp|cnn}_baseline_*`、`cosine_*`、`dropout_*`，叠加图 `figs/part_c_{mlp|cnn}_overlay.png`，权重 `best_models/part_c_*`。

如需自选多条已有 `.npz` 叠图，可在 Python 里调用 `draw_tools.metrics_overlay.load_metrics` + `plot_many`。

---

## `test_train.py`（通用训练）

```bash
python test_train.py [选项...]
python test_train.py --help
```

| 参数 | 默认值 | 可选值 / 说明 |
| --- | --- | --- |
| `--model` | `mlp` | `mlp`、`cnn` |
| `--epochs` | `5` | 正整数 |
| `--batch_size` | `32` | 正整数 |
| `--lr` | `0.06` | 正浮点（SGD 初始学习率） |
| `--seed` | `309` | 整数 |
| `--log_iters` | `100` | 正整数（每隔多少 iteration 打日志） |
| `--no_plot` | 不传（会尝试弹窗绘图） | 传入则关闭弹窗 |
| `--save_fig` | `''`（不保存） | 任意路径字符串，保存学习曲线 PNG |
| `--eval_per_iter` | 不传 | 传入则每个 iter 验证 dev（很慢） |
| `--eval_interval` | `0` | `0`：每 epoch 末验证；正整数：每 N iter 验证 |
| `--dropout_p` | `0` | `[0,1)` 浮点，全连接 Dropout 概率 |
| `--scheduler` | `none` | `none`：恒定 LR；`cosine`：余弦退火 |
| `--scheduler_step_on` | `epoch` | 仅 `--scheduler cosine` 时有效：`epoch`、`iter` |
| `--cosine_t_max` | `0` | `0`：按训练总长自动；否则为余弦周期参数 |
| `--cosine_eta_min` | `0` | 浮点，余弦最小学习率 |
| `--train_limit` | `0` | `0`：全量训练集；正整数：只用划分后训练集前 N 条 |
| `--reuse_idx` | 不传 | 传入且存在 `idx_path` 则复用划分 |
| `--idx_path` | `idx.pickle` | 划分文件路径 |
| `--save_metrics` | `''`（不保存） | 路径字符串，保存 `.npz` |
| `--save_dir` | `''` | 空则 `./best_models/<model>/` |
| `--eval_test` | 不传 | 传入则用 best dev 权重评官方 test |
| `--no_auto_download` | 不传 | 传入则缺失数据时不自动下载 MNIST |

---

## `test_model.py`（仅 MLP 权重评估）

```bash
python test_model.py
python test_model.py --ckpt .\best_models\mlp\best_model.pickle --no_auto_download
```

| 参数 | 默认值 | 可选值 / 说明 |
| --- | --- | --- |
| `--ckpt` | `.\best_models\mlp\best_model.pickle` | 任意 `best_model.pickle` 路径 |
| `--test_images` | `.\dataset\MNIST\t10k-images-idx3-ubyte.gz` | 官方测试集图像 `.gz` |
| `--test_labels` | `.\dataset\MNIST\t10k-labels-idx1-ubyte.gz` | 官方测试集标签 `.gz` |
| `--no_auto_download` | 不传 | 传入则禁止自动下载 MNIST |

（脚本加载的是 **MLP**；CNN 请用 `test_train.py` 训练结束时的 `--eval_test`。）
