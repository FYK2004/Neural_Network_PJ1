# 项目框架

基于 NumPy 实现 MNIST 上的 **MLP / CNN**；训练入口统一为 `test_train.py`，其余脚本多为封装或辅助。

```
codes/
├── mynn/                  # 核心实现
│   ├── op.py              # Linear、conv2D、Softmax+交叉熵、ReLU、Dropout 等
│   ├── models.py          # Model_MLP、Model_CNN
│   ├── optimizer.py       # SGD（及预留 MomentGD）
│   ├── lr_scheduler.py    # MultiStep / Step / Exponential / Cosine 等
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
├── hyperparameter_search.py
└── dataset_explore.ipynb
```

**数据**：默认路径 `dataset/MNIST/`。本地若无四个 `*-ubyte.gz`，`test_train.py` / `test_model.py` 会先自动下载（可加 `--no_auto_download` 关闭）。

### 推荐运行顺序（项目逻辑）

与作业叙事一致，建议按下面顺序跑完实验：

1. **Part A**：训练 MLP → 训练结束后在**官方测试集**上评估（脚本已开 `--eval_test`），终端打印 test accuracy / loss；曲线与 `figs/part_a_metrics.npz`。
2. **Part B**：训练 CNN（`--reuse_idx` 与 Part A 同划分）→ 同样在测试集评估 → 与 Part A 的 MLP 在 dev 曲线上可比；如需叠图见下一步。
3. **Part C**：默认对 **MLP 与 CNN** 分别跑 baseline / cosine / dropout 三组；每组实验 **5 epoch**（Part A/B 为 **8 epoch**），各架构一张组内叠加图与终端汇总。

**一键顺序执行**（在 `codes/` 下）：

```bash
python run_pipeline.py
python run_pipeline.py --pause-between-steps   # 每大步完成后按 Enter 再继续（便于逐步盯进度）
```

流水线对子进程使用 **`python -u`** 且设置 **`PYTHONUNBUFFERED`**，训练日志会**实时打印**；步骤标题为 **`# 步骤 i/N:`**。若要完全手动分命令执行，可依次：`python run_part_a.py` → `python run_part_b.py` →（可选叠图见 `run_pipeline` 源码 `_overlay_ab_mlp_cnn`）→ `python run_part_c.py`。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--part-c-model` | `both` | Part C 跑 `mlp` / `cnn` / `both` |
| `--epochs-c` | `5` | Part C 每组 epoch 数（A/B 为各 8） |
| `--pause-between-steps` | 关 | 每完成一大步后等待 Enter 再继续 |
| `--skip-a` / `--skip-b` / `--skip-c` | 关 | 复用已有输出时跳过对应阶段 |
| `--skip-ab-overlay` | 关 | 不生成 MLP vs CNN 的 dev 叠图 |

流水线会在 Part B 之后生成 **`figs/part_ab_mlp_vs_cnn_overlay.png`**（Part A 与 Part B 的验证集曲线对比），并在末尾再次打印 Part A / Part B 的 dev 与 test 数值表；Part C 结束后打印三组 metrics 路径对应的摘要。

---

## Part A — MLP 基线

**命令**（在 `codes/` 下）：

```bash
python run_part_a.py
```

**等价于**（脚本内写死的参数，无额外 CLI）：

| 参数 | 取值 |
|------|------|
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

**命令**：

```bash
python run_part_b.py
```

**等价参数**（相对 Part A 仅换模型并固定划分）：

| 参数 | 取值 |
|------|------|
| `--model` | `cnn` |
| 其余与 Part A 相同 | 同上表 |
| `--reuse_idx` | **开启**（复用 `idx.pickle`，须先跑过 Part A 或已有划分文件） |

**输出**：`figs/part_b_cnn_learning_curve.png`、`figs/part_b_metrics.npz`，权重目录 `best_models/cnn/`。

---

## Part C — 两个拓展方向（余弦 LR / Dropout）

**命令**：

```bash
python run_part_c.py [--model mlp|cnn|both] [--epochs N]
```

| CLI | 默认 | 说明 |
|-----|------|------|
| `--model` | `both` | `mlp`：只跑 MLP 三组；`cnn`：只跑 CNN；`both`：先 MLP 再三组 CNN |
| `--epochs` | `5` | 每一组实验的 epoch 数（Part A/B 由 `run_part_a/b.py` 固定为 8） |

**每组固定**（写在 `_run_suite` 内）：`batch_size=64`，`lr=0.06`，`seed=309`，`scheduler_step_on=epoch`（若用 cosine），`eval_interval=100`（约**每 100 iteration**在 dev 上验证一次），`log_iters=100`，`--reuse_idx`，`--eval_test`，`--no_plot`。

**三轮实验**：

1. baseline：`scheduler=none`，`dropout_p=0`  
2. 优化：`scheduler=cosine`，`cosine_eta_min=0`，`dropout_p=0`  
3. 正则：`scheduler=none`，`dropout_p=0.5`  

产物：`figs/part_c_{mlp|cnn}_baseline_*`、`cosine_*`、`dropout_*`，叠加图 `figs/part_c_{mlp|cnn}_overlay.png`，权重 `best_models/part_c_*`。

如需自选多条已有 `.npz` 叠图，可在 Python 里调用 `draw_tools.metrics_overlay.load_metrics` + `plot_many`。

---

## 通用训练：`test_train.py`

```bash
python test_train.py [参数...]
python test_train.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `mlp` | `mlp` \| `cnn` |
| `--epochs` | `5` | 训练轮数 |
| `--batch_size` | `32` | 批大小 |
| `--lr` | `0.06` | SGD 学习率 |
| `--seed` | `309` | 随机种子 |
| `--log_iters` | `100` | 每隔多少 iteration 打印日志 |
| `--eval_interval` | `0` | `0`：每 epoch 末验证；`>0`：每 N iter 验证 |
| `--eval_per_iter` | 关 | 每个 iter 验证（很慢） |
| `--dropout_p` | `0` | 全连接隐层 Dropout 概率 |
| `--scheduler` | `none` | `none` \| `multistep` \| `step` \| `exponential` \| `cosine` |
| `--scheduler_step_on` | `epoch` | `epoch` \| `iter`，调度器步进单位 |
| `--milestones` | `2,4` | MultiStepLR，逗号分隔 |
| `--gamma` | `0.5` | 衰减因子 |
| `--step_size` | `2` | StepLR 步长 |
| `--cosine_t_max` | `0` | Cosine `T_max`，`0` 为自动 |
| `--cosine_eta_min` | `0` | Cosine 最小学习率 |
| `--train_limit` | `0` | 划分后仅用训练集前 N 条，`0` 为全量 |
| `--reuse_idx` | 关 | 复用已有 `idx.pickle` |
| `--idx_path` | `idx.pickle` | 划分文件路径 |
| `--save_dir` | `''` | checkpoint 目录；空则为 `./best_models/<model>/` |
| `--save_fig` | `''` | 保存学习曲线 PNG |
| `--save_metrics` | `''` | 保存 `.npz`（train/dev 曲线、`best_score`，可选 `test_*`） |
| `--eval_test` | 关 | 用最优 dev 权重在官方测试集上评估 |
| `--no_plot` | 关 | 不弹窗显示图 |
| `--no_auto_download` | 关 | 禁止自动下载 MNIST |

---

## 测试集评估已保存模型：`test_model.py`

```bash
python test_model.py [--ckpt 路径] [--test_images 路径] [--test_labels 路径] [--no_auto_download]
```

| 参数 | 默认 |
|------|------|
| `--ckpt` | `.\best_models\mlp\best_model.pickle` |
| `--test_images` | `.\dataset\MNIST\t10k-images-idx3-ubyte.gz` |
| `--test_labels` | `.\dataset\MNIST\t10k-labels-idx1-ubyte.gz` |

（当前脚本面向 **MLP**；CNN 可用 `test_train.py` 训练结束时的 `--eval_test`。）
