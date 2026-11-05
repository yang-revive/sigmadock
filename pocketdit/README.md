# PocketDiT

一个精简的、基于口袋的分子对接模型：**Pocket-conditioned Geometric DiT + Flow Matching**。
完全独立于上一级目录的 `sigmadock`（不 import 它的任何代码），只依赖 `torch`、`numpy`、`rdkit`。

```
pocketdit/
  chem.py    PDB/SDF 解析、口袋提取、配体与口袋特征化              ~220 行
  fragment.py 刚性片段分解（切可旋转键 + 合并），尚未接入模型       ~310 行
  data.py    预处理成 .pt 缓存、Dataset、padding 批处理              ~160 行
  model.py   静态口袋编码器 + Geometric DiT + 等变输出头              ~260 行
  flow.py    先验、插值路径、损失、Euler 采样器                       ~85 行
  train.py   纯 PyTorch 训练循环（AdamW、warmup+cosine、EMA、bf16）    ~200 行
  dock.py    推理命令行，输出 SDF                                     ~120 行
tests/       等变性、padding、采样器、单复合物过拟合、端到端命令行
```

## 模型

```
Protein Pocket ──► PocketEncoder（静态，与 t 和配体位姿无关；采样时只算一次）──► h_p
                                                                                │
Ligand 分子图 + x_t ──► Atom Embedding ──► Geometric DiT × depth ──► Geometric Head ──► x̂_1
                                              ├ Ligand Self-Attention   bias: RBF(d_ij(x_t)) + 键类型 + 拓扑距离
                                              ├ Pocket Cross-Attention  bias: RBF(d_ik(x_t, 口袋原子))
                                              ├ MLP
                                              └ adaLN-Zero（flow time t 调制三个子层）
```

- **Token**：配体每个重原子一个 token；口袋取参考配体 8 Å 内残基的全部重原子（约 350–450 个）。
- **几何只通过注意力 bias 进入主干**：用 `F.scaled_dot_product_attention` 的加性 mask 实现，主干几乎全是稠密矩阵乘法，没有 PairFormer 和 triangle 操作。
- **输出头**：`Δx_i = mean_j w_ij·r̂_ij + mean_k w_ik·r̂_ik`，其中 w 是不变标量，r̂ 是相对向量，所以 `x̂_1 = x_t + Δx` 在构造上就是 SE(3) 等变的，不需要旋转数据增强。
- **手性**：只看距离的模型分不清镜像，生成时会翻转手性中心。所以给每个立体中心加一个特征 `sign · 有向体积(x_t)`：它在旋转下不变，在镜像下变号。参考符号取自输入的 3D 构象，避开了 RDKit 中 CW/CCW 邻居顺序的约定。这样模型是 SE(3) 而不是 E(3) 等变的，有测试覆盖。
- adaLN-Zero 和输出头都是零初始化：初始化时模型恒等输出 `x̂_1 = x_t`。

## Flow Matching

| | |
|---|---|
| 路径 | `x_t = (1−t)·x_0 + t·x_1`，t=1 是真实位姿 |
| 先验 | `x_0 ~ N(口袋中心, 4² Å²)`，训练时中心另加 1 Å 抖动 |
| 预测目标 | `x̂_1 = E[x_1 \| x_t]`，速度 `v = (x̂_1 − x_t)/(1−t)` |
| 损失 | 坐标 MSE（Å²，即每个复合物的 RMSD²）+ `t > 0.75` 时对真实距离 < 3 Å 的原子对（键长、键角）加距离损失 |
| 采样 | 20 步 Euler ODE，最后一步正好落在 `x̂_1` 上；口袋编码缓存复用 |

## 使用

```bash
pip install -e ".[test]"        # 在 pocketdit/ 目录下
```

**1. 预处理**：每个复合物只用 RDKit 解析一次，存成 `.pt` 文件。目录结构与 PDBBind 相同，每个复合物一个子文件夹：

```bash
python -m pocketdit.data --root data/pdbbind/refined-set --out cache/train \
    --protein_glob "*_protein.pdb" --ligand_glob "*_ligand.sdf" --pocket_cutoff 8
python -m pocketdit.data --root data/pdbbind/core-set --out cache/val
```

也可以用 `--csv index.csv`，列为 `protein,ligand[,ref]`，路径相对于 CSV 所在目录。交叉对接时用 `--ref_glob` 指定参考配体，由它来定义口袋。解析失败的复合物会记录在 `<out>/failures.txt`。

**2. 训练**：`TrainConfig`、`ModelConfig`、`FlowConfig` 的每个字段都是命令行参数。

```bash
python -m pocketdit.train --train_dir cache/train --val_dir cache/val --out_dir runs/exp1 \
    --batch_size 16 --lr 3e-4 --max_steps 200000 --model.dim 256 --model.depth 6 --flow.prior_std 4
```

日志写到 `runs/exp1/log.jsonl`。验证时用 EMA 模型报告 loss、采样 RMSD 中位数和 RMSD < 2 Å 的比例。`--resume runs/exp1/last.pt` 可以断点续训。

**3. 对接**：

```bash
# 重对接（口袋由配体自身坐标定义）
python -m pocketdit.dock --ckpt runs/exp1/last.pt --protein p.pdb --ligand l.sdf --num_samples 10
# 交叉对接 / 实际使用：用参考配体或口袋中心定义口袋
python -m pocketdit.dock --ckpt runs/exp1/last.pt --protein p.pdb --ligand query.sdf --ref ref.sdf
python -m pocketdit.dock --ckpt runs/exp1/last.pt --protein p.pdb --ligand query.sdf --center 12.3 4.5 -7.8
```

每个复合物输出 `docked/<name>.sdf`，包含 `--num_samples` 个位姿，**未排序**。属性 `rmsd_to_input` 是相对输入坐标的对称性校正 RMSD，只在重对接时有意义。

## 测试

```bash
pytest -q        # CPU 约 45 秒，使用 ../notebooks/dummy_data 中的 10 个 Astex 复合物
```

覆盖内容：PDB 解析和口袋提取；手性特征的符号；随机权重下的 SE(3) 等变性；镜像下输出确实不同；padding 不变性；用"总是给出真值的 oracle"检验采样器的积分；小模型从噪声过拟合单个复合物（中位 RMSD < 1.5 Å）；预处理 → 训练 → 对接的端到端命令行。

## 刚性片段分解（`fragment.py`）

当前模型是全原子扩散。`fragment.py` 是片段化版本的第一块，**还没有接入模型**，可以单独跑来看统计：

```bash
python -m pocketdit.fragment --root data/pdbbind/refined-set --selection compact
python -m pocketdit.fragment --smiles "CCc1ccccc1" "COc1ccccc1"   # 快速查看单个分子
```

做法：切开所有可旋转键，然后在保持每个片段刚性的前提下尽量把切口合并回去。判定刚性时只看片段子图——一条在整个分子里可旋转的键，切断之后往往因为一侧没有重原子邻居而不再是扭转键，这正是合并能成立的原因。`selection` 控制合并顺序：`compact`（确定性，推理用）、`random`（训练用增强）、`none`（消融）。

几个在实现过程中确认下来的结论：

- **最小片段尺寸是最终状态的性质，不是逐步不变量。** 十碳烷切开 7 条扭转键会一次隔离出 6 个单原子，任何单步合并都只能救回一个，所以逐步卡最小尺寸会把所有合并都拒掉。只有刚性约束逐步生效。
- **2 原子片段无法避免，而且很常见。** 甲氧基、乙基、羟甲基、硫甲基都挂在分子唯一的可旋转键上，切完必然是 2 个原子。这类片段绕自身轴的旋转不改变任何原子坐标，所以它有 5 个真实自由度而 SE(3) 给了 6 个；多出来的那一维是 gauge 冗余而不是信息缺失，必须由做刚体拟合的那一方在数值上定住（`is_collinear` 负责标记）。
- **合并保护不了手性中心。** 四面体中心至少有 3 个重原子邻居，所以它自己的刚体块已经超过豁免尺寸，而且相邻扭转键两侧都还留着邻居，这条键在任何合并下都会存活。结论是立体中心相邻的扭转键一定被切开，手性只能交给模型通过 `chem.chirality_reference` 的有向体积特征处理——和全原子模式完全一样。
- 合并顺序是贪心的，不是最优的。两个启发式（填满到豁免尺寸、优先合并最小片段）各有死角且互不支配，所以两个都跑，按"先避免孤立原子、再比片段数"取优。

10 个 Astex 配体上的数字（平均 26 个重原子、6.4 条扭转键）：

| | `none` | `compact` |
|---|---|---|
| 平均片段数 | 7.4 | 5.4 |
| 合并掉的切口 | 0 / 64 | 20 / 64 |
| gauge 退化片段占比 | 54.1% | **16.7%** |

## 已知局限和下一步

- **还没有在真实数据集上训练过**，没有 benchmark 数字。默认超参数（先验 4 Å、口袋 8 Å、距离损失权重等）都是合理的起点，没有经过调优。
- **没有置信度或打分模型**，多个采样位姿无法排序。可以加一个 confidence head，或者接 GNINA / Vina 重打分。
- 坐标 MSE 没有处理**对称等价原子**（例如羧基的两个 O、对称苯环），评估时请用对称性校正的 RMSD（`dock.py` 中已用 RDKit `CalcRMS`）。
- 训练时口袋由真实配体选出，口袋形状会泄露一部分配体信息；交叉对接时可以加大 `--pocket_cutoff` 或对 cutoff 加噪声。
- 可尝试的扩展：harmonic prior（按键图的拉普拉斯构造先验）、self-conditioning、非均匀的 t 采样和时间步长，以及 SigmaDock 式的刚性片段参数化（输出头不变，把逐原子向量聚合成片段的力和力矩）。
