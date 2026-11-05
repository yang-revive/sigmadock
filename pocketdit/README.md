# PocketDiT

一个精简的、基于口袋的分子对接模型：**Pocket-conditioned Geometric DiT + Flow Matching**。
完全独立于上一级目录的 `sigmadock`（不 import 它的任何代码），只依赖 `torch`、`numpy`、`rdkit`。

```
pocketdit/
  chem.py    PDB/SDF 解析、口袋提取与中心、配体与口袋特征化          ~280 行
  data.py    预处理成 .pt 缓存、口袋预算裁剪、Dataset、padding 批处理  ~250 行
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

- **Token**：配体每个重原子一个 token；口袋是**按距离排序、整残基选取、截到 256 个原子**的重原子集合（见下文"口袋预算"）。
- **几何只通过注意力 bias 进入主干**：用 `F.scaled_dot_product_attention` 的加性 mask 实现，主干几乎全是稠密矩阵乘法，没有 PairFormer 和 triangle 操作。
- **输出头**：`Δx_i = mean_j w_ij·r̂_ij + mean_k w_ik·r̂_ik`，其中 w 是不变标量，r̂ 是相对向量，所以 `x̂_1 = x_t + Δx` 在构造上就是 SE(3) 等变的，不需要旋转数据增强。
- **手性**：只看距离的模型分不清镜像，生成时会翻转手性中心。所以给每个立体中心加一个特征 `sign · 有向体积(x_t)`：它在旋转下不变，在镜像下变号。参考符号取自输入的 3D 构象，避开了 RDKit 中 CW/CCW 邻居顺序的约定。这样模型是 SE(3) 而不是 E(3) 等变的，有测试覆盖。
- adaLN-Zero 和输出头都是零初始化：初始化时模型恒等输出 `x̂_1 = x_t`。

## 口袋预算

缓存时按 12 Å 存下最宽的那一层壳（739–1019 个原子，约 58 KB/复合物），加载每个复合物时再裁到预算（`data.PocketCrop`）。缓存留的余量是给方向性噪声用的，必须覆盖 `cutoff + cutoff_jitter + dir_noise`，否则位移后的壳会在那一侧被悄悄切掉 —— `ComplexDataset` 启动时会检查并报错。

- **只按整残基选，按残基到参考配体的最小距离排序。** 4 Å 以内的是接触壳，优先保留；其余按从近到远填满预算。
- **4 Å 核心 + 不设 cutoff 单靠 256 预算填**，在 10 个示例复合物上等效 cutoff 自己就落在平均 **6.5 Å**（范围 5.8–6.9）—— 而这正是 SigmaDock 训练用的口袋尺度（`pocket_distance_cutoff: 6.0` + `pocket_distance_noise: 0.5`）。所以 `cutoff` 默认取 6.5：两个控制量互相印证，实测下来两者都会生效（约 4 个复合物先碰到预算，约 6 个先碰到 cutoff），最终 **203–253 个原子 / 等效 5.8–6.5 Å**，对比 8 Å 整壳的 348–444 个。
- 显存上，10 个复合物一个 batch：`PocketEncoder` 里 `[B,M,M,16]` 的 RBF 从 126 MB 降到 41 MB，`z_lp` 的 `[B,N,M,64]` 从 40 MB 降到 23 MB。
- 半个残基对 `ATOM_NAMES` / `RESIDUES` 嵌入是分布外的（没有 CA 的骨架 N、苯环的三个碳），所以不做原子级选取；按距离截断也优于在 4–8 Å 带里均匀采样 —— 后者会留下 7.9 Å 的原子而丢掉 4.1 Å 的。
- 顺带的收益：输出头把口袋项按 `n_p` 取平均，400 个大多很远的原子会把约 37 个真实接触稀释 10 倍，而 `n_p` 在不同复合物间从 126 变到 600 又会让这一项的尺度跟着变。
- **预算是上限不是配额。** 小而深的口袋就保持小，不会用溶剂暴露的残基填到 256；核心本身超预算时丢掉它最远的残基，大配体退化成最近的接触而不是整条样本作废。
- **随机性加在边界上，不加在单个原子上**：方向性位移（下面单独说）+ cutoff 抖动（`U[cutoff ± 1.5]`）+ 整残基 dropout。随机丢原子会在口袋壁上打洞，同一个复合物在不同 epoch 学到的立体场就不自洽了。验证和推理用确定性的那层壳。

### 方向性位移（`dir_noise`）

不加噪声时，壳的边界在 ~1 Å 尺度上描出了配体的轮廓 —— 模型可以直接从口袋形状把答案读出来。所以外壳不是按真实参考配体选的，而是按**刚体位移过的**参考配体选：`u` 方向球面均匀，`|u| ~ U[0, dir_noise]`（所以 `dir_noise` 是上界，均值是它的一半，这样模型能同时见到口袋定义很准和很差的情况）。

实测（10 个示例复合物 × 20 次采样），"选中残基与无位移时的差异比例"和"口袋质心到配体质心的距离"：

| `dir_noise` | 原子数 | 残基集合变化 | 口袋质心离配体质心 |
|---|---|---|---|
| 0 | 237 | 0% | 2.21 Å |
| 2 | 239 | 17% | 2.47 Å |
| **4**（默认） | 241 | **31%** | **3.05 Å** |

边界被实质性打散，而原子数几乎不动（237→241），预算不受影响。

两点说明：

- **它同时在合成交叉对接。** 推理时用参考配体定口袋，参考配体的原子离查询配体的真实位姿就是几 Å —— 位移过的真配体正是这个分布。所以不需要真实的交叉对接数据就能把那个输入分布造出来。
- **接触核心故意不位移。** 那 15 个残基是信号本身；它泄漏的是"配体的大致位置经 4 Å 膨胀后的轮廓"，而结合位点的位置本来就是任务的合法输入，而且已经通过 `pocket_com` 明确给了模型。

它**没有**解决的是壳的**拉伸方向**（主轴夹角在 28–33° 之间，刚体位移保留了它）。但这一项大部分是正当信号：结合位点本来就是一道沟，配体沿沟躺着，沟的方向是蛋白的性质而不是从答案泄漏的。

### 为什么不用球

把区域改成"中心 + 半径 R 的球"能彻底去掉形状泄漏，但实测代价太大：一个刚好装下 4 Å 核心残基的球要 **13.9 Å 半径 / 493 个原子**（最大 16.9 Å / 782 个），装下 6.5 Å 整壳要 **16.6 Å / 795 个**（最大 19.5 Å / 1076 个），而贴配体的壳只要 243 个 —— **同样的接触残基，球要花 2–3.3 倍的原子，其中 68%（50–84%）离配体超过 6.5 Å**。根本原因是配体细长、区域需要各向异性，而球是等向的。

## Flow Matching

| | |
|---|---|
| 路径 | `x_t = (1−t)·x_0 + t·x_1`，t=1 是真实位姿 |
| 先验 | `x_0 ~ N(口袋中心, 4² Å²)`，中心抖动训练 1 Å / 采样 0.25 Å |
| 预测目标 | `x̂_1 = E[x_1 \| x_t]`，速度 `v = (x̂_1 − x_t)/(1−t)` |
| 损失 | 坐标 MSE（Å²，即每个复合物的 RMSD²）+ `t > 0.75` 时对真实距离 < 3 Å 的原子对（键长、键角）加距离损失 |
| 采样 | 20 步 Euler ODE，最后一步正好落在 `x̂_1` 上；口袋编码缓存复用 |

**口袋中心（`pocket_com`）**：参考配体 6 Å 以内**蛋白重原子**的质心，和 SigmaDock 的 `extract_pocket_com` 一致 —— 不是参考配体自己的质心。模型是平移等变的（主干只看距离，输出头只输出相对向量），绝对坐标进不了网络，所以中心**只**通过先验起作用；要训练和推理一致的，是"先验中心到真实配体质心"这个偏移的**分布**。

用配体质心做中心会破坏这个一致性：重对接训练时它就是答案的质心，模型不需要学平移；到推理时中心偏出几 Å，模型就系统性地平移不足。口袋质心在 10 个示例复合物上离真实配体质心 1.7 Å（0.8–3.5 Å），是一个真要去解的平移；抖动则负责把推理时的中心误差覆盖进训练分布，所以采样时也保留（0.25 Å）而不是关掉。

诊断用 **`centroid_err`**（`flow.centroid_error`）：训练指标和 `val/centroid_err_median` 都有。它是唯一能暴露先验泄漏的数 —— 如果重对接上它趋近 0 而交叉对接上明显变大，说明模型在从先验里读答案，而总 RMSD 会把这件事藏起来。

## 使用

```bash
pip install -e ".[test]"        # 在 pocketdit/ 目录下
```

**1. 预处理**：每个复合物只用 RDKit 解析一次，存成 `.pt` 文件。目录结构与 PDBBind 相同，每个复合物一个子文件夹：

```bash
python -m pocketdit.data --root data/pdbbind/refined-set --out cache/train \
    --protein_glob "*_protein.pdb" --ligand_glob "*_ligand.sdf" --pocket_cutoff 12
python -m pocketdit.data --root data/pdbbind/core-set --out cache/val
```

`--pocket_cutoff` 是**缓存**的最宽壳，不是训练用的口袋 —— 预算裁剪在加载时做，所以改预算、改 cutoff 抖动、改 `dir_noise` 都不需要重跑预处理，只要缓存半径够（`cutoff + cutoff_jitter + dir_noise`，默认 12 ≥ 6.5+1.5+4）。`--com_cutoff` 决定先验中心取哪一层壳的质心（默认 6 Å）。

⚠️ 用完整蛋白（`*_protein.pdb`），**不要**用 PDBBind 自带的 `*_pocket.pdb` —— 那本身就是"配体 10 Å 以内"的裁剪，形状泄漏在文件层面就固化了，后面怎么裁、怎么加噪声都救不回来。

也可以用 `--csv index.csv`，列为 `protein,ligand[,ref]`，路径相对于 CSV 所在目录。交叉对接时用 `--ref_glob` 指定参考配体，由它来定义口袋。解析失败的复合物会记录在 `<out>/failures.txt`。

**2. 训练**：`TrainConfig`、`ModelConfig`、`FlowConfig`、`PocketCrop` 的每个字段都是命令行参数。

```bash
python -m pocketdit.train --train_dir cache/train --val_dir cache/val --out_dir runs/exp1 \
    --batch_size 16 --lr 3e-4 --max_steps 200000 --model.dim 256 --model.depth 6 \
    --crop.max_atoms 256 --crop.core_cutoff 4 --crop.cutoff 6.5 \
    --crop.dir_noise 4 --crop.cutoff_jitter 1.5 \
    --flow.prior_std 4 --flow.center_noise 1.0
```

日志写到 `runs/exp1/log.jsonl`。验证时用 EMA 模型报告 loss、采样 RMSD 中位数、RMSD < 2 Å 的比例和质心误差中位数。`--resume runs/exp1/last.pt` 可以断点续训。

**3. 对接**：

```bash
# 重对接（口袋由配体自身坐标定义）
python -m pocketdit.dock --ckpt runs/exp1/last.pt --protein p.pdb --ligand l.sdf --num_samples 10
# 交叉对接 / 实际使用：用参考配体或口袋中心定义口袋
python -m pocketdit.dock --ckpt runs/exp1/last.pt --protein p.pdb --ligand query.sdf --ref ref.sdf
python -m pocketdit.dock --ckpt runs/exp1/last.pt --protein p.pdb --ligand query.sdf --center 12.3 4.5 -7.8
```

口袋预算和先验设置都存在 checkpoint 里，推理会自动复现训练时的口袋，不需要重复传参（`--max_pocket_atoms` 可以覆盖）。两种定义口袋的方式里**参考配体更可靠**：`--center` 只给一个点，口袋只能是绕它的一个球（`--center_margin` 默认 6 Å，约一个配体半径），而不是贴着一个合理配体的壳。

每个复合物输出 `docked/<name>.sdf`，包含 `--num_samples` 个位姿，**未排序**。属性 `rmsd_to_input` 是相对输入坐标的对称性校正 RMSD，只在重对接时有意义。

## 测试

```bash
pytest -q        # CPU 约 45 秒，使用 ../notebooks/dummy_data 中的 10 个 Astex 复合物
```

覆盖内容：PDB 解析和口袋提取；口袋预算（整残基、按距离排序、上限语义、核心超预算的退化、cutoff 抖动不拆残基、验证时确定性）；方向性位移（重选外壳但绝不丢接触核心、把口袋质心推离配体质心、推理时关闭、缓存半径不足时报错）；先验中心取的是口袋质心而非配体质心；手性特征的符号；随机权重下的 SE(3) 等变性；镜像下输出确实不同；padding 不变性；用"总是给出真值的 oracle"检验采样器的积分；小模型从噪声过拟合单个复合物（中位 RMSD < 1.5 Å）；预处理 → 训练 → 对接的端到端命令行，含 `--center` 路径。

## 已知局限和下一步

- **还没有在真实数据集上训练过**，没有 benchmark 数字。默认超参数（先验 4 Å、预算 256、cutoff 抖动 1.5 Å、距离损失权重等）都是合理的起点，没有经过调优。`crop.shell_dropout` 默认关着：cutoff 抖动已经在动边界了，再叠一层整残基 dropout 是否有用需要实测。
- **没有置信度或打分模型**，多个采样位姿无法排序。可以加一个 confidence head，或者接 GNINA / Vina 重打分。
- 坐标 MSE 没有处理**对称等价原子**（例如羧基的两个 O、对称苯环），评估时请用对称性校正的 RMSD（`dock.py` 中已用 RDKit `CalcRMS`）。
- **形状泄漏还剩两项**：4 Å 核心的轮廓（故意保留，见上）和壳的拉伸方向。外壳的细轮廓已经被 `dir_noise` 打散，中心泄漏靠 `pocket_com` 堵住。还剩多少值得用 `centroid_err` 的重对接/交叉对接差值去量。
- **还不是"盒子无关"的通用对接程序。** 现在的输入模式是"参考配体定口袋"（SigmaDock 式的 `--autobox`），`dir_noise` 把这个模式的训推分布对上了。要支持"给一个中心 + 半径的盒子"还缺三样：训练时随机化 (中心偏移, R) 并把 R 作为标量 conditioning 喂进 `c`；先验拆成分子级共享平移 + 局部抖动（现在 `x_0` 的质心只在 1.26 Å 内变动，`σ/√n` 压着，所以多个样本的落点几乎重合、根本不探索区域）；以及跨中心排序所需的 confidence head。
- 先验是逐原子独立的高斯，原子间距期望约 9.8 Å，比真实分子散得多；harmonic prior（按键图的拉普拉斯构造先验）应该更贴。
- 其他可尝试的扩展：self-conditioning、非均匀的 t 采样和时间步长，以及 SigmaDock 式的刚性片段参数化（输出头不变，把逐原子向量聚合成片段的力和力矩）。
