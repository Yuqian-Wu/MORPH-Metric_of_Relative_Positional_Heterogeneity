# 在当前 G-TAF 框架中引入贝叶斯方法的可行性分析

**文档版本**：v1.6
**最后更新**：2026-03-25
**所属子课题**：MORPH（Sub-project 1）
**负责人**：JerryWu

---

## 版本记录

| 版本 | 日期 | 更新内容 |
|---|---|---|
| v1.0 | 2026-03-13 | 初稿：原有四个贝叶斯思路（方案A~D）+ 综合推荐路线 |
| v1.1 | 2026-03-14 | 新增：① 几何增强型 TEI（基于 Table 3.2 特征）② 宏观层级 TEI 分析 ③ 评估框架更新建议 |
| v1.2 | 2026-03-15 | 方案A+B 正式实施：修改 3.2.4 notebook（Cell 6/10/14/18/22），B-GNN 名实相符 |
| v1.3 | 2026-03-23 | 4.2 评估实验状态更新：新增 CI 宽度时序（3.3.1）、GM-TEI 时序（3.3.2）、CI 按情境（3.3.2）、GM-TEI 与变化点关联（3.3.3）四项 ✅ 已实施 |
| v1.4 | 2026-03-24 | 4.2 更新：事件研究（TEI vs 进球/换人）标记为 ✅ 已对接；_load_events() 动态加载 Event Data/10517.json，TEI/JSD 时序图新增黄牌+换人标注 |
| v1.5 | 2026-03-24 | 4.2 更新：① 置信区间与战术事件相关性 ✅ 已实施（3.3.3 Cell 7，Mann-Whitney U 检验变化点附近 CI 宽度 vs 稳定期）；② 事件研究窗口扩展至 ±60s（3.3.2 Cell 9，原 ±30s） |
| v1.6 | 2026-03-25 | 4.2 更新：③ TEI 置信带时序图 ✅ 已实施（3.3.2 Cell 7.6，窗口级 TEI±CI 阴影）；④ MC Dropout 方差时序 ✅ 已实施（3.3.2 Cell 7.7，帧级 epistemic 曲线 + Spearman 检验） |

---

## 0. 前置说明：当前输出的性质

当前 B1 方法（聚类原型 + 余弦相似度 + 温度缩放 Softmax）的输出是**离散概率分布**，但严格来说是"软指派分布"（soft assignment），不是贝叶斯后验。

- 它的概率值反映的是"与各阵型原型的相似度"
- 而非"给定观测数据后各阵型的后验概率 P(formation | data)"

这两者的区别很微妙但在学术上很重要。引入贝叶斯方法的核心动机，是将当前的"软指派分布"升级为"有理论保证的概率后验分布"，并为每个预测提供不确定性量化（Uncertainty Quantification, UQ）。

**B-GNN 命名问题**：~~当前代码中的"B-GNN"名称是历史遗留，实际并无贝叶斯推断。若需学术严谨性，应改名为"Prototype-GNN"或在引入贝叶斯方法后名实相符。~~
**✅ 2026-03-15 已解决**：方案A（MC Dropout）+ 方案B（Dirichlet聚合）已实施，B-GNN 现名实相符，可正式在论文中声称"Bayesian Graph Neural Network"。

---

## 一、原有四个贝叶斯方案（A ~ D）

### 总览

| 方案 | 名称 | 核心操作 | 工作量 | 学术价值 | 状态 |
|---|---|---|---|---|---|
| **A** | MC Dropout | 推断时保持 Dropout 激活，N 次采样取均值/方差 | 低（~30行代码）| ★★★ | ✅ 已实施 |
| **B** | Dirichlet 窗口聚合 | 以帧级频数为充分统计量更新 Dirichlet 先验 | 低（~20行代码）| ★★★★ | ✅ 已实施 |
| **C** | BOCPD 变化点检测 | 贝叶斯在线变化点检测，输出 P(changepoint at t) | 中 | ★★★★★ | 待定 |
| **D** | 变分嵌入 | 嵌入空间建立概率分布，VAE 风格 | 极高（需重训练）| ★★★★★ | 不推荐 |

---

### 方案 A：MC Dropout（推断期不确定性）

**原理**（Gal & Ghahramani 2016）：
训练时使用 Dropout（已有），推断时**不关闭** Dropout，对同一输入运行 N 次前向传播，得到 N 个不同的软概率输出，再统计均值和方差。

```
对每帧 t，运行 N=50 次前向传播（Dropout 保持激活）：
  p_n(t) = softmax( cosine_sim( z_t^(n), mu_k ) / tau )   for n=1,...,N

均值：  p_mean(t) = (1/N) * sum( p_n(t) )
方差：  p_var(t)  = (1/N) * sum( (p_n(t) - p_mean(t))^2 )
```

**关键指标**：
- `p_mean` 作为最终概率分布（替换当前确定性 softmax 输出）
- `p_var` 最大分量的方差作为"认知不确定性"（epistemic uncertainty）
- 当 `p_var` 高时，说明嵌入空间中当前帧位于决策边界附近，模型真正"不确定"

**优势**：
- 无需重新训练，推断阶段修改即可
- 输出的 p_mean 是近似贝叶斯后验（有理论保证）
- 在论文中可正式声称"MC Dropout 近似变分推断"

**局限**：
- Dropout 概率 p 和采样次数 N 是超参数，需要交叉验证
- 计算量增加 N 倍

**✅ 实施状态（2026-03-15）**：已在 `3.2.4_test_TwoStage_B1_ClusterPrototype.ipynb` 中实施。
- BGNN 新增 `embed_mc()` 方法（MCDropout 推断期恒激活）
- Cell 14：N_MC=50 次采样 → `frame_probs`(N,K) 均值 + `frame_probs_var`(N,K) 认知不确定性 + `frame_epistemic`(N,) 帧级最大方差
- 输出：`b1_frame_epistemic.npy`

---

### 方案 B：Dirichlet-Multinomial 窗口聚合（推荐主力方案）

**原理**：
Dirichlet 分布是多项分布的共轭先验，天然适合对"K 类阵型的概率分布"进行贝叶斯更新。

```
先验：  alpha_0 = [1, 1, ..., 1]   (K=43 维均匀先验)

对窗口 W 内的 T 帧，统计各阵型 top-1 出现频数：
  n_k = count(top-1 prediction == formation_k)   for k=1,...,K

后验（Dirichlet）：  alpha_k = alpha_0_k + n_k

后验均值（最终概率）：  p_k = alpha_k / sum(alpha)

后验方差（不确定性）：  Var(p_k) = (alpha_k * (sum(alpha) - alpha_k)) / (sum(alpha)^2 * (sum(alpha) + 1))
```

**优势**：
- 窗口聚合从"频率统计"升级为"贝叶斯后验"，有完整概率论基础
- 先验 alpha_0 可以用历史比赛的阵型先验频率替代均匀先验（更合理）
- 输出的后验方差直接量化窗口内的预测不确定性
- 与当前 stability_weight 机制完全兼容（可用稳定性作为 alpha 权重）

**实现位置**：3.2.4 的窗口聚合函数

**✅ 实施状态（2026-03-15）**：已在 `3.2.4_test_TwoStage_B1_ClusterPrototype.ipynb` Cell 18 中实施。
- 以稳定性加权 top-1 频数（`n_k.scatter_add_(0, top1_win, s_win)`）更新均匀先验 Dir(1,...,1)
- 输出 `P_win`（后验均值，替代原稳定性加权均值）+ `P_win_var`（Dirichlet后验方差）
- parquet 新增 `probvar_*` 列（每阵型后验方差）

---

### 方案 C：BOCPD 变化点检测（替换当前 PELT 算法）

**原理**（Adams & MacKay 2007）：
贝叶斯在线变化点检测（Bayesian Online Changepoint Detection），在每个时刻 t 输出 `P(changepoint at t | data_1:t)`，而非二元标签。

```
当前方案（PELT，3.3.3）：  binary_label(t) in {0, 1}
BOCPD 升级：             P_cp(t) = P(changepoint at t | data)   in [0, 1]
```

**优势**：
- 用概率替代二元判断，与全框架的"概率化"目标一致
- 可与 TEI 时序图叠加，提供"战术转变概率曲线"
- 比 PELT 更稳健（PELT 对噪声敏感）

**实现**：引入 `bayesian-changepoint-detection` 或 `ruptures` 库（后者有 BOCPD 接口）

---

### 方案 D：变分嵌入（暂不推荐，备选长远目标）

**原理**：将 B-GNN 的编码器改为变分自编码器（VAE）风格，输出嵌入分布参数 `(mu_z, sigma_z)` 而非确定性向量 `z_t`，通过重参数化技巧（reparameterization trick）对嵌入空间进行采样，使嵌入本身具有概率分布属性。

```
现有架构（v7.0）：
  GATConv → z_t（128维确定性向量）→ cosine_sim(z_t, μ_k) / τ → softmax → p_t

方案 D 目标架构：
  GATConv → (μ_z, σ_z)（各128维）→ z_t ~ N(μ_z, σ_z²)（采样）
         → cosine_sim(z_t, μ_k) / τ → softmax → p_t（每次采样结果不同）
```

**与当前架构的冲突点（v7.0 具体影响）**：

| 影响层 | 当前状态 | 方案 D 所需修改 |
|---|---|---|
| B-GNN 编码器输出 | 确定性 `z_t`（128维） | 改为 `(μ_z, σ_z)`，输出维度翻倍 |
| 损失函数 | 交叉熵（65类分类） | 改为 ELBO = 重建损失 + KL(q(z\|x) \|\| p(z)) |
| 原型 μ_k 计算（B1） | z_t 分组均值 | 需改为分布均值，或采样后计算 |
| `best_model.pth` | Acc 91%/F1 87.4% | **完全作废，需从零重训** |
| ELBO 优化稳定性 | — | 对学习率极敏感，KL 权重需精细 β 退火调参 |

**工作量评估**：
- 编码器架构重设计：~2天
- ELBO 损失函数实现 + KL 退火调参：~3天
- 重训练 + 超参搜索：耗时数倍于现有训练（收敛更难）
- 原型计算逻辑重写：~1天
- **总计：远超方案 A+B 的 ~50 行代码，且存在训练失败风险**

**预期效果提升（不确定）**：
- 收益：嵌入空间本身具有不确定性，`σ_z` 可作为帧级认知不确定性指标
- 风险：VAE 在判别任务上的分类精度历史上经常不如判别模型，Acc 可能低于当前 91%
- 对 TEI 的增益：间接——通过 p_t 的随机性引入额外熵，但这与方案 A（MC Dropout）的效果高度重叠

**结论**：方案 D 的核心收益（嵌入不确定性量化）与方案 A（MC Dropout 近似变分推断）高度重叠，而代价是推倒现有已验证的 Acc 91% 成果从头重来。对硕士课题而言，投入产出比极差。**在方案 A+B 组合已能满足"B-GNN"学术定义的前提下，方案 D 无实施必要。**

---

## 二、新想法 ① —— 几何增强型 TEI（Geometry-Augmented TEI）

**文档版本**：v1.1（基于 Hadi 博士论文 Table 3.2 Shape Graph 几何特征更新）

---

### 2.0 原始问题

Shannon 熵 `H(t) = -sum(p_k * log(p_k))` 只反映概率分布的"宽窄"，不区分战术含义：

- **场景A**：4-5-1 → 3-5-2 → 2-3-5（进攻型演化，重心持续前移）
- **场景B**：4-5-1 → 5-3-2 → 4-3-3（防守型收缩，重心持续后撤）

这两条序列的熵值序列可能完全相同，但战术含义截然相反。Shannon 熵对这两条路径"失明"，需要几何特征提供方向感。

---

### 2.1 可用几何特征分类（来源：Hadi 博士论文 Table 3.2）

**【E类：基础尺度与定位特征】**

| 特征名 | 计算方式 | 战术含义 |
|---|---|---|
| Team centroid | 所有球员位置坐标的均值 | 全队重心坐标，是其他位置类特征的基准参考点 |
| Width | 持有最大/最小横向坐标的两名球员之间的横向距离 | 阵型横向展开宽度 |
| Length | 持有最大/最小纵向坐标的两名球员之间的纵向距离 | 阵型纵向深度 |

**【A类：扩散程度——衡量队形松紧】**

| 特征名 | 计算方式 | 战术含义 |
|---|---|---|
| Stretch | 球员到重心距离均值 | 球员离重心的平均扩张度 |
| Centroid size | 球员到重心距离平方和的平方根 | 全队围绕重心的整体扩散量（比 Stretch 对离群球员更敏感） |
| Spread | 位置矩阵的 Frobenius 范数 | 全队分布的整体离散度 |
| Dispersion | 球员间两两距离矩阵均值 | 平均球员间距 |
| Surface area | 凸包面积 | 全队占据空间大小 |
| Compactness | 最小外接矩形面积 | 战术紧凑性 |

**【B类：攻守倾向——衡量阵型重心位置】**

| 特征名 | 计算方式 | 战术含义 |
|---|---|---|
| Defensive line height | 最后防线球员到己方球门线的纵向距离 | 防线高度（低=防守收缩，高=高位逼抢） |
| Highest player's location | 最前球员到对方球门线的纵向距离 | 进攻前压深度 |
| Centroid goal distance | 全队重心到己方球门中心的距离 | 全队整体重心位置 |
| Team-ball distance | 离球最近球员到球的欧氏距离 | 反映全队对球的即时压迫距离 |

**【C类：形状结构——衡量阵型"几何形状"】**

| 特征名 | 计算方式 | 战术含义 |
|---|---|---|
| Length per Width (LpW) | 纵向距离 / 横向距离 | >1 = 阵型偏长（压迫型），<1 = 阵型偏宽（防守型） |
| Elongation | 1 − Width / Length | 越接近 1 = 阵型越"瘦长"，与 LpW 互补 |
| Layer Ratio (LR) | 内凸包面积 / 外凸包面积 | 反映阵型的"层次感"（中场球员是否填充内部） |
| Rectangularity | 凸包面积 / 最小外接矩形面积 | 接近 1 = 队形规整，接近 0 = 队形散乱 |
| Circularity | 周长² / 凸包面积 | 反映凸包是否紧凑 |
| Area/Perimeter ratio | 凸包面积 / 凸包周长 | Circularity 的互补指标，值越大 = 形状越紧凑 |
| Convex hull centroid | 凸包上所有球员位置的均值 | 反映"边界球员"的平均位置，与全队重心对比可知内部密度 |
| Centroid-convex distance | 全队重心与凸包重心之间的欧氏距离 | 距离越大 = 内部球员分布越不均匀，队形结构越不对称 |

**【D类：Delaunay 三角网格特征——衡量局部结构均匀性与复杂度】**

| 特征名 | 计算方式 | 战术含义 |
|---|---|---|
| Triangles | DT 三角形数量 | 间接反映边界球员的凸包复杂程度 |
| Triangle Area | DT 三角形平均面积 | 面积越大 = 球员间距越大 = 阵型越松 |
| Edge length | DT 边长均值 | 反映局部球员间距 |
| Angle | DT 三角形最小角均值 | 角度越小 = 三角结构越"扁" = 局部结构不均匀 |
| Box-counting metrics | 使用最大分形值和分形面积量化凸包复杂度 | 分形值越高 = 队形轮廓越不规则 = 战术阵型越混乱 |

---

### 2.2 两个 GM-TEI 具体构造方案

> **设计原则**：两个方案均以 `TacDir(t)` 作为符号门控，赋予熵值"进攻性（正）/ 防守性（负）"方向感，解决 Shannon 熵对战术演化方向失明的核心问题。

**共同组件：TacDir(t)**

```
d_DLH(t) = Defensive_line_height(t) - Defensive_line_height(t - delta)
d_HPL(t) = Highest_player_location(t)  - Highest_player_location(t - delta)

TacDir(t) = sign(d_DLH(t) + d_HPL(t))
           = +1  (进攻型演化：防线上提 + 前锋前压)
           = -1  (防守型演化：防线后撤 + 前锋回撤)
           =  0  (中性，实践中做平滑处理，见注意事项)
```

> ⚠️ TacDir 是二值符号函数，翻转时会产生阶跃。建议对 `d_DLH + d_HPL` 做 5 帧滑动平均后再取符号，以减少短暂震荡。

---

**方案 GM-TEI_AB：扩散调制 × 方向型（轻量主方案）**

```
GM-TEI_AB(t) = H(t) × (1 + β × Spread(t) / Spread_max) × TacDir(t)
```

- `β`：超参，建议 0.5（固定即可，无需学习）
- `Spread_max`：全场最大 Spread，用于归一化
- **输出范围**：`[-(1+β)·H, +(1+β)·H]`

四象限语义：

| | TacDir = +1（进攻型） | TacDir = -1（防守型） |
|---|---|---|
| Spread 大（队形松散） | 大正值：**进攻混乱** | 大负值：**防守混乱** |
| Spread 小（队形紧凑） | 小正值：**进攻有序** | 小负值：**防守有序** |

**特征来源**：
- `Spread`：`global_features[:, 15]`（已在 3.2.2 计算 ✅）
- `DLH / HPL`（用于 TacDir）：`global_features[:, 20/21]`（已在 3.2.2 计算 ✅）
- **无需修改上游任何 notebook**

---

**方案 GM-TEI_CB：多维几何加权 × 方向型（增强主方案）**

```
g'(t) = [Spread(t), LpW(t), convex_hull_area(t), compactness(t), LR(t), Rectangularity(t)]
        （各维归一化到 [0, 1]，6维）

GM-TEI_CB(t) = H(t) × (1 + w'ᵀ · g'(t)) × TacDir(t)

w' = [w1, w2, w3, w4, w5, w6]   （权重向量，待学习）
```

> ⚠️ **设计原则**：g'(t) 中**不含** DLH / HPL，因为这两个特征已通过 TacDir(t) 引入，避免重复计数。

各维度含义：

| 维度 | 特征 | global_features 索引 | 含义 | 是否已有 |
|---|---|---|---|---|
| w1 | Spread | 15 | 队形扩散程度 | ✅ 已有 |
| w2 | LpW | 17 | 阵型纵横比（压迫 vs 防守） | ✅ 已有 |
| w3 | convex_hull_area | 18 | 全队占据空间大小 | ✅ 已有 |
| w4 | compactness | 19 | 最小外接矩形面积（紧凑性） | ✅ 已有 |
| w5 | LR（Layer Ratio） | 需补充 | 内/外凸包面积比，中场充实度 | ⚠️ 需改 3.2.2 |
| w6 | Rectangularity | 需补充 | 凸包面积 / 最小外接矩形面积 | ⚠️ 需改 3.2.2 |

**特征来源**：前 4 维零成本直接可用；w5/w6 需在 3.2.2 的 `compute_advanced_geometric_features` 中补充约 15 行代码（均基于已有凸包计算）。

---

### 2.3 场景A vs 场景B 的区分能力对比

| 指标 | 场景A（4-5-1→2-3-5，进攻演化） | 场景B（4-5-1→5-3-2，防守演化） |
|---|---|---|
| Shannon 熵 H(t) | 中-高 | 中-高（**无法区分**） |
| Defensive line height 变化 | 上升（防线前压） | 下降（防线后撤） |
| Highest player's location 变化 | 上升（前锋前压） | 下降（前锋回撤） |
| LpW 变化 | 增大（阵型纵向拉伸） | 减小（阵型变扁） |
| LR（层次比）变化 | 减小（前锋拉空中场） | 增大（球员聚集防守） |
| **TacDir(t) 符号** | **+1（进攻型）** | **-1（防守型）** |
| **GM-TEI_AB 值** | **正值，大小由 Spread 调制** | **负值，大小由 Spread 调制** |
| **GM-TEI_CB 值** | **正值，大小由 6 维几何特征加权** | **负值，大小由 6 维几何特征加权** |

---

### 2.4 贝叶斯化路径（可选，针对 GM-TEI_CB）

对 GM-TEI_CB 中的权重向量 w' 建立先验：

```
w'_i ~ Normal(0, sigma_i^2)    for i = 1,...,6
sigma_i ~ Half-Normal(1)
```

训练数据：标注了关键战术事件（进球、换人、战术转换时刻）的比赛片段。
目标：最大化"GM-TEI_CB 在已知战术转换事件周围 ±10 帧内的响应绝对值"。
工具：PyMC 或 NumPyro。

论文陈述示例：
> 权重向量 w' 通过贝叶斯推断从有标注的战术事件数据中学习，使得 GM-TEI_CB 的峰值与已知战术转变时刻高度对齐，并给出权重的后验置信区间，量化各几何特征对战术不确定性的贡献度。

---

### 2.5 实现说明（特征索引与上游依赖）

**✅ 已在 global_features 中的特征（global_features 索引对照）**：

| 特征 | global_features 索引 | 实测值（Frame 4630） | 是否需要改 3.2.2 |
|---|---|---|---|
| macro_phase (one-hot) | 0–1 | [0, 1] | — |
| fine_intent (one-hot) | 2–6 | [0,0,0,0,1] | — |
| score_diff / time_elapsed | 7–8 | [0, 0] | — |
| fifa_ranking_diff | 9 | -1.0 | — |
| odds (home/draw/away) | 10–12 | [2.5, 3.0, 2.8] | — |
| centroid (x, y) | 13–14 | [-8.29, -0.58] | — |
| **team_spread** | **15** | 15.18 | **✅ 已有** |
| team_diameter | 16 | 49.07 | — |
| **LpW** | **17** | 0.534 | **✅ 已有** |
| **convex_hull_area** | **18** | 847.4 | **✅ 已有** |
| **compactness** | **19** | 1213.8 | **✅ 已有** |
| **DLH**（TacDir用） | **20** | 20.48 | **✅ 已有** |
| **HPL**（TacDir用） | **21** | 4.99 | **✅ 已有** |

**⚠️ 需在 3.2.2 补充计算的特征**：

| 特征 | 计算方式 | 改动位置 | 估算代码量 |
|---|---|---|---|
| LR（Layer Ratio） | 内凸包面积 / 外凸包面积；内凸包 = 去掉外层球员后再做凸包 | `compute_advanced_geometric_features()` | ~10行 |
| Rectangularity | 凸包面积 / 最小外接矩形面积；最小外接矩形可用 `cv2.minAreaRect` 或旋转卡壳算法 | 同上 | ~5行 |

补充后 global_features 维度：22 → **24维**，需同步更新 3.2.3（BGNN 模型的 `gd` 参数）和 3.2.4（`GLOBAL_DIM` 常量）。

---

## 三、新想法 ② —— 宏观层级 TEI 聚合分析

### 3.1 问题动机

当前 TEI 是逐帧（逐窗口）的微观指标，缺少"整场比赛尺度"或"战术阶段尺度"的宏观分析。

---

### 3.2 多层级 TEI 聚合体系

**层级 1：帧级 TEI（现有，基础）**

```
H(t) = -sum( p_k(t) * log(p_k(t)) )    for k = 1,...,K
```

---

**层级 2：阶段级 TEI（Phase-level TEI）**

对每个宏观阶段（Build-up / High-block / Transition / Dead-ball 等）内所有帧的 TEI 求稳定性加权均值：

```
H_phase = sum( stability(t) * H(t) ) / sum( stability(t) )    for all t in phase
```

- 权重 `stability(t)` 来自现有帧级稳定性分数，过渡期帧权重更低
- 输出：一张"战术阶段不确定性热力图"（x 轴：比赛时间；颜色：Phase-TEI 大小）

---

**层级 3：比赛级 TEI（Match-level TEI）**

```
H_match_mean = mean( H(t) )    for all t in match
H_match_std  = std( H(t) )
```

跨比赛比较意义：H_match_mean 低且 H_match_std 低 → 战术纪律性强（如曼城）；H_match_mean 高 → 战术混乱或频繁切换（如弱队）。

---

### 3.3 配套分析模块

**① 阵型偏好矩阵（Formation Preference by Phase）**

对每个战术阶段，统计各阵型出现频率，输出：

| 阶段 | 最常用阵型 | 第二常用 | Phase-TEI |
|---|---|---|---|
| Build-up | 4-3-3（67%） | 4-5-1（18%） | 低 |
| High-block | 4-4-2（43%） | 4-2-4（25%） | 中 |

**② TEI 时序注释图（TEI Annotated Timeline）**

在 TEI 时序曲线上叠加：
- 宏观阶段分段（颜色底纹）
- 关键事件（进球、换人、红牌）
- TEI 峰值自动标记（用于事件研究法验证）

**③ 宏观变化点阈值标记**

```
delta_H(t) = |H_window(t+1) - H_window(t)| > tau
```

tau 为阈值（可用贝叶斯优化学习），满足条件的 t 标记为"战术转变点"。可与 BOCPD 输出叠加对比。

---

### 3.4 贝叶斯化路径（分层贝叶斯，契合导师方向）

跨比赛、跨球队分析时，构建分层贝叶斯模型：

```
H_phase,i ~ Normal( mu_phase, sigma_phase^2 )
mu_phase   ~ Normal(0, 1)
sigma_phase ~ Half-Normal(1)
```

可回答的研究问题：
- 不同球队在同一战术阶段的 Phase-TEI 分布是否显著不同？
- 同一球队在不同战术阶段的不确定性水平差异是否具有统计显著性？

---

## 四、新想法 ③ —— 评估方法：JSD、时间相干性、事件研究法

### 4.1 核心概念解释

**JSD（Jensen-Shannon Divergence，Jensen-Shannon 散度）**

JSD 是对称的概率分布距离度量，值域 [0, 1]（以 log2 为底）：

```
M = (P + Q) / 2
JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
```

其中 KL 是 KL 散度。JSD 是 KL 的对称版本，避免了 KL 不对称的缺陷。

在本课题中：P 和 Q 是相邻两个窗口的阵型概率分布。JSD 越高 = 分布差异越大（战术突变）；JSD 越低 = 分布平滑演化（战术稳定）。

---

**时间相干性（Temporal Coherence）**

逐帧输出随时间的平滑性，用相邻帧的 JSD 均值衡量：

```
TC = (1 / (T-1)) * sum( JSD(p(t), p(t+1)) )    for t = 1,...,T-1
```

- EFPI（模板匹配）：输出离散标签，切换时 JSD = 最大，不切换时 JSD = 0，是粗糙度量
- B1（概率分布）：即使 top-1 不变，分布也可以平滑演化，TC 是连续小值

这是 B1 优于 EFPI 的核心优势之一。

---

**事件研究法（Event Study Methodology）**

以某事件（如进球）发生时刻为 t=0，计算事件前后时间窗口（如前30秒到后30秒）内的**平均 TEI 曲线**：

```
TEI_event_avg(tau) = mean( H(t0 + tau) )    across all event instances
   where tau in [-30s, +30s]
```

若 TEI 在 t=0 附近出现显著峰值，验证了 TEI 对"混乱/过渡"状态的捕捉能力。

---

### 4.2 当前评估实验实施状态

| 评估实验 | 当前实现情况 | 结果 | 结论 |
|---|---|---|---|
| 概率分布平滑性（JSD） | 已实现 | 99.71% 帧内 JSD 均值 = 0.0039 | 优秀，显著优于 EFPI |
| Top-1 标签切换率对比 | 已实现 | B1 = 553 次，EFPI = 168 次 | 已统计（B1 更精细，EFPI 更粗粒度）|
| TEI 与切换事件相关性 | 已实现 | 切换帧 TEI 是非切换帧 1.24 倍，p = 1.10e-118 | 高度显著，TEI 有效 |
| **置信区间可视化（条形图误差棒）** | **✅ 已实施（3.2.5）** | Dirichlet 后验 95% CI，误差棒叠加在概率条形图上 | 直观呈现 B-GNN 贝叶斯特性 |
| **CI 宽度时序平滑性（维度C）** | **✅ 已实施（3.3.1 Cell 9）** | Dirichlet 95% CI 半宽时序图 + CI vs TEI 散点，输出 `eval_ci_smoothness_10517.png` | 验证 Dirichlet 不确定性输出的时序稳定性 |
| **GM-TEI 时序分析** | **✅ 已实施（3.3.2 Cell 13-14）** | GM-TEI_AB / GM-TEI_CB 按 Period 绘制，TacDir 着色，输出 `gm_tei_timeseries_10517.png` | 可视化几何增强型战术熵的方向性信息 |
| **CI 宽度按战术情境** | **✅ 已实施（3.3.2 Cell 17-18）** | Dirichlet 95% CI 半宽按 macro_phase / fine_intent 分组箱线图，输出 `ci_by_context_10517.png` | 验证高不确定性情境 CI 更宽（贝叶斯输出自洽） |
| **GM-TEI 与变化点关联** | **✅ 已实施（3.3.3 Cell 12-13）** | 变化点 vs 非变化点 GM-TEI/Spread/LR/Rect 箱线图 + TacDir饼图 + Mann-Whitney U，输出 `gm_tei_changepoints_10517.png` | 验证 GM-TEI 在战术转变时刻的响应能力 |
| **TEI 置信带时序图** | **✅ 已实施（3.3.2 Cell 7.6）** | 窗口级 TEI±Dirichlet 95% CI 阴影 + CI 宽度子图，CI 窄区段 TEI 更可信，输出 `tei_confidence_band_10517.png` | 量化 TEI 时序的可靠性 |
| **MC Dropout 方差热力图/时序** | **✅ 已实施（3.3.2 Cell 7.7）** | 帧级 epistemic 曲线（紫）vs TEI（红虚线）双轴，高不确定性区域（mean+2σ）紫色阴影并标注 Top-5 时刻，Spearman 相关检验，输出 `mc_dropout_epistemic_10517.png` | 展示帧级模型认知不确定性 |
| 置信区间与战术事件相关性 | **✅ 已实施（3.3.3 Cell 7）** | Mann-Whitney U 检验变化点附近（±3窗口）CI 宽度 vs 稳定期，输出 `ci_vs_changepoints_10517.png` | 验证贝叶斯不确定性在战术转变时刻的响应能力 |
| 下游任务预测增益（ΔEPV） | ❌ 未实施 | — | 待子课题二实施 |
| 事件研究（TEI 与进球/换人）| **✅ 已实施（3.3.2 Cell 9，±60s）** | 动态加载 `Event Data/10517.json`，以进球/黄牌/换人为 t=0，绘制前后 ±60s 平均 TEI + GM-TEI_AB + GM-TEI_CB 曲线（含 IQR 阴影） | 验证 TEI 对关键事件前后战术混乱度的捕捉能力 |

---

### 4.3 科研方案评估部分更新建议

与科研方案 3.4/3.5（Sub-step 3.3）的对应关系：

| 科研方案 3.4/3.5 描述 | 实际实现情况 | 操作建议 |
|---|---|---|
| "B-GNN 两阶段架构" | 已改为 B1 聚类原型方法 | 更新方法名称 |
| "MC Dropout 推断" | 未实现，用确定性 Softmax | 删除或标注为"可选未来工作" |
| "维度A：JSD 概率平滑性" | 已实现，结果数值已有 | 方案正确，补充实验数值 |
| "维度B：Top-1 切换率" | 已实现，EFPI=168 次 | 方案正确，补充实验数值 |
| "TEI 与切换事件相关性" | 已实现（Mann-Whitney U 检验，p 极显著）| 3.4 版本没有，**需要添加** |
| "维度二：ΔEPV 预测增益" | 未实施 | 保留计划，注明"待子课题二实施" |
| "维度三：不确定性与事件相关性" | 未实施 | 保留计划，是下一步工作 |

---

## 五、综合推荐路线图

### 按性价比排序（工作量 vs. 学术价值 vs. 导师满意度）

**第一优先级（强烈推荐，必做）**

> 方案B（Dirichlet 窗口聚合）+ 想法①（GM-TEI：Defensive line height + Spread）

- Dirichlet 聚合：~20 行代码，升级窗口聚合为真贝叶斯后验
- GM-TEI_B：~10 行代码，使用已有几何特征，区分进攻/防守型不确定性
- 论文声称："基于 Dirichlet-Multinomial 贝叶斯框架 + 几何校准的战术熵指数（GM-TEI）"

**第二优先级（推荐，中等工作量）**

> 想法②（宏观 TEI 分析：Phase-TEI + 阵型偏好矩阵）+ 想法③（评估框架完善）

- Phase-level TEI 聚合 + TEI 注释时序图：约 1-2 天
- 科研方案更新：将已实现 JSD/TEI 实验数值填入 3.3 节

**第三优先级（可选，导师偏好贝叶斯则做）**

> 方案A（MC Dropout）+ 方案C（BOCPD 替换 PELT）

- MC Dropout：~30 行代码修改
- BOCPD：引入 `bayesian-changepoint-detection` 库，约 1 天

**第四优先级（论文有余力再做）**

> 想法②扩展：分层贝叶斯模型（跨比赛比较球队 Phase-TEI）

- 需要多场比赛数据，工作量较大，但完全契合导师专长

---

## 六、一句话结论

> 当前框架已实现核心概率性识别目标。最低成本贝叶斯化路径：**将 Dirichlet 先验加入窗口聚合**（让输出成为真贝叶斯后验，~20行代码）+ **用 Defensive line height 调制 TEI**（提出 GM-TEI，~10行代码）。两者合计约 2-3 天工作量，但可在论文中完整声称"基于贝叶斯框架"，满足导师贝叶斯 UQ 方向要求。

---

## 附录：相关参考文献

- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. ICML.
- Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection. arXiv:0710.3742.
- Hadi (博士论文). Shape Graphs for Soccer Analytics. [Table 3.2 几何特征来源]
- Shannon, C. E. (1948). A Mathematical Theory of Communication.
- Jensen, J. L. W. V. (1906). Sur les fonctions convexes. [JSD 理论基础]
