# MORPH 项目适配指南

## 项目概述

本文档记录了MORPH（Metric of Relative Positional Heterogeneity）子课题的完整实施过程。MORPH是G-TAF框架的第一个子课题，专注于**动态、情境感知的概率性足球战术结构识别**。

## 快速开始

### 1. 环境配置（推荐方法）

**使用py启动器（推荐，最可靠）⭐**

```cmd
# 1. 切换到MORPH目录
E:
cd E:\JerryWu\Master\SoccerAnalytics\G-TAF\MORPH

# 2. 使用Python 3.12创建虚拟环境
py -3.12 -m venv MORPHenv

# 3. 激活虚拟环境
MORPHenv\Scripts\activate

# 4. 升级pip
python -m pip install --upgrade pip

# 5. 安装依赖包
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 安装unravelsports（可选，使用modified版本）
set PYTHONUTF8=1
pip install "E:\JerryWu\Master\SoccerAnalytics\TrackingData_literature_code\unravelsports-main (modified for 2022 WC)"

# 7. 配置Jupyter内核
python -m ipykernel install --user --name=MORPHenv --display-name="Python (MORPH)"

# 8. 验证安装
python -c "import torch, pyro, kloppy; print('安装成功！')"
```

### 2. 数据准备

确保2022世界杯数据集位于正确路径：
```
E:\JerryWu\Master\SoccerAnalytics\OpenData\TrackingData\Gradient Sports  Enhanced 2022 World Cup Dataset
```

### 3. 运行流程

**测试版（决赛单场数据）- 推荐首次使用**

1. **Step 1 - 数据预处理**
   - `Step1_Contextualization_Scaling/Test/1.1_test_Convert_TrackingData.ipynb`
   - `Step1_Contextualization_Scaling/Test/1.2_test_Phase_Classification.ipynb`
   - `Step1_Contextualization_Scaling/Test/1.3_test_Tactical_Intent.ipynb`
   - `Step1_Contextualization_Scaling/Test/1.4_test_Scaling.ipynb`

2. **Step 2 - 图表示**
   - `Step2_Graph_Representation/Test/2.1_test_Delaunay_Triangulation.ipynb`
   - `Step2_Graph_Representation/Test/2.2_test_Shape_Graph_Pruning.ipynb`
   - `Step2_Graph_Representation/Test/2.3_test_Batch_Processing.ipynb`

3. **Step 3 - 概率识别**
   - `Step3_Probabilistic_Identification/3.1_Baseline/Test/3.1.1_test_EFPI_Formation_Identification.ipynb`
   - `Step3_Probabilistic_Identification/3.1_Baseline/Test/3.1.2_test_ShapeGraphs_Formation_Identification.ipynb`
   - `Step3_Probabilistic_Identification/3.1_Baseline/Test/3.1.3_test_visualize_ShapeGraphs_Video.ipynb`

### 研究信息
- **课题名称**: 图神经网络驱动的概率性足球战术识别与效益评估框架 (G-TAF)
- **子课题一**: MORPH - 战术识别
- **数据集**: Gradient Sports Enhanced 2022 World Cup Dataset
- **测试比赛**: 决赛 (gameID: 10517, 阿根廷 vs 法国)

## 研究框架

MORPH采用**四步走的集成方法**：

### Step 1: 两层级战术情境化与空间对齐

**层级一：基于事件流标签的宏观阶段划分**
- 利用数据集已有标签（`home_ball`、`setpieceType`等）
- 100%准确性，避免建模误差
- 阶段类型：进攻-运动战、防守-运动战、定位球、其他

**层级二：基于CNN的细粒度意图识别**
- 对"进攻-运动战"细分为：Build-up、Attacking-play
- 对"防守-运动战"细分为：High-block、Mid-block、Low-block
- CNN优势：学习相对空间关系，对球队整体移动具有鲁棒性
- 性能：准确率~90%，批处理速度提升100倍

**空间对齐 (Scaling)**
- 计算10名外场球员的最小外接矩形
- 仿射变换缩放至标准尺寸
- 实现尺度不变性

### Step 2: 鲁棒的图表示 (Shape Graphs)

- Delaunay三角剖分构建初始图
- 角度稳定性剪枝生成鲁棒骨架图
- 解决球员个体非结构性移动的噪声

### Step 3: 概率识别与量化

**Sub-step 3.1: 确定性基准**
- 基准A: EFPI (模板匹配法)
- 基准B: Shape Graphs (几何规则法 + K-Means聚类)

**Sub-step 3.2: 概率性统一模型 (U-B-GNN)**
- 贝叶斯图神经网络
- 变分推断(ADVI)训练
- 输出完整概率分布
- TEI 统计分析（全场时序曲线、战术情境分组、事件研究法）
- 时序分割（JSD 变化点检测，SoccerCPD 思路）

**创新指标**:
- **MORPH指数**: 量化战术结构的位置异质性
- **TEI (战术熵指数)**: 量化战术识别的不确定性，公式：`TEI = -Σ(p_i * log(p_i))`

## 项目结构

```
MORPH/
├── MORPH_ADAPTATION_GUIDE.md              # 本文件：项目实施总日志
├── BAYESIAN_FEASIBILITY_ANALYSIS.md       # 贝叶斯方法可行性分析（独立文档，见下）
├── Step1_Contextualization_Scaling/
│   ├── Test/                              # 测试版（决赛）
│   └── General/                           # 完整版（64场）
├── Step2_Graph_Representation/
│   ├── Test/
│   └── General/
└── Step3_Probabilistic_Identification/
    ├── 3.1_Baseline/
    ├── 3.2_Probabilistic_Model/
    └── 3.3_Evaluation/
```

## 重要关联文档

| 文档 | 路径 | 说明 |
|---|---|---|
| **贝叶斯方法可行性分析** | `MORPH/BAYESIAN_FEASIBILITY_ANALYSIS.md` | 在当前框架中引入贝叶斯方法的完整分析，含方案A~D、几何增强型TEI（GM-TEI）、宏观层级TEI、评估方法（JSD/时间相干性/事件研究法）及实施路线图 |
| 科研方案 | `2025.9.7 论文写作 参考文献/G-TAF 科研方案 3.5.docx` | 最新科研方案（需手动更新：方法名由"两阶段架构"改为"B1聚类原型"，删MC Dropout描述，补TEI事件相关性实验数值）|

---

## 统一数据格式标准

### 1. 追踪数据格式 (Polars DataFrame)
```python
{
    'frame_id': int64,
    'period_id': int64,
    'timestamp': datetime,
    'team_id': str,
    'id': str,
    'x': float64,
    'y': float64,
    'v': float64,
    'a': float64,
}
```

### 2. 阵型时段数据 (Pandas DataFrame)
```python
{
    'game_id': str,
    'period_id': int,
    'form_period': int,
    'start_frame': int,
    'end_frame': int,
    'formation': str,
    'formation_probs': dict,      # MORPH特有
    'tactical_entropy': float,    # MORPH特有
    'morph_score': float,         # MORPH特有
}
```

## 环境配置

### Python环境
- **Python版本**: 3.10+ (推荐3.12)
- **虚拟环境名称**: MORPHenv

### 核心依赖包
```
# 数据处理
numpy>=1.23.0
pandas>=1.5.0
polars>=1.0.0

# 图神经网络
torch>=2.0.0
torch-geometric>=2.3.0
networkx>=3.0

# 追踪数据处理
kloppy>=3.17.0
unravelsports>=1.1.0

# 贝叶斯推断
pyro-ppl>=1.8.0
numpyro>=0.13.0
arviz>=0.16.0

# 可视化
matplotlib>=3.6.0
mplsoccer>=1.6.0

# Jupyter
jupyter
notebook
ipykernel
```

## 实施进度

### ✅ 已完成

**阶段1: 数据预处理与格式统一**
- [x] 创建项目结构
- [x] 设计统一数据格式
- [x] 创建数据加载模块

**阶段2: Step 1 两层级情境化实现**
- [x] 层级一：事件流标签解析
- [x] 层级二：CNN意图识别（启发式+CNN）
- [x] EFPI位置缩放机制
- [x] 测试版notebooks（1.1-1.4）

**阶段3: Step 2 实现**
- [x] Delaunay三角剖分 (2.1)
- [x] 角度稳定性剪枝 (2.2)
- [x] 批量处理与验证 (2.3)

**阶段4: Step 3.1 确定性基准（Test版）**
- [x] 基准A: EFPI阵型识别 (3.1.1)
- [x] 基准B: Shape Graphs阵型识别 (3.1.2)
- [x] 可视化工具 (3.1.3)

**阶段5: Step 3.2 概率性模型（Test版）**
- [x] 数据加载与特征工程 (3.2.1)
- [x] 图数据集构建 (3.2.2)，节点特征41维，全局特征24维
- [x] B-GNN模型架构 (3.2.3)，紧凑版 GCN+GlobalFeatureFusion+MCDropout
- [x] B1 聚类原型推断 (3.2.4) — v7.0，方案A（MC Dropout N=50）+ 方案B（Dirichlet窗口聚合）
  - 性能：Acc 91%，Macro F1 87.4%，JSD稳定性 5.7× 优于 EFPI
  - 输出：`b1_window_distributions.parquet`（prob_* + probvar_* + GM-TEI_AB/CB）、`b1_frame_epistemic.npy`
- [x] 可视化 (3.2.5)

**阶段6: Step 3.3 评估（Test版）**
- [x] 3.3.1 时间相干性评估（JSD + Dirichlet CI宽度平滑性）
- [x] 3.3.2 TEI统计分析（GM-TEI时序 + CI按情境分组 + 事件研究±60s）
- [x] 3.3.3 变化点检测（GM-TEI与变化点几何特征联合分析）

**阶段7: General版 Step 2-3.1（全量64场）**
- [x] Step 3.1.1 General版 notebook（并行优化版，4 workers）
- [x] Step 3.1.2.2 General版 notebook（并行优化版，8 workers）
- [x] Step 3.1.2.5 General版 notebook
- [x] Step 2.3 **数据格式重构（2026-04-20）+ 双队覆盖（2026-04-21）**：
  - 旧格式：每帧一个 pkl → 单场10万文件，无法压缩下载
  - **新格式**：每场两个 parquet（主队+客队）：
    - `shape_graph_nodes_{gid}.parquet`（列：frame_id, **team_side**, node_idx, x, y, n_players, n_removed, n_initial, n_edges）
    - `shape_graph_edges_{gid}.parquet`（列：frame_id, **team_side**, src, dst, distance）
  - `team_side`：`'home'` 或 `'away'`，下游按此列过滤单队
  - 命名规范对齐项目现有：`tracking_data_{gid}_suffix.parquet`
  - field_players 字段已确认下游完全不使用，已删除
  - NetworkX 对象改为边列表存储，下游重建 edge_index 零开销
  - 已更新文件：`2.3_general_Batch_Processing.ipynb`、`General/scripts/step2_3_batch_hpc.py`、`General/slurm/slurm_step2_3.sh`、`Step2_Graph_Representation/General/batch_worker.py`
  - **超算正在运行**（2026-04-21）：删除旧 parquet 后用新脚本重新生成，16进程并行

**阶段8: General版 Step 1.3 + Step 3.2（进行中）**
- [x] 3.2.1 General版 notebook：`3.2.1_general_Data_Loading_Feature_Engineering.ipynb`（formation_mapping + feature_config）
- [x] Step 1.3 **双队覆盖改造（2026-04-21）**：
  - `1.3_general_Tactical_Intent.ipynb` cell-16/18 已更新：对主客双队各推理一次
  - 输出新增列：`attack_intent_home`, `defense_intent_home`, `attack_intent_away`, `defense_intent_away`
  - 新增超算脚本：`General/scripts/step1_3_tactical_intent.py` + `General/slurm/slurm_step1_3.sh`
  - **超算正在运行**（2026-04-21），预计2-4小时
- [x] `3.1.2.5_general_EFPI_ShapeGraphs_Comparison.ipynb` 已更新：从 metadata 读取真实 home/away team_id，支持双队对比

### 🔄 待完成

**Step 2.3 数据重新生成**
- [x] 超算运行完成（2026-04-21）
- [x] 下载生成的 `shape_graph_nodes_*.parquet` + `shape_graph_edges_*.parquet`（共128个文件，含 `team_side` 列）

**Step 1.3 数据重新生成**
- [x] 超算运行完成（2026-04-21）
- [x] 下载生成的 `tracking_data_{gid}_intent.parquet`（64个文件，含双队意图列）
- [x] 本地重跑 cell-18（合并标签），生成 `tracking_data_{gid}_tactical_labels.parquet`

**Step 1.4 Scaling**
- [x] 本地运行完成（2026-04-24），生成 `tracking_data_{gid}_scaled.parquet`（64个文件）
- [x] 已上传到超算 `/public/home/hpc242111131/G-TAF/MORPH/data/morph_general/`

**General版 Step 3.1 数据运行**
- [x] 运行 `3.1.1_general_EFPI_Formation_Identification.ipynb`（本地，64场全部成功）
- [x] 运行 `3.1.2.2_general_ShapeGraphs_Batch_Processing.ipynb`（超算完成，64场）
- [x] 下载 `efpi_baseline/{gid}/efpi_results_{gid}.parquet` 和 `shapegraphs_baseline/{gid}/sg_results_{gid}.parquet`

**General版 Step 3.2 剩余**
- [x] 3.2.2 脚本创建：`General/scripts/step3_2_2_dataset.py` + `General/slurm/slurm_step3_2_2.sh`
  - **关键修改（2026-04-25）**：`encode_intent`/`encode_macro` 改为直接读取字符串值
  - **关键修复（2026-05-18）**：标签来源从 EFPI `iloc[0]`（含 ball 行，占 93.6%）改为按 team_id 过滤 EFPI；`build_formation_mapping` 过滤 `"ball"`；预建 frame_id 索引消除逐帧 O(N) 扫描；Polars `group_by` key 类型修复（tuple→int）
  - ⏳ 超算重跑中（2026-05-18 16:27 提交），预计完成时间 2026-05-19 上午；已验证 game 3812：201924样本，node_dim=41，global_dim=24，num_classes=61（无ball），最多类占比 6.8%，home/away 各半
- [x] 3.2.2 General版 notebook：`3.2.2_general_Graph_Dataset_Construction.ipynb`（本地验证，单场 10517）
- [x] 3.2.3 General版 notebook：`3.2.3_general_BGNN_Training.ipynb`（本地单折验证，TEST_GID=10517）
- [x] 3.2.3 超算脚本：`General/scripts/step3_2_3_train.py`（LOGO 64折，num_classes 从 formation_mapping.json 动态读取，num_workers=0）
- [x] 3.2.3 slurm：`General/slurm/slurm_step3_2_3.sh`（1 GPU + 8 CPU + 128G，30天上限）
  - ✅ **已完成（2026-06-10）**：64/64 折，Mean Test Acc=7.21%，Mean Macro F1=4.23%，耗时约60小时
  - **LOGO 性能问题（2026-06-07 确认）**：训练精度7%，权重含NaN（class_weights导致梯度爆炸所致）；嵌入空间完全崩塌（embed()输出全零向量）
  - 修复记录：`drop_last=True` 修复 BatchNorm crash；class_weights 效果更差已回滚
- [x] 3.2.4 超算脚本 v2.0：`General/scripts/step3_2_4_b1_inference_2.0.py`（多轮修复，含以下改动）
  - ① `model.eval()` + `nan_to_num` + 安全归一化（修复NaN）
  - ② 多 game_id 支持（`--game_id 3812 3820 ...`）
  - ③ 按 frame_id 排序恢复时序结构（修复 DLH/HPL 差分全零）
  - ④ Top-K + MeanRank 判别力诊断输出
  - ⑤ GM-TEI_AB/CB 去除恒零的 tac_dir 乘数（简化为纯几何调制熵）
  - ⑥ `THRESHOLD=0.030`（主流阵型4~6个/场）
  - ⑦ home/away 分离：每场输出14文件（home/away各7个），文件名含`_home`/`_away`后缀
  - ⑧ 横轴改为真实比赛时间（分钟）：读 `tracking_data_{gid}_scaled.parquet` 中 period_id+timestamp，按实际段末累计偏移转换 frame_id → game_time_min
  - ⑨ 战术阶段色块修复：用 `ball_owning_team_id` + `attack/defense_intent_home/away` 四列区分持球/防守意图（修复 global_features[1:6] 导致防守状态不显示的bug）
  - ⑩ 阶段图例改为图底色块+标签（`fig.legend`），不再写在图内
  - ⑪ 新增 `tei_combined_{gid}.png`：home vs away TEI/GM-TEI_AB/GM-TEI_CB 同轴对比图
  - ⑫ 新增 `dashboard_{gid}_{side}.png`：4子图（TEI/GM-TEI_AB/GM-TEI_CB/mainstream_prob）共享横轴，含战术阶段色块+关键事件标注
  - ⑬ 关键事件标注（`_load_events` 读 Event Data JSON）：进球（附实时比分+球员姓）、黄/红牌、换人（附换上/下球员姓名+位置）、角球/任意球/点球、中场休息/加时中场
  - ⑭ DataLoader 导入修复（`torch_geometric.loader`）；`weights_only=True` 消除 torch.load 警告
  - 旧版备份为 `step3_2_4_b1_inference_1.0.py`
- [x] 3.2.4 v2.0 测试运行（9场）已完成（2026-06-27）：
  - Top-3 ≈ 27~31%（随机6.7%，**4~5倍**）✅
  - MeanRank ≈ 10~13/45（随机期望23）✅
  - TEI mean=3.67~3.75，std=0.035~0.095（**有时序变化**）✅
  - GM-TEI_AB/CB 非零（去除tac_dir后正常）✅
  - 主流阵型 3~6 个/场（THRESHOLD=0.030 合理）✅
  - **结论**：v2.0 输出有效，可提交全量64场作业
- [x] 3.2.4 v2.0 可视化增强测试（4场，2026-06-28）：真实时间轴、防守阶段显示、事件标注均验证通过
- [ ] step3_2_4 v2.0 全量64场：已提交（2026-06-28），运行中
- [ ] 3.2.4 General版 notebook：`3.2.4_general_B1_ClusterPrototype.ipynb`（本地验证用）

**⚠️ 待导师确认的三个方向（2026-06-20）**
- [x] **方向A（降低类别粒度）** ← 已选定（2026-06-20）
- [ ] **方向B（绕过GNN，用EFPI直接计算TEI）**：放弃，丧失GNN核心贡献
- [ ] **方向C（保持现状，重新定框）**：放弃

**方向A v2.0 脚本改造计划（2026-06-20）**
- [x] **[已完成] [改动2] step3_2_2_dataset_2.0.py**：分层降采样（按阵型类均匀采样，替代纯随机），旧版备份为 step3_2_2_dataset_1.0.py
- [x] **[已完成] [改动1] step3_2_2_dataset_2.0.py**：阵型合并（61类→~45类），旧方案"12粗粒度类"已放弃
  - 数据分析发现：46个阵型出现在全部64场，16个稀少/短暂（出现<50场 或 平均帧<10k）
  - 合并规则：稀少阵型并入语义最近的稳定邻居（同后卫数+数字序列最近）
  - 合并映射（16个）：3421flat→3421, 31312→31213, 3322→3232, 4131→4141, 531→532, 4212→42121, 342→3421, 4221→4231, 432→4321, 312112→31213, 3411→3412, 351→352, 441→442, 4311→4312, 422→4222, 341→3421
  - 实现：新增 `FORMATION_MERGE_MAP` 字典；`build_formation_mapping` 和 `build_graph_for_frame` 均已更新
- [x] **[已完成] [改动3] step3_2_3_train_2.0.py**：去除class_weights，改用label_smoothing=0.1；旧版备份为 step3_2_3_train_1.0.py
- [x] **[已跳过] [改动4]**：加深网络——经分析，45类难度低于61类，2层GCN已够用，跳过
- [x] **[已完成] [改动5] step3_2_2_dataset_2.0.py + config.py**：清理无效全局特征（encode_macro恒零第二位 + zeros[7:13]），global_dim 24→17
- [x] **[已完成] [改动6] step3_2_4_b1_inference_2.0.py**：适配新全局特征索引（global_dim 24→17），更新 aggregate_windows 中的几何特征索引（spread:15→8, lpw:17→10, hull:18→11, compact:19→12, dlh:20→13, hpl:21→14, lr:22→15, rect:23→16）；旧版备份为 step3_2_4_b1_inference_1.0.py
- ✅ v2.0 改动1~6 全部完成
- ✅ **step3_2_2 v2.0 超算运行完成（2026-06-21）**：64/64场，45类，配额/类=444，共~19500~19980样本/场，0失败
- ✅ **step3_2_3 v2.0 超算完成（2026-06-23 提交）**：Mean Acc~8.4%，Mean F1~5.9%（随机基线2.2%），无NaN崩溃，训练loss正常下降
  - LOGO 8% Acc 符合预期（跨队战术泛化难），论文评估重心已转移至嵌入空间判别力+TEI语义（见下方评估框架）

**General版 Step 3.3（2026-07-04 更新）**
- [x] 超算脚本 v3.0：`step3_3_1_temporal_coherence.py`
  - v2.0基础上新增：TEI置信带图、MC Dropout epistemic时序图
  - v3.0修复：`bgnn_sw_roll` 维度不匹配（`eval_temporal_coherence` 图生成失败）
  - v3.0新增：全类事件标注（角球/任意球/点球/球门球/开球）+ 图右侧 Events 图例
  - 输出子目录：`bgnn_analysis/{gid}/3.3.1_temporal_coherence/`（共5图×2队 + jsd parquet）
- [x] step3_3_1 全量64场：✅ 已完成（2026-06-29初跑，2026-07-04修复重跑）
- [x] step3_3_2~3.3.5 超算脚本：✅ 已按方向重写完成（2026-06-30），共5个脚本：
  - `step3_3_1_temporal_coherence.py`（时间相干性）
  - `step3_3_2_tei_semantic.py`（TEI语义有效性）+ 修复：fine_intent merge bug，已重跑
  - `step3_3_3_bayesian_predictive.py`（贝叶斯不确定性预测价值）+ 修复：CJK字体、observed=False、tick_labels，已重跑
  - `step3_3_4_bayesian_novel.py`（贝叶斯专属创新指标：IG/BS）✅ 64/64无错误
  - `step3_3_5_changepoint.py`（PELT变化点检测）+ PELT惩罚系数调整（2×→5×）+ 全类事件标注
- [x] step3_3_1/2/3/4 全量64场：✅ 全部完成
- [ ] step3_3_5 全量64场：待重跑（PELT调参+事件标注更新）
- [x] 新增脚本：`step3_3_2_cross_game_aggregate.py`（跨场聚合分析）
  - 聚合64场×128队所有进球/换人/黄牌/红牌前后±60s TEI窗口
  - TEI按fine_intent分组跨场统计 + CI预测有效性Spearman r分布
  - 点球大战过滤修复（period==4 & clock>7300s）
  - 输出：`bgnn_analysis/cross_game_aggregate/`（4个文件）
- [ ] step3_3_2_cross 全量：待提交

**评估指标分类（2026-06-30 重新划分）**
- [x] **时间相干性**（step3_3_1）：JSD平滑度、切换率、CI宽度时序、TEI置信带、epistemic时序
- [x] **TEI语义有效性**（step3_3_2）：TEI按战术情境Mann-Whitney、CI按情境、事件研究±60s
- [x] **贝叶斯不确定性预测价值**（step3_3_3）：CI→下窗口切换率、Epistemic vs Aleatoric四象限
- [x] **贝叶斯专属创新指标**（step3_3_4）：后验信息增益IG、贝叶斯惊异度BS
- [x] **变化点检测**（step3_3_5）：PELT变化点+事件对齐

**已废弃的早期评估框架（存档）**
- [ ] **维度二：下游任务预测增益**（B-GNN prob vec vs EFPI one-hot，子课题二）
- [ ] **维度三：嵌入空间判别力**（原型语义聚类、Mean Rank、Top-K准确率）
- [ ] **维度四：TEI场景分组差异**（已整合进step3_3_2）
- [ ] **维度五：不确定性-关键事件相关性**（已整合进step3_3_2）
- [x] 删除未运行的冗余脚本：`step1_1_load.py`、`step1_2_macro_phase.py`、`step1_3_fine_intent.py`、`step1_preprocessing.py`、`step2_1_shape_graph.py`
- [x] 删除早期草稿 slurm：`dataset.sh`、`efpi.sh`、`inference.sh`、`preprocess.sh`、`train.sh`
- 保留脚本（11个）：`step1_3_tactical_intent.py`、`step1_4_scaling.py`、`step2_3_batch_hpc.py`、`step3_1_2_shapegraphs.py`、`step3_1_efpi.py`、`step3_2_2_dataset.py`、`step3_2_3_train.py`、`step3_2_4_b1_inference.py`、`step3_2_5_b1_visualization.py`（v2.0）、`step3_2_5_b1_video.py`（新增）、`step3_3_1_temporal_coherence.py`（v2.0）
  - `step3_2_5_b1_visualization.py` v2.0改动：路径修复 + home/away循环 + 文件名`_{side}`后缀 + `--game_id` 支持多ID
  - `step3_2_5_b1_video.py`（新增）：B1阵型概率动态MP4，条形图+TEI时序，支持 `--start_min`/`--end_min`/`--fps`，依赖 ffmpeg
- 保留 slurm（6个）：`slurm_step1_3.sh`、`slurm_step1_4.sh`、`slurm_step2_3.sh`、`slurm_step3_1_2.sh`、`slurm_step3_2_2.sh`、`slurm_step3_2_3.sh`、`slurm_step3_2_4.sh`

---

## 3.3 评估要点（✅ 3.3.1/3.3.2/3.3.3 均已完成重跑）

**定位**：数据驱动的无监督评估，从三个维度量化 B-GNN 相对于 EFPI 基准的优势。

### 输入文件
| 文件 | 路径 |
|------|------|
| TEI 时序 | `data/morph_test/bgnn_analysis/tei_timeseries_10517.parquet` |
| JSD 时序 | `data/morph_test/bgnn_analysis/jsd_timeseries_10517.parquet` |
| 变化点 | `data/morph_test/bgnn_analysis/changepoints_10517.json` |
| EFPI | `data/morph_test/efpi_baseline/efpi_baseline_results_10517_fullmatch.parquet` |
| 图数据集 | `data/morph_test/bgnn_dataset/graph_dataset_10517_full.pkl` |
| 模型权重 | `Step3_Probabilistic_Identification/3.2_Probabilistic_Model/Test/best_model.pth` |

### 三个评估维度

**维度一：时间相干性（✅ 3.3.1 已完成）**
- B-GNN JSD 均值=0.003860，EFPI 变化率均值=0.0034，切换集中在高TEI时刻（1.24×，p=1.10e-118）
- 新增维度C：Dirichlet 95% CI宽度时序 + CI vs TEI 散点（`eval_ci_smoothness_10517.png`）
- 注意：EFPI 必须先 `drop_duplicates('frame_id')` 再计算变化率

**维度二：下游任务预测增益**
- 三模型对比：基础特征 / +EFPI one-hot / +B-GNN 65维概率分布
- 推荐 XGBoost，目标变量用"未来5秒内是否射门"（代理任务，无需额外 EPV 数据）
- 当前只有 Period 1（~49927帧），样本量足够交叉验证

**维度三：不确定性与关键事件相关性（✅ 3.3.2 已完成）**
- 事件研究法：进球/黄牌/换人动态加载 `Event Data/10517.json`
- TEI 峰值与攻防转换正相关已验证

### 关键已知结论
| 指标 | 数值 |
|------|------|
| B-GNN Acc / Macro F1 | 91% / 87.4% |
| 变化点数量（Period 1） | 22 |
| TEI Pearson r（vs JSD） | 0.027 |

### 注意事项
- 只有 Period 1 数据，所有结论均只反映上半场
- GM-TEI_AB 所需特征（Spread/DLH/HPL）全部在 `global_features`（索引15/20/21），零成本
- GM-TEI_CB 需补充 LR/Rectangularity（约15行代码），会导致 `best_model.pth` 失效需重训
- TacDir 建议对 `d_DLH + d_HPL` 做5帧滑动平均后再取符号，避免阶跃跳变
- NotebookEdit 静默失败：所有 notebook 无 `id` 字段，必须用 Bash+Python 按数组索引操作

## 参考文献

### 核心参考文献（已复现代码）
1. **EFPI (2025)**: Bekkers J. EFPI: Elastic formation and position identification in football (soccer) using template matching and linear assignment.
2. **SoccerCPD (2022)**: Kim H, et al. Formation and role change-point detection in soccer matches using spatiotemporal tracking data. KDD 2022.
3. **GAT模型**: Everett G, et al. Evaluating defensive influence in multi-agent systems using graph attention networks. DSAA 2025.

### 待复现文献
4. **Shape Graphs (2025)**: Brandes U, et al. Shape graphs and the instantaneous inference of tactical positions in soccer. npj Complexity, 2025.
5. **Bauer et al. (2023)**: Putting team formations in association football into context. Journal of Sports Analytics, 2023.

### 贝叶斯方法文献
6. **Scholtes & Karakuş (2024)**: Bayes-xG模型 - 分层建模思想
7. **Robberechts et al. (2021)**: 变分推断在足球分析中的应用
8. **Ievoli et al. (2023)**: 马蹄铁先验用于特征选择

## 关键技术点

### 1. 数据加载
```python
from kloppy import pff
from unravel.soccer import KloppyPolarsDataset

kloppy_dataset = pff.load_tracking(
    meta_data="path/to/Metadata/10517.json",
    roster_meta_data="path/to/Rosters/10517.json",
    raw_data="path/to/Tracking Data/10517.jsonl.bz2",
    coordinates="secondspectrum",
    only_alive=True
)

dataset = KloppyPolarsDataset(
    kloppy_dataset=kloppy_dataset,
    orient_ball_owning=True,
    add_smoothing=True
)
```

### 2. 图构建
```python
from scipy.spatial import Delaunay

# Delaunay三角剖分
tri = Delaunay(player_positions)

# 角度稳定性剪枝
stable_edges = prune_by_angle_stability(tri, time_window=5)
```

### 3. B-GNN模型（核心创新）
```python
import pyro
from pyro.infer import SVI, Trace_ELBO
from torch_geometric.nn import GATConv

class BayesianGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, n_formations):
        super().__init__()
        self.gat1 = GATConv(node_features, hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, n_formations)
    
    def predict(self, x, edge_index, num_samples=100):
        """返回阵型概率分布"""
        predictive = pyro.infer.Predictive(self.model, guide=self.guide, num_samples=num_samples)
        samples = predictive(x, edge_index)
        probs = torch.softmax(samples['obs'], dim=-1).mean(dim=0)
        return probs
```

## 故障排除

### Q1: unravelsports 依赖冲突
`keras==2.14.0` 与 TensorFlow 2.18+ 冲突 → 改用 PYTHONPATH 方式加载（绕过 pip install）

### Q2: Step 3.2 使用 CPU 而非 GPU
MORPHenv 安装的是 CPU-only PyTorch wheel，代码逻辑本身正确。
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```
Step 3.1 无 PyTorch，无需修改。General 版（64场）必须 GPU，CPU 估计耗时数天。

### Q3: 网络超时
```cmd
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 核心创新说明

### 1. 从确定性到概率性的范式升级
- **传统方法**: 输出单一阵型标签
- **MORPH方法**: 输出完整概率分布，捕捉混合状态

### 2. 两层级战术情境化
- **层级一**: 利用已有标签，100%准确
- **层级二**: CNN深度学习，细粒度识别

### 3. MORPH指数
量化战术结构的位置异质性：
- 高值：阵型模糊（"无定形体"）
- 低值：阵型清晰

### 4. 战术熵(TEI)
量化战术识别的不确定性：
- 高熵：混合状态（如0.7×4-3-3 + 0.3×4-5-1）
- 低熵：单一阵型（如0.95×4-4-2）

## General 版本建立备注

### 背景：Test 版本（单场决赛）的局限性与 General 版本的设计方向

**Test 版本现状（game_id=10517，2022世界杯决赛）**
- 测试集 Acc 91%，Macro F1 0.87，模型结构验证通过
- Val Loss 全程低于 Train Loss（Dropout 正则化正常现象，非过拟合）
- 弱点：Class 46/47（support≤6）F1 极低，属数据稀缺问题，非模型问题

**General 版本的核心验证目标**
- 若多场数据上 Macro F1 仍保持高位 → 证明 B-GNN 学到的是**跨队通用的阵型图拓扑规律**
- 若 TEI 在战术过渡期（攻守转换、换人后）显著升高 → 证明概率性输出的实际价值
- 消融实验：B-GNN vs EFPI 确定性基准，量化图结构的独立贡献

**两阶段验证设计**

| 阶段 | 数据 | 目的 |
|------|------|------|
| Test 版（已完成） | 1场决赛（game_id=10517） | 验证代码可行性、模型结构正确性 |
| General 版（待完成） | 64场全部比赛 | 验证泛化能力、TEI 的跨队/跨赛事价值 |

---

## 3.3.2 参考要点（✅ 已完成并重跑）

**定位**：全场批量推理 + TEI 统计分析（全场时序曲线、情境分组箱线图、B-GNN vs EFPI 一致性、事件研究法、JSD时间相干性）

**tei_timeseries_10517.parquet 列结构**（供 3.3.3 读取）：
```
frame_id | period_id | time_sec | tei | top1_formation | top1_prob | macro_phase | fine_intent
```
> ⚠️ 每帧完整概率分布（65维）未保存，需要时必须从图数据集重新推理

**输出文件**（`data/morph_test/bgnn_analysis/`）：
```
tei_timeseries_10517.parquet / tei_timeseries_10517.png / tei_by_context_10517.png
bgnn_vs_efpi_10517.png / event_study_tei_10517.png / jsd_temporal_coherence_10517.png
```

**关键 bug（已修复，重跑时注意）**：
- `result_df` 重启内核后丢失 → Cell 12/18 已加 parquet 回退加载
- 进球时间用 period 内相对秒数：Period1 进球 23'(梅西点球)=1380s、36'(迪玛利亚)=2160s
- EFPI 需先 `drop_duplicates('frame_id')` 再计算变化率
- `timestamp` 是每段内相对时间（从0开始），非全场累计

---

## 更新日志

### 当前进度（2026-03-08）

**已完成:**
- ✅ Step 1-2 全部完成
- ✅ Step 3.1 确定性基准（EFPI + Shape Graphs）完成
- ✅ Step 3.2.1-3.2.3 图数据集与 B-GNN 架构完成
- ✅ Step 3.2.4 重建：两阶段架构（v6.0）训练 notebook 创建完成
- ✅ Step 3.2.5 重建：两阶段可视化 notebook 创建完成
- ✅ Step 3.2.6-3.2.7 完成后迁移为 Step 3.3.2-3.3.3（已迁移至 3.3_Evaluation/Test/）

**架构调整:**
- 原 Step 4（时序分割）整合为 Sub-step 3.2.7，MORPH 框架统一为三步走（Step 1-2-3）
- 3.2.4/3.2.5 已从旧方案（帧级软标签）彻底切换为 v6.0 两阶段架构

**下一步（按顺序）:**
1. ▶️ 运行 `3.2.4_test_TwoStage_Architecture.ipynb` 全程（训练 Stage2，生成 `best_stage2_model.pth` 和 `mainstream_mapping.json`）
2. ▶️ 运行 `3.2.5_test_TwoStage_Visualization.ipynb` 全程（验证推理 + 生成视频）
3. 用 Stage2 窗口级输出重跑 3.2.6（TEI 统计分析）
4. 用 Stage2 输出重跑 3.2.7（变化点检测）
5. 创建 Sub-step 3.3 评估 notebook

---

### 3.2.5 关键数据结构参考（旧版 per-frame 已替换为新版 TwoStage）

> 新版 notebook：`3.2.5_test_TwoStage_Visualization.ipynb`（已创建，依赖 3.2.4 输出）

**graph 对象属性**：`g.x`(10,41), `g.global_features`(1,22), `g.y`, `g.frame_id`

**tracking_data 关键列**：`frame_id`, `id`, `x`, `y`, `team_id`(363=法国/364=阿根廷), `period_id`, `timestamp`, `macro_phase`, `fine_intent`, `position_name`

**frame_id 范围**：4630–264811（graph 数据集 Period1：4630–98893）

**视频布局**（左：球场+Shape Graph 边+球员节点；右：主流阵型 Top-7 概率条形图）
- 表头第一行显示当前帧 EFPI 阵型标签（`EFPI: {efpi_form}`）
- 阵型名在柱状图**左侧**，概率%在右侧，平滑动画 `smooth_alpha=0.3`
- 全局变量 `_prev_rankings = {}` 每次视频前必须重置
- `mplsoccer.Pitch(pitch_type='secondspectrum', pitch_length=105, pitch_width=68)`
- 视频用 `FuncAnimation` + `FFMpegWriter(fps=25, codec='h264')`



### 关键技术坑（必读）

#### ⚠️ NotebookEdit 对无 id 字段的 notebook 静默失败
- 本项目所有 notebook 单元格均无 `id` 字段
- NotebookEdit 按 cell_id 匹配时找不到目标，**静默失败，不报错**
- **必须用 Bash+Python 按数组索引操作**: `nb['cells'][index]['source'] = ...`

#### ⚠️ f-string 内不能含真实换行符（Python < 3.12）
- 用脚本写入 notebook source 时，`\n` 在 f-string 内会被存为 chr(10)，导致跨行 f-string 非法
- **解决方案**: 将 `print(f'\n...')` 拆为 `print()` + `print(f'...')`

#### ⚠️ 数据集时序偏差导致严重过拟合
- `all_graphs` 按帧ID顺序排列，直接切分导致训练集/验证集分布完全不同
- **解决方案**: 切分前加 `random.shuffle(all_graphs)`

#### ⚠️ num_classes 必须来自 metadata，不能用 len(label_counts)
- 全量数据集有 65 种阵型，但单场比赛只出现 43 种
- `class_weights` 大小必须等于模型输出维度（65），否则越界

---

### 修复记录（摘要）

| 日期 | 步骤 | 修复要点 |
|------|------|---------|
| 2026-01-07 | 1.3 | macro_phase/fine_intent 双列，废弃 tactical_label |
| 2026-01-20 | 3.2.1/3.2.2 | 节点特征 99→41维，全局 24→22维 |
| 2026-02-24 | 3.2.4 | NaN 图过滤；shuffle 防时序偏差；Acc 91%/F1 87% |
| 2026-02-25 | 3.2.5 | MC Dropout 可视化；GK 节点索引错位修复 |
| 2026-02-25 | 3.2.6/3.2.7 | period 自适应；rbf→l2（内存溢出）；auto_pen BIC-like |
| 2026-03-11 | 3.2.4b | `median_prob → mean_prob`（softmax winner-takes-all 导致主流阵型识别为空）|
| 2026-03-11 | 3.2.4c | B2 域偏移 0.446（目标 >0.7），放弃 B-GNN 几何原型路径 |
| 2026-03-12 | 3.2.4d | B2-D Procrustes 三次迭代均失败（TEI=0.96/0.97），放弃，保留 notebook 作为失败案例 |
| 2026-03-13 | 3.2.4 | STRIDE 150→75（75% 重叠）；重跑窗口聚合 |
| 2026-03-15 | 3.2.4 | 实施贝叶斯方案 A+B：MC Dropout embed_mc() + Dirichlet-Multinomial 窗口聚合 |
| 2026-03-15 | 3.2.5 | 新增 Dirichlet 后验 95% CI 误差棒 |
| 2026-03-23 | 3.2.6/3.2.7 | 迁移为 3.3.2/3.3.3；3.3 评估三联强化（GM-TEI_AB/CB、CI 宽度分析）|
| 2026-03-24 | 3.3.2/3.3.3 | 事件数据对接（进球/黄牌/换人动态加载 Event Data/10517.json）|
| 2026-04-08 | 3.3.2/3.3.3 | ✅ 重跑顺利完成 |

---

## 3.3.3 参考要点（✅ 已完成并重跑）

**定位**：基于概率分布序列的战术变化点检测（JSD 时序 + PELT 变化点检测）

**输入**（均已存在）：
- `data/morph_test/bgnn_analysis/tei_timeseries_10517.parquet`（TEI 时序）
- `data/morph_test/bgnn_dataset/graph_dataset_10517_full.pkl`（图数据集，65维概率须重算）
- `data/morph_test/efpi_baseline/efpi_baseline_results_10517_fullmatch.parquet`（EFPI）

**输出文件**（`data/morph_test/bgnn_analysis/`）：
```
jsd_timeseries_10517.parquet / changepoints_10517.json
jsd_vs_efpi_10517.png / changepoint_analysis_10517.png / tei_jsd_correlation_10517.png
```

**关键技术点**：
- JSD 须除以 log₂(NUM_CLASSES=65) 归一化 + `np.clip(..., 0, 1)`（不要除 ln(2)）
- PELT 用 `model='l2'`（rbf 需 N×N Gram 矩阵 → 18.6GB 内存溢出）
- `auto_pen = max(2 * sig_var * log(n), 1e-8)`（BIC-like，非 n×var；否则变化点=0）
- EFPI 先 `drop_duplicates('frame_id')` 再计算变化率（原始数据每帧含多行球员记录）
- 图数据集只有 Period 1（上半场），所有分析均只反映上半场
- `ruptures` 安装：`pip install ruptures -i https://pypi.tuna.tsinghua.edu.cn/simple`

---

## 两阶段架构设计方案（v6.0，已废弃）

> ❌ **已废弃**（2026-03-08）。v6.0 Stage 2 GT 生成将非主流阵型过滤掉，导致 51.4% 帧数据丢失，切换为 v7.0 B1 原型方案。备份：`Test - 备份 (per-frame 分类框架)/`

**设计要点**：Stage 1（冻结 B-GNN）→ 128维 z_t；Stage 2 对 300帧窗口 z_t 加权聚合 → 主流阵型概率分布；训练目标 = 窗口级 KL 散度。因预设白名单导致数据丢失严重，放弃。

---

## v7.0 架构方案：原型匹配（2026-03-10，当前采用）

**v6.0 失败根因**：Stage 2 GT 生成过滤非主流阵型，51.4% 帧数据丢失，GT 严重失真。

**v7.0 核心设计**：B-GNN（Stage 1）→ z_t（128维）→ 与 65 个 EFPI 原型向量 μ_k 做 cosine 相似度 → softmax(·/τ) → 65维软概率分布 → Dirichlet 窗口聚合 → 主流阵型（无需白名单）。

### 路径 B1：学习型原型（✅ 当前采用）

**原理**：按 EFPI 标签对 z_t 分组取均值得到 μ_k，leave-one-game-out 策略避免循环依赖。

**THRESHOLD 参数**：
- 单场 Test：0.030（K=43，基线=0.023，实测 442/5221/4132 可识别）
- General 版（64场）：0.040（原型质量更高，均值概率预计 0.06~0.10）
- 诊断：阈值应为均匀基线（1/K）的 1.3~2.0 倍

**对应 notebook**：`3.2.4_test_TwoStage_B1_ClusterPrototype.ipynb`

### 路径 B2：几何型原型（❌ 已放弃）

域偏移中位余弦相似度 0.446（目标 >0.7），合成图（速度=0）与真实帧联合分布差异过大，放弃。notebook：`3.2.4c_DEPRECATED_B2_GeometricPrototype.ipynb`

### 路径 B2-D：Procrustes 纯几何对比（❌ 已放弃）

VL 层级重心 Procrustes 距离 → softmax，TEI=0.96/0.97，阵型层数语义不匹配（442=3层 vs 32122=5层），判别力为零，放弃。notebook：`3.2.4d_DEPRECATED_B2D_Procrustes.ipynb`

### 后处理：时间一致性识别主流阵型（两条路径共用）

```python
# 方法：对全场所有窗口的 P_window 做稳定性筛选
# "主流阵型" = 在多个连续窗口中均保持高概率的阵型

window_distributions = [P_w1, P_w2, ..., P_wN]  # 每个 65 维

# 每种阵型的"时间稳定性"= 在所有窗口中概率的中位数
median_prob = torch.stack(window_distributions).median(dim=0).values
mainstream_candidates = [
    formations[k] for k in range(65)
    if median_prob[k] > threshold  # threshold 由数据驱动确定，如 0.05
]
```

---

### 实施计划（分步进行）

| 步骤 | 状态 | 内容 | 对应文件 |
|------|------|------|---------|
| Step 1 | ✅ 已完成 | 创建 B1 notebook：聚类原型 + 帧级相似度 + 窗口聚合 | `3.2.4_test_TwoStage_B1_ClusterPrototype.ipynb` |
| Step 2 | ✅ 已完成 | 创建 B2 notebook：模板图构造 + 几何原型 + 域偏移验证 | `3.2.4c_DEPRECATED_B2_GeometricPrototype.ipynb` |
| Step 3 | ✅ 已完成 | 运行 B1 + B2；B2 域偏移中位 cos=0.446（失败，目标>0.7）→ B-GNN几何原型路径放弃 | 两个 notebook 均已运行 |
| Step 4 | ✅ 已完成 | 确认 B2 根因（合成图联合分布偏移），选择 Method D（Procrustes 纯几何对比）作为独立验证层 | 见 B2-D 节 |
| Step 5 | ✅ 已完成 | 修复 B1 bug：`median_prob → mean_prob`（Cell 18/22）；创建 B2-D notebook | `3.2.4`（已修复）、`3.2.4d_DEPRECATED`（已废弃）|
| Step 6 | ✅ 已完成 | B2-D 多次迭代失败（15维几何特征/5层VL重心/分层Procrustes），根因：阵型层数多样性（442=3层/4231=4层/32122=5层）与真实帧固定5层VL标签语义不匹配 | B2-D 路径放弃 |
| Step 7 | ✅ 已完成 | 重跑 B1（STRIDE=75）→ 确认主流阵型识别有效 → 更新 3.2.5 可视化（Top-7、EFPI表头、ball行过滤）| 按第一阶段计划执行 |

> ⚠️ **废弃说明**：v6.0 的 `3.2.4_test_TwoStage_Architecture.ipynb` 和 `Stage2Aggregator` 方案（29类过滤+KL损失）**不再作为主推方案**，但保留文件供参考。

### Notebook 结构说明

**B1 Notebook**（24 个单元格）：
1. 导入 → 2. 加载数据 → 3. Stage 1 模型加载 → 4. 稳定性分数
→ 5. 批量提取 z_t → **6. 聚类原型计算**（核心）→ 7. 温度参数与帧级相似度
→ 8. 温度敏感性分析 → 9. 窗口聚合与时间一致性 → 10. 可视化 → 11. 保存

**输出文件**：`b1_prototypes.pth`、`b1_window_distributions.parquet`、`b1_mainstream_result.json`、`b1_tau_sensitivity.png`、`b1_formation_timeseries.png`、`b1_tei_timeseries.png`

**B2 Notebook**（28 个单元格）：
1-5. 同 B1 → **6. 模板 Shape Graph 构造函数**（核心）→ **7. 几何原型 μ_k_B2 计算**
→ **8. 域偏移验证**（B1 vs B2 余弦相似度）→ 9. 域偏移可视化
→ 10. B2 帧级相似度与窗口聚合 → **11. B1 vs B2 结果对比** → 12. 可视化 → 13. 保存

**输出文件**：`b2_prototypes.pth`、`b2_window_distributions.parquet`、`b2_mainstream_result.json`、`b2_domain_shift.png`、`b2_formation_timeseries.png`

**运行顺序**：先运行 B1（生成 `b1_prototypes.pth`），再运行 B2（自动加载 B1 结果做对比）。

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `TAU` | 0.07 | 温度参数，越小分布越尖锐（过小→过拟合单一阵型） |
| `WINDOW` | 300 帧 | 聚合窗口大小（≈10秒@29.97fps） |
| `STRIDE` | 75 帧 | 窗口步长（75%重叠，≈2.5秒）|
| `THRESHOLD` | 0.04 | 主流阵型识别的中位概率阈值 |
| 域偏移警戒线 | 0.7 | B1 vs B2 余弦相似度中位数 <0.7 则 B2 可信度下降 |

---

---

## General 版本超算部署方案（已暂停，保留备用）

> ⚠️ **状态更新（2026-04-09）**：由于超算排队时间长、报错多，已改为**本地 GPU 实施**（见下一节）。本节内容保留备用，供未来需要时参考。

### 关键信息速查

**CSUHPC 平台**：用户 `hpc242111131`，账户 `pi_wanghongqiao`，GPU 队列 `gpu4Q`  
**数据路径**：`/public/home/hpc242111131/Gradient Sports  Enhanced 2022 World Cup Dataset/`（含两个空格）  
**环境激活**（每次登录）：
```bash
module load anaconda3/4.9.2 CUDA/12.1.0 GNU/gcc-12.2.0
source activate morph_env  # 注意：用 source，不是 conda activate
export PYTHONPATH="/public/home/hpc242111131/unravelsports-main (modified for 2022 WC):$PYTHONPATH"
```

### General .py 脚本（已完成）

所有脚本位于 `MORPH/General/scripts/`，对应 Test notebooks：
- `step1_1_load.py` → 1.1（kloppy 加载）
- `step1_2_macro_phase.py` → 1.2（宏观阶段）
- `step1_3_fine_intent.py` → 1.3（细粒度意图，启发式规则版）
- `step1_4_scaling.py` → 1.4（EFPI 缩放）
- `step2_1_shape_graph.py` → 2.1-2.2（Delaunay + 剪枝）
- `step3_2_2_dataset.py` → 3.2.2（PyG 图数据集，24维 global_features）
- `step3_2_3_train.py` → 3.2.3（B-GNN 训练）
- `step3_2_4_b1_inference.py` → 3.2.4（B1 推断，待创建）

### 关键参数差异

| 参数 | Test | General | 说明 |
|------|------|---------|------|
| `THRESHOLD` | 0.030 | **0.040** | 多场数据阈值提高 |
| 原型计算 | 全场均值 | **Leave-one-game-out** | 当前比赛不参与 μ_k 计算 |

### SLURM 提交示例

```bash
# 训练
sbatch slurm/train.sh

# 推断（需先设置 GAME_ID）
export GAME_ID=10517
sbatch slurm/inference.sh
```

详细配置见 `MORPH/General/` 目录下的脚本和 SLURM 模板。

---

## General 版本双队覆盖要求（2026-04-21 确定）

### 核心要求

General 版本需**同时处理每场比赛的主队（home）和客队（away）**，不能只处理主队。

### 各文件双队状态

| 文件 | 状态 | 说明 |
|------|------|------|
| 1.1 / 1.2 / 1.4 | ✅ 原生双队 | tracking parquet 含主客双队，无需修改 |
| 1.3 `Tactical_Intent` | ✅ 已改（2026-04-21） | 新增 `attack_intent_home/away` + `defense_intent_home/away` 列 |
| 2.3 / `batch_worker.py` / `step2_3_batch_hpc.py` | ✅ 已改（2026-04-21） | parquet 新增 `team_side`（`'home'`/`'away'`）列 |
| 3.1.1 EFPI | ✅ 原生双队 | kloppy+unravel 内部处理双队，输出含 `team_id` 列 |
| 3.1.2.2 ShapeGraphs | ✅ 已改（2026-04-21） | 按 `team_side` 过滤后分别计算阵型，输出含 `team_side` 列 |
| 3.1.2.5 比较分析 | ✅ 已改（2026-04-21） | 从 metadata 读取真实 home/away team_id，双队分别对比 |
| 3.2.1 特征配置 | ✅ 无需改 | 生成共享配置文件，与球队无关 |

### 新增超算脚本

- `General/scripts/step1_3_tactical_intent.py`：Step 1.3 超算版（双队推理）
- `General/slurm/slurm_step1_3.sh`：对应 SLURM 脚本

---

## Step 1 General 版本本地实施（2026-04-09）

### 背景

原计划在 CSUHPC 超算运行 General 版本（64场比赛），但遇到排队时间长、报错多等问题。决定改为**本地完成 General 版本代码撰写和运行**，超算版本 .py 脚本保留备用。

### 环境配置

**硬件**：
- GPU: NVIDIA GeForce RTX 4050 Laptop (6GB VRAM)
- Driver: CUDA 13.0 (581.83)
- OS: Windows 11 Home China 10.0.26200

**软件环境**：
- Python: 3.12.4 (MORPHenv venv)
- PyTorch: 2.6.0+cu124（从 2.9.1+cpu 升级，支持 GPU 加速）
- torch-geometric: 2.7.0

**PyTorch CUDA 版本安装**（已完成，2026-04-09）：
```bash
# 使用阿里云镜像直接下载 whl（官方源断网，阿里云 ~6 MB/s 稳定）
pip install \
  "https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.6.0+cu124-cp312-cp312-win_amd64.whl" \
  "https://mirrors.aliyun.com/pytorch-wheels/cu124/torchvision-0.21.0+cu124-cp312-cp312-win_amd64.whl" \
  "https://mirrors.aliyun.com/pytorch-wheels/cu124/torchaudio-2.6.0+cu124-cp312-cp312-win_amd64.whl"
```
验证结果：`torch.cuda.is_available() = True`，GPU = RTX 4050 Laptop，CUDA 12.4，显存 6.0 GB

### 已完成的 Notebooks

所有 General 版本 notebooks 已创建在 `Step1_Contextualization_Scaling/General/` 目录：

| Notebook | 功能 | 关键特性 |
|----------|------|----------|
| **1.1_general_Convert_TrackingData.ipynb** | 批量加载64场比赛追踪数据 | • 断点续传<br>• 并行处理选项<br>• 错误日志记录<br>• 统计可视化 |
| **1.2_general_Phase_Classification.ipynb** | 宏观阶段划分（事件流标签） | • 批量事件解析<br>• 帧级阶段映射<br>• 阶段分布统计 |
| **1.3_general_Tactical_Intent.ipynb** | 细粒度战术意图识别（CNN） | • **GPU 加速**（AMP 混合精度）<br>• 向量化热图生成<br>• 批量推理优化<br>• 模型复用（训练一次，推理64场）<br>• 断点续传 |
| **1.4_general_Scaling.ipynb** | 空间对齐（EFPI 缩放） | • 向量化 groupby 缩放<br>• 内存优化<br>• 可视化验证 |

### GPU 加速优化（1.3 notebook）

**优化策略**：
1. **AMP 混合精度训练**：`torch.autocast` + `GradScaler`，显存占用减半，速度提升 ~2x
2. **批量热图生成**：向量化 numpy 操作，避免逐帧循环，速度提升 5-10x
3. **DataLoader 多线程**：`num_workers=4`（Windows 建议 0-4）
4. **推理批量优化**：`batch_size=256`（无梯度，可用更大批量）
5. **模型复用**：训练一次（决赛数据），保存权重，批量推理64场

**预期性能**（RTX 4050）：
- 训练：2-3 分钟（vs CPU 10+ 分钟）
- 单场推理：1-2 分钟（vs CPU 10 分钟）
- 64场总时间：约 2-4 小时

### 输出文件结构

```
data/morph_general/
├── tracking_data_{gid}.parquet              # 1.1 输出：原始追踪数据
├── metadata_{gid}.json                      # 1.1 输出：比赛元数据
├── tracking_data_{gid}_phase.parquet        # 1.2 输出：含宏观阶段标签
├── tracking_data_{gid}_intent.parquet       # 1.3 输出：含 CNN 推理结果
├── tracking_data_{gid}_tactical_labels.parquet  # 1.3 输出：合并 phase + intent
├── tracking_data_{gid}_scaled.parquet       # 1.4 输出：最终版（含缩放坐标）
├── shape_graph_nodes_{gid}.parquet          # 2.3 输出：节点表（frame_id, node_idx, x, y, n_players, n_removed, n_initial, n_edges）
├── shape_graph_edges_{gid}.parquet          # 2.3 输出：边表（frame_id, src, dst, distance）
├── step1_1_summary.csv                      # 各步骤汇总
├── step1_2_summary.csv
├── step1_3_summary.csv
├── step1_4_summary.csv
└── step2_3_summary.csv

models/
├── attack_model_best.pth                    # CNN 进攻意图分类器
└── defense_model_best.pth                   # CNN 防守意图分类器
```

### 与 Test 版本的差异

| 项目 | Test 版 | General 版 |
|------|---------|------------|
| 比赛数量 | 1场（决赛10517） | 全部64场 |
| 处理方式 | 单文件，逐步执行 | 批量并行，断点续传 |
| GPU 加速 | 基础（可选） | 完整优化（AMP + 向量化） |
| 错误处理 | 基础 | 完善（记录失败比赛，保存日志） |
| 输出格式 | 单文件 parquet | 每场独立 parquet + 汇总 CSV |
| 模型训练 | 每次运行重新训练 | 训练一次，保存复用 |

### 下一步

1. **运行 General notebooks**：在本地 GPU 环境执行 1.1-1.4
2. **验证输出**：检查 64 场比赛的处理结果
3. **Step 2-4**：根据需要创建 General 版本的后续步骤

### 注意事项

- **显存管理**：RTX 4050 仅 6GB，batch_size 不宜过大（训练128，推理256）
- **断点续传**：所有 notebooks 支持跳过已处理比赛，可随时中断恢复
- **数据路径**：确保数据集路径正确（含两个空格）：
  ```
  E:\JerryWu\Master\SoccerAnalytics\OpenData\TrackingData\Gradient Sports  Enhanced 2022 World Cup Dataset
  ```

---

**文档版本**: v10.0
**最后更新**: 2026-04-20（v10.0 Step 2.3 数据格式重构：旧格式每帧一个 pkl → 新格式每场两个 parquet；补充 shape_graph_nodes/edges 到输出文件结构图；field_players 确认冗余已删除，NetworkX 改为边列表存储；v9.9 更新 PyTorch CUDA 安装方式为阿里云镜像直链；v9.8 精简超算部署方案章节；v9.7 新增 Step 1 General 版本本地实施记录）
