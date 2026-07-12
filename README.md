# MORPH: Probabilistic Football Tactical Structure Recognition via Bayesian Graph Neural Networks

**MORPH** (Metric of Relative Positional Heterogeneity) is a dynamic, context-aware framework for probabilistic football tactical structure identification, powered by Bayesian Graph Neural Networks (B-GNN). This is the first sub-project of the **G-TAF** (Graph Neural Network-driven Tactical Analysis Framework).

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Key Features

- **Bayesian Uncertainty Quantification**: MC Dropout + Dirichlet-Multinomial aggregation for epistemic/aleatoric uncertainty
- **Context-Aware Recognition**: Two-level tactical contextualization (macro-phase + fine-intent)
- **Graph-Based Representation**: Delaunay triangulation + shape-graph pruning for spatial structure
- **Temporal Coherence**: 5.7× more stable than baseline (JSD = 0.106 vs 0.607)
- **High Accuracy**: 91.0% frame-level accuracy, 87.4% macro F1 on FIFA World Cup 2022 dataset
- **Comprehensive Evaluation**: 7-dimensional assessment framework (temporal coherence, semantic validity, predictive power, changepoint detection, novel Bayesian metrics)

---

## 📊 Dataset

**Gradient Sports Enhanced 2022 World Cup Dataset**
- 64 matches, 128 team-sides
- Synchronized tracking data (25 FPS) + event labels
- Test match: Final (Argentina vs France, gameID: 10517)

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MORPH.git
cd MORPH

# Create virtual environment (Python 3.12 required)
python -m venv MORPHenv
source MORPHenv/bin/activate  # On Windows: MORPHenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, pyro, kloppy; print('Installation successful!')"
```

### 2. Data Preparation

Place the 2022 World Cup dataset in:
```
/path/to/Gradient Sports  Enhanced 2022 World Cup Dataset/
```

Update `config.py` with your dataset path.

### 3. Run Pipeline

**Test Version (Single Match - Recommended for First Run)**

```bash
# Step 1: Data Preprocessing
jupyter notebook Step1_Contextualization_Scaling/Test/1.1_test_Convert_TrackingData.ipynb

# Step 2: Graph Representation
jupyter notebook Step2_Graph_Representation/Test/2.1_test_Delaunay_Triangulation.ipynb

# Step 3: B-GNN Inference
jupyter notebook Step3_Probabilistic_Identification/3.2_BGNN/Test/3.2.4_test_b1_inference.ipynb
```

**General Version (All 64 Matches - HPC Recommended)**

```bash
# Run scripts sequentially
python General/scripts/step3_2_4_b1_inference_2.0.py --game_id 10517
python General/scripts/step3_3_1_temporal_coherence.py --game_id 10517
python General/scripts/step3_3_2_tei_semantic.py --game_id 10517
```

For HPC cluster submission:
```bash
sbatch General/slurm/run_step3_3_1.slurm
```

---

## 🏗️ Architecture

### Four-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Contextualization & Scaling                         │
│  ├─ Macro-phase: attack / defense                           │
│  ├─ Fine-intent: BUILD_UP / ATTACKING_PLAY / HIGH_BLOCK /   │
│  │                MID_BLOCK / LOW_BLOCK                      │
│  └─ Spatial normalization (0-1 scaling)                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Graph Representation                                │
│  ├─ Delaunay triangulation                                  │
│  ├─ Shape-graph pruning (remove long edges)                 │
│  └─ Geometric features: 24-dim global + edge attributes     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Bayesian GNN Inference                              │
│  ├─ Stage1 B-GNN: GCN encoder (frozen after training)       │
│  │   ├─ embed(): deterministic 128-dim embedding            │
│  │   └─ embed_mc(): MC Dropout (N=50 samples)              │
│  ├─ Prototype-based classification (45 formations)          │
│  │   └─ Cosine similarity + temperature-scaled softmax      │
│  ├─ Frame-level: MC Dropout → epistemic uncertainty         │
│  └─ Window-level: Dirichlet-Multinomial → P_window ± CI     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Evaluation & Analysis (7 dimensions)                │
│  ├─ 3.3.1 Temporal Coherence (JSD, switch rate, CI smooth)  │
│  ├─ 3.3.2 Semantic Validity (TEI by context, event study)   │
│  ├─ 3.3.3 Predictive Validity (CI→formation change, r=0.19) │
│  ├─ 3.3.4 Novel Bayesian Metrics (Info Gain, Surprise)      │
│  └─ 3.3.5 Changepoint Detection (PELT on JSD)               │
└─────────────────────────────────────────────────────────────┘
```

### Bayesian Framework

**Method A: MC Dropout (Frame-level Epistemic Uncertainty)**
```python
# Inference with dropout enabled
for n in range(N_MC):
    z_mc = model.embed_mc(graph)  # MC Dropout active
    p_n = softmax(cosine_sim(z_mc, prototypes) / tau)
frame_probs = mean(p_n)
frame_epistemic = var(p_n)
```

**Method B: Dirichlet-Multinomial (Window-level Aggregation)**
```python
# Sliding window (300 frames, stride 75)
counts = stability_weighted_counts(frame_probs_window)
alpha = 1 + counts  # Dirichlet prior update
P_window = alpha / sum(alpha)
P_window_CI = dirichlet_credible_interval(alpha, level=0.95)
```

---

## 📈 Results

### Cross-Game Aggregate (64 matches)

| Metric | Value | p-value | Significance |
|--------|-------|---------|--------------|
| **Goal events → TEI** | n=388 | 3.39e-06 | ✅ Highly significant |
| **Substitution → TEI** | n=1150 | 4.14e-06 | ✅ Highly significant |
| **Yellow cards → TEI** | n=210 | 0.384 | ❌ Not significant |
| **CI predictive validity** | r=0.189 | — | ✅ 57% team-sides significant |

### Key Findings

1. **TEI rises after goals** (3.718 → 3.721 bits): Captures tactical transition period
2. **TEI drops after substitutions**: Tactical intervention stabilizes formation
3. **Attack vs Defense TEI**: No significant difference (p=0.295) — Elite teams maintain high discipline in both phases
4. **Bayesian uncertainty predicts formation changes**: CI width correlates with next-window change (Spearman r=0.189, 57% significant)

---

## 📂 Project Structure

```
MORPH/
├── Step1_Contextualization_Scaling/
│   ├── Test/                    # Single-match notebooks
│   └── General/                 # Scripts for all 64 matches
├── Step2_Graph_Representation/
│   ├── Test/
│   └── General/
├── Step3_Probabilistic_Identification/
│   ├── 3.1_Baseline/            # EFPI baseline
│   ├── 3.2_BGNN/                # B-GNN implementation
│   │   ├── Test/
│   │   └── General/
│   │       └── scripts/         # Evaluation scripts (3.3.1-3.3.5)
│   └── 3.3_Evaluation/          # Analysis notebooks
├── General/
│   ├── config.py                # Global configuration
│   ├── scripts/                 # Production scripts
│   │   ├── step3_2_4_b1_inference_2.0.py
│   │   ├── step3_3_1_temporal_coherence.py
│   │   ├── step3_3_2_tei_semantic.py
│   │   ├── step3_3_3_bayesian_predictive.py
│   │   ├── step3_3_4_bayesian_novel.py
│   │   ├── step3_3_5_changepoint.py
│   │   └── step3_3_2_cross_game_aggregate.py
│   └── slurm/                   # HPC job scripts
├── MORPH_ADAPTATION_GUIDE.md    # Implementation log
├── BAYESIAN_FEASIBILITY_ANALYSIS.md
└── requirements.txt
```

---

## 🔬 Evaluation Framework (7 Dimensions)

### 3.3.1 Temporal Coherence
- **JSD stability**: B-GNN 0.106 vs EFPI 0.607 (5.7× better)
- **Switch rate**: 0.67% vs 15.8%
- **CI smoothness**: Dirichlet CI width temporal consistency

### 3.3.2 Semantic Validity
- **TEI by context**: Attack/defense, fine-intent grouping
- **Event study**: ±60s window around goals/substitutions/cards
- **GM-TEI**: Geometry-modulated TEI (incorporates spread/compactness)

### 3.3.3 Bayesian Predictive Validity
- **CI→formation change**: Spearman r median = 0.189
- **Epistemic/Aleatoric quadrant**: High epistemic → reactive defense

### 3.3.4 Novel Bayesian Metrics
- **Information Gain**: KL(P_window || Uniform) — strength of data evidence
- **Bayesian Surprise**: KL(P_t || P_{t-1}) — temporal belief shift

### 3.3.5 Changepoint Detection
- **PELT algorithm** on JSD timeseries (penalty = 5×σ²ln(n))
- **Alignment with events**: Mean distance to nearest goal/substitution

---

## 🛠️ Dependencies

```
torch>=2.1.0
pyro-ppl>=1.8.6
torch_geometric>=2.4.0
kloppy>=3.10.0
polars>=0.19.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
ruptures>=1.1.8
tqdm>=4.66.0
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{morph2026,
  author = {Wu, Jerry},
  title = {MORPH: Probabilistic Football Tactical Structure Recognition via Bayesian Graph Neural Networks},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/MORPH}
}
```

---

## 📧 Contact

- **Author**: JerryWu
- **Project**: G-TAF (Graph Neural Network-driven Tactical Analysis Framework)
- **Institution**: [Your Institution]

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Gradient Sports Enhanced 2022 World Cup Dataset
- **Baseline Method**: EFPI (Evolutionary Formation Pattern Identification)
- **Key References**:
  - Gal & Ghahramani (2016): Dropout as a Bayesian Approximation
  - Minka (2000): Estimating a Dirichlet distribution
  - Killick et al. (2012): PELT algorithm for changepoint detection

---

## 🔄 Development Status

**Current Version**: v7.0 (Bayesian Integration Complete)

- ✅ Step 1-2: Data preprocessing & graph representation
- ✅ Step 3.1: Baseline methods (EFPI)
- ✅ Step 3.2: B-GNN training & inference (single match + all 64 matches)
- ✅ Step 3.3: Evaluation framework (7 dimensions)
- 🚧 Step 4: Tactical benefit assessment (planned)

**Recent Updates**:
- [2026-07] Event annotations (goals, subs, cards, setpieces) added to all visualizations
- [2026-07] Tactical phase color blocks integrated across evaluation plots
- [2026-07] Cross-game aggregate analysis (64 matches) completed
- [2026-06] GM-TEI (Geometry-Modulated TEI) implemented with 24-dim global features
- [2026-03] Bayesian framework (MC Dropout + Dirichlet) integrated

---

## 🚀 Future Work

- [ ] Extend to other datasets (tracking data from other leagues)
- [ ] Real-time inference optimization
- [ ] Interactive visualization dashboard
- [ ] Tactical benefit assessment framework (Sub-project 2)
