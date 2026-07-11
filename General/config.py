"""
MORPH General版本全局配置
本地开发 / 超算部署 通过 ENV 环境变量切换
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# 运行环境检测
# ─────────────────────────────────────────────
ENV = os.environ.get("MORPH_ENV", "local")  # "local" | "hpc"

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
if ENV == "hpc":
    HOME = Path("/public/home/hpc242111131")
    DATA_ROOT = HOME / "Gradient Sports  Enhanced 2022 World Cup Dataset"
    PROJECT_ROOT = HOME / "G-TAF/MORPH/General"
else:
    HOME = Path(r"E:\JerryWu\Master\SoccerAnalytics")
    DATA_ROOT = HOME / "OpenData/TrackingData/Gradient Sports  Enhanced 2022 World Cup Dataset"
    PROJECT_ROOT = HOME / "G-TAF/MORPH/General"

# 原始数据子目录
TRACKING_DIR  = DATA_ROOT / "Tracking Data"
METADATA_DIR  = DATA_ROOT / "Metadata"
ROSTERS_DIR   = DATA_ROOT / "Rosters"
EVENT_DIR     = DATA_ROOT / "Event Data"

# 输出目录
OUTPUT_ROOT      = PROJECT_ROOT / "data"
if ENV == "hpc":
    MORPH_GENERAL = HOME / "G-TAF/MORPH/data/morph_general"
else:
    MORPH_GENERAL = Path(r"E:\JerryWu\Master\SoccerAnalytics\G-TAF\MORPH\data\morph_general")
TRACKING_OUT     = OUTPUT_ROOT / "tracking"       # step1 输出
SHAPE_GRAPH_OUT  = OUTPUT_ROOT / "shape_graphs"   # step2 输出
EFPI_OUT         = MORPH_GENERAL / "efpi"         # step3.1 输出
DATASET_OUT      = OUTPUT_ROOT / "bgnn_dataset"   # step3.2.2 输出
MODEL_OUT        = OUTPUT_ROOT / "models"         # step3.2.3 输出
ANALYSIS_OUT     = OUTPUT_ROOT / "bgnn_analysis"  # step3.2.4 / step3.3 输出

# ─────────────────────────────────────────────
# 比赛列表（64场，gameID）
# ─────────────────────────────────────────────
# 小组赛 3812-3859，淘汰赛 10502-10517
ALL_GAME_IDS = [
    3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819,
    3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827,
    3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835,
    3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843,
    3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851,
    3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859,
    10502, 10503, 10504, 10505, 10506, 10507, 10508,
    10509, 10510, 10511, 10512, 10513, 10514, 10515,
    10516, 10517,
]

TEST_GAME_ID = 10517  # 决赛（阿根廷 vs 法国）

# ─────────────────────────────────────────────
# Step 1 参数
# ─────────────────────────────────────────────
ORIENT_BALL_OWNING = True
ADD_SMOOTHING      = True

# ─────────────────────────────────────────────
# Step 2 参数
# ─────────────────────────────────────────────
ALPHA_THRESHOLD = 3 * 3.14159265 / 4  # 135° 角度稳定性剪枝阈值

# ─────────────────────────────────────────────
# Step 3.2.2 图数据集参数
# ─────────────────────────────────────────────
NODE_DIM    = 41
GLOBAL_DIM  = 17   # v2.0：macro(1)+intent(5)+centroid(2)+spread/diam(2)+geom(7)，已删除恒零维度
NUM_CLASSES = 45   # v2.0 预估（合并16个稀少阵型后，实际值由 formation_mapping.json 确定）

# ─────────────────────────────────────────────
# Step 3.2.3 B-GNN 训练参数
# ─────────────────────────────────────────────
HIDDEN_DIM          = 128
GRAPH_EMBEDDING_DIM = 64
FUSION_DIM          = 64
NUM_CONV_LAYERS     = 2
DROPOUT             = 0.3
MC_DROPOUT          = 0.5
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-4
NUM_EPOCHS          = 80
PATIENCE            = 12
BATCH_SIZE_TRAIN    = 128
BATCH_SIZE_EVAL     = 256

# ─────────────────────────────────────────────
# Step 3.2.4 B1 推断参数
# ─────────────────────────────────────────────
WINDOW    = 300    # 窗口帧数（≈10秒 @29.97fps）
STRIDE    = 75     # 步长帧数（75% 重叠，≈2.5秒）
TAU       = 0.07   # 温度参数
THRESHOLD = 0.030  # 主流阵型概率阈值（高于均匀基线1/45=0.022的50%，预期每场4~6个主流阵型）
N_MC      = 50     # MC Dropout 采样次数
BETA          = 0.5    # GM-TEI_AB 调制系数
FPS           = 25.0
INTENT_LABELS = ["BUILD_UP", "ATTACKING_PLAY", "HIGH_BLOCK", "MID_BLOCK", "LOW_BLOCK"]
INTENT_COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#C4AD66", "#B47CC7"]

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def make_dirs():
    """创建所有输出目录"""
    for d in [TRACKING_OUT, SHAPE_GRAPH_OUT, DATASET_OUT, MODEL_OUT, ANALYSIS_OUT]:
        d.mkdir(parents=True, exist_ok=True)


def game_tracking_path(game_id: int) -> Path:
    return TRACKING_DIR / f"{game_id}.jsonl.bz2"


def game_metadata_path(game_id: int) -> Path:
    return METADATA_DIR / f"{game_id}.json"


def game_roster_path(game_id: int) -> Path:
    return ROSTERS_DIR / f"{game_id}.json"


def game_event_path(game_id: int) -> Path:
    return EVENT_DIR / f"{game_id}.json"


def tracking_out_path(game_id: int) -> Path:
    return TRACKING_OUT / f"tracking_{game_id}.parquet"


def shape_graph_dir(game_id: int) -> Path:
    return SHAPE_GRAPH_OUT / str(game_id)


def dataset_path(game_id: int) -> Path:
    return DATASET_OUT / f"graph_dataset_{game_id}.pkl"


def model_path() -> Path:
    return MODEL_OUT / "best_model.pth"


def prototypes_path() -> Path:
    return MODEL_OUT / "b1_prototypes.pth"


def analysis_dir(game_id: int) -> Path:
    return ANALYSIS_OUT / str(game_id)
