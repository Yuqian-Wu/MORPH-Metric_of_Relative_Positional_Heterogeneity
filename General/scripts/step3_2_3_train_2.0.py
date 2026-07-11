"""
Step 3.2.3: B-GNN Leave-One-Game-Out 训练
输入：morph_general/bgnn_dataset/graph_dataset_{gid}.pkl（64场）
输出：morph_general/bgnn_models/model_fold_{test_gid}.pth（64个）
       morph_general/bgnn_models/logo_summary.json

用法：
  python step3_2_3_train.py --all              # 全部64折
  python step3_2_3_train.py --game_id 10517    # 单折（测试用）
  python step3_2_3_train.py --all --workers 4  # 指定数据加载进程数
"""

import sys, argparse, logging, json, pickle, random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"设备: {device}")

MODEL_DIR = C.MORPH_GENERAL / "bgnn_models"
BGNN_DIR  = C.MORPH_GENERAL / "bgnn_dataset"


# ─────────────────────────────────────────────
# 模型定义
# ─────────────────────────────────────────────
class GraphConvModule(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, dropout=C.DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid_ch)
        self.bn1   = nn.BatchNorm1d(hid_ch)
        self.conv2 = GCNConv(hid_ch, out_ch)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.bn2(self.conv2(x, edge_index))


class GlobalFeatureFusion(nn.Module):
    def __init__(self, graph_dim, global_dim, fusion_dim):
        super().__init__()
        self.fg = nn.Linear(graph_dim, fusion_dim)
        self.fl = nn.Linear(global_dim, fusion_dim)
        self.bn = nn.BatchNorm1d(fusion_dim)

    def forward(self, graph_emb, global_feat):
        if global_feat.dim() == 3:
            global_feat = global_feat.squeeze(1)
        return F.relu(self.bn(self.fg(graph_emb) + self.fl(global_feat)))


class MCDropout(nn.Module):
    def __init__(self, p=C.MC_DROPOUT):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


class BGNN(nn.Module):
    def __init__(self, node_dim=C.NODE_DIM, global_dim=C.GLOBAL_DIM,
                 hidden_dim=C.HIDDEN_DIM, num_classes=C.NUM_CLASSES):
        super().__init__()
        self.gcm = GraphConvModule(node_dim, hidden_dim, hidden_dim)
        self.fus = GlobalFeatureFusion(hidden_dim, global_dim, hidden_dim)
        self.mcd = MCDropout()
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        ne = self.gcm(data.x, data.edge_index)
        ge = global_mean_pool(ne, data.batch)
        z  = self.fus(ge, data.global_features)
        return self.cls(self.mcd(z))

    def embed(self, data):
        """确定性嵌入（用于原型计算）"""
        self.eval()
        with torch.no_grad():
            ne = self.gcm(data.x, data.edge_index)
            ge = global_mean_pool(ne, data.batch)
            return self.fus(ge, data.global_features)

    def embed_mc(self, data):
        """MC Dropout 嵌入（推断期不确定性量化）"""
        ne = self.gcm(data.x, data.edge_index)
        ge = global_mean_pool(ne, data.batch)
        return self.mcd(self.fus(ge, data.global_features))


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────
def load_game(gid: int, num_classes: int) -> list:
    p = BGNN_DIR / f"graph_dataset_{gid}.pkl"
    if not p.exists():
        return []
    with open(p, "rb") as f:
        graphs = pickle.load(f)
    graphs = [g for g in graphs
              if not torch.isnan(g.x).any()
              and g.y_hard.item() < num_classes
              and g.y_hard.item() != 61]  # 排除 "ball" 标签
    return graphs


# ─────────────────────────────────────────────
# 单折训练
# ─────────────────────────────────────────────
def train_fold(test_gid: int, all_data: dict, num_workers: int, args, num_classes: int) -> dict:
    out_path = MODEL_DIR / f"model_fold_{test_gid}.pth"
    if out_path.exists():
        log.info(f"[fold {test_gid}] 已存在，跳过")
        return {"game_id": test_gid, "status": "skipped"}

    # 构建训练集（排除测试场次）
    train_graphs = []
    for gid, graphs in all_data.items():
        if gid != test_gid:
            train_graphs.extend(graphs)
    test_graphs = all_data.get(test_gid, [])

    if not train_graphs or not test_graphs:
        log.warning(f"[fold {test_gid}] 数据不足，跳过")
        return {"game_id": test_gid, "status": "empty"}

    random.shuffle(train_graphs)
    n_val = max(1, int(len(train_graphs) * 0.1))
    val_graphs   = train_graphs[:n_val]
    train_graphs = train_graphs[n_val:]

    tr_loader = DataLoader(train_graphs, batch_size=C.BATCH_SIZE_TRAIN,
                           shuffle=True,  num_workers=0, pin_memory=False, drop_last=True)
    va_loader = DataLoader(val_graphs,   batch_size=C.BATCH_SIZE_EVAL,
                           shuffle=False, num_workers=0, pin_memory=False)
    te_loader = DataLoader(test_graphs,  batch_size=C.BATCH_SIZE_EVAL,
                           shuffle=False, num_workers=0, pin_memory=False)

    model = BGNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.cross_entropy(out, batch.y_hard.view(-1), label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * batch.num_graphs
        tr_loss /= len(train_graphs)

        model.eval()
        va_loss = correct = total = 0
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                out = model(batch)
                va_loss += F.cross_entropy(out, batch.y_hard.view(-1),
                                           reduction="sum").item()
                correct += (out.argmax(-1) == batch.y_hard.view(-1)).sum().item()
                total   += batch.num_graphs
        va_loss /= len(val_graphs)
        va_acc   = correct / total
        scheduler.step(va_loss)

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"  [fold {test_gid}] epoch {epoch:3d} "
                     f"tr={tr_loss:.4f} va={va_loss:.4f} acc={va_acc:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            no_improve = 0
            torch.save(model.state_dict(), out_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                log.info(f"  [fold {test_gid}] early stop @ epoch {epoch}")
                break

    # 测试集评估
    model.load_state_dict(torch.load(out_path, map_location=device))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in te_loader:
            batch = batch.to(device)
            out = model(batch)
            preds.extend(out.argmax(-1).cpu().tolist())
            trues.extend(batch.y_hard.view(-1).cpu().tolist())

    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    log.info(f"[fold {test_gid}] Test Acc={acc:.4f}  Macro F1={f1:.4f}")
    return {"game_id": test_gid, "status": "done", "test_acc": acc, "macro_f1": f1}


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--game_id",  type=int)
    parser.add_argument("--epochs",   type=int, default=C.NUM_EPOCHS)
    parser.add_argument("--patience", type=int, default=C.PATIENCE)
    parser.add_argument("--workers",  type=int, default=4)
    args = parser.parse_args()

    game_ids = C.ALL_GAME_IDS if args.all else ([args.game_id] if args.game_id else None)
    if not game_ids:
        parser.print_help()
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 从 formation_mapping.json 读取实际阵型数
    mapping_path = BGNN_DIR / "formation_mapping.json"
    if not mapping_path.exists():
        log.error(f"缺少 formation_mapping.json: {mapping_path}")
        sys.exit(1)
    with open(mapping_path) as f:
        num_classes = len(json.load(f)["formation_to_idx"])
    log.info(f"num_classes = {num_classes}")

    # 一次性加载所有可用数据集到内存
    log.info("加载所有数据集...")
    all_data = {}
    for gid in tqdm(C.ALL_GAME_IDS, desc="loading"):
        graphs = load_game(gid, num_classes)
        if graphs:
            all_data[gid] = graphs
    log.info(f"已加载 {len(all_data)} 场，共 {sum(len(v) for v in all_data.values())} 个图")

    # 逐折训练
    results = []
    for test_gid in tqdm(game_ids, desc="LOGO folds"):
        if test_gid not in all_data:
            log.warning(f"[fold {test_gid}] 无数据，跳过")
            continue
        r = train_fold(test_gid, all_data, args.workers, args, num_classes)
        results.append(r)

    # 汇总
    done = [r for r in results if r["status"] == "done"]
    if done:
        accs = [r["test_acc"] for r in done]
        f1s  = [r["macro_f1"] for r in done]
        log.info(f"\n=== LOGO 汇总 ({len(done)}/{len(results)} 折完成) ===")
        log.info(f"Acc:      mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
        log.info(f"Macro F1: mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")

    summary = {
        "n_folds": len(done),
        "mean_acc": float(np.mean([r["test_acc"] for r in done])) if done else 0,
        "mean_f1":  float(np.mean([r["macro_f1"] for r in done])) if done else 0,
        "folds": results,
    }
    with open(MODEL_DIR / "logo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"汇总已保存 → {MODEL_DIR / 'logo_summary.json'}")


if __name__ == "__main__":
    main()
