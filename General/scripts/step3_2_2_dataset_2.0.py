"""
Step 3.2.2: 图数据集构建 - General版（适配 parquet 格式）v2.0
v2.0 改动：
  [改动1] 阵型合并：16个稀少/短暂阵型并入最近语义邻居，类别数61→~45
  [改动2] 分层降采样：按阵型类别均匀采样，替代纯随机采样
  [改动5] 清理无效全局特征：删除恒零维度，global_dim 24→17

输入：
  morph_general/tracking_data_{gid}_scaled.parquet
  morph_general/shape_graph_nodes_{gid}.parquet
  morph_general/shape_graph_edges_{gid}.parquet
  morph_general/efpi_baseline/{gid}/efpi_results_{gid}.parquet
  morph_general/shapegraphs_baseline/{gid}/sg_results_{gid}.parquet
输出：
  morph_general/bgnn_dataset/graph_dataset_{gid}.pkl

节点特征 41维：[x_scaled,y_scaled,vx,vy,dist_ball](5)+roster_pos(25)+vert(5)+horiz(5)+closest(1)
全局特征 24维：macro(2)+intent(5)+score/time(2)+ranking(1)+odds(3)+centroid(2)+spread/diam(2)+geom(7)
"""

import sys, argparse, json, pickle, logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch_geometric.data import Data
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = C.MORPH_GENERAL / "bgnn_dataset"

VERT_LEVELS  = ["B", "DM", "M", "AM", "F"]
HORIZ_POS    = ["L", "LC", "C", "RC", "R"]
FINE_INTENTS = ["BUILD_UP", "ATTACKING_PLAY", "HIGH_BLOCK", "MID_BLOCK", "LOW_BLOCK"]

ROSTER_POSITION_MAP = {
    "GK":  [(2,2,1.0)],
    "LCB": [(0,1,1.0)], "RCB": [(0,3,1.0)], "MCB": [(0,2,1.0)],
    "LB":  [(0,0,1.0)], "RB":  [(0,4,1.0)],
    "LWB": [(1,0,0.7),(0,0,0.3)], "RWB": [(1,4,0.7),(0,4,0.3)],
    "DM":  [(1,2,1.0)], "CM":  [(2,2,1.0)], "AM":  [(3,2,1.0)],
    "LW":  [(3,0,1.0)], "RW":  [(3,4,1.0)], "CF":  [(4,2,1.0)],
}

# [改动1] 合并16个稀少/短暂阵型到最近语义邻居（数据驱动：平均帧<10k或出现场次<50）
FORMATION_MERGE_MAP = {
    "3421flat": "3421",  "31312": "31213",  "3322": "3232",   "4131": "4141",
    "531":      "532",   "4212":  "42121",  "342":  "3421",   "4221": "4231",
    "432":      "4321",  "312112":"31213",  "3411": "3412",   "351":  "352",
    "441":      "442",   "4311":  "4312",   "422":  "4222",   "341":  "3421",
}


def _onehot(val, cats):
    v = np.zeros(len(cats), dtype=np.float32)
    if val in cats:
        v[cats.index(val)] = 1.0
    return v


def encode_roster_position(pos):
    mat = np.zeros((5, 5), dtype=np.float32)
    for r, c, w in ROSTER_POSITION_MAP.get(pos, []):
        mat[r, c] = w
    return mat.flatten()


ATK_INTENTS = {"ATTACKING_PLAY", "BUILD_UP"}

def encode_intent(row: dict) -> np.ndarray:
    for col in ("attack_intent_home", "attack_intent_away"):
        v = row.get(col)
        if v in FINE_INTENTS:
            return _onehot(v, FINE_INTENTS)
    for col in ("defense_intent_home", "defense_intent_away"):
        v = row.get(col)
        if v in FINE_INTENTS:
            return _onehot(v, FINE_INTENTS)
    return _onehot("LOW_BLOCK", FINE_INTENTS)


def encode_macro(row: dict) -> np.ndarray:
    is_atk = any(row.get(c) in ATK_INTENTS
                 for c in ("attack_intent_home", "attack_intent_away"))
    return np.array([1.0 if is_atk else 0.0], dtype=np.float32)


def compute_advanced_geom(positions: np.ndarray) -> dict:
    x, y = positions[:, 0], positions[:, 1]
    length  = max(x.max() - x.min(), 1e-6)
    width   = max(y.max() - y.min(), 1e-6)
    compact = length * width
    hull_area = lr = rect = 0.0
    if len(positions) >= 3:
        try:
            hull = ConvexHull(positions)
            hull_area = hull.volume
            rect = hull_area / compact
            inner = positions[[i for i in range(len(positions)) if i not in hull.vertices]]
            if len(inner) >= 3:
                lr = ConvexHull(inner).volume / hull_area
        except Exception:
            pass
    return dict(lpw=length/width, hull=hull_area, compact=compact,
                dlh=float(abs(x.min())), hpl=float(abs(x.max())), lr=lr, rect=rect)


def build_formation_mapping(game_ids):
    formations = set()
    for gid in game_ids:
        f = C.MORPH_GENERAL / f"efpi_baseline/{gid}/efpi_results_{gid}.parquet"
        if f.exists():
            vals = pl.read_parquet(f)["formation"].drop_nulls().unique().to_list()
            formations.update(v for v in vals if v != "ball")
    # [改动1] 应用合并映射，稀少阵型归并到稳定类
    formations = {FORMATION_MERGE_MAP.get(f, f) for f in formations}
    formations = sorted(formations)
    f2i = {f: i for i, f in enumerate(formations)}
    return f2i, {i: f for f, i in f2i.items()}


def build_graph_for_frame(frame_id, team_side, nodes_df, edges_df,
                           tracking_pd, efpi_idx, sg_idx,
                           player_pos_map, formation_to_idx, team_id):
    n = nodes_df.filter(
        (pl.col("frame_id") == frame_id) & (pl.col("team_side") == team_side)
    )
    e = edges_df.filter(
        (pl.col("frame_id") == frame_id) & (pl.col("team_side") == team_side)
    )
    if len(n) < 3 or len(e) == 0:
        return None

    positions = n.select(["x", "y"]).to_numpy()
    N = len(positions)

    ef = efpi_idx.get(frame_id)
    if ef is None:
        return None
    ef = ef[ef["team_id"] == team_id]
    if len(ef) == 0:
        return None
    formation = ef.iloc[0]["formation"]
    formation = FORMATION_MERGE_MAP.get(formation, formation)  # [改动1] 应用合并
    if formation not in formation_to_idx:
        return None
    y_hard = formation_to_idx[formation]

    sg = sg_idx.get((frame_id, team_side))
    if sg is None or len(sg) == 0:
        return None
    vert_levels  = sg.iloc[0]["vertical_levels"]
    horiz_levels = sg.iloc[0]["horizontal_positions"]
    if len(vert_levels) != N or len(horiz_levels) != N:
        return None

    ball = tracking_pd[tracking_pd["id"] == "ball"]
    ball_x = float(ball["x_scaled"].values[0]) if len(ball) > 0 else 0.0
    ball_y = float(ball["y_scaled"].values[0]) if len(ball) > 0 else 0.0
    players = tracking_pd[tracking_pd["id"] != "ball"].reset_index(drop=True)

    if len(players) > 0:
        tr_xy = players[["x", "y"]].to_numpy()
        matched_rows = []
        for i in range(N):
            dists_to_players = np.linalg.norm(tr_xy - positions[i], axis=1)
            matched_rows.append(players.iloc[int(dists_to_players.argmin())].to_dict())
    else:
        matched_rows = [{} for _ in range(N)]

    scaled_xy = np.array([
        [float(r.get("x_scaled") or 0), float(r.get("y_scaled") or 0)]
        for r in matched_rows
    ], dtype=np.float32)

    node_feats = []
    for i in range(N):
        row = matched_rows[i]
        x_s = scaled_xy[i, 0]
        y_s = scaled_xy[i, 1]
        vx = float(row.get("vx") or 0)
        vy = float(row.get("vy") or 0)
        dist_ball = float(np.sqrt((x_s - ball_x)**2 + (y_s - ball_y)**2))
        pid = str(row.get("id", ""))
        pos_group = player_pos_map.get(pid, {}).get("positionGroupType", "Unknown")
        node_feats.append(np.concatenate([
            [x_s, y_s, vx, vy, dist_ball],
            encode_roster_position(pos_group),
            _onehot(vert_levels[i],  VERT_LEVELS),
            _onehot(horiz_levels[i], HORIZ_POS),
        ]))

    dmat = cdist(scaled_xy, scaled_xy)
    np.fill_diagonal(dmat, np.inf)
    closest = dmat.min(axis=1)
    node_feats = np.concatenate(
        [np.array(node_feats, dtype=np.float32), closest[:, None]], axis=1
    )

    src   = e["src"].to_list()
    dst   = e["dst"].to_list()
    dists = e["distance"].to_list()
    max_d = max(dists) if dists else 1.0
    edges = [[s, d] for s, d in zip(src, dst)] + [[d, s] for s, d in zip(src, dst)]
    eattr = [[d, d/max_d] for d in dists] * 2
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(eattr, dtype=torch.float32)

    row0     = matched_rows[0] if matched_rows else {}
    geom     = compute_advanced_geom(scaled_xy)
    centroid = scaled_xy.mean(axis=0)
    spread   = float(np.sqrt(((scaled_xy - centroid)**2).sum(axis=1).mean()))
    diameter = float(dmat[dmat != np.inf].max()) if (dmat != np.inf).any() else 0.0

    global_feat = np.concatenate([
        encode_macro(row0),
        encode_intent(row0),
        centroid,
        [spread, diameter],
        [geom["lpw"], geom["hull"], geom["compact"],
         geom["dlh"], geom["hpl"], geom["lr"], geom["rect"]],
    ], dtype=np.float32)

    return Data(
        x=torch.tensor(node_feats, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_hard=torch.tensor([y_hard], dtype=torch.long),
        global_features=torch.tensor(global_feat, dtype=torch.float32).unsqueeze(0),
        frame_id=frame_id,
        team_side=team_side,
        num_nodes=N,
    )


def process_game(gid: int, formation_to_idx: dict) -> dict:
    out_pkl = OUTPUT_DIR / f"graph_dataset_{gid}.pkl"
    if out_pkl.exists():
        return {"game_id": gid, "status": "skipped"}

    files = {
        "scaled": C.MORPH_GENERAL / f"tracking_data_{gid}_scaled.parquet",
        "nodes":  C.MORPH_GENERAL / f"shape_graph_nodes_{gid}.parquet",
        "edges":  C.MORPH_GENERAL / f"shape_graph_edges_{gid}.parquet",
        "efpi":   C.MORPH_GENERAL / f"efpi_baseline/{gid}/efpi_results_{gid}.parquet",
        "sg":     C.MORPH_GENERAL / f"shapegraphs_baseline/{gid}/sg_results_{gid}.parquet",
    }
    for k, f in files.items():
        if not f.exists():
            return {"game_id": gid, "status": "missing", "reason": k}

    try:
        tracking = pl.read_parquet(files["scaled"])
        nodes_df = pl.read_parquet(files["nodes"])
        edges_df = pl.read_parquet(files["edges"])
        efpi_pd  = pl.read_parquet(files["efpi"]).to_pandas()
        sg_pd    = pd.read_parquet(files["sg"])

        with open(C.game_roster_path(gid)) as fp:
            roster = json.load(fp)
        player_pos_map = {p["player"]["id"]: {"positionGroupType": p["positionGroupType"]}
                          for p in roster}

        with open(C.game_metadata_path(gid)) as fp:
            meta = json.load(fp)
        meta = meta[0] if isinstance(meta, list) else meta
        team_id_map = {
            "home": str(meta["homeTeam"]["id"]),
            "away": str(meta["awayTeam"]["id"]),
        }

        efpi_pd["frame_id"] = efpi_pd["frame_id"].astype(int)
        sg_pd["frame_id"]   = sg_pd["frame_id"].astype(int)
        efpi_idx = {int(fid): grp for fid, grp in efpi_pd.groupby("frame_id")}
        sg_idx   = {(int(fid), side): grp
                    for (fid, side), grp in sg_pd.groupby(["frame_id", "team_side"])}
        tracking_idx = {int(fid[0]) if isinstance(fid, tuple) else int(fid): grp.to_pandas()
                        for fid, grp in tracking.group_by("frame_id")}

        valid_frames = sorted(
            set(int(f) for f in nodes_df["frame_id"].unique().to_list()) &
            set(efpi_idx.keys())
        )

        dataset = []
        for fid in valid_frames:
            tr_pd = tracking_idx.get(fid)
            if tr_pd is None:
                continue
            for side in ["home", "away"]:
                data = build_graph_for_frame(
                    fid, side, nodes_df, edges_df,
                    tr_pd, efpi_idx, sg_idx,
                    player_pos_map, formation_to_idx, team_id_map[side],
                )
                if data is not None:
                    dataset.append(data)

        if not dataset:
            return {"game_id": gid, "status": "empty"}

        # [v2.0 改动2] 分层降采样：按阵型类别均匀采样，保证稀有类不被随机丢弃
        import random as _random
        _random.seed(42)
        MAX_PER_GAME = 20000
        label_groups = defaultdict(list)
        for g in dataset:
            label_groups[g.y_hard.item()].append(g)
        quota = max(1, MAX_PER_GAME // len(label_groups)) if label_groups else MAX_PER_GAME
        sampled = []
        for graphs_k in label_groups.values():
            sampled.extend(_random.sample(graphs_k, min(len(graphs_k), quota)))
        if len(sampled) > MAX_PER_GAME:
            _random.shuffle(sampled)
            sampled = sampled[:MAX_PER_GAME]
        dataset = sampled
        log.info(f"[{gid}] 分层采样：{len(label_groups)} 类，配额/类={quota}，共 {len(dataset)} 样本")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_pkl, "wb") as fp:
            pickle.dump(dataset, fp)

        return {"game_id": gid, "status": "done", "n_samples": len(dataset)}

    except Exception as ex:
        log.error(f"[{gid}] 失败: {ex}")
        return {"game_id": gid, "status": "error", "reason": str(ex)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--game_id", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    game_ids = C.ALL_GAME_IDS if args.all else ([args.game_id] if args.game_id else None)
    if not game_ids:
        parser.print_help()
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("构建全局阵型映射...")
    formation_to_idx, _ = build_formation_mapping(C.ALL_GAME_IDS)
    log.info(f"阵型总数: {len(formation_to_idx)}")

    with open(OUTPUT_DIR / "formation_mapping.json", "w") as fp:
        json.dump({"formation_to_idx": formation_to_idx}, fp, indent=2)

    ok = fail = 0
    for gid in tqdm(game_ids, desc="3.2.2 dataset v2.0"):
        r = process_game(gid, formation_to_idx)
        if r["status"] in ("done", "skipped"):
            ok += 1
        else:
            fail += 1
            log.warning(f"[{gid}] {r['status']}: {r.get('reason', '')}")

    log.info(f"完成：{ok} 成功/跳过，{fail} 失败")


if __name__ == "__main__":
    main()
