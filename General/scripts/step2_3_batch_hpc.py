# -*- coding: utf-8 -*-
"""
Step 2.3: Shape Graph 批量处理（超算版）- 新格式，双队覆盖

输入：data/morph_general/tracking_data_{gid}_scaled.parquet
      data/morph_general/metadata_{gid}.json
输出：data/morph_general/shape_graph_nodes_{gid}.parquet   ← 节点坐标（每帧×每队×每节点）
      data/morph_general/shape_graph_edges_{gid}.parquet   ← 边列表（每帧×每队×每边）
      data/morph_general/step2_3_summary.csv

列说明：
  nodes: frame_id, team_side('home'/'away'), node_idx, x, y,
         n_players, n_removed, n_initial, n_edges
  edges: frame_id, team_side('home'/'away'), src, dst, distance

用法：
  python step2_3_batch_hpc.py --all              # 处理全部64场
  python step2_3_batch_hpc.py --game_id 3813     # 处理单场
  python step2_3_batch_hpc.py --workers 16       # 指定进程数（默认16）

超算运行（SLURM）：
  sbatch slurm_step2_3.sh
"""

import sys, argparse, logging, heapq, json, time
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
from scipy.spatial import Delaunay

# ─── 路径配置 ───
HPC_HOME = Path("/public/home/hpc242111131")
DATA_DIR = HPC_HOME / "G-TAF/MORPH/data/morph_general"

ALPHA_THRESHOLD = 3 * np.pi / 4   # 135°

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(DATA_DIR / "step2_3_batch.log"), mode='a'),
    ]
)
log = logging.getLogger(__name__)

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


# ─────────────────────────────────────────────
# Shape Graph 核心算法
# ─────────────────────────────────────────────
def _angle(p, q, r):
    v1, v2 = p - r, q - r
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def _opposing_angle(p_idx, q_idx, face, positions):
    others = [v for v in face if v != p_idx and v != q_idx]
    if not others:
        return 0.0
    return min(_angle(positions[p_idx], positions[q_idx], positions[r]) for r in others)


def build_shape_graph(positions, alpha=ALPHA_THRESHOLD):
    tri = Delaunay(positions)
    edge_to_faces = defaultdict(list)
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            edge_to_faces[edge].append(list(simplex))

    n_initial = len(edge_to_faces)
    pq = []
    for edge, faces in edge_to_faces.items():
        p, q = edge
        a_pq = _opposing_angle(p, q, faces[0], positions)
        a_qp = _opposing_angle(q, p, faces[1], positions) if len(faces) >= 2 else 0.0
        heapq.heappush(pq, (-(a_pq + a_qp), edge))

    n_removed = 0
    while pq:
        neg_s, edge = heapq.heappop(pq)
        if -neg_s <= alpha:
            break
        if edge in edge_to_faces:
            del edge_to_faces[edge]
            n_removed += 1

    # 只返回边列表，不建 NetworkX 对象（节省内存）
    edges = list(edge_to_faces.keys())
    return edges, n_removed, n_initial


# ─────────────────────────────────────────────
# 守门员识别
# ─────────────────────────────────────────────
def identify_goalkeeper_simple(tracking_pl, team_id, frame_id):
    team_data = tracking_pl.filter(
        (pl.col('team_id') == team_id) & (pl.col('id').is_not_null())
    )
    gk = team_data.filter(pl.col('position_name') == 'GK')
    if len(gk) > 0:
        return gk['id'][0]
    frame_data = tracking_pl.filter(
        (pl.col('frame_id') == frame_id) &
        (pl.col('team_id') == team_id) &
        (pl.col('id').is_not_null())
    )
    if frame_data.is_empty():
        return None
    pdf = frame_data.to_pandas()
    mean_x = pdf['x'].mean()
    if mean_x >= 0:
        return pdf.loc[pdf['x'].idxmin(), 'id']
    else:
        return pdf.loc[pdf['x'].idxmax(), 'id']


# ─────────────────────────────────────────────
# 单队单帧 Shape Graph 构建（内部辅助函数）
# ─────────────────────────────────────────────
def _process_team_frame(tracking_pl, team_id, team_side, fid, node_rows, edge_rows):
    """
    为单场比赛单帧单队构建 Shape Graph，结果追加到 node_rows / edge_rows。
    返回 True 表示成功，False 表示跳过（球员不足）。
    """
    gk_id = identify_goalkeeper_simple(tracking_pl, team_id, fid)
    frame_data = tracking_pl.filter(
        (pl.col('frame_id') == fid) &
        (pl.col('team_id') == team_id) &
        (pl.col('id').is_not_null()) &
        (pl.col('id') != gk_id)
    )
    if len(frame_data) < 3:
        return False

    positions = frame_data.select(['x', 'y']).to_numpy()
    edges, n_removed, n_initial = build_shape_graph(positions)

    # 节点行
    for node_idx, (x, y) in enumerate(positions):
        node_rows.append({
            'frame_id':  fid,
            'team_side': team_side,       # 'home' 或 'away'
            'node_idx':  node_idx,
            'x':         float(x),
            'y':         float(y),
            'n_players': len(positions),
            'n_removed': n_removed,
            'n_initial': n_initial,
            'n_edges':   len(edges),
        })

    # 边行
    for (src, dst) in edges:
        dist = float(np.linalg.norm(positions[src] - positions[dst]))
        edge_rows.append({
            'frame_id':  fid,
            'team_side': team_side,       # 'home' 或 'away'
            'src':       int(src),
            'dst':       int(dst),
            'distance':  dist,
        })

    return True


# ─────────────────────────────────────────────
# 单场处理（多进程 worker）
# ─────────────────────────────────────────────
def process_game_worker(gid):
    sf = DATA_DIR / f"tracking_data_{gid}_scaled.parquet"
    mf = DATA_DIR / f"metadata_{gid}.json"

    nodes_out = DATA_DIR / f"shape_graph_nodes_{gid}.parquet"
    edges_out = DATA_DIR / f"shape_graph_edges_{gid}.parquet"

    if not sf.exists() or not mf.exists():
        return {'game_id': gid, 'status': 'error', 'reason': '文件不存在'}

    try:
        tracking_pl = pl.read_parquet(sf)
        with open(mf) as f:
            meta = json.load(f)
        home_team_id = str(meta['home_team_id'])
        away_team_id = str(meta['away_team_id'])
    except Exception as e:
        return {'game_id': gid, 'status': 'error', 'reason': str(e)}

    all_frames = tracking_pl['frame_id'].unique().sort().to_list()
    node_rows = []
    edge_rows = []
    n_failed = 0

    for fid in all_frames:
        try:
            # 主队
            ok_home = _process_team_frame(
                tracking_pl, home_team_id, 'home', fid, node_rows, edge_rows
            )
            # 客队
            ok_away = _process_team_frame(
                tracking_pl, away_team_id, 'away', fid, node_rows, edge_rows
            )
            if not ok_home and not ok_away:
                n_failed += 1
        except Exception:
            n_failed += 1
            continue

    if not node_rows:
        return {'game_id': gid, 'status': 'empty',
                'n_frames_total': len(all_frames), 'n_frames_ok': 0}

    # 保存两张 parquet
    pd.DataFrame(node_rows).to_parquet(nodes_out, index=False)
    pd.DataFrame(edge_rows).to_parquet(edges_out, index=False)

    # 统计成功帧数（主队或客队至少有一个成功即算）
    ok_frames = set(r['frame_id'] for r in node_rows)
    return {
        'game_id':        gid,
        'status':         'done',
        'n_frames_total': len(all_frames),
        'n_frames_ok':    len(ok_frames),
        'n_frames_failed': n_failed,
        'n_node_rows':    len(node_rows),
        'n_edge_rows':    len(edge_rows),
    }


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
def get_todo_games(game_ids):
    """过滤已完成的场次（两张 parquet 都存在）"""
    todo = []
    for gid in game_ids:
        nodes_out = DATA_DIR / f"shape_graph_nodes_{gid}.parquet"
        edges_out = DATA_DIR / f"shape_graph_edges_{gid}.parquet"
        if not (nodes_out.exists() and edges_out.exists()):
            todo.append(gid)
    return todo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='处理全部64场')
    parser.add_argument('--game_id', type=int, help='处理单场')
    parser.add_argument('--workers', type=int, default=16, help='并行进程数（默认16）')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='跳过已完成场次（默认True）')
    args = parser.parse_args()

    if args.all:
        game_ids = ALL_GAME_IDS
    elif args.game_id:
        game_ids = [args.game_id]
    else:
        parser.print_help()
        sys.exit(1)

    todo = get_todo_games(game_ids) if args.skip_existing else game_ids
    log.info(f"总场次: {len(game_ids)}，待处理: {len(todo)}，已完成: {len(game_ids)-len(todo)}")

    if not todo:
        log.info("所有场次已完成，退出。")
        return

    n_workers = min(args.workers, mp.cpu_count(), len(todo))
    log.info(f"使用 {n_workers} 个进程处理 {len(todo)} 场（主队+客队双队模式）...")

    t0 = time.time()
    all_results = []

    with mp.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_game_worker, todo), 1):
            all_results.append(result)
            elapsed = time.time() - t0
            eta_h = elapsed / i * (len(todo) - i) / 3600
            log.info(
                f"[{i:02d}/{len(todo)}] {result['game_id']}: {result['status']} | "
                f"ok={result.get('n_frames_ok',0):,} frames | "
                f"elapsed={elapsed/3600:.1f}h eta={eta_h:.1f}h"
            )

    total_h = (time.time() - t0) / 3600
    log.info(f"完成！总用时: {total_h:.2f}h")

    summary_path = DATA_DIR / 'step2_3_summary.csv'
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    log.info(f"汇总已保存: {summary_path}")


if __name__ == '__main__':
    main()
