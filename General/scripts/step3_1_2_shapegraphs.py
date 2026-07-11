"""
Step 3.1.2 Shape Graph 阵型识别 - HPC版
输入: morph_general/shape_graph_nodes_{gid}.parquet
输出: morph_general/shapegraphs_baseline/{gid}/sg_results_{gid}.parquet
"""
import sys, time, argparse
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

OUTPUT_DIR = C.MORPH_GENERAL / "shapegraphs_baseline"
SMOOTH_WINDOW = 5


def hull_split(pos):
    if len(pos) < 3:
        return np.median(pos[:, 0])
    try:
        return np.median(pos[ConvexHull(pos).vertices, 0])
    except Exception:
        return np.median(pos[:, 0])


def assign_vertical_levels(pos):
    N = len(pos)
    lvl = np.zeros(N, dtype=int)
    split1 = hull_split(pos)
    fwd = pos[:, 0] > split1
    bwd = ~fwd
    if fwd.sum() > 0:
        fi = np.where(fwd)[0]
        fp = pos[fwd]
        sf = hull_split(fp) if len(fp) > 1 else pos[fi[0], 0]
        for i in fi:
            lvl[i] = 4 if pos[i, 0] > sf else 3
    if bwd.sum() > 0:
        bi = np.where(bwd)[0]
        bp = pos[bwd]
        if len(bp) > 2:
            sb = hull_split(bp)
            for i in bi:
                if pos[i, 0] < sb:
                    lvl[i] = 0
                else:
                    mid = bi[pos[bi, 0] >= sb]
                    sm = np.median(pos[mid, 0]) if len(mid) > 1 else pos[i, 0]
                    lvl[i] = 2 if pos[i, 0] > sm else 1
        else:
            for k, i in enumerate(bi):
                lvl[i] = 0 if k < len(bi) // 2 else 1
    return lvl


def assign_horizontal_positions(pos, lvl):
    h = np.zeros(len(pos), dtype=int)
    for lv in range(5):
        idx = np.where(lvl == lv)[0]
        if len(idx) == 0:
            continue
        order = np.argsort(pos[idx, 1])
        n = len(order)
        for rank, orig in enumerate(order):
            h[idx[orig]] = int(rank * 4 / max(n - 1, 1))
    return h


def smooth_formations(forms, w):
    out = []
    for i in range(len(forms)):
        s, e = max(0, i - w // 2), min(len(forms), i + w // 2 + 1)
        out.append(Counter(forms[s:e]).most_common(1)[0][0])
    return out


def process_team_frames(nodes_team):
    LN = ['B', 'DM', 'M', 'AM', 'F']
    HN = ['L', 'LC', 'C', 'RC', 'R']
    frame_ids = nodes_team['frame_id'].unique().sort().to_list()
    rows = []
    for fid in frame_ids:
        fn = nodes_team.filter(pl.col('frame_id') == fid)
        pos = fn.select(['x', 'y']).to_numpy()
        if len(pos) < 3:
            continue
        lvl = assign_vertical_levels(pos)
        hp = assign_horizontal_positions(pos, lvl)
        lc = np.bincount(lvl, minlength=5)
        rows.append({
            'frame_id': fid,
            'formation': f'{lc[0]}-{lc[1]+lc[2]+lc[3]}-{lc[4]}',
            'formation_detailed': ''.join(map(str, lc)),
            'n_defenders': int(lc[0]),
            'n_dm': int(lc[1]),
            'n_midfielders': int(lc[2]),
            'n_am': int(lc[3]),
            'n_forwards': int(lc[4]),
            'vertical_levels': [LN[l] for l in lvl],
            'horizontal_positions': [HN[h] for h in hp],
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values('frame_id').reset_index(drop=True)
    df['formation_smoothed'] = smooth_formations(df['formation'].tolist(), SMOOTH_WINDOW)
    df['formation_detailed_smoothed'] = smooth_formations(df['formation_detailed'].tolist(), SMOOTH_WINDOW)
    return df


def process_single_game(gid):
    out_file = OUTPUT_DIR / str(gid) / f'sg_results_{gid}.parquet'
    if out_file.exists():
        return {'game_id': gid, 'status': 'skipped'}

    nodes_f = C.MORPH_GENERAL / f'shape_graph_nodes_{gid}.parquet'
    if not nodes_f.exists():
        return {'game_id': gid, 'status': 'missing', 'reason': 'nodes parquet不存在'}

    try:
        nodes_all = pl.read_parquet(nodes_f)
        has_team_side = 'team_side' in nodes_all.columns
        sides = ['home', 'away'] if has_team_side else [None]

        out_dir = OUTPUT_DIR / str(gid)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_dfs = []
        for side in sides:
            nodes = nodes_all.filter(pl.col('team_side') == side) if side else nodes_all
            df = process_team_frames(nodes)
            if df is None:
                continue
            if side:
                df.insert(1, 'team_side', side)
            all_dfs.append(df)

        if not all_dfs:
            return {'game_id': gid, 'status': 'empty', 'reason': '无有效帧'}

        result_df = pd.concat(all_dfs, ignore_index=True)
        result_df.to_parquet(out_file)
        return {
            'game_id': gid,
            'status': 'done',
            'n_frames': result_df['frame_id'].nunique(),
            'top_formation': result_df['formation_smoothed'].mode()[0],
        }
    except Exception as e:
        return {'game_id': gid, 'status': 'error', 'reason': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--game_id', type=int)
    parser.add_argument('--workers', type=int, default=min(8, cpu_count()))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        games = C.ALL_GAME_IDS
    elif args.game_id:
        games = [args.game_id]
    else:
        parser.print_help()
        sys.exit(1)

    todo = [g for g in games if not (OUTPUT_DIR / str(g) / f'sg_results_{g}.parquet').exists()]
    print(f'总场次: {len(games)}，待处理: {len(todo)}，workers: {args.workers}')

    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(process_single_game, todo), 1):
            elapsed = time.time() - t0
            eta_h = elapsed / idx * (len(todo) - idx) / 3600 if idx < len(todo) else 0
            if result['status'] == 'done':
                print(f"  OK [{idx}/{len(todo)}] {result['game_id']}: {result['n_frames']:,}帧 主阵型={result['top_formation']} eta={eta_h:.1f}h")
            elif result['status'] == 'skipped':
                print(f"  SKIP [{idx}/{len(todo)}] {result['game_id']}")
            else:
                print(f"  FAIL [{idx}/{len(todo)}] {result['game_id']}: {result.get('reason')}")

    print(f'完成！总用时: {(time.time()-t0)/3600:.2f}h')


if __name__ == '__main__':
    main()
