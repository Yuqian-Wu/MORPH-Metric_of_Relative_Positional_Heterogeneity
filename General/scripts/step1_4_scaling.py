"""
Step 1.4: 空间缩放（仿射变换）
输入：data/morph_general/tracking_data_{game_id}_intent.parquet
输出：data/morph_general/tracking_data_{game_id}_scaled.parquet

对每帧每队的外场球员（非GK、非ball）做仿射缩放，主客队均处理。
GK 和 ball 保留原坐标。

用法：
  python step1_4_scaling.py --game_id 10517
  MORPH_ENV=hpc python step1_4_scaling.py --all
"""

import sys, argparse, logging
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TARGET_W = 100.0
TARGET_H = 68.0


def _scale_group(x: np.ndarray, y: np.ndarray) -> tuple:
    cx, cy = x.mean(), y.mean()
    w = max(x.max() - x.min(), 1e-6)
    h = max(y.max() - y.min(), 1e-6)
    xs = (x - cx) * (TARGET_W / w)
    ys = (y - cy) * (TARGET_H / h)
    return xs, ys


def add_scaled_positions(df: pl.DataFrame) -> pl.DataFrame:
    field = df.filter(
        (pl.col("id") != "ball") &
        (pl.col("position_name") != "goalkeeper")
    )
    other = df.filter(
        (pl.col("id") == "ball") |
        (pl.col("position_name") == "goalkeeper")
    )

    pdf = field.select(["frame_id", "team_id", "id", "x", "y"]).to_pandas()
    scaled_parts = []

    for (frame_id, team_id), grp in pdf.groupby(["frame_id", "team_id"]):
        xs, ys = _scale_group(grp["x"].values, grp["y"].values)
        tmp = grp[["frame_id", "team_id", "id"]].copy()
        tmp["x_scaled"] = xs
        tmp["y_scaled"] = ys
        scaled_parts.append(tmp)

    scaled_map = pl.from_pandas(pd.concat(scaled_parts, ignore_index=True))

    # 外场球员：join 缩放坐标
    field = field.join(
        scaled_map.select(["frame_id", "team_id", "id", "x_scaled", "y_scaled"]),
        on=["frame_id", "team_id", "id"], how="left"
    )

    # GK / ball：用原坐标
    other = other.with_columns([
        pl.col("x").alias("x_scaled"),
        pl.col("y").alias("y_scaled"),
    ])

    return pl.concat([field, other], how="diagonal").sort(["frame_id", "id"])


def process_game(game_id: int) -> bool:
    inp = C.MORPH_GENERAL / f"tracking_data_{game_id}_intent.parquet"
    out = C.MORPH_GENERAL / f"tracking_data_{game_id}_scaled.parquet"

    if out.exists():
        log.info(f"[{game_id}] 已存在，跳过")
        return True
    if not inp.exists():
        log.error(f"[{game_id}] 输入不存在: {inp}，请先运行 step1_3_fine_intent.py（超算）")
        return False

    try:
        log.info(f"[{game_id}] 空间缩放...")
        df = pl.read_parquet(inp)
        df = add_scaled_positions(df)
        df.write_parquet(out)
        log.info(f"[{game_id}] 保存 → {out}")
        return True
    except Exception as e:
        log.error(f"[{game_id}] 失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_id", type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    game_ids = C.ALL_GAME_IDS if args.all else ([args.game_id] if args.game_id else None)
    if not game_ids:
        parser.print_help(); sys.exit(1)

    ok = fail = 0
    for gid in tqdm(game_ids, desc="1.4 scaling"):
        if process_game(gid): ok += 1
        else: fail += 1
    log.info(f"完成：{ok} 成功，{fail} 失败")


if __name__ == "__main__":
    main()
