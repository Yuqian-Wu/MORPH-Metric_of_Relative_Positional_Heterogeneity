"""
Step 3.1: EFPI 基准（General 版，64场）
输入：Tracking Data / Metadata / Rosters（原始数据）
输出：
  - data/efpi/efpi_{game_id}.parquet（每场 EFPI 逐帧结果）
  - data/bgnn_dataset/formation_mapping.json（全局阵型→索引映射，需在 step3_2_2 前生成）

用法：
  python step3_1_efpi.py --game_id 10517
  MORPH_ENV=hpc python step3_1_efpi.py --all
  MORPH_ENV=hpc python step3_1_efpi.py --make_mapping   # 仅生成 formation_mapping.json
"""

import sys, argparse, logging, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

import polars as pl
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EFPI_OUT = C.EFPI_OUT


def run_efpi(game_id: int) -> bool:
    out_path = EFPI_OUT / f"efpi_{game_id}.parquet"
    if out_path.exists():
        log.info(f"[{game_id}] 已存在，跳过")
        return True

    try:
        from kloppy import pff
        from unravel.soccer import KloppyPolarsDataset, EFPI

        log.info(f"[{game_id}] 加载追踪数据...")
        kloppy_ds = pff.load_tracking(
            meta_data=str(C.game_metadata_path(game_id)),
            roster_meta_data=str(C.game_roster_path(game_id)),
            raw_data=str(C.game_tracking_path(game_id)),
            coordinates="secondspectrum",
            only_alive=True,
        )
        dataset = KloppyPolarsDataset(
            kloppy_dataset=kloppy_ds,
            ball_carrier_threshold=25.0,
            max_player_speed=12.0,
            max_ball_speed=28.0,
            max_player_acceleration=6.0,
            max_ball_acceleration=13.5,
            orient_ball_owning=C.ORIENT_BALL_OWNING,
            add_smoothing=False,
        )

        log.info(f"[{game_id}] 运行 EFPI（逐帧）...")
        efpi_model = EFPI(dataset=dataset)
        efpi_model.fit(
            formations=None,
            every="frame",
            substitutions="drop",
            change_threshold=0.1,
            change_after_possession=True,
        )

        EFPI_OUT.mkdir(parents=True, exist_ok=True)
        efpi_model.output.write_parquet(out_path)
        log.info(f"[{game_id}] 保存 {len(efpi_model.output)} 行 → {out_path}")

        # ── 描述性统计 ──
        out_df = efpi_model.output
        n_frames = out_df["frame_id"].n_unique() if "frame_id" in out_df.columns else 0
        log.info(f"[{game_id}] 有效帧数={n_frames:,}")
        # 阵型分布（排除 ball 行）
        if "formation" in out_df.columns and "team_id" in out_df.columns:
            form_dist = (
                out_df.filter(pl.col("team_id") != "ball")
                .group_by("formation").agg(pl.len().alias("n"))
                .sort("n", descending=True)
                .head(10)
            )
            log.info(f"[{game_id}] Top-10 阵型分布:\n{form_dist}")

        return True

    except Exception as e:
        log.error(f"[{game_id}] 失败: {e}")
        import traceback; traceback.print_exc()
        return False


def make_formation_mapping():
    """
    扫描所有已生成的 efpi_*.parquet，收集全部唯一阵型，
    生成 data/bgnn_dataset/formation_mapping.json。
    必须在所有 EFPI 跑完后执行。
    """
    parquets = sorted(EFPI_OUT.glob("efpi_*.parquet"))
    if not parquets:
        log.error(f"未找到任何 EFPI parquet，请先运行 --all")
        return False

    formations = set()
    for p in tqdm(parquets, desc="扫描阵型"):
        df = pl.read_parquet(p, columns=["formation"])
        formations.update(df["formation"].drop_nulls().unique().to_list())

    # 排除 "ball"
    formations.discard("ball")
    formations = sorted(formations)
    formation_to_idx = {f: i for i, f in enumerate(formations)}

    C.DATASET_OUT.mkdir(parents=True, exist_ok=True)
    out = C.DATASET_OUT / "formation_mapping.json"
    with open(out, "w") as f:
        json.dump({"formation_to_idx": formation_to_idx, "n_formations": len(formations)}, f, indent=2)

    log.info(f"formation_mapping.json: {len(formations)} 种阵型 → {out}")

    # ── 全局阵型频率汇总图 ──
    try:
        import polars as _pl
        all_counts: dict = {}
        for p in parquets:
            df = _pl.read_parquet(p, columns=["team_id", "formation"])
            df = df.filter(_pl.col("team_id") != "ball")
            for row in df.group_by("formation").agg(_pl.len().alias("n")).iter_rows():
                all_counts[row[0]] = all_counts.get(row[0], 0) + row[1]
        top = sorted(all_counts.items(), key=lambda x: -x[1])[:20]
        labels, vals = zip(*top)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(labels)), vals, color="steelblue")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("帧数（所有场次合计）"); ax.set_title("Top-20 阵型频率（64场汇总）")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig_path = C.OUTPUT_ROOT / "efpi" / "formation_distribution_all.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"阵型分布图已保存 → {fig_path}")
    except Exception as e:
        log.warning(f"阵型分布图保存失败: {e}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_id", type=int)
    parser.add_argument("--all", action="store_true", help="处理全部64场")
    parser.add_argument("--make_mapping", action="store_true", help="仅生成 formation_mapping.json")
    args = parser.parse_args()

    if args.make_mapping:
        make_formation_mapping()
        return

    game_ids = C.ALL_GAME_IDS if args.all else ([args.game_id] if args.game_id else None)
    if not game_ids:
        parser.print_help(); sys.exit(1)

    ok = fail = 0
    for gid in tqdm(game_ids, desc="Step3.1 EFPI"):
        if run_efpi(gid): ok += 1
        else: fail += 1

    log.info(f"完成：{ok} 成功，{fail} 失败")

    if args.all and ok > 0:
        log.info("自动生成 formation_mapping.json ...")
        make_formation_mapping()


if __name__ == "__main__":
    main()
