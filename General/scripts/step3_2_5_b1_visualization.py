"""
Step 3.2.5: B1 静态可视化 v2.0（适配 home/away 分离输出）
输入（per game per side，来自 step3_2_4 v2.0）：
  - morph_general/bgnn_analysis/{game_id}/b1_window_distributions_{side}.parquet
  - morph_general/bgnn_analysis/{game_id}/b1_mainstream_result_{side}.json
  - morph_general/bgnn_analysis/{game_id}/b1_frame_tei_{side}.parquet
输出（per game per side）：
  - b1_formation_heatmap_{game_id}_{side}.png
  - b1_prob_overview_{game_id}_{side}.png
  - b1_frame_tei_dist_{game_id}_{side}.png

用法：
  python step3_2_5_b1_visualization.py --game_id 10517
  python step3_2_5_b1_visualization.py --game_id 3812 3820 10517
  MORPH_ENV=hpc python step3_2_5_b1_visualization.py --all
"""

import sys, argparse, logging, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ANALYSIS_DIR = C.MORPH_GENERAL / "bgnn_analysis"


def process_game(game_id: int) -> bool:
    out_dir = ANALYSIS_DIR / str(game_id)
    if not out_dir.exists():
        log.error(f"[{game_id}] 分析目录不存在，请先运行 step3_2_4")
        return False

    ok = True
    for team_side in ["home", "away"]:
        win_path = out_dir / f"b1_window_distributions_{team_side}.parquet"
        res_path = out_dir / f"b1_mainstream_result_{team_side}.json"
        tei_path = out_dir / f"b1_frame_tei_{team_side}.parquet"

        if not win_path.exists() or not res_path.exists():
            log.warning(f"[{game_id}/{team_side}] 缺少 b1 推断结果，跳过")
            ok = False
            continue

        try:
            b1_win = pd.read_parquet(win_path)
            with open(res_path, encoding="utf-8") as f:
                b1_res = json.load(f)

            available_forms = b1_res.get("available_forms", [])
            mainstream      = b1_res.get("mainstream", [])

            if not any(f"prob_{f}" in b1_win.columns for f in available_forms):
                log.warning(f"[{game_id}/{team_side}] 无概率列，跳过")
                continue

            center_fids = b1_win["center_fid"].values
            n_win = len(b1_win)
            mean_probs = {f: b1_win[f"prob_{f}"].mean() for f in available_forms if f"prob_{f}" in b1_win.columns}
            top20 = sorted(mean_probs, key=lambda x: -mean_probs[x])[:20]
            prob_mat = np.array([b1_win[f"prob_{f}"].values for f in top20])

            # ── 图1：阵型概率热图 ─────────────────────────────────────────────
            fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]})
            im = axes[0].imshow(prob_mat, aspect="auto", cmap="YlOrRd",
                                vmin=0, vmax=prob_mat.max(),
                                extent=[0, n_win, len(top20)-0.5, -0.5])
            axes[0].set_yticks(range(len(top20)))
            axes[0].set_yticklabels(top20, fontsize=8)
            axes[0].set_xlabel("Window index")
            axes[0].set_ylabel("Formation")
            axes[0].set_title(f"B1 Formation heatmap Top-20  game={game_id}  [{team_side}]")
            plt.colorbar(im, ax=axes[0], label="P(formation)")
            for i, f in enumerate(top20):
                if f in mainstream:
                    axes[0].text(n_win + 0.5, i, "*", fontsize=8, color="red", va="center")
            if "tei" in b1_win.columns:
                axes[1].plot(range(n_win), b1_win["tei"].values, color="steelblue", lw=1.0)
                axes[1].set_ylabel("TEI"); axes[1].set_xlabel("Window index")
                axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            fp = out_dir / f"b1_formation_heatmap_{game_id}_{team_side}.png"
            plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
            log.info(f"[{game_id}/{team_side}] 热图 → {fp}")

            log.info(f"[{game_id}/{team_side}] 窗口数={n_win}  主流阵型={mainstream}")
            for f in top20[:5]:
                log.info(f"  {f}: {mean_probs[f]:.4f}")

            # ── 图3：帧级 TEI 分布 ────────────────────────────────────────────
            if tei_path.exists():
                frame_tei = pd.read_parquet(tei_path)
                fig, axes3 = plt.subplots(1, 2, figsize=(12, 4))
                axes3[0].hist(frame_tei["tei"].values, bins=60, color="steelblue",
                              alpha=0.75, edgecolor="white", lw=0.3)
                axes3[0].axvline(frame_tei["tei"].mean(), color="red", lw=1.5, ls="--",
                                 label=f"mean={frame_tei['tei'].mean():.3f}")
                axes3[0].set_xlabel("TEI"); axes3[0].set_ylabel("Frames")
                axes3[0].set_title(f"Frame TEI dist  game={game_id}  [{team_side}]")
                axes3[0].legend(fontsize=9); axes3[0].grid(True, alpha=0.3)
                top1_counts = frame_tei["top1_formation"].value_counts().head(10)
                axes3[1].bar(range(len(top1_counts)), top1_counts.values, color="steelblue", alpha=0.8)
                axes3[1].set_xticks(range(len(top1_counts)))
                axes3[1].set_xticklabels(top1_counts.index, rotation=45, ha="right", fontsize=8)
                axes3[1].set_ylabel("Frames"); axes3[1].set_title("Top-10 frame top-1 formation")
                axes3[1].grid(True, axis="y", alpha=0.3)
                plt.tight_layout()
                fp3 = out_dir / f"b1_frame_tei_dist_{game_id}_{team_side}.png"
                plt.savefig(fp3, dpi=150, bbox_inches="tight"); plt.close()
                log.info(f"[{game_id}/{team_side}] 帧级 TEI 分布 → {fp3}")

        except Exception as e:
            log.error(f"[{game_id}/{team_side}] 失败: {e}")
            import traceback; traceback.print_exc()
            ok = False

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_id", type=int, nargs="+")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    game_ids = C.ALL_GAME_IDS if args.all else (args.game_id if args.game_id else None)
    if not game_ids:
        parser.print_help(); sys.exit(1)

    ok = fail = 0
    for gid in tqdm(game_ids, desc="3.2.5 B1 visualization"):
        if process_game(gid): ok += 1
        else: fail += 1
    log.info(f"完成：{ok} 成功，{fail} 失败")


if __name__ == "__main__":
    main()
