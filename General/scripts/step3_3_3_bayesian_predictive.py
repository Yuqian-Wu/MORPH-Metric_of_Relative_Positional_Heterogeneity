"""
Step 3.3.3: 贝叶斯不确定性预测有效性
输出目录：morph_general/bgnn_analysis/{game_id}/3.3.3_bayesian_predictive/

评估角度：
  ② 不确定性的预测有效性（CI宽度 → 下一窗口阵型切换率）
  ⑦ Epistemic vs Aleatoric 四象限分析

核心命题：证明贝叶斯不确定性是有预测价值的信号，不是随机噪声

输出（per game per side）：
  - uncertainty_predictive_{gid}_{side}.png  ② CI四分位 vs 下窗口切换率
  - epistemic_aleatoric_{gid}_{side}.png     ⑦ 四象限散点图 + 各象限TEI分布
  - bayesian_predictive_summary_{gid}_{side}.json  统计数值

用法：
  python step3_3_3_bayesian_predictive.py --game_id 10517
  python step3_3_3_bayesian_predictive.py --game_id 3812 3820 10517
  MORPH_ENV=hpc python step3_3_3_bayesian_predictive.py --all
"""

import sys, argparse, json, logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ANALYSIS_DIR = C.MORPH_GENERAL / "bgnn_analysis"
SUB_DIR      = "3.3.3_bayesian_predictive"


# ─────────────────────────────────────────────
# ② 不确定性预测有效性
# ─────────────────────────────────────────────
def _plot_uncertainty_predictive(game_id, team_side, sub_dir, b1_win, summary):
    try:
        probvar_cols = [c for c in b1_win.columns if c.startswith("probvar_")]
        if not probvar_cols:
            return
        df = b1_win.copy()
        df["ci_half"] = df[probvar_cols].clip(lower=0).pow(0.5).mean(axis=1) * 1.96
        prob_cols = [c for c in df.columns if c.startswith("prob_") and not c.startswith("probvar_")]
        df["top1_form"] = df[prob_cols].idxmax(axis=1)
        df["next_top1"]  = df["top1_form"].shift(-1)
        df["formation_changed"] = (df["top1_form"] != df["next_top1"]).astype(float)
        df = df.dropna(subset=["ci_half", "formation_changed"])

        df["ci_quartile"] = pd.qcut(df["ci_half"], q=4,
                                     labels=["Q1\n(low CI)", "Q2", "Q3", "Q4\n(high CI)"])
        change_by_q = df.groupby("ci_quartile", observed=False)["formation_changed"].agg(["mean", "count"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(change_by_q.index.astype(str), change_by_q["mean"],
                    color=["#4878CF", "#6ACC65", "#C4AD66", "#D65F5F"], alpha=0.8)
        axes[0].set_ylabel("Next-window formation change rate")
        axes[0].set_title("Does high CI predict formation change?\n"
                          "(Predictive validity of Bayesian uncertainty)")
        for i, (rate, n) in enumerate(zip(change_by_q["mean"], change_by_q["count"])):
            axes[0].text(i, rate + 0.005, f"{rate:.3f}\nn={n}", ha="center", fontsize=8)
        axes[0].grid(True, axis="y", alpha=0.3)

        r, p = stats.spearmanr(df["ci_half"], df["formation_changed"])
        summary["uncertainty_spearman_r"] = float(r)
        summary["uncertainty_spearman_p"] = float(p)
        axes[1].scatter(df["ci_half"], df["formation_changed"], alpha=0.2, s=5, color="steelblue")
        axes[1].set_xlabel("CI half-width (uncertainty)"); axes[1].set_ylabel("Formation changed next window")
        axes[1].set_title(f"CI vs Next-window change\nSpearman r={r:.3f}, p={p:.2e}")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Uncertainty Predictive Validity  game={game_id}  [{team_side}]",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fp = sub_dir / f"uncertainty_predictive_{game_id}_{team_side}.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"[{game_id}/{team_side}] 预测有效性图 → {fp}")
    except Exception as e:
        log.warning(f"[{game_id}/{team_side}] 预测有效性图失败: {e}")


# ─────────────────────────────────────────────
# ⑦ Epistemic vs Aleatoric 四象限分析
# ─────────────────────────────────────────────
def _plot_epistemic_aleatoric(game_id, team_side, sub_dir, b1_win, ep_path, frame_ids, summary):
    try:
        probvar_cols = [c for c in b1_win.columns if c.startswith("probvar_")]
        if not probvar_cols or not ep_path.exists():
            log.warning(f"[{game_id}/{team_side}] 缺少 probvar 或 epistemic，跳过四象限")
            return
        b1 = b1_win.copy()
        b1["aleatoric"] = b1[probvar_cols].clip(lower=0).pow(0.5).mean(axis=1) * 1.96

        ep_vals = np.load(str(ep_path))
        if len(ep_vals) != len(frame_ids):
            log.warning(f"[{game_id}/{team_side}] epistemic长度不匹配，跳过四象限")
            return
        ep_df = pd.DataFrame({"frame_id": frame_ids, "epistemic": ep_vals})
        b1["frame_id"] = b1["center_fid"].astype(int)
        b1 = b1.merge(ep_df, on="frame_id", how="left")
        b1 = b1.dropna(subset=["aleatoric", "epistemic"])

        ale_med = b1["aleatoric"].median()
        epi_med = b1["epistemic"].median()
        b1["quadrant"] = b1.apply(
            lambda r: (
                "High-E / High-A\n(True transition)" if r["epistemic"] >= epi_med and r["aleatoric"] >= ale_med
                else "High-E / Low-A\n(Model blind spot)" if r["epistemic"] >= epi_med
                else "Low-E / High-A\n(Known switch)" if r["aleatoric"] >= ale_med
                else "Low-E / Low-A\n(Stable formation)"), axis=1)

        quad_colors = {
            "High-E / High-A\n(True transition)":  "#E74C3C",
            "High-E / Low-A\n(Model blind spot)":  "#E67E22",
            "Low-E / High-A\n(Known switch)":       "#3498DB",
            "Low-E / Low-A\n(Stable formation)":   "#2ECC71",
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for quad, grp in b1.groupby("quadrant"):
            c = quad_colors.get(quad, "gray")
            axes[0].scatter(grp["epistemic"], grp["aleatoric"],
                            alpha=0.4, s=8, color=c, label=f"{quad} (n={len(grp)})")
        axes[0].axvline(epi_med, color="black", lw=0.8, ls="--", alpha=0.5)
        axes[0].axhline(ale_med, color="black", lw=0.8, ls="--", alpha=0.5)
        axes[0].set_xlabel("Epistemic uncertainty (MC Dropout)")
        axes[0].set_ylabel("Aleatoric uncertainty (Dirichlet CI)")
        axes[0].set_title(f"Epistemic vs Aleatoric  game={game_id}  [{team_side}]")
        axes[0].legend(fontsize=7, loc="upper right")
        axes[0].grid(True, alpha=0.3)

        if "tei" in b1.columns:
            quad_order = list(quad_colors.keys())
            groups = [b1[b1["quadrant"] == q]["tei"].values for q in quad_order if q in b1["quadrant"].values]
            labels = [q.split("\n")[0] for q in quad_order if q in b1["quadrant"].values]
            bp = axes[1].boxplot(groups, tick_labels=labels, patch_artist=True,
                                  medianprops=dict(color="black", lw=2))
            for patch, q in zip(bp["boxes"], [q for q in quad_order if q in b1["quadrant"].values]):
                patch.set_facecolor(quad_colors[q]); patch.set_alpha(0.7)
            axes[1].set_ylabel("TEI (bits)")
            axes[1].set_title("TEI distribution by quadrant")
            axes[1].tick_params(axis="x", rotation=10); axes[1].grid(True, axis="y", alpha=0.3)
            summary["quad_counts"] = b1["quadrant"].value_counts().to_dict()

        plt.suptitle(f"Bayesian Uncertainty Quadrant Analysis  game={game_id}  [{team_side}]\n"
                     "Dual-layer structure's unique value",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fp = sub_dir / f"epistemic_aleatoric_{game_id}_{team_side}.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"[{game_id}/{team_side}] 四象限图 → {fp}")
    except Exception as e:
        log.warning(f"[{game_id}/{team_side}] 四象限图失败: {e}")


# ─────────────────────────────────────────────
# 主处理函数
# ─────────────────────────────────────────────
def process_game(game_id: int) -> bool:
    out_dir = ANALYSIS_DIR / str(game_id)
    if not out_dir.exists():
        log.error(f"[{game_id}] 分析目录不存在"); return False

    sub_dir = out_dir / SUB_DIR
    sub_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    for team_side in ["home", "away"]:
        win_path = out_dir / f"b1_window_distributions_{team_side}.parquet"
        ep_path  = out_dir / f"b1_frame_epistemic_{team_side}.npy"
        tei_path = out_dir / f"b1_frame_tei_{team_side}.parquet"

        if not win_path.exists():
            log.warning(f"[{game_id}/{team_side}] 缺少 b1_window_distributions，跳过")
            ok = False; continue

        try:
            b1_win = pd.read_parquet(win_path)
            frame_ids = pd.read_parquet(tei_path)["frame_id"].values if tei_path.exists() else np.array([])

            summary = {"game_id": game_id, "team_side": team_side}
            _plot_uncertainty_predictive(game_id, team_side, sub_dir, b1_win, summary)
            _plot_epistemic_aleatoric(game_id, team_side, sub_dir, b1_win, ep_path, frame_ids, summary)

            json.dump(summary, open(sub_dir / f"bayesian_predictive_summary_{game_id}_{team_side}.json", "w"),
                      indent=2, ensure_ascii=False)
            log.info(f"[{game_id}/{team_side}] 完成")
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
    for gid in tqdm(game_ids, desc="3.3.3 Bayesian predictive"):
        if process_game(gid): ok += 1
        else: fail += 1
    log.info(f"完成：{ok} 成功，{fail} 失败")


if __name__ == "__main__":
    main()
