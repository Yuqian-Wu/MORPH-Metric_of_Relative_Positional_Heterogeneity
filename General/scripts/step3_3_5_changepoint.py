"""
Step 3.3.5: 变化点检测
输出目录：morph_general/bgnn_analysis/{game_id}/3.3.5_changepoint/

评估角度：
  PELT 算法检测 JSD 时序中的结构断点
  变化点与真实事件（进球/换人/黄牌）对齐检验

核心命题：B-GNN变化点比EFPI标签切换更接近真实战术转变时刻

输出（per game per side）：
  - changepoints_{gid}_{side}.png        PELT变化点 + JSD时序 + 事件标注
  - changepoint_summary_{gid}_{side}.json  统计数值（变化点数、与事件距离）

依赖：
  - 需要 step3_3_1 输出的 jsd_timeseries_{gid}_{side}.parquet
  - 需要安装 ruptures (pip install ruptures)

用法：
  python step3_3_5_changepoint.py --game_id 10517
  python step3_3_5_changepoint.py --game_id 3812 3820 10517
  MORPH_ENV=hpc python step3_3_5_changepoint.py --all
"""

import sys, argparse, json, logging
from pathlib import Path
from itertools import groupby

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ANALYSIS_DIR = C.MORPH_GENERAL / "bgnn_analysis"
SUB_DIR_331  = "3.3.1_temporal_coherence"
SUB_DIR      = "3.3.5_changepoint"
PELT_MIN_SIZE = 50
SMOOTH_WIN    = 25
_FINE_MAP = {v: i for i, v in enumerate(
    ["BUILD_UP", "ATTACKING_PLAY", "HIGH_BLOCK", "MID_BLOCK", "LOW_BLOCK"])}
_PERIOD_STARTS_FIXED = {1: 0, 2: 2700, 3: 5400, 4: 6300}


# ─────────────────────────────────────────────
# 辅助：战术阶段色块
# ─────────────────────────────────────────────
def _draw_phase_spans(ax, x_vals, intent_indices):
    if x_vals is None or len(x_vals) < 2:
        return
    stride_est = (x_vals[-1] - x_vals[0]) / max(len(x_vals) - 1, 1)
    x_ends = list(x_vals[1:]) + [x_vals[-1] + stride_est]
    for intent_idx, grp in groupby(enumerate(intent_indices), key=lambda t: t[1]):
        idxs = [t[0] for t in grp]
        ax.axvspan(x_vals[idxs[0]], x_ends[idxs[-1]],
                   alpha=0.07, color=C.INTENT_COLORS[intent_idx], lw=0, zorder=0)


def _phase_legend_patches():
    return [mpatches.Patch(color=C.INTENT_COLORS[i], alpha=0.55, label=C.INTENT_LABELS[i])
            for i in range(len(C.INTENT_LABELS))]


def _load_events(game_id: int) -> dict:
    ev_path = C.EVENT_DIR / f"{game_id}.json"
    if not ev_path.exists():
        return {}
    track_path = C.MORPH_GENERAL / f"tracking_data_{game_id}_scaled.parquet"
    cum_offset = {}
    if track_path.exists():
        import polars as pl
        df = pl.read_parquet(track_path, columns=["period_id", "timestamp"]).to_pandas()
        df["tsec"] = df["timestamp"].dt.total_seconds()
        # 修复：用各周期最大时间戳，而非 unique 取任意行
        pm = df.groupby("period_id")["tsec"].max().sort_index()
        cum_offset = pm.shift(1, fill_value=0.0).cumsum().to_dict()

    def to_min(period, clock):
        if clock is None: return None
        in_p = clock - _PERIOD_STARTS_FIXED.get(period, 0)
        return (in_p + cum_offset.get(period, 0)) / 60

    ev_list = json.load(open(ev_path))
    goals, subs, cards, setpieces = [], [], [], []
    hs = aw = 0
    for e in ev_list:
        ge = e.get("gameEvents") or {}
        pe = e.get("possessionEvents") or {}
        fo = e.get("fouls") or {}
        period, clock = ge.get("period", 1), ge.get("startGameClock")
        is_home = ge.get("homeTeam")
        t = to_min(period, clock)
        if t is None: continue
        if pe.get("shotOutcomeType") == "G":
            if is_home: hs += 1
            else: aw += 1
            goals.append({"time_min": t, "is_home": is_home, "score": f"{hs}-{aw}"})
        if ge.get("gameEventType") == "SUB":
            subs.append({"time_min": t, "is_home": is_home})
        fc = fo.get("finalFoulOutcomeType")
        if fc in ("Y", "R"):
            cards.append({"time_min": t, "is_home": is_home, "card": fc})
        sp = ge.get("setpieceType")
        if sp in ("C", "F", "P", "K", "G"):
            setpieces.append({"time_min": t, "is_home": is_home, "type": sp})
    return {"goals": goals, "subs": subs, "cards": cards, "setpieces": setpieces}


def _event_legend_handles():
    from matplotlib.lines import Line2D
    return [
        Line2D([0],[0], color="#2ECC71", lw=1.2, ls="--", label="Goal scored"),
        Line2D([0],[0], color="#E74C3C", lw=1.2, ls="--", label="Goal conceded"),
        Line2D([0],[0], color="#F1C40F", lw=1.0, ls=":", label="Yellow card"),
        Line2D([0],[0], color="#C0392B", lw=1.0, ls=":", label="Red card"),
        Line2D([0],[0], color="#27AE60", lw=0.8, ls="-.", label="Substitution"),
        Line2D([0],[0], color="#3498DB", lw=0.7, ls=":", label="Corner"),
        Line2D([0],[0], color="#E67E22", lw=0.7, ls=":", label="Free kick"),
        Line2D([0],[0], color="#8E44AD", lw=0.7, ls="--", label="Penalty"),
        Line2D([0],[0], color="#16A085", lw=0.6, ls=":", label="Goal kick"),
        Line2D([0],[0], color="#95A5A6", lw=0.6, ls=":", label="Kickoff"),
    ]


_SP_STYLE = {"C": ("#3498DB", ":", 0.45), "F": ("#E67E22", ":", 0.35),
             "P": ("#8E44AD", "--", 0.80), "K": ("#95A5A6", ":", 0.40),
             "G": ("#16A085", ":", 0.40)}


def _annotate_events(ax, events, team_side, y_top_frac=0.93):
    is_home = (team_side == "home")
    ylim = ax.get_ylim()
    y_top = ylim[0] + (ylim[1] - ylim[0]) * y_top_frac
    for g in events.get("goals", []):
        color = "#2ECC71" if g["is_home"] == is_home else "#E74C3C"
        ax.axvline(g["time_min"], color=color, lw=1.2, ls="--", alpha=0.85)
        ax.text(g["time_min"] + 0.2, y_top, g["score"],
                fontsize=6, color=color, va="top", rotation=90)
    for s in events.get("subs", []):
        if s.get("is_home") not in (is_home, None): continue
        ax.axvline(s["time_min"], color="#27AE60", lw=0.8, ls="-.", alpha=0.6)
    for c in events.get("cards", []):
        if c.get("is_home") not in (is_home, None): continue
        ax.axvline(c["time_min"], color="#F1C40F" if c["card"]=="Y" else "#C0392B",
                   lw=0.8, ls=":", alpha=0.7)
    for sp in events.get("setpieces", []):
        style = _SP_STYLE.get(sp["type"])
        if not style: continue
        ax.axvline(sp["time_min"], color=style[0], lw=0.7, ls=style[1], alpha=style[2])


# ─────────────────────────────────────────────
# PELT 变化点检测
# ─────────────────────────────────────────────
def _plot_changepoints(game_id, team_side, sub_dir, jsd_df, events, summary, phase_data=None):
    try:
        import ruptures as rpt
    except ImportError:
        log.warning("ruptures 未安装，跳过变化点检测 (pip install ruptures)")
        return

    try:
        jsd_smooth = pd.Series(jsd_df["jsd"].values).rolling(
            SMOOTH_WIN, center=True, min_periods=1).mean().values
        signal = jsd_smooth.reshape(-1, 1)
        pen = 5 * jsd_smooth.var() * np.log(len(signal))
        model = rpt.Pelt(model="l2", min_size=PELT_MIN_SIZE, jump=5).fit(signal)
        bkpts = model.predict(pen=pen)
        bkpts = [b for b in bkpts if b < len(signal)]

        x_vals = jsd_df["game_min"].values if "game_min" in jsd_df.columns else np.arange(len(jsd_df))
        cp_times = [x_vals[b] for b in bkpts if b < len(x_vals)]

        fig, ax = plt.subplots(figsize=(22, 5))

        # 战术阶段色块（最底层）
        if phase_data:
            _draw_phase_spans(ax, phase_data[0], phase_data[1])

        ax.plot(x_vals, jsd_smooth, color="steelblue", lw=1.2, alpha=0.9,
                label="JSD (smoothed)", zorder=3)
        ax.axhline(jsd_smooth.mean(), color="steelblue", lw=0.8, ls=":", alpha=0.5)

        # 变化点：在 x 轴底边用橙色倒三角标记，不与事件竖线冲突
        # blended_transform: x 轴用数据坐标，y 轴用 axes 坐标（0=底, 1=顶）
        if cp_times:
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.plot(cp_times, [0.02] * len(cp_times), marker="^",
                    transform=trans, color="#FF6B35", markersize=6,
                    ls="none", zorder=5, clip_on=False,
                    label=f"Changepoint (n={len(cp_times)})")

        _annotate_events(ax, events, team_side)
        ax.set_xlabel("Time (min)"); ax.set_ylabel("JSD (normalized)")
        ax.set_title(f"PELT changepoints on JSD  game={game_id}  [{team_side}]  n_cp={len(bkpts)}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # 右侧图例（事件 + 相位 + changepoint 已在主图例里）
        from matplotlib.lines import Line2D
        cp_handle = Line2D([0], [0], marker="^", color="#FF6B35", ls="none",
                           markersize=6, label=f"Changepoint (n={len(cp_times)})")
        leg_handles = _event_legend_handles() + _phase_legend_patches() + [cp_handle]
        fig.legend(handles=leg_handles, loc="center left", ncol=1,
                   fontsize=6, framealpha=0.85, bbox_to_anchor=(1.01, 0.5),
                   title="Events / Phase / CP", title_fontsize=6)
        plt.tight_layout(rect=[0, 0, 0.84, 1])
        fp = sub_dir / f"changepoints_{game_id}_{team_side}.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
        summary["n_changepoints"] = len(bkpts)

        # 计算变化点与最近事件的距离
        all_events = [g["time_min"] for g in events.get("goals", [])]
        is_home = (team_side == "home")
        all_events += [s["time_min"] for s in events.get("subs", [])
                       if s.get("is_home") in (is_home, None)]
        if all_events and cp_times:
            dists = [min(abs(t - et) for et in all_events) for t in cp_times]
            summary["mean_dist_to_event"] = float(np.mean(dists))

        log.info(f"[{game_id}/{team_side}] PELT变化点 n={len(bkpts)} → {fp}")
    except Exception as e:
        log.warning(f"[{game_id}/{team_side}] PELT变化点失败: {e}")
        import traceback; traceback.print_exc()


# ─────────────────────────────────────────────
# 主处理函数
# ─────────────────────────────────────────────
def process_game(game_id: int) -> bool:
    out_dir = ANALYSIS_DIR / str(game_id)
    if not out_dir.exists():
        log.error(f"[{game_id}] 分析目录不存在"); return False

    sub_dir = out_dir / SUB_DIR
    sub_dir.mkdir(parents=True, exist_ok=True)
    events = _load_events(game_id)

    # 加载战术阶段数据
    track_path = C.MORPH_GENERAL / f"tracking_data_{game_id}_scaled.parquet"
    phase_by_side = {"home": None, "away": None}
    if track_path.exists():
        import polars as pl
        meta_path = C.MORPH_GENERAL / f"metadata_{game_id}.json"
        home_id = ""
        if meta_path.exists():
            home_id = str(json.load(open(meta_path)).get("home_team_id", ""))
        cols = ["frame_id", "period_id", "timestamp", "ball_owning_team_id",
                "attack_intent_home", "defense_intent_home",
                "attack_intent_away", "defense_intent_away"]
        tm = pl.read_parquet(track_path, columns=cols).unique("frame_id").to_pandas()
        tm["tsec"] = tm["timestamp"].dt.total_seconds()
        pm = tm.groupby("period_id")["tsec"].max().sort_index()
        co = pm.shift(1, fill_value=0.0).cumsum()
        tm["game_min"] = (tm["tsec"] + tm["period_id"].map(co)) / 60
        home_poss = tm["ball_owning_team_id"].astype(str) == home_id
        intent_h = tm["attack_intent_home"].where(home_poss, tm["defense_intent_home"])
        intent_a = tm["attack_intent_away"].where(~home_poss, tm["defense_intent_away"])
        for side, intent_s in [("home", intent_h), ("away", intent_a)]:
            phase_df = tm[["game_min"]].copy()
            phase_df["fine_intent"] = intent_s.values
            phase_df = phase_df.dropna(subset=["game_min", "fine_intent"]).sort_values("game_min")
            x_phase = phase_df["game_min"].values
            idx_phase = phase_df["fine_intent"].map(_FINE_MAP).fillna(4).astype(int).values
            phase_by_side[side] = (x_phase, idx_phase)

    ok = True
    for team_side in ["home", "away"]:
        jsd_path = out_dir / SUB_DIR_331 / f"jsd_timeseries_{game_id}_{team_side}.parquet"
        if not jsd_path.exists():
            log.warning(f"[{game_id}/{team_side}] jsd_timeseries 不存在，跳过 "
                         "(请先运行 step3_3_1)")
            ok = False; continue
        try:
            jsd_df = pd.read_parquet(jsd_path)
            summary = {"game_id": game_id, "team_side": team_side}
            _plot_changepoints(game_id, team_side, sub_dir, jsd_df, events, summary,
                               phase_by_side[team_side])
            json.dump(summary, open(sub_dir / f"changepoint_summary_{game_id}_{team_side}.json", "w"),
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
    for gid in tqdm(game_ids, desc="3.3.5 changepoint"):
        if process_game(gid): ok += 1
        else: fail += 1
    log.info(f"完成：{ok} 成功，{fail} 失败")


if __name__ == "__main__":
    main()
