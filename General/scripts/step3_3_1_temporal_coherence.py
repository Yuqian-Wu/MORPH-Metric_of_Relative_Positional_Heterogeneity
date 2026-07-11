"""
Step 3.3.1: 时间相干性评估 v3.0
输出目录：morph_general/bgnn_analysis/{game_id}/3.3.1_temporal_coherence/

输出（per game per side）：
  - jsd_timeseries_{gid}_{side}.parquet         ← 下游3.3.2/3.3.3依赖
  - eval_temporal_coherence_{gid}_{side}.png    维度A JSD + 维度B切换率 + TEI箱线
  - eval_ci_smoothness_{gid}_{side}.png         维度C CI宽度时序 + CI vs TEI散点
  - eval_ci_band_{gid}_{side}.png               TEI置信带时序（7.6）
  - eval_epistemic_{gid}_{side}.png             MC Dropout帧级认知不确定性（7.7）

用法：
  python step3_3_1_temporal_coherence.py --game_id 10517
  python step3_3_1_temporal_coherence.py --game_id 3812 3820 10517
  MORPH_ENV=hpc python step3_3_1_temporal_coherence.py --all
"""

import sys, argparse, json, logging, math
from pathlib import Path
from itertools import groupby

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

NUM_CLASSES  = C.NUM_CLASSES
JSD_THRESH   = 0.05
SMOOTH_WIN   = 25
WROLL_CI     = 8
WROLL_EP     = 500
ANALYSIS_DIR = C.MORPH_GENERAL / "bgnn_analysis"
SUB_DIR      = "3.3.1_temporal_coherence"

_FINE_MAP = {v: i for i, v in enumerate(
    ["BUILD_UP", "ATTACKING_PLAY", "HIGH_BLOCK", "MID_BLOCK", "LOW_BLOCK"])}
_PERIOD_STARTS_FIXED = {1: 0, 2: 2700, 3: 5400, 4: 6300}


# ─────────────────────────────────────────────
# 辅助：战术阶段色块（与 step3_2_4 保持一致）
# ─────────────────────────────────────────────
def _draw_phase_spans(ax, x_vals, intent_indices):
    """在 ax 上叠加战术阶段色块，x_vals 与 intent_indices 等长"""
    if x_vals is None or len(x_vals) < 2:
        return
    stride_est = (x_vals[-1] - x_vals[0]) / max(len(x_vals) - 1, 1)
    x_ends = list(x_vals[1:]) + [x_vals[-1] + stride_est]
    for intent_idx, grp in groupby(enumerate(intent_indices), key=lambda t: t[1]):
        idxs = [t[0] for t in grp]
        x0 = x_vals[idxs[0]]
        x1 = x_ends[idxs[-1]]
        ax.axvspan(x0, x1, alpha=0.07, color=C.INTENT_COLORS[intent_idx], lw=0, zorder=0)


def _phase_legend_patches():
    return [mpatches.Patch(color=C.INTENT_COLORS[i], alpha=0.55, label=C.INTENT_LABELS[i])
            for i in range(len(C.INTENT_LABELS))]


def _get_phase_arrays(time_meta):
    """从 time_meta 提取 (x_vals, intent_indices) 供色块绘制，已排序去重"""
    if time_meta is None:
        return None, None
    df = time_meta.dropna(subset=["game_min", "fine_intent"]).sort_values("game_min")
    x_vals = df["game_min"].values
    intents = df["fine_intent"].map(_FINE_MAP).fillna(4).astype(int).values
    return x_vals, intents


# ─────────────────────────────────────────────
# 辅助：JSD
# ─────────────────────────────────────────────
def jsd_consecutive(probs: np.ndarray) -> np.ndarray:
    p, q = probs[:-1], probs[1:]
    m = 0.5 * (p + q)
    eps = 1e-10
    kl_pm = np.sum(p * np.log2(p / (m + eps) + eps), axis=-1)
    kl_qm = np.sum(q * np.log2(q / (m + eps) + eps), axis=-1)
    return np.clip(0.5 * kl_pm + 0.5 * kl_qm, 0, None)


# ─────────────────────────────────────────────
# 辅助：加载时间+意图元数据
# ─────────────────────────────────────────────
def _load_time_meta(game_id: int, team_side: str):
    """返回 DataFrame: frame_id, period_id, time_sec, game_min, fine_intent"""
    path = C.MORPH_GENERAL / f"tracking_data_{game_id}_scaled.parquet"
    if not path.exists():
        return None
    cols = ["frame_id", "period_id", "timestamp", "ball_owning_team_id",
            "attack_intent_home", "defense_intent_home",
            "attack_intent_away", "defense_intent_away"]
    df = pl.read_parquet(path, columns=cols).unique("frame_id").to_pandas()
    df["tsec"] = df["timestamp"].dt.total_seconds()
    period_max  = df.groupby("period_id")["tsec"].max().sort_index()
    cum_offset  = period_max.shift(1, fill_value=0.0).cumsum()
    df["game_min"] = (df["tsec"] + df["period_id"].map(cum_offset)) / 60

    meta_path = C.MORPH_GENERAL / f"metadata_{game_id}.json"
    home_id = ""
    if meta_path.exists():
        m = json.load(open(meta_path))
        home_id = str(m.get("home_team_id", ""))
    home_poss = df["ball_owning_team_id"].astype(str) == home_id
    if team_side == "home":
        intent_s = df["attack_intent_home"].where(home_poss, df["defense_intent_home"])
    else:
        intent_s = df["attack_intent_away"].where(~home_poss, df["defense_intent_away"])
    df["fine_intent"] = intent_s.fillna("LOW_BLOCK")
    df["macro_phase"] = df["fine_intent"].apply(
        lambda x: "attack" if x in ("BUILD_UP", "ATTACKING_PLAY") else "defense")
    return df[["frame_id", "period_id", "tsec", "game_min",
               "fine_intent", "macro_phase"]].drop_duplicates("frame_id")


# ─────────────────────────────────────────────
# 辅助：加载事件（仅用于标注）
# ─────────────────────────────────────────────
def _load_events_for_annotation(game_id: int, cum_offset_sec: dict) -> dict:
    ev_path = C.EVENT_DIR / f"{game_id}.json"
    if not ev_path.exists():
        return {}
    ev = json.load(open(ev_path))

    def to_min(period, clock):
        if clock is None:
            return None
        in_p = clock - _PERIOD_STARTS_FIXED.get(period, 0)
        return (in_p + cum_offset_sec.get(period, 0)) / 60

    goals, subs, cards, setpieces = [], [], [], []
    hs = aw = 0
    for e in ev:
        ge = e.get("gameEvents") or {}
        pe = e.get("possessionEvents") or {}
        fo = e.get("fouls") or {}
        period  = ge.get("period", 1)
        clock   = ge.get("startGameClock")
        is_home = ge.get("homeTeam")
        t = to_min(period, clock)
        if t is None:
            continue
        if pe.get("shotOutcomeType") == "G":
            if is_home: hs += 1
            else:        aw += 1
            goals.append({"time_min": t, "is_home": is_home,
                           "score": f"{hs}-{aw}",
                           "player": (pe.get("shooterPlayerName") or "?").split()[-1]})
        if ge.get("gameEventType") == "SUB":
            subs.append({"time_min": t, "is_home": is_home,
                          "player_on": (ge.get("playerOnName") or "?").split()[-1]})
        fc = fo.get("finalFoulOutcomeType")
        if fc in ("Y", "R"):
            cards.append({"time_min": t, "is_home": is_home,
                           "card": fc,
                           "player": (fo.get("onFieldCulpritPlayerName") or "?").split()[-1]})
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


_SP_STYLE = {"C": ("#3498DB", ":", 0.45),
             "F": ("#E67E22", ":", 0.35),
             "P": ("#8E44AD", "--", 0.80),
             "K": ("#95A5A6", ":",  0.40),
             "G": ("#16A085", ":",  0.40)}


def _annotate_events(ax, events, team_side, y_top_frac=0.95):
    """在 ax 上标注全类事件竖线"""
    if not events:
        return
    is_home = (team_side == "home")
    ylim = ax.get_ylim()
    y_top = ylim[0] + (ylim[1] - ylim[0]) * y_top_frac
    for g in events.get("goals", []):
        color = "#2ECC71" if g["is_home"] == is_home else "#E74C3C"
        ax.axvline(g["time_min"], color=color, lw=1.2, ls="--", alpha=0.85)
        ax.text(g["time_min"] + 0.2, y_top,
                f"{g['score']} {g['player']}", fontsize=6,
                color=color, va="top", rotation=90)
    for s in events.get("subs", []):
        if s["is_home"] not in (is_home, None): continue
        ax.axvline(s["time_min"], color="#27AE60", lw=0.8, ls="-.", alpha=0.6)
    for c in events.get("cards", []):
        if c["is_home"] not in (is_home, None): continue
        color = "#F1C40F" if c["card"] == "Y" else "#C0392B"
        ax.axvline(c["time_min"], color=color, lw=0.8, ls=":", alpha=0.7)
    for sp in events.get("setpieces", []):
        style = _SP_STYLE.get(sp["type"])
        if not style: continue
        color, ls, alpha = style
        ax.axvline(sp["time_min"], color=color, lw=0.7, ls=ls, alpha=alpha)


# ─────────────────────────────────────────────
# 主处理函数
# ─────────────────────────────────────────────
def process_game(game_id: int) -> bool:
    out_dir = ANALYSIS_DIR / str(game_id)
    if not out_dir.exists():
        log.error(f"[{game_id}] 分析目录不存在")
        return False

    sub_dir = out_dir / SUB_DIR
    sub_dir.mkdir(parents=True, exist_ok=True)

    time_meta = _load_time_meta(game_id, "home")  # 先加载一份获取cum_offset
    if time_meta is None:
        log.warning(f"[{game_id}] tracking数据缺失")
        return False

    # 构建事件 cum_offset
    period_max = time_meta.groupby("period_id")["tsec"].max().sort_index()
    cum_offset = period_max.shift(1, fill_value=0.0).cumsum().to_dict()
    events = _load_events_for_annotation(game_id, cum_offset)

    ok = True
    for team_side in ["home", "away"]:
        probs_path = out_dir / f"b1_frame_probs_{team_side}.npy"
        tei_path   = out_dir / f"b1_frame_tei_{team_side}.parquet"
        win_path   = out_dir / f"b1_window_distributions_{team_side}.parquet"
        ep_path    = out_dir / f"b1_frame_epistemic_{team_side}.npy"

        for p in [probs_path, tei_path, win_path]:
            if not p.exists():
                log.warning(f"[{game_id}/{team_side}] 缺少 {p.name}，跳过")
                ok = False
                break
        else:
            try:
                _process_side(game_id, team_side, sub_dir,
                              probs_path, tei_path, win_path, ep_path, events)
                continue
            except Exception as e:
                log.error(f"[{game_id}/{team_side}] 失败: {e}")
                import traceback; traceback.print_exc()
                ok = False
    return ok


def _process_side(game_id, team_side, sub_dir, probs_path, tei_path, win_path, ep_path, events):
    frame_probs = np.load(str(probs_path))
    frame_tei   = pd.read_parquet(tei_path).sort_values("frame_id").reset_index(drop=True)
    frame_ids   = frame_tei["frame_id"].values
    time_meta   = _load_time_meta(game_id, team_side)

    # ── 维度 A：JSD ───────────────────────────────────────────────────────
    jsd_vals = jsd_consecutive(frame_probs)
    jsd_norm = np.clip(jsd_vals / np.log2(NUM_CLASSES), 0, 1)
    jsd_fids = frame_ids[1:]
    bgnn_jsd_mean = float(np.mean(jsd_norm))
    bgnn_jsd_std  = float(np.std(jsd_norm))
    high_jsd_frac = float((jsd_norm > JSD_THRESH).mean())
    log.info(f"[{game_id}/{team_side}] 维度A JSD 均值={bgnn_jsd_mean:.6f} std={bgnn_jsd_std:.6f}")
    log.info(f"[{game_id}/{team_side}] 高 JSD 帧占比={high_jsd_frac:.2%}")

    jsd_df = pd.DataFrame({"frame_id": jsd_fids, "jsd": jsd_vals})
    if time_meta is not None:
        jsd_df = jsd_df.merge(time_meta, on="frame_id", how="left")
    jsd_path = sub_dir / f"jsd_timeseries_{game_id}_{team_side}.parquet"
    jsd_df.to_parquet(jsd_path, index=False)
    log.info(f"[{game_id}/{team_side}] jsd_timeseries → {jsd_path}")

    # ── 维度 B：top-1 切换率 ──────────────────────────────────────────────
    bgnn_labels  = frame_tei["top1_formation"].values
    bgnn_switch  = (bgnn_labels[1:] != bgnn_labels[:-1]).astype(float)
    bgnn_switch_rate = float(bgnn_switch.mean())
    bgnn_switch_n    = int(bgnn_switch.sum())
    log.info(f"[{game_id}/{team_side}] 维度B B-GNN 切换率={bgnn_switch_rate:.6f} ({bgnn_switch_n}次)")

    # TEI差异检验（切换 vs 非切换）
    tei_aligned      = frame_tei["tei"].values[1:]
    tei_at_switch    = tei_aligned[bgnn_switch == 1]
    tei_at_no_switch = tei_aligned[bgnn_switch == 0]
    u_stat = p_val = float("nan")
    if len(tei_at_switch) > 0 and len(tei_at_no_switch) > 0:
        u_stat, p_val = stats.mannwhitneyu(tei_at_switch, tei_at_no_switch, alternative="greater")
        sig = "显著 (p<0.05)" if p_val < 0.05 else "不显著"
        log.info(f"[{game_id}/{team_side}] TEI倍数={(tei_at_switch.mean()/(tei_at_no_switch.mean()+1e-8)):.2f}× Mann-Whitney p={p_val:.2e} {sig}")

    # ── 维度 C：Dirichlet CI 宽度 ────────────────────────────────────────
    b1_win = pd.read_parquet(win_path)
    if time_meta is not None:
        b1_win = b1_win.merge(time_meta.rename(columns={"frame_id":"center_fid"}),
                               on="center_fid", how="left")
    probvar_cols = [c for c in b1_win.columns if c.startswith("probvar_")]
    mean_ci = std_ci = high_frac = r_ci = p_ci = float("nan")
    if probvar_cols:
        b1_win["mean_ci_half"] = (b1_win[probvar_cols].clip(lower=0).pow(0.5).mean(axis=1) * 1.96)
        mean_ci   = float(b1_win["mean_ci_half"].mean())
        std_ci    = float(b1_win["mean_ci_half"].std())
        high_frac = float((b1_win["mean_ci_half"] > mean_ci + 2 * std_ci).mean())
        log.info(f"[{game_id}/{team_side}] 维度C CI半宽均值={mean_ci:.5f} std={std_ci:.5f}")
        if "tei" in b1_win.columns:
            r_ci, p_ci = stats.pearsonr(b1_win["tei"], b1_win["mean_ci_half"])
            log.info(f"[{game_id}/{team_side}] CI vs TEI: Pearson r={r_ci:.3f} p={p_ci:.2e}")

    # ── 绘图 ──────────────────────────────────────────────────────────────
    _plot_coherence(game_id, team_side, sub_dir, jsd_norm, jsd_fids, bgnn_switch,
                    bgnn_switch_rate, bgnn_switch_n, tei_at_switch, tei_at_no_switch,
                    p_val, events, time_meta)
    if probvar_cols:
        _plot_ci_smoothness(game_id, team_side, sub_dir, b1_win, mean_ci, std_ci, r_ci, p_ci, events)
        _plot_ci_band(game_id, team_side, sub_dir, b1_win, events)

    if ep_path.exists():
        _plot_epistemic(game_id, team_side, sub_dir, ep_path, frame_ids, time_meta, b1_win, events)


# ─────────────────────────────────────────────
# 图1：JSD + 切换率 + TEI箱线
# ─────────────────────────────────────────────
def _plot_coherence(game_id, team_side, sub_dir, jsd_norm, jsd_fids, bgnn_switch,
                    bgnn_switch_rate, bgnn_switch_n, tei_at_switch, tei_at_no_switch,
                    p_val, events, time_meta):
    try:
        # 将帧ID转换为比赛分钟数（修复事件标注堆叠在x=0的bug）
        if time_meta is not None:
            fid_to_min = time_meta.set_index("frame_id")["game_min"].to_dict()
            x_vals = np.array([fid_to_min.get(int(f), np.nan) for f in jsd_fids])
        else:
            x_vals = np.asarray(jsd_fids, dtype=float) / (C.FPS * 60)
        valid = ~np.isnan(x_vals)

        jsd_smooth   = pd.Series(jsd_norm).rolling(SMOOTH_WIN, center=True, min_periods=1).mean().values
        bgnn_sw_roll = pd.Series(bgnn_switch).rolling(SMOOTH_WIN, center=True, min_periods=1).mean().values

        # 战术阶段色块数据
        x_phase, intents_phase = _get_phase_arrays(time_meta)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

        axes[0].plot(x_vals[valid], jsd_smooth[valid], color="steelblue", lw=1.2, alpha=0.9,
                     label=f"B-GNN JSD ({SMOOTH_WIN}-frame rolling)", zorder=3)
        axes[0].axhline(jsd_norm.mean(), color="steelblue", lw=1.5, ls="--", alpha=0.7,
                        label=f"mean={jsd_norm.mean():.4f}")
        _draw_phase_spans(axes[0], x_phase, intents_phase)
        _annotate_events(axes[0], events, team_side)
        axes[0].set_xlabel("Time (min)"); axes[0].set_ylabel("JSD (normalized)")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f"Dim A: Frame-wise JSD  game={game_id}  [{team_side}]")
        axes[0].legend(fontsize=8); axes[0].set_ylim(bottom=0)

        axes[1].plot(x_vals[valid], bgnn_sw_roll[valid], color="steelblue", lw=1.2,
                     label=f"B-GNN switch rate (mean={bgnn_switch_rate:.4f}, {bgnn_switch_n}x)",
                     zorder=3)
        _draw_phase_spans(axes[1], x_phase, intents_phase)
        axes[1].set_xlabel("Time (min)"); axes[1].set_ylabel("Switch rate")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Dim B: Top-1 label switch rate")
        axes[1].legend(fontsize=8); axes[1].set_ylim(bottom=0)

        if len(tei_at_switch) > 0 and len(tei_at_no_switch) > 0:
            axes[2].boxplot([tei_at_no_switch, tei_at_switch],
                            tick_labels=["No switch", "Switch"], patch_artist=True,
                            boxprops=dict(facecolor="lightblue", alpha=0.7),
                            medianprops=dict(color="navy", lw=2))
            p_str = f"p={p_val:.2e}" if not np.isnan(p_val) else ""
            axes[2].set_ylabel("TEI"); axes[2].set_title(f"Switch vs No-switch TEI ({p_str})")
            axes[2].grid(True, axis="y", alpha=0.3)

        plt.suptitle(f"Temporal Coherence  game={game_id}  [{team_side}]", fontsize=12, fontweight="bold")
        leg_handles = _event_legend_handles() + _phase_legend_patches()
        fig.legend(handles=leg_handles, loc="center left", ncol=1,
                   fontsize=6, framealpha=0.85, bbox_to_anchor=(1.01, 0.5),
                   title="Events / Phase", title_fontsize=6)
        plt.tight_layout(rect=[0, 0, 0.84, 1])
        fp = sub_dir / f"eval_temporal_coherence_{game_id}_{team_side}.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"[{game_id}/{team_side}] 时间相干性图 → {fp}")
    except Exception as e:
        log.warning(f"[{game_id}/{team_side}] 时间相干性图失败: {e}")
        import traceback; traceback.print_exc()


# ─────────────────────────────────────────────
# 图2：CI宽度时序 + CI vs TEI 散点
# ─────────────────────────────────────────────
def _plot_ci_smoothness(game_id, team_side, sub_dir, b1_win, mean_ci, std_ci, r_ci, p_ci, events):
    try:
        ci_smooth = b1_win["mean_ci_half"].rolling(WROLL_CI, center=True, min_periods=1).mean()
        x_vals = b1_win["game_min"].values if "game_min" in b1_win.columns else b1_win.index.values

        # 战术阶段色块数据（b1_win 已 merge 了 time_meta，包含 fine_intent）
        x_phase = intents_phase = None
        if "fine_intent" in b1_win.columns and "game_min" in b1_win.columns:
            phase_df = b1_win.dropna(subset=["game_min", "fine_intent"]).sort_values("game_min")
            x_phase = phase_df["game_min"].values
            intents_phase = phase_df["fine_intent"].map(_FINE_MAP).fillna(4).astype(int).values

        fig, axes = plt.subplots(2, 1, figsize=(14, 7))
        axes[0].plot(x_vals, ci_smooth, color="purple", lw=1.2, zorder=3,
                     label=f"Mean CI half-width ({WROLL_CI}-window rolling)")
        axes[0].axhline(mean_ci, color="purple", lw=1.2, ls="--", alpha=0.6,
                        label=f"mean={mean_ci:.5f}")
        _draw_phase_spans(axes[0], x_phase, intents_phase)
        _annotate_events(axes[0], events, team_side)
        axes[0].set_xlabel("Time (min)"); axes[0].set_ylabel("95% CI half-width")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Dim C: Dirichlet posterior CI width timeseries")
        axes[0].legend(fontsize=9); axes[0].set_ylim(bottom=0)

        if "tei" in b1_win.columns and not np.isnan(r_ci):
            axes[1].scatter(b1_win["tei"], b1_win["mean_ci_half"], alpha=0.3, s=5, color="purple")
            axes[1].set_xlabel("TEI"); axes[1].set_ylabel("95% CI half-width")
            axes[1].set_title(f"CI width vs TEI  Pearson r={r_ci:.3f} p={p_ci:.2e}")
            axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Dim C: Bayesian CI stability  game={game_id}  [{team_side}]",
                     fontsize=12, fontweight="bold")
        leg_handles = _event_legend_handles() + _phase_legend_patches()
        fig.legend(handles=leg_handles, loc="center left", ncol=1,
                   fontsize=6, framealpha=0.85, bbox_to_anchor=(1.01, 0.5),
                   title="Events / Phase", title_fontsize=6)
        plt.tight_layout(rect=[0, 0, 0.84, 1])
        fp = sub_dir / f"eval_ci_smoothness_{game_id}_{team_side}.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"[{game_id}/{team_side}] CI平滑性图 → {fp}")
    except Exception as e:
        log.warning(f"[{game_id}/{team_side}] CI平滑性图失败: {e}")


# ─────────────────────────────────────────────
# 图3：TEI 置信带时序（7.6）
# ─────────────────────────────────────────────
def _plot_ci_band(game_id, team_side, sub_dir, b1_win, events):
    try:
        if "mean_ci_half" not in b1_win.columns:
            return
        b1 = b1_win.copy()
        b1["tei_upper"] = b1["tei"] + b1["mean_ci_half"]
        b1["tei_lower"] = (b1["tei"] - b1["mean_ci_half"]).clip(lower=0)
        x_vals = b1["game_min"].values if "game_min" in b1.columns else b1.index.values

        # 战术阶段色块数据
        x_phase = intents_phase = None
        if "fine_intent" in b1.columns:
            phase_df = b1.dropna(subset=["game_min", "fine_intent"]).sort_values("game_min")
            x_phase = phase_df["game_min"].values
            intents_phase = phase_df["fine_intent"].map(_FINE_MAP).fillna(4).astype(int).values

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        tei_s   = b1["tei"].rolling(WROLL_CI, center=True, min_periods=1).mean()
        upper_s = b1["tei_upper"].rolling(WROLL_CI, center=True, min_periods=1).mean()
        lower_s = b1["tei_lower"].rolling(WROLL_CI, center=True, min_periods=1).mean()
        ci_s    = b1["mean_ci_half"].rolling(WROLL_CI, center=True, min_periods=1).mean()

        axes[0].plot(x_vals, tei_s, lw=1.5, color="#D32F2F", zorder=3, label="TEI (window mean)")
        axes[0].fill_between(x_vals, lower_s, upper_s, alpha=0.25, color="#EF9A9A",
                              label="95% CI (shaded width = uncertainty)", zorder=2)
        axes[0].axhline(float(tei_s.mean()), color="gray", lw=0.8, ls=":", alpha=0.7,
                        label=f"mean={tei_s.mean():.3f}")
        _draw_phase_spans(axes[0], x_phase, intents_phase)
        _annotate_events(axes[0], events, team_side)
        axes[0].set_ylabel("TEI (bits)"); axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f"TEI confidence band  game={game_id}  [{team_side}]")
        axes[0].legend(fontsize=8)

        axes[1].fill_between(x_vals, 0, ci_s, alpha=0.5, color="#7E57C2", label="CI half-width", zorder=2)
        axes[1].plot(x_vals, ci_s, lw=1, color="#4527A0", alpha=0.8, zorder=3)
        axes[1].axhline(float(ci_s.mean()), color="gray", lw=0.8, ls=":",
                        label=f"mean={ci_s.mean():.5f}")
        _draw_phase_spans(axes[1], x_phase, intents_phase)
        axes[1].set_ylabel("CI half-width"); axes[1].set_xlabel("Time (min)")
        axes[1].set_title("Bayesian CI width (wider → less certain about formation)")
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"TEI Confidence Band (Dirichlet 95% CI)  game={game_id}  [{team_side}]",
                     fontsize=11, fontweight="bold")
        leg_handles = _event_legend_handles() + _phase_legend_patches()
        fig.legend(handles=leg_handles, loc="center left", ncol=1,
                   fontsize=6, framealpha=0.85, bbox_to_anchor=(1.01, 0.5),
                   title="Events / Phase", title_fontsize=6)
        plt.tight_layout(rect=[0, 0, 0.84, 1])
        fp = sub_dir / f"eval_ci_band_{game_id}_{team_side}.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"[{game_id}/{team_side}] TEI置信带图 → {fp}")
    except Exception as e:
        log.warning(f"[{game_id}/{team_side}] TEI置信带图失败: {e}")


# ─────────────────────────────────────────────
# 图4：MC Dropout 帧级认知不确定性时序（7.7）
# ─────────────────────────────────────────────
def _plot_epistemic(game_id, team_side, sub_dir, ep_path, frame_ids, time_meta, b1_win, events):
    try:
        ep_vals = np.load(str(ep_path))
        if len(ep_vals) != len(frame_ids):
            log.warning(f"[{game_id}/{team_side}] epistemic长度({len(ep_vals)})与frame_ids不匹配，跳过")
            return

        ep_df = pd.DataFrame({"frame_id": frame_ids, "epistemic": ep_vals})
        if time_meta is not None:
            ep_df = ep_df.merge(time_meta[["frame_id", "game_min"]], on="frame_id", how="left")
        else:
            ep_df["game_min"] = np.arange(len(ep_df)) / (C.FPS * 60)

        ep_df = ep_df.dropna(subset=["game_min"]).sort_values("game_min").reset_index(drop=True)

        x_ep = ep_df["game_min"].values
        ep_s = ep_df["epistemic"].rolling(WROLL_EP, center=True, min_periods=1).mean()
        # 阈值从平滑后的曲线计算（原始值波动太大，平滑后几乎不可能超过原始mean+2σ）
        ep_thresh = float(ep_s.mean() + 2 * ep_s.std())

        # 战术阶段色块数据
        x_phase, intents_phase = _get_phase_arrays(time_meta)

        fig, ax = plt.subplots(figsize=(14, 5))

        # 相位色块先绘（zorder=0），曲线在上
        _draw_phase_spans(ax, x_phase, intents_phase)

        # 原始帧级曲线：中度紫色（比主曲线 #6A1B9A 浅，比浅灰更可辨）
        ax.plot(x_ep, ep_df["epistemic"].values, lw=0.3, color="#B39DDB", alpha=0.4, zorder=1,
                label="Raw epistemic (per-frame)")
        ax.plot(x_ep, ep_s, lw=1.8, color="#6A1B9A", alpha=0.95, zorder=3,
                label=f"Epistemic uncertainty (smoothed, w={WROLL_EP})")
        ax.fill_between(x_ep, ep_s, ep_s.min(),
                         where=(ep_s >= ep_thresh), alpha=0.45, color="#FFB74D", zorder=2,
                         label=f"High uncertainty region (≥{ep_thresh:.4f})")
        ax.axhline(ep_df["epistemic"].mean(), color="#9C27B0", lw=0.8, ls=":", alpha=0.6)

        # 右轴叠加窗口级 TEI
        ax2 = ax.twinx()
        if "tei" in b1_win.columns and "game_min" in b1_win.columns:
            win_t = b1_win["game_min"].values
            win_tei = b1_win["tei"].rolling(5, center=True, min_periods=1).mean()
            ax2.plot(win_t, win_tei, lw=1.4, color="#D32F2F", alpha=0.6,
                     ls="--", label="TEI (window, bits)")
        ax2.set_ylabel("TEI (bits)", color="#D32F2F", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#D32F2F")

        _annotate_events(ax, events, team_side)

        ax.set_xlabel("Time (min)")
        ax.set_ylabel("MC Dropout variance (epistemic uncertainty)", color="#6A1B9A")
        ax.tick_params(axis="y", labelcolor="#6A1B9A")
        ax.set_title(f"MC Dropout epistemic uncertainty  game={game_id}  [{team_side}]")
        lines1, lbs1 = ax.get_legend_handles_labels()
        lines2, lbs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbs1 + lbs2, fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.25)

        leg_handles = _event_legend_handles() + _phase_legend_patches()
        fig.legend(handles=leg_handles, loc="center left", ncol=1,
                   fontsize=6, framealpha=0.85, bbox_to_anchor=(1.01, 0.5),
                   title="Events / Phase", title_fontsize=6)
        plt.tight_layout(rect=[0, 0, 0.84, 1])
        fp = sub_dir / f"eval_epistemic_{game_id}_{team_side}.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"[{game_id}/{team_side}] epistemic图 → {fp}")

        # Spearman 相关
        if "tei" in b1_win.columns:
            merged = ep_df.merge(b1_win[["center_fid", "tei"]].rename(
                columns={"center_fid": "frame_id"}), on="frame_id", how="inner")
            if len(merged) > 10:
                r, p = stats.spearmanr(merged["epistemic"], merged["tei"])
                log.info(f"[{game_id}/{team_side}] Spearman(epistemic, TEI): r={r:.4f} p={p:.2e}")
    except Exception as e:
        log.warning(f"[{game_id}/{team_side}] epistemic图失败: {e}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_id", type=int, nargs="+")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    game_ids = C.ALL_GAME_IDS if args.all else (args.game_id if args.game_id else None)
    if not game_ids:
        parser.print_help(); sys.exit(1)
    ok = fail = 0
    for gid in tqdm(game_ids, desc="3.3.1 temporal coherence"):
        if process_game(gid): ok += 1
        else: fail += 1
    log.info(f"完成：{ok} 成功，{fail} 失败")


if __name__ == "__main__":
    main()
