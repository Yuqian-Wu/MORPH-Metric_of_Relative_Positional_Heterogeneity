"""
Step 3.3.2: TEI 语义有效性分析
输出目录：morph_general/bgnn_analysis/{game_id}/3.3.2_tei_semantic/

评估角度：
  ① TEI 按战术情境分组差异（Mann-Whitney U）
  ③ 事件研究法：进球/换人/黄牌前后 TEI/GM-TEI ±60s 时序
  CI 按战术情境分组（从 step3_3_1 移入）

核心命题：证明 TEI 捕捉到有语义的战术状态，而非随机噪声

输出（per game per side）：
  - tei_by_context_{gid}_{side}.png          ① TEI按fine_intent/macro_phase箱线图
  - ci_by_context_{gid}_{side}.png           CI宽度按fine_intent/macro_phase箱线图
  - event_study_{gid}_{side}.png             ③ 事件研究±60s曲线
  - tei_semantic_summary_{gid}_{side}.json   统计数值汇总

用法：
  python step3_3_2_tei_semantic.py --game_id 10517
  python step3_3_2_tei_semantic.py --game_id 3812 3820 10517
  MORPH_ENV=hpc python step3_3_2_tei_semantic.py --all
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
SUB_DIR      = "3.3.2_tei_semantic"
EVENT_WIN_S  = 60
_PERIOD_STARTS_FIXED = {1: 0, 2: 2700, 3: 5400, 4: 6300}
_FINE_LABELS = ["BUILD_UP", "ATTACKING_PLAY", "HIGH_BLOCK", "MID_BLOCK", "LOW_BLOCK"]
_FINE_COLORS = {"BUILD_UP": "#4878CF", "ATTACKING_PLAY": "#6ACC65",
                "HIGH_BLOCK": "#D65F5F", "MID_BLOCK": "#C4AD66", "LOW_BLOCK": "#B47CC7"}


def _load_context(game_id: int, team_side: str):
    """返回 frame_id, period_id, tsec, game_min, fine_intent, macro_phase"""
    path = C.MORPH_GENERAL / f"tracking_data_{game_id}_scaled.parquet"
    if not path.exists():
        return None
    import polars as pl
    cols = ["frame_id", "period_id", "timestamp", "ball_owning_team_id",
            "attack_intent_home", "defense_intent_home",
            "attack_intent_away", "defense_intent_away"]
    df = pl.read_parquet(path, columns=cols).unique("frame_id").to_pandas()
    df["tsec"] = df["timestamp"].dt.total_seconds()
    period_max = df.groupby("period_id")["tsec"].max().sort_index()
    cum_offset = period_max.shift(1, fill_value=0.0).cumsum()
    df["game_min"] = (df["tsec"] + df["period_id"].map(cum_offset)) / 60

    meta_path = C.MORPH_GENERAL / f"metadata_{game_id}.json"
    home_id = ""
    if meta_path.exists():
        home_id = str(json.load(open(meta_path)).get("home_team_id", ""))
    home_poss = df["ball_owning_team_id"].astype(str) == home_id
    if team_side == "home":
        intent = df["attack_intent_home"].where(home_poss, df["defense_intent_home"])
    else:
        intent = df["attack_intent_away"].where(~home_poss, df["defense_intent_away"])
    df["fine_intent"] = intent.fillna("LOW_BLOCK")
    df["macro_phase"] = df["fine_intent"].apply(
        lambda x: "attack" if x in ("BUILD_UP", "ATTACKING_PLAY") else "defense")
    return df[["frame_id", "period_id", "tsec", "game_min",
               "fine_intent", "macro_phase"]].drop_duplicates("frame_id")


def _load_events(game_id: int) -> dict:
    """加载进球/换人/黄牌事件，返回 time_min"""
    ev_path = C.EVENT_DIR / f"{game_id}.json"
    if not ev_path.exists():
        return {}
    track_path = C.MORPH_GENERAL / f"tracking_data_{game_id}_scaled.parquet"
    cum_offset = {}
    if track_path.exists():
        import polars as pl
        df = pl.read_parquet(track_path, columns=["period_id", "timestamp"]).unique("period_id").to_pandas()
        df["tsec"] = df["timestamp"].dt.total_seconds()
        pm = df.set_index("period_id")["tsec"].sort_index()
        cum_offset = pm.shift(1, fill_value=0.0).cumsum().to_dict()

    def to_min(period, clock):
        if clock is None: return None
        in_p = clock - _PERIOD_STARTS_FIXED.get(period, 0)
        return (in_p + cum_offset.get(period, 0)) / 60

    ev_list = json.load(open(ev_path))
    goals, subs, cards = [], [], []
    hs = aw = 0
    for e in ev_list:
        ge = e.get("gameEvents") or {}
        pe = e.get("possessionEvents") or {}
        fo = e.get("fouls") or {}
        period, clock = ge.get("period", 1), ge.get("startGameClock")
        is_home = ge.get("homeTeam")
        if pe.get("shotOutcomeType") == "G":
            if is_home: hs += 1
            else: aw += 1
            t = to_min(period, clock)
            if t is not None:
                goals.append({"time_min": t, "is_home": is_home,
                               "score": f"{hs}-{aw}",
                               "player": (pe.get("shooterPlayerName") or "?").split()[-1]})
        if ge.get("gameEventType") == "SUB":
            t = to_min(period, clock)
            if t is not None:
                subs.append({"time_min": t, "is_home": is_home,
                              "player_on": (ge.get("playerOnName") or "?").split()[-1]})
        fc = fo.get("finalFoulOutcomeType")
        if fc in ("Y", "R"):
            t = to_min(period, clock)
            if t is not None:
                cards.append({"time_min": t, "is_home": is_home, "card": fc})
    return {"goals": goals, "subs": subs, "cards": cards}


# ─────────────────────────────────────────────
# ① TEI 按战术情境分组
# ─────────────────────────────────────────────
def _plot_tei_by_context(game_id, team_side, sub_dir, b1_win, context_df, summary):
    try:
        df = b1_win.dropna(subset=["fine_intent", "tei"]) if "fine_intent" in b1_win.columns else pd.DataFrame()
        if df.empty:
            log.warning(f"[{game_id}/{team_side}] TEI情境图：b1_win 无fine_intent列，跳过")
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, col, title in [
            (axes[0], "macro_phase",  "TEI by Macro Phase"),
            (axes[1], "fine_intent",  "TEI by Fine Intent"),
        ]:
            groups = {k: v["tei"].values for k, v in df.groupby(col)}
            colors  = [_FINE_COLORS.get(k, "#999") if col == "fine_intent" else
                       ("#6ACC65" if k == "attack" else "#D65F5F")
                       for k in groups]
            bp = ax.boxplot(groups.values(), labels=groups.keys(), patch_artist=True,
                            medianprops=dict(color="black", lw=2))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            for i, (k, v) in enumerate(groups.items()):
                ax.text(i + 1, float(np.median(v)) * 1.005,
                        f"n={len(v)}\nμ={np.mean(v):.3f}", ha="center", fontsize=7)
            ax.set_title(title); ax.set_ylabel("TEI (bits)")
            ax.tick_params(axis="x", rotation=15); ax.grid(True, axis="y", alpha=0.3)
            keys = list(groups.keys())
            if len(keys) == 2:
                _, p = stats.mannwhitneyu(groups[keys[0]], groups[keys[1]], alternative="two-sided")
                ax.set_xlabel(f"Mann-Whitney p={p:.4e}")
                summary[f"tei_mw_p_{col}"] = float(p)
        plt.suptitle(f"TEI by tactical context  game={game_id}  [{team_side}]\n"
                     "(higher TEI → more formation uncertainty)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fp = sub_dir / f"tei_by_context_{game_id}_{team_side}.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"[{game_id}/{team_side}] TEI情境图 → {fp}")
    except Exception as e:
        log.warning(f"[{game_id}/{team_side}] TEI情境图失败: {e}")


# ─────────────────────────────────────────────
# CI 按战术情境分组（从 step3_3_1 移入）
# ─────────────────────────────────────────────
def _plot_ci_by_context(game_id, team_side, sub_dir, b1_win, context_df, summary):
    try:
        probvar_cols = [c for c in b1_win.columns if c.startswith("probvar_")]
        if not probvar_cols:
            return
        df = b1_win.copy()
        df["ci_half"] = df[probvar_cols].clip(lower=0).pow(0.5).mean(axis=1) * 1.96
        if "fine_intent" not in df.columns:
            log.warning(f"[{game_id}/{team_side}] CI情境图：b1_win 无fine_intent列，跳过")
            return
        df = df.dropna(subset=["fine_intent", "ci_half"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, col, title in [
            (axes[0], "macro_phase", "CI by Macro Phase"),
            (axes[1], "fine_intent", "CI by Fine Intent"),
        ]:
            groups = {k: v["ci_half"].values for k, v in df.groupby(col)}
            colors  = [_FINE_COLORS.get(k, "#999") if col == "fine_intent" else
                       ("#6ACC65" if k == "attack" else "#D65F5F")
                       for k in groups]
            bp = ax.boxplot(groups.values(), labels=groups.keys(), patch_artist=True,
                            medianprops=dict(color="purple", lw=2))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.6)
            for i, (k, v) in enumerate(groups.items()):
                ax.text(i + 1, float(np.median(v)) * 1.02, f"n={len(v)}",
                        ha="center", fontsize=7)
            ax.set_title(title); ax.set_ylabel("Mean 95% CI half-width")
            ax.tick_params(axis="x", rotation=15); ax.grid(True, axis="y", alpha=0.3)
            keys = list(groups.keys())
            if len(keys) == 2:
                _, p = stats.mannwhitneyu(groups[keys[0]], groups[keys[1]], alternative="two-sided")
                ax.set_xlabel(f"Mann-Whitney p={p:.4e}")
                summary[f"ci_mw_p_{col}"] = float(p)
        plt.suptitle(f"CI width by tactical context  game={game_id}  [{team_side}]\n"
                     "(wider CI → model more uncertain)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fp = sub_dir / f"ci_by_context_{game_id}_{team_side}.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"[{game_id}/{team_side}] CI情境图 → {fp}")
    except Exception as e:
        log.warning(f"[{game_id}/{team_side}] CI情境图失败: {e}")


# ─────────────────────────────────────────────
# ③ 事件研究法 ±60s
# ─────────────────────────────────────────────
def _plot_event_study(game_id, team_side, sub_dir, b1_win, events, summary):
    try:
        if "game_min" not in b1_win.columns:
            return
        df = b1_win.dropna(subset=["game_min", "tei"]).sort_values("game_min").reset_index(drop=True)
        win_sec = EVENT_WIN_S / 60
        n_pts = 61
        t_rel = np.linspace(-win_sec, win_sec, n_pts)

        event_types = [
            ("goals",  "Goal",        "#E74C3C"),
            ("subs",   "Substitution","#27AE60"),
            ("cards",  "Yellow card", "#F1C40F"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
        is_home = (team_side == "home")
        any_plotted = False

        for col_i, (ev_key, ev_label, color) in enumerate(event_types):
            ev_list = events.get(ev_key, [])
            if ev_key != "goals":
                ev_list = [e for e in ev_list if e.get("is_home") in (is_home, None)]

            windows_tei = []
            windows_gm = []
            for ev in ev_list:
                t0 = ev["time_min"]
                t_query = t0 + t_rel
                tei_interp = np.interp(t_query, df["game_min"].values, df["tei"].values,
                                       left=np.nan, right=np.nan)
                if not np.all(np.isnan(tei_interp)):
                    windows_tei.append(tei_interp)
                if "gm_tei_ab" in df.columns:
                    gm_interp = np.interp(t_query, df["game_min"].values, df["gm_tei_ab"].values,
                                          left=np.nan, right=np.nan)
                    if not np.all(np.isnan(gm_interp)):
                        windows_gm.append(gm_interp)

            # 上图：TEI
            ax0 = axes[0, col_i]
            if windows_tei:
                mat = np.array(windows_tei)
                mean_tei = np.nanmean(mat, axis=0)
                sem_tei  = np.nanstd(mat, axis=0) / np.sqrt((~np.isnan(mat)).sum(axis=0).clip(min=1))
                ax0.plot(t_rel * 60, mean_tei, color=color, lw=2, label=f"mean (n={len(windows_tei)})")
                ax0.fill_between(t_rel * 60, mean_tei - sem_tei, mean_tei + sem_tei,
                                  alpha=0.25, color=color)
                ax0.axvline(0, color="black", lw=1.5, ls="--", alpha=0.8)
                ax0.axhline(df["tei"].mean(), color="gray", lw=0.8, ls=":", alpha=0.6)
                summary[f"event_study_{ev_key}_n"] = len(windows_tei)
                pre_mask  = (t_rel * 60 >= -30) & (t_rel * 60 < 0)
                post_mask = (t_rel * 60 > 0)  & (t_rel * 60 <= 30)
                pre_tei  = mean_tei[pre_mask]
                post_tei = mean_tei[post_mask]
                if len(pre_tei) and len(post_tei):
                    _, p_es = stats.mannwhitneyu(post_tei, pre_tei, alternative="two-sided")
                    summary[f"event_study_{ev_key}_tei_p"] = float(p_es)
                any_plotted = True
            else:
                ax0.text(0.5, 0.5, f"No {ev_label} events", ha="center", va="center",
                         transform=ax0.transAxes, fontsize=10, color="gray")
            ax0.set_title(f"{ev_label} (n={len(windows_tei)})")
            ax0.set_ylabel("TEI (bits)"); ax0.legend(fontsize=7); ax0.grid(True, alpha=0.3)

            # 下图：GM-TEI_AB
            ax1 = axes[1, col_i]
            if windows_gm:
                mat_gm = np.array(windows_gm)
                mean_gm = np.nanmean(mat_gm, axis=0)
                sem_gm  = np.nanstd(mat_gm, axis=0) / np.sqrt((~np.isnan(mat_gm)).sum(axis=0).clip(min=1))
                ax1.plot(t_rel * 60, mean_gm, color="darkorange", lw=2, label=f"mean (n={len(windows_gm)})")
                ax1.fill_between(t_rel * 60, mean_gm - sem_gm, mean_gm + sem_gm,
                                  alpha=0.25, color="darkorange")
                ax1.axvline(0, color="black", lw=1.5, ls="--", alpha=0.8)
            ax1.set_xlabel("Time relative to event (s)")
            ax1.set_ylabel("GM-TEI_AB"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

        if any_plotted:
            plt.suptitle(f"Event study: TEI / GM-TEI_AB ±{EVENT_WIN_S}s  game={game_id}  [{team_side}]",
                         fontsize=11, fontweight="bold")
            plt.tight_layout()
            fp = sub_dir / f"event_study_{game_id}_{team_side}.png"
            plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
            log.info(f"[{game_id}/{team_side}] 事件研究图 → {fp}")
    except Exception as e:
        log.warning(f"[{game_id}/{team_side}] 事件研究图失败: {e}")


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

    ok = True
    for team_side in ["home", "away"]:
        win_path = out_dir / f"b1_window_distributions_{team_side}.parquet"
        if not win_path.exists():
            log.warning(f"[{game_id}/{team_side}] 缺少 b1_window_distributions，跳过")
            ok = False; continue

        try:
            b1_win = pd.read_parquet(win_path)
            context_df = _load_context(game_id, team_side)
            if context_df is not None:
                b1_win = b1_win.merge(context_df[["frame_id", "game_min", "fine_intent", "macro_phase"]]
                                       .rename(columns={"frame_id": "center_fid"}),
                                       on="center_fid", how="left")

            summary = {"game_id": game_id, "team_side": team_side}
            _plot_tei_by_context(game_id, team_side, sub_dir, b1_win, context_df, summary)
            _plot_ci_by_context(game_id, team_side, sub_dir, b1_win, context_df, summary)
            _plot_event_study(game_id, team_side, sub_dir, b1_win, events, summary)

            summary["tei_mean"]  = float(b1_win["tei"].mean()) if "tei" in b1_win.columns else None
            summary["tei_std"]   = float(b1_win["tei"].std())  if "tei" in b1_win.columns else None
            json.dump(summary, open(sub_dir / f"tei_semantic_summary_{game_id}_{team_side}.json", "w"),
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
    for gid in tqdm(game_ids, desc="3.3.2 TEI semantic"):
        if process_game(gid): ok += 1
        else: fail += 1
    log.info(f"完成：{ok} 成功，{fail} 失败")


if __name__ == "__main__":
    main()
