"""
Step 3.3.2 Cross-Game Aggregate Analysis
跨场聚合分析：将64场所有事件研究窗口合并，输出具有统计意义的均值曲线

输出目录：morph_general/bgnn_analysis/cross_game_aggregate/

输出：
  - event_study_aggregate.png      所有场次进球/换人/黄牌前后TEI均值曲线
  - tei_context_aggregate.png      64场×128队mergedTEI按fine_intent分组
  - bayesian_predictive_agg.png    CI预测有效性跨场汇总
  - aggregate_summary.json         关键数值

用法：
  python step3_3_2_cross_game_aggregate.py
  MORPH_ENV=hpc python step3_3_2_cross_game_aggregate.py
"""

import sys, json, logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ANALYSIS_DIR  = C.MORPH_GENERAL / "bgnn_analysis"
OUT_DIR       = ANALYSIS_DIR / "cross_game_aggregate"
EVENT_WIN_S   = 60
_PERIOD_STARTS_FIXED = {1: 0, 2: 2700, 3: 5400, 4: 6300}
_FINE_COLORS  = {"BUILD_UP": "#4878CF", "ATTACKING_PLAY": "#6ACC65",
                 "HIGH_BLOCK": "#D65F5F", "MID_BLOCK": "#C4AD66", "LOW_BLOCK": "#B47CC7"}


def _load_events_with_time(game_id: int) -> dict:
    """加载事件并转为全场分钟数"""
    ev_path = C.EVENT_DIR / f"{game_id}.json"
    if not ev_path.exists():
        return {}
    track_path = C.MORPH_GENERAL / f"tracking_data_{game_id}_scaled.parquet"
    cum_offset = {}
    if track_path.exists():
        import polars as pl
        df = pl.read_parquet(track_path, columns=["period_id","timestamp"]).unique("period_id").to_pandas()
        df["tsec"] = df["timestamp"].dt.total_seconds()
        pm = df.set_index("period_id")["tsec"].sort_index()
        cum_offset = pm.shift(1, fill_value=0.0).cumsum().to_dict()

    def to_min(period, clock):
        if clock is None: return None
        return (clock - _PERIOD_STARTS_FIXED.get(period, 0) + cum_offset.get(period, 0)) / 60

    ev_list = json.load(open(ev_path))
    goals, subs, yellow_cards, red_cards = [], [], [], []
    hs = aw = 0
    for e in ev_list:
        ge = e.get("gameEvents") or {}
        pe = e.get("possessionEvents") or {}
        fo = e.get("fouls") or {}
        period, clock = ge.get("period", 1), ge.get("startGameClock")
        is_home = ge.get("homeTeam")
        # 排除点球大战（period 4 中 abs_clock > 7300s，即120分钟后）
        t = to_min(period, clock)
        if t is None: continue
        if period == 4 and clock is not None and clock > 7300: continue
        if pe.get("shotOutcomeType") == "G":
            if is_home: hs += 1
            else: aw += 1
            goals.append({"time_min": t, "is_home": is_home, "score": f"{hs}-{aw}"})
        if ge.get("gameEventType") == "SUB":
            subs.append({"time_min": t, "is_home": is_home})
        fc = fo.get("finalFoulOutcomeType")
        if fc == "Y":
            yellow_cards.append({"time_min": t, "is_home": is_home, "card": "Y"})
        elif fc == "R":
            red_cards.append({"time_min": t, "is_home": is_home, "card": "R"})
    return {"goals": goals, "subs": subs, "yellow_cards": yellow_cards, "red_cards": red_cards}


def _collect_event_windows(game_id: int, team_side: str,
                            events: dict, win_min: float) -> dict:
    """从单场单队提取所有事件的±win_min TEI窗口"""
    win_path = ANALYSIS_DIR / str(game_id) / f"b1_window_distributions_{team_side}.parquet"
    if not win_path.exists():
        return {}
    b1 = pd.read_parquet(win_path)

    # 补充 game_min
    if "game_min" not in b1.columns:
        track_path = C.MORPH_GENERAL / f"tracking_data_{game_id}_scaled.parquet"
        if track_path.exists():
            import polars as pl
            tm = pl.read_parquet(track_path, columns=["frame_id","period_id","timestamp"])\
                   .unique("frame_id").to_pandas()
            tm["tsec"] = tm["timestamp"].dt.total_seconds()
            pm = tm.groupby("period_id")["tsec"].max().sort_index()
            co = pm.shift(1, fill_value=0.0).cumsum()
            tm["game_min"] = (tm["tsec"] + tm["period_id"].map(co)) / 60
            b1 = b1.merge(tm[["frame_id","game_min"]].rename(
                columns={"frame_id":"center_fid"}), on="center_fid", how="left")
        else:
            return {}

    df = b1.dropna(subset=["game_min","tei"]).sort_values("game_min").reset_index(drop=True)
    x = df["game_min"].values
    t_rel = np.linspace(-win_min, win_min, int(win_min * 60 + 1))   # 1点/秒

    is_home = (team_side == "home")
    result = {}
    for ev_key in ["goals", "subs", "yellow_cards", "red_cards"]:
        ev_list = events.get(ev_key, [])
        if ev_key != "goals":
            ev_list = [e for e in ev_list if e.get("is_home") in (is_home, None)]
        windows_tei = []
        windows_gm  = []
        for ev in ev_list:
            t0 = ev["time_min"]
            q  = t0 + t_rel
            tei_w = np.interp(q, x, df["tei"].values, left=np.nan, right=np.nan)
            if not np.all(np.isnan(tei_w)):
                windows_tei.append(tei_w)
            if "gm_tei_ab" in df.columns:
                gm_w = np.interp(q, x, df["gm_tei_ab"].values, left=np.nan, right=np.nan)
                if not np.all(np.isnan(gm_w)):
                    windows_gm.append(gm_w)
        result[ev_key] = {"tei": windows_tei, "gm": windows_gm}
    return result


def plot_event_study_aggregate(all_windows: dict, win_min: float, summary: dict):
    """跨场聚合事件研究图"""
    t_rel_s = np.linspace(-win_min * 60, win_min * 60,
                           int(win_min * 60 + 1))
    ev_types = [("goals","Goal","#E74C3C"),
                ("subs","Substitution","#27AE60"),
                ("yellow_cards","Yellow card","#F1C40F"),
                ("red_cards","Red card","#C0392B")]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True)
    for col_i, (ev_key, ev_label, color) in enumerate(ev_types):
        tei_mats = all_windows.get(ev_key, {}).get("tei", [])
        gm_mats  = all_windows.get(ev_key, {}).get("gm", [])

        for row_i, (mats, ylabel, title_suffix) in enumerate([
            (tei_mats, "TEI (bits)", "TEI"),
            (gm_mats,  "GM-TEI_AB", "GM-TEI_AB"),
        ]):
            ax = axes[row_i, col_i]
            if mats:
                mat = np.array(mats)
                n = len(mats)
                mean  = np.nanmean(mat, axis=0)
                sem   = np.nanstd(mat, axis=0) / np.sqrt((~np.isnan(mat)).sum(axis=0).clip(min=1))
                ax.plot(t_rel_s, mean, color=color, lw=2,
                        label=f"mean (n={n})")
                ax.fill_between(t_rel_s, mean - sem, mean + sem,
                                 alpha=0.2, color=color)
                ax.axvline(0, color="black", lw=1.5, ls="--", alpha=0.8)
                ax.axhline(float(np.nanmean(mean)), color="gray",
                            lw=0.8, ls=":", alpha=0.6,
                            label=f"baseline={np.nanmean(mean):.3f}")

                # 统计检验：-30~0 vs 0~+30s
                pre_mask  = (t_rel_s >= -30) & (t_rel_s < 0)
                post_mask = (t_rel_s > 0)   & (t_rel_s <= 30)
                pre_vals  = mean[pre_mask];  post_vals = mean[post_mask]
                if len(pre_vals) and len(post_vals):
                    _, p = stats.mannwhitneyu(post_vals, pre_vals,
                                               alternative="two-sided")
                    ax.set_xlabel(f"p={p:.3e}")
                    summary[f"{ev_key}_{ylabel}_p"] = float(p)
                    summary[f"{ev_key}_{ylabel}_n"] = n
            else:
                ax.text(0.5, 0.5, f"No {ev_label}", ha="center",
                         va="center", transform=ax.transAxes, color="gray")

            ax.set_ylabel(ylabel); ax.legend(fontsize=7)
            ax.set_title(f"{ev_label} – {title_suffix}")
            ax.grid(True, alpha=0.3)

    plt.suptitle(f"Event Study (aggregated across all 64 games × home/away)  ±{int(win_min*60)}s",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    fp = OUT_DIR / "event_study_aggregate.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"事件研究聚合图 → {fp}")


def plot_tei_context_aggregate(all_tei_by_intent: dict, summary: dict):
    """跨场 TEI 按 fine_intent 分组箱线图"""
    # 过滤空组
    groups = {k: np.concatenate(v) for k, v in all_tei_by_intent.items() if v}
    if len(groups) < 2:
        log.warning("TEI情境聚合：数据不足，跳过")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    macro = {"attack": np.concatenate([groups.get("BUILD_UP", np.array([])),
                                        groups.get("ATTACKING_PLAY", np.array([]))]),
             "defense": np.concatenate([groups.get("HIGH_BLOCK", np.array([])),
                                         groups.get("MID_BLOCK", np.array([])),
                                         groups.get("LOW_BLOCK", np.array([]))])}
    macro = {k: v for k, v in macro.items() if len(v) > 0}

    for ax, (data, title) in [
        (axes[0], (macro, "TEI by Macro Phase (aggregated)")),
        (axes[1], (groups, "TEI by Fine Intent (aggregated)")),
    ]:
        colors = [("#6ACC65" if k == "attack" else "#D65F5F") if title.startswith("TEI by Macro")
                  else _FINE_COLORS.get(k, "#999") for k in data]
        bp = ax.boxplot(data.values(), tick_labels=data.keys(),
                         patch_artist=True, medianprops=dict(color="black", lw=2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for i, (k, v) in enumerate(data.items()):
            ax.text(i+1, float(np.median(v))*1.001,
                    f"n={len(v)//1000}k\nμ={np.mean(v):.3f}", ha="center", fontsize=6)
        ax.set_title(title); ax.set_ylabel("TEI (bits)")
        ax.tick_params(axis="x", rotation=15); ax.grid(True, axis="y", alpha=0.3)
        keys = list(data.keys())
        if len(keys) == 2:
            _, p = stats.mannwhitneyu(data[keys[0]], data[keys[1]], alternative="two-sided")
            ax.set_xlabel(f"Mann-Whitney p={p:.4e}")
            summary[f"tei_mw_{keys[0]}_vs_{keys[1]}"] = float(p)

    plt.suptitle("TEI by tactical context (all 64 games, 128 team-sides)",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    fp = OUT_DIR / "tei_context_aggregate.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"TEI情境聚合图 → {fp}")


def plot_bayesian_predictive_aggregate(all_spearman: list, summary: dict):
    """CI预测有效性跨场分布"""
    if not all_spearman:
        return
    rs   = [x["r"] for x in all_spearman]
    ps   = [x["p"] for x in all_spearman]
    sig  = sum(1 for p in ps if p < 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(rs, bins=20, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(np.median(rs), color="red", lw=2, ls="--",
                     label=f"median r={np.median(rs):.3f}")
    axes[0].axvline(0, color="gray", lw=1, ls=":")
    axes[0].set_xlabel("Spearman r"); axes[0].set_ylabel("Count")
    axes[0].set_title("CI→next-window change: r distribution\nacross 128 team-sides")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    pct_sig = sig / len(ps) * 100
    axes[1].bar(["p<0.05 (sig)", "p≥0.05"], [sig, len(ps)-sig],
                 color=["#2ECC71","#E74C3C"], alpha=0.8)
    axes[1].set_title(f"Significance: {pct_sig:.0f}% of team-sides significant")
    axes[1].set_ylabel("Count"); axes[1].grid(True, axis="y", alpha=0.3)

    plt.suptitle("Uncertainty Predictive Validity (aggregated 64 games × 2 sides)",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    fp = OUT_DIR / "bayesian_predictive_agg.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
    summary["bayesian_predictive_median_r"] = float(np.median(rs))
    summary["bayesian_predictive_pct_significant"] = float(pct_sig)
    log.info(f"CI预测有效性聚合图 → {fp}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    win_min = EVENT_WIN_S / 60

    # 收集容器
    all_windows    = {"goals":{"tei":[],"gm":[]}, "subs":{"tei":[],"gm":[]},
                      "yellow_cards":{"tei":[],"gm":[]}, "red_cards":{"tei":[],"gm":[]}}
    tei_by_intent  = {k: [] for k in ["BUILD_UP","ATTACKING_PLAY","HIGH_BLOCK","MID_BLOCK","LOW_BLOCK"]}
    all_spearman   = []

    total_goals = total_subs = 0

    for gid in C.ALL_GAME_IDS:
        events = _load_events_with_time(gid)
        total_goals += len(events.get("goals", []))
        total_subs  += len(events.get("subs",  []))

        for team_side in ["home", "away"]:
            # ── 事件研究窗口收集 ───────────────────────────────────────
            windows = _collect_event_windows(gid, team_side, events, win_min)
            for ev_key in ["goals", "subs", "yellow_cards", "red_cards"]:
                for metric in ["tei", "gm"]:
                    all_windows[ev_key][metric].extend(windows.get(ev_key, {}).get(metric, []))

            # ── TEI 按情境收集 ────────────────────────────────────────
            win_path = ANALYSIS_DIR / str(gid) / f"b1_window_distributions_{team_side}.parquet"
            ctx_path = ANALYSIS_DIR / str(gid) / "3.3.2_tei_semantic" / f"tei_semantic_summary_{gid}_{team_side}.json"
            if win_path.exists():
                try:
                    b1 = pd.read_parquet(win_path)
                    # 尝试从context加载fine_intent
                    track_path = C.MORPH_GENERAL / f"tracking_data_{gid}_scaled.parquet"
                    if track_path.exists():
                        import polars as pl
                        cols = ["frame_id","period_id","timestamp","ball_owning_team_id",
                                "attack_intent_home","defense_intent_home",
                                "attack_intent_away","defense_intent_away"]
                        ctx = pl.read_parquet(track_path, columns=cols).unique("frame_id").to_pandas()
                        ctx["tsec"] = ctx["timestamp"].dt.total_seconds()
                        pm = ctx.groupby("period_id")["tsec"].max().sort_index()
                        co = pm.shift(1, fill_value=0.0).cumsum()
                        ctx["game_min"] = (ctx["tsec"] + ctx["period_id"].map(co)) / 60
                        meta_path = C.MORPH_GENERAL / f"metadata_{gid}.json"
                        home_id = ""
                        if meta_path.exists():
                            home_id = str(json.load(open(meta_path)).get("home_team_id",""))
                        hp = ctx["ball_owning_team_id"].astype(str) == home_id
                        if team_side == "home":
                            intent = ctx["attack_intent_home"].where(hp, ctx["defense_intent_home"])
                        else:
                            intent = ctx["attack_intent_away"].where(~hp, ctx["defense_intent_away"])
                        ctx["fine_intent"] = intent.fillna("LOW_BLOCK")
                        b1 = b1.merge(ctx[["frame_id","fine_intent"]].rename(
                            columns={"frame_id":"center_fid"}), on="center_fid", how="left")
                    if "fine_intent" in b1.columns and "tei" in b1.columns:
                        for intent_label, grp in b1.dropna(subset=["fine_intent","tei"]).groupby("fine_intent"):
                            if intent_label in tei_by_intent:
                                tei_by_intent[intent_label].append(grp["tei"].values)
                except Exception as ex:
                    log.debug(f"[{gid}/{team_side}] TEI情境: {ex}")

            # ── CI预测有效性 Spearman ────────────────────────────────
            pred_path = ANALYSIS_DIR / str(gid) / "3.3.3_bayesian_predictive" / \
                        f"bayesian_predictive_summary_{gid}_{team_side}.json"
            if pred_path.exists():
                try:
                    d = json.load(open(pred_path))
                    r = d.get("uncertainty_spearman_r")
                    p = d.get("uncertainty_spearman_p")
                    if r is not None and p is not None:
                        all_spearman.append({"r": float(r), "p": float(p), "gid": gid, "side": team_side})
                except Exception:
                    pass

        log.info(f"[{gid}] 收集完成")

    summary["total_goals_events"] = total_goals
    summary["total_subs_events"]  = total_subs
    summary["total_team_sides"]   = len(all_spearman)

    log.info(f"聚合完成：{len(C.ALL_GAME_IDS)}场 × 2队  进球事件={total_goals}  换人事件={total_subs}")
    log.info(f"CI预测有效性: {len(all_spearman)} 个team-side")

    plot_event_study_aggregate(all_windows, win_min, summary)
    plot_tei_context_aggregate(tei_by_intent, summary)
    plot_bayesian_predictive_aggregate(all_spearman, summary)

    json.dump(summary, open(OUT_DIR / "aggregate_summary.json","w"),
              indent=2, ensure_ascii=False)
    log.info(f"汇总已保存 → {OUT_DIR}/aggregate_summary.json")


if __name__ == "__main__":
    main()
