"""
Step 3.2.5 视频生成：B1 阵型概率动态可视化
输入（来自 step3_2_4 v2.0）：
  - morph_general/bgnn_analysis/{game_id}/b1_window_distributions_{side}.parquet
  - morph_general/bgnn_analysis/{game_id}/b1_mainstream_result_{side}.json
  - morph_general/tracking_data_{game_id}_scaled.parquet（用于游戏时间轴）
输出：
  - morph_general/bgnn_analysis/{game_id}/b1_viz_video_{game_id}_{side}[_Xmin-Ymin].mp4

面板布局：
  左：Top-8 阵型概率水平条形图（含 95% CI 误差棒，主流阵型标红）
  右：TEI / GM-TEI_AB 时序 + 当前时刻红色竖线

依赖：需要 ffmpeg（超算：module load ffmpeg）

用法：
  python step3_2_5_b1_video.py --game_id 10517
  python step3_2_5_b1_video.py --game_id 10517 --team_side home --start_min 80 --end_min 120
  MORPH_ENV=hpc python step3_2_5_b1_video.py --game_id 10517 --fps 10
"""

import sys, argparse, json, logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ANALYSIS_DIR = C.MORPH_GENERAL / "bgnn_analysis"
TOP_N = 8  # 每帧显示概率最高的前N个阵型


def _build_game_time(game_id: int) -> dict:
    """返回 fid → game_time_min 映射，使用与 step3_2_4 一致的 period 累计偏移"""
    path = C.MORPH_GENERAL / f"tracking_data_{game_id}_scaled.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["frame_id", "period_id", "timestamp"]).drop_duplicates("frame_id").copy()
    df["tsec"] = df["timestamp"].dt.total_seconds()
    period_max = df.groupby("period_id")["tsec"].max().sort_index()
    cum_offset = period_max.shift(1, fill_value=0.0).cumsum()
    df["game_min"] = (df["tsec"] + df["period_id"].map(cum_offset)) / 60
    return dict(zip(df["frame_id"].astype(int), df["game_min"]))


def make_video(game_id: int, team_side: str, start_min, end_min, fps: int, out_path: Path):
    out_dir = ANALYSIS_DIR / str(game_id)
    win_df = pd.read_parquet(out_dir / f"b1_window_distributions_{team_side}.parquet")
    with open(out_dir / f"b1_mainstream_result_{team_side}.json", encoding="utf-8") as f:
        res = json.load(f)

    fid_to_min = _build_game_time(game_id)
    win_df["game_min"] = win_df["center_fid"].map(
        lambda fid: fid_to_min.get(int(fid), int(fid) / (C.FPS * 60)))

    if start_min is not None:
        win_df = win_df[win_df["game_min"] >= start_min]
    if end_min is not None:
        win_df = win_df[win_df["game_min"] <= end_min]
    win_df = win_df.reset_index(drop=True)

    if len(win_df) == 0:
        log.error(f"[{game_id}/{team_side}] 指定时间范围内无窗口")
        return

    available_forms = res["available_forms"]
    mainstream      = set(res.get("mainstream", []))
    x_time          = win_df["game_min"].values
    tei_vals        = win_df["tei"].values
    has_gm          = "gm_tei_ab" in win_df.columns

    # ── 建立图形 ──────────────────────────────────────────────────────────
    fig, (ax_bar, ax_tei) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={"width_ratios": [1.2, 1]})
    plt.subplots_adjust(wspace=0.38)

    # 右图：静态 TEI 曲线，动态红色竖线
    ax_tei.plot(x_time, tei_vals, color="steelblue", lw=1.0, alpha=0.7, label="TEI")
    if has_gm:
        ax_tei.plot(x_time, win_df["gm_tei_ab"].values, color="darkorange", lw=1.0, alpha=0.7, label="GM-TEI_AB")
    ax_tei.set_xlabel("Time (min)"); ax_tei.set_ylabel("TEI")
    ax_tei.set_title("TEI timeseries"); ax_tei.legend(fontsize=8); ax_tei.grid(True, alpha=0.3)
    time_marker, = ax_tei.plot([x_time[0], x_time[0]], ax_tei.get_ylim(), color="red", lw=1.5, ls="--")

    def update(i):
        row = win_df.iloc[i]
        # Top-N 阵型（按本窗口概率降序）
        probs = {f: row[f"prob_{f}"] for f in available_forms if f"prob_{f}" in row.index}
        top_forms = sorted(probs, key=lambda x: -probs[x])[:TOP_N]
        top_p     = np.array([probs[f] * 100 for f in top_forms])
        ci        = np.array([
            np.sqrt(max(row.get(f"probvar_{f}", 0), 0)) * 1.96 * 100
            for f in top_forms])

        ax_bar.cla()
        colors = ["#E53935" if f in mainstream else "steelblue" for f in top_forms]
        ax_bar.barh(range(len(top_forms)), top_p, color=colors, alpha=0.85,
                    xerr=ci, error_kw=dict(ecolor="#888", capsize=3, lw=0.8))
        ax_bar.set_yticks(range(len(top_forms)))
        ax_bar.set_yticklabels(top_forms, fontsize=9)
        ax_bar.set_xlim(0, max(top_p.max() * 1.35, 5))
        ax_bar.set_xlabel("P(formation) %")
        ax_bar.axvline(C.THRESHOLD * 100, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax_bar.invert_yaxis()
        ax_bar.grid(True, axis="x", alpha=0.3)

        tei_str = f"TEI={row['tei']:.3f}"
        if has_gm:
            tei_str += f"   GM-TEI_AB={row['gm_tei_ab']:.3f}"
        ax_bar.set_title(f"game={game_id}  [{team_side}]   t={row['game_min']:.1f} min\n{tei_str}", fontsize=9)

        # 更新时间线
        ymin, ymax = ax_tei.get_ylim()
        time_marker.set_data([row["game_min"], row["game_min"]], [ymin, ymax])
        return (ax_bar, time_marker)

    n_frames = len(win_df)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=1500)
    log.info(f"[{game_id}/{team_side}] 生成 {n_frames} 帧 @ {fps}fps → {out_path}")
    anim.save(str(out_path), writer=writer, dpi=120)
    plt.close()
    log.info(f"[{game_id}/{team_side}] 视频已保存 → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_id",    type=int, required=True)
    parser.add_argument("--team_side",  choices=["home", "away", "both"], default="both")
    parser.add_argument("--start_min",  type=float, default=None, help="截取起始分钟（默认全场）")
    parser.add_argument("--end_min",    type=float, default=None, help="截取结束分钟（默认全场）")
    parser.add_argument("--fps",        type=int,   default=8,    help="视频帧率（默认8，推荐5-15）")
    args = parser.parse_args()

    out_dir = ANALYSIS_DIR / str(args.game_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    sides = ["home", "away"] if args.team_side == "both" else [args.team_side]
    for side in sides:
        suffix = f"_{args.game_id}_{side}"
        if args.start_min is not None:
            e = int(args.end_min) if args.end_min else "end"
            suffix += f"_{int(args.start_min)}-{e}min"
        out_path = out_dir / f"b1_viz_video{suffix}.mp4"
        try:
            make_video(args.game_id, side, args.start_min, args.end_min, args.fps, out_path)
        except Exception as e:
            log.error(f"[{args.game_id}/{side}] 失败: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
