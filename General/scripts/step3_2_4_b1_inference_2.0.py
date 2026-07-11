"""
Step 3.2.4: B1 原型推断（MC Dropout + Dirichlet 窗口聚合 + GM-TEI）
输入：
  - data/models/best_model.pth（step3_2_3 输出）
  - data/bgnn_dataset/graph_dataset_{game_id}.pkl
  - data/bgnn_dataset/dataset_metadata_{game_id}.json
输出（per game）：
  - data/bgnn_analysis/{game_id}/b1_window_distributions.parquet
  - data/bgnn_analysis/{game_id}/b1_frame_epistemic.npy
  - data/bgnn_analysis/{game_id}/b1_mainstream_result.json
  - data/models/b1_prototypes.pth（全局原型，leave-one-game-out 时按场次生成）

用法：
  python step3_2_4_b1_inference.py --game_id 10517
  ON_HPC=1 python step3_2_4_b1_inference.py --all
"""

import sys, argparse, logging, pickle, json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

# 复用 BGNN 定义
from step3_2_3_train import BGNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 原型计算（leave-one-game-out）
# ─────────────────────────────────────────────
def compute_prototypes(model: BGNN, all_game_ids: list, exclude_game_id: int,
                       formation_to_idx: dict) -> tuple:
    """
    对除 exclude_game_id 外的所有场次计算原型 μ_k
    返回：(proto_mat_n, available_forms)
      proto_mat_n: (K, hidden_dim) 归一化原型矩阵
      available_forms: list of formation strings
    """
    idx_to_form = {v: k for k, v in formation_to_idx.items()}
    form2zs = defaultdict(list)

    model.eval()
    for gid in all_game_ids:
        if gid == exclude_game_id:
            continue
        path = game_dataset_path(gid)
        if not path.exists():
            continue
        with open(path, "rb") as f:
            graphs = pickle.load(f)

        loader = DataLoader(graphs, batch_size=C.BATCH_SIZE_EVAL, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                z = torch.nan_to_num(model.embed(batch), nan=0.0)  # 模型权重含NaN时清零
                labels = batch.y_hard.view(-1).tolist()
                for z_i, lbl in zip(z, labels):
                    form = idx_to_form.get(lbl)
                    if form:
                        form2zs[form].append(z_i.cpu())

    available_forms = sorted(form2zs.keys())
    if not available_forms:
        return None, []

    proto_mat = torch.stack([torch.stack(form2zs[f]).mean(0) for f in available_forms])
    # 安全归一化：零原型（embed输出全零）返回零向量而非 NaN
    proto_mat_n = proto_mat / (proto_mat.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    return proto_mat_n, available_forms


# ─────────────────────────────────────────────
# 帧级推断（方案A：MC Dropout）
# ─────────────────────────────────────────────
def infer_frame_probs(model: BGNN, graphs: list,
                      proto_mat_n: torch.Tensor) -> tuple:
    """
    返回：
      frame_probs:    (N, K) 后验均值
      frame_epistemic:(N,)   认知不确定性（最大方差）
      frame_ids:      (N,)   帧 ID 列表
    """
    loader = DataLoader(graphs, batch_size=C.BATCH_SIZE_EVAL, shuffle=False)
    all_z_mc = []
    all_fids = []

    for _ in range(C.N_MC):
        z_run = []
        model.eval()  # BatchNorm 用 running stats；MCDropout 已硬编码 training=True，无需 model.train()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                z_mc = model.embed_mc(batch).cpu()
                # 检查并清理 NaN（极端情况：embed_mc 输出 NaN，替换为零向量）
                if torch.isnan(z_mc).any():
                    z_mc = torch.nan_to_num(z_mc, nan=0.0)
                z_run.append(z_mc)
        all_z_mc.append(torch.cat(z_run, dim=0))

    # 收集 frame_ids（只需一次）
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=C.BATCH_SIZE_EVAL, shuffle=False):
            all_fids.extend([int(fid) for fid in batch.frame_id])

    all_z_mc = torch.stack(all_z_mc)          # (N_MC, N, D)
    # 安全归一化：零向量（embed_mc 经 ReLU 后可能全零）返回零向量而非 NaN
    norms    = all_z_mc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    z_mc_n   = all_z_mc / norms
    proto_n  = proto_mat_n.to("cpu")

    cos_mc   = torch.einsum("mni,ki->mnk", z_mc_n, proto_n)  # (N_MC, N, K)
    probs_mc = F.softmax(cos_mc / C.TAU, dim=-1)              # (N_MC, N, K)

    frame_probs     = probs_mc.mean(0)                        # (N, K)
    frame_probs_var = probs_mc.var(0)                         # (N, K)
    frame_epistemic = frame_probs_var.max(dim=-1).values      # (N,)

    return frame_probs, frame_epistemic, all_fids


# ─────────────────────────────────────────────
# 窗口聚合（方案B：Dirichlet-Multinomial）
# ─────────────────────────────────────────────
def aggregate_windows(frame_probs: torch.Tensor, frame_ids: list,
                      graphs: list, available_forms: list) -> list:
    """
    返回 window_results: list of dict
    """
    N = len(frame_probs)
    K = len(available_forms)

    # 稳定性权重（帧级最大概率）
    stab = frame_probs.max(dim=-1).values  # (N,)

    # 几何特征（从 global_features 取）
    geom_keys = ["spread", "lpw", "hull", "compact", "dlh", "hpl", "lr", "rect"]
    geom_arr = {k: np.zeros(N) for k in geom_keys}
    for i, g in enumerate(graphs):
        gf = g.global_features.squeeze(0).numpy()
        # global_features 布局 v2.0：macro(1)+intent(5)+centroid(2)+spread/diam(2)+geom(7)
        # 索引：spread=8, diam=9, lpw=10, hull=11, compact=12, dlh=13, hpl=14, lr=15, rect=16
        geom_arr["spread"][i]  = gf[8]
        geom_arr["lpw"][i]     = gf[10]
        geom_arr["hull"][i]    = gf[11]
        geom_arr["compact"][i] = gf[12]
        geom_arr["dlh"][i]     = gf[13]
        geom_arr["hpl"][i]     = gf[14]
        geom_arr["lr"][i]      = gf[15]
        geom_arr["rect"][i]    = gf[16]

    window_results = []
    for s in range(0, N - C.WINDOW + 1, C.STRIDE):
        e = s + C.WINDOW
        top1 = frame_probs[s:e].argmax(dim=-1)   # (WINDOW,)
        s_w  = stab[s:e]

        # 稳定性加权频数
        n_k = torch.zeros(K)
        n_k.scatter_add_(0, top1, s_w)

        # Dirichlet 后验
        alpha   = torch.ones(K) + n_k
        alpha_s = alpha.sum()
        P_win   = alpha / alpha_s
        P_var   = (alpha * (alpha_s - alpha)) / (alpha_s ** 2 * (alpha_s + 1))

        geom = {k: float(geom_arr[k][s:e].mean()) for k in geom_keys}

        window_results.append({
            "s": s, "e": e,
            "center_fid": frame_ids[(s + e) // 2] if (s + e) // 2 < len(frame_ids) else frame_ids[-1],
            "probs": P_win,
            "probs_var": P_var,
            "alpha": alpha,
            "geom": geom,
        })

    return window_results


# ─────────────────────────────────────────────
# GM-TEI 计算
# ─────────────────────────────────────────────
def compute_gm_tei(window_results: list) -> list:
    """在 window_results 上添加 tei / gm_tei_ab / gm_tei_cb / tac_dir"""
    dlh_seq = np.array([w["geom"]["dlh"] for w in window_results])
    hpl_seq = np.array([w["geom"]["hpl"] for w in window_results])
    delta   = (np.diff(np.concatenate([[dlh_seq[0]], dlh_seq])) +
               np.diff(np.concatenate([[hpl_seq[0]], hpl_seq])))
    delta_sm = np.convolve(delta, np.ones(5) / 5, mode="same")
    tac_dir  = np.sign(delta_sm).astype(float)

    spread_max = max(w["geom"]["spread"] for w in window_results) + 1e-8
    W_CB = np.ones(6, dtype=np.float32) / 6
    geom_keys_cb = ["spread", "lpw", "hull", "compact", "lr", "rect"]

    for i, w in enumerate(window_results):
        p = w["probs"].clamp(min=1e-10)
        H = float((-p * p.log()).sum())

        g_prime = np.array([w["geom"][k] for k in geom_keys_cb], dtype=np.float32)
        g_prime = g_prime / (np.abs(g_prime).max() + 1e-8)  # 归一化

        w["tei"]       = H
        w["tac_dir"]   = float(tac_dir[i])
        w["gm_tei_ab"] = H * (1.0 + C.BETA * w["geom"]["spread"] / spread_max)  # 去除恒零的tac_dir乘数
        w["gm_tei_cb"] = float(H * (1.0 + float(W_CB @ g_prime)))  # 去除恒零的tac_dir乘数

    return window_results


# ─────────────────────────────────────────────
# 主流阵型识别
# ─────────────────────────────────────────────
def detect_mainstream(window_results: list, available_forms: list) -> list:
    all_P = torch.stack([w["probs"] for w in window_results])
    mean_p = all_P.mean(0)
    std_p  = all_P.std(0)
    cv_p   = std_p / (mean_p + 1e-8)

    mainstream = []
    for k, f in enumerate(available_forms):
        if mean_p[k].item() > C.THRESHOLD:
            mainstream.append({
                "formation": f,
                "mean_prob": round(mean_p[k].item(), 4),
                "cv": round(cv_p[k].item(), 4),
            })
    return sorted(mainstream, key=lambda x: -x["mean_prob"])


def _draw_phase_spans(ax, x_vals, intents):
    if not x_vals:
        return
    # 每个窗口的色块延伸到下一窗口起始，避免单窗口零宽度间隙
    stride_est = (x_vals[-1] - x_vals[0]) / max(len(x_vals) - 1, 1)
    x_ends = list(x_vals[1:]) + [x_vals[-1] + stride_est]
    from itertools import groupby
    for intent_idx, grp in groupby(enumerate(intents), key=lambda t: t[1]):
        idxs = [t[0] for t in grp]
        x0 = x_vals[idxs[0]]
        x1 = x_ends[idxs[-1]]
        ax.axvspan(x0, x1, alpha=0.07, color=C.INTENT_COLORS[intent_idx], lw=0)


def _phase_legend_patches():
    return [mpatches.Patch(color=C.INTENT_COLORS[i], alpha=0.5, label=C.INTENT_LABELS[i])
            for i in range(len(C.INTENT_LABELS))]


_FINE_MAP = {v: i for i, v in enumerate(["BUILD_UP", "ATTACKING_PLAY", "HIGH_BLOCK", "MID_BLOCK", "LOW_BLOCK"])}


def _load_tracking_time_intent(game_id: int):
    """返回 (fid_to_time_min, fid_to_intent_home, fid_to_intent_away, cum_offset_dict)"""
    path = C.MORPH_GENERAL / f"tracking_data_{game_id}_scaled.parquet"
    if not path.exists():
        log.warning(f"[{game_id}] tracking parquet 不存在，时间轴退化为 fid/FPS")
        return {}, {}, {}, {}
    cols = ["frame_id", "period_id", "timestamp", "ball_owning_team_id",
            "attack_intent_home", "defense_intent_home",
            "attack_intent_away", "defense_intent_away"]
    df = pd.read_parquet(path, columns=cols).drop_duplicates("frame_id").copy()
    df["tsec"] = df["timestamp"].dt.total_seconds()
    period_max = df.groupby("period_id")["tsec"].max().sort_index()
    cum_offset = period_max.shift(1, fill_value=0.0).cumsum()
    cum_offset_dict = cum_offset.to_dict()
    df["game_time_min"] = (df["tsec"] + df["period_id"].map(cum_offset)) / 60
    fids = df["frame_id"].astype(int)

    meta = json.load(open(C.MORPH_GENERAL / f"metadata_{game_id}.json"))
    home_id = str(meta["home_team_id"])
    home_poss = df["ball_owning_team_id"].astype(str) == home_id

    intent_h = df["attack_intent_home"].where(home_poss,  df["defense_intent_home"])
    intent_a = df["attack_intent_away"].where(~home_poss, df["defense_intent_away"])

    return (
        dict(zip(fids, df["game_time_min"])),
        dict(zip(fids, intent_h.map(_FINE_MAP).fillna(4).astype(int))),
        dict(zip(fids, intent_a.map(_FINE_MAP).fillna(4).astype(int))),
        cum_offset_dict,
    )


_PERIOD_STARTS_FIXED = {1: 0, 2: 2700, 3: 5400, 4: 6300}  # Gradient Sports 固定基准（秒）


def _load_events(game_id: int, cum_offset: dict) -> dict:
    """从 Event Data JSON 提取关键事件，转换为全场绝对分钟数"""
    path = C.EVENT_DIR / f"{game_id}.json"
    if not path.exists():
        return {}
    ev = json.load(open(path))

    def to_min(period, clock):
        if clock is None:
            return None
        in_period = clock - _PERIOD_STARTS_FIXED.get(period, 0)
        return (in_period + cum_offset.get(period, 0)) / 60

    goals, cards, subs, setpieces = [], [], [], []
    home_score = away_score = skipped_cards = 0

    for e in ev:
        ge = e.get("gameEvents") or {}
        pe = e.get("possessionEvents") or {}
        fo = e.get("fouls") or {}
        period = ge.get("period", 1)
        clock  = ge.get("startGameClock")
        is_home = ge.get("homeTeam")

        if pe.get("shotOutcomeType") == "G":
            if is_home:   home_score += 1
            else:         away_score += 1
            goals.append({
                "time_min": to_min(period, clock),
                "is_home":  is_home,
                "score":    f"{home_score}-{away_score}",
                "player":   (pe.get("shooterPlayerName") or "?").split()[-1],
            })

        fc = fo.get("finalFoulOutcomeType")
        if fc in ("Y", "R"):
            t = to_min(period, clock)
            if t is not None:
                cards.append({
                    "time_min": t,
                    "is_home":  is_home,
                    "card":     fc,
                    "player":   (fo.get("onFieldCulpritPlayerName") or "?").split()[-1],
                })
            else:
                skipped_cards += 1

        if ge.get("gameEventType") == "SUB":
            if is_home is None:
                pid_on = ge.get("playerOnId")
                home_ids = {p["playerId"] for p in (e.get("homePlayers") or [])}
                away_ids = {p["playerId"] for p in (e.get("awayPlayers") or [])}
                if pid_on in home_ids:   is_home = True
                elif pid_on in away_ids: is_home = False
            all_pos = {p["playerId"]: p.get("positionGroupType", "?")
                       for p in (e.get("homePlayers") or []) + (e.get("awayPlayers") or [])}
            subs.append({
                "time_min":   to_min(period, clock),
                "is_home":    is_home,
                "player_on":  (ge.get("playerOnName")  or "?").split()[-1],
                "player_off": (ge.get("playerOffName") or "?").split()[-1],
                "pos_on":     all_pos.get(ge.get("playerOnId"),  "?"),
                "pos_off":    all_pos.get(ge.get("playerOffId"), "?"),
            })

        sp = ge.get("setpieceType")
        if sp in ("C", "F", "P", "K", "G"):
            setpieces.append({
                "time_min": to_min(period, clock),
                "is_home":  is_home,
                "type":     sp,
            })

    _PERIOD_LABELS = {2: "HT", 3: "ET", 4: "ET-HT"}
    periods = sorted(cum_offset.keys())
    boundaries = [{"time_min": cum_offset[p] / 60,
                   "label": _PERIOD_LABELS.get(p, f"P{p}")}
                  for p in periods if p > min(periods)]
    return {"goals": goals, "cards": cards, "subs": subs,
            "setpieces": setpieces, "boundaries": boundaries,
            "skipped_cards": skipped_cards}


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
        Line2D([0],[0], color="gray",    lw=1.0, ls="--", label="Half-time/ET"),
        Line2D([0],[0], color="#95A5A6", lw=0.6, ls=":", label="Kickoff"),
        Line2D([0],[0], color="#16A085", lw=0.6, ls=":", label="Goal kick"),
    ]


def _draw_events(axes, events: dict, team_side: str):
    """在一组共享 x 轴的坐标轴上绘制关键事件标记"""
    if not events:
        return
    axs = axes if hasattr(axes, "__len__") else [axes]
    ax0 = axs[0]
    is_home_team = (team_side == "home")

    for b in events.get("boundaries", []):
        for ax in axs:
            ax.axvline(b["time_min"], color="gray", lw=1.0, ls="--", alpha=0.5)
        ax0.text(b["time_min"] + 0.3, 0.99, b["label"],
                 transform=ax0.get_xaxis_transform(), fontsize=7,
                 color="gray", va="top")

    for g in events.get("goals", []):
        color = "#2ECC71" if g["is_home"] == is_home_team else "#E74C3C"
        label = f"{g['score']} {g['player']}"
        for ax in axs:
            ax.axvline(g["time_min"], color=color, lw=1.2, ls="--", alpha=0.85)
        ax0.text(g["time_min"] + 0.3, 0.99, label,
                 transform=ax0.get_xaxis_transform(), fontsize=6.5,
                 color=color, va="top", rotation=90)

    for c in events.get("cards", []):
        if c["is_home"] not in (is_home_team, None):
            continue
        color = "#F1C40F" if c["card"] == "Y" else "#C0392B"
        for ax in axs:
            ax.axvline(c["time_min"], color=color, lw=1.0, ls=":", alpha=0.8)
        ax0.text(c["time_min"] + 0.3, 0.02, c["player"],
                 transform=ax0.get_xaxis_transform(), fontsize=5.5,
                 color=color, va="bottom", rotation=90)

    sub_history = []  # (time_min, y_used)
    for s in events.get("subs", []):
        if s["is_home"] not in (is_home_team, None):
            continue
        for ax in axs:
            ax.axvline(s["time_min"], color="#27AE60", lw=0.8, ls="-.", alpha=0.6)
        label = f"+{s['player_on']}({s['pos_on']}) -{s['player_off']}({s['pos_off']})"
        nearby_y = {y for t, y in sub_history if abs(s["time_min"] - t) < 3}
        y_pos = 0.35 if 0.55 in nearby_y else 0.55
        sub_history.append((s["time_min"], y_pos))
        ax0.text(s["time_min"] + 0.3, y_pos, label,
                 transform=ax0.get_xaxis_transform(), fontsize=5.5,
                 color="#27AE60", va="center", rotation=90)

    _SP_STYLE = {"C": ("#3498DB", ":", 0.45),
                 "F": ("#E67E22", ":", 0.35),
                 "P": ("#8E44AD", "--", 0.80),
                 "K": ("#95A5A6", ":",  0.40),
                 "G": ("#16A085", ":",  0.40)}
    for sp in events.get("setpieces", []):
        style = _SP_STYLE.get(sp["type"])
        if not style: continue
        color, ls, alpha = style
        for ax in axs:
            ax.axvline(sp["time_min"], color=color, lw=0.7, ls=ls, alpha=alpha)
        if sp["type"] == "P":
            ax0.text(sp["time_min"] + 0.3, 0.80, "PK",
                     transform=ax0.get_xaxis_transform(), fontsize=6.5,
                     color=color, va="center", rotation=90)

    skipped = events.get("skipped_cards", 0)
    if skipped > 0:
        ax0.get_figure().text(
            0.5, 0.001,
            f"* {skipped} yellow card event(s) omitted — missing timestamp in source data",
            ha="center", va="bottom", fontsize=6, color="gray", style="italic")


def save_dashboard(game_id: int, team_side: str, window_results: list,
                   frame_probs: torch.Tensor, available_forms: list,
                   mainstream: list, x_vals: list, intents: list,
                   events: dict):
    """4子图 dashboard：TEI / GM-TEI_AB / GM-TEI_CB / mainstream_prob，共享横轴"""
    out_dir = game_analysis_dir(game_id)
    try:
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1, 1, 1.2]})
        tei_vals = np.array([w["tei"]       for w in window_results])
        gm_ab    = np.array([w["gm_tei_ab"] for w in window_results])
        gm_cb    = np.array([w["gm_tei_cb"] for w in window_results])

        axes[0].plot(x_vals, tei_vals, color="steelblue", lw=1.2)
        axes[0].set_ylabel("TEI (Shannon H)"); axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f"Dashboard  game={game_id}  [{team_side}]", fontsize=11)

        axes[1].plot(x_vals, gm_ab, color="darkorange", lw=1.2)
        axes[1].set_ylabel("GM-TEI_AB"); axes[1].grid(True, alpha=0.3)

        axes[2].plot(x_vals, gm_cb, color="purple", lw=1.2)
        axes[2].set_ylabel("GM-TEI_CB"); axes[2].grid(True, alpha=0.3)

        if mainstream:
            for m in mainstream[:5]:
                f = m["formation"]
                if f not in available_forms: continue
                probs = [w["probs"][available_forms.index(f)].item() for w in window_results]
                axes[3].plot(x_vals, probs, lw=1.2, label=f)
            axes[3].axhline(C.THRESHOLD, color="gray", ls="--", lw=0.8,
                            label=f"threshold={C.THRESHOLD}")
            axes[3].legend(fontsize=8)
        axes[3].set_ylabel("P(formation)"); axes[3].set_xlabel("Time (min)")
        axes[3].grid(True, alpha=0.3)

        for ax in axes:
            _draw_phase_spans(ax, x_vals, intents)
        _draw_events(axes, events, team_side)

        fig.legend(handles=_phase_legend_patches(), loc="lower center", ncol=5,
                   fontsize=7, framealpha=0.5, bbox_to_anchor=(0.5, 0))
        fig.legend(handles=_event_legend_handles(), loc="center left", ncol=1,
                   fontsize=6, framealpha=0.85, bbox_to_anchor=(1.01, 0.5),
                   title="Events", title_fontsize=6)
        plt.tight_layout(rect=[0, 0.04, 0.84, 1])
        fig_path = out_dir / f"dashboard_{game_id}_{team_side}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"[{game_id}] dashboard已保存 → {fig_path}")
    except Exception as e:
        log.warning(f"[{game_id}] dashboard生成失败: {e}")


def save_combined_tei(game_id: int, home_results: list, away_results: list,
                      fid_to_time: dict):
    out_dir = game_analysis_dir(game_id)
    try:
        hx = [fid_to_time.get(int(w["center_fid"]), int(w["center_fid"]) / (C.FPS * 60)) for w in home_results]
        ax_ = [fid_to_time.get(int(w["center_fid"]), int(w["center_fid"]) / (C.FPS * 60)) for w in away_results]
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        for i, (key, ylabel) in enumerate([("tei", "TEI"), ("gm_tei_ab", "GM-TEI_AB"), ("gm_tei_cb", "GM-TEI_CB")]):
            axes[i].plot(hx,  [w[key] for w in home_results], color="steelblue", lw=1.2, label="home")
            axes[i].plot(ax_, [w[key] for w in away_results], color="tomato",    lw=1.2, label="away")
            axes[i].set_ylabel(ylabel); axes[i].legend(fontsize=8); axes[i].grid(True, alpha=0.3)
        axes[0].set_title(f"TEI home vs away  game={game_id}")
        axes[2].set_xlabel("Time (min)")
        plt.tight_layout()
        fig_path = out_dir / f"tei_combined_{game_id}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"[{game_id}] combined TEI图已保存 → {fig_path}")
    except Exception as e:
        log.warning(f"[{game_id}] combined TEI图失败: {e}")


# ─────────────────────────────────────────────
# 保存结果
# ─────────────────────────────────────────────
def save_results(game_id: int, window_results: list, frame_epistemic: torch.Tensor,
                 frame_ids: list, available_forms: list, mainstream: list,
                 frame_probs: torch.Tensor, team_side: str = "home",
                 fid_to_time: dict = None, fid_to_intent: dict = None,
                 events: dict = None):
    out_dir = game_analysis_dir(game_id)

    out_dir.mkdir(parents=True, exist_ok=True)

    # window distributions parquet
    records = []
    for w in window_results:
        row = {
            "center_fid": w["center_fid"],
            "window_start": w["s"],
            "window_end": w["e"],
            "tei": w["tei"],
            "tac_dir": w["tac_dir"],
            "gm_tei_ab": w["gm_tei_ab"],
            "gm_tei_cb": w["gm_tei_cb"],
        }
        for k, f in enumerate(available_forms):
            row[f"prob_{f}"]    = w["probs"][k].item()
            row[f"probvar_{f}"] = w["probs_var"][k].item()
        for gk in ["spread", "lpw", "hull", "compact", "dlh", "hpl", "lr", "rect"]:
            row[f"geom_{gk}"] = w["geom"][gk]
        records.append(row)

    pd.DataFrame(records).to_parquet(out_dir / f"b1_window_distributions_{team_side}.parquet", index=False)

    # frame epistemic
    np.save(str(out_dir / f"b1_frame_epistemic_{team_side}.npy"), frame_epistemic.numpy())

    # 帧级 TEI + top-1 阵型（供 step3_3_x 使用）
    tei_frame = -(frame_probs.clamp(min=1e-10) * frame_probs.clamp(min=1e-10).log2()).sum(dim=-1)
    top1_idx_f = frame_probs.argmax(dim=-1)
    top1_prob_f = frame_probs.max(dim=-1).values
    top1_form_f = [available_forms[i] for i in top1_idx_f.tolist()]
    pd.DataFrame({
        "frame_id":       frame_ids,
        "tei":            tei_frame.tolist(),
        "top1_formation": top1_form_f,
        "top1_prob":      top1_prob_f.tolist(),
    }).to_parquet(out_dir / f"b1_frame_tei_{team_side}.parquet", index=False)

    # 帧级概率矩阵（N, K），供 JSD 计算
    np.save(str(out_dir / f"b1_frame_probs_{team_side}.npy"), frame_probs.numpy())
    log.info(f"[{game_id}/{team_side}] 帧级 TEI / 概率矩阵已保存")

    # mainstream result json
    result = {
        "game_id": game_id,
        "team_side": team_side,
        "mainstream": [m["formation"] for m in mainstream],
        "threshold": C.THRESHOLD,
        "tau": C.TAU,
        "n_mc": C.N_MC,
        "window": C.WINDOW,
        "stride": C.STRIDE,
        "details": mainstream,
        "available_forms": available_forms,
    }
    with open(out_dir / f"b1_mainstream_result_{team_side}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(f"[{game_id}] 保存至 {out_dir}")

    # ── 描述性统计 ──
    tei_vals = np.array([w["tei"] for w in window_results])
    gm_ab    = np.array([w["gm_tei_ab"] for w in window_results])
    gm_cb    = np.array([w["gm_tei_cb"] for w in window_results])
    log.info(f"[{game_id}] TEI: mean={tei_vals.mean():.4f} std={tei_vals.std():.4f} "
             f"min={tei_vals.min():.4f} max={tei_vals.max():.4f}")
    log.info(f"[{game_id}] GM-TEI_AB: mean={gm_ab.mean():.4f} std={gm_ab.std():.4f}")
    log.info(f"[{game_id}] GM-TEI_CB: mean={gm_cb.mean():.4f} std={gm_cb.std():.4f}")
    log.info(f"[{game_id}] 主流阵型: {[m['formation'] for m in mainstream]}")
    for m in mainstream:
        log.info(f"  {m['formation']}: mean_prob={m['mean_prob']:.4f}  CV={m['cv']:.4f}")

    # ── TEI 时序图 ──
    center_fids = [w["center_fid"] for w in window_results]
    if fid_to_time:
        x_vals  = [fid_to_time.get(int(fid), int(fid) / (C.FPS * 60)) for fid in center_fids]
        intents = [fid_to_intent.get(int(fid), 0) if fid_to_intent else 0 for fid in center_fids]
    else:
        x_vals  = [int(fid) / (C.FPS * 60) for fid in center_fids]
        intents = [0] * len(center_fids)
    try:
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(x_vals, tei_vals, color="steelblue", lw=1.2)
        axes[0].set_ylabel("TEI (Shannon H)"); axes[0].set_title(f"TEI timeseries  game={game_id}  [{team_side}]")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(x_vals, gm_ab, color="darkorange", lw=1.2)
        axes[1].set_ylabel("GM-TEI_AB"); axes[1].grid(True, alpha=0.3)
        axes[2].plot(x_vals, [w["gm_tei_cb"] for w in window_results], color="purple", lw=1.2)
        axes[2].set_ylabel("GM-TEI_CB"); axes[2].set_xlabel("Time (min)")
        axes[2].grid(True, alpha=0.3)
        for ax in axes:
            _draw_phase_spans(ax, x_vals, intents)
        _draw_events(axes, events, team_side)
        fig.legend(handles=_phase_legend_patches(), loc="lower center", ncol=5,
                   fontsize=7, framealpha=0.5, bbox_to_anchor=(0.5, 0))
        fig.legend(handles=_event_legend_handles(), loc="center left", ncol=1,
                   fontsize=6, framealpha=0.85, bbox_to_anchor=(1.01, 0.5),
                   title="Events", title_fontsize=6)
        plt.tight_layout(rect=[0, 0.04, 0.84, 1])
        fig_path = out_dir / f"tei_timeseries_{game_id}_{team_side}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"[{game_id}] TEI 时序图已保存 → {fig_path}")
    except Exception as e:
        log.warning(f"[{game_id}] TEI 时序图保存失败: {e}")

    # ── 主流阵型概率时序图 ──
    try:
        if mainstream:
            top_forms = [m["formation"] for m in mainstream[:5]]
            fig, ax = plt.subplots(figsize=(14, 4))
            for f in top_forms:
                probs = [w["probs"][available_forms.index(f)].item()
                         if f in available_forms else 0.0
                         for w in window_results]
                ax.plot(x_vals, probs, lw=1.2, label=f)
            ax.axhline(C.THRESHOLD, color="gray", linestyle="--", lw=0.8, label=f"threshold={C.THRESHOLD}")
            _draw_phase_spans(ax, x_vals, intents)
            _draw_events([ax], events, team_side)
            ax.set_ylabel("P(formation)"); ax.set_xlabel("Time (min)")
            ax.set_title(f"Mainstream formation prob  game={game_id}  [{team_side}]")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            fig.legend(handles=_phase_legend_patches(), loc="lower center", ncol=5,
                       fontsize=7, framealpha=0.5, bbox_to_anchor=(0.5, 0))
            fig.legend(handles=_event_legend_handles(), loc="center left", ncol=1,
                       fontsize=6, framealpha=0.85, bbox_to_anchor=(1.01, 0.5),
                       title="Events", title_fontsize=6)
            plt.tight_layout(rect=[0, 0.08, 0.84, 1])
            fig_path2 = out_dir / f"mainstream_prob_{game_id}_{team_side}.png"
            plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"[{game_id}] 主流阵型概率图已保存 → {fig_path2}")
    except Exception as e:
        log.warning(f"[{game_id}] 主流阵型概率图保存失败: {e}")


BGNN_DIR    = C.MORPH_GENERAL / "bgnn_dataset"
ANALYSIS_DIR = C.MORPH_GENERAL / "bgnn_analysis"
MODEL_DIR_G  = C.MORPH_GENERAL / "bgnn_models"


def game_analysis_dir(game_id: int) -> Path:
    return ANALYSIS_DIR / str(game_id)


def game_dataset_path(game_id: int) -> Path:
    return BGNN_DIR / f"graph_dataset_{game_id}.pkl"


# ─────────────────────────────────────────────
# 单场处理
# ─────────────────────────────────────────────
def process_game(game_id: int, model: BGNN, all_game_ids: list,
                 formation_to_idx: dict) -> bool:
    out_dir = game_analysis_dir(game_id)
    if (out_dir / "b1_mainstream_result_home.json").exists() and \
       (out_dir / "b1_mainstream_result_away.json").exists():
        log.info(f"[{game_id}] 已存在，跳过")
        return True

    dataset_path = game_dataset_path(game_id)
    if not dataset_path.exists():
        log.error(f"[{game_id}] 数据集不存在: {dataset_path}")
        return False

    try:
        with open(dataset_path, "rb") as f:
            graphs = pickle.load(f)
        graphs = [g for g in graphs if not torch.isnan(g.x).any()]
        # 按 frame_id 排序，恢复时序结构（降采样打乱了顺序，DLH/HPL 趋势需要时序才有意义）
        graphs.sort(key=lambda g: g.frame_id if isinstance(g.frame_id, int) else int(g.frame_id.item()))
        if not graphs:
            log.warning(f"[{game_id}] 无有效图")
            return False

        log.info(f"[{game_id}] 计算原型（leave-one-game-out）...")
        proto_mat_n, available_forms = compute_prototypes(
            model, all_game_ids, game_id, formation_to_idx
        )
        if proto_mat_n is None:
            log.error(f"[{game_id}] 原型计算失败")
            return False

        fid_to_time, fid_to_intent_h, fid_to_intent_a, cum_offset = _load_tracking_time_intent(game_id)
        fid_to_intent_map = {"home": fid_to_intent_h, "away": fid_to_intent_a}
        events = _load_events(game_id, cum_offset)

        log.info(f"[{game_id}] MC Dropout 帧级推断（N_MC={C.N_MC}）...")
        idx_to_form = {v: k for k, v in formation_to_idx.items()}
        K = len(available_forms)
        side_results = {}

        for team_side in ["home", "away"]:
            team_graphs = [g for g in graphs if g.team_side == team_side]
            if not team_graphs:
                log.warning(f"[{game_id}/{team_side}] 无有效图，跳过")
                continue

            frame_probs, frame_epistemic, frame_ids = infer_frame_probs(
                model, team_graphs, proto_mat_n
            )

            # 嵌入空间判别力诊断：Top-1/3/5 + Mean Rank
            match1 = match3 = match5 = total = 0
            rank_sum = 0
            for i, g in enumerate(team_graphs[:len(frame_ids)]):
                gt_form = idx_to_form.get(int(g.y_hard.item()))
                if not gt_form or gt_form not in available_forms:
                    continue
                total += 1
                gt_k = available_forms.index(gt_form)
                probs_i = frame_probs[i]
                sorted_idx = probs_i.argsort(descending=True).tolist()
                rank = sorted_idx.index(gt_k) + 1
                rank_sum += rank
                if rank <= 1: match1 += 1
                if rank <= 3: match3 += 1
                if rank <= 5: match5 += 1
            mean_rank = rank_sum / total if total > 0 else 0.0
            log.info(f"[{game_id}/{team_side}] Top-1={match1/total:.3f}  Top-3={match3/total:.3f}  "
                     f"Top-5={match5/total:.3f}  MeanRank={mean_rank:.1f}/{K}  "
                     f"(随机期望: Top-1={1/K:.3f}, MeanRank={(K+1)/2:.0f})")

            log.info(f"[{game_id}/{team_side}] Dirichlet 窗口聚合...")
            window_results = aggregate_windows(frame_probs, frame_ids, team_graphs, available_forms)

            log.info(f"[{game_id}/{team_side}] GM-TEI 计算...")
            window_results = compute_gm_tei(window_results)

            mainstream = detect_mainstream(window_results, available_forms)
            log.info(f"[{game_id}/{team_side}] 主流阵型: {[m['formation'] for m in mainstream]}")

            save_results(game_id, window_results, frame_epistemic,
                         frame_ids, available_forms, mainstream, frame_probs, team_side,
                         fid_to_time, fid_to_intent_map[team_side], events)
            side_results[team_side] = (window_results, frame_probs)
        if len(side_results) == 2:
            save_combined_tei(game_id,
                              side_results["home"][0], side_results["away"][0], fid_to_time)
        # dashboard per team
        for ts, (wr, fp) in side_results.items():
            x_v = [fid_to_time.get(int(w["center_fid"]), int(w["center_fid"]) / (C.FPS * 60)) for w in wr]
            ints = [fid_to_intent_map[ts].get(int(w["center_fid"]), 0) for w in wr]
            ms   = detect_mainstream(wr, available_forms)
            save_dashboard(game_id, ts, wr, fp, available_forms, ms, x_v, ints, events)
        return True

    except Exception as e:
        log.error(f"[{game_id}] 失败: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_id", type=int, nargs="+")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    game_ids = C.ALL_GAME_IDS if args.all else (args.game_id if args.game_id else None)
    if not game_ids:
        parser.print_help(); sys.exit(1)

    # 加载 formation_to_idx
    meta_path = C.MORPH_GENERAL / "bgnn_dataset" / "formation_mapping.json"
    if not meta_path.exists():
        log.error(f"缺少 formation_mapping.json: {meta_path}")
        sys.exit(1)
    with open(meta_path) as f:
        formation_to_idx = json.load(f)["formation_to_idx"]

    ok = fail = 0
    for gid in tqdm(game_ids, desc="3.2.4 B1 inference"):
        # LOGO：每场用对应折模型
        model_path = MODEL_DIR_G / f"model_fold_{gid}.pth"
        if not model_path.exists():
            log.warning(f"[{gid}] 模型不存在: {model_path}，跳过")
            fail += 1
            continue
        model = BGNN(num_classes=len(formation_to_idx)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        if process_game(gid, model, C.ALL_GAME_IDS, formation_to_idx):
            ok += 1
        else:
            fail += 1
    log.info(f"完成：{ok} 成功，{fail} 失败")


if __name__ == "__main__":
    main()
