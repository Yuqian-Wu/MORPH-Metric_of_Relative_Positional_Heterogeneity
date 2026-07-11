# -*- coding: utf-8 -*-
"""
Step 1.3: 战术意图识别（CNN批量推理）- 超算版，双队覆盖

输入：data/morph_general/tracking_data_{gid}.parquet
      data/morph_general/metadata_{gid}.json
输出：data/morph_general/tracking_data_{gid}_intent.parquet
      （新增列：attack_intent_home, defense_intent_home,
               attack_intent_away, defense_intent_away）

用法：
  python step1_3_tactical_intent.py --all              # 处理全部64场
  python step1_3_tactical_intent.py --game_id 3813     # 处理单场
  python step1_3_tactical_intent.py --batch_size 256   # 推理批量大小

超算运行（SLURM）：
  sbatch slurm_step1_3.sh
"""

import sys, argparse, logging, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import torch
import torch.nn as nn

# ─── 路径配置 ───
HPC_HOME = Path("/public/home/hpc242111131")
DATA_DIR = HPC_HOME / "G-TAF/MORPH/data/morph_general"
MODEL_DIR = HPC_HOME / "G-TAF/MORPH/models"

GRID_SIZE = (34, 52)
SIGMA = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(DATA_DIR / "step1_3_batch.log"), mode='a'),
    ]
)
log = logging.getLogger(__name__)

ALL_GAME_IDS = [
    3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819,
    3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827,
    3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835,
    3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843,
    3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851,
    3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859,
    10502, 10503, 10504, 10505, 10506, 10507, 10508,
    10509, 10510, 10511, 10512, 10513, 10514, 10515,
    10516, 10517,
]

ATTACK_LABELS = {'BUILD_UP': 0, 'ATTACKING_PLAY': 1}
DEFENSE_LABELS = {'HIGH_BLOCK': 0, 'MID_BLOCK': 1, 'LOW_BLOCK': 2}
label_map_att = {v: k for k, v in ATTACK_LABELS.items()}
label_map_def = {v: k for k, v in DEFENSE_LABELS.items()}


# ─────────────────────────────────────────────
# 热图生成
# ─────────────────────────────────────────────
def create_heatmaps_batch(frame_ids, tracking_data, team_id, image_size=GRID_SIZE, sigma=SIGMA):
    H, W = image_size
    N = len(frame_ids)
    heatmaps = np.zeros((N, 3, H, W), dtype=np.float32)

    team_data = tracking_data.filter(
        (pl.col('team_id') == team_id) & pl.col('id').is_not_null()
    )
    ball_data = tracking_data.filter(pl.col('team_id') == 'ball')

    team_pd = team_data.select(['frame_id','x','y','v']).to_pandas()
    ball_pd = ball_data.select(['frame_id','x','y']).to_pandas()

    team_grouped = team_pd.groupby('frame_id')
    ball_grouped = ball_pd.groupby('frame_id')

    for i, fid in enumerate(frame_ids):
        hm = np.zeros((3, H, W), dtype=np.float32)

        if fid in team_grouped.groups:
            grp = team_grouped.get_group(fid)
            xi = ((grp['x'].values + 52.5) / 105 * W).astype(int)
            yi = ((grp['y'].values + 34.0) / 68.0 * H).astype(int)
            xi = np.clip(xi, 0, W-1)
            yi = np.clip(yi, 0, H-1)
            for x_, y_ in zip(xi, yi):
                hm[0, y_, x_] += 1.0
            hm[0] = gaussian_filter(hm[0], sigma=sigma)

            if 'v' in grp.columns:
                for x_, y_, v_ in zip(xi, yi, grp['v'].values):
                    hm[2, y_, x_] += float(v_)
                hm[2] = gaussian_filter(hm[2], sigma=sigma)

        if fid in ball_grouped.groups:
            bgrp = ball_grouped.get_group(fid).iloc[0]
            bx = int(np.clip((bgrp['x'] + 52.5) / 105 * W, 0, W-1))
            by = int(np.clip((bgrp['y'] + 34.0) / 68.0 * H, 0, H-1))
            hm[1, by, bx] = 1.0
            hm[1] = gaussian_filter(hm[1], sigma=sigma/2)

        for c in range(3):
            mx = hm[c].max()
            if mx > 0:
                hm[c] /= mx

        heatmaps[i] = hm

    return heatmaps


# ─────────────────────────────────────────────
# CNN 模型
# ─────────────────────────────────────────────
class TacticalCNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
# 批量推理
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_all_frames(model, tracking_data, team_id, label_map, batch_size, device, use_amp):
    model.eval()
    all_frame_ids = tracking_data['frame_id'].unique().sort().to_list()
    predictions = {}

    for i in range(0, len(all_frame_ids), batch_size):
        batch_fids = all_frame_ids[i:i+batch_size]
        heatmaps = create_heatmaps_batch(batch_fids, tracking_data, team_id)
        imgs = torch.FloatTensor(heatmaps).to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(imgs)
        else:
            logits = model(imgs)

        preds = logits.argmax(1).cpu().numpy()
        for fid, pred in zip(batch_fids, preds):
            predictions[fid] = label_map[int(pred)]

    return predictions


# ─────────────────────────────────────────────
# 单场处理
# ─────────────────────────────────────────────
def process_game(gid, attack_model, defense_model, device, use_amp, batch_size):
    tracking_file = DATA_DIR / f"tracking_data_{gid}.parquet"
    meta_file = DATA_DIR / f"metadata_{gid}.json"
    out_file = DATA_DIR / f"tracking_data_{gid}_intent.parquet"

    if not tracking_file.exists() or not meta_file.exists():
        return {'game_id': gid, 'status': 'error', 'reason': '文件不存在'}

    try:
        tracking = pl.read_parquet(tracking_file)
        with open(meta_file) as f:
            meta = json.load(f)
        home_id = str(meta['home_team_id'])
        away_id = str(meta['away_team_id'])

        tracking_pd = tracking.to_pandas()
        tracking_pd['attack_intent_home'] = None
        tracking_pd['defense_intent_home'] = None
        tracking_pd['attack_intent_away'] = None
        tracking_pd['defense_intent_away'] = None

        for team_id, team_side in [(home_id, 'home'), (away_id, 'away')]:
            att_preds = predict_all_frames(attack_model, tracking, team_id,
                                          label_map_att, batch_size, device, use_amp)
            def_preds = predict_all_frames(defense_model, tracking, team_id,
                                          label_map_def, batch_size, device, use_amp)
            tracking_pd[f'attack_intent_{team_side}'] = tracking_pd['frame_id'].map(att_preds)
            tracking_pd[f'defense_intent_{team_side}'] = tracking_pd['frame_id'].map(def_preds)

        pl.from_pandas(tracking_pd).write_parquet(out_file)

        n_frames = tracking['frame_id'].n_unique()
        return {
            'game_id': gid, 'status': 'done',
            'n_frames': n_frames,
            'file_mb': out_file.stat().st_size / 1024 / 1024,
        }

    except Exception as e:
        return {'game_id': gid, 'status': 'error', 'reason': str(e)}


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
def get_todo_games(game_ids):
    return [gid for gid in game_ids
            if not (DATA_DIR / f"tracking_data_{gid}_intent.parquet").exists()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='处理全部64场')
    parser.add_argument('--game_id', type=int, help='处理单场')
    parser.add_argument('--batch_size', type=int, default=256, help='推理批量大小（默认256）')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='跳过已完成场次（默认True）')
    args = parser.parse_args()

    if args.all:
        game_ids = ALL_GAME_IDS
    elif args.game_id:
        game_ids = [args.game_id]
    else:
        parser.print_help()
        sys.exit(1)

    todo = get_todo_games(game_ids) if args.skip_existing else game_ids
    log.info(f"总场次: {len(game_ids)}，待处理: {len(todo)}，已完成: {len(game_ids)-len(todo)}")

    if not todo:
        log.info("所有场次已完成，退出。")
        return

    # GPU 检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    log.info(f"使用设备: {device}")
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载模型
    attack_model = TacticalCNN(num_classes=2).to(device)
    defense_model = TacticalCNN(num_classes=3).to(device)

    attack_path = MODEL_DIR / 'attack_model_best.pth'
    defense_path = MODEL_DIR / 'defense_model_best.pth'

    if not attack_path.exists() or not defense_path.exists():
        log.error(f"模型文件不存在: {attack_path} 或 {defense_path}")
        sys.exit(1)

    attack_model.load_state_dict(torch.load(attack_path, map_location=device, weights_only=True))
    defense_model.load_state_dict(torch.load(defense_path, map_location=device, weights_only=True))
    log.info("模型加载完成")

    # 批量推理
    import time
    t0 = time.time()
    all_results = []

    for i, gid in enumerate(todo, 1):
        result = process_game(gid, attack_model, defense_model, device, use_amp, args.batch_size)
        all_results.append(result)

        elapsed = time.time() - t0
        eta_h = elapsed / i * (len(todo) - i) / 3600
        log.info(
            f"[{i:02d}/{len(todo)}] {result['game_id']}: {result['status']} | "
            f"frames={result.get('n_frames',0):,} | "
            f"elapsed={elapsed/3600:.1f}h eta={eta_h:.1f}h"
        )

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    total_h = (time.time() - t0) / 3600
    log.info(f"完成！总用时: {total_h:.2f}h")

    summary_path = DATA_DIR / 'step1_3_summary.csv'
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    log.info(f"汇总已保存: {summary_path}")


if __name__ == '__main__':
    main()
