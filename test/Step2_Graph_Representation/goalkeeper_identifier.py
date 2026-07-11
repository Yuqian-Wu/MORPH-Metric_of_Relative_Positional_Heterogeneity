"""
守门员识别辅助函数
提供鲁棒的守门员识别方法

策略说明：
1. 统计方法（全局）：基于整场比赛的平均位置和位置变化识别守门员
   - 优点：准确识别真正的守门员，不受单帧异常影响
   - 缺点：某些帧中守门员可能不在场（球队压上进攻时）
   
2. 位置方法（局部）：基于单帧X坐标最小值识别守门员
   - 优点：保证该帧有守门员数据
   - 缺点：可能误判（守门员前插、后卫回撤等情况）
   - 适用场景：球队在本方半场防守时，守门员必然是最靠后的球员

3. 混合策略（推荐）：
   - 先用统计方法识别真正的守门员ID
   - 如果该守门员不在指定帧中，说明球队整体压上（守门员数据缺失）
   - 此时用位置方法识别该帧最靠后的球员作为"守门员"
   - 这种情况下，Shape Graph只需要10名外场球员，"守门员"位置仅用于可视化
"""
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

def identify_goalkeeper(tracking_data, team_id, frame_id=None, method='statistical', quiet=False):
    """
    识别守门员
    
    Parameters:
    -----------
    tracking_data : pl.DataFrame
        追踪数据
    team_id : str
        球队ID
    frame_id : int, optional
        特定帧ID。如果提供，会验证守门员在该帧中存在
    method : str
        识别方法：'statistical'（统计）, 'positional'（位置）
    quiet : bool, optional
        是否抑制输出信息（默认False）
    
    Returns:
    --------
    goalkeeper_id : str
        守门员的ID
        
    Notes:
    ------
    当统计方法识别的守门员不在指定帧中时，说明：
    - 球队整体压上进攻，守门员位置数据缺失
    - 此时Shape Graph只需要场上的10名外场球员
    - 位置方法识别的"守门员"仅用于可视化和数据完整性
    """
    
    if method == 'statistical':
        # 方法1：统计方法（推荐）- 基于position_name识别
        # 🔧 修复：直接使用position_name=='GK'识别守门员，避免位置判断错误
        team_data = tracking_data.filter(
            (pl.col('team_id') == team_id) &
            (pl.col('id').is_not_null())
        )
        
        # 优先使用position_name识别守门员
        gk_by_position = team_data.filter(
            pl.col('position_name') == 'GK'
        ).select('id').unique()
        
        if len(gk_by_position) > 0:
            # 找到了标记为GK的球员
            goalkeeper_id = gk_by_position['id'][0]
        else:
            # 回退方案：使用位置统计（但这种情况不应该发生）
            avg_positions = team_data.group_by('id').agg([
                pl.col('x').mean().alias('avg_x'),
                pl.col('x').std().alias('std_x'),
                pl.col('x').count().alias('count')
            ]).to_pandas()
            
            # 守门员特征：平均位置最靠后，且位置变化较小
            avg_positions['score'] = avg_positions['avg_x'] - 0.5 * avg_positions['std_x']
            goalkeeper_id = avg_positions.loc[avg_positions['score'].idxmin(), 'id']
            
            if not quiet:
                print(f"⚠️ 未找到position_name=='GK'的球员，使用位置方法识别: {goalkeeper_id}")
        
        # 🔧 混合策略：如果提供了frame_id，验证守门员在该帧中存在
        if frame_id is not None:
            frame_players = tracking_data.filter(
                (pl.col('frame_id') == frame_id) &
                (pl.col('team_id') == team_id) &
                (pl.col('id').is_not_null())
            )['id'].to_list()
            
            # 如果守门员不在该帧中，使用该帧的位置方法
            if goalkeeper_id not in frame_players:
                if not quiet:
                    print(f"ℹ️ 守门员{goalkeeper_id}不在帧{frame_id}中（球队可能整体压上）")
                    print(f"   使用位置方法识别该帧最靠后的球员...")
                return identify_goalkeeper(tracking_data, team_id, frame_id, method='positional', quiet=quiet)
        
        return goalkeeper_id
    
    elif method == 'positional':
        # 方法2：位置方法（回退方案）
        # 使用特定帧或所有帧的位置
        if frame_id is not None:
            team_data = tracking_data.filter(
                (pl.col('frame_id') == frame_id) &
                (pl.col('team_id') == team_id) &
                (pl.col('id').is_not_null())
            ).to_pandas()
        else:
            team_data = tracking_data.filter(
                (pl.col('team_id') == team_id) &
                (pl.col('id').is_not_null())
            ).to_pandas()
        
        if len(team_data) == 0:
            raise ValueError(f"帧{frame_id}中没有找到球队{team_id}的球员")
        
        # 找到X坐标最小的球员（最靠后）
        goalkeeper_id = team_data.loc[team_data['x'].idxmin(), 'id']
        
        return goalkeeper_id

def separate_players_and_ball(tracking_data, team_id, frame_id, goalkeeper_id=None):
    """
    分离外场球员、守门员和球
    
    Parameters:
    -----------
    tracking_data : pl.DataFrame
        追踪数据
    team_id : str
        球队ID
    frame_id : int
        帧ID
    goalkeeper_id : str, optional
        守门员ID。如果为None，自动识别
    
    Returns:
    --------
    field_players : pd.DataFrame
        外场球员数据
    goalkeeper : pd.DataFrame
        守门员数据
    ball : pd.DataFrame
        球数据
    """
    
    # 如果没有提供守门员ID，自动识别
    if goalkeeper_id is None:
        goalkeeper_id = identify_goalkeeper(tracking_data, team_id, method='statistical')
    
    # 加载该帧的所有球员
    all_players = tracking_data.filter(
        (pl.col('frame_id') == frame_id) &
        (pl.col('team_id') == team_id) &
        (pl.col('id').is_not_null())
    ).to_pandas()
    
    # 分离守门员和外场球员
    goalkeeper = all_players[all_players['id'] == goalkeeper_id].copy()
    field_players = all_players[all_players['id'] != goalkeeper_id].copy()
    
    # 加载球
    ball = tracking_data.filter(
        (pl.col('frame_id') == frame_id) &
        (pl.col('team_id') == 'ball')
    ).to_pandas()
    
    return field_players, goalkeeper, ball

# 使用示例
if __name__ == '__main__':
    DATA_DIR = Path('../../data/morph_test')
    GAME_ID = 10517
    HOME_TEAM_ID = '364'
    
    # 加载数据
    tracking = pl.read_parquet(DATA_DIR / f'tracking_data_{GAME_ID}_scaled.parquet')
    
    # 识别守门员
    gk_id = identify_goalkeeper(tracking, HOME_TEAM_ID, method='statistical')
    print(f"守门员ID: {gk_id}")
    
    # 测试帧
    period_1_frames = tracking.filter(pl.col('period_id') == 1)['frame_id'].unique().sort()
    test_frame_id = period_1_frames[len(period_1_frames) // 2]
    
    # 分离球员和球
    field, gk, ball = separate_players_and_ball(tracking, HOME_TEAM_ID, test_frame_id, gk_id)
    print(f"\n帧{test_frame_id}:")
    print(f"外场球员: {len(field)} 人")
    print(f"守门员: {len(gk)} 人 (ID={gk['id'].values[0]}, X={gk['x'].values[0]:.2f})")
    print(f"球: {len(ball)} 个")