"""
B2-D改进版：基于丰富几何特征的阵型匹配

核心改进：
1. 不再使用5个VL重心点（信息量太少）
2. 计算15+维几何特征向量（参考文献Table 3.2）
3. 使用余弦相似度/欧氏距离匹配

特征列表（参考3.2.2和文献）：
- 基础形状：Width, Length, LpW, Stretch, Spread, Dispersion
- 凸包特征：Convex hull area, Compactness, Rectangularity, Circularity
- 位置特征：Defensive line height, Highest player location, Centroid goal distance
- DT图特征：Triangle area (mean), Edge length (mean), Angle (mean)
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist, cdist


def compute_geometric_feature_vector(positions):
    """
    计算完整的几何特征向量（15维）

    参数:
        positions: numpy array of shape (N, 2) - 球员位置坐标

    返回:
        features: numpy array of shape (15,) - 几何特征向量
    """
    N = len(positions)
    if N < 3:
        return np.zeros(15)

    # 1. Team centroid
    centroid = positions.mean(axis=0)

    # 2. Width & Length
    width = positions[:, 1].max() - positions[:, 1].min()  # Y方向
    length = positions[:, 0].max() - positions[:, 0].min()  # X方向

    # 3. Length per Width (LpW)
    lpw = length / width if width > 1e-6 else 1.0

    # 4. Stretch (平均到质心距离)
    stretch = np.linalg.norm(positions - centroid, axis=1).mean()

    # 5. Spread (Frobenius范数)
    spread = np.sqrt((pdist(positions) ** 2).sum())

    # 6. Dispersion (平均成对距离)
    dispersion = pdist(positions).mean()

    # 7. Compactness (最小外接矩形面积)
    compactness = length * width

    # 8. Convex hull area
    try:
        hull = ConvexHull(positions)
        convex_area = hull.volume  # 2D中volume=area
    except:
        convex_area = 0.0

    # 9. Rectangularity (凸包面积/外接矩形面积)
    rectangularity = convex_area / compactness if compactness > 1e-6 else 0.0

    # 10. Circularity (周长^2 / 凸包面积)
    try:
        perimeter = hull.area  # 2D中area=perimeter
        circularity = (perimeter ** 2) / convex_area if convex_area > 1e-6 else 0.0
    except:
        circularity = 0.0

    # 11. Defensive line height (最小X坐标的绝对值)
    defensive_line = abs(positions[:, 0].min())

    # 12. Highest player location (最大X坐标的绝对值)
    highest_player = abs(positions[:, 0].max())

    # 13. Centroid goal distance (质心到球门距离，假设球门在X=0)
    centroid_goal_dist = abs(centroid[0])

    # 14-16. Delaunay三角形特征
    try:
        tri = Delaunay(positions)
        triangles = tri.simplices

        # 平均三角形面积
        areas = []
        for t in triangles:
            pts = positions[t]
            area = 0.5 * abs(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
            areas.append(area)
        avg_triangle_area = np.mean(areas) if areas else 0.0

        # 平均边长
        edges = []
        for t in triangles:
            for i in range(3):
                edge_len = np.linalg.norm(positions[t[i]] - positions[t[(i+1)%3]])
                edges.append(edge_len)
        avg_edge_length = np.mean(edges) if edges else 0.0
    except:
        avg_triangle_area = 0.0
        avg_edge_length = 0.0

    # 拼接特征向量
    features = np.array([
        width, length, lpw, stretch, spread, dispersion,
        compactness, convex_area, rectangularity, circularity,
        defensive_line, highest_player, centroid_goal_dist,
        avg_triangle_area, avg_edge_length
    ], dtype=np.float32)

    return features


def compute_template_features(pitch, formation_name, pos_to_vl=None):
    """
    计算mplsoccer模板的几何特征向量

    参数:
        pitch: mplsoccer Pitch对象
        formation_name: str, 阵型名称（如'442'）
        pos_to_vl: dict, 位置到VL的映射（可选，用于兼容）

    返回:
        features: numpy array of shape (15,) - 几何特征向量
        positions: numpy array of shape (N, 2) - 球员位置（去掉GK）
    """
    try:
        positions_obj = pitch.get_formation(formation_name)
    except:
        return None, None

    # 去掉守门员
    outfield = [p for p in positions_obj if p.name != 'GK']
    positions = np.array([(p.x, p.y) for p in outfield], dtype=np.float32)

    if len(positions) < 3:
        return None, None

    features = compute_geometric_feature_vector(positions)
    return features, positions


def cosine_similarity(v1, v2):
    """余弦相似度"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def euclidean_distance(v1, v2):
    """欧氏距离（归一化）"""
    return np.linalg.norm(v1 - v2)
