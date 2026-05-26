import os
import math
import json
import numpy as np
import cv2
from skimage import measure
from collections import deque

# ==========================================
# 模块1: 物理坐标转换器 (PhysicalTransformer)
# 职责: 负责像素与米之间的数学转换
# ==========================================
class PhysicalTransformer:
    def __init__(self, origin_px, radius_px):
        """
        :param origin_px: 扇形中心点 (x, y) 像素坐标
        :param radius_px: 图像有效量程对应的像素半径
        """
        self.origin_px = np.array(origin_px)
        self.radius_px = radius_px

    def parse_metadata(self, txt_path):
        """解析声呐元数据TXT"""
        data = {}
        if not os.path.exists(txt_path): return None
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    try: data[k.strip()] = float(v.strip())
                    except: data[k.strip()] = v.strip()
        
        # 计算比例尺 (米/像素)
        bin_size = data['sound_velocity'] / (2 * data['sample_rate'])
        total_range_m = data['nRanges'] * bin_size
        scale = total_range_m / self.radius_px
        
        return {
            "scale": scale,
            "timestamp": data['timestamp_hardware'],
            "bin_size": bin_size,
            "range": total_range_m
        }

    def pixel_to_real(self, px_coord, scale):
        """像素坐标转物理坐标 (单位: 米)"""
        # px_coord: (x, y)
        dx = px_coord[0] - self.origin_px[0]
        dy = self.origin_px[1] - px_coord[1]  # 图像坐标系Y向下，物理向上
        return np.array([dx * scale, dy * scale])

# ==========================================
# 模块2: 目标提取器 (TargetExtractor)
# 职责: 从 Mask 中提取连通域属性
# ==========================================
class TargetExtractor:
    def __init__(self, threshold=0.5, min_area_px=5):
        self.threshold = threshold
        self.min_area_px = min_area_px

    def extract(self, prob_map):
        """提取当前帧中所有目标的像素特征"""
        binary = (prob_map > self.threshold).astype(np.uint8)
        labels = measure.label(binary, connectivity=2)
        props = measure.regionprops(labels)
        
        targets = []
        for p in props:
            if p.area < self.min_area_px: continue
            targets.append({
                "centroid_px": np.array([p.centroid[1], p.centroid[0]]), # (x, y)
                "area_px": p.area,
                "bbox_px": p.bbox, # (min_row, min_col, max_row, max_col)
                "major_axis_px": p.major_axis_length
            })
        return targets

# ==========================================
# 模块3: 动态分析器 (MotionAnalyzer)
# 职责: 跨帧目标关联、速度计算、尺寸转换
# ==========================================
class MotionAnalyzer:
    def __init__(self, max_history=5):
        # 存储每个目标的轨迹: { track_id: deque([ (time, pos_real, area_px), ... ]) }
        self.tracks = {}
        self.next_id = 0
        self.max_history = max_history

    def update(self, current_targets, timestamp, scale):
        """
        使用简单的质心距离匹配目标并计算速度
        """
        current_results = []
        matched_ids = set()

        # 1. 简单的质心匹配逻辑 (可根据需要替换为更强的匈牙利算法/SORT)
        for target in current_targets:
            pos_real = target['centroid_px'] * scale # 简化处理，实际应调用Transformer
            
            best_id = None
            min_dist = 2.0 # 阈值：2米内视为同一目标
            
            for tid, history in self.tracks.items():
                if tid in matched_ids: continue
                last_pos = history[-1][1]
                dist = np.linalg.norm(pos_real - last_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid
            
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = deque(maxlen=self.max_history)

            # 更新轨迹
            self.tracks[best_id].append((timestamp, pos_real, target['area_px'], target['major_axis_px']))
            matched_ids.add(best_id)

            # 2. 计算速度 (至少需要2帧数据)
            velocity = np.array([0.0, 0.0])
            speed = 0.0
            if len(self.tracks[best_id]) >= 2:
                t2, p2, _, _ = self.tracks[best_id][-1]
                t1, p1, _, _ = self.tracks[best_id][-2]
                dt = t2 - t1
                if dt > 0:
                    velocity = (p2 - p1) / dt
                    speed = np.linalg.norm(velocity)

            # 3. 物理尺寸计算
            # 面积 (m^2) = 像素面积 * scale^2
            real_area = target['area_px'] * (scale ** 2)
            # 长度 (m) = 像素长度 * scale
            real_length = target['major_axis_px'] * scale

            current_results.append({
                "track_id": best_id,
                "pos_m": pos_real,
                "velocity_m_s": velocity,
                "speed_m_s": speed,
                "area_m2": real_area,
                "length_m": real_length
            })

        # 清理消失的目标 (此处可添加生命周期管理)
        active_ids = list(self.tracks.keys())
        for tid in active_ids:
            if tid not in matched_ids:
                # 如果超过一定时间没更新，可以 pop 掉
                pass 

        return current_results

# ==========================================
# 4. 流程集成示例 (Main logic)
# ==========================================
def main_process_example():
    # 初始化配置
    transformer = PhysicalTransformer(origin_px=(391.5, 511), radius_px=399)
    extractor = TargetExtractor(threshold=0.5)
    analyzer = MotionAnalyzer()

    # 模拟推理循环 (假设你在第一个脚本的 evaluate_sequence 循环中)
    # for frame_idx in range(len(prob_maps)):
    
    # 模拟输入数据
    mock_prob_map = np.random.rand(512, 512) # 你的推理输出
    mock_txt_path = r"E:\AI\Code\WorkCode\Multi-beam\DataProcess\day1\testing\yuan\DataRecord_2025-12-08_110354\0585.txt"               # 对应的元数据
    
    # --- 开始计算 ---
    
    # Step A: 解析当前帧物理环境
    meta = transformer.parse_metadata(mock_txt_path)
    if not meta: return

    # Step B: 提取像素级目标
    pixel_targets = extractor.extract(mock_prob_map)

    # Step C: 计算物理属性与动态信息
    results = analyzer.update(pixel_targets, meta['timestamp'], meta['scale'])

    # Step D: 输出结果
    for res in results:
        print(f"Target ID: {res['track_id']}")
        print(f"  位置: X={res['pos_m'][0]:.2f}m, Y={res['pos_m'][1]:.2f}m")
        print(f"  尺寸: 长度={res['length_m']:.2f}m, 面积={res['area_m2']:.4f}m2")
        print(f"  速度: {res['speed_m_s']:.2f} m/s")
main_process_example()