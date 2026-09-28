import pandas as pd
import numpy as np
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')


class UrbanMetricsCalculator:
    def __init__(self):
        self.results = None

    def calculate_file1_metrics(self, file1_path: str, col_mapping: Dict[str, str]) -> pd.DataFrame:
        """计算绿视度和天空开阔度"""
        df = pd.read_csv(file1_path)
        df['绿视度'] = df[col_mapping['vegetation']] / df[col_mapping['total_pixels']]
        df['天空开阔度'] = df[col_mapping['sky']] / df[col_mapping['total_pixels']]
        result = df[[col_mapping['lon'], col_mapping['lat'], '绿视度', '天空开阔度']]
        result.columns = ['经度', '纬度', '绿视度', '天空开阔度']
        return result.set_index(['经度', '纬度'])

    def calculate_file2_metrics(self, file2_path: str, col_mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """计算复杂度、围合感、通畅度（四个方向平均）"""
        df = pd.read_csv(file2_path)
        lon_col, lat_col, dir_col, total_col, sky_col = col_mapping['base']

        # 只保留0,90,180,270四个方向的数据
        valid_directions = ['0', '90', '180', '270']
        df = df[df[dir_col].astype(str).isin(valid_directions)]

        if df.empty:
            raise ValueError("没有找到四个方向的图片数据")

        metrics_by_direction = []

        for direction in valid_directions:
            dir_df = df[df[dir_col] == direction].copy()

            if len(dir_df) == 0:
                print(f"警告: 方向 {direction} 没有数据")
                continue

            # 1. 计算复杂度（信息熵）
            proportions = dir_df[col_mapping['complexity']].div(dir_df[total_col], axis=0)
            proportions = proportions.replace(0, np.nan)
            entropy = -np.sum(proportions * np.log2(proportions), axis=1)
            dir_df['复杂度'] = entropy.fillna(0)

            # 2. 计算围合感（确保分母不为零）
            enclosure_sum = dir_df[col_mapping['enclosure']].sum(axis=1)
            denominator = dir_df[total_col] - dir_df[sky_col]
            # 当分母为0时，围合感设为0（只有天空，无围合）
            dir_df['围合感'] = np.where(denominator > 0, enclosure_sum / denominator, 0)

            # 3. 计算车辆通畅度
            vehicle_sum = dir_df[col_mapping['vehicle']].sum(axis=1)
            # 查找道路列
            road_cols = [col for col in df.columns if 'road' in col.lower()]
            road_col = road_cols[0] if road_cols else None

            if road_col:
                road_sum = vehicle_sum + dir_df[road_col]
                # 分母为0时，通畅度设为1（无车辆）
                vehicle_access = np.where(road_sum > 0, 1 - (vehicle_sum / road_sum), 1.0)
            else:
                # 没有道路列，假设道路像素为0
                vehicle_access = np.where(vehicle_sum > 0, 0.0, 1.0)

            dir_df['车辆通畅度'] = vehicle_access

            # 4. 计算人行通畅度
            ped_col, sidewalk_col = col_mapping['pedestrian']
            denominator_ped = dir_df[ped_col] + dir_df[sidewalk_col]
            # 分母为0时，通畅度设为1（无行人）
            pedestrian_access = np.where(denominator_ped > 0,
                                         1 - (dir_df[ped_col] / denominator_ped), 1.0)
            dir_df['人行通畅度'] = pedestrian_access

            # 5. 计算最终通畅度（取最小值）
            dir_df['通畅度'] = dir_df[['车辆通畅度', '人行通畅度']].min(axis=1)

            metrics_by_direction.append(dir_df[[lon_col, lat_col, '复杂度', '围合感',
                                                '人行通畅度', '车辆通畅度', '通畅度']])

        # 合并所有方向并计算平均值
        all_directions = pd.concat(metrics_by_direction)

        # 按经纬度分组，计算四个方向的平均值
        result = all_directions.groupby([lon_col, lat_col]).mean().reset_index()
        result.columns = ['经度', '纬度', '复杂度', '围合感', '人行通畅度', '车辆通畅度', '通畅度']

        print(f"处理了 {len(metrics_by_direction)} 个方向的数据")
        print(f"最终得到 {len(result)} 个经纬度点的平均指标")

        return result.set_index(['经度', '纬度'])

    def merge_and_save(self, file1_result: pd.DataFrame, file2_result: pd.DataFrame,
                       output_path: str = 'urban_metrics.csv'):
        """save"""
        self.results = pd.concat([file1_result, file2_result], axis=1).reset_index()
        self.results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"save: {output_path}")
        return self.results


def quick_calculate(file1_path: str, file2_path: str,
                    file1_mapping: Dict[str, str] = None,
                    file2_mapping: Dict[str, List[str]] = None,
                    output_path: str = 'output_metrics.csv'):
    """compute"""
    if file1_mapping is None:
        file1_mapping = {
            'lon': 'longitude',
            'lat': 'latitude',
            'vegetation': 'vegetation',
            'sky': 'sky',
            'total_pixels': 'total_pixels'
        }

    if file2_mapping is None:
        file2_mapping = {
            'base': ['longitude', 'latitude', 'direction', 'total_pixels', 'sky'],
            'complexity': ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                           'traffic light', 'traffic sign', 'vegetation', 'terrain',
                           'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                           'motorcycle', 'bicycle'],
            'enclosure': ['building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign'],
            'vehicle': ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'],
            'pedestrian': ['person', 'sidewalk']
        }

    calculator = UrbanMetricsCalculator()
    file1_metrics = calculator.calculate_file1_metrics(file1_path, file1_mapping)
    file2_metrics = calculator.calculate_file2_metrics(file2_path, file2_mapping)
    return calculator.merge_and_save(file1_metrics, file2_metrics, output_path)


if __name__ == "__main__":
    file1_mapping = {
        'lon': 'longitude',
        'lat': 'latitude',
        'vegetation': 'vegetation',
        'sky': 'sky',
        'total_pixels': 'total_pixels'
    }

    file2_mapping = {
        'base': ['longitude', 'latitude', 'direction', 'total_pixels', 'sky'],
        'complexity': ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                       'traffic light', 'traffic sign', 'vegetation', 'terrain',
                       'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                       'motorcycle', 'bicycle'],
        'enclosure': ['building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign'],
        'vehicle': ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'],
        'pedestrian': ['person', 'sidewalk']
    }

    try:
        results = quick_calculate(
            file1_path='dir/result/shenzhen_virtual_filtered_final.csv',
            file2_path='dir/result/shenzhen_seg_all_final.csv',
            file1_mapping=file1_mapping,
            file2_mapping=file2_mapping,
            output_path='dir/result/final_results.csv'
        )
        print("\nfinish：")
        print(results.head())
        print(f"\nhave {len(results)} points")

        #
        print("\nsta_information：")
        print(results.describe())

    except Exception as e:
        print(f"error: {e}")
        import traceback

        traceback.print_exc()