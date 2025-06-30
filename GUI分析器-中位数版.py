#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚗 乘车体验对比分析系统 - 中位数版
======================================
专注于传统车vs自动驾驶车的客观对比分析

特点：
- 客观的中位数平衡因子算法
- 基于评分函数中位数特性
- 不受极端值影响的稳健权重计算

作者: 本科生研究项目
版本: v2.1 - 中位数平衡因子版
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import seaborn as sns
import os
import glob
from datetime import datetime
from math import radians, cos, sin, asin, sqrt, atan2
from scipy import signal, integrate, optimize
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式 - 增强版
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'Kaiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'Kaiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 强制设置中文字体
try:
    # 尝试使用微软雅黑
    from matplotlib.font_manager import FontProperties
    font_prop = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = font_prop.get_name()
except:
    # 备用字体设置
    plt.rcParams['font.family'] = ['sans-serif']

sns.set_style("whitegrid")
sns.set_palette("husl")

def 设置中文字体():
    """智能设置中文字体显示"""
    import matplotlib.font_manager as fm
    
    # 检查系统可用字体
    font_list = [f.name for f in fm.fontManager.ttflist]
    
    # 按优先级设置中文字体
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'FangSong', 'Kaiti', 'SimSun']
    
    for font in chinese_fonts:
        if font in font_list:
            plt.rcParams['font.sans-serif'] = [font]
            print(f"✅ 成功设置中文字体: {font}")
            return
    
    # 如果都没有，使用系统默认
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("⚠️ 未找到中文字体，使用默认字体")

# 初始化字体设置
设置中文字体()

class 数据分析引擎:
    """核心分析引擎 - 专注于对比分析"""
    def __init__(self, data_folder, price=None):
        self.data_folder = data_folder
        self.price = price
        self.results = {}
        
    def load_and_analyze(self):
        """加载数据并分析"""
        try:
            # 加载数据
            self.acc_data = pd.read_csv(os.path.join(self.data_folder, 'Accelerometer.csv'))
            self.gyro_data = pd.read_csv(os.path.join(self.data_folder, 'Gyroscope.csv'))
            self.gps_data = pd.read_csv(os.path.join(self.data_folder, 'Location.csv'))
            self.mic_data = pd.read_csv(os.path.join(self.data_folder, 'Microphone.csv'))
            
            # 数据同步处理
            self._synchronize_sensors()
            
            # 计算各项指标
            self._calculate_smoothness()
            self._calculate_stability()
            self._calculate_noise_level()
            self._calculate_time_efficiency()
            if self.price is not None:
                self._calculate_price_value()
            
            return True
        except Exception as e:
            print(f"分析失败: {e}")
            return False
    
    def _synchronize_sensors(self):
        """传感器数据时间同步"""
        min_time = max(
            self.acc_data['seconds_elapsed'].min(),
            self.gyro_data['seconds_elapsed'].min(),
            self.gps_data['seconds_elapsed'].min(),
            self.mic_data['seconds_elapsed'].min()
        )
        max_time = min(
            self.acc_data['seconds_elapsed'].max(),
            self.gyro_data['seconds_elapsed'].max(), 
            self.gps_data['seconds_elapsed'].max(),
            self.mic_data['seconds_elapsed'].max()
        )
        
        # 过滤到共同时间范围
        acc_mask = (self.acc_data['seconds_elapsed'] >= min_time) & (self.acc_data['seconds_elapsed'] <= max_time)
        self.acc_data = self.acc_data[acc_mask].copy().reset_index(drop=True)
        
        gyro_mask = (self.gyro_data['seconds_elapsed'] >= min_time) & (self.gyro_data['seconds_elapsed'] <= max_time)
        self.gyro_data = self.gyro_data[gyro_mask].copy().reset_index(drop=True)
        
        gps_mask = (self.gps_data['seconds_elapsed'] >= min_time) & (self.gps_data['seconds_elapsed'] <= max_time)
        self.gps_data = self.gps_data[gps_mask].copy().reset_index(drop=True)
        
        mic_mask = (self.mic_data['seconds_elapsed'] >= min_time) & (self.mic_data['seconds_elapsed'] <= max_time)
        self.mic_data = self.mic_data[mic_mask].copy().reset_index(drop=True)
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """使用Haversine公式精确计算距离"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        R = 6371000
        return R * c
    
    def _remove_gravity_component(self, acc_x, acc_y, acc_z, sample_rate=50):
        """正确去除重力分量"""
        try:
            nyquist = sample_rate / 2
            cutoff = 0.5 / nyquist
            b, a = signal.butter(4, cutoff, btype='high')
            acc_x_filtered = signal.filtfilt(b, a, acc_x)
            acc_y_filtered = signal.filtfilt(b, a, acc_y)
            acc_z_filtered = signal.filtfilt(b, a, acc_z)
            return acc_x_filtered, acc_y_filtered, acc_z_filtered
        except Exception as e:
            # 备用方案：移动平均去趋势
            window = min(100, len(acc_x) // 10)
            if window < 3:
                window = 3
            acc_x_series = pd.Series(acc_x)
            acc_y_series = pd.Series(acc_y)
            acc_z_series = pd.Series(acc_z)
            acc_x_ma = acc_x_series.rolling(window=window, center=True).mean()
            acc_y_ma = acc_y_series.rolling(window=window, center=True).mean()
            acc_z_ma = acc_z_series.rolling(window=window, center=True).mean()
            acc_x_ma = acc_x_ma.fillna(method='bfill').fillna(method='ffill')
            acc_y_ma = acc_y_ma.fillna(method='bfill').fillna(method='ffill')
            acc_z_ma = acc_z_ma.fillna(method='bfill').fillna(method='ffill')
            return (acc_x_series - acc_x_ma).values, (acc_y_series - acc_y_ma).values, (acc_z_series - acc_z_ma).values
    
    def _calculate_smoothness(self):
        """计算平顺性（基于ISO 2631-1频率加权RMS加速度）"""
        acc_x_clean, acc_y_clean, acc_z_clean = self._remove_gravity_component(
            self.acc_data['x'].values, 
            self.acc_data['y'].values, 
            self.acc_data['z'].values
        )
        
        # 计算三轴RMS加速度（符合ISO 2631-1标准）
        rms_acc_x = np.sqrt(np.mean(acc_x_clean**2))
        rms_acc_y = np.sqrt(np.mean(acc_y_clean**2))
        rms_acc_z = np.sqrt(np.mean(acc_z_clean**2))
        
        # 应用ISO 2631-1权重系数（简化版本）
        # Wd用于水平方向(x,y)，Wk用于垂直方向(z)
        # 在1-4Hz频率范围内，权重系数接近1
        weighted_rms_horizontal = np.sqrt(rms_acc_x**2 + rms_acc_y**2)
        weighted_rms_vertical = rms_acc_z
        
        # 总体频率加权RMS加速度
        total_weighted_rms = np.sqrt(weighted_rms_horizontal**2 + weighted_rms_vertical**2)
        
        # 基于ISO 2631-1标准的评分系统
        # 舒适度阈值：<0.315 m/s² (优秀), <0.63 m/s² (良好), <1.0 m/s² (尚可)
        if total_weighted_rms <= 0.315:
            score = 100 * np.exp(-2.0 * total_weighted_rms / 0.315)
        elif total_weighted_rms <= 0.63:
            score = 85 * np.exp(-1.5 * (total_weighted_rms - 0.315) / 0.315)
        elif total_weighted_rms <= 1.0:
            score = 60 * np.exp(-1.0 * (total_weighted_rms - 0.63) / 0.37)
        elif total_weighted_rms <= 2.0:
            score = 35 * np.exp(-0.8 * (total_weighted_rms - 1.0) / 1.0)
        else:
            score = 15 * np.exp(-0.5 * (total_weighted_rms - 2.0) / 2.0)
        
        self.results['平顺性'] = {
            'score': max(0.0, min(100.0, float(score))),
            'value': total_weighted_rms,
            'unit': 'm/s²'
        }
    
    def _calculate_stability(self):
        """计算稳定性（基于Continental真实驾驶数据）"""
        gyro_x = self.gyro_data['x'] - np.mean(self.gyro_data['x'])
        gyro_y = self.gyro_data['y'] - np.mean(self.gyro_data['y'])
        gyro_z = self.gyro_data['z'] - np.mean(self.gyro_data['z'])
        
        angular_velocity = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        rms_angular_velocity = np.sqrt(np.mean(angular_velocity**2))
        
        # 基于Continental汽车公司真实驾驶数据调整评分标准
        # 冷静驾驶: ±0.64 rad/s, 激进驾驶: ±1.24 rad/s, 城市驾驶: ±0.62 rad/s
        # 参考：Kontos et al. 2023, "Prediction for Future Yaw Rate Values of Vehicles"
        if rms_angular_velocity <= 0.175:      # ≈10°/s，优秀驾驶稳定性
            score = 100
        elif rms_angular_velocity <= 0.35:     # ≈20°/s，良好驾驶稳定性  
            score = 90 - 20 * (rms_angular_velocity - 0.175) / 0.175
        elif rms_angular_velocity <= 0.61:     # ≈35°/s，正常驾驶上限（城市/冷静驾驶）
            score = 70 - 30 * (rms_angular_velocity - 0.35) / 0.26
        elif rms_angular_velocity <= 1.0:      # ≈57°/s，激烈驾驶
            score = 40 - 25 * (rms_angular_velocity - 0.61) / 0.39
        elif rms_angular_velocity <= 1.5:      # ≈86°/s，极限驾驶
            score = 15 - 10 * (rms_angular_velocity - 1.0) / 0.5
        else:                                  # >86°/s，危险驾驶
            score = 5
            
        self.results['稳定性'] = {
            'score': max(0.0, min(100.0, float(score))),
            'value': rms_angular_velocity,
            'unit': 'rad/s'
        }
    
    def _calculate_noise_level(self):
        """计算噪声（基于WHO 2018环境噪声指南）"""
        # 获取有效的dBFS数据
        valid_db = self.mic_data[self.mic_data['dBFS'] > -150]['dBFS']
        
        if len(valid_db) == 0:
            avg_dbfs = -80
        else:
            avg_dbfs = np.mean(valid_db)
        
        # 将dBFS转换为近似的dB(A)声压级
        # 经验转换公式：dB(A) ≈ dBFS + 96 (针对手机麦克风)
        # 这是一个近似转换，实际需要麦克风校准
        estimated_dba = avg_dbfs + 96
        
        # 基于WHO 2018环境噪声指南的评分
        # 道路交通噪声建议值：<45 dB(A) Lnight, <53 dB(A) Lden
        # 车内噪声参考标准：<50 dB(A) 优秀，<60 dB(A) 良好
        if estimated_dba <= 45:
            score = 100  # 优秀：符合WHO夜间标准
        elif estimated_dba <= 50:
            score = 90 - 20 * (estimated_dba - 45) / 5  # 很好
        elif estimated_dba <= 55:
            score = 70 - 20 * (estimated_dba - 50) / 5  # 良好
        elif estimated_dba <= 65:
            score = 50 - 30 * (estimated_dba - 55) / 10  # 一般
        elif estimated_dba <= 75:
            score = 20 - 15 * (estimated_dba - 65) / 10  # 较差
        else:
            score = 5  # 差：超过75 dB(A)严重噪声污染
            
        self.results['噪声水平'] = {
            'score': max(0.0, min(100.0, float(score))),
            'value': estimated_dba,
            'unit': 'dB(A)'
        }
    
    def _calculate_time_efficiency(self):
        """计算时间效率（基于武汉市真实交通数据）"""
        total_distance = 0
        
        for i in range(1, len(self.gps_data)):
            lat1, lon1 = self.gps_data.iloc[i-1][['latitude', 'longitude']]
            lat2, lon2 = self.gps_data.iloc[i][['latitude', 'longitude']]
            distance = self._haversine_distance(lat1, lon1, lat2, lon2)
            total_distance += distance
            
        total_time = self.gps_data['seconds_elapsed'].iloc[-1] - self.gps_data['seconds_elapsed'].iloc[0]
        if total_time > 0:
            avg_speed_kmh = (total_distance / total_time) * 3.6
        else:
            avg_speed_kmh = 0
            
        # 基于武汉市真实交通数据调整评分标准
        # 武汉市平均速度27.75km/h，延时指数1.760
        if avg_speed_kmh >= 35:      # 优秀水平（高于武汉快速路况）
            score = 100
        elif avg_speed_kmh >= 27.75: # 武汉市实际平均速度
            score = 85 + 15 * (avg_speed_kmh - 27.75) / 7.25
        elif avg_speed_kmh >= 20:    # 轻度拥堵
            score = 65 + 20 * (avg_speed_kmh - 20) / 7.75
        elif avg_speed_kmh >= 15:    # 中度拥堵
            score = 40 + 25 * (avg_speed_kmh - 15) / 5
        elif avg_speed_kmh >= 10:    # 重度拥堵
            score = 20 + 20 * (avg_speed_kmh - 10) / 5
        else:                        # 严重拥堵
            score = 5 + 15 * avg_speed_kmh / 10
            
        self.results['时间效率'] = {
            'score': max(0.0, min(100.0, float(score))),
            'value': avg_speed_kmh,
            'unit': 'km/h'
        }
    
    def _calculate_price_value(self):
        """计算价格性价比（基于武汉市真实价格基准）"""
        if self.price is None:
            return
            
        total_distance = 0
        for i in range(1, len(self.gps_data)):
            lat1, lon1 = self.gps_data.iloc[i-1][['latitude', 'longitude']]
            lat2, lon2 = self.gps_data.iloc[i][['latitude', 'longitude']]
            distance = self._haversine_distance(lat1, lon1, lat2, lon2)
            total_distance += distance
        
        if total_distance > 0:
            price_per_km = self.price / (total_distance / 1000)
        else:
            price_per_km = self.price
        
        # 基于武汉市真实价格数据调整评分标准
        # 萝卜快跑：1.2元/km（补贴价），网约车：2.0-2.5元/km，出租车：2.5-3.0元/km
        if price_per_km <= 1.5:      # 萝卜快跑补贴价格区间
            score = 100
        elif price_per_km <= 2.0:    # 优惠网约车价格
            score = 90 - 20 * (price_per_km - 1.5) / 0.5
        elif price_per_km <= 2.5:    # 正常网约车价格
            score = 70 - 25 * (price_per_km - 2.0) / 0.5
        elif price_per_km <= 3.0:    # 出租车价格区间
            score = 45 - 25 * (price_per_km - 2.5) / 0.5
        elif price_per_km <= 4.0:    # 高价区间
            score = 20 - 15 * (price_per_km - 3.0) / 1.0
        else:                        # 过高价格
            score = 5 - 5 * min(1, (price_per_km - 4.0) / 2.0)
        
        self.results['价格性价比'] = {
            'score': max(0.0, min(100.0, float(score))),
            'value': price_per_km,
            'unit': '元/km'
        }
    
    def get_score_vector(self):
        """获取得分向量（用于多样本分析）"""
        indicators = ['平顺性', '稳定性', '噪声水平', '时间效率']
        if '价格性价比' in self.results:
            indicators.append('价格性价比')
        return [self.results[ind]['score'] for ind in indicators]

class 对比分析系统:
    """对比分析核心类"""
    def __init__(self):
        self.paired_data = {}
        
    def scan_paired_data(self):
        """扫描配对数据"""
        self.paired_data.clear()
        pattern = "*-[01]-*"
        folders = glob.glob(pattern)
        
        groups = {}
        for folder in folders:
            if os.path.isdir(folder):
                parts = folder.split('-')
                if len(parts) == 3:
                    name_part = parts[0]  # 改为通用的名字部分
                    car_type = parts[1]
                    price = parts[2]
                    
                    if name_part not in groups:
                        groups[name_part] = {}
                    
                    groups[name_part][car_type] = {
                        'folder': folder,
                        'price': float(price)
                    }
        
        # 只保留配对完整的数据（同一名字下有0和1）
        for name, data in groups.items():
            if '0' in data and '1' in data:
                self.paired_data[name] = data
                
        return self.paired_data
    
    def calculate_balance_factors(self, n_indicators):
        """计算平衡因子（基于评分函数中位数特性的稳健方法）"""
        
        # === 定义各指标的评分函数 ===
        def smoothness_score_func(rms_acc):
            """平顺性评分函数（基于ISO 2631-1）"""
            if rms_acc <= 0.315:
                return 100 * np.exp(-2.0 * rms_acc / 0.315)
            elif rms_acc <= 0.63:
                return 85 * np.exp(-1.5 * (rms_acc - 0.315) / 0.315)
            elif rms_acc <= 1.0:
                return 60 * np.exp(-1.0 * (rms_acc - 0.63) / 0.37)
            elif rms_acc <= 2.0:
                return 35 * np.exp(-0.8 * (rms_acc - 1.0) / 1.0)
            else:
                return 15 * np.exp(-0.5 * (rms_acc - 2.0) / 2.0)
        
        def stability_score_func(angular_vel):
            """稳定性评分函数（基于Continental数据）"""
            if angular_vel <= 0.175:
                return 100
            elif angular_vel <= 0.35:
                return 90 - 20 * (angular_vel - 0.175) / 0.175
            elif angular_vel <= 0.61:
                return 70 - 30 * (angular_vel - 0.35) / 0.26
            elif angular_vel <= 1.0:
                return 40 - 25 * (angular_vel - 0.61) / 0.39
            elif angular_vel <= 1.5:
                return 15 - 10 * (angular_vel - 1.0) / 0.5
            else:
                return 5
        
        def noise_score_func(db_a):
            """噪声评分函数（基于WHO 2018标准）"""
            if db_a <= 45:
                return 100
            elif db_a <= 50:
                return 90 - 20 * (db_a - 45) / 5
            elif db_a <= 55:
                return 70 - 20 * (db_a - 50) / 5
            elif db_a <= 65:
                return 50 - 30 * (db_a - 55) / 10
            elif db_a <= 75:
                return 20 - 15 * (db_a - 65) / 10
            else:
                return 5
        
        def efficiency_score_func(speed_kmh):
            """时间效率评分函数（基于武汉市数据）"""
            if speed_kmh >= 35:
                return 100
            elif speed_kmh >= 27.75:
                return 85 + 15 * (speed_kmh - 27.75) / 7.25
            elif speed_kmh >= 20:
                return 65 + 20 * (speed_kmh - 20) / 7.75
            elif speed_kmh >= 15:
                return 40 + 25 * (speed_kmh - 15) / 5
            elif speed_kmh >= 10:
                return 20 + 20 * (speed_kmh - 10) / 5
            else:
                return 5 + 15 * speed_kmh / 10
        
        def price_score_func(price_per_km):
            """价格评分函数（基于武汉市价格）"""
            if price_per_km <= 1.5:
                return 100
            elif price_per_km <= 2.0:
                return 90 - 20 * (price_per_km - 1.5) / 0.5
            elif price_per_km <= 2.5:
                return 70 - 25 * (price_per_km - 2.0) / 0.5
            elif price_per_km <= 3.0:
                return 45 - 25 * (price_per_km - 2.5) / 0.5
            elif price_per_km <= 4.0:
                return 20 - 15 * (price_per_km - 3.0) / 1.0
            else:
                return max(0, 5 - 5 * min(1, (price_per_km - 4.0) / 2.0))
        
        # === 定义各指标的搜索范围（基于物理意义） ===
        input_ranges = [
            (0.0, 3.0),      # 平顺性: RMS加速度 0-3 m/s²
            (0.0, 2.0),      # 稳定性: 角速度 0-2 rad/s  
            (40.0, 80.0),    # 噪声: 40-80 dB(A)
            (5.0, 50.0),     # 时间效率: 速度 5-50 km/h
            (0.5, 5.0)       # 价格: 0.5-5 元/km
        ]
        
        score_functions = [
            smoothness_score_func,
            stability_score_func, 
            noise_score_func,
            efficiency_score_func,
            price_score_func
        ]
        
        # === 计算中位数输入值（解方程：score_function(x) = 50） ===
        median_inputs = []
        target_score = 50.0  # 中位数目标分数
        
        for i in range(n_indicators):
            func = score_functions[i]
            input_range = input_ranges[i]
            
            # 定义求解方程：score_function(x) - 50 = 0
            def equation(x):
                return func(x) - target_score
            
            try:
                # 使用scipy.optimize.brentq求解方程
                # 首先检查边界值
                f_min = equation(input_range[0])
                f_max = equation(input_range[1])
                
                if f_min * f_max > 0:
                    # 如果没有根，使用二分搜索找到最接近50分的输入值
                    x_samples = np.linspace(input_range[0], input_range[1], 1000)
                    scores = [func(x) for x in x_samples]
                    distances = [abs(score - target_score) for score in scores]
                    min_idx = np.argmin(distances)
                    median_input = x_samples[min_idx]
                else:
                    # 使用Brent方法求解
                    median_input = optimize.brentq(equation, input_range[0], input_range[1])
                
                median_inputs.append(median_input)
                
            except Exception as e:
                # 备用方案：数值搜索
                x_samples = np.linspace(input_range[0], input_range[1], 1000)
                scores = [func(x) for x in x_samples]
                distances = [abs(score - target_score) for score in scores]
                min_idx = np.argmin(distances)
                median_input = x_samples[min_idx]
                median_inputs.append(median_input)
        
        # === 计算中位数输入值的归一化位置 ===
        median_positions = []
        for i in range(n_indicators):
            input_range = input_ranges[i]
            # 计算中位数输入值在其范围内的相对位置 (0-1)
            position = (median_inputs[i] - input_range[0]) / (input_range[1] - input_range[0])
            median_positions.append(position)
        
        # === 计算平衡因子 ===
        # 位置越靠前（越容易达到50分），需要的平衡因子越小
        # 位置越靠后（越难达到50分），需要的平衡因子越大
        target_position = 0.5  # 理想的中位数位置（范围中点）
        
        balance_factors = np.array([target_position / pos if pos > 0.01 else 1.0 
                                   for pos in median_positions])
        
        # 归一化平衡因子（保持相对比例）
        balance_factors = balance_factors / np.mean(balance_factors)
        
        return balance_factors, median_inputs, median_positions

    def calculate_entropy_weights(self, score_matrix):
        """计算双层权重系统：平衡因子 × 熵权重"""
        decision_matrix = np.array(score_matrix)
        n_samples, n_indicators = decision_matrix.shape
        
        # === 第一层：传统熵权重计算 ===
        # 归一化处理
        col_sums = decision_matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1
        normalized_matrix = decision_matrix / col_sums
        
        # 计算熵值
        entropies = []
        ln_n = np.log(n_samples)
        
        for j in range(n_indicators):
            col_data = normalized_matrix[:, j] + 1e-10
            entropy_j = -1/ln_n * np.sum(col_data * np.log(col_data))
            entropies.append(entropy_j)
        
        entropies = np.array(entropies)
        
        # 计算原始熵权重
        divergences = 1 - entropies
        original_weights = divergences / np.sum(divergences)
        
        # === 第二层：中位数平衡因子修正 ===
        balance_factors, median_inputs, median_positions = self.calculate_balance_factors(n_indicators)
        
        # 计算最终权重：平衡因子 × 熵权重
        corrected_weights = balance_factors * original_weights
        final_weights = corrected_weights / np.sum(corrected_weights)  # 重新归一化
        
        # 返回详细信息（用于调试和展示）
        return {
            'final_weights': final_weights,
            'original_weights': original_weights,
            'balance_factors': balance_factors,
            'median_inputs': median_inputs,
            'median_positions': median_positions,
            'correction_effect': np.abs(final_weights - original_weights).sum()
        }
    
    def single_group_comparison(self, group_name):
        """单组对比分析"""
        if group_name not in self.paired_data:
            return None
            
        group_data = self.paired_data[group_name]
        
        # 分析传统车和自动驾驶车
        trad_engine = 数据分析引擎(group_data['0']['folder'], group_data['0']['price'])
        auto_engine = 数据分析引擎(group_data['1']['folder'], group_data['1']['price'])
        
        if not (trad_engine.load_and_analyze() and auto_engine.load_and_analyze()):
            return None
        
        # 构建决策矩阵
        trad_scores = trad_engine.get_score_vector()
        auto_scores = auto_engine.get_score_vector()
        score_matrix = [trad_scores, auto_scores]
        
        # 计算双层权重系统
        weight_info = self.calculate_entropy_weights(score_matrix)
        final_weights = weight_info['final_weights']
        
        # 计算加权得分
        trad_final = np.dot(trad_scores, final_weights)
        auto_final = np.dot(auto_scores, final_weights)
        
        return {
            'group_name': group_name,
            'traditional': {'engine': trad_engine, 'scores': trad_scores, 'final': trad_final},
            'autonomous': {'engine': auto_engine, 'scores': auto_scores, 'final': auto_final},
            'weights': final_weights,
            'weight_info': weight_info,  # 包含详细的权重分析信息
            'indicators': ['平顺性', '稳定性', '噪声水平', '时间效率', '价格性价比'][:len(final_weights)]
        }
    
    def multi_group_comparison(self, selected_groups=None):
        """多组综合对比分析"""
        if selected_groups is None:
            selected_groups = list(self.paired_data.keys())
        
        all_traditional_scores = []
        all_autonomous_scores = []
        valid_groups = []
        
        # 收集所有数据
        for group_name in selected_groups:
            if group_name in self.paired_data:
                result = self.single_group_comparison(group_name)
                if result:
                    all_traditional_scores.append(result['traditional']['scores'])
                    all_autonomous_scores.append(result['autonomous']['scores'])
                    valid_groups.append(group_name)
        
        if len(valid_groups) < 2:
            return None
        
        # 构建大决策矩阵
        all_scores = all_traditional_scores + all_autonomous_scores
        
        # 计算多样本双层权重系统
        weight_info = self.calculate_entropy_weights(all_scores)
        final_weights = weight_info['final_weights']
        
        # 计算每次行程的加权得分
        trad_weighted_scores = [np.dot(scores, final_weights) for scores in all_traditional_scores]
        auto_weighted_scores = [np.dot(scores, final_weights) for scores in all_autonomous_scores]
        
        # 计算最终综合得分
        trad_final = np.mean(trad_weighted_scores)
        auto_final = np.mean(auto_weighted_scores)
        
        return {
            'groups': valid_groups,
            'sample_count': len(all_scores),
            'traditional_final': trad_final,
            'autonomous_final': auto_final,
            'traditional_scores': all_traditional_scores,
            'autonomous_scores': all_autonomous_scores,
            'weights': final_weights,
            'weight_info': weight_info,  # 包含详细的权重分析信息
            'indicators': ['平顺性', '稳定性', '噪声水平', '时间效率', '价格性价比'][:len(final_weights)]
        }

class 乘车体验对比分析GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚗 乘车体验对比分析系统 v2.1 - 中位数版")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f8f9fa')
        
        # 设置窗口居中
        self.center_window()
        
        self.analysis_system = 对比分析系统()
        self.create_modern_ui()
        self.scan_data()
        
    def center_window(self):
        """窗口居中显示"""
        self.root.update_idletasks()
        width = 1600
        height = 1000
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_modern_ui(self):
        """创建现代化界面"""
        # 主容器
        main_container = tk.Frame(self.root, bg='#f8f9fa')
        main_container.pack(fill='both', expand=True)
        
        # 顶部标题栏
        self.create_header(main_container)
        
        # 主要内容区域
        content_area = tk.Frame(main_container, bg='#f8f9fa')
        content_area.pack(fill='both', expand=True, padx=30, pady=20)
        
        # 创建两个主要模式的卡片
        self.create_mode_cards(content_area)
        
        # 底部状态栏
        self.create_status_bar(main_container)
    
    def create_header(self, parent):
        """创建顶部标题栏"""
        header = tk.Frame(parent, bg='#2c3e50', height=120)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        # 主标题
        title_label = tk.Label(header, text="🚗 乘车体验对比分析系统 - 中位数版", 
                              font=('微软雅黑', 24, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # 副标题：算法说明
        subtitle_label = tk.Label(header, text="基于评分函数中位数特性的稳健平衡因子算法", 
                                 font=('微软雅黑', 14), fg='#ecf0f1', bg='#2c3e50')
        subtitle_label.pack(pady=5)
        
        # 副标题
        subtitle_label = tk.Label(header, text="智能分析 • 科学对比 • 直观展示", 
                                 font=('微软雅黑', 16), fg='#ecf0f1', bg='#2c3e50')
        subtitle_label.pack()
    
    def create_mode_cards(self, parent):
        """创建两个分析模式的卡片"""
        # 卡片容器
        cards_frame = tk.Frame(parent, bg='#f8f9fa')
        cards_frame.pack(fill='both', expand=True)
        
        # 左侧卡片：单组精准对比
        self.create_single_mode_card(cards_frame)
        
        # 右侧卡片：多组综合对比  
        self.create_multi_mode_card(cards_frame)
    
    def create_single_mode_card(self, parent):
        """创建单组对比模式卡片"""
        # 左侧卡片框架
        left_card = tk.Frame(parent, bg='white', relief='raised', bd=2)
        left_card.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # 卡片标题
        title_frame = tk.Frame(left_card, bg='#3498db', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="🎯 单组精准对比", 
                font=('微软雅黑', 20, 'bold'), fg='white', bg='#3498db').pack(pady=25)
        
        # 卡片内容
        content_frame = tk.Frame(left_card, bg='white', padx=30, pady=30)
        content_frame.pack(fill='both', expand=True)
        
        # 步骤指导
        steps_text = """📋 操作步骤：
1️⃣ 从下方列表选择要分析的数据组
2️⃣ 点击 "开始单组对比" 按钮
3️⃣ 查看详细的对比分析结果

💡 适用场景：
• 想了解某次具体出行的体验差异
• 需要详细的单次对比数据
• 分析特定条件下的表现"""
        
        tk.Label(content_frame, text=steps_text, font=('微软雅黑', 11), 
                bg='white', justify='left', anchor='nw').pack(anchor='nw')
        
        # 数据选择区域
        tk.Label(content_frame, text="📊 选择数据组：", 
                font=('微软雅黑', 14, 'bold'), bg='white').pack(anchor='w', pady=(20, 10))
        
        # 数据列表框
        listbox_frame = tk.Frame(content_frame, bg='white')
        listbox_frame.pack(fill='x', pady=(0, 20))
        
        self.single_listbox = tk.Listbox(listbox_frame, height=8, font=('微软雅黑', 11),
                                        selectmode='single', bg='#f8f9fa', 
                                        selectbackground='#3498db', selectforeground='white')
        scrollbar1 = tk.Scrollbar(listbox_frame, orient="vertical", command=self.single_listbox.yview)
        self.single_listbox.configure(yscrollcommand=scrollbar1.set)
        
        self.single_listbox.pack(side="left", fill="both", expand=True)
        scrollbar1.pack(side="right", fill="y")
        
        # 开始分析按钮
        start_btn1 = tk.Button(content_frame, text="🚀 开始单组对比", 
                              font=('微软雅黑', 14, 'bold'), bg='#3498db', fg='white',
                              relief='flat', padx=30, pady=10, cursor='hand2',
                              command=self.single_day_analysis)
        start_btn1.pack(anchor='w')
    
    def create_multi_mode_card(self, parent):
        """创建多组对比模式卡片"""
        # 右侧卡片框架
        right_card = tk.Frame(parent, bg='white', relief='raised', bd=2)
        right_card.pack(side='right', fill='both', expand=True, padx=(15, 0))
        
        # 卡片标题
        title_frame = tk.Frame(right_card, bg='#e74c3c', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="📈 多组综合对比", 
                font=('微软雅黑', 20, 'bold'), fg='white', bg='#e74c3c').pack(pady=25)
        
        # 卡片内容
        content_frame = tk.Frame(right_card, bg='white', padx=30, pady=30)
        content_frame.pack(fill='both', expand=True)
        
        # 步骤指导
        steps_text = """📋 操作步骤：
1️⃣ 勾选下方要包含在分析中的数据组
2️⃣ 至少选择2组数据（建议3组以上）
3️⃣ 点击 "开始多组对比" 按钮
4️⃣ 获得统计学意义的综合结论

💡 适用场景：
• 想了解整体期间的综合表现
• 需要统计学意义的结论
• 分析长期使用体验差异"""
        
        tk.Label(content_frame, text=steps_text, font=('微软雅黑', 11), 
                bg='white', justify='left', anchor='nw').pack(anchor='nw')
        
        # 数据选择区域
        select_frame = tk.Frame(content_frame, bg='white')
        select_frame.pack(fill='both', expand=True, pady=(20, 0))
        
        # 选择标题和全选按钮
        select_header = tk.Frame(select_frame, bg='white')
        select_header.pack(fill='x', pady=(0, 10))
        
        tk.Label(select_header, text="📊 选择数据组：", 
                font=('微软雅黑', 14, 'bold'), bg='white').pack(side='left')
        
        select_all_btn = tk.Button(select_header, text="全选", 
                                  font=('微软雅黑', 10), bg='#f39c12', fg='white',
                                  relief='flat', padx=15, pady=2, cursor='hand2',
                                  command=self.select_all_data)
        select_all_btn.pack(side='right')
        
        clear_all_btn = tk.Button(select_header, text="清空", 
                                 font=('微软雅黑', 10), bg='#95a5a6', fg='white',
                                 relief='flat', padx=15, pady=2, cursor='hand2',
                                 command=self.clear_all_data)
        clear_all_btn.pack(side='right', padx=(0, 10))
        
        # 复选框容器
        checkbox_container = tk.Frame(select_frame, bg='white')
        checkbox_container.pack(fill='both', expand=True)
        
        # 滚动框架
        canvas = tk.Canvas(checkbox_container, bg='#f8f9fa', height=200)
        scrollbar2 = tk.Scrollbar(checkbox_container, orient="vertical", command=canvas.yview)
        self.checkboxes_frame = tk.Frame(canvas, bg='#f8f9fa')
        
        self.checkboxes_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.checkboxes_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar2.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar2.pack(side="right", fill="y")
        
        # 开始分析按钮
        start_btn2 = tk.Button(content_frame, text="🚀 开始多组对比", 
                              font=('微软雅黑', 14, 'bold'), bg='#e74c3c', fg='white',
                              relief='flat', padx=30, pady=10, cursor='hand2',
                              command=self.multi_day_analysis)
        start_btn2.pack(anchor='w', pady=(20, 0))
        
        # 初始化复选框变量
        self.date_vars = {}
    
    def create_status_bar(self, parent):
        """创建底部状态栏"""
        status_bar = tk.Frame(parent, bg='#34495e', height=40)
        status_bar.pack(fill='x', side='bottom')
        status_bar.pack_propagate(False)
        
        self.status_label = tk.Label(status_bar, text="💾 就绪 | 等待数据选择...", 
                                    font=('微软雅黑', 10), fg='white', bg='#34495e')
        self.status_label.pack(side='left', padx=20, pady=10)
        
        # 版本信息
        version_label = tk.Label(status_bar, text="v2.1 直观UI版", 
                                font=('微软雅黑', 10), fg='#bdc3c7', bg='#34495e')
        version_label.pack(side='right', padx=20, pady=10)
    
    def select_all_data(self):
        """全选所有数据"""
        for var in self.date_vars.values():
            var.set(True)
        self.update_status("✅ 已全选所有数据组")
    
    def clear_all_data(self):
        """清空所有选择"""
        for var in self.date_vars.values():
            var.set(False)
        self.update_status("🔄 已清空所有选择")
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)
        self.root.after(3000, lambda: self.status_label.config(text="💾 就绪 | 等待数据选择..."))
    
    def scan_data(self):
        """扫描配对数据并更新界面"""
        paired_data = self.analysis_system.scan_paired_data()
        
        # 清空现有数据
        self.single_listbox.delete(0, tk.END)
        
        # 清空复选框
        for widget in self.checkboxes_frame.winfo_children():
            widget.destroy()
        self.date_vars.clear()
        
        # 添加数据到界面
        dates = sorted(paired_data.keys())
        for i, date in enumerate(dates):
            trad_price = paired_data[date]['0']['price']
            auto_price = paired_data[date]['1']['price']
            display_text = f"📊 {date} (传统车:{trad_price}元 vs 自动驾驶:{auto_price}元)"
            
            # 添加到单选列表
            self.single_listbox.insert(tk.END, display_text)
            
            # 添加到多选复选框
            var = tk.BooleanVar()
            self.date_vars[date] = var
            checkbox = tk.Checkbutton(self.checkboxes_frame, text=display_text, 
                                    variable=var, font=('微软雅黑', 11), bg='#f8f9fa',
                                    selectcolor='#3498db', activebackground='#f8f9fa')
            checkbox.pack(anchor='w', pady=3, padx=10, fill='x')
        
        # 默认选择第一项
        if dates:
            self.single_listbox.selection_set(0)
            
        # 更新状态
        if len(dates) > 0:
            self.update_status(f"📊 扫描完成 | 发现 {len(dates)} 组配对数据")
        else:
            self.update_status("⚠️ 未发现配对数据 | 请检查数据文件")
    
    def single_day_analysis(self):
        """执行单组分析"""
        selection = self.single_listbox.curselection()
        if not selection:
            messagebox.showwarning("选择提示", "请先选择要分析的数据组！")
            return
        
        groups = list(self.analysis_system.paired_data.keys())
        selected_group = groups[selection[0]]
        
        self.update_status("🔄 正在进行单组精准对比分析...")
        
        # 创建结果窗口
        self.create_result_window("单组精准对比", lambda: self.analysis_system.single_group_comparison(selected_group))
    
    def multi_day_analysis(self):
        """执行多组分析"""
        selected_groups = [group for group, var in self.date_vars.items() if var.get()]
        
        if len(selected_groups) < 2:
            messagebox.showwarning("选择提示", "多组分析至少需要选择2组数据！\n建议选择3组以上获得更可靠的结果。")
            return
        
        self.update_status(f"🔄 正在分析 {len(selected_groups)} 组数据...")
        
        # 创建结果窗口
        self.create_result_window("多组综合对比", lambda: self.analysis_system.multi_group_comparison(selected_groups))
    
    def create_result_window(self, title, analysis_func):
        """创建结果显示窗口"""
        # 创建新窗口
        result_window = tk.Toplevel(self.root)
        result_window.title(f"📊 {title} - 分析结果")
        result_window.geometry("1400x900")
        result_window.configure(bg='white')
        
        # 窗口居中
        result_window.update_idletasks()
        x = (result_window.winfo_screenwidth() // 2) - 700
        y = (result_window.winfo_screenheight() // 2) - 450
        result_window.geometry(f'1400x900+{x}+{y}')
        
        # 加载提示
        loading_frame = tk.Frame(result_window, bg='white')
        loading_frame.pack(expand=True, fill='both')
        
        loading_label = tk.Label(loading_frame, text="🔄 正在分析数据...\n请稍候", 
                                font=('微软雅黑', 18), bg='white', fg='#7f8c8d')
        loading_label.pack(expand=True)
        
        # 更新界面
        result_window.update()
        
        # 执行分析
        try:
            result = analysis_func()
            if result:
                # 清空加载界面
                for widget in loading_frame.winfo_children():
                    widget.destroy()
                
                # 显示结果
                self.show_enhanced_result(loading_frame, result, title.startswith("多组"))
                self.update_status("✅ 分析完成！")
            else:
                messagebox.showerror("分析错误", "分析过程中出现错误，请检查数据完整性！")
                result_window.destroy()
                self.update_status("❌ 分析失败")
        except Exception as e:
            messagebox.showerror("系统错误", f"系统运行出错：{str(e)}")
            result_window.destroy()
            self.update_status("❌ 系统错误")
    
    def show_enhanced_result(self, parent, result, is_multi_day):
        """显示增强的分析结果"""
        # 主容器
        main_frame = tk.Frame(parent, bg='white')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 标题和摘要
        self.create_result_header(main_frame, result, is_multi_day)
        
        # 图表区域
        self.create_enhanced_charts(main_frame, result, is_multi_day)
    
    def create_result_header(self, parent, result, is_multi_day):
        """创建结果标题和摘要"""
        header_frame = tk.Frame(parent, bg='#ecf0f1', relief='raised', bd=1)
        header_frame.pack(fill='x', pady=(0, 20))
        
        if is_multi_day:
            title = f"📈 多组综合对比分析"
            trad_score = result['traditional_final']
            auto_score = result['autonomous_final']
            sample_info = f"样本数量: {result['sample_count']}个"
        else:
            title = f"🎯 单组精准对比分析 - {result['group_name']}"
            trad_score = result['traditional']['final']
            auto_score = result['autonomous']['final']
            sample_info = f"分析组: {result['group_name']}"
        
        advantage = auto_score - trad_score
        
        # 标题
        tk.Label(header_frame, text=title, font=('微软雅黑', 18, 'bold'), 
                bg='#ecf0f1', fg='#2c3e50').pack(pady=15)
        
        # 核心结果
        result_frame = tk.Frame(header_frame, bg='#ecf0f1')
        result_frame.pack(pady=(0, 15))
        
        # 得分显示
        scores_frame = tk.Frame(result_frame, bg='#ecf0f1')
        scores_frame.pack()
        
        tk.Label(scores_frame, text=f"🚗 传统车得分: {trad_score:.1f}分", 
                font=('微软雅黑', 14), bg='#ecf0f1', fg='#e74c3c').pack(side='left', padx=20)
        
        tk.Label(scores_frame, text="VS", font=('微软雅黑', 14, 'bold'), 
                bg='#ecf0f1', fg='#7f8c8d').pack(side='left', padx=10)
        
        tk.Label(scores_frame, text=f"🤖 自动驾驶车得分: {auto_score:.1f}分", 
                font=('微软雅黑', 14), bg='#ecf0f1', fg='#3498db').pack(side='left', padx=20)
        
        # 优势分析
        advantage_color = '#27ae60' if advantage > 0 else '#e74c3c' if advantage < 0 else '#f39c12'
        advantage_text = "自动驾驶车优势" if advantage > 0 else "传统车优势" if advantage < 0 else "平分秋色"
        
        tk.Label(result_frame, text=f"📊 {advantage_text}: {abs(advantage):.1f}分 | {sample_info}", 
                font=('微软雅黑', 12), bg='#ecf0f1', fg=advantage_color).pack(pady=5)
    
    def create_enhanced_charts(self, parent, result, is_multi_day):
        """创建增强的图表显示"""
        # 创建图表
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('white')
        
        if is_multi_day:
            # 多组对比布局：2x2 - 优化间距
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35, 
                                 left=0.08, right=0.95, top=0.92, bottom=0.08)
            
            # 综合得分对比
            ax1 = fig.add_subplot(gs[0, 0])
            self.plot_comprehensive_comparison(ax1, result)
            
            # 指标权重分析
            ax2 = fig.add_subplot(gs[0, 1])
            self.plot_weights_analysis(ax2, result)
            
            # 得分分布对比
            ax3 = fig.add_subplot(gs[1, 0])
            self.plot_score_distribution_enhanced(ax3, result)
            
            # 趋势分析
            ax4 = fig.add_subplot(gs[1, 1])
            self.plot_trend_analysis(ax4, result)
            
        else:
            # 单组对比布局：2x2 - 优化间距
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35,
                                 left=0.08, right=0.95, top=0.92, bottom=0.08)
            
            # 雷达图对比
            ax1 = fig.add_subplot(gs[0, 0], projection='polar')
            self.plot_radar_comparison(ax1, result)
            
            # 指标详细对比
            ax2 = fig.add_subplot(gs[0, 1])
            self.plot_detailed_indicators(ax2, result)
            
            # 权重重要性
            ax3 = fig.add_subplot(gs[1, 0])
            self.plot_weights_analysis(ax3, result)
            
            # 原始数据对比
            ax4 = fig.add_subplot(gs[1, 1])
            self.plot_raw_data_comparison(ax4, result)
        
        # 嵌入图表
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_comprehensive_comparison(self, ax, result):
        """绘制综合得分对比"""
        trad_score = result['traditional_final']
        auto_score = result['autonomous_final']
        
        categories = ['传统车', '自动驾驶车']
        scores = [trad_score, auto_score]
        colors = ['#e74c3c', '#3498db']
        
        bars = ax.bar(categories, scores, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}分', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_title('📊 综合得分对比', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('得分', fontsize=12)
        ax.set_ylim(0, max(scores) * 1.2)
        ax.grid(True, alpha=0.3)
    
    def plot_weights_analysis(self, ax, result):
        """绘制双层权重系统分析"""
        indicators = result['indicators']
        weight_info = result.get('weight_info', {})
        
        if 'original_weights' in weight_info:
            # 双层权重系统：智能显示逻辑
            original_weights = weight_info['original_weights']
            final_weights = weight_info['final_weights']
            balance_factors = weight_info['balance_factors']
            correction_effect = weight_info.get('correction_effect', 0)
            
            # 判断是否需要显示对比（指标数量>=3且修正效果明显）
            should_show_comparison = (len(indicators) >= 3 and correction_effect > 0.01)
            
            y_pos = np.arange(len(indicators))
            
            if should_show_comparison:
                # 显示完整的双层权重对比
                height = 0.35
                
                # 绘制原始权重和修正后权重
                bars1 = ax.barh(y_pos - height/2, original_weights, height, 
                               label='原始熵权重', color='#95a5a6', alpha=0.7)
                bars2 = ax.barh(y_pos + height/2, final_weights, height, 
                               label='修正后权重', color='#f39c12', alpha=0.9)
                
                # 添加数值标签和平衡因子
                for i, (orig, final, balance) in enumerate(zip(original_weights, final_weights, balance_factors)):
                    # 原始权重标签
                    ax.text(orig + 0.005, bars1[i].get_y() + bars1[i].get_height()/2,
                           f'{orig:.3f}', ha='left', va='center', fontsize=9, color='gray')
                    
                    # 修正后权重标签
                    ax.text(final + 0.005, bars2[i].get_y() + bars2[i].get_height()/2,
                           f'{final:.3f}', ha='left', va='center', fontsize=10, 
                           fontweight='bold')
                    
                    # 平衡因子标签（放在中间位置，对齐指标行）
                    max_width = max(max(original_weights), max(final_weights))
                    # 计算两个条形图中间的Y位置
                    middle_y = (bars1[i].get_y() + bars1[i].get_height()/2 + 
                               bars2[i].get_y() + bars2[i].get_height()/2) / 2
                    ax.text(max_width * 0.8, middle_y, f'×{balance:.2f}', 
                           ha='center', va='center', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
                
                ax.set_title('⚖️ 双层权重系统分析', fontsize=14, fontweight='bold', pad=20)
                ax.legend(loc='lower right', fontsize=9)
                
            else:
                # 简化显示：仅显示最终权重
                bars = ax.barh(y_pos, final_weights, color='#f39c12', alpha=0.9)
                
                # 添加数值标签和平衡因子
                for i, (final, balance) in enumerate(zip(final_weights, balance_factors)):
                    # 最终权重标签
                    ax.text(final + 0.005, bars[i].get_y() + bars[i].get_height()/2,
                           f'{final:.3f}', ha='left', va='center', fontsize=10, 
                           fontweight='bold')
                    
                    # 如果平衡因子不是1.0，显示修正标记
                    if abs(balance - 1.0) > 0.05:  # 修正幅度超过5%才显示
                        ax.text(final * 0.7, bars[i].get_y() + bars[i].get_height()/2, 
                               f'×{balance:.2f}', ha='center', va='center', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
                
                # 显示简化标题
                if len(indicators) < 3:
                    ax.set_title('⚖️ 权重分布（数据有限）', fontsize=14, fontweight='bold', pad=20)
                else:
                    ax.set_title('⚖️ 修正后权重分布', fontsize=14, fontweight='bold', pad=20)
            
            ax.set_xlabel('权重值', fontsize=12)
            
        else:
            # 传统权重显示（兼容旧版本）
            weights = result['weights']
            y_pos = np.arange(len(indicators))
            bars = ax.barh(y_pos, weights, color='#f39c12', alpha=0.8)
            
            # 添加数值标签
            for i, (bar, weight) in enumerate(zip(bars, weights)):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                       f'{weight:.3f}', ha='left', va='center', fontsize=10)
            
            ax.set_title('⚖️ 指标权重重要性', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('权重值', fontsize=12)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(indicators, fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
    
    def plot_score_distribution_enhanced(self, ax, result):
        """绘制得分分布对比"""
        trad_scores = result['traditional_scores']
        auto_scores = result['autonomous_scores']
        
        # 计算每个指标的平均得分
        indicators = result['indicators']
        trad_means = np.mean(trad_scores, axis=0)
        auto_means = np.mean(auto_scores, axis=0)
        
        x = np.arange(len(indicators))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, trad_means, width, label='传统车', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, auto_means, width, label='自动驾驶车', color='#3498db', alpha=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('指标', fontsize=12)
        ax.set_ylabel('平均得分', fontsize=12)
        ax.set_title('📈 各指标平均得分对比', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(indicators, rotation=30, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def plot_trend_analysis(self, ax, result):
        """绘制趋势分析"""
        trad_scores = result['traditional_scores']
        auto_scores = result['autonomous_scores']
        
        # 计算综合得分趋势
        trad_finals = [np.dot(scores, result['weights']) for scores in trad_scores]
        auto_finals = [np.dot(scores, result['weights']) for scores in auto_scores]
        
        x = range(1, len(trad_finals) + 1)
        
        ax.plot(x, trad_finals, 'o-', color='#e74c3c', linewidth=2, markersize=6, label='传统车', alpha=0.8)
        ax.plot(x, auto_finals, 's-', color='#3498db', linewidth=2, markersize=6, label='自动驾驶车', alpha=0.8)
        
        ax.set_xlabel('数据组序号', fontsize=12)
        ax.set_ylabel('综合得分', fontsize=12)
        ax.set_title('📊 得分趋势变化', fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_radar_comparison(self, ax, result):
        """绘制雷达图对比"""
        indicators = result['indicators']
        trad_scores = result['traditional']['scores']
        auto_scores = result['autonomous']['scores']
        
        # 设置雷达图
        angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        trad_values = np.concatenate((trad_scores, [trad_scores[0]]))
        auto_values = np.concatenate((auto_scores, [auto_scores[0]]))
        
        ax.plot(angles, trad_values, 'o-', linewidth=2, color='#e74c3c', label='传统车')
        ax.fill(angles, trad_values, alpha=0.25, color='#e74c3c')
        
        ax.plot(angles, auto_values, 's-', linewidth=2, color='#3498db', label='自动驾驶车')
        ax.fill(angles, auto_values, alpha=0.25, color='#3498db')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(indicators, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title('🎯 综合能力雷达图', fontsize=14, fontweight='bold', pad=20)
        # 将图例放到下方，避免覆盖雷达图
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                 ncol=2, fontsize=9, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3)
    
    def plot_detailed_indicators(self, ax, result):
        """绘制详细指标对比"""
        indicators = result['indicators']
        trad_scores = result['traditional']['scores']
        auto_scores = result['autonomous']['scores']
        
        x = np.arange(len(indicators))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, trad_scores, width, label='传统车', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, auto_scores, width, label='自动驾驶车', color='#3498db', alpha=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('评价指标', fontsize=12)
        ax.set_ylabel('得分', fontsize=12)
        ax.set_title('📋 详细指标对比', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(indicators, rotation=30, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def plot_raw_data_comparison(self, ax, result):
        """绘制关键原始数据对比"""
        # 获取原始引擎对象来访问原始数据
        trad_engine = result['traditional']['engine']
        auto_engine = result['autonomous']['engine']
        
        # 准备数据
        indicators = ['RMS加速度', '角速度标准差', '平均噪声', '时间效率', '价格效率']
        units = ['m/s²', 'rad/s', 'dB(A)', 'min/km', '元/km']
        
        # 获取原始数值
        trad_values = [
            trad_engine.results['平顺性']['value'],
            trad_engine.results['稳定性']['value'], 
            trad_engine.results['噪声水平']['value'],
            trad_engine.results['时间效率']['value'],
            trad_engine.results['价格指标']['value'] if '价格指标' in trad_engine.results else 0
        ]
        
        auto_values = [
            auto_engine.results['平顺性']['value'],
            auto_engine.results['稳定性']['value'],
            auto_engine.results['噪声水平']['value'], 
            auto_engine.results['时间效率']['value'],
            auto_engine.results['价格指标']['value'] if '价格指标' in auto_engine.results else 0
        ]
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 设置表格标题 - 调整位置避免覆盖
        ax.text(0.5, 0.93, '🔢 关键指标原始数值', fontsize=13, fontweight='bold', 
               ha='center', va='center', transform=ax.transAxes)
        
        # 创建表格数据
        table_data = []
        table_data.append(['指标名称', '传统车', '自动驾驶车', '差异'])
        
        for i, (indicator, unit, trad_val, auto_val) in enumerate(zip(indicators, units, trad_values, auto_values)):
            if trad_val > 0 and auto_val > 0:  # 确保数据有效
                # 计算差异百分比
                if indicator in ['RMS加速度', '角速度标准差', '平均噪声', '时间效率']:
                    # 这些指标越小越好
                    diff_percent = (trad_val - auto_val) / trad_val * 100
                    arrow = '↓' if diff_percent > 0 else '↑'
                else:
                    # 价格指标单独处理
                    diff_percent = (auto_val - trad_val) / trad_val * 100
                    arrow = '↑' if diff_percent > 0 else '↓'
                
                trad_str = f'{trad_val:.2f} {unit}'
                auto_str = f'{auto_val:.2f} {unit}'
                diff_str = f'{arrow}{abs(diff_percent):.0f}%'
                
                table_data.append([indicator, trad_str, auto_str, diff_str])
        
        # 绘制表格 - 调整位置和大小避免覆盖
        y_start = 0.80  # 降低表格起始位置
        row_height = 0.11  # 稍微减小行高
        col_widths = [0.25, 0.24, 0.24, 0.15]
        col_starts = [0.08, 0.33, 0.57, 0.81]
        
        # 绘制表头
        for j, (text, x_start, width) in enumerate(zip(table_data[0], col_starts, col_widths)):
            ax.text(x_start + width/2, y_start, text, fontsize=10, fontweight='bold',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='#ecf0f1', alpha=0.9))
        
        # 绘制数据行
        for i, row in enumerate(table_data[1:], 1):
            y_pos = y_start - i * row_height
            for j, (text, x_start, width) in enumerate(zip(row, col_starts, col_widths)):
                color = '#2c3e50'
                if j == 3:  # 差异列
                    if '↓' in text:
                        color = '#27ae60'  # 绿色表示改善
                    elif '↑' in text and '价格' not in row[0]:
                        color = '#e74c3c'  # 红色表示变差
                    elif '↑' in text and '价格' in row[0]:
                        color = '#e74c3c'  # 价格上涨也是红色
                
                ax.text(x_start + width/2, y_pos, text, fontsize=9,
                       ha='center', va='center', transform=ax.transAxes, color=color,
                       bbox=dict(boxstyle='round,pad=0.18', facecolor='white', alpha=0.7))

    def run(self):
        """运行GUI"""
        self.root.mainloop()

def main():
    app = 乘车体验对比分析GUI()
    app.run()

if __name__ == "__main__":
    main() 