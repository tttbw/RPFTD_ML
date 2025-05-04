import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import akshare as ak
from typing import Union, Tuple, Dict, List, Optional

class TimeSeriesFeatureEngineer:
    def __init__(self, data: pd.DataFrame = None, show_plot: bool = False):
        """
        初始化时间序列特征工程器
        
        Args:
            data: 输入的时间序列数据，DataFrame格式
            show_plot: 是否显示绘图，默认为False
        """
        self.data = data
        self.show_plot = show_plot
        
        # 特征类型映射
        self.feature_type_map = {
            'high': 'price',
            'low': 'price',
            'open': 'price',
            'close': 'price',
            'volume': 'volume',
            're': 'return',
            'amount': 'amount',
            'turnover': 'turnover'
        }
    
    @classmethod
    def from_akshare(cls, 
                    symbol: str = "sh000001", 
                    start_date: str = '2008-01-01', 
                    end_date: str = '2025-01-27',
                    show_plot: bool = False):
        """
        从akshare加载数据创建特征工程器实例
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            show_plot: 是否显示绘图，默认为False
            
        Returns:
            TimeSeriesFeatureEngineer实例
        """
        # 加载股票数据
        data = ak.stock_zh_a_daily(symbol=symbol)
        data['re'] = data['close'].pct_change(1)
        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= pd.Timestamp(start_date)) & 
                   (data['date'] <= pd.Timestamp(end_date))]
        
        return cls(data=data, show_plot=show_plot)

    @staticmethod
    def energy_filter(power_spectrum: np.ndarray, 
                     energy_ratio: float = 0.1) -> float:
        """基于能量谱的滤波器，去掉能量最低的频段
        
        Args:
            power_spectrum: 功率谱
            energy_ratio: 要去除的最低能量部分的比例，默认0.1（10%）
            
        Returns:
            float: 计算得到的能量阈值
        """
        sorted_power = np.sort(power_spectrum)
        threshold_idx = int(len(sorted_power) * energy_ratio)
        return sorted_power[threshold_idx]

    def draw_origin(self, feature: str):
        """绘制原始时间序列"""
        sns.set_theme(style="whitegrid", palette="muted")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=self.data['date'], y=self.data[feature], 
                    color='blue', linewidth=1.5, label=feature)
        plt.title("Original Time Series", fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10)
        plt.legend(loc='upper left', fontsize=10, title="Data", 
                  title_fontsize=11, frameon=True, shadow=True)
        plt.tight_layout()
        if self.show_plot:
            plt.show()
        else:
            plt.close()

    def draw_spectral(self, feature: str, limit_freq_ratio: float = 1, 
                     log_scale: bool = True) -> np.ndarray:
        """绘制频谱分析图"""
        val = self.data[feature].values
        fft_val = np.fft.fft(val)
        freqs = np.fft.fftfreq(len(fft_val), 1 / len(fft_val))
        power = np.abs(fft_val) ** 2
        
        half_len = len(freqs) // 2
        freqs = freqs[:half_len]
        power = power[:half_len]
        
        n_limit = int(len(freqs) * limit_freq_ratio)
        limited_freqs = freqs[:n_limit]
        limited_power = power[:n_limit]
        
        if self.show_plot:
            sns.set_theme(style="whitegrid", palette="muted")
            plt.figure(figsize=(12, 6))
            
            if log_scale:
                plt.plot(limited_freqs, np.log10(limited_power), 
                        color='blue', linewidth=1.5, label=f"{feature}")
                plt.ylabel('Log Power')
            else:
                plt.plot(limited_freqs, limited_power, 
                        color='blue', linewidth=1.5, label=f"{feature}")
                plt.ylabel('Power')
            
            plt.title('Frequency Spectrum Analysis', fontsize=16, fontweight='bold')
            plt.xlabel('Frequency [Hz]', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
            plt.legend(loc='upper right', fontsize=10, frameon=True, 
                      shadow=True, title='Legend', title_fontsize=11)
            plt.tight_layout()
            plt.show()
        
        return fft_val

    def denoise_with_padding(self, 
                           feature: str, 
                           energy_ratio: float = 0.1, 
                           padding_length: int = 100, 
                           show_plot: Optional[bool] = None) -> pd.DataFrame:
        """使用反射填充进行FFT降噪
        
        Args:
            feature: 要处理的特征名
            energy_ratio: 要去除的最低能量部分的比例，默认0.1（10%）
            padding_length: 填充长度，默认100
            show_plot: 是否显示图像，默认None（使用实例的show_plot设置）
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        if show_plot is None:
            show_plot = self.show_plot
            
        # 获取原始序列
        original_seq = self.data[feature].values
        
        # 使用反射填充扩展序列
        padded_seq = np.pad(original_seq, (padding_length, padding_length), 
                          mode='reflect')
        
        # FFT变换
        fft_result = np.fft.fft(padded_seq)
        power_spectrum = np.abs(fft_result) ** 2
        
        # 基于能量的滤波
        threshold = self.energy_filter(power_spectrum, energy_ratio)
        fft_filtered = fft_result.copy()
        fft_filtered[power_spectrum < threshold] = 0
        
        # 计算噪声
        fft_noise = fft_result - fft_filtered

        # 逆FFT变换
        cleaned_padded_seq = np.real(np.fft.ifft(fft_filtered))
        cleaned_seq = cleaned_padded_seq[padding_length:-padding_length]
        noise_padded_seq = np.real(np.fft.ifft(fft_noise))
        noise_seq = noise_padded_seq[padding_length:-padding_length]

        # 保存结果
        self.data[f'denoised_{feature}'] = cleaned_seq
        self.data[f'noise_{feature}'] = noise_seq

        # 可视化
        if show_plot:
            sns.set_theme(style="whitegrid", palette="muted")
            plt.figure(figsize=(14, 12))

            plt.subplot(2, 1, 1)
            plt.plot(self.data['date'], original_seq, label="Original Signal", 
                    color="red", linestyle="--", linewidth=2, alpha=0.85)
            plt.plot(self.data['date'], cleaned_seq, label="Denoised Signal", 
                    color="blue", linewidth=1.5)
            plt.title(f"P-FTD Denoised {feature} (Remove Lowest {energy_ratio:.1%} Energy)", 
                     fontsize=16, fontweight="bold", pad=15)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Signal Value", fontsize=12)
            plt.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
            plt.xticks(rotation=30, fontsize=10)
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(self.data['date'], noise_seq, label="Extracted Noise", 
                    color="green", linewidth=1.2, alpha=0.8)
            plt.title("Noise Component Visualization", fontsize=16, fontweight="bold", pad=15)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Noise Amplitude", fontsize=12)
            plt.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
            plt.xticks(rotation=30, fontsize=10)
            plt.legend()

            plt.tight_layout()
            plt.show()

        return self.data

    def prepare_features(self, 
                        features: List[str] = None,
                        target: str = 'close',
                        sequence_length: int = 10, 
                        denoise: bool = True, 
                        target_energy_ratio: float = 0.1,
                        feature_energy_ratio: Optional[float] = None) -> Tuple[np.ndarray, ...]:
        """准备特征数据
        
        Args:
            features: 要使用的特征列表，如果为None则使用所有可用特征
            target: 目标变量，默认为'close'
            sequence_length: 序列长度，默认10
            denoise: 是否对特征进行傅立叶降噪，默认True
            target_energy_ratio: 目标变量的降噪能量比例，默认0.1（10%）
            feature_energy_ratio: 特征的降噪能量比例，如果为None则使用target_energy_ratio
            
        Returns:
            Tuple[np.ndarray, ...]: (X_train, X_test, y_train, y_test)
        """
        if features is None:
            features = [col for col in self.data.columns if col not in ['date', target]]
        
        if feature_energy_ratio is None:
            feature_energy_ratio = target_energy_ratio
        
        # 首先处理缺失值
        self.data = self.data.ffill()  # 向前填充
        self.data = self.data.bfill()  # 向后填充（处理开始的NaN）
        
        # 对每个特征进行傅立叶降噪
        if denoise:
            # 对目标变量进行降噪
            self.denoise_with_padding(target, energy_ratio=target_energy_ratio, show_plot=False)
            
            # 对特征进行降噪
            for feature in features:
                self.denoise_with_padding(feature, energy_ratio=feature_energy_ratio, show_plot=False)
        
        # 准备训练数据
        X = self.data[features].values
        y = self.data[target].values
        
        # 数据标准化
        from sklearn.preprocessing import StandardScaler
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        X = self.scaler_x.fit_transform(X)
        y = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        # 划分训练集和测试集
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return X_train, X_test, y_train, y_test, self.scaler_x, self.scaler_y

    def add_technical_indicators(self):
        """添加技术指标
        
        计算以下技术指标：
        - MA (移动平均线): ma5, ma10, ma20
        - RSI (相对强弱指标)
        - MACD (移动平均收敛散度)
        - Bollinger Bands (布林带)
        """
        # 移动平均线
        self.data['ma5'] = self.data['close'].rolling(window=5).mean()
        self.data['ma10'] = self.data['close'].rolling(window=10).mean()
        self.data['ma20'] = self.data['close'].rolling(window=20).mean()
        
        # RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['close'].ewm(span=26, adjust=False).mean()
        self.data['macd'] = exp1 - exp2
        self.data['signal'] = self.data['macd'].ewm(span=9, adjust=False).mean()
        self.data['macd_hist'] = self.data['macd'] - self.data['signal']
        
        # Bollinger Bands
        self.data['bb_middle'] = self.data['close'].rolling(window=20).mean()
        std = self.data['close'].rolling(window=20).std()
        self.data['bb_upper'] = self.data['bb_middle'] + (std * 2)
        self.data['bb_lower'] = self.data['bb_middle'] - (std * 2)
        
        return self.data 