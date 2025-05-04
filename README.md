# 一种基于反射填充的傅立叶降噪的金融数据降噪方法

[English Version](README_EN.md)

这是一个专门用于金融时间序列分析和预测的 Python 工具包，特别针对股票市场的价格预测。该工具包提供了完整的特征工程流程和多种深度学习模型，其中基于反射填充的傅立叶降噪方法（P-FTD）是我们的核心创新，能够有效处理金融时间序列中的噪声和异常值。

## 主要特性

1. **金融数据获取**

   - 支持从 akshare 获取 A 股数据
   - 支持自定义时间范围
   - 自动计算收益率（re）

2. **特征工程**

   - 基于反射填充的傅立叶降噪（P-FTD）
   - 技术指标计算（MA、RSI、MACD、Bollinger Bands 等）
   - 数据标准化和归一化
   - 缺失值处理

3. **模型训练**
   - 多种深度学习模型（RNN、LSTM、CNN、GNN）
   - 支持向量回归（SVR）
   - 模型参数优化
   - 模型性能评估

## 特征工程详解

### 1. 基于反射填充的傅立叶降噪（P-FTD）

P-FTD 是一种专门针对金融时间序列设计的降噪方法，它通过以下步骤实现：

1. **数据预处理**

   - 获取原始时间序列数据
   - 可选参数：
     ```python
     {
         'feature': str,  # 要处理的特征名
         'show_plot': bool  # 是否显示图像
     }
     ```

2. **反射填充**

   - 使用 numpy 的 pad 函数进行反射填充
   - 在序列两端进行对称填充，避免边界效应
   - 可选参数：
     ```python
     {
         'padding_length': int,  # 填充长度，默认为100
         'pad_mode': str  # 填充模式，默认为'reflect'
     }
     ```

3. **傅立叶变换**

   - 对填充后的序列进行快速傅立叶变换（FFT）
   - 计算频谱能量分布
   - 可选参数：
     ```python
     {
         'limit_freq_ratio': float,  # 频率限制比例，默认为1.0
         'log_scale': bool  # 是否使用对数刻度，默认为True
     }
     ```

4. **能量滤波**

   - 基于能量阈值过滤噪声
   - 可选参数：
     ```python
     {
         'energy_ratio': float  # 要去除的最低能量部分的比例，默认为0.1
     }
     ```

5. **噪声提取**

   - 计算噪声成分
   - 可选参数：
     ```python
     {
         'show_plot': bool  # 是否显示噪声图像
     }
     ```

6. **逆变换和后处理**
   - 使用逆傅立叶变换（IFFT）重构信号
   - 去除填充部分
   - 可选参数：
     ```python
     {
         'show_plot': bool  # 是否显示处理后的图像
     }
     ```

### 2. 技术指标计算

我们实现了多种常用的金融技术指标：

1. **移动平均线（MA）**

   - MA5：5 日移动平均
   - MA10：10 日移动平均
   - MA20：20 日移动平均
   - 可选参数：
     ```python
     {
         'window': int  # 移动窗口大小，默认为5/10/20
     }
     ```

2. **相对强弱指标（RSI）**

   - 计算周期：14 天
   - 可选参数：
     ```python
     {
         'window': int  # RSI计算周期，默认为14
     }
     ```

3. **MACD 指标**

   - 快线：12 日 EMA
   - 慢线：26 日 EMA
   - 信号线：9 日 EMA
   - MACD 柱状图：MACD - 信号线
   - 可选参数：
     ```python
     {
         'fast_period': int,  # 快线周期，默认为12
         'slow_period': int,  # 慢线周期，默认为26
         'signal_period': int  # 信号线周期，默认为9
     }
     ```

4. **布林带（Bollinger Bands）**
   - 中轨：20 日移动平均
   - 上轨：中轨 + 2 倍标准差
   - 下轨：中轨 - 2 倍标准差
   - 可选参数：
     ```python
     {
         'window': int,  # 计算周期，默认为20
         'std_dev': float  # 标准差倍数，默认为2
     }
     ```

### 3. 数据预处理

1. **缺失值处理**

   - 前向填充（ffill）
   - 后向填充（bfill）
   - 可选参数：
     ```python
     {
         'method': str  # 填充方法，可选'ffill'或'bfill'
     }
     ```

2. **数据标准化**

   - 使用 StandardScaler 进行标准化
   - 可选参数：
     ```python
     {
         'scaler_type': str  # 标准化类型，默认为'standard'
     }
     ```

3. **特征准备**
   - 准备训练数据
   - 划分训练集和测试集
   - 可选参数：
     ```python
     {
         'features': list,  # 特征列表
         'target': str,  # 目标变量
         'sequence_length': int,  # 序列长度
         'denoise': bool,  # 是否进行降噪
         'target_energy_ratio': float,  # 目标变量的降噪能量比例
         'feature_energy_ratio': float  # 特征的降噪能量比例
     }
     ```

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/tttbw/RPFTD_ML.git
cd RPFTD_ML
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用示例

```python
from feature_engineering import TimeSeriesFeatureEngineer
from model_trainer import ModelTrainer

# 1. 数据加载
engineer = TimeSeriesFeatureEngineer.from_akshare(
    symbol="sh000001",  # 上证指数
    start_date="2020-01-01",
    end_date="2023-12-31",
    show_plot=True
)

# 2. 特征工程
# 添加技术指标
engineer.add_technical_indicators()

# 准备特征数据，使用P-FTD降噪
features = ['close', 'volume', 'ma5', 'ma10', 'ma20', 'rsi', 'macd']
X_train, X_test, y_train, y_test, scaler_x, scaler_y = engineer.prepare_features(
    features=features,
    target='close',
    sequence_length=10,
    denoise=True,
    target_energy_ratio=0.1
)

# 3. 模型训练
trainer = ModelTrainer(
    X_train, X_test, y_train, y_test,
    scaler_y=scaler_y,
    sequence_length=10
)

# 训练LSTM模型
model, r2, mse = trainer.train_model(
    model_type='lstm',
    model_params={
        'hidden_size': 128,
        'num_layers': 2,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'batch_size': 32
    }
)
```

## 项目结构

```
.
├── feature_engineering.py  # 特征工程模块
├── model_trainer.py       # 模型训练模块
├── demo.py               # 示例代码
├── requirements.txt      # 依赖包
└── README.md            # 项目说明
```

## 主要类说明

### TimeSeriesFeatureEngineer

特征工程类，提供以下功能：

- 金融数据加载和预处理
- 技术指标计算
- 频谱分析
- 傅立叶降噪（P-FTD）
- 数据可视化
- 异常值检测

### ModelTrainer

模型训练类，支持以下模型：

- RNN：适合短期预测
- LSTM：适合长期依赖关系
- CNN：适合捕捉局部模式
- GNN：适合处理图结构数据
- SVR：适合小样本预测

## 注意事项

1. 使用前请确保已安装所有依赖包
2. 建议使用 GPU 进行模型训练
3. 数据获取需要网络连接
4. P-FTD 参数需要根据具体数据特点调整
5. 技术指标参数可以根据不同市场特点调整
6. 建议先进行小规模测试，再扩大数据规模

## 许可证

MIT License
