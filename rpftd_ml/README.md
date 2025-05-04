# RPFTD-ML

A Financial Data Denoising Method Based on Reflection-Padded Fourier Transform

## Installation

1. Clone the repository:

```bash
git clone https://github.com/tttbw/RPFTD_ML.git
cd RPFTD_ML
```

2. Install the package (development mode):

```bash
pip install -e .
```

## Usage

```python
from rpftd_ml import TimeSeriesFeatureEngineer, ModelTrainer

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

For more detailed documentation, please visit our [GitHub repository](https://github.com/tttbw/RPFTD_ML).
