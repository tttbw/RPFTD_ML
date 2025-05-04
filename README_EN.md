# A Financial Data Denoising Method Based on Reflection-Padded Fourier Transform

[中文版本](README.md)

This is a Python toolkit specifically designed for financial time series analysis and prediction, with a focus on stock market price forecasting. The toolkit provides a complete feature engineering pipeline and various deep learning models. Our core innovation is the Reflection-Padded Fourier Transform Denoising (P-FTD) method, which effectively handles noise and outliers in financial time series.

## Main Features

1. **Financial Data Acquisition**

   - Support for A-share data from akshare
   - Customizable time range
   - Automatic calculation of returns (re)

2. **Feature Engineering**

   - Reflection-Padded Fourier Transform Denoising (P-FTD)
   - Technical indicators calculation (MA, RSI, MACD, Bollinger Bands, etc.)
   - Data standardization and normalization
   - Missing value handling

3. **Model Training**
   - Multiple deep learning models (RNN, LSTM, CNN, GNN)
   - Support Vector Regression (SVR)
   - Model parameter optimization
   - Model performance evaluation

## Feature Engineering Details

### 1. Reflection-Padded Fourier Transform Denoising (P-FTD)

P-FTD is a denoising method specifically designed for financial time series, implemented through the following steps:

1. **Data Preprocessing**

   - Get original time series data
   - Optional parameters:
     ```python
     {
         'feature': str,  # Feature name to process
         'show_plot': bool  # Whether to display plots
     }
     ```

2. **Reflection Padding**

   - Use numpy's pad function for reflection padding
   - Symmetric padding at both ends to avoid boundary effects
   - Optional parameters:
     ```python
     {
         'padding_length': int,  # Padding length, default 100
         'pad_mode': str  # Padding mode, default 'reflect'
     }
     ```

3. **Fourier Transform**

   - Fast Fourier Transform (FFT) on padded sequence
   - Calculate spectral energy distribution
   - Optional parameters:
     ```python
     {
         'limit_freq_ratio': float,  # Frequency limit ratio, default 1.0
         'log_scale': bool  # Whether to use log scale, default True
     }
     ```

4. **Energy Filtering**

   - Filter noise based on energy threshold
   - Optional parameters:
     ```python
     {
         'energy_ratio': float  # Ratio of lowest energy to remove, default 0.1
     }
     ```

5. **Noise Extraction**

   - Calculate noise components
   - Optional parameters:
     ```python
     {
         'show_plot': bool  # Whether to display noise plots
     }
     ```

6. **Inverse Transform and Post-processing**
   - Reconstruct signal using Inverse FFT
   - Remove padding
   - Optional parameters:
     ```python
     {
         'show_plot': bool  # Whether to display processed plots
     }
     ```

### 2. Technical Indicators Calculation

We implement various commonly used financial technical indicators:

1. **Moving Average (MA)**

   - MA5: 5-day moving average
   - MA10: 10-day moving average
   - MA20: 20-day moving average
   - Optional parameters:
     ```python
     {
         'window': int  # Moving window size, default 5/10/20
     }
     ```

2. **Relative Strength Index (RSI)**

   - Calculation period: 14 days
   - Optional parameters:
     ```python
     {
         'window': int  # RSI calculation period, default 14
     }
     ```

3. **MACD Indicator**

   - Fast line: 12-day EMA
   - Slow line: 26-day EMA
   - Signal line: 9-day EMA
   - MACD histogram: MACD - Signal line
   - Optional parameters:
     ```python
     {
         'fast_period': int,  # Fast line period, default 12
         'slow_period': int,  # Slow line period, default 26
         'signal_period': int  # Signal line period, default 9
     }
     ```

4. **Bollinger Bands**
   - Middle band: 20-day moving average
   - Upper band: Middle band + 2 × standard deviation
   - Lower band: Middle band - 2 × standard deviation
   - Optional parameters:
     ```python
     {
         'window': int,  # Calculation period, default 20
         'std_dev': float  # Standard deviation multiplier, default 2
     }
     ```

### 3. Data Preprocessing

1. **Missing Value Handling**

   - Forward fill (ffill)
   - Backward fill (bfill)
   - Optional parameters:
     ```python
     {
         'method': str  # Fill method, options: 'ffill' or 'bfill'
     }
     ```

2. **Data Standardization**

   - Use StandardScaler for standardization
   - Optional parameters:
     ```python
     {
         'scaler_type': str  # Standardization type, default 'standard'
     }
     ```

3. **Feature Preparation**
   - Prepare training data
   - Split training and testing sets
   - Optional parameters:
     ```python
     {
         'features': list,  # Feature list
         'target': str,  # Target variable
         'sequence_length': int,  # Sequence length
         'denoise': bool,  # Whether to denoise
         'target_energy_ratio': float,  # Target variable denoising energy ratio
         'feature_energy_ratio': float  # Feature denoising energy ratio
     }
     ```

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

## Usage Example

```python
from feature_engineering import TimeSeriesFeatureEngineer
from model_trainer import ModelTrainer

# 1. Load data
engineer = TimeSeriesFeatureEngineer.from_akshare(
    symbol="sh000001",  # SSE Composite Index
    start_date="2020-01-01",
    end_date="2023-12-31",
    show_plot=True
)

# 2. Feature engineering
# Add technical indicators
engineer.add_technical_indicators()

# Prepare feature data with P-FTD denoising
features = ['close', 'volume', 'ma5', 'ma10', 'ma20', 'rsi', 'macd']
X_train, X_test, y_train, y_test, scaler_x, scaler_y = engineer.prepare_features(
    features=features,
    target='close',
    sequence_length=10,
    denoise=True,
    target_energy_ratio=0.1
)

# 3. Model training
trainer = ModelTrainer(
    X_train, X_test, y_train, y_test,
    scaler_y=scaler_y,
    sequence_length=10
)

# Train LSTM model
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

## Project Structure

```
.
├── feature_engineering.py  # Feature engineering module
├── model_trainer.py       # Model training module
├── demo.py               # Example code
├── requirements.txt      # Dependencies
└── README.md            # Project documentation
```

## Main Classes

### TimeSeriesFeatureEngineer

Feature engineering class providing:

- Financial data loading and preprocessing
- Technical indicator calculation
- Spectral analysis
- Fourier denoising (P-FTD)
- Data visualization
- Outlier detection

### ModelTrainer

Model training class supporting:

- RNN: Suitable for short-term prediction
- LSTM: Suitable for long-term dependencies
- CNN: Suitable for capturing local patterns
- GNN: Suitable for graph-structured data
- SVR: Suitable for small sample prediction

## Notes

1. Ensure all dependencies are installed before use
2. GPU is recommended for model training
3. Internet connection required for data acquisition
4. P-FTD parameters should be adjusted based on data characteristics
5. Technical indicator parameters can be adjusted for different markets
6. Start with small-scale testing before scaling up

## License

MIT License
