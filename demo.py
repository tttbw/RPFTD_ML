import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_engineering import TimeSeriesFeatureEngineer
from model_trainer import ModelTrainer

def main():
    # 1. 数据加载和特征工程
    print("1. 加载数据并进行特征工程...")
    engineer = TimeSeriesFeatureEngineer.from_akshare(
        symbol="sh000001",  # 上证指数
        start_date="2021-01-01",  # 使用更长的时间范围
        end_date="2023-12-31",
        show_plot=True
    )
    
    # 验证数据加载
    print(f"加载数据形状: {engineer.data.shape}")
    print("数据预览:")
    print(engineer.data.head())
    
    # 添加技术指标
    engineer.add_technical_indicators()
    print("\n添加技术指标后的列名:")
    print(engineer.data.columns.tolist())
    
    # 检查缺失值
    print("\n检查缺失值:")
    print(engineer.data.isnull().sum())
    
    # 可视化原始数据
    engineer.draw_origin('close')
    
    # 频谱分析
    engineer.draw_spectral('close', limit_freq_ratio=0.5)
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    # 准备特征数据
    features = ['close', 'volume', 'ma5', 'ma10', 'ma20', 'rsi', 'macd', 'signal', 'macd_hist']
    
    # 检查特征是否存在
    missing_features = [f for f in features if f not in engineer.data.columns]
    if missing_features:
        raise ValueError(f"缺失以下特征: {missing_features}")
        
    # 打印特征数据的形状
    print("\n特征数据形状:")
    for feature in features:
        print(f"{feature}: {engineer.data[feature].shape}")
        print(f"非空值数量: {engineer.data[feature].count()}")
        print(f"空值数量: {engineer.data[feature].isnull().sum()}")
        print("---")
    
    X_train, X_test, y_train, y_test, scaler_x, scaler_y = engineer.prepare_features(
        features=features,
        target='close',
        sequence_length=10,
        denoise=True,
        target_energy_ratio=0.1
    )
    
    print(f"\n训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    # 3. 模型训练
    print("\n3. 训练模型...")
    trainer = ModelTrainer(
        X_train, X_test, y_train, y_test,
        scaler_y=scaler_y,
        sequence_length=10
    )
    
    # 存储模型结果
    model_results = {}
    
    # 训练RNN模型
    print("\n训练RNN模型...")
    rnn_model, rnn_r2, rnn_mse = trainer.train_model(
        model_type='rnn',
        model_params={
            'hidden_size': 128,
            'num_layers': 2,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'batch_size': 32
        }
    )
    model_results['RNN'] = {'R²': rnn_r2, 'MSE': rnn_mse}
    print(f"RNN模型 R²: {rnn_r2:.4f}, MSE: {rnn_mse:.4f}")
    
    # 训练LSTM模型
    print("\n训练LSTM模型...")
    lstm_model, lstm_r2, lstm_mse = trainer.train_model(
        model_type='lstm',
        model_params={
            'hidden_size': 128,
            'num_layers': 2,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'batch_size': 32
        }
    )
    model_results['LSTM'] = {'R²': lstm_r2, 'MSE': lstm_mse}
    print(f"LSTM模型 R²: {lstm_r2:.4f}, MSE: {lstm_mse:.4f}")
    
    # 训练CNN模型
    print("\n训练CNN模型...")
    cnn_model, cnn_r2, cnn_mse = trainer.train_model(
        model_type='cnn',
        model_params={
            'hidden_size': 128,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'batch_size': 32
        }
    )
    model_results['CNN'] = {'R²': cnn_r2, 'MSE': cnn_mse}
    print(f"CNN模型 R²: {cnn_r2:.4f}, MSE: {cnn_mse:.4f}")
    
    # 训练GNN模型
    print("\n训练GNN模型...")
    gnn_model, gnn_r2, gnn_mse = trainer.train_model(
        model_type='gnn',
        model_params={
            'hidden_size': 128,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'batch_size': 32
        }
    )
    model_results['GNN'] = {'R²': gnn_r2, 'MSE': gnn_mse}
    print(f"GNN模型 R²: {gnn_r2:.4f}, MSE: {gnn_mse:.4f}")
    
    # 训练SVR模型
    print("\n训练SVR模型...")
    svr_model, svr_r2, svr_mse = trainer.train_model(
        model_type='svr',
        model_params={
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'epsilon': 0.1
        }
    )
    model_results['SVR'] = {'R²': svr_r2, 'MSE': svr_mse}
    print(f"SVR模型 R²: {svr_r2:.4f}, MSE: {svr_mse:.4f}")
    
    # 优化SVR参数
    print("\n优化SVR参数...")
    best_params, best_r2, best_mse = trainer.optimize_svr_params()
    model_results['SVR_optimized'] = {'R²': best_r2, 'MSE': best_mse}
    print(f"最优参数: {best_params}")
    print(f"最优R²: {best_r2:.4f}, 最优MSE: {best_mse:.4f}")
    
    # 4. 模型比较
    print("\n4. 模型比较:")
    print("\n模型性能对比:")
    print("=" * 50)
    print(f"{'模型':<15} {'R²':<10} {'MSE':<10}")
    print("-" * 50)
    for model_name, metrics in model_results.items():
        print(f"{model_name:<15} {metrics['R²']:.4f}    {metrics['MSE']:.4f}")
    print("=" * 50)
    
    # 可视化模型性能
    plt.figure(figsize=(12, 6))
    models = list(model_results.keys())
    r2_scores = [metrics['R²'] for metrics in model_results.values()]
    mse_scores = [metrics['MSE'] for metrics in model_results.values()]
    
    plt.subplot(1, 2, 1)
    plt.bar(models, r2_scores)
    plt.title('R² Score Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('R² Score')
    
    plt.subplot(1, 2, 2)
    plt.bar(models, mse_scores)
    plt.title('MSE Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('MSE')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 