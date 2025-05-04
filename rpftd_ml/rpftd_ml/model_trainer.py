import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from typing import Union, Tuple, Dict, List, Optional, Literal
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 10):
        """
        时间序列数据集
        
        Args:
            X: 特征数据
            y: 目标数据
            sequence_length: 序列长度
        """
        self.sequence_length = sequence_length
        
        # 创建序列数据
        X_seq = []
        y_seq = []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        
        self.X = torch.FloatTensor(np.array(X_seq))
        self.y = torch.FloatTensor(np.array(y_seq))
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BaseModel(nn.Module):
    """所有模型的基类"""
    def __init__(self):
        super().__init__()
        self.model_type = "base"
        
    def get_default_params(self) -> Dict:
        """获取默认参数"""
        return {}

class RNN(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.model_type = "rnn"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
        
    @staticmethod
    def get_default_params() -> Dict:
        return {
            "hidden_size": 64,
            "num_layers": 2,
            "learning_rate": 0.001,
            "num_epochs": 100,
            "batch_size": 32
        }

class LSTM(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.model_type = "lstm"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 修改LSTM层的输入维度
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        n_features = x.size(2)
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只使用最后一个时间步的输出
        out = out[:, -1, :]
        
        # 添加BatchNorm和Dropout
        out = self.bn(out)
        out = self.dropout(out)
        
        # 全连接层
        out = self.fc(out)
        return out
        
    @staticmethod
    def get_default_params() -> Dict:
        return {
            "hidden_size": 128,
            "num_layers": 2,
            "learning_rate": 0.001,
            "num_epochs": 100,
            "batch_size": 32,
            "dropout": 0.3
        }

class CNN(BaseModel):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super().__init__()
        self.model_type = "cnn"
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
        
    @staticmethod
    def get_default_params() -> Dict:
        return {
            "hidden_size": 64,
            "learning_rate": 0.001,
            "num_epochs": 100,
            "batch_size": 32,
            "dropout": 0.3
        }

class GNN(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.model_type = "gnn"
        self.conv1 = nn.Linear(input_size, hidden_size)
        self.conv2 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, adj):
        # 重塑输入数据以适应BatchNorm
        batch_size = x.size(0)
        seq_length = x.size(1)
        n_features = x.size(2)
        
        # 重塑x为(batch_size * seq_length, n_features)
        x = x.reshape(-1, n_features)
        
        # 应用图卷积和BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # 重塑回(batch_size, seq_length, hidden_size)
        x = x.reshape(batch_size, seq_length, -1)
        
        # 计算平均值
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x
        
    @staticmethod
    def get_default_params() -> Dict:
        return {
            "hidden_size": 64,
            "learning_rate": 0.001,
            "num_epochs": 100,
            "batch_size": 32,
            "dropout": 0.3
        }

class ModelTrainer:
    def __init__(self, 
                 X_train: np.ndarray, 
                 X_test: np.ndarray, 
                 y_train: np.ndarray, 
                 y_test: np.ndarray,
                 scaler_y,
                 sequence_length: int = 10):
        """
        模型训练器
        
        Args:
            X_train: 训练集特征
            X_test: 测试集特征
            y_train: 训练集标签
            y_test: 测试集标签
            scaler_y: 标签的标准化器
            sequence_length: 序列长度
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler_y = scaler_y
        self.sequence_length = sequence_length
        
        # 创建数据集
        self.train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
        self.test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)
        
    def create_model(self, 
                    model_type: Literal['rnn', 'lstm', 'cnn', 'gnn', 'svr'],
                    model_params: Optional[Dict] = None) -> BaseModel:
        """
        创建模型
        
        Args:
            model_type: 模型类型
            model_params: 模型参数
            
        Returns:
            模型实例
        """
        input_size = self.X_train.shape[1]
        output_size = 1
        
        if model_type == 'rnn':
            params = RNN.get_default_params()
            if model_params:
                params.update(model_params)
            return RNN(input_size, params['hidden_size'], 
                      params['num_layers'], output_size)
        
        elif model_type == 'lstm':
            params = LSTM.get_default_params()
            if model_params:
                params.update(model_params)
            return LSTM(input_size, params['hidden_size'], 
                       params['num_layers'], output_size)
        
        elif model_type == 'cnn':
            params = CNN.get_default_params()
            if model_params:
                params.update(model_params)
            return CNN(input_size, output_size, params['hidden_size'])
        
        elif model_type == 'gnn':
            params = GNN.get_default_params()
            if model_params:
                params.update(model_params)
            return GNN(input_size, params['hidden_size'], output_size)
        
        elif model_type == 'svr':
            params = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            }
            if model_params:
                params.update(model_params)
            return SVR(**params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_model(self, 
                   model_type: Literal['rnn', 'lstm', 'cnn', 'gnn', 'svr'],
                   model_params: Optional[Dict] = None) -> Tuple[BaseModel, float, float]:
        """
        训练模型
        
        Args:
            model_type: 模型类型
            model_params: 模型参数
            
        Returns:
            Tuple[模型, R²得分, MSE]
        """
        if model_type == 'svr':
            return self._train_svr(model_params)
        else:
            return self._train_neural_network(model_type, model_params)
    
    def _train_neural_network(self, 
                            model_type: str,
                            model_params: Optional[Dict] = None) -> Tuple[BaseModel, float, float]:
        """训练神经网络模型"""
        # 获取默认参数
        if model_type == 'rnn':
            params = RNN.get_default_params()
        elif model_type == 'lstm':
            params = LSTM.get_default_params()
        elif model_type == 'cnn':
            params = CNN.get_default_params()
        elif model_type == 'gnn':
            params = GNN.get_default_params()
            
        if model_params:
            params.update(model_params)
            
        # 创建模型和数据加载器
        model = self.create_model(model_type, model_params)
        train_loader = DataLoader(self.train_dataset, 
                                batch_size=params['batch_size'], 
                                shuffle=True)
        test_loader = DataLoader(self.test_dataset, 
                               batch_size=params['batch_size'], 
                               shuffle=False)
        
        # 训练设置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # 训练模型
        for epoch in range(params['num_epochs']):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                if model_type == 'gnn':
                    # 创建邻接矩阵
                    adj = torch.eye(self.sequence_length).to(device)
                    for i in range(self.sequence_length - 1):
                        adj[i, i+1] = 1
                        adj[i+1, i] = 1
                    adj = adj / adj.sum(dim=1, keepdim=True)
                    outputs = model(batch_X, adj)
                else:
                    outputs = model(batch_X)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{params["num_epochs"]}], Loss: {loss.item():.4f}')
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                if model_type == 'gnn':
                    adj = torch.eye(self.sequence_length).to(device)
                    for i in range(self.sequence_length - 1):
                        adj[i, i+1] = 1
                        adj[i+1, i] = 1
                    adj = adj / adj.sum(dim=1, keepdim=True)
                    outputs = model(batch_X, adj)
                else:
                    outputs = model(batch_X)
                y_pred.extend(outputs.cpu().numpy())
                y_true.extend(batch_y.cpu().numpy())
            
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            
            y_pred = self.scaler_y.inverse_transform(y_pred)
            y_true = self.scaler_y.inverse_transform(y_true)
            
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
        
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        
        return model, r2, mse
    
    def _train_svr(self, model_params: Optional[Dict] = None) -> Tuple[SVR, float, float]:
        """训练SVR模型"""
        # 准备数据
        X_train = self.train_dataset.X.numpy()
        y_train = self.train_dataset.y.numpy()
        X_test = self.test_dataset.X.numpy()
        y_test = self.test_dataset.y.numpy()
        
        # 重塑数据
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # 设置默认参数
        params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'epsilon': 0.1
        }
        if model_params:
            params.update(model_params)
        
        # 训练模型
        model = SVR(**params)
        model.fit(X_train, y_train.ravel())
        
        # 预测和评估
        y_pred = model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_test = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        
        return model, r2, mse
    
    def optimize_svr_params(self, param_grid: Optional[Dict] = None) -> Tuple[Dict, float, float]:
        """
        优化SVR模型的超参数
        
        Args:
            param_grid: 参数网格
            
        Returns:
            Tuple[最优参数, 最优R²得分, 最优MSE]
        """
        # 准备数据
        X_train = self.train_dataset.X.numpy()
        y_train = self.train_dataset.y.numpy()
        X_train = X_train.reshape(X_train.shape[0], -1)
        
        # 默认参数网格
        default_param_grid = {
            'kernel': ['rbf', 'linear', 'poly'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2]
        }
        
        if param_grid:
            default_param_grid.update(param_grid)
        
        # 创建SVR模型
        svr = SVR()
        
        # 使用GridSearchCV进行参数优化
        grid_search = GridSearchCV(
            estimator=svr,
            param_grid=default_param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # 拟合数据
        grid_search.fit(X_train, y_train.ravel())
        
        # 获取最优参数和得分
        best_params = grid_search.best_params_
        best_r2 = grid_search.best_score_
        
        # 使用最优参数训练模型并计算MSE
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_train)
        y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_train = self.scaler_y.inverse_transform(y_train.reshape(-1, 1))
        best_mse = mean_squared_error(y_train, y_pred)
        
        print(f"最优参数: {best_params}")
        print(f"最优R²得分: {best_r2:.4f}")
        print(f"最优MSE: {best_mse:.4f}")
        
        return best_params, best_r2, best_mse 