import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pybaseball import batting_stats
import joblib
import os
import numpy as np
import torch
import torch.nn as nn

# NNモデルの定義（pred_NN.pyから移植）
class Net(nn.Module):
    def __init__(self, input_dim=4, output_dim=5):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def predict(model, sample):
    """
    sample は {'K%': ..., 'BB%': ..., 'Hard%': ..., 'GB%': ...} の形式。
    1選手分の予測値を返す。
    線形回帰モデル用の関数（後方互換性のため）
    """
    # 入力データの検証
    required_features = ['K%', 'BB%', 'Hard%', 'GB%']
    for feature in required_features:
        if feature not in sample:
            raise ValueError(f"必須特徴量 '{feature}' が入力に含まれていません。")
    
    X_new = pd.DataFrame([sample])
    return model.predict(X_new)[0]

def predict_linear(model, sample):
    """
    線形回帰モデル用の予測関数
    """
    return predict(model, sample)

def predict_nn(model, sample):
    """
    ニューラルネットワークモデル用の予測関数
    sample は {'K%': ..., 'BB%': ..., 'Hard%': ..., 'GB%': ...} の形式。
    1選手分の予測値を返す。
    """
    # 入力データの検証
    required_features = ['K%', 'BB%', 'Hard%', 'GB%']
    for feature in required_features:
        if feature not in sample:
            raise ValueError(f"必須特徴量 '{feature}' が入力に含まれていません。")
    
    # データを配列に変換
    X_new = pd.DataFrame([sample]).values.astype(np.float32)
    tensor_input = torch.from_numpy(X_new)
    
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_input).numpy()
    
    return prediction[0]  # 出力は5要素のベクトル

def predict_auto(model, sample, model_type='linear'):
    """
    モデルタイプに応じて適切な予測関数を呼び出す
    
    Parameters:
    -----------
    model : sklearn model or torch model
        予測に使用するモデル
    sample : dict
        特徴量のディクショナリ {'K%': ..., 'BB%': ..., 'Hard%': ..., 'GB%': ...}
    model_type : str
        'linear' または 'nn'
    
    Returns:
    --------
    numpy.ndarray
        予測結果（5要素のベクトル）
    """
    if model_type == 'linear':
        return predict_linear(model, sample)
    elif model_type == 'nn':
        return predict_nn(model, sample)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
