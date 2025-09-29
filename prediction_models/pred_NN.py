import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

def fetch_data_from_pybaseball(start_year=2022, end_year=2024, min_pa=100):
    """pybaseballを用いて2022～2024年のデータを取得し、特徴量と、全打数における各アウト・安打割合のターゲットベクトルを抽出・整形します"""
    from pybaseball import batting_stats
    frames = []
    for year in range(start_year, end_year + 1):
        df = batting_stats(year)
        df['year'] = year
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    
    # 最小打席数でフィルタリング
    data = data[data['PA'] >= min_pa].reset_index(drop=True)
    
    # 1Bカラムがない場合は計算する
    if '1B' not in data.columns:
        data['1B'] = data['H'] - (data['2B'] + data['3B'] + data['HR'])
    
    # 欠損値の処理
    # 主要な特徴量に欠損値があるレコードを除外
    required_cols = ['K%', 'BB%', 'Hard%', 'GB%', 'PA', 'SO', 'BB', '1B', '2B', '3B', 'HR']
    data = data.dropna(subset=required_cols).reset_index(drop=True)
    
    # ターゲットの計算：PAを分母として各割合を計算（PA=全打数）
    # ゼロ除算を防ぐための安全な割り算
    data['SO_rate'] = data['SO'] / data['PA'].replace(0, 1)
    data['BB_rate'] = data['BB'] / data['PA'].replace(0, 1)
    data['1B_rate'] = data['1B'] / data['PA'].replace(0, 1)
    data['2B_rate'] = data['2B'] / data['PA'].replace(0, 1)
    data['3B_rate'] = data['3B'] / data['PA'].replace(0, 1)
    data['HR_rate'] = data['HR'] / data['PA'].replace(0, 1)
    data['OTH_rate'] = (data['PA'] - (data['SO'] + data['BB'] + data['1B'] + data['2B'] + data['3B'] + data['HR'])) / data['PA'].replace(0, 1)
    
    # 特徴量と目標変数の選択
    cols_to_return = ['year', 'Name', 'Team', 'PA', 'K%', 'BB%', 'Hard%', 'GB%', 
                     'SO_rate', 'BB_rate', '1B_rate', '2B_rate', '3B_rate', 'HR_rate', 'OTH_rate']
    # 選択した列のうち、データに存在する列のみを返す
    existing_cols = [col for col in cols_to_return if col in data.columns]
    
    return data[existing_cols]

# PyTorchによるモデル定義
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 回帰（MSE）用にそのまま出力
        return x

def train_model(data):
    # 特徴量は ["K%", "BB%", "Hard%", "GB%"], 目的変数は [1B_rate, 2B_rate, 3B_rate, HR_rate]
    features = ["K%", "BB%", "Hard%", "GB%"]
    target = ["1B_rate", "2B_rate", "3B_rate", "HR_rate"]

    df = data.copy()
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32)
    # 数値誤差対策としてクリップのみ（正規化は行わない）
    y = np.clip(y, 0.0, None)
    
    # データを学習・検証用に分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # numpy配列をtorchテンソルに変換
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = Net(input_dim, output_dim)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 1000
    batch_size = 32
    # --- 重み付きサンプリング（正規分布を仮定）---
    # 学習特徴量 X_train について多変量正規分布 N(μ, Σ) を仮定し、
    # サンプル重みを w_i ∝ 1 / pdf(x_i) で定義（平均1に正規化、上位分位でクリップ）。
    k = X_train.shape[1]
    mu = X_train.mean(axis=0)
    diff = X_train - mu
    # 共分散行列（数値安定化のため微小な対角正則化を付与）
    Sigma = np.cov(X_train, rowvar=False)
    Sigma = Sigma + np.eye(k, dtype=Sigma.dtype) * 1e-6
    # log|Σ| と Σ^-1 を計算
    sign, logdet = np.linalg.slogdet(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    # マハラノビス距離の二乗
    mahal2 = np.einsum('ij,jk,ik->i', diff, Sigma_inv, diff)
    log_pdf = -0.5 * (k * np.log(2 * np.pi) + logdet + mahal2)
    # 1/pdf に比例する重み（相対値で十分なので平均0化してexp）
    weights = np.exp(-(log_pdf - log_pdf.mean())).astype(np.float32)
    # 外れ値による極端な重みを抑える（上位5%でクリップ）
    # cap = np.quantile(weights, 0.999)
    # weights = np.minimum(weights, cap)
    # 平均を1に正規化
    weights = weights / (weights.mean() + 1e-8)

    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(weights), replacement=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    loss_history = []
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_losses = []
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
    
    # エポックごとの平均損失をプロット
    plt.figure()
    plt.plot(range(1, epochs+1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss over Epochs')
    plt.show()
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
    mse = mean_squared_error(y_test, y_pred)
    print(f"テストMSE: {mse:.6f}")
    # モデルを保存する
    import os
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'trained_model_NN.pth')
    torch.save(model.state_dict(), model_path)
    print(f"モデルを保存しました: {model_path}")
    return model

def predict(model, sample):
    # sampleは{'K%': ..., 'BB%': ..., 'Hard%': ..., 'GB%': ...}の形式で与える
    X_new = pd.DataFrame([sample]).values.astype(np.float32)
    tensor_input = torch.from_numpy(X_new)
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_input).numpy()
    return prediction[0]  # 出力は4要素のベクトル

if __name__ == "__main__":
    print("データ取得中...")
    data = fetch_data_from_pybaseball()
    # 同じ選手の複数年は別データとして扱う（= 選手-年レコードをそのまま件数化）
    record_count = len(data)
    print(f"データ取得完了: 選手-年レコード {record_count}件")
    model = train_model(data)
    
    # 実際の1人の選手について、特徴量およびターゲット割合ベクトルを取得
    features = ["K%", "BB%", "Hard%", "GB%"]
    target = ["1B_rate", "2B_rate", "3B_rate", "HR_rate"]
    player_sample = data.iloc[1]
    sample_features = player_sample[features].to_dict()
    predicted = predict(model, sample_features)
    actual = player_sample[target].values
    
    # 追加：選手の入力特徴量も表示する
    print("選手の入力特徴量 (K%, BB%, Hard%, GB%):", sample_features)
    print("実際の選手成績割合ベクトル（）:", actual)
    print("予測される選手成績割合ベクトル:", predicted)
