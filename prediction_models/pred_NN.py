"""Neural network based batting outcome prediction."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from prediction_models.data_utils import FEATURE_COLUMNS, fetch_batting_data

TARGET_COLUMNS = ["1B_rate", "2B_rate", "3B_rate", "HR_rate"]


def fetch_data_from_pybaseball(start_year: int = 2022, end_year: int = 2024, min_pa: int = 100) -> pd.DataFrame:
    """Load data via :mod:`pybaseball` and keep the columns used by the NN model."""

    data = fetch_batting_data(start_year=start_year, end_year=end_year, min_pa=min_pa)
    columns = [
        column
        for column in [
            "year",
            "Name",
            "Team",
            "PA",
            *FEATURE_COLUMNS,
            "SO_rate",
            "BB_rate",
            "1B_rate",
            "2B_rate",
            "3B_rate",
            "HR_rate",
            "OTH_rate",
        ]
        if column in data.columns
    ]
    return data[columns]


class Net(nn.Module):
    """Simple feed-forward network for rate prediction."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):  # type: ignore[override]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_model(data: pd.DataFrame) -> nn.Module:
    """Train the neural network to predict extra-base hit rates."""

    features = FEATURE_COLUMNS
    target = TARGET_COLUMNS

    df = data.copy()
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32)
    y = np.clip(y, 0.0, None)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    k = X_train.shape[1]
    mu = X_train.mean(axis=0)
    diff = X_train - mu
    sigma = np.cov(X_train, rowvar=False) + np.eye(k, dtype=X_train.dtype) * 1e-6
    sign, logdet = np.linalg.slogdet(sigma)
    sigma_inv = np.linalg.inv(sigma)
    mahal2 = np.einsum("ij,jk,ik->i", diff, sigma_inv, diff)
    log_pdf = -0.5 * (k * np.log(2 * np.pi) + logdet + mahal2)
    weights = np.exp(-(log_pdf - log_pdf.mean())).astype(np.float32)
    weights = weights / (weights.mean() + 1e-8)

    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights), num_samples=len(weights), replacement=True
    )
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
        loss_history.append(float(np.mean(epoch_losses)))

    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss over Epochs")
    plt.show()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
    mse = mean_squared_error(y_test, y_pred)
    print(f"テストMSE: {mse:.6f}")

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "trained_model_NN.pth")
    torch.save(model.state_dict(), model_path)
    print(f"モデルを保存しました: {model_path}")
    return model


def predict(model: nn.Module, sample: dict) -> np.ndarray:
    """Run inference for a single sample."""

    X_new = pd.DataFrame([sample]).values.astype(np.float32)
    tensor_input = torch.from_numpy(X_new)
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_input).numpy()
    return prediction[0]


if __name__ == "__main__":
    print("データ取得中...")
    data = fetch_data_from_pybaseball()
    record_count = len(data)
    print(f"データ取得完了: 選手-年レコード {record_count}件")
    model = train_model(data)

    features = FEATURE_COLUMNS
    target = TARGET_COLUMNS
    player_sample = data.iloc[1]
    sample_features = player_sample[features].to_dict()
    predicted = predict(model, sample_features)
    actual = player_sample[target].values

    print("選手の入力特徴量 (K%, BB%, Hard%, GB%):", sample_features)
    print("実際の選手成績割合ベクトル:", actual)
    print("予測される選手成績割合ベクトル:", predicted)
