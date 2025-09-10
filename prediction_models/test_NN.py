import os
import torch
from pred_NN import Net, predict

# テスト用のサンプルデータ
features = ['K%', 'BB%', 'Hard%', 'GB%']
status = [0.228, 0.085, 0.386, 0.446]
sample = dict(zip(features, status))

# モデルファイルのパスを指定
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/trained_model_NN.pth')

# ファイルの存在確認
if not os.path.exists(model_path):
    print(f"モデルファイルが見つかりません: {model_path}")
    exit(1)

# ニューラルネットワークモデルをロードして初期化
input_dim = 4  # ['K%', 'BB%', 'Hard%', 'GB%']
output_dim = 5  # ['1B_rate', '2B_rate', '3B_rate', 'HR_rate', 'OTH_rate']
model = Net(input_dim, output_dim)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# サンプルデータによる予測
print("予測実行中...")
prediction_result = predict(model, sample)
labels = ["単打", "二塁打", "三塁打", "本塁打", "アウト"]

print("\n=== 予測結果 ===")
for i, label in enumerate(labels):
    if i < len(prediction_result):
        print(f"{label}: {prediction_result[i]:.1%}")

# 基本統計
k_rate = status[0]   # K% (三振率)
bb_rate = status[1]  # BB% (四球率)
hit_rate = sum(prediction_result[:4])  # 安打率

# 確率計算
ab_rate = 1 - bb_rate  # 打数率（四球以外）
batting_average = hit_rate / ab_rate if ab_rate > 0 else 0
obp = hit_rate + bb_rate  # 出塁率
slugging = (prediction_result[0] + prediction_result[1]*2 + prediction_result[2]*3 + prediction_result[3]*4) / ab_rate if ab_rate > 0 else 0
ops = obp + slugging

print(f"\n=== 基本統計 ===")
print(f"打率: {batting_average:.3f}")
print(f"OPS: {ops:.3f}")

# 全確率の合計チェック
total_probability = k_rate + bb_rate + sum(prediction_result)
print(f"\n=== 確率合計 ===")
print(f"全確率の合計: {total_probability:.3f}")
