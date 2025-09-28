import os
import torch
try:
    from prediction_models.prediction import Net
    from prediction_models.pred_NN import predict
except ModuleNotFoundError:
    import sys as _sys
    _sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prediction_models.prediction import Net
    from prediction_models.pred_NN import predict

# テスト用のサンプルデータ
features = ['K%', 'BB%', 'Hard%', 'GB%']
status = [0.228, 0.085, 0.386, 0.446]
status = [0.226, 0.090, 0.302, 0.486]
sample = dict(zip(features, status))

# モデルファイルのパスを指定
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/trained_model_NN.pth')

# ファイルの存在確認
if not os.path.exists(model_path):
    print(f"モデルファイルが見つかりません: {model_path}")
    exit(1)

# ニューラルネットワークモデルをロードして初期化（出力次元は保存内容から推定）
state_dict = torch.load(model_path, map_location='cpu')
out_dim = 5
try:
    w = state_dict.get('fc3.weight')
    if w is not None and hasattr(w, 'shape'):
        out_dim = int(w.shape[0])
except Exception:
    out_dim = 5

input_dim = 4  # ['K%', 'BB%', 'Hard%', 'GB%']
model = Net(input_dim, out_dim)
model.load_state_dict(state_dict)
model.eval()

print("予測実行中...")
raw = predict(model, sample)
labels = ["単打", "二塁打", "三塁打", "本塁打", "アウト"]

# 入力ステータスからK/BBを設定
k_rate = status[0]
bb_rate = status[1]
other_prob = max(0.0, 1.0 - k_rate - bb_rate)

# モデル出力を4出力/5出力のどちらにも対応させる
single = double = triple = hr = out_woSO = 0.0
vals = list(raw)
if len(vals) >= 5:
    single, double, triple, hr, out_woSO = [max(0.0, float(v)) for v in vals[:5]]
    total = single + double + triple + hr + out_woSO
    if total > 0:
        scale = other_prob / total
        single = max(0.0, single * scale)
        double = max(0.0, double * scale)
        triple = max(0.0, triple * scale)
        hr = max(0.0, hr * scale)
        out_woSO = max(0.0, out_woSO * scale)
    else:
        single = double = triple = hr = 0.0
        out_woSO = other_prob
elif len(vals) == 4:
    single, double, triple, hr = [max(0.0, float(v)) for v in vals]
    total_hits = single + double + triple + hr
    if total_hits > other_prob and total_hits > 0:
        scale = other_prob / total_hits
        single = max(0.0, single * scale)
        double = max(0.0, double * scale)
        triple = max(0.0, triple * scale)
        hr = max(0.0, hr * scale)
    out_woSO = max(0.0, other_prob - (single + double + triple + hr))
else:
    # 予期しない形状は簡易デフォルト
    single, double, triple, hr, out_woSO = 0.15, 0.05, 0.01, 0.03, other_prob - 0.24
    out_woSO = max(0.0, out_woSO)

scaled = [single, double, triple, hr, out_woSO]

print("\n=== 予測結果(スケーリング後) ===")
for i, label in enumerate(labels):
    print(f"{label}: {scaled[i]:.1%}")

# 基本統計（スケーリング後の値で計算）
hit_rate = single + double + triple + hr
ab_rate = 1 - bb_rate
batting_average = hit_rate / ab_rate if ab_rate > 0 else 0
obp = hit_rate + bb_rate
slugging = (single + double*2 + triple*3 + hr*4) / ab_rate if ab_rate > 0 else 0
ops = obp + slugging

print(f"\n=== 基本統計 ===")
print(f"打率: {batting_average:.3f}")
print(f"OPS: {ops:.3f}")

# 全確率の合計チェック（K/BB/モデル由来成分）
total_probability = k_rate + bb_rate + hit_rate + out_woSO
print(f"\n=== 確率合計 ===")
print(f"全確率の合計: {total_probability:.3f}")
