import os
import math
import torch
import matplotlib.pyplot as plt
import pandas as pd
try:
    from prediction_models.prediction import Net
    from prediction_models.pred_NN import predict, fetch_data_from_pybaseball
except ModuleNotFoundError:
    import sys as _sys
    _sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prediction_models.prediction import Net
    from prediction_models.pred_NN import predict, fetch_data_from_pybaseball

# モデルファイルのパスを指定
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/trained_model_NN.pth')

# ファイルの存在確認
if not os.path.exists(model_path):
    print(f"モデルファイルが見つかりません: {model_path}")
    exit(1)

# ニューラルネットワークモデルをロードして初期化（出力次元は保存内容から推定）
state_dict = torch.load(model_path, map_location='cpu')
out_dim = 4
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

# =============================
# 10年分データでOPS散布図の作成
# =============================
try:
    print("\nデータ取得中 (2015-2024)...")
    data = fetch_data_from_pybaseball(start_year=2015, end_year=2024, min_pa=100)
    print(f"取得件数: {len(data)}")

    def _to_frac(x):
        try:
            v = float(x)
        except Exception:
            return 0.0
        return v / 100.0 if v > 1.0 else v

    actual_ops_list = []
    pred_ops_list = []

    features = ['K%', 'BB%', 'Hard%', 'GB%']
    targets = ['1B_rate', '2B_rate', '3B_rate', 'HR_rate', 'BB_rate']

    for _, row in data.iterrows():
        # 実績OPS
        if not all(col in row for col in targets):
            continue
        single_r, double_r, triple_r, hr_r, bb_r = [float(row[col]) for col in targets]
        hit_r = single_r + double_r + triple_r + hr_r
        ab_r = max(1e-8, 1.0 - bb_r)
        tb_r = single_r + 2 * double_r + 3 * triple_r + 4 * hr_r
        actual_obp = hit_r + bb_r
        actual_slg = tb_r / ab_r
        actual_ops = actual_obp + actual_slg

        # 予測OPS
        if not all(f in row for f in features):
            continue
        sample_features = {f: float(row[f]) for f in features}
        raw = predict(model, sample_features)
        vals = list(raw)

        k_rate = _to_frac(row['K%'])
        bb_rate = _to_frac(row['BB%'])
        other_prob = max(0.0, 1.0 - k_rate - bb_rate)

        single = double = triple = hr = out_woSO = 0.0
        if len(vals) >= 5:
            single, double, triple, hr, out_woSO = [max(0.0, float(v)) for v in vals[:5]]
            total = single + double + triple + hr + out_woSO
            if total > 0:
                scale = other_prob / total
                single *= scale
                double *= scale
                triple *= scale
                hr *= scale
                out_woSO *= scale
            else:
                single = double = triple = hr = 0.0
                out_woSO = other_prob
        elif len(vals) == 4:
            single, double, triple, hr = [max(0.0, float(v)) for v in vals]
            total_hits = single + double + triple + hr
            if total_hits > other_prob and total_hits > 0:
                scale = other_prob / total_hits
                single *= scale
                double *= scale
                triple *= scale
                hr *= scale
            out_woSO = max(0.0, other_prob - (single + double + triple + hr))
        else:
            # 予期しない形状はスキップ
            continue

        pred_hit_r = single + double + triple + hr
        pred_ab_r = max(1e-8, 1.0 - bb_rate)
        pred_tb_r = single + 2 * double + 3 * triple + 4 * hr
        pred_obp = pred_hit_r + bb_rate
        pred_slg = pred_tb_r / pred_ab_r
        pred_ops = pred_obp + pred_slg

        # 値が有限であることを確認
        if not (math.isfinite(actual_ops) and math.isfinite(pred_ops)):
            continue

        actual_ops_list.append(actual_ops)
        pred_ops_list.append(pred_ops)

    if actual_ops_list and pred_ops_list:
        # 散布図
        plt.figure(figsize=(6, 6))
        plt.scatter(actual_ops_list, pred_ops_list, alpha=0.4, s=12)
        mn = min(min(actual_ops_list), min(pred_ops_list))
        mx = max(max(actual_ops_list), max(pred_ops_list))
        pad = 0.05
        lo = max(0.0, mn - pad)
        hi = mx + pad
        plt.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='y=x')
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel('Actual OPS')
        plt.ylabel('Predicted OPS')
        plt.title('Actual vs Predicted OPS (2014-2023)')

        # 簡易相関係数の表示
        try:
            import numpy as np

            r = np.corrcoef(actual_ops_list, pred_ops_list)[0, 1]
            plt.legend(title=f'r = {r:.3f}')
        except Exception:
            plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("散布図を作成する十分なデータがありませんでした。")
except Exception as e:
    print(f"散布図作成でエラー: {e}")

# 600打席での期待成績
PA = 600
k_cnt = round(PA * k_rate)
bb_cnt = round(PA * bb_rate)
single_cnt = round(PA * single)
double_cnt = round(PA * double)
triple_cnt = round(PA * triple)
hr_cnt = round(PA * hr)

# 残りは非三振アウトに割当（合計がPAになるように調整）
consumed = k_cnt + bb_cnt + single_cnt + double_cnt + triple_cnt + hr_cnt
out_woSO_cnt = max(0, PA - consumed)

hits_cnt = single_cnt + double_cnt + triple_cnt + hr_cnt
ab_cnt = PA - bb_cnt
tb_cnt = single_cnt + 2 * double_cnt + 3 * triple_cnt + 4 * hr_cnt
batting_avg_cnt = (hits_cnt / ab_cnt) if ab_cnt > 0 else 0.0
obp_cnt = (hits_cnt + bb_cnt) / PA
slg_cnt = (tb_cnt / ab_cnt) if ab_cnt > 0 else 0.0
ops_cnt = obp_cnt + slg_cnt

print("\n=== 600打席の期待成績（概算） ===")
print(f"PA: {PA}")
print(f"BB: {bb_cnt}")
print(f"SO: {k_cnt}")
print(f"1B: {single_cnt}, 2B: {double_cnt}, 3B: {triple_cnt}, HR: {hr_cnt}")
print(f"Out(非三振): {out_woSO_cnt}")
print(f"H: {hits_cnt}, AB: {ab_cnt}, TB: {tb_cnt}")
print(f"AVG: {batting_avg_cnt:.3f}, OBP: {obp_cnt:.3f}, SLG: {slg_cnt:.3f}, OPS: {ops_cnt:.3f}")
