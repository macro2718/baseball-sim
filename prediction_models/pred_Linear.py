import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pybaseball import batting_stats
import joblib
import os

def fetch_data_from_pybaseball(start_year=2020, end_year=2024, min_pa=100):
    """pybaseballを用いて2021～2023年のデータを取得し、NN版と同じ特徴量・ターゲット列を返す"""
    from pybaseball import batting_stats
    frames = []
    for year in range(start_year, end_year + 1):
        df = batting_stats(year)
        df['year'] = year
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data[data['PA'] >= min_pa].reset_index(drop=True)
    if '1B' not in data.columns:
        data['1B'] = data['H'] - (data['2B'] + data['3B'] + data['HR'])
    required = ['K%', 'BB%', 'Hard%', 'GB%', 'PA', 'SO', 'BB', '1B', '2B', '3B', 'HR']
    data = data.dropna(subset=required).reset_index(drop=True)
    data['SO_rate']  = data['SO']  / data['PA'].replace(0,1)
    data['BB_rate']  = data['BB']  / data['PA'].replace(0,1)
    data['1B_rate']  = data['1B']  / data['PA'].replace(0,1)
    data['2B_rate']  = data['2B']  / data['PA'].replace(0,1)
    data['3B_rate']  = data['3B']  / data['PA'].replace(0,1)
    data['HR_rate']  = data['HR']  / data['PA'].replace(0,1)
    data['OTH_rate'] = (data['PA'] - (data[['SO','BB','1B','2B','3B','HR']].sum(axis=1))) / data['PA'].replace(0,1)
    cols = ['year','Name','Team','PA','K%','BB%','Hard%','GB%',
            'SO_rate','BB_rate','1B_rate','2B_rate','3B_rate','HR_rate','OTH_rate']
    existing = [c for c in cols if c in data.columns]
    return data[existing]

def train_model(data, min_pa=100):
    """
    線形回帰モデルで、全打数における一塁打、二塁打、三塁打、本塁打、三振以外のアウト割合を予測する。
    特徴量: ['K%', 'BB%', 'Hard%', 'GB%']
    ターゲット: ['1B_rate', '2B_rate', '3B_rate', 'HR_rate', 'OTH_rate']
    
    Parameters:
    -----------
    data : pandas.DataFrame
        訓練データ
    min_pa : int, default=100
        分析対象とする最小の打席数
    """
    # 最小打数未満のデータを除外
    filtered_data = data[data['PA'] >= min_pa]
    print(f"PA >= {min_pa}の選手のみ使用: {len(filtered_data)}人（元の{len(data)}人から）")
    
    features = ['K%', 'BB%', 'Hard%', 'GB%']
    target = ['1B_rate', '2B_rate', '3B_rate', 'HR_rate', 'OTH_rate']
    X = filtered_data[features]
    y = filtered_data[target]
    
    # 全データを訓練に使用
    model = LinearRegression()
    model.fit(X, y)
    
    # 訓練データにおけるMSEを計算
    train_preds = model.predict(X)
    train_mse = mean_squared_error(y, train_preds)
    
    print("訓練データにおけるMSE:", train_mse)
    
    # モデルパラメータの出力
    print("\nモデルパラメータ:")
    print("切片 (intercept):", model.intercept_)
    print("係数 (coefficients):")
    for i, feature in enumerate(features):
        print(f"  {feature}:")
        for j, t in enumerate(target):
            print(f"    {t}: {model.coef_[j][i]:.6f}")
    
    # モデル情報を辞書で返す
    model_info = {
        'model': model,
        'train_mse': train_mse,
        'features': features,
        'target': target,
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }
    
    return model_info

def predict(model, sample):
    """
    sample は {'K%': ..., 'BB%': ..., 'Hard%': ..., 'GB%': ...} の形式。
    1選手分の予測値を返す。
    """
    # 入力データの検証
    required_features = ['K%', 'BB%', 'Hard%', 'GB%']
    for feature in required_features:
        if feature not in sample:
            raise ValueError(f"必須特徴量 '{feature}' が入力に含まれていません。")
    
    X_new = pd.DataFrame([sample])
    return model.predict(X_new)[0]

def predict_multiple(model, samples_df, features):
    """
    複数の選手データに対して予測を行います。
    
    Parameters:
    -----------
    model : 訓練済みモデル
    samples_df : pandas.DataFrame, 選手データを含むデータフレーム
    features : list, 使用する特徴量のリスト
    
    Returns:
    --------
    pandas.DataFrame : 選手名と予測結果を含むデータフレーム
    """
    if samples_df.empty:
        raise ValueError("空のデータフレームが提供されました。")
    
    results = []
    for idx, player in samples_df.iterrows():
        try:
            sample_features = player[features].to_dict()
            predicted = predict(model, sample_features)
            
            result = {
                'Name': player['Name'] if 'Name' in player else f"Player_{idx}",
                '1B_rate_pred': predicted[0],
                '2B_rate_pred': predicted[1],
                '3B_rate_pred': predicted[2],
                'HR_rate_pred': predicted[3],
                'OTH_rate_pred': predicted[4]
            }
            results.append(result)
        except Exception as e:
            print(f"プレイヤー {idx} の予測中にエラーが発生しました: {e}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("データ取得中...")
    data = fetch_data_from_pybaseball()
    print(f"データ取得完了。{len(data)}人の選手データを取得しました。")
    # 欠損値除去
    features = ['K%','BB%','Hard%','GB%']
    target   = ['1B_rate','2B_rate','3B_rate','HR_rate','OTH_rate']
    data = data.dropna(subset=features+target).reset_index(drop=True)
    # モデル学習
    model_info = train_model(data)
    model = model_info['model']
    # モデル保存
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'batting_model.joblib')
    joblib.dump(model_info, model_path)
    print(f"モデルを保存しました: {model_path}")
    # サンプル1人で予測
    player = data.iloc[1]
    sample_features = player[features].to_dict()
    predicted = predict(model, sample_features)
    actual = player[target].values
    print("選手の入力特徴量 (K%, BB%, Hard%, GB%):", sample_features)
    print("実際の選手成績割合ベクトル:", actual)
    print("予測される選手成績割合ベクトル:", predicted)
