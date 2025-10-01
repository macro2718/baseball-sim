# baseball-sim

野球シミュレーションと簡易Web UI（Flask）を備えたリポジトリです。ブラウザでプレイの進行、代打・代走・投手交代などの操作、結果ログの確認、複数試合シミュレーションの実行ができます。機械学習を用いた打席結果予測（線形／NN）の読み込みにも対応（未インストール時は自動フォールバック）。

## 特長
- FlaskベースのWeb UI（テンプレート・静的ファイル同梱）
- 選手・チームデータのJSON読込（`player_data/data/`）
- 代打／代走／守備交代／投手交代などの戦略操作
- 複数試合の一括シミュレーションと結果出力（`simulation_results/`）
- 予測モデルのプラガブル読み込み（`prediction_models/models/`）

## 依存関係（最小）
- Python 3.9以上（`list[str]`表記などを使用）
- Flask（Webアプリ実行に必要）

requirements.txt は最小構成（Flask のみ）です。予測モデルや学習スクリプトを使う場合の追加パッケージは「オプション依存関係」を参照してください。

## セットアップ
- 仮想環境の作成と有効化（例）
  - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
  - Windows(PowerShell): `py -3 -m venv .venv; .venv\\Scripts\\Activate.ps1`
- 依存インストール: `pip install -r requirements.txt`

## 実行（Web UI）
- 環境変数を設定して起動
  - macOS/Linux: `export FLASK_APP=baseball_sim.ui.app:app && flask run`
  - Windows(PowerShell): `$Env:FLASK_APP='baseball_sim.ui.app:app'; flask run`
- デフォルト: http://127.0.0.1:5000/
- `.venv/bin/python -m flask --app baseball_sim.ui.app run`

## 設定
- 設定ファイル: `baseball_sim/config/config.json`
  - `simulation.prediction_model_type`: `"linear"` または `"nn"`
  - `simulation.use_ml_prediction`: 予測モデル利用のON/OFF（未インストール時は自動フォールバック）

### リーグ平均の管理（K%, BB%, Hard%, GB%）
- 参照元は `baseball_sim/config/league.py` の `LeagueAverages`。
- デフォルト値は `settings.py` の `league.averages` に定義（K%: 22.8, BB%: 8.5, Hard%: 38.6, GB%: 44.6）。
- ユーザー設定で上書きする場合は `baseball_sim/config/config.json` に次を追加:
  ```json
  {
    "league": {
      "averages": { "k_pct": 22.8, "bb_pct": 8.5, "hard_pct": 38.6, "gb_pct": 44.6 }
    }
  }
  ```
- 利用箇所:
  - 打席結果の確率分布計算: `baseball_sim/gameplay/outcomes/probability.py`
  - 新規選手作成のデフォルト能力値: `baseball_sim/data/player_factory.py`
  - 一部ユーティリティ（バント計算など）: `baseball_sim/gameplay/utils.py`

## データ配置
- 選手・チーム定義: `player_data/data/players.json`, `player_data/data/teams.json`
- チームライブラリ（UIからの読み書き対象）: `player_data/teams/*.json`
- 予測モデル（任意）: `prediction_models/models/`
  - 線形モデル（joblib）: `batting_model.joblib`
  - NNモデル（PyTorch）: `trained_model_NN.pth`

## シミュレーションの一括実行（スクリプトから）
- 例: `python -c "from baseball_sim.interface.simulation import simulate_games; simulate_games(num_games=20)"`
- 結果は戻り値として返されるため、必要に応じて任意の形式で保存してください。
- リーグ戦モード: `simulate_games(league_options={"teams": [...], "games_per_card": 3, "cards_per_opponent": 2})`
  - `teams` には `player_data/teams.json` と同形式の辞書をリストで渡します（同一チームを複数回指定可能）。
  - チーム数は偶数である必要があります。1カードあたりの試合数（c）、カードの繰り返し回数（d）を設定すると、
    全チームが1日1試合ずつ消化する日程が自動生成されます。

## オプション依存関係（必要に応じて）
- 予測モデルの利用／学習用
  - joblib（線形モデル読込）
  - PyTorch（NNモデル読込・学習）
  - scikit-learn, pandas, numpy, matplotlib, tqdm, pybaseball（学習スクリプト）
- これらは requirements.txt には含めていません。必要に応じて個別にインストールしてください。

インストール例（学習・検証を行う場合）:
- `pip install joblib scikit-learn pandas numpy matplotlib tqdm`
- PyTorch は環境に応じて公式サイトのインストール手順に従ってください。

## ディレクトリ構成（抜粋）
- `baseball_sim/`
  - `ui/` Flaskアプリ（`app.py`, `routes.py`, `templates/`, `static/`）
  - `gameplay/` 試合進行・確率計算・結果処理
  - `data/` データローダと選手・チーム生成
  - `prediction/` 予測モデルの読み込み
  - `config/` 設定・パス管理
  - `infrastructure/` ロギング等の基盤
- `player_data/` データファイル・チーム定義
- `prediction_models/` 予測モデルや学習スクリプト
- `simulation_results/` 実行ログ・出力

## トラブルシュート
- Flask が見つからない: `pip install -r requirements.txt`
- PyTorch 未インストールで `nn` を選択: 自動で線形モデルにフォールバックします（ログに警告が出ます）。
- joblib 未インストールまたはモデル未配置: デフォルトの確率を用いたシミュレーションにフォールバックします。

---
Contributions are welcome.
