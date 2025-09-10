import json
import random
import os
import sys

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .player_factory import PlayerFactory

# パスの設定
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# === 設定項目 ===
SAMPLE_MODE = False  # True: 平均値固定モード, False: ランダムモード
SEED_VALUE = 42      # 乱数シード値

# 乱数シードを設定（再現性のため）
random.seed(SEED_VALUE)

# 日本人の姓と名のサンプル
LAST_NAMES = [
    "Suzuki", "Tanaka", "Sato", "Takahashi", "Watanabe", "Ito", "Yamamoto", "Nakamura", 
    "Kobayashi", "Kato", "Matsumoto", "Yamada", "Inoue", "Kimura", "Hayashi", "Shimizu", 
    "Saito", "Yamazaki", "Abe", "Mori", "Ikeda", "Hashimoto", "Ishikawa", "Ogawa", 
    "Yamashita", "Okada", "Goto", "Hasegawa", "Murakami", "Kondo", "Ishii", "Maeda", 
    "Fujita", "Aoki", "Sakamoto", "Endo", "Arai", "Ota", "Fujii", "Nishimura", "Fukuda", 
    "Miura", "Takeuchi", "Matsuda", "Nakajima", "Ueda", "Takagi"
]

FIRST_NAMES = [
    "Hiroshi", "Takashi", "Taro", "Jiro", "Kazuki", "Yuki", "Kenta", "Daiki", "Tatsuya", 
    "Ryo", "Yuta", "Kazuma", "Shota", "Naoki", "Akira", "Yuji", "Koji", "Takuya", "Yosuke", 
    "Makoto", "Yu", "Ryota", "Yusuke", "Satoshi", "Shigeru", "Keisuke", "Daisuke", "Takeshi", 
    "Masaki", "Soichiro", "Eiji", "Hideo", "Dai", "Mamoru", "Tsuyoshi", "Ken", "Masaru", 
    "Jin", "Shingo", "Kazuo", "Tadashi", "Sho", "Ichiro", "Takumi", "Kenji"
]

# ポジションのリスト
POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]

# ポジション適性のルール
def get_eligible_positions(primary_position):
    """主ポジションに基づいて守備可能ポジションを決定"""
    eligible = [primary_position]
    
    # DHは全ての選手が付けられるように追加
    eligible.append("DH")
    
    # ランダムにポジションを拡張（0～2個の範囲）
    additional_positions = random.randint(0, 2)
    available_positions = [pos for pos in POSITIONS if pos != primary_position]
    
    if additional_positions > 0 and available_positions:
        # 利用可能なポジションからランダムに選択
        num_to_add = min(additional_positions, len(available_positions))
        additional = random.sample(available_positions, num_to_add)
        eligible.extend(additional)
        
    return list(set(eligible))  # 重複を除去

def generate_random_name():
    """ランダムな日本人名を生成する"""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    return f"{first_name} {last_name}"

def generate_unique_names(count):
    """指定数の重複しない名前を生成する"""
    names = set()
    while len(names) < count:
        name = generate_random_name()
        if name not in names:
            names.add(name)
    return list(names)

def generate_pitcher(name, pitcher_type="SP", use_sample_mode=False):
    """投手データを生成する（PlayerFactoryに委譲）"""
    pitcher = PlayerFactory.create_pitcher(name, pitcher_type, use_sample_mode)
    # 辞書形式で返す（既存コードとの互換性のため）
    return {
        "name": pitcher.name,
        "pitcher_type": pitcher.pitcher_type,
        "k_pct": pitcher.k_pct,
        "bb_pct": pitcher.bb_pct,
        "hard_pct": pitcher.hard_pct,
        "gb_pct": pitcher.gb_pct,
        "stamina": pitcher.stamina,
        "throws": pitcher.throws
    }

def generate_batter(name, primary_position=None, use_sample_mode=False):
    """野手データを生成する（PlayerFactoryに委譲）"""
    if primary_position is None:
        primary_position = random.choice(POSITIONS)
    
    eligible_positions = get_eligible_positions(primary_position)
    batter = PlayerFactory.create_batter(name, eligible_positions, use_sample_mode)
    
    # 辞書形式で返す（既存コードとの互換性のため）
    return {
        "name": batter.name,
        "eligible_positions": batter.eligible_positions,
        "k_pct": batter.k_pct,
        "bb_pct": batter.bb_pct,
        "hard_pct": batter.hard_pct,
        "gb_pct": batter.gb_pct,
        "speed": batter.speed,
        "fielding_skill": batter.fielding_skill,
        "bats": batter.bats
    }

def create_optimal_lineup(player_names, all_batters):
    """選手リストから9つの相異なるポジション（C～RF＋DH）でラインナップを作成"""
    # 選手名からデータを取り出す
    players_data = {b["name"]: b for b in all_batters if b["name"] in player_names}
    lineup = []
    used_players = set()

    # 各ポジションを固定順（C～RF, DH）で割当
    for pos in POSITIONS + ["DH"]:
        # 未割当選手の中で該当ポジション適性のある選手を探す
        candidates = [n for n in player_names if n not in used_players and pos in players_data[n]["eligible_positions"]]
        if candidates:
            selected = candidates[0]
        else:
            # 適性がなければ未割当選手から先頭で選択
            selected = next(n for n in player_names if n not in used_players)
        used_players.add(selected)
        lineup.append({"name": selected, "position": pos})

    return lineup

def generate_teams_and_players(pitchers_per_team=10, batters_per_team=15, use_sample_mode=False):
    """チームデータとプレイヤーデータを生成する"""
    # 必要な選手名の総数を計算
    total_pitchers = pitchers_per_team * 2  # ホームとアウェイ
    total_batters = batters_per_team * 2    # ホームとアウェイ
    
    # すべての選手名を一度に生成（重複なし）
    all_unique_names = generate_unique_names(total_pitchers + total_batters)
    
    # 名前を投手と野手に分配
    all_pitchers_names = all_unique_names[:total_pitchers]
    all_batters_names = all_unique_names[total_pitchers:]
    
    # チーム別に分配
    home_pitchers_names = all_pitchers_names[:pitchers_per_team]
    away_pitchers_names = all_pitchers_names[pitchers_per_team:]
    
    home_batters_names = all_batters_names[:batters_per_team]
    away_batters_names = all_batters_names[batters_per_team:]
    
    # 選手データ生成
    all_pitchers = []
    
    # 各チームの投手を生成（先発投手3-4人、中継ぎ投手6-7人）
    starters_per_team = pitchers_per_team // 3  # 約1/3が先発投手
    
    pitcher_idx = 0
    for team_idx in range(2):  # ホームとアウェイ
        team_pitcher_names = all_pitchers_names[team_idx * pitchers_per_team:(team_idx + 1) * pitchers_per_team]
        
        # 先発投手を生成
        for i in range(starters_per_team):
            name = team_pitcher_names[i]
            all_pitchers.append(generate_pitcher(name, "SP", use_sample_mode))
        
        # 中継ぎ投手を生成
        for i in range(starters_per_team, pitchers_per_team):
            name = team_pitcher_names[i]
            all_pitchers.append(generate_pitcher(name, "RP", use_sample_mode))
    
    all_batters = []
    # 各チームで8つのポジションを保証
    for team_idx, team_batters_names in enumerate([home_batters_names, away_batters_names]):
        # 最初の8人は各ポジション1人ずつ
        for i, position in enumerate(POSITIONS):
            name = team_batters_names[i]
            all_batters.append(generate_batter(name, position, use_sample_mode))
        
        # 残りの選手はランダムポジション
        for name in team_batters_names[8:]:
            all_batters.append(generate_batter(name, None, use_sample_mode))
    
    # 適切なポジション配置でチームデータ構築
    home_lineup = create_optimal_lineup(home_batters_names[:9], all_batters)
    away_lineup = create_optimal_lineup(away_batters_names[:9], all_batters)
    
    home_team = {
        "name": "Home Team",
        "pitchers": home_pitchers_names,
        "lineup": home_lineup,
        "bench": home_batters_names[9:]
    }
    
    away_team = {
        "name": "Away Team",
        "pitchers": away_pitchers_names,
        "lineup": away_lineup,
        "bench": away_batters_names[9:]
    }
    
    teams_data = {
        "home_team": home_team,
        "away_team": away_team
    }
    
    players_data = {
        "pitchers": all_pitchers,
        "batters": all_batters
    }
    
    return teams_data, players_data

def save_json(data, filename):
    """JSONデータをファイルに保存"""
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"データを {file_path} に保存しました")

def main():
    """メイン関数"""
    mode = "サンプル" if SAMPLE_MODE else "ランダム"
    print(f"野球シミュレーションデータ生成を開始...（{mode}モード、シード: {SEED_VALUE}）")
    
    teams_data, players_data = generate_teams_and_players(13, 16, SAMPLE_MODE)
    save_json(teams_data, "teams.json")
    save_json(players_data, "players.json")
    print("データ生成完了！")

if __name__ == "__main__":
    main()
