"""
ベースボールシミュレーションの定数定義
"""

# 野球基本ルール定数
INNINGS_PER_GAME = 9
PLAYERS_PER_TEAM = 9
OUTS_PER_INNING = 3
BASES_COUNT = 3
MAX_EXTRA_INNINGS = 12

# バント処理用定数
class BuntConstants:
    """バント処理に使用する定数"""
    # 走力関連（評価値、100が平均）
    STANDARD_RUNNER_SPEED = 100.0
    FAST_RUNNER_SPEED = 107.5
    VERY_FAST_RUNNER_SPEED = 113.2
    
    # 成功確率
    SACRIFICE_BUNT_SUCCESS_RATE = 0.75  # 75%（より現実的な値）
    SQUEEZE_PLAY_SUCCESS_RATE = 0.65   # 65%（スクイズは難しい）
    GENERAL_BUNT_SUCCESS_RATE = 0.70   # 70%
    SQUEEZE_PROBABILITY = 0.4  # スクイズプレーが選択される確率
    
    # ホームイン確率
    HOME_IN_PROBABILITY_FROM_THIRD = 0.6  # 60%（より現実的）
    HOME_IN_PROBABILITY_FAST_RUNNER = 0.5
    RUNNER_OUT_ON_SQUEEZE_FAIL = 0.3
    TRIPLE_ADVANCE_PROBABILITY = 0.2  # 一塁から三塁まで進む確率（減少）
    
    # 確率範囲制限
    MIN_ADVANCE_PROBABILITY = 0.4  # 最低40%
    MAX_ADVANCE_PROBABILITY = 0.90  # 最高90%
    TWO_OUT_PENALTY = 0.85  # 2アウト時の成功率減少

# ポジション定数
class Positions:
    CATCHER = "C"
    FIRST_BASE = "1B"
    SECOND_BASE = "2B"
    THIRD_BASE = "3B"
    SHORTSTOP = "SS"
    LEFT_FIELD = "LF"
    CENTER_FIELD = "CF"
    RIGHT_FIELD = "RF"
    DESIGNATED_HITTER = "DH"
    PITCHER = "P"
    
    # 必須守備ポジション（投手以外）
    DEFENSIVE_POSITIONS = [CATCHER, FIRST_BASE, SECOND_BASE, THIRD_BASE, 
                          SHORTSTOP, LEFT_FIELD, CENTER_FIELD, RIGHT_FIELD]
    
    # 全てのポジション（DH含む）
    ALL_POSITIONS = DEFENSIVE_POSITIONS + [DESIGNATED_HITTER]
    
    # 必須ポジション
    REQUIRED_POSITIONS = ALL_POSITIONS

# ゲーム結果の定数
class GameResults:
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    HOME_RUN = "home_run"
    WALK = "walk"
    STRIKEOUT = "strikeout"
    GROUND_OUT = "ground_out"
    FLY_OUT = "fly_out"
    SACRIFICE = "sacrifice"
    STOLEN_BASE = "stolen_base"
    CAUGHT_STEALING = "caught_stealing"
    STEAL_NOT_ALLOWED = "steal_not_allowed"

    # バント関連の結果
    BUNT_SINGLE = "bunt_single"
    SACRIFICE_BUNT = "sacrifice_bunt"
    BUNT_OUT = "bunt_out"
    BUNT_FAILED = "bunt_failed"
    SQUEEZE_SUCCESS = "squeeze_success"
    SQUEEZE_FAIL = "squeeze_fail"

    # 良い結果（攻撃側にとって）
    POSITIVE_RESULTS = [
        SINGLE,
        DOUBLE,
        TRIPLE,
        HOME_RUN,
        SACRIFICE,
        WALK,
        BUNT_SINGLE,
        SACRIFICE_BUNT,
        STOLEN_BASE,
        SQUEEZE_SUCCESS,
    ]

    # アウトになる結果
    OUT_RESULTS = [
        STRIKEOUT,
        GROUND_OUT,
        FLY_OUT,
        BUNT_OUT,
        BUNT_FAILED,
        CAUGHT_STEALING,
        SQUEEZE_FAIL,
    ]

    # バント関連の結果
    BUNT_RESULTS = [BUNT_SINGLE, SACRIFICE_BUNT, BUNT_OUT, BUNT_FAILED]

# ファイルパス定数
class FilePaths:
    DATA_DIR = "player_data/data"
    PLAYERS_JSON = "players.json"
    TEAMS_JSON = "teams.json"
    MODELS_DIR = "prediction_models/models"
    BATTING_MODEL = "batting_model.joblib"
    NN_MODEL = "trained_model_NN.pth"
    DEFAULT_SIMULATION_OUTPUT = "simulation_results.txt"
    TEAM_LIBRARY_DIR = "teams"
    TEAM_SELECTION_JSON = "team_selection.json"

# UI関連定数
class UIConstants:
    WINDOW_WIDTH = 1664  # 1280 * 1.3
    WINDOW_HEIGHT = 1040  # 800 * 1.3
    FIELD_CANVAS_WIDTH = 650  # 500 * 1.3
    FIELD_CANVAS_HEIGHT = 520  # 400 * 1.3
    ROSTER_DISPLAY_COUNT = 9
    
# 統計関連定数
class StatColumns:
    PLATE_APPEARANCES = "PA"
    AT_BATS = "AB"
    SINGLES = "1B"
    DOUBLES = "2B"
    TRIPLES = "3B"
    HOME_RUNS = "HR"
    WALKS = "BB"
    STRIKEOUTS = "SO"
    RUNS_BATTED_IN = "RBI"
    STOLEN_BASES = "SB"
    STEAL_ATTEMPTS = "SBA"

