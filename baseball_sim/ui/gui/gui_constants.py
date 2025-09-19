"""
GUI constants and text definitions
"""
import sys

def get_font_settings():
    """システムに応じて適切なフォントを取得"""
    if sys.platform.startswith('win'):
        font_family = "MS Gothic"
    elif sys.platform.startswith('darwin'):
        font_family = "Hiragino Sans GB"
    else:
        font_family = "TkDefaultFont"
    
    default_font = (font_family, 10)
    title_font = (font_family, 24, "bold")
    
    return default_font, title_font

def get_ui_text():
    """UIテキストの辞書を取得"""
    return {
        "title": "Baseball Simulation",
        "new_game": "Start New Game",
        "exit": "Exit",
        "scoreboard": "Scoreboard",
        "team": "Team",
        "score": "Score",
        "hits": "Hits",
        "errors": "Errors",
        "away": "Away",
        "home": "Home",
        "inning_top": "Inning {0} Top",
        "inning_bottom": "Inning {0} Bottom",
        "outs": "Outs: {0}",
        "game_situation": "Game Situation",
        "pitcher": "Pitcher: {0}",
        "batter": "Batter: {0}",
        "strategy": "Strategy",
        "offense_strategy": "Offense Strategy",
        "defense_strategy": "Defense Strategy",
        "next_play": "Next Play",
        "play_result": "Play Result",
        "current_batter": "★Current Batter",
        "batting_order": "Batting Order",
        "bench": "Bench",
        "pinch_hit": "Pinch Hit",
        "close": "Close",
        "pitchers": "Pitchers",
        "current_pitcher": "Current Pitcher: {0}",
        "change": "Change",
        "fielders": "Fielders",
        "defense_sub": "Defensive Sub",
        "select_player": "Select player to replace:",
        "cancel": "Cancel",
        "error": "Error",
        "pinch_hit_error": "Could not make pinch hit substitution",
        "pitcher_change_error": "Could not change pitcher",
        "defense_sub_error": "Could not make defensive substitution",
        "game_start": "===== GAME START =====\n{0} vs {1}",
        "game_over": "===== GAME OVER =====",
        "final_score": "Final Score: {0} {1} - {2} {3}",
        "victory": "{0} wins",
        "tie": "It's a tie",
        "strike": "Strike out",
        "ball": "Walk",
        "stats": "Player Stats",
        "batting_stats": "Batting Stats",
        "pitching_stats": "Pitching Stats",
        "player": "Player",
        "ab": "AB",
        "h": "H",
        "avg": "AVG",
        "hr": "HR",
        "rbi": "RBI",
        "runs": "R",
        "bb": "BB",
        "k": "K",
        "single": "1B",
        "double": "2B",
        "triple": "3B",
        "ip": "IP",
        "er": "ER",
        "era": "ERA",
        "whip": "WHIP",
        "show_away": "Show Away Team",
        "show_home": "Show Home Team",
    }
