"""
Scoreboard display functionality for the Baseball GUI
"""
import tkinter as tk
from tkinter import ttk

class ScoreboardManager:
    def __init__(self, parent_frame, text):
        self.parent_frame = parent_frame
        self.text = text
        self.game_state = None  # ゲーム状態を保存
        self.away_team_label = None
        self.home_team_label = None
        self.away_score_label = None
        self.home_score_label = None
        self.away_hits_label = None
        self.home_hits_label = None
        self.away_errors_label = None
        self.home_errors_label = None
        self.away_inning_labels = []
        self.home_inning_labels = []
        self.inning_label = None
        self.pitcher_label = None
        self.batter_label = None
    
    def set_game_state(self, game_state):
        """ゲーム状態を設定"""
        self.game_state = game_state
    
    def create_scoreboard(self):
        """スコアボードを作成"""
        # スコアボードをフィールド画面の下部に配置
        scoreboard_frame = ttk.LabelFrame(self.parent_frame, text=self.text["scoreboard"])
        scoreboard_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.BOTTOM)
        
        # イニングヘッダー行を12回分に変更
        inning_header_frame = ttk.Frame(scoreboard_frame)
        inning_header_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(inning_header_frame, text="Team", width=7).grid(row=0, column=0, padx=1, pady=1, sticky="w")
        for i in range(12):
            ttk.Label(inning_header_frame, text=str(i+1), width=3).grid(row=0, column=i+1, padx=1, pady=1)
        ttk.Label(inning_header_frame, text="R", width=3).grid(row=0, column=13, padx=1, pady=1)
        ttk.Label(inning_header_frame, text="H", width=3).grid(row=0, column=14, padx=1, pady=1)
        ttk.Label(inning_header_frame, text="E", width=3).grid(row=0, column=15, padx=1, pady=1)
        
        # チーム得点表（アウェイチーム）を12回分に変更
        away_frame = ttk.Frame(scoreboard_frame)
        away_frame.pack(fill=tk.X, padx=5, pady=2)
        self.away_team_label = ttk.Label(away_frame, text=self.text["away"], width=7)
        self.away_team_label.grid(row=0, column=0, padx=1, pady=1, sticky="w")
        self.away_inning_labels = []
        for i in range(12):
            label = ttk.Label(away_frame, text="", width=3, relief="solid", borderwidth=1)
            label.grid(row=0, column=i+1, padx=1, pady=1)
            self.away_inning_labels.append(label)
        self.away_score_label = ttk.Label(away_frame, text="0", width=3, relief="solid", borderwidth=1)
        self.away_score_label.grid(row=0, column=13, padx=1, pady=1)
        self.away_hits_label = ttk.Label(away_frame, text="0", width=3, relief="solid", borderwidth=1)
        self.away_hits_label.grid(row=0, column=14, padx=1, pady=1)
        self.away_errors_label = ttk.Label(away_frame, text="0", width=3, relief="solid", borderwidth=1)
        self.away_errors_label.grid(row=0, column=15, padx=1, pady=1)
        
        # チーム得点表（ホームチーム）を12回分に変更
        home_frame = ttk.Frame(scoreboard_frame)
        home_frame.pack(fill=tk.X, padx=5, pady=2)
        self.home_team_label = ttk.Label(home_frame, text=self.text["home"], width=7)
        self.home_team_label.grid(row=0, column=0, padx=1, pady=1, sticky="w")
        self.home_inning_labels = []
        for i in range(12):
            label = ttk.Label(home_frame, text="", width=3, relief="solid", borderwidth=1)
            label.grid(row=0, column=i+1, padx=1, pady=1)
            self.home_inning_labels.append(label)
        self.home_score_label = ttk.Label(home_frame, text="0", width=3, relief="solid", borderwidth=1)
        self.home_score_label.grid(row=0, column=13, padx=1, pady=1)
        self.home_hits_label = ttk.Label(home_frame, text="0", width=3, relief="solid", borderwidth=1)
        self.home_hits_label.grid(row=0, column=14, padx=1, pady=1)
        self.home_errors_label = ttk.Label(home_frame, text="0", width=3, relief="solid", borderwidth=1)
        self.home_errors_label.grid(row=0, column=15, padx=1, pady=1)
        
        # イニング情報表示（アウトカウント表示は削除）
        inning_frame = ttk.Frame(scoreboard_frame)
        inning_frame.pack(fill=tk.X, padx=5, pady=5)
        self.inning_label = ttk.Label(inning_frame, text=self.text["inning_top"].format(1))
        self.inning_label.pack(side=tk.LEFT, padx=5)
        
        return scoreboard_frame
    
    def update_scoreboard(self, game_state):
        """スコアボードの情報を更新"""
        # チーム名
        self.away_team_label.config(text=game_state.away_team.name)
        self.home_team_label.config(text=game_state.home_team.name)
        
        # スコア
        self.away_score_label.config(text=str(game_state.away_score))
        self.home_score_label.config(text=str(game_state.home_score))
        
        # イニング得点を更新（12回分に変更）
        for label in self.away_inning_labels:
            label.config(text="")
        for label in self.home_inning_labels:
            label.config(text="")

        if hasattr(game_state, 'inning_scores'):
            away_scores = game_state.inning_scores[0] if len(game_state.inning_scores) > 0 else []
            home_scores = game_state.inning_scores[1] if len(game_state.inning_scores) > 1 else []

            for i, value in enumerate(away_scores[:len(self.away_inning_labels)]):
                self.away_inning_labels[i].config(text=str(value))

            for i, value in enumerate(home_scores[:len(self.home_inning_labels)]):
                self.home_inning_labels[i].config(text=str(value))
        
        # 安打数（シングルヒット、ダブル、トリプル、ホームランの合計）
        away_hits = sum(player.stats.get("1B", 0) + player.stats.get("2B", 0) + player.stats.get("3B", 0) + player.stats.get("HR", 0) for player in game_state.away_team.lineup)
        home_hits = sum(player.stats.get("1B", 0) + player.stats.get("2B", 0) + player.stats.get("3B", 0) + player.stats.get("HR", 0) for player in game_state.home_team.lineup)
        
        self.away_hits_label.config(text=str(away_hits))
        self.home_hits_label.config(text=str(home_hits))
        
        # エラー数（サンプル - 実際にはエラー数を追跡する機能が必要）
        self.away_errors_label.config(text="0")
        self.home_errors_label.config(text="0")
        
        # イニング情報
        inning_str = self.text["inning_top" if game_state.is_top_inning else "inning_bottom"].format(game_state.inning)
        self.inning_label.config(text=inning_str)
        
        # 投手と打者の情報
        if self.pitcher_label and self.batter_label:
            pitcher_obj = getattr(game_state.fielding_team, 'current_pitcher', None)
            batter_obj = getattr(game_state.batting_team, 'current_batter', None)

            self.pitcher_label.config(text=self.text["pitcher"].format(pitcher_obj if pitcher_obj else ""))
            self.batter_label.config(text=self.text["batter"].format(batter_obj if batter_obj else ""))
    
    def create_situation_panel(self, parent_frame):
        """試合状況パネルを作成"""
        situation_frame = ttk.LabelFrame(parent_frame, text=self.text["game_situation"])
        situation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.pitcher_label = ttk.Label(situation_frame, text=self.text["pitcher"].format(""))
        self.pitcher_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.batter_label = ttk.Label(situation_frame, text=self.text["batter"].format(""))
        self.batter_label.pack(anchor=tk.W, padx=5, pady=2)
        
        return situation_frame
