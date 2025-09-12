"""
Statistics and player stats display functionality for the Baseball GUI
"""
import tkinter as tk
from tkinter import ttk
import sys
import os

# プロジェクト設定を使用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_code.config import setup_project_environment
setup_project_environment()

from main_code.core.stats_calculator import StatsCalculator

class StatsManager:
    def __init__(self, root, text):
        self.root = root
        self.text = text
        self.game_state = None
    
    def set_game_state(self, game_state):
        """ゲーム状態を設定"""
        self.game_state = game_state
    
    def show_player_stats(self):
        """選手成績ウィンドウを表示"""
        if not self.game_state:
            return
        
        stats_window = tk.Toplevel(self.root)
        stats_window.title(self.text["stats"])
        stats_window.geometry("1105x780")  # 850x600 * 1.3
        
        # チーム選択用ノートブック
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # アウェイチームタブ
        away_tab = ttk.Frame(notebook)
        notebook.add(away_tab, text=f"{self.game_state.away_team.name} ({self.text['away']})")
        
        # ホームチームタブ
        home_tab = ttk.Frame(notebook)
        notebook.add(home_tab, text=f"{self.game_state.home_team.name} ({self.text['home']})")
        
        # 各タブに打撃・投球成績を表示
        self._create_team_stats_tab(away_tab, self.game_state.away_team)
        self._create_team_stats_tab(home_tab, self.game_state.home_team)
        
        # 閉じるボタン
        ttk.Button(stats_window, text=self.text["close"], command=stats_window.destroy).pack(pady=10)
    
    def _create_team_stats_tab(self, parent, team):
        """チームの選手成績タブを作成"""
        # 打撃・投球切り替え用ノートブック
        stats_notebook = ttk.Notebook(parent)
        stats_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 打撃成績タブ
        batting_tab = ttk.Frame(stats_notebook)
        stats_notebook.add(batting_tab, text=self.text["batting_stats"])
        
        # 投球成績タブ
        pitching_tab = ttk.Frame(stats_notebook)
        stats_notebook.add(pitching_tab, text=self.text["pitching_stats"])
        
        # 打撃成績テーブル
        self._create_batting_stats_table(batting_tab, team)
        
        # 投球成績テーブル
        self._create_pitching_stats_table(pitching_tab, team)
    
    def _create_batting_stats_table(self, parent, team):
        """打撃成績テーブルを作成"""
        # ヘッダー行
        columns = [self.text["player"], self.text["ab"], self.text["single"], 
                  self.text["double"], self.text["triple"], self.text["hr"], 
                  self.text["runs"], self.text["rbi"], self.text["bb"], 
                  self.text["k"], self.text["avg"]]
        
        # 統一されたテーブル作成メソッドを使用
        tree = self._create_stats_table(parent, columns)
        
        # 選手データを行として追加
        self._fill_batting_stats(tree, team)
    
    def _fill_batting_stats(self, tree, team):
        """打撃成績テーブルにデータを入力"""
        # 先発選手
        for player in team.lineup:
            if player.position != "P":  # 投手以外（DHは含む）
                self._add_batter_stats_row(tree, player)
        
        # ベンチ選手
        for player in team.bench:
            if player.position != "P":  # 投手以外
                self._add_batter_stats_row(tree, player)
    
    def _add_batter_stats_row(self, tree, player):
        """打者の成績行を追加"""
        # 打率計算（統一されたフォーマット関数を使用）
        avg = StatsCalculator.format_average(player.get_avg())
        
        # 各種統計データ
        values = [
            player.name,  # 選手名
            player.stats.get("AB", 0),  # 打数
            player.stats.get("1B", 0),  # シングルヒット
            player.stats.get("2B", 0),  # 二塁打
            player.stats.get("3B", 0),  # 三塁打
            player.stats.get("HR", 0),  # 本塁打
            player.stats.get("R", 0),   # 得点
            player.stats.get("RBI", 0), # 打点
            player.stats.get("BB", 0),  # 四球
            player.stats.get("K", 0),   # 三振
            avg                         # 打率
        ]
        
        # 行として追加
        tree.insert("", tk.END, values=values)
    
    def _create_pitching_stats_table(self, parent, team):
        """投球成績テーブルを作成"""
        # ヘッダー行
        columns = [self.text["player"], self.text["ip"], self.text["h"], 
                  self.text["runs"], self.text["er"], self.text["bb"], 
                  self.text["k"], self.text["era"], self.text["whip"]]
        
        # 統一されたテーブル作成メソッドを使用
        tree = self._create_stats_table(parent, columns)
        
        # 選手データを行として追加
        self._fill_pitching_stats(tree, team)
    
    def _fill_pitching_stats(self, tree, team):
        """投球成績テーブルにデータを入力"""
        # 先発投手
        for player in team.lineup:
            if player.position == "P":  # 投手のみ
                self._add_pitcher_stats_row(tree, player)
        
        # リリーフ投手
        for player in team.pitchers:
            if player not in team.lineup:  # 先発にいない投手
                self._add_pitcher_stats_row(tree, player)
        
        # ベンチの投手
        for player in team.bench:
            if player.position == "P" and player not in team.pitchers:  # 投手だがピッチャーリストにいない
                self._add_pitcher_stats_row(tree, player)
    
    def _add_pitcher_stats_row(self, tree, player):
        """投手の成績行を追加（統一処理）"""
        # 投手統計の取得と表示処理を統一
        pitching_data = self._get_pitcher_stats(player)
        
        # 各種統計データの表示値を作成
        values = [
            player.name,                           # 選手名
            pitching_data['ip_display'],           # 投球イニング
            pitching_data['hits'],                 # 被安打
            pitching_data['runs'],                 # 失点
            pitching_data['earned_runs'],          # 自責点
            pitching_data['walks'],                # 四球
            pitching_data['strikeouts'],           # 奪三振
            pitching_data['era'],                  # 防御率
            pitching_data['whip']                  # WHIP
        ]
        
        tree.insert("", tk.END, values=values)
    
    def _get_pitcher_stats(self, player):
        """投手の統計データを統一的に取得"""
        # 投手データの取得方法を統一
        if hasattr(player, 'pitching_stats') and player.pitching_stats:
            stats = player.pitching_stats
            ip = stats.get("IP", 0)
            strikeouts = stats.get("SO", stats.get("K", 0))  # SOとKの両方をチェック
        else:
            stats = player.stats
            ip = stats.get("IP", 0)
            strikeouts = stats.get("K", 0)
        
        # 共通の計算処理
        era = self._calculate_era_display(player, ip)
        whip = self._calculate_whip_display(player, ip)
        ip_display = StatsCalculator.format_inning_display(ip)
        
        return {
            'ip_display': ip_display,
            'hits': stats.get("H", 0),
            'runs': stats.get("R", 0),
            'earned_runs': stats.get("ER", 0),
            'walks': stats.get("BB", 0),
            'strikeouts': strikeouts,
            'era': era,
            'whip': whip
        }
    
    def _calculate_era_display(self, player, ip):
        """防御率の表示値を計算"""
        if ip > 0 and hasattr(player, 'get_era'):
            return StatsCalculator.format_average(player.get_era(), 2)
        return "-.--"
    
    def _calculate_whip_display(self, player, ip):
        """WHIPの表示値を計算"""
        if ip > 0 and hasattr(player, 'get_whip'):
            return StatsCalculator.format_average(player.get_whip(), 2)
        return "-.--"
    
    def _create_stats_table(self, parent, columns):
        """統一されたTreeviewテーブルを作成"""
        # TreeviewでテーブルUIを作成
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)
        for i, col in enumerate(columns):
            tree.heading(i, text=col)
            # 列幅を調整
            width = 60 if i == 0 else 40
            tree.column(i, width=width, anchor="center" if i > 0 else "w")
        
        # スクロールバー
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # 配置
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        return tree
