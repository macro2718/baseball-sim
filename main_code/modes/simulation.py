from main_code.core.game import GameState
from main_code.core.stats_calculator import StatsCalculator
import sys
import os
from datetime import datetime

# プロジェクト設定を使用
from main_code.config import setup_project_environment
setup_project_environment()

def simulate_games(num_games=10, output_file=None):
    """指定された回数の試合をシミュレーションし、結果をファイルに出力する"""
    # データローダーを直接インポート
    from player_data.data_loader import DataLoader
    
    # デフォルトのファイル名にタイムスタンプを追加
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # modes/ からプロジェクトルートへは3階層上がる
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, "simulation_results")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"simulation_results_{timestamp}.txt")
    
    # 結果を保存するデータ構造を初期化
    results = {
        "games": [],
        "team_stats": {},
        "teams": {},      # チームオブジェクトを保存
        "players": {},    # 選手オブジェクトを保存
        "pitchers": {}    # 投手オブジェクトを保存
    }
    
    # 最初にチームを作成し、全試合で再利用する
    home_team, away_team = DataLoader.create_teams_from_data()
    
    # ホームチーム、アウェイチームそれぞれの選手・投手をループでまとめて保存
    for team in (home_team, away_team):
        results["teams"][team.name] = team
        for player in team.lineup:
            results["players"][player.name] = player
        for pitcher in team.pitchers:
            results["pitchers"][pitcher.name] = pitcher
    
    for game_num in range(1, num_games + 1):
        print(f"シミュレーション中... 試合 {game_num}/{num_games}")
        
        # 試合前にチームと選手の状態をリセット
        reset_team_and_players(home_team, away_team)
        
        # 試合状態を初期化
        game = GameState(home_team, away_team)
        
        # 試合をシミュレーション
        game_result = simulate_single_game(game)
        
        # 結果を保存
        results["games"].append(game_result)
        
        # チームの勝敗統計を更新
        update_statistics(results, game, game_result)
    
    # 結果をファイルに出力
    output_results(results, output_file)
    print(f"シミュレーション完了。結果は {output_file} に保存されました。")
    
    return results

def reset_team_and_players(home_team, away_team):
    """チームと選手の状態をリセットする（投手のスタミナなど）"""
    # 各チームの投手のスタミナをリセット
    for pitcher in home_team.pitchers:
        pitcher.current_stamina = pitcher.stamina
    
    for pitcher in away_team.pitchers:
        pitcher.current_stamina = pitcher.stamina
    
    # 打順をリセット
    home_team.current_batter_index = 0
    away_team.current_batter_index = 0
    
    # 先発投手をリセット（必要に応じて）
    home_team.current_pitcher = home_team.pitchers[0]
    away_team.current_pitcher = away_team.pitchers[0]

def simulate_single_game(game):
    """1試合をシミュレーション実行し、結果を返す"""
    game_over = False
    game_result = {
        "home_team": game.home_team.name,
        "away_team": game.away_team.name,
        "home_score": 0,
        "away_score": 0,
        "innings": 0,
        "events": []
    }
    
    while not game_over:
        # 打席の結果を計算
        batter = game.batting_team.lineup[game.batting_team.current_batter_index]
        pitcher = game.fielding_team.current_pitcher
        
        result = game.calculate_result(batter, pitcher)
        message = game.apply_result(result, batter)
        
        # 注: 打者と投手の統計は apply_result 内で更新されるため、ここでの更新は削除
        
        # 得点情報を抽出
        runs_scored = get_runs_from_message(message)
        
        # イベントを記録（得点情報を追加）
        event = {
            "inning": f"{game.inning} {'Top' if game.is_top_inning else 'Bottom'}",
            "batter": batter.name,
            "pitcher": pitcher.name,
            "result": result,
            "message": message,
            "runs_scored": runs_scored,  # メッセージから得点情報を抽出
            "batting_team": game.batting_team.name,
            "fielding_team": game.fielding_team.name
        }
        game_result["events"].append(event)
        
        # 投手のスタミナを減少
        pitcher.decrease_stamina()
        
        # 次の打者へ
        game.batting_team.next_batter()
        
        # 試合終了チェック - 修正
        # イニング終了時の判定（9回以降、両チームの点数が異なる場合）
        if game.game_ended:
            game_over = True
        
        # サヨナラ勝ち判定（9回裏以降、ホームチームがリードした時点で即終了）
        if game.inning >= 9 and not game.is_top_inning and game.home_score > game.away_score:
            game_over = True
        
        # 最大12イニングまで
        if game.inning > 12:
            game_over = True
    
    # 最終結果を更新
    game_result["home_score"] = game.home_score
    game_result["away_score"] = game.away_score
    game_result["innings"] = game.inning
    
    return game_result

# メッセージから得点情報を抽出する関数を追加
def get_runs_from_message(message):
    """メッセージから得点情報を抽出する"""
    # "得点" または "score" という単語が含まれているかチェック
    if "得点" in message or "score" in message:
        # 数字を探す（1得点、2得点など）
        for i in range(4, 0, -1):  # 最大4点まで想定（グランドスラム）
            if str(i) in message:
                return i
        return 1  # 具体的な数字がなければ1点とみなす
    return 0  # 得点がない場合は0

def update_statistics(results, game, game_result):
    """チームの勝敗統計を更新する"""
    # ホームチームの統計
    home_name = game.home_team.name
    if home_name not in results["team_stats"]:
        results["team_stats"][home_name] = {
            "wins": 0, "losses": 0, "draws": 0, 
            "runs_scored": 0, "runs_allowed": 0
        }
    
    # アウェイチームの統計
    away_name = game.away_team.name
    if away_name not in results["team_stats"]:
        results["team_stats"][away_name] = {
            "wins": 0, "losses": 0, "draws": 0, 
            "runs_scored": 0, "runs_allowed": 0
        }
    
    # 勝敗の記録
    if game.home_score > game.away_score:
        results["team_stats"][home_name]["wins"] += 1
        results["team_stats"][away_name]["losses"] += 1
    elif game.away_score > game.home_score:
        results["team_stats"][home_name]["losses"] += 1
        results["team_stats"][away_name]["wins"] += 1
    else:
        results["team_stats"][home_name]["draws"] += 1
        results["team_stats"][away_name]["draws"] += 1
    
    # 得点と失点の記録
    results["team_stats"][home_name]["runs_scored"] += game.home_score
    results["team_stats"][home_name]["runs_allowed"] += game.away_score
    results["team_stats"][away_name]["runs_scored"] += game.away_score
    results["team_stats"][away_name]["runs_allowed"] += game.home_score

def output_results(results, output_file):
    """シミュレーション結果をファイルに出力する"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("===== シミュレーション結果 =====\n\n")
        
        # チーム成績の基本出力
        f.write("===== チーム成績 =====\n")
        f.write(f"{'チーム名':<15} {'試合':<5} {'勝':<5} {'負':<5} {'分':<5} {'得点':<5} {'失点':<5} {'勝率':<5}\n")
        f.write("-" * 60 + "\n")
        
        for team_name, stats in results["team_stats"].items():
            games_played = stats["wins"] + stats["losses"] + stats["draws"]
            win_pct = stats["wins"] / games_played if games_played > 0 else 0
            
            f.write(f"{team_name:<15} {games_played:<5} {stats['wins']:<5} {stats['losses']:<5} {stats['draws']:<5} "
                   f"{stats['runs_scored']:<5} {stats['runs_allowed']:<5} {win_pct:.3f}\n")

        # チーム打撃成績の出力
        f.write("\n===== チーム打撃成績 =====\n")
        f.write(f"{'チーム名':<15} {'打率':<5} {'OBP':<5} {'SLG':<5} {'OPS':<5} {'1B計':<5} {'2B計':<5} {'3B計':<5} {'HR計':<5} "
               f"{'三振':<5} {'四球':<5}\n")
        f.write("-" * 80 + "\n")
        
        # チーム打撃成績を計算して出力
        for team_name, team in results["teams"].items():
            # チーム打撃成績の集計
            team_pa = 0
            team_ab = 0
            team_hits = 0
            team_bb = 0
            team_hr = 0
            team_so = 0
            team_singles = 0
            team_doubles = 0
            team_triples = 0
            
            # チーム全体の打撃成績を集計
            for player in team.lineup:
                team_pa += player.stats["PA"]
                team_ab += player.stats["AB"]
                team_singles += player.stats["1B"]
                team_doubles += player.stats.get("2B", 0)
                team_triples += player.stats.get("3B", 0)
                team_hr += player.stats["HR"]
                team_hits += player.stats["1B"] + player.stats["2B"] + player.stats["3B"] + player.stats["HR"]
                team_bb += player.stats["BB"]
                team_so += player.stats["SO"]
                
            # チーム打率計算
            team_avg = team_hits / team_ab if team_ab > 0 else 0
            
            # チーム出塁率計算（PA未集計のため、OBP = (H + BB) / (AB + BB) と等価の簡略式を使用）
            team_obp = StatsCalculator.calculate_obp(team_hits, team_bb, team_ab)
            
            # チーム長打率計算（修正版）
            singles = team_hits - team_doubles - team_triples - team_hr
            total_bases = singles + (team_doubles * 2) + (team_triples * 3) + (team_hr * 4)
            team_slg = total_bases / team_ab if team_ab > 0 else 0
            
            # チームOPS計算
            team_ops = team_obp + team_slg
            
            # チーム打撃成績の出力
            f.write(f"{team_name:<15} {team_avg:.3f} {team_obp:.3f} {team_slg:.3f} {team_ops:.3f} "
                   f"{team_singles:<5} {team_doubles:<5} {team_triples:<5} {team_hr:<5} "
                   f"{team_so:<5} {team_bb:<5}\n")
        
        # チーム投手成績の出力
        f.write("\n===== チーム投手成績 =====\n")
        f.write(f"{'チーム名':<15} {'防御率':<6} {'WHIP':<6} {'K/9':<6} {'BB/9':<6}\n")
        f.write("-" * 60 + "\n")
        
        # チーム投手成績を計算して出力
        for team_name, team in results["teams"].items():
            # チーム投手成績の集計
            team_ip = 0
            team_h = 0
            team_bb_allowed = 0
            team_so_pitched = 0
            team_er = 0
            
            # チーム全体の投手成績を集計
            for pitcher in team.pitchers:
                team_ip += pitcher.pitching_stats["IP"]
                team_h += pitcher.pitching_stats["H"]
                team_bb_allowed += pitcher.pitching_stats["BB"]
                team_so_pitched += pitcher.pitching_stats["SO"]
                team_er += pitcher.pitching_stats["ER"]
            
            # チーム防御率計算
            team_era = (team_er * 9) / team_ip if team_ip > 0 else 0
            
            # チームWHIP計算
            team_whip = (team_h + team_bb_allowed) / team_ip if team_ip > 0 else 0
            
            # チームK/9計算
            team_k_per_9 = (team_so_pitched * 9) / team_ip if team_ip > 0 else 0
            
            # チームBB/9計算
            team_bb_per_9 = (team_bb_allowed * 9) / team_ip if team_ip > 0 else 0
            
            # チーム投手成績の出力
            f.write(f"{team_name:<15} {team_era:<6.2f} {team_whip:<6.2f} {team_k_per_9:<6.2f} {team_bb_per_9:<6.2f}\n")

        # チームごとの選手成績の出力
        f.write("\n===== チーム詳細成績 =====\n")
        f.write(f"{'チーム名':<15} {'打率':<4} {'OBP':<5} {'SLG':<4} {'OPS':<4} {'1B計':<4} {'2B計':<4} {'3B計':<5} {'HR計':<4} "
               f"{'三振':<3} {'四球':<3} {'防御率':<3} {'WHIP':<3} {'K/9':<3} {'BB/9':<5}\n")
        f.write("-" * 120 + "\n")
        
        # チーム詳細成績を計算して出力
        for team_name, team in results["teams"].items():
            # チーム打撃成績の集計
            team_pa = 0
            team_ab = 0
            team_hits = 0
            team_bb = 0
            team_hr = 0
            team_so = 0
            team_singles = 0
            team_doubles = 0
            team_triples = 0
            
            # チーム全体の打撃成績を集計
            for player in team.lineup:
                team_pa += player.stats["PA"]
                team_ab += player.stats["AB"]
                team_singles += player.stats["1B"]
                team_doubles += player.stats.get("2B", 0)
                team_triples += player.stats.get("3B", 0)
                team_hr += player.stats["HR"]
                team_hits += player.stats["1B"] + player.stats["2B"] + player.stats["3B"] + player.stats["HR"]
                team_bb += player.stats["BB"]
                team_so += player.stats["SO"]
                
            # チーム打率計算
            team_avg = team_hits / team_ab if team_ab > 0 else 0
            
            # チーム出塁率計算（簡略式）
            team_obp = StatsCalculator.calculate_obp(team_hits, team_bb, team_ab)
            
            # チーム長打率計算（修正版）
            singles = team_hits - team_doubles - team_triples - team_hr
            total_bases = singles + (team_doubles * 2) + (team_triples * 3) + (team_hr * 4)
            team_slg = total_bases / team_ab if team_ab > 0 else 0
            
            # チームOPS計算
            team_ops = team_obp + team_slg
            
            # チーム投手成績の集計
            team_ip = 0
            team_h = 0
            team_bb_allowed = 0
            team_so_pitched = 0
            team_er = 0
            
            # チーム全体の投手成績を集計
            for pitcher in team.pitchers:
                team_ip += pitcher.pitching_stats["IP"]
                team_h += pitcher.pitching_stats["H"]
                team_bb_allowed += pitcher.pitching_stats["BB"]
                team_so_pitched += pitcher.pitching_stats["SO"]
                team_er += pitcher.pitching_stats["ER"]
            
            # チーム防御率計算
            team_era = (team_er * 9) / team_ip if team_ip > 0 else 0
            
            # チームWHIP計算
            team_whip = (team_h + team_bb_allowed) / team_ip if team_ip > 0 else 0
            
            # チームK/9計算
            team_k_per_9 = (team_so_pitched * 9) / team_ip if team_ip > 0 else 0
            
            # チームBB/9計算
            team_bb_per_9 = (team_bb_allowed * 9) / team_ip if team_ip > 0 else 0
            
            # チーム詳細成績の出力（ヒット種類ごとの合計を追加）
            f.write(f"{team_name:<17} {team_avg:.3f} {team_obp:.3f} {team_slg:.3f} {team_ops:.3f} "
                   f"{team_singles:<5} {team_doubles:<5} {team_triples:<5} {team_hr:<5} "
                   f"{team_so:<5} {team_bb:<5} "
                   f"{team_era:.2f} {team_whip:.2f} {team_k_per_9:.2f} {team_bb_per_9:.2f}\n")
        
        # チームごとの選手成績の出力
        for team_name, team in results["teams"].items():
            f.write(f"\n===== {team_name} 選手成績 =====\n")
            
            # 打者成績
            f.write("\n【打者成績】\n")
            f.write(f"{'選手名':<18} {'PA':<4} {'AB':<4} {'1B':<4} {'2B':<4} {'3B':<4} {'HR':<4} {'RBI':<4} {'BB':<4} {'K':<4} "
                  f"{'AVG':<6} {'OBP':<5} {'SLG':<5} {'OPS':<5} {'K%':<5} {'BB%':<5}\n")
            f.write("-" * 130 + "\n")
            
            # チーム選手の打撃成績を表示
            for player in team.lineup:
                # 選手の成績を取得
                pa = player.stats["PA"]
                ab = player.stats["AB"]
                singles = player.stats["1B"]
                doubles = player.stats.get("2B", 0)
                triples = player.stats.get("3B", 0)
                hr = player.stats["HR"]
                rbi = player.stats["RBI"]
                bb = player.stats["BB"]
                so = player.stats["SO"]
                
                # 指標を計算
                avg = player.get_avg()
                obp = player.get_obp()
                slg = player.get_slg()
                ops = player.get_ops()
                
                # K%とBB%を計算
                k_pct = (so / pa * 100) if pa > 0 else 0
                bb_pct = (bb / pa * 100) if pa > 0 else 0
                
                # 打者成績を出力
                f.write(f"{player.name:<20} {pa:<4} {ab:<4} {singles:<4} {doubles:<4} {triples:<4} "
                      f"{hr:<4} {rbi:<4} {bb:<4} {so:<4} "
                      f"{avg:.3f} {obp:.3f} {slg:.3f} {ops:.3f} {k_pct:.1f} {bb_pct:.1f}\n")
            
            # 投手成績
            f.write("\n【投手成績】\n")
            f.write(f"{'選手名':<15} {'IP':<5} {'H':<4} {'BB':<4} {'K':<4} {'R':<4} {'ER':<4} {'HR':<4} "
                  f"{'ERA':<5} {'WHIP':<5} {'K/9':<5} {'BB/9':<5} {'HR/9':<5}\n")
            f.write("-" * 120 + "\n")
            
            # チーム選手の投手成績を表示
            for pitcher in team.pitchers:
                # 投手の成績を取得
                ip = pitcher.pitching_stats["IP"]
                h = pitcher.pitching_stats["H"]
                bb = pitcher.pitching_stats["BB"]
                so = pitcher.pitching_stats["SO"]
                r = pitcher.pitching_stats["R"]
                er = pitcher.pitching_stats["ER"]
                hr = pitcher.pitching_stats["HR"]
                
                # 指標を計算
                era = pitcher.get_era()
                whip = pitcher.get_whip()
                k_per_9 = pitcher.get_k_per_9()
                bb_per_9 = pitcher.get_bb_per_9()
                
                # HR/9を計算
                hr_per_9 = (hr * 9) / ip if ip > 0 else 0
                
                # 投手成績を出力
                f.write(f"{pitcher.name:<15} {ip:<5.1f} {h:<4} {bb:<4} {so:<4} {r:<4} {er:<4} {hr:<4} "
                      f"{era:<5.2f} {whip:<5.2f} {k_per_9:<5.2f} {bb_per_9:<5.2f} {hr_per_9:<5.2f}\n")
        
        # 全選手の成績
        f.write("\n===== 全選手成績 =====\n")
        
        # 打者成績の出力
        f.write("\n===== 打者成績 =====\n")
        f.write(f"{'選手名':<18} {'チーム':<10} {'PA':<4} {'AB':<4} {'1B':<4} {'2B':<4} {'3B':<4} {'HR':<4} {'RBI':<4} {'BB':<4} {'K':<4} "
              f"{'AVG':<6} {'OBP':<5} {'SLG':<5} {'OPS':<5} {'K%':<5} {'BB%':<5}\n")
        f.write("-" * 140 + "\n")
        
        # チームに所属する選手をキーとして保存
        player_team = {}
        for team_name, team in results["teams"].items():
            for player in team.lineup:
                player_team[player.name] = team_name
        
        # すべての打者の成績を出力
        for name, player in results["players"].items():
            # 選手の成績を取得
            pa = player.stats["PA"]
            
            # 打席がゼロなら表示しない
            if pa == 0:
                continue
                
            ab = player.stats["AB"]
            singles = player.stats["1B"]
            doubles = player.stats.get("2B", 0)
            triples = player.stats.get("3B", 0)
            hr = player.stats["HR"]
            rbi = player.stats["RBI"]
            bb = player.stats["BB"]
            so = player.stats["SO"]
            
            # 指標を計算
            avg = player.get_avg()
            obp = player.get_obp()
            slg = player.get_slg()
            ops = player.get_ops()
            
            # K%とBB%を計算
            k_pct = (so / pa * 100) if pa > 0 else 0
            bb_pct = (bb / pa * 100) if pa > 0 else 0
            
            team_name = player_team.get(name, "不明")
            
            # 打者成績を出力
            f.write(f"{name:<20} {team_name:<10} {pa:<4} {ab:<4} {singles:<4} {doubles:<4} {triples:<4} "
                  f"{hr:<4} {rbi:<4} {bb:<4} {so:<4} "
                  f"{avg:.3f} {obp:.3f} {slg:.3f} {ops:.3f} {k_pct:.1f} {bb_pct:.1f}\n")
        
        # 投手成績の出力
        f.write("\n===== 投手成績 =====\n")
        f.write(f"{'選手名':<15} {'チーム':<10} {'IP':<5} {'H':<4} {'BB':<4} {'K':<4} {'R':<4} {'ER':<4} {'HR':<4} "
              f"{'ERA':<5} {'WHIP':<5} {'K/9':<5} {'BB/9':<5} {'HR/9':<5}\n")
        f.write("-" * 120 + "\n")
        
        # チームに所属する投手をキーとして保存
        pitcher_team = {}
        for team_name, team in results["teams"].items():
            for pitcher in team.pitchers:
                pitcher_team[pitcher.name] = team_name
        
        # すべての投手の成績を出力
        for name, pitcher in results["pitchers"].items():
            # 投手の成績を取得
            ip = pitcher.pitching_stats["IP"]
            
            # 投球イニングがゼロなら表示しない
            if ip == 0:
                continue
                
            h = pitcher.pitching_stats["H"]
            bb = pitcher.pitching_stats["BB"]
            so = pitcher.pitching_stats["SO"]
            r = pitcher.pitching_stats["R"]
            er = pitcher.pitching_stats["ER"]
            hr = pitcher.pitching_stats["HR"]
            
            # 指標を計算
            era = pitcher.get_era()
            whip = pitcher.get_whip()
            k_per_9 = pitcher.get_k_per_9()
            bb_per_9 = pitcher.get_bb_per_9()
            hr_per_9 = pitcher.get_hr_per_9()
            
            team_name = pitcher_team.get(name, "不明")
            
            # 投手成績を出力
            f.write(f"{name:<15} {team_name:<10} {ip:<5.1f} {h:<4} {bb:<4} {so:<4} {r:<4} {er:<4} {hr:<4} "
                  f"{era:<5.2f} {whip:<5.2f} {k_per_9:<5.2f} {bb_per_9:<5.2f} {hr_per_9:<5.2f}\n")
        
        # 各試合の結果サマリー
        f.write("\n===== 試合結果サマリー =====\n")
        for i, game in enumerate(results["games"], 1):
            f.write(f"試合 {i}: {game['away_team']} {game['away_score']} - {game['home_score']} {game['home_team']} "
                  f"({game['innings']}回)\n")
