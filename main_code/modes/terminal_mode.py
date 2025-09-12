from main_code.core.game import GameState
from main_code.core.substitution_manager import SubstitutionManager  # 追加
from main_code.core.stats_calculator import StatsCalculator

def display_game_info(game_state):
    """Display current game state"""
    inning_str = f"{game_state.inning} {'Top' if game_state.is_top_inning else 'Bottom'}"
    print(f"\n===== {inning_str} =====")
    print(f"Score: {game_state.away_team.name} {game_state.away_score} - {game_state.home_score} {game_state.home_team.name}")
    
    # イニングスコアの表示
    print("\nInning Score:")
    print(f"{game_state.away_team.name}: ", end="")
    for score in game_state.inning_scores[0]:
        print(f"{score} ", end="")
    print(f"Total: {game_state.away_score}")
    
    print(f"{game_state.home_team.name}: ", end="")
    for i, score in enumerate(game_state.inning_scores[1]):
        if i < len(game_state.inning_scores[0]) - 1 or not game_state.is_top_inning:
            print(f"{score} ", end="")
    if not game_state.is_top_inning:
        print(f"{game_state.inning_scores[1][-1]} ", end="")
    print(f"Total: {game_state.home_score}")
    
    # ランナーの状況
    print(f"\nOuts: ", end="")
    for i in range(3):
        if i < game_state.outs:
            print("●", end="")
        else:
            print("○", end="")
    print()
    
    # ランナー表示
    runners = []
    if game_state.bases[0] is not None:
        runners.append(f"1B({game_state.bases[0].name})")
    if game_state.bases[1] is not None:
        runners.append(f"2B({game_state.bases[1].name})")
    if game_state.bases[2] is not None:
        runners.append(f"3B({game_state.bases[2].name})")
    
    if runners:
        print(f"Runners on: {', '.join(runners)}")
    else:
        print("No runners")
        
    # 投手と打者の情報
    pitcher = game_state.fielding_team.current_pitcher
    batter = game_state.batting_team.next_batter()
    game_state.batting_team.current_batter_index = (game_state.batting_team.current_batter_index - 1) % 9
    
    print(f"\nPitcher: {pitcher} - Stamina: {pitcher.stamina}% (ERA: {pitcher.get_era():.2f}, WHIP: {pitcher.get_whip():.2f})")
    print(f"Batter: {batter} (AVG: {batter.get_avg():.3f}, OPS: {batter.get_ops():.3f})")

def display_defensive_formation(game_state):
    """守備陣形を表示"""
    print(f"\n===== {game_state.fielding_team.name} Defensive Formation =====")
    
    # ポジション順で表示
    position_order = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
    
    for pos in position_order:
        for player in game_state.fielding_team.lineup:
            if player.position == pos:
                fielding_rating = f"Fielding: {player.fielding_skill}"
                print(f"{pos:>3}: {player.name:<15} ({fielding_rating})")
                break
        else:
            print(f"{pos:>3}: Empty")  # ポジションが埋まっていない場合

def manage_team(game_state, is_offense):
    """Manage team"""
    team = game_state.batting_team if is_offense else game_state.fielding_team
    substitution_manager = SubstitutionManager(team)
    
    while True:
        print("\n===== Management Menu =====")
        if is_offense:
            print("1. 代打")
            print("2. 出場可能選手表示")
        else:
            print("1. 投手交代")
            print("2. 守備交代")
            print("3. 守備位置交代")
            print("4. 守備陣形表示")
        print("0. 管理終了")
        
        choice = input("Select an option: ")
        
        if choice == "0":
            break
            
        elif choice == "1" and is_offense:
            # 代打（統一化された処理を使用）
            _handle_pinch_hit(substitution_manager)
                
        elif choice == "2" and is_offense:
            # 出場可能選手表示
            _display_available_players(substitution_manager)
                
        elif choice == "1" and not is_offense:
            # 投手交代（統一化された処理を使用）
            _handle_pitcher_change(substitution_manager)
                
        elif choice == "2" and not is_offense:
            # 守備交代（統一化された処理を使用）
            _handle_defensive_substitution(substitution_manager)
                
        elif choice == "3" and not is_offense:
            # 守備位置交代（統一化された処理を使用）
            _handle_position_switch(substitution_manager)
                
        elif choice == "4" and not is_offense:
            # 守備陣形表示
            display_defensive_formation(game_state)

def _handle_pinch_hit(substitution_manager):
    """代打処理（統一化）"""
    available_bench = substitution_manager.get_available_bench_players()
    if not available_bench:
        print("出場可能なベンチ選手がいません。")
        return
        
    print("\n===== 出場可能ベンチ選手 =====")
    formatted_bench = substitution_manager.get_formatted_bench()
    for info in formatted_bench:
        print(info)
        
    try:
        bench_choice = int(input("代打選手を選択 (0でキャンセル): ")) - 1
        if bench_choice == -1:
            return
            
        if 0 <= bench_choice < len(available_bench):
            lineup_index = substitution_manager.team.current_batter_index
            success, message = substitution_manager.execute_pinch_hit(bench_choice, lineup_index)
            print(f"結果: {message}")
        else:
            print("無効な選択です。")
    except ValueError:
        print("無効な入力です。")

def _display_available_players(substitution_manager):
    """出場可能選手表示（統一化）"""
    available_bench = substitution_manager.get_available_bench_players()
    if not available_bench:
        print("出場可能なベンチ選手がいません。")
        return
    
    print("\n===== 出場可能ベンチ選手 =====")
    formatted_bench = substitution_manager.get_formatted_bench()
    for info in formatted_bench:
        print(info)

def _handle_pitcher_change(substitution_manager):
    """投手交代処理（統一化）"""
    available_pitchers = substitution_manager.get_available_pitchers()
    if not available_pitchers:
        print("出場可能な投手がいません。")
        return
        
    print(f"\n現在の投手: {substitution_manager.team.current_pitcher.name if substitution_manager.team.current_pitcher else 'なし'}")
    print("\n===== 出場可能投手 =====")
    formatted_pitchers = substitution_manager.get_formatted_pitchers()
    for info in formatted_pitchers:
        print(info)
        
    try:
        pitcher_choice = int(input("投手を選択 (0でキャンセル): ")) - 1
        if pitcher_choice == -1:
            return
            
        if 0 <= pitcher_choice < len(available_pitchers):
            success, message = substitution_manager.execute_pitcher_change(pitcher_choice)
            print(f"結果: {message}")
        else:
            print("無効な選択です。")
    except ValueError:
        print("無効な入力です。")

def _handle_defensive_substitution(substitution_manager):
    """守備交代処理（統一化）"""
    available_bench = substitution_manager.get_available_bench_players()
    if not available_bench:
        print("出場可能なベンチ選手がいません。")
        return
        
    print("\n===== 出場可能ベンチ選手 =====")
    formatted_bench = substitution_manager.get_formatted_bench()
    for info in formatted_bench:
        print(info)
        
    print("\n===== 現在のラインナップ =====")
    formatted_lineup = substitution_manager.get_formatted_lineup()
    for info in formatted_lineup:
        print(info)
        
    try:
        bench_choice = int(input("代替選手を選択 (0でキャンセル): ")) - 1
        if bench_choice == -1:
            return
            
        lineup_choice = int(input("交代される選手を選択 (0でキャンセル): ")) - 1
        if lineup_choice == -1:
            return
            
        if 0 <= bench_choice < len(available_bench) and 0 <= lineup_choice < len(substitution_manager.team.lineup):
            success, message = substitution_manager.execute_defensive_substitution(bench_choice, lineup_choice)
            print(f"結果: {message}")
        else:
            print("無効な選択です。")
    except ValueError:
        print("無効な入力です。")

def _handle_position_switch(substitution_manager):
    """守備位置交代処理（統一化）"""
    print("\n===== 現在のフィールダー =====")
    eligible_players = []
    for i, player in enumerate(substitution_manager.team.lineup):
        if player.current_position != "DH":  # DHは交代対象外
            print(f"{len(eligible_players)+1}. {player.name} ({player.current_position})")
            eligible_players.append(i)
        
    if len(eligible_players) < 2:
        print("ポジション交代可能な選手が不足しています。")
        return
        
    try:
        player1_choice = int(input("1人目を選択 (0でキャンセル): ")) - 1
        if player1_choice == -1:
            return
        
        if not (0 <= player1_choice < len(eligible_players)):
            print("無効な選択です。")
            return
            
        player1_index = eligible_players[player1_choice]
        
        print("\n===== 2人目を選択 =====")
        for i, player_index in enumerate(eligible_players):
            if i != player1_choice:
                player = substitution_manager.team.lineup[player_index]
                print(f"{i+1}. {player.name} ({player.current_position})")
        
        player2_choice = int(input("2人目を選択 (0でキャンセル): ")) - 1
        if player2_choice == -1:
            return
            
        if not (0 <= player2_choice < len(eligible_players)) or player2_choice == player1_choice:
            print("無効な選択です。")
            return
            
        player2_index = eligible_players[player2_choice]
        
        success, message = substitution_manager.execute_position_switch(player1_index, player2_index)
        print(f"結果: {message}")
        
    except ValueError:
        print("無効な入力です。")

def play_game_terminal(home_team, away_team):
    """Terminal-based game loop"""
    game = GameState(home_team, away_team)
    
    game_over = False
    
    while not game_over:
        display_game_info(game)
        display_defensive_formation(game)  # 守備陣形の表示を追加
        
        # Fielding team management
        print(f"\n{game.fielding_team.name} management")
        manage_team(game, False)
        
        # Batting team management
        print(f"\n{game.batting_team.name} management")
        manage_team(game, True)
        
        # At-bat selection (normal batting or bunt)
        batter = game.batting_team.lineup[game.batting_team.current_batter_index]
        pitcher = game.fielding_team.current_pitcher
        
        print(f"\n===== {batter.name} vs {pitcher.name} =====")
        
        # Show current situation
        runners = []
        if game.bases[0] is not None:
            runners.append("1B")
        if game.bases[1] is not None:
            runners.append("2B")
        if game.bases[2] is not None:
            runners.append("3B")
        
        if runners:
            runners_text = f"Runners: {', '.join(runners)}"
        else:
            runners_text = "No runners"
        
        print(f"Situation: {game.inning}{'T' if game.is_top_inning else 'B'}, {game.outs} outs, {runners_text}")
        
        # 守備位置エラーチェック - ゲームアクションが許可されているかの確認
        allowed, error_msg = game.is_game_action_allowed()
        if not allowed:
            print(f"\n❌ {error_msg}")
            print("Please fix the defensive position issues before continuing.")
            print("\nReturning to team management...")
            return False
        
        # Check if bunt is possible
        can_bunt = game.can_bunt()
        
        print("\nSelect batting action:")
        print("1: Normal batting")
        
        if can_bunt:
            # Show bunt success probability
            from main_code.core.game_utils import BuntCalculator
            try:
                bunt_success_rate = BuntCalculator.calculate_bunt_success_probability(batter, pitcher)
                print(f"2: Bunt (Success rate: {bunt_success_rate:.1%})")
            except:
                print("2: Bunt")
        else:
            if not any(game.bases):
                print("2: Bunt (Unavailable - No runners)")
            elif game.outs >= 2:
                print("2: Bunt (Unavailable - 2 outs)")
            else:
                print("2: Bunt (Unavailable)")
        
        # Get player choice
        while True:
            choice = input("Choose action (1-2): ")
            if choice == "1":
                # Normal batting
                result = game.calculate_result(batter, pitcher)
                message = game.apply_result(result, batter)
                print(f"\nResult: {message}")
                break
            elif choice == "2":
                if can_bunt:
                    # Execute bunt
                    result_message = game.execute_bunt(batter, pitcher)
                    print(f"\nBunt Result: {result_message}")
                    break
                else:
                    print("Bunt is not available in this situation. Please choose option 1.")
            else:
                print("Invalid choice. Please enter 1 or 2.")
        
        # Decrease pitcher's stamina
        pitcher.decrease_stamina()
        
        # Move to next batter
        game.batting_team.next_batter()
        
        # Check for game end using game_ended flag
        if game.game_ended:
            game_over = True
        
        # Additional check for extra innings limit
        if game.inning > 12:
            game_over = True
            
        # Check for mercy rule (optional)
        run_difference = abs(game.home_score - game.away_score)
        if (game.inning >= 7 and run_difference >= 10) or (game.inning >= 5 and run_difference >= 15):
            game_over = True
            print("\nMercy rule applied!")
        
        input("\nPress Enter to continue...")
    
    # End of game
    print("\n===== End of Game =====")
    print(f"Final Score: {game.away_team.name} {game.away_score} - {game.home_score} {game.home_team.name}")
    
    # Final inning score display
    print("\nInning Score:")
    print(f"{game.away_team.name}: ", end="")
    for score in game.inning_scores[0]:
        print(f"{score} ", end="")
    print(f"Total: {game.away_score}")
    
    print(f"{game.home_team.name}: ", end="")
    for score in game.inning_scores[1]:
        print(f"{score} ", end="")
    print(f"Total: {game.home_score}")
    
    # Display final result
    if game.home_score > game.away_score:
        print(f"{game.home_team.name} wins!")
    elif game.away_score > game.home_score:
        print(f"{game.away_team.name} wins!")
    else:
        print("Draw!")
    
    # Display game summary with ejected players
    game.home_team.game_summary()
    game.away_team.game_summary()
    
    # Display team stats
    print("\n===== Team Stats =====")
    
    # Display batting stats
    print(f"\n{game.home_team.name} Batting:")
    for player in game.home_team.lineup:
        if player.stats["PA"] > 0:
            total_hits = player.stats["1B"] + player.stats.get("2B", 0) + player.stats.get("3B", 0) + player.stats.get("HR", 0)
            avg_display = StatsCalculator.format_average(player.get_avg())
            print(f"{player.name}: {total_hits}/{player.stats['AB']} ({avg_display}), {player.stats.get('HR', 0)} HR, {player.stats['RBI']} RBI")
    
    print(f"\n{game.away_team.name} Batting:")
    for player in game.away_team.lineup:
        if player.stats["PA"] > 0:
            total_hits = player.stats["1B"] + player.stats.get("2B", 0) + player.stats.get("3B", 0) + player.stats.get("HR", 0)
            avg_display = StatsCalculator.format_average(player.get_avg())
            print(f"{player.name}: {total_hits}/{player.stats['AB']} ({avg_display}), {player.stats.get('HR', 0)} HR, {player.stats['RBI']} RBI")
    
    # Display pitching stats
    print(f"\n{game.home_team.name} Pitching:")
    for pitcher in game.home_team.pitchers:
        if pitcher.pitching_stats["IP"] > 0:
            ip = pitcher.pitching_stats["IP"]
            er = pitcher.pitching_stats["ER"]
            so = pitcher.pitching_stats["SO"]
            bb = pitcher.pitching_stats["BB"]
            print(f"{pitcher.name}: {ip:.1f} IP, {er} ER, {so} K, {bb} BB, ERA: {pitcher.get_era():.2f}")
    
    print(f"\n{game.away_team.name} Pitching:")
    for pitcher in game.away_team.pitchers:
        if pitcher.pitching_stats["IP"] > 0:
            ip = pitcher.pitching_stats["IP"]
            er = pitcher.pitching_stats["ER"]
            so = pitcher.pitching_stats["SO"]
            bb = pitcher.pitching_stats["BB"]
            print(f"{pitcher.name}: {ip:.1f} IP, {er} ER, {so} K, {bb} BB, ERA: {pitcher.get_era():.2f}")
