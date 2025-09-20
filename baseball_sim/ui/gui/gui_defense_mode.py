"""
Defense Change Mode for GUI
守備変更モードの管理クラス - パワプロスタイルの守備変更UI
"""
import tkinter as tk
from tkinter import ttk, messagebox
import copy


class DefenseChangeMode:
    """
    守備変更モードを管理するクラス
    パワプロ風の守備変更UIを提供
    """
    
    def __init__(self, parent, substitution_manager, main_gui=None):
        """
        Args:
            parent: 親ウィンドウ
            substitution_manager: SubstitutionManager インスタンス
            main_gui: メインGUIインスタンス（表示更新用）
        """
        self.parent = parent
        self.substitution_manager = substitution_manager
        self.main_gui = main_gui
        self.team = substitution_manager.team
        
        # 守備変更モードの状態管理
        self.is_mode_active = False
        self.temp_lineup = None  # 作業用ラインナップ
        self.temp_bench = None   # 作業用ベンチ
        self.retired_players = []  # 退場した選手のリスト（再使用不可）
        self.retired_details = []  # 退場選手の詳細（表示用）
        self.changes_made = []   # 変更履歴
        
        # GUI要素
        self.mode_window = None
        self.position_buttons = {}
        self.bench_frame = None
        self.status_label = None
        
    def start_defense_change_mode(self):
        """守備変更モードを開始"""
        if self.is_mode_active:
            messagebox.showinfo("Info", "Defense change mode is already active")
            return
            
        self.is_mode_active = True
        
        # 現在の状態をコピーして作業用として保存
        self.temp_lineup = copy.deepcopy(self.team.lineup)
        self.temp_bench = copy.deepcopy(self.team.bench)
        self.retired_players = []  # 退場選手リストをリセット
        self.retired_details = []  # 退場選手詳細もリセット
        self.changes_made = []
        
        # 各選手にcurrent_positionが設定されていることを確認
        for player in self.temp_lineup:
            if not hasattr(player, 'current_position') or player.current_position is None:
                player.current_position = getattr(player, 'position', None)
        
        self._create_defense_mode_window()
        
    def end_defense_change_mode(self, apply_changes=True):
        """守備変更モードを終了
        
        Args:
            apply_changes (bool): 変更を適用するかどうか
        """
        if not self.is_mode_active:
            return
            
        if apply_changes:
            # 守備位置の妥当性をチェック
            validation_issues = self._validate_defense_positions()
            
            # エラーのみをチェック（警告は無視）
            errors = [issue for issue in validation_issues if not issue.startswith("⚠")]
            
            if errors:
                error_message = "Cannot apply changes due to errors:\n" + "\n".join(errors)
                messagebox.showerror("Invalid Formation", error_message)
                return False
            
            # 警告がある場合は確認
            warnings = [issue for issue in validation_issues if issue.startswith("⚠")]
            if warnings:
                warning_message = "The following warnings exist:\n" + "\n".join(warnings) + "\n\nDo you want to continue anyway?"
                if not messagebox.askyesno("Warnings Found", warning_message):
                    return False
            
            # 変更を適用
            self._apply_changes()
            messagebox.showinfo("Complete", "Defense changes applied successfully!")
        else:
            messagebox.showinfo("Cancelled", "Defense changes cancelled")
            
        self.is_mode_active = False
        if self.mode_window:
            self.mode_window.destroy()
            self.mode_window = None
            
        return True
        
    def _create_defense_mode_window(self):
        """守備変更モードのウィンドウを作成"""
        self.mode_window = tk.Toplevel(self.parent)
        self.mode_window.title("Defense Change Mode")
        self.mode_window.geometry("1100x820")
        self.mode_window.protocol("WM_DELETE_WINDOW", lambda: self.end_defense_change_mode(False))
        
        # メインフレーム
        main_frame = ttk.Frame(self.mode_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # タイトルとステータス
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text="Defense Change Mode", 
                 font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(title_frame, text="Ready", 
                                     font=("Arial", 10), foreground="green")
        self.status_label.pack(side=tk.RIGHT)
        
        # メインコンテンツフレーム
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左側: 守備位置表示
        left_frame = ttk.LabelFrame(content_frame, text="Field Positions")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self._create_field_display(left_frame)
        
        # 右側: ベンチ選手とコントロール
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # ベンチ選手フレーム
        bench_label_frame = ttk.LabelFrame(right_frame, text="Bench Players")
        bench_label_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self._create_bench_display(bench_label_frame)
        
        # 退場選手フレーム
        retired_label_frame = ttk.LabelFrame(right_frame, text="Retired Players")
        retired_label_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_retired_display(retired_label_frame)
        
        # コントロールボタン
        control_frame = ttk.LabelFrame(right_frame, text="Controls")
        control_frame.pack(fill=tk.X)
        
        self._create_control_buttons(control_frame)
        
        # 変更履歴
        history_frame = ttk.LabelFrame(main_frame, text="Changes Made")
        history_frame.pack(fill=tk.X, pady=(10, 0))
        
        self._create_history_display(history_frame)
        
        # 更新
        self._update_display()
        
    def _create_field_display(self, parent):
        """守備位置の表示を作成"""
        # 野球場のレイアウトを模した配置
        field_frame = ttk.Frame(parent)
        field_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # 位置の配置座標（行、列、位置名）- ピッチャーを除外
        # 外野配置は現状を維持しつつ、内野のx軸間隔を圧縮して等脚台形を形成
        positions = [
            (0, 1, "CF", 2),  # 中堅手（中央に寄せて配置）
            (1, 0, "LF", 1),  # 左翼手
            (1, 3, "RF", 1),  # 右翼手
            (2, 2, "2B", 1),  # 二塁手（上辺右の頂点）
            (2, 1, "SS", 1),  # 遊撃手（上辺左の頂点）
            (3, 3, "1B", 1),  # 一塁手（下辺右の頂点）
            (3, 0, "3B", 1),  # 三塁手（下辺左の頂点）
            (4, 1, "C", 2),   # 捕手（ホームプレート付近）
        ]

        # グリッドを設定（行5×列4のフィールド）
        for i in range(5):
            field_frame.grid_rowconfigure(i, weight=1)
        for j in range(4):
            field_frame.grid_columnconfigure(j, weight=1)
            
        # 各ポジションのボタンを作成
        for row, col, position, colspan in positions:
            button_frame = ttk.Frame(field_frame)
            button_frame.grid(row=row, column=col, columnspan=colspan,
                              padx=5, pady=5, sticky="nsew")
            
            # ポジション名ラベル
            from .gui_colors import get_position_color
            color = get_position_color(position)
            pos_label = ttk.Label(button_frame, text=position, 
                                 font=("Arial", 10, "bold"),
                                 foreground=(color or "black"))
            pos_label.pack()
            
            # 選手ボタン
            player_button = ttk.Button(button_frame, text="", width=18,
                                     command=lambda pos=position: self._on_position_click(pos))
            player_button.pack(pady=2, ipady=5)  # 打率がないので高さを少し減らす
            
            self.position_buttons[position] = player_button
            
        # DH（指名打者）は別途右下に配置
        dh_frame = ttk.Frame(field_frame)
        dh_frame.grid(row=4, column=3, padx=5, pady=5, sticky="se")
        
        ttk.Label(dh_frame, text="DH", font=("Arial", 10, "bold")).pack()
        dh_button = ttk.Button(dh_frame, text="", width=18,
                              command=lambda: self._on_position_click("DH"))
        dh_button.pack(pady=2, ipady=5)  # 打率がないので高さを少し減らす
        self.position_buttons["DH"] = dh_button
        
    def _create_bench_display(self, parent):
        """ベンチ選手の表示を作成"""
        # スクロール可能なフレーム
        canvas = tk.Canvas(parent, height=300)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.bench_frame = ttk.Frame(canvas)
        
        self.bench_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.bench_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def _create_retired_display(self, parent):
        """退場選手の表示を作成"""
        self.retired_text = tk.Text(parent, height=4, wrap=tk.WORD, state=tk.DISABLED)
        retired_scrollbar = ttk.Scrollbar(parent, orient="vertical", 
                                        command=self.retired_text.yview)
        self.retired_text.configure(yscrollcommand=retired_scrollbar.set)
        
        self.retired_text.pack(side="left", fill="both", expand=True)
        retired_scrollbar.pack(side="right", fill="y")
        
    def _create_control_buttons(self, parent):
        """コントロールボタンを作成"""
        # 完了ボタン
        ttk.Button(parent, text="Apply Changes", 
                  command=lambda: self.end_defense_change_mode(True)).pack(fill=tk.X, pady=2)
        
        # キャンセルボタン
        ttk.Button(parent, text="Cancel", 
                  command=lambda: self.end_defense_change_mode(False)).pack(fill=tk.X, pady=2)
        
        # リセットボタン
        ttk.Button(parent, text="Reset All", 
                  command=self._reset_changes).pack(fill=tk.X, pady=2)
        
        # バリデーションボタン
        ttk.Button(parent, text="Check Formation", 
                  command=self._check_formation).pack(fill=tk.X, pady=2)
        
    def _create_history_display(self, parent):
        """変更履歴の表示を作成"""
        self.history_text = tk.Text(parent, height=4, wrap=tk.WORD)
        history_scrollbar = ttk.Scrollbar(parent, orient="vertical", 
                                        command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_text.pack(side="left", fill="both", expand=True)
        history_scrollbar.pack(side="right", fill="y")
        
    def _update_display(self):
        """表示を更新"""
        # 守備位置の更新
        for position, button in self.position_buttons.items():
            player = self._find_player_at_position(position)
            if player:
                # 選手名と守備可能位置を表示（現在のポジションを太字で）
                # ピッチャーは考慮しない
                playable_positions = []
                for pos in ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]:
                    if player.can_play_position(pos):
                        if pos == position:
                            playable_positions.append(f"[{pos}]")  # 現在のポジションを括弧で強調
                        else:
                            playable_positions.append(pos)
                
                # 長すぎる場合は省略
                positions_text = ", ".join(playable_positions)
                if len(positions_text) > 20:
                    # 現在のポジションと主要な数個のみ表示
                    main_positions = [pos for pos in playable_positions[:4]]
                    if len(playable_positions) > 4:
                        positions_text = ", ".join(main_positions) + "..."
                    else:
                        positions_text = ", ".join(main_positions)
                
                display_text = f"{player.name}\n{positions_text}"
                button.configure(text=display_text)
            else:
                button.configure(text="Empty")
                
        # ベンチ選手の更新
        self._update_bench_display()
        
        # 退場選手の更新  
        self._update_retired_display()
        
        # 変更履歴の更新
        self._update_history_display()
        
    def _update_bench_display(self):
        """ベンチ選手表示を更新"""
        # 既存のウィジェットを削除
        for widget in self.bench_frame.winfo_children():
            widget.destroy()
            
        if not self.temp_bench:
            ttk.Label(self.bench_frame, text="No bench players").pack(pady=10)
            return
            
        for i, player in enumerate(self.temp_bench):
            player_frame = ttk.Frame(self.bench_frame)
            player_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # 選手情報（ボタンは名前のみ）
            info_text = f"{player.name}"
            player_button = ttk.Button(player_frame, text=info_text, width=20,
                                     command=lambda idx=i: self._on_bench_player_click(idx))
            player_button.pack(side=tk.LEFT, padx=2)
            
            # 守備位置適性表示（DH含む）をトークン毎に色付け
            from .gui_colors import get_position_color
            positions = [pos for pos in ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"] if player.can_play_position(pos)]
            pos_text = tk.Text(player_frame, height=1, wrap=tk.NONE, font=("Arial", 8))
            if positions:
                for idx_p, pos in enumerate(positions):
                    if idx_p > 0:
                        pos_text.insert(tk.END, ", ")
                    color = get_position_color(pos, getattr(player, 'pitcher_type', None) if pos in {"P","SP","RP"} else None)
                    if color:
                        tag = f"pos_{pos}_{color}"
                        pos_text.tag_configure(tag, foreground=color)
                        pos_text.insert(tk.END, pos, tag)
                    else:
                        pos_text.insert(tk.END, pos)
            else:
                pos_text.insert(tk.END, "No positions available")
            pos_text.configure(state=tk.DISABLED)
            pos_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
    def _update_retired_display(self):
        """退場選手表示を更新"""
        self.retired_text.configure(state=tk.NORMAL)
        self.retired_text.delete(1.0, tk.END)
        
        if self.retired_players:
            # 詳細情報を利用して表示（ポジション語は部分着色）
            try:
                from .gui_colors import get_position_color
            except Exception:
                get_position_color = None

            # プレイヤー順に詳細を見つけて出力
            for player in self.retired_players:
                # 対応する詳細を探す
                detail = None
                for d in self.retired_details:
                    if d.get('player') == player:
                        detail = d
                        break

                name = getattr(player, 'name', 'Unknown')
                actual_pos = detail.get('actual_position') if detail else None
                eligible = detail.get('eligible_positions', []) if detail else []

                # 名前
                self.retired_text.insert(tk.END, name)

                # 実際の守備位置
                self.retired_text.insert(tk.END, " (")
                if actual_pos and get_position_color:
                    color = get_position_color(actual_pos)
                    if color:
                        tag = f"ret_act_{actual_pos}_{color}"
                        self.retired_text.tag_configure(tag, foreground=color)
                        self.retired_text.insert(tk.END, actual_pos, tag)
                    else:
                        self.retired_text.insert(tk.END, actual_pos)
                elif actual_pos:
                    self.retired_text.insert(tk.END, actual_pos)
                else:
                    self.retired_text.insert(tk.END, "-")
                self.retired_text.insert(tk.END, ") ")

                # Eligible 表示
                self.retired_text.insert(tk.END, "[Eligible: ")
                for idx, pos in enumerate(eligible):
                    if idx > 0:
                        self.retired_text.insert(tk.END, ", ")
                    if get_position_color:
                        color = get_position_color(pos)
                        if color:
                            tag = f"ret_elig_{pos}_{color}"
                            self.retired_text.tag_configure(tag, foreground=color)
                            self.retired_text.insert(tk.END, pos, tag)
                        else:
                            self.retired_text.insert(tk.END, pos)
                    else:
                        self.retired_text.insert(tk.END, pos)
                self.retired_text.insert(tk.END, "]\n")
        else:
            self.retired_text.insert(tk.END, "No retired players")
        
        self.retired_text.configure(state=tk.DISABLED)
            
    def _update_history_display(self):
        """変更履歴表示を更新"""
        self.history_text.delete(1.0, tk.END)
        if self.changes_made:
            for change in self.changes_made:
                self.history_text.insert(tk.END, change + "\n")
        else:
            self.history_text.insert(tk.END, "No changes made yet")
            
    def _find_player_at_position(self, position):
        """指定されたポジションの選手を検索"""
        if not self.temp_lineup:
            return None
            
        for player in self.temp_lineup:
            current_pos = getattr(player, 'current_position', getattr(player, 'position', None))
            if current_pos == position:
                return player
        return None
        
    def _on_position_click(self, position):
        """守備位置クリック時の処理"""
        player = self._find_player_at_position(position)
        
        if player:
            # 選手が配置されている場合、選択メニューを表示
            self._show_position_menu(position, player)
        else:
            # 空のポジションの場合、ベンチからの配置メニューを表示
            self._show_bench_assignment_menu(position)
            
    def _on_bench_player_click(self, bench_index):
        """ベンチ選手クリック時の処理"""
        if bench_index >= len(self.temp_bench):
            return
            
        player = self.temp_bench[bench_index]
        self._show_bench_player_menu(bench_index, player)
        
    def _show_position_menu(self, position, player):
        """守備位置選手のメニューを表示"""
        # 直接交代候補の選手を表示
        self._show_exchange_menu(player)
            
    def _show_bench_assignment_menu(self, position):
        """ベンチからの配置メニューを表示"""
        if not self.temp_bench:
            messagebox.showinfo("Info", "No bench players available")
            return
            
        # 適性のある選手のみを表示
        suitable_players = []
        for i, player in enumerate(self.temp_bench):
            if player.can_play_position(position):
                suitable_players.append((i, player))
                
        if not suitable_players:
            messagebox.showinfo("Info", f"No bench players can play {position}")
            return
            
        # 選択ダイアログを表示
        self._show_bench_selection_dialog(position, suitable_players)
        
    def _show_bench_player_menu(self, bench_index, player):
        """ベンチ選手のメニューを表示"""
        # 直接交代候補の選手を表示
        self._show_bench_exchange_menu(bench_index, player)
            
    def _retire_player(self, position, player):
        """選手を退場させる（再使用不可）"""
        # 確認ダイアログ
        confirm = messagebox.askyesno("Confirm Retirement", 
                                    f"Are you sure you want to retire {player.name}?")
        if not confirm:
            return
            
        # 表示用に退場時点の情報を保存
        self._record_retired_detail(player, position)

        # ポジションから削除
        setattr(player, 'current_position', None)
        
        # 退場選手リストに追加
        if player not in self.retired_players:
            self.retired_players.append(player)
            
        # ベンチからも削除（もしベンチにいる場合）
        if player in self.temp_bench:
            self.temp_bench.remove(player)
            
        # ラインナップからも削除
        if player in self.temp_lineup:
            self.temp_lineup.remove(player)
            
        # 変更を記録
        change_text = f"{player.name} retired from {position}"
        self.changes_made.append(change_text)
        
        self._update_display()

    def _record_retired_detail(self, player, actual_position):
        """退場選手の表示用詳細を記録する"""
        # 既に記録済みならスキップ
        for d in self.retired_details:
            if d.get('player') == player:
                return
        eligible_positions = []
        if hasattr(player, 'get_display_eligible_positions'):
            eligible_positions = player.get_display_eligible_positions()
        elif hasattr(player, 'eligible_positions') and player.eligible_positions:
            eligible_positions = list(player.eligible_positions)
        self.retired_details.append({
            'player': player,
            'name': getattr(player, 'name', 'Unknown'),
            'actual_position': actual_position,
            'eligible_positions': eligible_positions,
        })
        
    def _assign_to_position(self, bench_index, position):
        """ベンチ選手をポジションに配置"""
        if bench_index >= len(self.temp_bench):
            return
            
        player = self.temp_bench[bench_index]
        
        # ポジションに配置
        setattr(player, 'current_position', position)
        
        # ベンチから削除
        self.temp_bench.remove(player)
        
        # 変更を記録
        change_text = f"{player.name} assigned to {position}"
        self.changes_made.append(change_text)
        
        self._update_display()
        
    def _replace_player_at_position(self, bench_index, position):
        """ベンチ選手とポジション選手を交代"""
        if bench_index >= len(self.temp_bench):
            return
            
        bench_player = self.temp_bench[bench_index]
        field_player = self._find_player_at_position(position)
        
        if not field_player:
            self._assign_to_position(bench_index, position)
            return
            
        # 交代実行
        setattr(bench_player, 'current_position', position)
        setattr(field_player, 'current_position', None)
        
        # ベンチの更新
        self.temp_bench[bench_index] = field_player
        
        # 変更を記録
        change_text = f"{bench_player.name} replaces {field_player.name} at {position}"
        self.changes_made.append(change_text)
        
        self._update_display()
        
    def _show_exchange_menu(self, player):
        """他選手との交換メニューを表示"""
        current_pos = getattr(player, 'current_position', getattr(player, 'position', None))
        
        if not current_pos:
            messagebox.showinfo("Info", f"{player.name} is not assigned to any position")
            return
        
        menu = tk.Menu(self.mode_window, tearoff=0)
        
        # フィールドの他の選手との交換
        field_exchange_options = []
        for other_player in self.temp_lineup:
            if other_player == player:
                continue
                
            other_pos = getattr(other_player, 'current_position', getattr(other_player, 'position', None))
            
            if not other_pos:
                continue  # ポジションが設定されていない選手はスキップ
            
            # 相互の適性状況を表示
            can_player_play_other = player.can_play_position(other_pos)
            can_other_play_current = other_player.can_play_position(current_pos)
            
            if can_player_play_other and can_other_play_current:
                status = "✓✓"
            elif can_player_play_other:
                status = "✓⚠"
            elif can_other_play_current:
                status = "⚠✓"
            else:
                status = "⚠⚠"
            
            field_exchange_options.append((other_player, other_pos, status))
        
        # フィールド選手との交換オプションを追加
        if field_exchange_options:
            menu.add_command(label="--- Exchange with Field Players ---", state="disabled")
            for other_player, other_pos, status in field_exchange_options:
                menu.add_command(label=f"{status} Exchange with {other_player.name} ({other_pos})",
                               command=lambda op=other_player: self._exchange_positions(player, op))
        
        # ベンチ選手との交代オプション
        bench_options = []
        for bench_player in self.temp_bench:
            # ベンチ選手が現在のポジションを守れるかチェック
            can_bench_play_current = bench_player.can_play_position(current_pos)
            status = "✓" if can_bench_play_current else "⚠"
            bench_options.append((bench_player, status))
        
        # ベンチ選手との交代オプションを追加
        if bench_options:
            if field_exchange_options:  # フィールド選手のオプションがある場合はセパレータを追加
                menu.add_separator()
            menu.add_command(label="--- Replace with Bench Players ---", state="disabled")
            for bench_player, status in bench_options:
                menu.add_command(label=f"{status} Replace with {bench_player.name} (retire current)",
                               command=lambda bp=bench_player: self._replace_with_bench_player(player, bp))
        
        if not field_exchange_options and not bench_options:
            messagebox.showinfo("Info", "No suitable players for exchange")
            return
        
        # キャンセルボタンを追加
        menu.add_separator()
        menu.add_command(label="Cancel", command=lambda: menu.unpost())
        
        # メニューを表示
        try:
            menu.tk_popup(self.mode_window.winfo_pointerx(), 
                         self.mode_window.winfo_pointery())
        finally:
            menu.grab_release()
            
    def _exchange_positions(self, player1, player2):
        """二人の選手のポジションを交換"""
        pos1 = getattr(player1, 'current_position', getattr(player1, 'position', None))
        pos2 = getattr(player2, 'current_position', getattr(player2, 'position', None))
        
        if not pos1 or not pos2:
            messagebox.showerror("Error", "Cannot exchange positions: One or both players have no assigned position")
            return
        
        # ポジション交換
        setattr(player1, 'current_position', pos2)
        setattr(player2, 'current_position', pos1)
        
        # 変更を記録
        change_text = f"{player1.name} and {player2.name} exchanged positions ({pos1} ↔ {pos2})"
        self.changes_made.append(change_text)
        
        self._update_display()
        
    def _replace_with_bench_player(self, field_player, bench_player):
        """フィールド選手をベンチ選手と交代させ、フィールド選手を退場させる"""
        current_pos = getattr(field_player, 'current_position', getattr(field_player, 'position', None))
        
        if not current_pos:
            messagebox.showerror("Error", f"{field_player.name} has no assigned position")
            return
        
        # 確認ダイアログ
        confirm = messagebox.askyesno("Confirm Substitution", 
                                    f"Replace {field_player.name} with {bench_player.name} at {current_pos}?\n"
                                    f"{field_player.name} will be retired and cannot be reused.")
        if not confirm:
            return
        
        # ベンチ選手を現在のポジションに配置
        setattr(bench_player, 'current_position', current_pos)
        
        # ベンチ選手をラインナップに追加
        if bench_player not in self.temp_lineup:
            self.temp_lineup.append(bench_player)
        
        # 退場時点の情報を保存
        self._record_retired_detail(field_player, current_pos)

        # フィールド選手を退場させる
        setattr(field_player, 'current_position', None)
        if field_player not in self.retired_players:
            self.retired_players.append(field_player)
        
        # ラインナップからフィールド選手を削除
        if field_player in self.temp_lineup:
            self.temp_lineup.remove(field_player)
        
        # ベンチからベンチ選手を削除
        if bench_player in self.temp_bench:
            self.temp_bench.remove(bench_player)
        
        # 変更を記録
        change_text = f"{bench_player.name} replaces {field_player.name} at {current_pos} ({field_player.name} retired)"
        self.changes_made.append(change_text)
        
        self._update_display()
        
    def _show_bench_exchange_menu(self, bench_index, bench_player):
        """ベンチ選手の交換メニューを表示"""
        if bench_index >= len(self.temp_bench):
            return
            
        menu = tk.Menu(self.mode_window, tearoff=0)
        
        # フィールド選手との交代オプション
        field_options = []
        for field_player in self.temp_lineup:
            field_pos = getattr(field_player, 'current_position', getattr(field_player, 'position', None))
            
            if not field_pos:
                continue  # ポジションが設定されていない選手はスキップ
            
            # ベンチ選手がそのポジションを守れるかチェック
            can_bench_play_field = bench_player.can_play_position(field_pos)
            status = "✓" if can_bench_play_field else "⚠"
            field_options.append((field_player, field_pos, status))
        
        # フィールド選手との交代オプションを追加
        if field_options:
            menu.add_command(label="--- Replace Field Players ---", state="disabled")
            for field_player, field_pos, status in field_options:
                menu.add_command(label=f"{status} Replace {field_player.name} at {field_pos} (retire current)",
                               command=lambda fp=field_player: self._bench_replace_field_player(bench_index, fp))
        
        # 他のベンチ選手との交換オプション
        other_bench_options = []
        for i, other_bench_player in enumerate(self.temp_bench):
            if i == bench_index or other_bench_player == bench_player:
                continue  # 自分自身はスキップ
            other_bench_options.append((i, other_bench_player))
        
        # 他のベンチ選手との交換オプションを追加
        if other_bench_options:
            if field_options:  # フィールド選手のオプションがある場合はセパレータを追加
                menu.add_separator()
            menu.add_command(label="--- Exchange with Bench Players ---", state="disabled")
            for other_index, other_bench_player in other_bench_options:
                menu.add_command(label=f"Exchange positions with {other_bench_player.name}",
                               command=lambda oi=other_index: self._exchange_bench_players(bench_index, oi))
        
        if not field_options and not other_bench_options:
            messagebox.showinfo("Info", "No players available for exchange")
            return
        
        # キャンセルボタンを追加
        menu.add_separator()
        menu.add_command(label="Cancel", command=lambda: menu.unpost())
        
        # メニューを表示
        try:
            menu.tk_popup(self.mode_window.winfo_pointerx(), 
                         self.mode_window.winfo_pointery())
        finally:
            menu.grab_release()
            
    def _bench_replace_field_player(self, bench_index, field_player):
        """ベンチ選手がフィールド選手と交代"""
        if bench_index >= len(self.temp_bench):
            return
            
        bench_player = self.temp_bench[bench_index]
        self._replace_with_bench_player(field_player, bench_player)
        
    def _exchange_bench_players(self, bench_index1, bench_index2):
        """ベンチ選手同士でベンチ内の順序を交換"""
        if (bench_index1 >= len(self.temp_bench) or 
            bench_index2 >= len(self.temp_bench) or 
            bench_index1 == bench_index2):
            return
            
        # ベンチ内での位置を交換
        player1 = self.temp_bench[bench_index1]
        player2 = self.temp_bench[bench_index2]
        
        self.temp_bench[bench_index1] = player2
        self.temp_bench[bench_index2] = player1
        
        # 変更を記録
        change_text = f"{player1.name} and {player2.name} exchanged bench positions"
        self.changes_made.append(change_text)
        
        self._update_display()
        
    def _show_bench_selection_dialog(self, position, suitable_players):
        """ベンチ選手選択ダイアログを表示"""
        dialog = tk.Toplevel(self.mode_window)
        dialog.title(f"Assign Player to {position}")
        dialog.geometry("480x360")  # 400x300の1.2倍
        
        ttk.Label(dialog, text=f"Select player for {position}", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        for bench_index, player in suitable_players:
            button_text = f"{player.name}"
            ttk.Button(dialog, text=button_text,
                      command=lambda idx=bench_index: [self._assign_to_position(idx, position), dialog.destroy()]).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=10)
        
    def _reset_changes(self):
        """すべての変更をリセット"""
        confirm = messagebox.askyesno("Confirm Reset", 
                                    "Are you sure you want to reset all changes?")
        if confirm:
            self.temp_lineup = copy.deepcopy(self.team.lineup)
            self.temp_bench = copy.deepcopy(self.team.bench)
            self.retired_players = []  # 退場選手もリセット
            self.changes_made = []
            
            # 各選手にcurrent_positionが設定されていることを確認
            for player in self.temp_lineup:
                if not hasattr(player, 'current_position') or player.current_position is None:
                    player.current_position = getattr(player, 'position', None)
            
            self._update_display()
            
    def _check_formation(self):
        """現在のフォーメーションをチェック"""
        issues = self._validate_defense_positions()
        
        if issues:
            # エラーと警告を分ける
            errors = [issue for issue in issues if not issue.startswith("⚠")]
            warnings = [issue for issue in issues if issue.startswith("⚠")]
            
            message_parts = []
            if errors:
                message_parts.append("ERRORS:\n" + "\n".join(errors))
            if warnings:
                message_parts.append("WARNINGS:\n" + "\n".join(warnings))
            
            issue_message = "\n\n".join(message_parts)
            
            if errors:
                messagebox.showerror("Formation Check", issue_message)
                self.status_label.configure(text="Invalid Formation", foreground="red")
            else:
                messagebox.showwarning("Formation Check", issue_message)
                self.status_label.configure(text="Formation has warnings", foreground="orange")
        else:
            messagebox.showinfo("Formation Check", "Formation is valid!")
            self.status_label.configure(text="Valid Formation", foreground="green")
            
    def _validate_defense_positions(self):
        """守備位置の妥当性をチェック"""
        errors = []
        warnings = []
        
        # 必要なポジション（ピッチャーは除外）
        required_positions = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
        
        # 各ポジションに選手が配置されているかチェック
        filled_positions = []
        for player in self.temp_lineup:
            current_pos = getattr(player, 'current_position', None)
            if current_pos and current_pos not in ["P", "DH"]:  # ピッチャーとDHは必須ではない
                filled_positions.append(current_pos)
                
                # 選手がそのポジションを守れるかチェック（警告のみ）
                if not player.can_play_position(current_pos):
                    warnings.append(f"⚠ {player.name} cannot play {current_pos} (risk of errors)")
        
        # 必要なポジションが埋まっているかチェック
        for pos in required_positions:
            if pos not in filled_positions:
                errors.append(f"No player assigned to {pos}")
                
        # 重複チェック
        position_counts = {}
        for pos in filled_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
            
        for pos, count in position_counts.items():
            if count > 1:
                errors.append(f"Multiple players assigned to {pos}")
        
        # 警告も含めて返す
        all_issues = errors + warnings
        return all_issues
        
    def _apply_changes(self):
        """変更を実際のチームに適用"""
        # ラインナップの更新（退場していない選手のみ）
        updated_lineup = []
        for temp_player in self.temp_lineup:
            if temp_player not in self.retired_players:
                # 対応する元の選手を見つけて更新
                for original_player in self.team.lineup:
                    if original_player.name == temp_player.name:
                        temp_pos = getattr(temp_player, 'current_position', None)
                        if temp_pos:
                            setattr(original_player, 'current_position', temp_pos)
                        updated_lineup.append(original_player)
                        break
        
        # 退場していないベンチ選手の更新
        updated_bench = []
        for temp_player in self.temp_bench:
            if temp_player not in self.retired_players:
                # 対応する元の選手を見つけて更新
                for original_player in self.team.bench:
                    if original_player.name == temp_player.name:
                        updated_bench.append(original_player)
                        break
        
        # チームのラインナップとベンチを更新
        self.team.lineup = [player for player in self.team.lineup if not any(retired.name == player.name for retired in self.retired_players)]
        self.team.bench = updated_bench
        
        # メインGUIの更新
        if self.main_gui:
            self.main_gui._update_scoreboard(self.main_gui.game_state)
