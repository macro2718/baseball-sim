"""
Field drawing and animation functionality for the Baseball GUI
"""
import tkinter as tk
import random
import math

class FieldManager:
    def __init__(self, canvas, text):
        self.canvas = canvas
        self.text = text
        self.game_state = None  # ゲーム状態を保存
    
    def set_game_state(self, game_state):
        """ゲーム状態を設定"""
        self.game_state = game_state
    
    def draw_field(self):
        """野球場を描画"""
        canvas = self.canvas
        canvas.delete("all")
        
        # キャンバスサイズ: 650x520 (500x400 * 1.3)
        width, height = 650, 520
        
        # フィールドの背景
        canvas.create_rectangle(0, 0, width, height, fill="#8FBC8F", outline="")
        
        # 内野 (中心座標を下に移動し、サイズを小さく調整)
        center_x, center_y = width // 2, int(height * 0.92)  # 325, 478（下に移動）
        infield_size = int(110 * 1.3)  # 143（小さく調整: 169→143）
        
        # 内野ダイヤモンド
        canvas.create_polygon(
            center_x, center_y,                    # ホーム
            center_x - infield_size, center_y - infield_size,  # 三塁
            center_x, center_y - infield_size * 2,  # 二塁
            center_x + infield_size, center_y - infield_size,  # 一塁
            fill="#D2B48C", outline="white", width=2
        )
        
        # ホームプレート
        home_x, home_y = center_x, center_y
        canvas.create_polygon(
            home_x, home_y, 
            home_x - 13, home_y + 13, 
            home_x + 13, home_y + 13, 
            fill="white", outline="black"
        )
        
        # 各ベース
        base_size = 26  # 20 * 1.3
        
        # 一塁
        first_x, first_y = center_x + infield_size, center_y - infield_size
        canvas.create_rectangle(
            first_x - base_size//2, first_y - base_size//2,
            first_x + base_size//2, first_y + base_size//2,
            fill="white", outline="black", tags="first_base"
        )
        
        # 二塁
        second_x, second_y = center_x, center_y - infield_size * 2
        canvas.create_rectangle(
            second_x - base_size//2, second_y - base_size//2,
            second_x + base_size//2, second_y + base_size//2,
            fill="white", outline="black", tags="second_base"
        )
        
        # 三塁
        third_x, third_y = center_x - infield_size, center_y - infield_size
        canvas.create_rectangle(
            third_x - base_size//2, third_y - base_size//2,
            third_x + base_size//2, third_y + base_size//2,
            fill="white", outline="black", tags="third_base"
        )
        
        # マウンド
        mound_size = int(30 * 1.3)  # 39
        mound_x, mound_y = center_x, center_y - infield_size // 2
        canvas.create_oval(
            mound_x - mound_size//2, mound_y - mound_size//2,
            mound_x + mound_size//2, mound_y + mound_size//2,
            fill="#D2B48C", outline="white"
        )
        
        # ベースライン
        canvas.create_line(home_x, home_y, first_x, first_y, fill="white", width=2)
        canvas.create_line(home_x, home_y, third_x, third_y, fill="white", width=2)
        canvas.create_line(third_x, third_y, second_x, second_y, fill="white", width=2)
        canvas.create_line(second_x, second_y, first_x, first_y, fill="white", width=2)
        
        # 外野の境界線（さらに下の位置に調整）
        arc_size = int(500 * 1.3)  # 650
        arc_start_x = center_x - arc_size//2
        arc_start_y = center_y - arc_size//2 - int(80 * 1.3)  # 104ピクセル上に移動（156→104に下げる）
        canvas.create_arc(
            arc_start_x, arc_start_y, 
            arc_start_x + arc_size, arc_start_y + arc_size,
            start=45, extent=90, style=tk.ARC, outline="white", width=4  # 線をさらに太く
        )
    
    def update_field(self, bases):
        """フィールド上の走者表示を更新"""
        canvas = self.canvas
        
        # ベース上の走者表示をリセット
        canvas.delete("runner")
        
        # キャンバスサイズ: 650x520
        width, height = 650, 520
        center_x, center_y = width // 2, int(height * 0.92)  # 325, 478（ダイヤモンドと同じ位置）
        infield_size = int(110 * 1.3)  # 143（ダイヤモンドと同じサイズ）
        base_size = 26
        
        # ベース座標を計算
        first_x, first_y = center_x + infield_size, center_y - infield_size
        second_x, second_y = center_x, center_y - infield_size * 2
        third_x, third_y = center_x - infield_size, center_y - infield_size
        
        # 一塁走者
        if bases[0] is not None:
            canvas.create_oval(
                first_x - base_size//2, first_y - base_size//2,
                first_x + base_size//2, first_y + base_size//2,
                fill="red", outline="black", tags="runner"
            )
            # 選手名を表示
            canvas.create_text(
                first_x, first_y + base_size + 8, 
                text=bases[0].name, 
                font=("Helvetica", 8, "bold"), 
                fill="white", 
                tags="runner"
            )
        
        # 二塁走者
        if bases[1] is not None:
            canvas.create_oval(
                second_x - base_size//2, second_y - base_size//2,
                second_x + base_size//2, second_y + base_size//2,
                fill="red", outline="black", tags="runner"
            )
            # 選手名を表示
            canvas.create_text(
                second_x, second_y - base_size - 8, 
                text=bases[1].name, 
                font=("Helvetica", 8, "bold"), 
                fill="white", 
                tags="runner"
            )
        
        # 三塁走者
        if bases[2] is not None:
            canvas.create_oval(
                third_x - base_size//2, third_y - base_size//2,
                third_x + base_size//2, third_y + base_size//2,
                fill="red", outline="black", tags="runner"
            )
            # 選手名を表示
            canvas.create_text(
                third_x - base_size - 8, third_y, 
                text=bases[2].name, 
                font=("Helvetica", 8, "bold"), 
                fill="white", 
                tags="runner"
            )
    
    def update_outs_display(self, outs):
        """アウトカウントを赤い円で表示する"""
        canvas = self.canvas
        
        # 既存のアウト表示を削除
        canvas.delete("out_display")
        
        # キャンバスサイズ: 650x520
        # フィールドの右上にアウトを表示（座標を1.3倍に調整）
        start_x, start_y = int(520 * 1.3), int(30 * 1.3)  # 676, 39 -> 調整して520, 39
        start_x = 520  # キャンバス幅内に収める
        
        for i in range(3):
            x = start_x + i * int(25 * 1.3)  # 32.5 -> 33
            y = start_y
            circle_size = int(20 * 1.3)  # 26
            
            if i < outs:
                canvas.create_oval(x, y, x+circle_size, y+circle_size, fill="red", outline="black", tags="out_display")
            else:
                canvas.create_oval(x, y, x+circle_size, y+circle_size, fill="white", outline="black", tags="out_display")
        
        # OUT テキスト
        text_x = start_x + int(40 * 1.3)  # 52
        text_y = start_y - int(15 * 1.3)  # -19.5 -> -20
        canvas.create_text(text_x, text_y, text="OUT", font=("Helvetica", 13, "bold"), tags="out_display")
    
    def animate_play_result(self, result, root):
        """プレー結果のアニメーション表示（非ブロッキング版）"""
        canvas = self.canvas
        
        # キャンバスサイズ: 650x520
        width, height = 650, 520
        center_x, center_y = width // 2, int(height * 0.92)  # 325, 478（ダイヤモンドと同じ位置）
        
        def animate_ball(dx, dy, steps, ball):
            if steps > 0:
                canvas.move(ball, dx, dy)
                root.after(5, lambda: animate_ball(dx, dy, steps-1, ball))  # さらに高速に（5ms→3ms）
            else:
                canvas.delete(ball)
        
        # ボールを作成（ホームプレート近くから開始）
        ball_size = int(10 * 1.3)  # 13
        ball = canvas.create_oval(
            center_x - ball_size//2, center_y - ball_size//2,
            center_x + ball_size//2, center_y + ball_size//2,
            fill="white", outline="black"
        )
        
        # 結果に応じてアニメーション
        if result in [
            "single",
            "double",
            "triple",
            "home_run",
            "groundout",
            "outfield_flyout",
            "infield_flyout",
        ]:
            # ヒットやアウトの場合のボール軌道
            if result == "single":
                steps = 100
                speed = 2.5
            elif result == "home_run":
                steps = 120
                speed = 4.0
            elif result == "double":
                steps = 120
                speed = 3
            elif result == "triple":
                steps = 100
                speed = 3.5
            elif result == "groundout":
                steps = 100
                speed = 1.9
            elif result in ("outfield_flyout", "infield_flyout"):
                steps = 130
                speed = 2.5
            rad = math.radians(random.uniform(45, 135))
            dx = speed * -math.cos(rad)
            dy = speed * -math.sin(rad)
            animate_ball(dx, dy, steps, ball)
            
            # ヒット結果のテキスト表示（中央に表示）
            hit_text = None
            text_x, text_y = center_x, center_y - int(100 * 1.3)  # 中央より少し上
            font_size = int(16 * 1.3)  # 21
            
            if result == "single":
                hit_text = canvas.create_text(text_x, text_y, text="Single", fill="blue", font=("Helvetica", font_size, "bold"))
            elif result == "double":
                hit_text = canvas.create_text(text_x, text_y, text="Double", fill="blue", font=("Helvetica", font_size, "bold"))
            elif result == "triple":
                hit_text = canvas.create_text(text_x, text_y, text="Triple", fill="blue", font=("Helvetica", font_size, "bold"))
            elif result == "home_run":
                hit_text = canvas.create_text(text_x, text_y, text="Home Run", fill="blue", font=("Helvetica", font_size, "bold"))
            
            if hit_text:
                root.after(1000, lambda: canvas.delete(hit_text))
        # バントの場合
        elif result == "bunt":
            # バント特有の短い軌道アニメーション
            steps = 50
            speed = 1.5
            # バントは前方への短い軌道
            rad = math.radians(random.uniform(60, 120))  # より前方向
            dx = speed * -math.cos(rad)
            dy = speed * -math.sin(rad)
            animate_ball(dx, dy, steps, ball)
            
            # バント結果のテキスト表示
            text_x, text_y = center_x, center_y - int(100 * 1.3)
            font_size = int(16 * 1.3)
            bunt_text = canvas.create_text(text_x, text_y, text="Bunt", fill="green", font=("Helvetica", font_size, "bold"))
            root.after(1000, lambda: canvas.delete(bunt_text))
        # 三振の場合
        elif result == "strikeout":
            canvas.delete(ball)
            def strike_animation(count):
                if count > 0:
                    text_x, text_y = center_x, center_y - int(100 * 1.3)
                    font_size = int(16 * 1.3)
                    strike_text = canvas.create_text(text_x, text_y, text=self.text["strike"], fill="red", font=("Helvetica", font_size, "bold"))
                    root.after(300, lambda: (canvas.delete(strike_text), strike_animation(count-1)))
            strike_animation(3)
        # 四球の場合
        elif result == "walk":
            text_x, text_y = center_x, center_y - int(100 * 1.3)
            font_size = int(16 * 1.3)
            ball_text = canvas.create_text(text_x, text_y, text=self.text["ball"], fill="green", font=("Helvetica", font_size, "bold"))
            root.after(1000, lambda: canvas.delete(ball_text))
    
    def animate_change(self, root):
        """攻守交代のアニメーション"""
        canvas = self.canvas
        
        # キャンバスサイズ: 650x520
        width, height = 650, 520
        center_x, center_y = width // 2, height // 2  # 325, 260
        font_size = int(20 * 1.3)  # 26
        
        change_text = canvas.create_text(center_x, center_y, text="Change Sides", fill="orange", font=("Helvetica", font_size, "bold"))
        root.after(2000, lambda: canvas.delete(change_text))
