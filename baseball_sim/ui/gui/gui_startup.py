"""GUI components for selecting modes and running simulations."""

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from baseball_sim.config import setup_project_environment
from baseball_sim.interface.simulation import simulate_games

setup_project_environment()


class StartupWindow:
    """アプリ起動時にモード選択を行うウィンドウ"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Baseball Simulation - Select Mode")
        self.root.geometry("800x520")
        self.root.resizable(False, False)

        self.selected_mode = None

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_widgets(self) -> None:
        """モード選択画面のウィジェットを構築"""
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)

        title_label = ttk.Label(
            main_frame,
            text="Welcome to the Baseball Simulator",
            font=("Helvetica", 25, "bold"),
            anchor="center"
        )
        title_label.pack(pady=(0, 20))

        description = ttk.Label(
            main_frame,
            text="Please select a mode to start.",
            font=("Helvetica", 16),
            anchor="center"
        )
        description.pack(pady=(0, 40))

        gui_button = ttk.Button(
            main_frame,
            text="GUI Mode (Team Management & Games)",
            command=lambda: self._select_mode("gui")
        )
        gui_button.pack(fill="x", pady=15)

        simulation_button = ttk.Button(
            main_frame,
            text="Simulation Mode (Automated Games)",
            command=lambda: self._select_mode("simulation")
        )
        simulation_button.pack(fill="x", pady=5)

        exit_button = ttk.Button(
            main_frame,
            text="Exit",
            command=self._on_close
        )
        exit_button.pack(fill="x", pady=(20, 0))

    def _select_mode(self, mode: str) -> None:
        """モードを選択しメインループを終了"""
        self.selected_mode = mode
        self.root.quit()

    def _on_close(self) -> None:
        """ウィンドウを閉じる際の処理"""
        self.selected_mode = None
        self.root.quit()

    def show(self):
        """ウィンドウを表示し、選択されたモードを返す"""
        self.root.mainloop()
        self.root.destroy()
        return self.selected_mode


class SimulationWindow:
    """シミュレーションモードの操作と進捗表示を行うウィンドウ"""

    POLL_INTERVAL_MS = 100

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Baseball Simulation - Simulation Mode")
        self.root.geometry("960x780")

        self.progress_queue = queue.Queue()
        self.simulation_thread = None
        self.is_running = False
        self.last_results = None
        self.total_games = 0

        self.num_games_var = tk.StringVar(value="10")
        self.output_file_var = tk.StringVar()
        self.progress_label_var = tk.StringVar(value="0 / 0 games")
        self.status_var = tk.StringVar(value="Waiting for simulation to start...")

        self._build_widgets()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        try:
            if self.root.winfo_exists():
                self.root.after(self.POLL_INTERVAL_MS, self._process_queue)
        except tk.TclError:
            pass

    def _build_widgets(self) -> None:
        """シミュレーションウィンドウのウィジェット構築"""
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)

        settings_frame = ttk.LabelFrame(main_frame, text="Simulation Settings", padding=15)
        settings_frame.pack(fill="x")

        ttk.Label(settings_frame, text="Number of Games:").grid(row=0, column=0, sticky="w")
        self.games_entry = ttk.Entry(settings_frame, textvariable=self.num_games_var, width=10)
        self.games_entry.grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(settings_frame, text="Output File Name:\n(Leave blank for auto-generate)").grid(
            row=1, column=0, sticky="nw", pady=(10, 0)
        )
        self.output_entry = ttk.Entry(settings_frame, textvariable=self.output_file_var)
        self.output_entry.grid(row=1, column=1, sticky="we", padx=(5, 0), pady=(10, 0))

        settings_frame.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(15, 0))

        self.start_button = ttk.Button(button_frame, text="Start Simulation", command=self._start_simulation)
        self.start_button.pack(side="left")

        self.close_button = ttk.Button(button_frame, text="Close", command=self._on_close)
        self.close_button.pack(side="right")

        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding=15)
        progress_frame.pack(fill="x", pady=(15, 0))

        self.progress_bar = ttk.Progressbar(progress_frame, maximum=1, value=0)
        self.progress_bar.pack(fill="x")

        progress_label = ttk.Label(progress_frame, textvariable=self.progress_label_var, anchor="e")
        progress_label.pack(fill="x", pady=(5, 0))

        status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="#333333")
        status_label.pack(fill="x", pady=(15, 0))

        log_frame = ttk.LabelFrame(main_frame, text="Progress Log", padding=10)
        log_frame.pack(fill="both", expand=True, pady=(15, 0))

        self.log_text = tk.Text(log_frame, height=12, state="disabled", wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _start_simulation(self) -> None:
        """シミュレーションを開始"""
        if self.is_running:
            return

        try:
            num_games = int(self.num_games_var.get())
            if num_games <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input error", "Please enter an integer greater than or equal to 1 for the number of games.")
            return

        output_file = self.output_file_var.get().strip() or None

        self.total_games = num_games
        self.progress_bar.config(maximum=num_games)
        self.progress_bar["value"] = 0
        self.progress_label_var.set(f"0 / {num_games} games")
        self.status_var.set("Preparing for simulation...")
        self._append_log(f"Starting simulation for {num_games} games.")

        self.games_entry.config(state=tk.DISABLED)
        self.output_entry.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.close_button.config(state=tk.DISABLED)
        self.is_running = True

        self.simulation_thread = threading.Thread(
            target=self._run_simulation,
            args=(num_games, output_file),
            daemon=True,
        )
        self.simulation_thread.start()

    def _run_simulation(self, num_games: int, output_file: Optional[str]) -> None:
        """シミュレーションを別スレッドで実行"""
        try:
            results = simulate_games(
                num_games=num_games,
                output_file=output_file,
                progress_callback=self._enqueue_progress,
                message_callback=self._enqueue_message,
            )
            self.progress_queue.put(("completed", results))
        except Exception as exc:  # pylint: disable=broad-except
            self.progress_queue.put(("error", str(exc)))

    def _enqueue_progress(self, current: int, total: int) -> None:
        """進捗更新をキューに送信"""
        self.progress_queue.put(("progress", current, total))

    def _enqueue_message(self, message: str) -> None:
        """メッセージ更新をキューに送信"""
        self.progress_queue.put(("message", message))

    def _process_queue(self) -> None:
        """バックグラウンド処理からのメッセージをUIに反映"""
        try:
            while True:
                item = self.progress_queue.get_nowait()
                event_type = item[0]

                if event_type == "progress":
                    _, current, total = item
                    self.progress_bar.config(maximum=max(total, 1))
                    self.progress_bar["value"] = current
                    self.progress_label_var.set(f"{current} / {total} games")
                elif event_type == "message":
                    _, message = item
                    self.status_var.set(message)
                    self._append_log(message)
                elif event_type == "completed":
                    _, results = item
                    self._on_simulation_complete(results)
                elif event_type == "error":
                    _, error_message = item
                    self._on_simulation_error(error_message)
        except queue.Empty:
            pass

        try:
            if self.root.winfo_exists():
                self.root.after(self.POLL_INTERVAL_MS, self._process_queue)
        except tk.TclError:
            pass

    def _on_simulation_complete(self, results):
        """シミュレーション完了時の処理"""
        self.is_running = False
        self.last_results = results

        self.games_entry.config(state=tk.NORMAL)
        self.output_entry.config(state=tk.NORMAL)
        self.start_button.config(state=tk.NORMAL)
        self.close_button.config(state=tk.NORMAL)

        output_path = results.get("output_file") if isinstance(results, dict) else None
        if output_path:
            completion_message = f"Simulation complete: {output_path}"
        else:
            completion_message = "Simulation complete."

        if self.status_var.get() != completion_message:
            self.status_var.set(completion_message)
            self._append_log(completion_message)

        messagebox.showinfo("Complete", completion_message)

    def _on_simulation_error(self, error_message: str) -> None:
        """エラー発生時の処理"""
        self.is_running = False

        self.games_entry.config(state=tk.NORMAL)
        self.output_entry.config(state=tk.NORMAL)
        self.start_button.config(state=tk.NORMAL)
        self.close_button.config(state=tk.NORMAL)

        self.status_var.set(f"Error: {error_message}")
        self._append_log(f"Error: {error_message}")
        messagebox.showerror("Error", error_message)

    def _append_log(self, message: str) -> None:
        """ログエリアにメッセージを追加"""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        """ウィンドウを閉じる"""
        if self.is_running:
            messagebox.showwarning("Simulation in Progress", "Please wait until the simulation is complete.")
            return

        self.root.destroy()

    def show(self):
        """ウィンドウを表示"""
        self.root.mainloop()
        return self.last_results


__all__ = ["StartupWindow", "SimulationWindow"]
