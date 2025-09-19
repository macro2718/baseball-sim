"""
Event Management for GUI
GUIコンポーネント間のイベント管理
"""

from baseball_sim.infrastructure.logging_utils import logger


class EventManager:
    """
    GUI内でのイベント管理クラス
    コンポーネント間の疎結合を実現
    """
    
    def __init__(self):
        self._handlers = {}
        
    def bind(self, event_name, handler):
        """イベントハンドラーを登録
        
        Args:
            event_name (str): イベント名
            handler (callable): イベントハンドラー関数
        """
        if event_name not in self._handlers:
            self._handlers[event_name] = []
        self._handlers[event_name].append(handler)
    
    def unbind(self, event_name, handler):
        """イベントハンドラーを解除
        
        Args:
            event_name (str): イベント名
            handler (callable): イベントハンドラー関数
        """
        if event_name in self._handlers:
            if handler in self._handlers[event_name]:
                self._handlers[event_name].remove(handler)
    
    def trigger(self, event_name, *args, **kwargs):
        """イベントを発火
        
        Args:
            event_name (str): イベント名
            *args, **kwargs: ハンドラーに渡す引数
        """
        if event_name in self._handlers:
            for handler in self._handlers[event_name]:
                try:
                    handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in event handler for '{event_name}': {e}")
    
    def clear_handlers(self, event_name=None):
        """イベントハンドラーをクリア
        
        Args:
            event_name (str, optional): 特定のイベント名。Noneの場合は全てクリア
        """
        if event_name is None:
            self._handlers.clear()
        elif event_name in self._handlers:
            self._handlers[event_name].clear()
    
    def list_events(self):
        """登録されているイベント一覧を取得
        
        Returns:
            list: イベント名のリスト
        """
        return list(self._handlers.keys())
    
    def has_handlers(self, event_name):
        """指定されたイベントにハンドラーが登録されているかチェック
        
        Args:
            event_name (str): イベント名
            
        Returns:
            bool: ハンドラーが存在するかどうか
        """
        return event_name in self._handlers and len(self._handlers[event_name]) > 0


# 定義されているイベント名の定数
class Events:
    """GUIで使用されるイベント名の定数"""
    
    # チーム関連
    TEAMS_CREATED = "teams_created"
    LINEUP_CHANGED = "lineup_changed"
    TEAM_VALIDATED = "team_validated"
    
    # ゲーム関連
    GAME_STARTED = "game_started"
    GAME_STATE_CHANGED = "game_state_changed"
    INNING_CHANGED = "inning_changed"
    GAME_ENDED = "game_ended"
    
    # プレイ関連
    BATTING_ACTION = "batting_action"
    PLAY_RESULT = "play_result"
    STRATEGY_EXECUTED = "strategy_executed"
    
    # UI関連
    SCREEN_CHANGED = "screen_changed"
    DIALOG_OPENED = "dialog_opened"
    DIALOG_CLOSED = "dialog_closed"
