"""Simulation helpers for :class:`WebGameSession`."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Tuple

from baseball_sim.interface.simulation import simulate_games
from baseball_sim.data.team_library import TeamLibraryError

from .exceptions import GameSessionError
from .simulation_summary import summarize_simulation_results


class SimulationControlsMixin:
    """Provide background simulation support for the web UI."""

    _simulation_state: Dict[str, Any]
    _home_team_source: Optional[Dict[str, Any]]
    _away_team_source: Optional[Dict[str, Any]]

    def _clear_simulation_results(self) -> None:
        self._simulation_state["last_run"] = None
        self._simulation_state["log"] = []
        self._simulation_state["running"] = False
        self._simulation_state["league"] = {
            "teams": [],
            "games_per_card": None,
            "cards_per_opponent": None,
        }
        self._simulation_state["playable"] = {
            "teams": [],
            "selection": {"home": None, "away": None},
        }
        if hasattr(self, "_simulation_playable"):
            self._simulation_playable = {}

    def get_simulation_state(self) -> Dict[str, Any]:
        state = {
            "enabled": bool(self.home_team and self.away_team),
            "running": bool(self._simulation_state.get("running", False)),
            "default_games": int(self._simulation_state.get("default_games", 20) or 20),
            "limits": self._simulation_state.get("limits", {"min_games": 1, "max_games": 200}),
            "last_run": self._simulation_state.get("last_run"),
            "log": list(self._simulation_state.get("log", []))[-20:],
        }
        league_state = self._simulation_state.get("league") or {}
        state["league"] = {
            "teams": list(league_state.get("teams", [])),
            "games_per_card": league_state.get("games_per_card"),
            "cards_per_opponent": league_state.get("cards_per_opponent"),
        }
        playable_state = self._simulation_state.get("playable") or {}
        teams = []
        for entry in playable_state.get("teams", []):
            if not isinstance(entry, Mapping):
                continue
            team_entry = {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "record": entry.get("record") or {},
                "roles": list(entry.get("roles", [])),
                "summary": entry.get("summary") or None,
            }
            if entry.get("source_id"):
                team_entry["source_id"] = entry.get("source_id")
            teams.append(team_entry)
        selection = playable_state.get("selection") or {}
        state["playable"] = {
            "teams": teams,
            "selection": {
                "home": selection.get("home"),
                "away": selection.get("away"),
            },
        }
        return state

    def _prepare_league_request(
        self, league_options: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not isinstance(league_options, Mapping):
            raise GameSessionError("リーグ設定の形式が不正です。")

        raw_teams = league_options.get("teams")
        if not isinstance(raw_teams, (list, tuple)) or len(raw_teams) < 2:
            raise GameSessionError("リーグには2チーム以上の参加が必要です。")

        team_datas: List[Dict[str, Any]] = []
        team_ids: List[str] = []
        for entry in raw_teams:
            team_payload: Optional[Mapping[str, Any]] = None
            team_id: Optional[str] = None
            if isinstance(entry, Mapping):
                maybe_id = entry.get("id") or entry.get("team_id")
                if isinstance(maybe_id, str):
                    team_id = maybe_id.strip()
                raw_payload = entry.get("data") or entry.get("team")
                if isinstance(raw_payload, Mapping):
                    team_payload = raw_payload
            elif isinstance(entry, str):
                team_id = entry.strip()
            else:
                raise GameSessionError("リーグチームの指定が不正です。")

            if team_payload is None:
                if not team_id:
                    raise GameSessionError("リーグチームのIDを指定してください。")
                try:
                    team_payload = self._team_library.load_team(team_id)
                except TeamLibraryError as exc:  # pragma: no cover - pass through
                    raise GameSessionError(str(exc)) from exc
            team_datas.append(dict(team_payload))
            team_ids.append(team_id or str(len(team_datas)))

        if len(team_datas) % 2 != 0:
            raise GameSessionError("リーグ参加チーム数は偶数で指定してください。")

        games_per_card = int(league_options.get("games_per_card", 0) or 0)
        cards_per_opponent = int(league_options.get("cards_per_opponent", 0) or 0)
        if games_per_card <= 0 or cards_per_opponent <= 0:
            raise GameSessionError("カード設定は1以上の数値で指定してください。")

        role_assignment_raw = league_options.get("role_assignment")
        role_assignment: Dict[str, int] = {}
        if isinstance(role_assignment_raw, Mapping):
            for role, index in role_assignment_raw.items():
                try:
                    normalized_index = int(index)
                except (TypeError, ValueError):
                    continue
                if 0 <= normalized_index < len(team_datas):
                    role_assignment[str(role)] = normalized_index

        league_request: Dict[str, Any] = {
            "teams": team_datas,
            "games_per_card": games_per_card,
            "cards_per_opponent": cards_per_opponent,
        }
        if role_assignment:
            league_request["role_assignment"] = role_assignment

        league_context: Dict[str, Any] = {
            "teams": team_ids,
            "games_per_card": games_per_card,
            "cards_per_opponent": cards_per_opponent,
        }
        if role_assignment:
            league_context["role_assignment"] = role_assignment

        return league_request, league_context

    def _iter_simulation_teams(self, results: Mapping[str, Any]):
        teams = results.get("teams") or {}
        seen: set[int] = set()
        for team in teams.values():
            if team is None:
                continue
            identifier = id(team)
            if identifier in seen:
                continue
            seen.add(identifier)
            yield team

    def run_simulation(
        self, num_games: Optional[int] = None, *, league_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        league_request: Optional[Dict[str, Any]] = None
        league_context: Optional[Dict[str, Any]] = None

        if league_options:
            league_request, league_context = self._prepare_league_request(league_options)
        else:
            if num_games is None:
                raise GameSessionError("シミュレーションする試合数を指定してください。")

        limits = self._simulation_state.get("limits", {"min_games": 1, "max_games": 200})
        min_games = int(limits.get("min_games", 1) or 1)
        max_games = int(limits.get("max_games", 200) or 200)

        if league_request is None:
            if num_games < min_games:
                raise GameSessionError(f"シミュレーション試合数は{min_games}以上で指定してください。")
            if num_games > max_games:
                raise GameSessionError(f"シミュレーション試合数は最大{max_games}試合までです。")

        self.ensure_teams()
        if league_request is None:
            if not self.home_team or not self.away_team:
                raise GameSessionError("チームが読み込まれていません。チームを選択してください。")

            if not self._home_team_source or not self._away_team_source:
                raise GameSessionError("チームデータを読み込めませんでした。チーム選択を確認してください。")

        self._simulation_state["running"] = True

        if league_request is not None:
            progress_messages: List[str] = ["Starting league simulation..."]
        else:
            progress_messages = [f"Simulating {num_games} games..."]

        def handle_message(message: str) -> None:
            if not message:
                return
            progress_messages.append(str(message))

        try:
            if league_request is not None:
                results = simulate_games(
                    num_games=None,
                    output_file=None,
                    progress_callback=None,
                    message_callback=handle_message,
                    league_options=league_request,
                    save_to_file=False,
                )
            else:
                results = simulate_games(
                    num_games=num_games,
                    output_file=None,
                    progress_callback=None,
                    message_callback=handle_message,
                    home_team_data=self._home_team_source,
                    away_team_data=self._away_team_source,
                    save_to_file=False,
                )
        except Exception as exc:  # pragma: no cover - defensive
            self._simulation_state["running"] = False
            raise GameSessionError(f"シミュレーションに失敗しました: {exc}") from exc

        self._simulation_state["running"] = False
        if league_context is not None:
            self._simulation_state["default_games"] = int(league_context.get("games_per_card", 0) or 0)
            self._simulation_state["league"] = league_context
        else:
            self._simulation_state["default_games"] = num_games
            self._simulation_state["league"] = {
                "teams": [self._home_team_id, self._away_team_id],
                "games_per_card": num_games,
                "cards_per_opponent": 1,
            }
        self._simulation_state["log"] = progress_messages[-20:]

        summary = summarize_simulation_results(
            results,
            home_team=self.home_team,
            away_team=self.away_team,
            num_games=num_games,
            league=league_context,
        )

        unique_sim_teams = list(self._iter_simulation_teams(results))
        summary_lookup: Dict[str, Dict[str, Any]] = {}
        for entry in summary.get("teams", []):
            if isinstance(entry, Mapping):
                key = entry.get("simulationKey")
                if isinstance(key, str):
                    summary_lookup[key] = entry  # type: ignore[assignment]

        playable_entries: List[Dict[str, Any]] = []
        playable_map: Dict[str, Dict[str, Any]] = {}
        playable_records: List[Dict[str, Any]] = []
        source_ids: List[Optional[str]] = []
        if league_context is not None:
            raw_ids = league_context.get("teams")
            if isinstance(raw_ids, list):
                source_ids = [str(team_id) if team_id is not None else None for team_id in raw_ids]
        else:
            source_ids = [self._away_team_id, self._home_team_id]

        for index, team_obj in enumerate(unique_sim_teams):
            sim_key = f"sim-team-{index + 1}"
            entry = summary_lookup.get(sim_key) or {}
            display_name = entry.get("name") if isinstance(entry.get("name"), str) else getattr(team_obj, "name", sim_key)
            record = entry.get("record") if isinstance(entry.get("record"), Mapping) else {}
            roles = list(entry.get("roles", [])) if isinstance(entry.get("roles"), list) else []
            source_id: Optional[str] = None
            if "away" in roles and self._away_team_id:
                source_id = self._away_team_id
            elif "home" in roles and self._home_team_id:
                source_id = self._home_team_id
            elif index < len(source_ids):
                source_id = source_ids[index]

            wins = int(record.get("wins", 0)) if isinstance(record, Mapping) else 0
            losses = int(record.get("losses", 0)) if isinstance(record, Mapping) else 0
            draws = int(record.get("draws", 0)) if isinstance(record, Mapping) else 0
            summary_label = None
            total = wins + losses + draws
            if total > 0:
                base = f"{wins}-{losses}"
                if draws:
                    base = f"{base}-{draws}"
                summary_label = base

            entry_payload = {
                "id": sim_key,
                "name": display_name,
                "record": record or {},
                "roles": roles,
            }
            if summary_label:
                entry_payload["summary"] = summary_label
            if source_id:
                entry_payload["source_id"] = source_id
            playable_entries.append(entry_payload)

            team_clone = deepcopy(team_obj)
            try:
                setattr(team_clone, "name", display_name)
            except Exception:
                pass
            playable_map[sim_key] = {
                "team": team_clone,
                "display_name": display_name,
                "source_id": source_id,
            }
            playable_records.append(
                {
                    "payload": entry_payload,
                    "map_entry": playable_map[sim_key],
                    "base_name": display_name,
                }
            )

        if playable_records:
            name_counts = Counter(
                record["base_name"] for record in playable_records if record.get("base_name")
            )
            duplicate_tracker: Dict[str, int] = defaultdict(int)
            for record in playable_records:
                payload = record["payload"]
                map_entry = record["map_entry"]
                base_name = record.get("base_name") or payload.get("id")
                if not base_name:
                    continue
                duplicate_tracker[base_name] += 1
                if name_counts.get(base_name, 0) > 1:
                    suffix_index = duplicate_tracker[base_name]
                    display_label = f"{base_name} #{suffix_index}"
                else:
                    display_label = base_name

                payload["name"] = display_label
                if map_entry:
                    map_entry["display_name"] = display_label
                    team_clone = map_entry.get("team")
                    if team_clone is not None:
                        try:
                            setattr(team_clone, "name", display_label)
                        except Exception:
                            pass

        default_selection = {"home": None, "away": None}
        for payload in playable_entries:
            roles = payload.get("roles") or []
            if not default_selection["away"] and "away" in roles:
                default_selection["away"] = payload["id"]
            if not default_selection["home"] and "home" in roles:
                default_selection["home"] = payload["id"]
        candidate_ids = [entry["id"] for entry in playable_entries if entry.get("id")]
        if not default_selection["away"] and candidate_ids:
            default_selection["away"] = candidate_ids[0]
        if not default_selection["home"]:
            for candidate in candidate_ids:
                if candidate != default_selection["away"]:
                    default_selection["home"] = candidate
                    break

        self._simulation_state["playable"] = {
            "teams": playable_entries,
            "selection": default_selection,
        }
        self._simulation_playable = playable_map

        self._simulation_state["last_run"] = summary

        total_games = summary.get("total_games") or num_games or 0
        if summary.get("mode") == "league":
            message = f"リーグシミュレーションが完了しました（{total_games}試合）。"
        else:
            message = f"{total_games}試合のシミュレーションが完了しました。"

        self._notifications.publish("success", message)

        return self.build_state()

    def start_simulation_match(
        self,
        *,
        away_key: str,
        home_key: str,
        control_mode: Optional[str] = None,
        user_team: Optional[str] = None,
    ) -> Dict[str, Any]:
        playable_map = getattr(self, "_simulation_playable", {}) or {}
        if not playable_map:
            raise GameSessionError("シミュレーション結果がありません。先にシミュレーションを実行してください。")

        away_selection = str(away_key or "").strip()
        home_selection = str(home_key or "").strip()

        if not away_selection or not home_selection:
            raise GameSessionError("ホームとアウェイのチームを選択してください。")
        if away_selection == home_selection:
            raise GameSessionError("ホームとアウェイには異なるチームを選択してください。")

        away_entry = playable_map.get(away_selection)
        home_entry = playable_map.get(home_selection)
        if not away_entry or not home_entry:
            raise GameSessionError("選択したシミュレーションチームが見つかりません。")

        away_template = away_entry.get("team")
        home_template = home_entry.get("team")
        if away_template is None or home_template is None:
            raise GameSessionError("シミュレーションチームの読み込みに失敗しました。")

        away_team = deepcopy(away_template)
        home_team = deepcopy(home_template)

        display_away = away_entry.get("display_name") or getattr(away_team, "name", away_selection)
        display_home = home_entry.get("display_name") or getattr(home_team, "name", home_selection)
        try:
            setattr(away_team, "name", display_away)
        except Exception:
            pass
        try:
            setattr(home_team, "name", display_home)
        except Exception:
            pass

        self.away_team = away_team
        self.home_team = home_team
        self._away_team_source = None
        self._home_team_source = None
        self._away_team_id = away_entry.get("source_id")
        self._home_team_id = home_entry.get("source_id")

        playable_state = self._simulation_state.get("playable") or {}
        playable_state["selection"] = {"home": home_selection, "away": away_selection}
        self._simulation_state["playable"] = playable_state

        if hasattr(self, "_prepare_game_setup"):
            try:
                self._prepare_game_setup(control_mode, user_team)  # type: ignore[attr-defined]
            except Exception:
                pass
        elif hasattr(self, "_configure_control_mode"):
            try:
                self._configure_control_mode(control_mode, user_team)  # type: ignore[attr-defined]
            except Exception:
                pass

        if hasattr(self, "_start_loaded_game"):
            return self._start_loaded_game()  # type: ignore[attr-defined]

        return self.build_state()
