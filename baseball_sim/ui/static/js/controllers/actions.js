// Note: Player management uses stable IDs from backend to avoid name collisions.
// All player list/detail/save/delete requests should use `player_id` when available.
import { CONFIG } from '../config.js';
import { elements } from '../dom.js';
import {
  stateCache,
  resetDefenseSelection,
  getDefensePlanInvalidAssignments,
  setUIView,
  getPinchRunSelectedBase,
} from '../state.js';
import { apiRequest } from '../services/apiClient.js';
import { renderDefensePanel, updateDefenseSelectionInfo } from '../ui/defensePanel.js';
import { showStatus } from '../ui/status.js';

function handleApiError(error, render) {
  if (error?.payload?.state) {
    render(error.payload.state);
  }
  if (error?.payload?.error) {
    showStatus(error.payload.error, 'danger');
  } else if (error instanceof Error) {
    showStatus(error.message, 'danger');
  } else {
    showStatus('リクエスト中に不明なエラーが発生しました。', 'danger');
  }
  console.warn(error);
}

export function createGameActions(render) {
  const ANALYTICS_STATUS_MESSAGE = '統計計算中です。CPU解析が完了するまでお待ちください。';

  function setAnalyticsPendingUI(isPending) {
    const pending = Boolean(isPending);
    stateCache.analyticsPending = pending;

    if (elements.appShell) {
      elements.appShell.classList.toggle('analytics-pending', pending);
    }

    if (elements.gameScreen) {
      elements.gameScreen.classList.toggle('analytics-lock', pending);
      if (pending) {
        elements.gameScreen.setAttribute('aria-busy', 'true');
      } else {
        elements.gameScreen.removeAttribute('aria-busy');
      }
    }

    if (elements.actionWarning) {
      if (pending) {
        elements.actionWarning.dataset.analyticsMessage = 'true';
        elements.actionWarning.textContent = ANALYTICS_STATUS_MESSAGE;
      } else if (elements.actionWarning.dataset.analyticsMessage) {
        delete elements.actionWarning.dataset.analyticsMessage;
      }
    }
  }

  async function requestLiveAnalytics(options = {}) {
    if (stateCache.analyticsPending) {
      return null;
    }

    const { force = false, samples = null } = options || {};
    const latestData = stateCache.data || {};
    const gameState = options?.gameState || latestData.game || null;
    const hasGame = gameState && typeof gameState === 'object';
    const isPlayable = hasGame && (gameState.active || gameState.game_over);
    const isGameView = stateCache.uiView === 'game' || isPlayable;

    if (!force && !isGameView) {
      setAnalyticsPendingUI(false);
      return null;
    }

    if (!force && !isPlayable) {
      setAnalyticsPendingUI(false);
      return null;
    }

    const requestBody = {};
    if (Number.isFinite(samples) && samples > 0) {
      requestBody.samples = Math.max(1, Math.round(samples));
    }

    setAnalyticsPendingUI(true);
    try {
      const requestOptions = { method: 'POST' };
      if (Object.keys(requestBody).length > 0) {
        requestOptions.body = JSON.stringify(requestBody);
      }

      const payload = await apiRequest(CONFIG.api.endpoints.gameAnalytics, requestOptions);
      stateCache.analyticsPending = false;
      render(payload);
      return payload;
    } catch (error) {
      handleApiError(error, render);
      return null;
    } finally {
      setAnalyticsPendingUI(false);
    }
  }

  async function handleStart(options = {}) {
    const payloadBody = {
      reload: Boolean(options?.reload),
    };
    if (options?.mode) {
      payloadBody.mode = options.mode;
    }
    if (options?.userTeam) {
      payloadBody.user_team = options.userTeam;
    }

    try {
      const payload = await apiRequest(CONFIG.api.endpoints.gameStart, {
        method: 'POST',
        body: JSON.stringify(payloadBody),
      });
      render(payload);
      await requestLiveAnalytics();
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handleReloadTeams() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.reloadTeams, { method: 'POST' });
      render(payload);
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handleReturnToTitle() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.gameStop, { method: 'POST' });
      setUIView('title');
      render(payload);
      setAnalyticsPendingUI(false);
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handleClearLog() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.clearLog, { method: 'POST' });
      render(payload);
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handleRunSimulation(requestBody, options = {}) {
    if (!requestBody || typeof requestBody !== 'object') {
      showStatus('シミュレーション設定を解釈できませんでした。', 'danger');
      return null;
    }

    const body = { ...requestBody };
    if (Object.prototype.hasOwnProperty.call(body, 'games')) {
      const parsedGames = Number.parseInt(body.games, 10);
      if (!Number.isFinite(parsedGames) || parsedGames <= 0) {
        showStatus('試合数には1以上の整数を入力してください。', 'danger');
        return null;
      }
      body.games = parsedGames;
    }

    if (!body.league && !body.games) {
      showStatus('シミュレーション設定を確認してください。', 'danger');
      return null;
    }

    try {
      const payload = await apiRequest(CONFIG.api.endpoints.simulationRun, {
        method: 'POST',
        body: JSON.stringify(body),
      });
      if (options?.setView) {
        setUIView(options.setView);
      }
      render(payload);
      return payload;
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  async function handleSimulationStart(config = {}) {
    const leagueConfig = config?.league || {};
    const leagueTeamsRaw = Array.isArray(leagueConfig.teams)
      ? leagueConfig.teams.filter((teamId) => typeof teamId === 'string' && teamId.trim())
      : [];
    const leagueTeams = leagueTeamsRaw.map((teamId) => teamId.trim());
    const gamesPerCard = Number.parseInt(leagueConfig.games_per_card, 10);
    const cardsPerOpponent = Number.parseInt(leagueConfig.cards_per_opponent, 10);

    if (leagueTeams.length > 0) {
      if (leagueTeams.length < 2) {
        showStatus('リーグには2チーム以上の参加が必要です。', 'danger');
        return;
      }
      if (leagueTeams.length % 2 !== 0) {
        showStatus('リーグ参加チーム数は偶数にしてください。', 'danger');
        return;
      }

      if (!Number.isFinite(gamesPerCard) || gamesPerCard <= 0) {
        showStatus('1カードあたりの試合数は1以上の整数を入力してください。', 'danger');
        return;
      }
      if (!Number.isFinite(cardsPerOpponent) || cardsPerOpponent <= 0) {
        showStatus('対戦カード数は1以上の整数を入力してください。', 'danger');
        return;
      }

      const [homeTeamId, awayTeamId] = leagueTeams;
      try {
        await apiRequest(CONFIG.api.endpoints.teamSelect, {
          method: 'POST',
          body: JSON.stringify({ home: homeTeamId, away: awayTeamId }),
        });
      } catch (error) {
        handleApiError(error, render);
        throw error;
      }

      const leaguePayload = {
        teams: leagueTeams,
        games_per_card: gamesPerCard,
        cards_per_opponent: cardsPerOpponent,
        role_assignment: { home: 0, away: 1 },
      };

      await handleRunSimulation({ league: leaguePayload }, { setView: 'simulation-results' });
      return;
    }

    if (Object.prototype.hasOwnProperty.call(config, 'games')) {
      await handleRunSimulation({ games: config.games }, { setView: 'simulation-results' });
      return;
    }

    showStatus('リーグに参加するチームを2チーム以上追加してください。', 'danger');
  }

  async function handleSimulationMatchStart(selection = {}, options = {}) {
    const homeKey = typeof selection.home === 'string' ? selection.home.trim() : '';
    const awayKey = typeof selection.away === 'string' ? selection.away.trim() : '';

    if (!homeKey || !awayKey) {
      showStatus('ホームとアウェイのチームを選択してください。', 'danger');
      return null;
    }
    if (homeKey === awayKey) {
      showStatus('ホームとアウェイには異なるチームを選んでください。', 'danger');
      return null;
    }

    const payloadBody = { home: homeKey, away: awayKey };
    if (options?.mode) {
      payloadBody.mode = options.mode;
    }
    if (options?.userTeam) {
      payloadBody.user_team = options.userTeam;
    }

    try {
      const payload = await apiRequest(CONFIG.api.endpoints.simulationMatchStart, {
        method: 'POST',
        body: JSON.stringify(payloadBody),
      });
      setUIView('game');
      render(payload);
      await requestLiveAnalytics();
      return payload;
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  async function handleSwing() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.gameSwing, { method: 'POST' });
      render(payload);
      await requestLiveAnalytics();
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handleBunt() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.gameBunt, { method: 'POST' });
      render(payload);
      await requestLiveAnalytics();
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handleSqueeze() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.gameSqueeze, { method: 'POST' });
      render(payload);
      await requestLiveAnalytics();
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handleProgress() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.gameProgress, { method: 'POST' });
      render(payload);
      await requestLiveAnalytics();
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handleSteal() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.strategySteal, { method: 'POST' });
      render(payload);
      await requestLiveAnalytics();
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handlePinchHit() {
    if (!elements.pinchPlayer) return;
    const benchValue = elements.pinchPlayer.value;
    const lineupIndex = stateCache.currentBatterIndex;
    if (!Number.isInteger(lineupIndex)) {
      showStatus('現在の打者が見つかりません。', 'danger');
      return;
    }
    if (!benchValue) {
      showStatus('代打に出す選手をカード（またはリスト）から選択してください。', 'danger');
      return;
    }

    const benchIndex = Number(benchValue);
    if (Number.isNaN(benchIndex)) {
      showStatus('選択内容を解釈できませんでした。', 'danger');
      return;
    }

    try {
      const payload = await apiRequest(CONFIG.api.endpoints.pinchHit, {
        method: 'POST',
        body: JSON.stringify({ lineup_index: lineupIndex, bench_index: benchIndex }),
      });
      render(payload);
      await requestLiveAnalytics();
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handlePinchRun() {
    if (!elements.pinchRunPlayer) return;
    const baseIndex = getPinchRunSelectedBase();
    if (!Number.isInteger(baseIndex)) {
      showStatus('代走させる走者を選択してください。', 'danger');
      return;
    }

    const benchValue = elements.pinchRunPlayer.value;
    if (!benchValue) {
      showStatus('代走に出す選手をカード（またはリスト）から選択してください。', 'danger');
      return;
    }

    const benchIndex = Number(benchValue);
    if (Number.isNaN(benchIndex)) {
      showStatus('選択内容を解釈できませんでした。', 'danger');
      return;
    }

    try {
      const payload = await apiRequest(CONFIG.api.endpoints.pinchRun, {
        method: 'POST',
        body: JSON.stringify({ base_index: baseIndex, bench_index: benchIndex }),
      });
      render(payload);
      await requestLiveAnalytics();
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handleDefenseSubstitution() {
    if (!stateCache.defenseContext.canSub) {
      showStatus('守備交代は現在行えません。', 'danger');
      return;
    }

    const plan = stateCache.defensePlan;
    const operations = plan?.operations || [];
    if (!plan || operations.length === 0) {
      showStatus('守備交代の変更内容が選択されていません。', 'danger');
      return;
    }

    const invalidAssignments = getDefensePlanInvalidAssignments(plan);
    let force = false;
    if (invalidAssignments.length > 0) {
      const details = invalidAssignments
        .map((entry) => `・${entry.player?.name ?? '-'} (${entry.positionLabel ?? entry.positionKey ?? '-'})`)
        .join('\n');
      const confirmMessage =
        '⚠️ 守備適性のない守備配置が含まれています。' +
        (details ? `\n${details}` : '') +
        '\nこのまま強制的に適用しますか？';
      const confirmed = window.confirm(confirmMessage);
      if (!confirmed) {
        showStatus('守備交代の適用をキャンセルしました。', 'warning');
        return;
      }
      force = true;
      showStatus('守備適性のない守備位置を含めたまま守備交代を適用します。', 'warning');
    }

    const swaps = operations
      .map((op) => {
        if (op.type === 'lineup_lineup') {
          return {
            a: { group: 'lineup', index: op.lineupIndexA },
            b: { group: 'lineup', index: op.lineupIndexB },
          };
        }
        if (op.type === 'bench_lineup') {
          return {
            a: { group: 'bench', index: op.benchIndex },
            b: { group: 'lineup', index: op.lineupIndex },
          };
        }
        return null;
      })
      .filter(Boolean);

    if (!swaps.length) {
      showStatus('守備交代の指示が正しく選択されていません。', 'danger');
      return;
    }

    try {
      const payload = await apiRequest(CONFIG.api.endpoints.defenseSubstitution, {
        method: 'POST',
        body: JSON.stringify({ swaps, force }),
      });
      stateCache.defensePlan = null;
      resetDefenseSelection();
      render(payload);
      await requestLiveAnalytics();
    } catch (error) {
      handleApiError(error, render);
    }
  }

  function handleDefenseReset() {
    const data = stateCache.data;
    const gameState = data?.game;
    const teams = data?.teams || {};
    const defenseKey = gameState?.defense;
    const defenseTeam = defenseKey ? teams[defenseKey] : null;

    stateCache.defensePlan = null;
    resetDefenseSelection();

    if (!defenseTeam) {
      updateDefenseSelectionInfo();
      return;
    }

    renderDefensePanel(defenseTeam, gameState);
    updateDefenseSelectionInfo();
  }

  async function handlePitcherChange() {
    if (!elements.pitcherSelect) return;
    const pitcherValue = elements.pitcherSelect.value;
    if (!pitcherValue) {
      showStatus('交代する投手を選択してください。', 'danger');
      return;
    }

    const pitcherIndex = Number(pitcherValue);
    if (Number.isNaN(pitcherIndex)) {
      showStatus('選択内容を解釈できませんでした。', 'danger');
      return;
    }

    try {
      const payload = await apiRequest(CONFIG.api.endpoints.changePitcher, {
        method: 'POST',
        body: JSON.stringify({ pitcher_index: pitcherIndex }),
      });
      render(payload);
      await requestLiveAnalytics();
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handleLineupUpdate(teamKey, lineupEntries) {
    const normalizedTeam = teamKey === 'home' ? 'home' : teamKey === 'away' ? 'away' : null;
    if (!normalizedTeam) {
      showStatus('スタメンを更新するチームを正しく選択してください。', 'danger');
      return null;
    }

    if (!Array.isArray(lineupEntries) || !lineupEntries.length) {
      showStatus('スタメンの候補が見つかりません。', 'danger');
      return null;
    }

    try {
      const payload = await apiRequest(CONFIG.api.endpoints.teamSetLineup, {
        method: 'POST',
        body: JSON.stringify({ team: normalizedTeam, lineup: lineupEntries }),
      });
      render(payload);
      return payload;
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  async function handleSetStartingPitcher(teamKey, pitcherName) {
    const normalizedTeam = teamKey === 'home' ? 'home' : teamKey === 'away' ? 'away' : null;
    if (!normalizedTeam) {
      showStatus('先発投手を設定するチームを正しく選択してください。', 'danger');
      return null;
    }

    const normalizedName = typeof pitcherName === 'string' ? pitcherName.trim() : '';
    if (!normalizedName) {
      showStatus('先発投手を選択してください。', 'danger');
      return null;
    }

    try {
      const payload = await apiRequest(CONFIG.api.endpoints.teamSetPitcher, {
        method: 'POST',
        body: JSON.stringify({ team: normalizedTeam, pitcher: normalizedName }),
      });
      render(payload);
      return payload;
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  async function loadInitialState() {
    try {
      const initialState = await apiRequest(CONFIG.api.endpoints.gameState);
      render(initialState);
      await requestLiveAnalytics({ gameState: initialState?.game || null });
    } catch (error) {
      console.error(error);
      showStatus('初期状態の取得に失敗しました。ページを再読み込みしてください。', 'danger');
    }
  }

  async function handleTeamSelection(homeId, awayId) {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.teamSelect, {
        method: 'POST',
        body: JSON.stringify({ home: homeId, away: awayId }),
      });
      setUIView('title');
      render(payload);
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  async function fetchTeamDefinition(teamId) {
    if (!teamId) return null;
    const endpoint = `${CONFIG.api.endpoints.teamDetail}/${encodeURIComponent(teamId)}`;
    try {
      const payload = await apiRequest(endpoint);
      return payload?.team || null;
    } catch (error) {
      handleApiError(error, render);
      return null;
    }
  }

  async function fetchPlayersCatalog() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.playersCatalog);
      return {
        batters: Array.isArray(payload?.batters) ? payload.batters : [],
        pitchers: Array.isArray(payload?.pitchers) ? payload.pitchers : [],
      };
    } catch (error) {
      handleApiError(error, render);
      return { batters: [], pitchers: [] };
    }
  }

  async function handleTeamSave(teamId, teamData) {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.teamSave, {
        method: 'POST',
        body: JSON.stringify({ team_id: teamId, team: teamData }),
      });
      if (payload?.state) {
        render(payload.state);
      } else {
        render(payload);
      }
      return payload?.team_id ?? teamId ?? null;
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  async function handleTeamDelete(teamId) {
    if (!teamId) {
      showStatus('削除するチームを選択してください。', 'danger');
      return null;
    }
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.teamDelete, {
        method: 'POST',
        body: JSON.stringify({ team_id: teamId }),
      });
      render(payload);
      return teamId;
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  async function fetchPlayersList(role, { folder = '' } = {}) {
    const params = new URLSearchParams();
    params.set('role', String(role || 'batter'));
    if (folder && typeof folder === 'string' && folder.trim()) {
      params.set('folder', folder.trim());
    }
    const endpoint = `${CONFIG.api.endpoints.playersList}?${params.toString()}`;
    try {
      const payload = await apiRequest(endpoint);
      const list = Array.isArray(payload?.players) ? payload.players : [];
      // Normalize: expect { id, name }
      return list
        .map((p) => (p && typeof p === 'object' ? { id: String(p.id || ''), name: String(p.name || '') } : null))
        .filter((p) => p && p.id && p.name);
    } catch (error) {
      handleApiError(error, render);
      return [];
    }
  }

  async function fetchPlayerFolders() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.playerFolders);
      return Array.isArray(payload?.folders) ? payload.folders.filter((f) => typeof f === 'string' && f.trim()).map((f) => f.trim()) : [];
    } catch (error) {
      handleApiError(error, render);
      return [];
    }
  }

  async function createPlayerFolder(name) {
    const cleaned = String(name || '').trim();
    if (!cleaned) return null;
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.playerFolderCreate, {
        method: 'POST',
        body: JSON.stringify({ name: cleaned }),
      });
      const folders = Array.isArray(payload?.folders) ? payload.folders : [];
      return folders;
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  async function deletePlayerFolder(name) {
    const cleaned = String(name || '').trim();
    if (!cleaned) return null;
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.playerFolderDelete, {
        method: 'POST',
        body: JSON.stringify({ name: cleaned }),
      });
      const folders = Array.isArray(payload?.folders) ? payload.folders : [];
      return folders;
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  async function renamePlayerFolder(oldName, newName) {
    const oldClean = String(oldName || '').trim();
    const newClean = String(newName || '').trim();
    if (!oldClean || !newClean) return null;
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.playerFolderRename, {
        method: 'POST',
        body: JSON.stringify({ old: oldClean, new: newClean }),
      });
      const folders = Array.isArray(payload?.folders) ? payload.folders : [];
      return folders;
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  async function fetchPlayerDefinition(idOrName) {
    if (!idOrName) return null;
    // Prefer id query param for precision; fall back to name for compatibility
    const paramKey = /^[0-9a-fA-F-]{36}$/.test(String(idOrName)) ? 'id' : 'name';
    const endpoint = `${CONFIG.api.endpoints.playerDetail}?${paramKey}=${encodeURIComponent(idOrName)}`;
    try {
      const payload = await apiRequest(endpoint);
      if (payload && typeof payload === 'object') {
        return {
          player: payload.player || null,
          role: payload.role || null,
          hasReferences: Boolean(payload.has_references),
          referencedBy: Array.isArray(payload.referenced_by) ? payload.referenced_by : [],
        };
      }
      return { player: null, role: null, hasReferences: false, referencedBy: [] };
    } catch (error) {
      handleApiError(error, render);
      return { player: null, role: null, hasReferences: false, referencedBy: [] };
    }
  }

  async function handlePlayerSave(playerData, role, originalName = null, playerId = null) {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.playerSave, {
        method: 'POST',
        body: JSON.stringify({ player: playerData, role, original_name: originalName, player_id: playerId }),
      });
      // After save, re-render state so any dependent UI updates (if any) happen
      if (payload?.state) {
        render(payload.state);
      }
      // Return id and name for caller convenience
      return { id: payload?.id || playerId || null, name: payload?.name || playerData?.name || null };
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  async function handlePlayerDelete(playerId, role, fallbackName = null) {
    if (!playerId && !fallbackName) {
      showStatus('削除する選手を選択してください。', 'danger');
      return null;
    }
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.playerDelete, {
        method: 'POST',
        body: JSON.stringify({ player_id: playerId, role, name: fallbackName }),
      });
      if (payload?.state) {
        render(payload.state);
      } else {
        render(payload);
      }
      return playerId || fallbackName;
    } catch (error) {
      handleApiError(error, render);
      throw error;
    }
  }

  return {
    requestLiveAnalytics,
    handleStart,
    handleReloadTeams,
    handleReturnToTitle,
    handleClearLog,
    startSimulation: handleSimulationStart,
    handleSimulationMatchStart,
    handleSwing,
    handleBunt,
    handleSqueeze,
    handleProgress,
    handleSteal,
    handlePinchHit,
    handlePinchRun,
    handleDefenseSubstitution,
    handleDefenseReset,
    handlePitcherChange,
    handleLineupUpdate,
    handleSetStartingPitcher,
    loadInitialState,
    handleTeamSelection,
    fetchTeamDefinition,
    fetchPlayersCatalog,
    handleTeamSave,
    fetchPlayersList,
    fetchPlayerDefinition,
    handlePlayerSave,
    handleTeamDelete,
    handlePlayerDelete,
    fetchPlayerFolders,
    createPlayerFolder,
    deletePlayerFolder,
    renamePlayerFolder,
  };
}
