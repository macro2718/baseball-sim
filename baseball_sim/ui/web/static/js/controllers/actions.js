import { CONFIG } from '../config.js';
import { elements } from '../dom.js';
import { stateCache, resetDefenseSelection, getDefensePlanInvalidAssignments } from '../state.js';
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
  async function handleStart(reload) {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.gameStart, {
        method: 'POST',
        body: JSON.stringify({ reload }),
      });
      render(payload);
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
      render(payload);
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

  async function handleSwing() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.gameSwing, { method: 'POST' });
      render(payload);
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function handleBunt() {
    try {
      const payload = await apiRequest(CONFIG.api.endpoints.gameBunt, { method: 'POST' });
      render(payload);
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
      showStatus('代打に出すベンチ選手を選択してください。', 'danger');
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
    } catch (error) {
      handleApiError(error, render);
    }
  }

  async function loadInitialState() {
    try {
      const initialState = await apiRequest(CONFIG.api.endpoints.gameState);
      render(initialState);
    } catch (error) {
      console.error(error);
      showStatus('初期状態の取得に失敗しました。ページを再読み込みしてください。', 'danger');
    }
  }

  return {
    handleStart,
    handleReloadTeams,
    handleReturnToTitle,
    handleClearLog,
    handleSwing,
    handleBunt,
    handlePinchHit,
    handleDefenseSubstitution,
    handleDefenseReset,
    handlePitcherChange,
    loadInitialState,
  };
}
