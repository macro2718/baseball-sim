import { elements } from '../dom.js';
import {
  stateCache,
  normalizePositionKey,
  canBenchPlayerCoverPosition,
  updateDefenseContext,
  resetDefenseSelection,
  getDefensePlanInvalidAssignments,
} from '../state.js';
import { escapeHtml, renderPositionList, renderPositionToken } from '../utils.js';

const DEFAULT_INFO = '出場選手のポジションや選手名、ベンチ選手をクリックして入れ替えを指示できます。';

function buildPlanSignature(team) {
  if (!team) return null;
  const lineupSig = (team.lineup || [])
    .map((player) => {
      const key = normalizePositionKey(player?.position_key || player?.position) || '-';
      return `${player?.index ?? ''}:${player?.name ?? ''}:${key}`;
    })
    .join('|');
  const benchSig = (team.bench || [])
    .map((player) => `${player?.index ?? ''}:${player?.name ?? ''}`)
    .join('|');
  const retiredSig = (team.retired || [])
    .map((player) => player?.name ?? '')
    .join('|');
  return `${team.name || ''}::${lineupSig}::${benchSig}::${retiredSig}`;
}

function cloneLineupPlayer(player, index) {
  const clone = {
    ...player,
    index,
    order: index + 1,
  };
  const positionKey = normalizePositionKey(player?.position_key || player?.position);
  clone.position_key = positionKey;
  clone.position = player?.position ?? positionKey ?? '-';
  const eligible = Array.isArray(player?.eligible) ? [...player.eligible] : [];
  const eligibleAll = Array.isArray(player?.eligible_all)
    ? [...player.eligible_all]
    : eligible.map((pos) => String(pos).toUpperCase());
  clone.eligible = eligible;
  clone.eligible_all = eligibleAll;
  return clone;
}

function cloneBenchPlayer(player, index) {
  const clone = {
    ...player,
    index,
  };
  const eligible = Array.isArray(player?.eligible) ? [...player.eligible] : [];
  const eligibleAll = Array.isArray(player?.eligible_all)
    ? [...player.eligible_all]
    : eligible.map((pos) => String(pos).toUpperCase());
  clone.eligible = eligible;
  clone.eligible_all = eligibleAll;
  return clone;
}

function cloneRetiredPlayer(player) {
  const clone = {
    ...player,
  };
  const eligible = Array.isArray(player?.eligible) ? [...player.eligible] : [];
  const eligibleAll = Array.isArray(player?.eligible_all)
    ? [...player.eligible_all]
    : eligible.map((pos) => String(pos).toUpperCase());
  clone.eligible = eligible;
  clone.eligible_all = eligibleAll;
  clone.position = player?.position ?? player?.position_key ?? '-';
  clone.position_key = normalizePositionKey(player?.position_key || player?.position);
  return clone;
}

function clearDefensePanels() {
  if (elements.defenseField) {
    elements.defenseField.innerHTML = '';
  }
  if (elements.defenseBench) {
    elements.defenseBench.innerHTML = '';
  }
  if (elements.defenseExtras) {
    elements.defenseExtras.classList.add('hidden');
    elements.defenseExtras.innerHTML = '';
  }
  if (elements.defenseRetired) {
    elements.defenseRetired.classList.add('hidden');
    elements.defenseRetired.innerHTML = '';
  }
}

function reindexPlanPlayers(plan) {
  if (!plan) return;
  plan.lineup.forEach((player, index) => {
    if (!player) return;
    player.index = index;
    player.order = index + 1;
  });
  plan.bench.forEach((player, index) => {
    if (!player) return;
    player.index = index;
  });
}

function updateDefensePlanContext(plan) {
  const lineupMap = {};
  const benchMap = {};
  (plan?.lineup || []).forEach((player, index) => {
    if (player) {
      lineupMap[index] = player;
    }
  });
  (plan?.bench || []).forEach((player, index) => {
    if (player) {
      benchMap[index] = player;
    }
  });
  updateDefenseContext(lineupMap, benchMap, stateCache.defenseContext.canSub);
}

function ensureDefensePlan(defenseTeam) {
  if (!defenseTeam) {
    stateCache.defensePlan = null;
    resetDefenseSelection();
    updateDefenseContext({}, {}, stateCache.defenseContext.canSub);
    clearDefensePanels();
    return null;
  }

  const signature = buildPlanSignature(defenseTeam);
  if (!stateCache.defensePlan || stateCache.defensePlan.signature !== signature) {
    const lineup = (defenseTeam.lineup || []).map((player, index) =>
      cloneLineupPlayer(player, index),
    );
    const bench = (defenseTeam.bench || []).map((player, index) =>
      cloneBenchPlayer(player, index),
    );
    const retired = (defenseTeam.retired || []).map((player) => cloneRetiredPlayer(player));

    stateCache.defensePlan = {
      signature,
      teamName: defenseTeam.name || '',
      lineup,
      bench,
      retired,
      operations: [],
    };
    resetDefenseSelection();
    reindexPlanPlayers(stateCache.defensePlan);
    updateDefensePlanContext(stateCache.defensePlan);
  } else {
    updateDefensePlanContext(stateCache.defensePlan);
  }

  return stateCache.defensePlan;
}

function getCurrentGameState(gameState) {
  if (gameState && typeof gameState === 'object') {
    return gameState;
  }
  return stateCache.data?.game || {};
}

function renderDefensePlanView(plan, gameState) {
  const lineup = plan?.lineup || [];
  const benchPlayers = plan?.bench || [];
  const activeGame = Boolean(gameState?.active);
  const canInteract = activeGame && stateCache.defenseContext.canSub;

  if (elements.defenseField) {
    elements.defenseField.innerHTML = '';
    if (!lineup.length) {
      const empty = document.createElement('p');
      empty.className = 'empty-message';
      empty.textContent = '出場中の選手情報がありません。';
      elements.defenseField.appendChild(empty);
    } else {
      lineup.forEach((player, index) => {
        if (!player) return;
        const row = document.createElement('div');
        row.className = 'defense-lineup-row';
        row.dataset.index = String(index);
        row.dataset.lineupIndex = String(index);

        const order = document.createElement('span');
        order.className = 'lineup-order';
        order.textContent = `${player.order ?? index + 1}.`;

        const positionButton = document.createElement('button');
        positionButton.type = 'button';
        positionButton.className = 'defense-action-button lineup-position-button';
        positionButton.dataset.role = 'lineup';
        positionButton.dataset.kind = 'position';
        positionButton.dataset.index = String(index);
        const displayPosition = player.position || player.position_key || '-';
        const positionToken = renderPositionToken(displayPosition, player.pitcher_type, 'position-token');
        positionButton.innerHTML = positionToken || escapeHtml(displayPosition);
        positionButton.disabled = !canInteract;

        const playerButton = document.createElement('button');
        playerButton.type = 'button';
        playerButton.className = 'defense-action-button lineup-player-button';
        playerButton.dataset.role = 'lineup';
        playerButton.dataset.kind = 'player';
        playerButton.dataset.index = String(index);
        playerButton.innerHTML = `<span>${escapeHtml(player.name ?? '-')}</span>`;
        playerButton.disabled = !canInteract;

        const meta = document.createElement('div');
        meta.className = 'lineup-meta';
        const label = document.createElement('span');
        label.className = 'eligible-label';
        label.textContent = '守備適性';
        const eligible = document.createElement('span');
        eligible.className = 'eligible-positions';
        eligible.innerHTML = renderPositionList(player.eligible || [], player.pitcher_type);
        meta.append(label, eligible);

        row.append(order, positionButton, playerButton, meta);
        elements.defenseField.appendChild(row);
      });
    }
  }

  if (elements.defenseBench) {
    elements.defenseBench.innerHTML = '';
    if (!benchPlayers.length) {
      const empty = document.createElement('p');
      empty.className = 'empty-message';
      empty.textContent = 'ベンチに交代可能な選手がいません。';
      elements.defenseBench.appendChild(empty);
    } else {
      benchPlayers.forEach((player) => {
        if (!player) return;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'defense-action-button bench-player-button';
        button.dataset.role = 'bench';
        button.dataset.kind = 'player';
        button.dataset.index = String(player.index);
        button.dataset.benchIndex = String(player.index);
        const eligibleHtml = renderPositionList(player.eligible || [], player.pitcher_type);
        button.innerHTML = `
          <strong>${escapeHtml(player.name ?? '-')}</strong>
          <span class="eligible-label">守備適性</span>
          <span class="eligible-positions">${eligibleHtml}</span>
        `;
        button.disabled = !stateCache.defenseContext.canSub;
        elements.defenseBench.appendChild(button);
      });
    }
  }

  if (elements.defenseExtras) {
    elements.defenseExtras.classList.add('hidden');
    elements.defenseExtras.innerHTML = '';
  }

  if (elements.defenseRetired) {
    elements.defenseRetired.classList.add('hidden');
    elements.defenseRetired.innerHTML = '';
  }
}

function setDefenseFeedback(message, variant = 'info') {
  if (message) {
    stateCache.defenseSelection.feedback = { message, variant };
  } else {
    stateCache.defenseSelection.feedback = null;
  }
}

function getPlanLineupPlayer(index) {
  const plan = stateCache.defensePlan;
  if (!plan) return null;
  if (!Number.isInteger(index) || index < 0 || index >= plan.lineup.length) return null;
  return plan.lineup[index] || null;
}

function getPlanBenchPlayer(index) {
  const plan = stateCache.defensePlan;
  if (!plan) return null;
  if (!Number.isInteger(index) || index < 0 || index >= plan.bench.length) return null;
  return plan.bench[index] || null;
}

function swapLineupPositions(plan, indexA, indexB) {
  if (!Number.isInteger(indexA) || !Number.isInteger(indexB)) {
    return { success: false, message: '選択された守備位置が無効です。', variant: 'danger' };
  }
  if (indexA === indexB) {
    return { success: false, message: null };
  }

  const playerA = getPlanLineupPlayer(indexA);
  const playerB = getPlanLineupPlayer(indexB);
  if (!playerA || !playerB) {
    return { success: false, message: '守備位置の情報を取得できませんでした。', variant: 'danger' };
  }

  const posAKey = normalizePositionKey(playerA.position_key || playerA.position);
  const posBKey = normalizePositionKey(playerB.position_key || playerB.position);
  const displayPosA = playerA.position || posAKey || '-';
  const displayPosB = playerB.position || posBKey || '-';

  const playerACanCover = !posBKey || canBenchPlayerCoverPosition(playerA, posBKey);
  const playerBCanCover = !posAKey || canBenchPlayerCoverPosition(playerB, posAKey);

  playerA.position = displayPosB;
  playerA.position_key = posBKey;
  playerB.position = displayPosA;
  playerB.position_key = posAKey;

  plan.operations.push({
    type: 'lineup_lineup',
    lineupIndexA: indexA,
    lineupIndexB: indexB,
  });

  let message = `${playerA.name} と ${playerB.name} の守備位置を入れ替える案を追加しました。`;
  const warnings = [];
  if (!playerACanCover) {
    warnings.push(`${playerA.name} は ${displayPosB} を守れません`);
  }
  if (!playerBCanCover) {
    warnings.push(`${playerB.name} は ${displayPosA} を守れません`);
  }

  if (warnings.length) {
    message += ` ただし ${warnings.join('、')}。`;
  }

  return {
    success: true,
    message,
    variant: warnings.length ? 'warning' : 'success',
  };
}

function swapBenchWithLineup(plan, lineupIndex, benchIndex) {
  const lineupPlayer = getPlanLineupPlayer(lineupIndex);
  const benchPlayer = getPlanBenchPlayer(benchIndex);
  if (!lineupPlayer || !benchPlayer) {
    return {
      success: false,
      message: '交代対象の選手情報を取得できませんでした。',
      variant: 'danger',
    };
  }

  const positionKey = normalizePositionKey(lineupPlayer.position_key || lineupPlayer.position);
  const positionLabel = lineupPlayer.position || positionKey || '守備位置';

  const canCover = !positionKey || canBenchPlayerCoverPosition(benchPlayer, positionKey);

  const entering = cloneBenchPlayer(benchPlayer, lineupIndex);
  entering.order = lineupIndex + 1;
  entering.position_key = positionKey;
  entering.position = positionLabel;

  plan.bench.splice(benchIndex, 1);
  plan.lineup[lineupIndex] = entering;

  const retiredEntry = cloneRetiredPlayer(lineupPlayer);
  if (retiredEntry.name) {
    plan.retired.push(retiredEntry);
  }

  plan.operations.push({
    type: 'bench_lineup',
    lineupIndex,
    benchIndex,
  });

  let message = `${entering.name} を ${positionLabel} に投入する案を追加しました（${lineupPlayer.name} はリタイア）。`;
  if (!canCover) {
    message += ` ただし ${entering.name} は ${positionLabel} を守れません。`;
  }

  return {
    success: true,
    message,
    variant: canCover ? 'success' : 'warning',
  };
}

function normalizeDefenseSelection(selection) {
  if (!selection) return null;
  const { role, kind } = selection;
  const index = Number(selection.index);
  if (!Number.isInteger(index) || index < 0) {
    return null;
  }

  if (kind === 'position') {
    return { type: 'lineup_position', lineupIndex: index };
  }

  if (kind === 'player') {
    if (role === 'lineup') {
      return { type: 'lineup_player', lineupIndex: index };
    }
    if (role === 'bench') {
      return { type: 'bench_player', benchIndex: index };
    }
  }

  return null;
}

function applyDefenseSwap(first, second) {
  if (!stateCache.defenseContext.canSub) {
    return { success: false, message: '守備交代は現在行えません。', variant: 'danger' };
  }
  const plan = stateCache.defensePlan;
  if (!plan) {
    return { success: false, message: '守備情報がありません。', variant: 'danger' };
  }

  if (!first || !second) {
    return { success: false, message: null };
  }

  const firstSelection = normalizeDefenseSelection(first);
  const secondSelection = normalizeDefenseSelection(second);

  if (!firstSelection || !secondSelection) {
    return {
      success: false,
      message: '選択した組み合わせでは入れ替えできません。',
      variant: 'danger',
    };
  }

  const firstLineupIndex = Number.isInteger(firstSelection.lineupIndex)
    ? firstSelection.lineupIndex
    : null;
  const secondLineupIndex = Number.isInteger(secondSelection.lineupIndex)
    ? secondSelection.lineupIndex
    : null;

  if (
    firstLineupIndex !== null &&
    secondLineupIndex !== null &&
    firstLineupIndex === secondLineupIndex
  ) {
    return { success: false, message: null };
  }

  if (firstSelection.type === 'bench_player' && secondSelection.type === 'bench_player') {
    return {
      success: false,
      message: 'ベンチ同士の入れ替えは行えません。',
      variant: 'warning',
    };
  }

  if (firstLineupIndex !== null && secondLineupIndex !== null) {
    return swapLineupPositions(plan, firstLineupIndex, secondLineupIndex);
  }

  if (firstSelection.type === 'bench_player' && secondLineupIndex !== null) {
    return swapBenchWithLineup(plan, secondLineupIndex, firstSelection.benchIndex);
  }

  if (secondSelection.type === 'bench_player' && firstLineupIndex !== null) {
    return swapBenchWithLineup(plan, firstLineupIndex, secondSelection.benchIndex);
  }

  return {
    success: false,
    message: '選択した組み合わせでは入れ替えできません。',
    variant: 'danger',
  };
}

function commitDefensePlanChange(plan, gameState) {
  reindexPlanPlayers(plan);
  updateDefensePlanContext(plan);
  renderDefensePlanView(plan, gameState);
}

export function renderDefensePanel(defenseTeam, gameState) {
  const plan = ensureDefensePlan(defenseTeam);
  if (!plan) {
    return;
  }
  const currentGame = getCurrentGameState(gameState);
  renderDefensePlanView(plan, currentGame);
}

export function updateDefenseBenchAvailability() {
  const plan = stateCache.defensePlan;
  const selection = stateCache.defenseSelection.first;
  const canSubBase = stateCache.defenseContext.canSub;

  const lineupSelectionIndex =
    selection && selection.role === 'lineup' ? Number(selection.index) : null;
  const lineupPlayer =
    plan && Number.isInteger(lineupSelectionIndex) ? plan.lineup[lineupSelectionIndex] : null;
  const lineupPositionKey = lineupPlayer
    ? normalizePositionKey(lineupPlayer.position_key || lineupPlayer.position)
    : null;
  const lineupPositionLabel = lineupPlayer ? lineupPlayer.position || lineupPositionKey || '' : '';

  const benchSelectionIndex =
    selection && selection.role === 'bench' && selection.kind === 'player'
      ? Number(selection.index)
      : null;
  const benchSelectionPlayer =
    plan && Number.isInteger(benchSelectionIndex) ? plan.bench[benchSelectionIndex] : null;

  if (elements.defenseBench) {
    elements.defenseBench
      .querySelectorAll('[data-role="bench"][data-kind="player"]')
      .forEach((button) => {
        const value = button.dataset.index ?? button.dataset.benchIndex;
        const benchIndex = Number(value);
        const benchPlayer =
          plan && Number.isInteger(benchIndex) ? plan.bench[benchIndex] : null;

        const enable = canSubBase && Boolean(benchPlayer);
        let markIneligible = false;
        let title = '';

        if (
          enable &&
          lineupPlayer &&
          lineupPositionKey &&
          benchPlayer &&
          !canBenchPlayerCoverPosition(benchPlayer, lineupPositionKey)
        ) {
          markIneligible = true;
          title = lineupPositionLabel
            ? `${benchPlayer.name} は ${lineupPositionLabel} を守れません。`
            : `${benchPlayer.name} はこの守備位置を守れません。`;
        }

        button.disabled = !enable;
        button.classList.toggle('ineligible', markIneligible);
        button.title = title;
      });
  }

  if (elements.defenseField) {
    elements.defenseField
      .querySelectorAll('[data-role="lineup"][data-kind="position"]')
      .forEach((button) => {
        const value = button.dataset.index ?? button.dataset.lineupIndex;
        const lineupIndex = Number(value);
        const rowPlayer =
          plan && Number.isInteger(lineupIndex) ? plan.lineup[lineupIndex] : null;
        const posKey = rowPlayer
          ? normalizePositionKey(rowPlayer.position_key || rowPlayer.position)
          : null;
        const posLabel = rowPlayer ? rowPlayer.position || posKey || '' : '';

        let markIneligible = false;
        let title = '';

        if (benchSelectionPlayer && posKey && !canBenchPlayerCoverPosition(benchSelectionPlayer, posKey)) {
          markIneligible = true;
          title = posLabel
            ? `${benchSelectionPlayer.name} は ${posLabel} を守れません。`
            : `${benchSelectionPlayer.name} はこの守備位置を守れません。`;
        }

        button.classList.toggle('ineligible', markIneligible);
        button.title = title;
      });
  }
}

export function applyDefenseSelectionHighlights() {
  const selection = stateCache.defenseSelection.first;
  const lineupIndex =
    selection && selection.role === 'lineup' ? Number(selection.index) : null;
  const benchIndex =
    selection && selection.role === 'bench' && selection.kind === 'player'
      ? Number(selection.index)
      : null;

  if (elements.defenseField) {
    elements.defenseField
      .querySelectorAll('[data-role="lineup"][data-kind="position"]')
      .forEach((button) => {
        const value = button.dataset.index ?? button.dataset.lineupIndex;
        const index = Number(value);
        const isSelected =
          selection &&
          selection.role === 'lineup' &&
          selection.kind === 'position' &&
          Number.isInteger(index) &&
          index === lineupIndex;
        button.classList.toggle('selected', isSelected);
      });

    elements.defenseField
      .querySelectorAll('[data-role="lineup"][data-kind="player"]')
      .forEach((button) => {
        const value = button.dataset.index ?? button.dataset.lineupIndex;
        const index = Number(value);
        const isSelected =
          selection &&
          selection.role === 'lineup' &&
          selection.kind === 'player' &&
          Number.isInteger(index) &&
          index === lineupIndex;
        button.classList.toggle('selected', isSelected);
      });

    elements.defenseField.querySelectorAll('.defense-lineup-row').forEach((row) => {
      const value = row.dataset.index ?? row.dataset.lineupIndex;
      const index = Number(value);
      const highlight =
        selection && selection.role === 'lineup' && Number.isInteger(index) && index === lineupIndex;
      row.classList.toggle('selected', highlight);
    });
  }

  if (elements.defenseBench) {
    elements.defenseBench
      .querySelectorAll('[data-role="bench"][data-kind="player"]')
      .forEach((button) => {
        const value = button.dataset.index ?? button.dataset.benchIndex;
        const index = Number(value);
        const isSelected =
          selection &&
          selection.role === 'bench' &&
          selection.kind === 'player' &&
          Number.isInteger(index) &&
          index === benchIndex;
        button.classList.toggle('selected', isSelected);
      });
  }
}

export function updateDefenseSelectionInfo() {
  updateDefenseBenchAvailability();

  const infoEl = elements.defenseSelectionInfo;
  if (!infoEl) return;

  const plan = stateCache.defensePlan;
  const selection = stateCache.defenseSelection.first;
  const feedback = stateCache.defenseSelection.feedback;
  const canSub = stateCache.defenseContext.canSub;
  const operationsCount = plan?.operations?.length || 0;
  const invalidAssignments = getDefensePlanInvalidAssignments(plan);

  infoEl.classList.remove('success', 'danger', 'warning');

  let message = DEFAULT_INFO;
  let variantClass = null;

  if (feedback?.message) {
    message = feedback.message;
    variantClass = feedback.variant && feedback.variant !== 'info' ? feedback.variant : null;
  } else if (!canSub) {
    message = '守備交代を行える状況ではありません。';
    variantClass = 'warning';
  } else if (!plan || !(plan.lineup || []).length) {
    message = '守備情報がありません。';
  } else if (selection?.role === 'lineup') {
    const index = Number(selection.index);
    const player = Number.isInteger(index) ? getPlanLineupPlayer(index) : null;
    const positionKey = player
      ? normalizePositionKey(player.position_key || player.position)
      : null;
    const positionLabel = player ? player.position || positionKey || '-' : '-';
    if (selection.kind === 'position') {
      message = player
        ? `${player.name} の守備位置（${positionLabel}）と入れ替える対象を選択してください。`
        : '守備位置を入れ替える対象を選択してください。';
    } else {
      message = player
        ? `${player.name} と入れ替える選手または守備位置を選択してください。`
        : '守備交代を行う守備位置を選択してください。';
    }
  } else if (selection?.role === 'bench') {
    const index = Number(selection.index);
    const player = Number.isInteger(index) ? getPlanBenchPlayer(index) : null;
    message = player
      ? `${player.name} を投入する守備位置や入れ替える選手を選択してください。`
      : '守備交代を行う守備位置を選択してください。';
  } else if (operationsCount > 0) {
    message = `未適用の守備交代案が ${operationsCount} 件あります。適用ボタンで確定してください。`;
    if (invalidAssignments.length) {
      message += ' ⚠️ 一部の選手は選択した守備位置を守れません。';
    }
    variantClass = 'warning';
  }

  infoEl.textContent = message;
  if (variantClass) {
    infoEl.classList.add(variantClass);
  }

  const canApply = canSub && operationsCount > 0;
  if (elements.defenseApplyButton) {
    elements.defenseApplyButton.disabled = !canApply;
  }

  const canReset = Boolean(plan) && operationsCount > 0;
  if (elements.defenseResetButton) {
    elements.defenseResetButton.disabled = !canReset;
  }

  applyDefenseSelectionHighlights();
}

export function handleDefensePlayerClick(event) {
  const button = event.target.closest('[data-role]');
  if (!button || button.disabled) return;

  const role = button.dataset.role;
  const kind = button.dataset.kind || 'player';
  if (!role || (role !== 'lineup' && role !== 'bench')) return;
  if (kind !== 'player' && kind !== 'position') return;

  const rawValue = button.dataset.index ?? button.dataset.lineupIndex ?? button.dataset.benchIndex;
  const index = Number(rawValue);
  if (!Number.isInteger(index) || index < 0) return;

  const selection = { role, kind, index };

  const currentSelection = stateCache.defenseSelection.first;

  if (
    currentSelection &&
    currentSelection.role === role &&
    currentSelection.kind === kind &&
    Number(currentSelection.index) === index
  ) {
    stateCache.defenseSelection.first = null;
    setDefenseFeedback(null);
    applyDefenseSelectionHighlights();
    updateDefenseSelectionInfo();
    return;
  }

  if (!currentSelection) {
    stateCache.defenseSelection.first = selection;
    setDefenseFeedback(null);
    applyDefenseSelectionHighlights();
    updateDefenseSelectionInfo();
    return;
  }

  const result = applyDefenseSwap(currentSelection, selection);
  if (result.success) {
    const plan = stateCache.defensePlan;
    stateCache.defenseSelection.first = null;
    setDefenseFeedback(result.message, result.variant || 'success');
    commitDefensePlanChange(plan, getCurrentGameState());
  } else if (result.message) {
    setDefenseFeedback(result.message, result.variant || 'danger');
  }

  applyDefenseSelectionHighlights();
  updateDefenseSelectionInfo();
}

export function resetDefenseSelectionsIfUnavailable(defenseLineup, defenseBenchPlayers) {
  const selection = stateCache.defenseSelection.first;
  if (!selection) return;

  if (
    selection.role === 'lineup' &&
    (!Array.isArray(defenseLineup) ||
      Number(selection.index) < 0 ||
      Number(selection.index) >= defenseLineup.length)
  ) {
    resetDefenseSelection();
  } else if (
    selection.role === 'bench' &&
    (!Array.isArray(defenseBenchPlayers) ||
      Number(selection.index) < 0 ||
      Number(selection.index) >= defenseBenchPlayers.length)
  ) {
    resetDefenseSelection();
  }
}

