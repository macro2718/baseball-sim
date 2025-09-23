import { FIELD_POSITIONS } from '../config.js';
import { elements } from '../dom.js';
import {
  stateCache,
  normalizePositionKey,
  canBenchPlayerCoverPosition,
  updateDefenseContext,
  resetDefenseSelection,
  isKnownFieldPosition,
} from '../state.js';
import { escapeHtml, renderPositionList, renderPositionToken } from '../utils.js';

const DEFAULT_INFO = '出場中またはベンチの選手を2人選択すると入れ替えできます。';

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
  if (elements.defenseExtras) {
    elements.defenseExtras.classList.add('hidden');
    elements.defenseExtras.innerHTML = '';
  }
  if (elements.defenseBench) {
    elements.defenseBench.innerHTML = '';
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

function renderRetiredList(container, retiredPlayers) {
  if (!container) return;
  container.innerHTML = '';
  if (!retiredPlayers.length) {
    container.classList.add('hidden');
    return;
  }

  container.classList.remove('hidden');
  const title = document.createElement('p');
  title.className = 'extras-title';
  title.textContent = 'リタイア選手';
  container.appendChild(title);

  retiredPlayers.forEach((player) => {
    const card = document.createElement('div');
    card.className = 'retired-card';
    const retiredToken = renderPositionToken(player?.position || '-', player?.pitcher_type);
    const eligibleHtml = renderPositionList(player?.eligible || [], player?.pitcher_type);
    card.innerHTML = `
      <strong>${escapeHtml(player?.name ?? '-')}</strong>
      <span class="retired-status">${retiredToken ? `${retiredToken} 退場` : '退場'}</span>
      <span class="eligible-label">適性</span>
      <span class="eligible-positions">${eligibleHtml}</span>
    `;
    container.appendChild(card);
  });
}

function renderDefensePlanView(plan, gameState) {
  const lineup = plan?.lineup || [];
  const benchPlayers = plan?.bench || [];
  const retiredPlayers = plan?.retired || [];
  const activeGame = Boolean(gameState?.active);
  const canInteract = activeGame && stateCache.defenseContext.canSub;

  if (elements.defenseField) {
    elements.defenseField.innerHTML = '';
    const assigned = new Map();
    const extras = [];

    lineup.forEach((player) => {
      if (!player) return;
      const key = normalizePositionKey(player.position_key || player.position);
      if (key && isKnownFieldPosition(key) && !assigned.has(key)) {
        assigned.set(key, player);
      } else {
        extras.push(player);
      }
    });

    FIELD_POSITIONS.forEach((slot) => {
      const player = assigned.get(slot.key);
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `position-slot ${slot.className}`;
      button.dataset.position = slot.key;
      button.dataset.role = 'lineup';
      if (player) {
        button.dataset.index = String(player.index);
        button.dataset.lineupIndex = String(player.index);
        const eligibleHtml = renderPositionList(player.eligible || [], player.pitcher_type);
        button.innerHTML = `
          ${renderPositionToken(slot.label, null, 'position-label')}
          <strong>${escapeHtml(player.name ?? '-')}</strong>
          <span class="eligible-positions">${eligibleHtml}</span>
        `;
        button.disabled = !canInteract;
      } else {
        button.dataset.index = '';
        button.dataset.lineupIndex = '';
        const emptyEligible = renderPositionList([], null);
        button.innerHTML = `
          ${renderPositionToken(slot.label, null, 'position-label')}
          <strong>空席</strong>
          <span class="eligible-positions">${emptyEligible}</span>
        `;
        button.disabled = true;
      }
      elements.defenseField.appendChild(button);
    });

    if (elements.defenseExtras) {
      if (extras.length) {
        elements.defenseExtras.classList.remove('hidden');
        elements.defenseExtras.innerHTML = '';
        const title = document.createElement('p');
        title.className = 'extras-title';
        title.textContent = '配置外の選手';
        elements.defenseExtras.appendChild(title);
        extras.forEach((player) => {
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'bench-card';
          button.dataset.role = 'lineup';
          button.dataset.index = String(player.index);
          button.dataset.lineupIndex = String(player.index);
          const currentPosition = player.position && player.position !== '-' ? player.position : '-';
          const currentPositionHtml = renderPositionToken(currentPosition, player.pitcher_type);
          const eligibleHtml = renderPositionList(player.eligible || [], player.pitcher_type);
          button.innerHTML = `
            <strong>${escapeHtml(player.name ?? '-')}</strong>
            <span class="eligible-label">現在: ${currentPositionHtml}</span>
            <span class="eligible-positions">適性: ${eligibleHtml}</span>
          `;
          button.disabled = !canInteract;
          elements.defenseExtras.appendChild(button);
        });
      } else {
        elements.defenseExtras.classList.add('hidden');
        elements.defenseExtras.innerHTML = '';
      }
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
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'bench-card';
        button.dataset.role = 'bench';
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

  if (elements.defenseRetired) {
    renderRetiredList(elements.defenseRetired, retiredPlayers);
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

  if (posBKey && !canBenchPlayerCoverPosition(playerA, posBKey)) {
    return {
      success: false,
      message: `${playerA.name} は ${displayPosB} を守れません。`,
      variant: 'danger',
    };
  }
  if (posAKey && !canBenchPlayerCoverPosition(playerB, posAKey)) {
    return {
      success: false,
      message: `${playerB.name} は ${displayPosA} を守れません。`,
      variant: 'danger',
    };
  }

  playerA.position = displayPosB;
  playerA.position_key = posBKey;
  playerB.position = displayPosA;
  playerB.position_key = posAKey;

  plan.operations.push({
    type: 'lineup_lineup',
    lineupIndexA: indexA,
    lineupIndexB: indexB,
  });

  return {
    success: true,
    message: `${playerA.name} と ${playerB.name} の守備位置を入れ替える案を追加しました。`,
    variant: 'success',
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

  if (positionKey && !canBenchPlayerCoverPosition(benchPlayer, positionKey)) {
    return {
      success: false,
      message: `${benchPlayer.name} は ${positionLabel} を守れません。別の選手を選択してください。`,
      variant: 'danger',
    };
  }

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

  return {
    success: true,
    message: `${entering.name} を ${positionLabel} に投入する案を追加しました（${lineupPlayer.name} はリタイア）。`,
    variant: 'success',
  };
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

  if (first.type === second.type && first.index === second.index) {
    return { success: false, message: null };
  }

  if (first.type === 'lineup' && second.type === 'lineup') {
    return swapLineupPositions(plan, first.index, second.index);
  }

  if ((first.type === 'lineup' && second.type === 'bench') || (first.type === 'bench' && second.type === 'lineup')) {
    const lineupIndex = first.type === 'lineup' ? first.index : second.index;
    const benchIndex = first.type === 'bench' ? first.index : second.index;
    return swapBenchWithLineup(plan, lineupIndex, benchIndex);
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
  if (!elements.defenseBench) return;

  const plan = stateCache.defensePlan;
  const selection = stateCache.defenseSelection.first;
  const canSubBase = stateCache.defenseContext.canSub;
  const hasLineupSelection = selection && selection.type === 'lineup';
  const lineupPlayer = hasLineupSelection ? getPlanLineupPlayer(selection.index) : null;
  const lineupPositionKey = lineupPlayer
    ? normalizePositionKey(lineupPlayer.position_key || lineupPlayer.position)
    : null;
  const positionLabel = lineupPlayer ? lineupPlayer.position || lineupPositionKey || '' : '';

  elements.defenseBench.querySelectorAll('[data-role="bench"]').forEach((button) => {
    const value = button.dataset.index ?? button.dataset.benchIndex;
    const benchIndex = Number(value);
    const benchPlayer = plan && Number.isInteger(benchIndex) ? plan.bench[benchIndex] : null;

    let enable = canSubBase && Boolean(benchPlayer);
    let markIneligible = false;

    if (enable && hasLineupSelection && lineupPositionKey) {
      enable = canBenchPlayerCoverPosition(benchPlayer, lineupPositionKey);
      markIneligible = !enable;
    }

    button.disabled = !enable;
    button.classList.toggle('ineligible', markIneligible);

    let hint = button.querySelector('.ineligible-hint');
    if (markIneligible) {
      if (!hint) {
        hint = document.createElement('span');
        hint.className = 'ineligible-hint';
        button.appendChild(hint);
      }
      hint.textContent = '守備不可';
      hint.classList.remove('hidden');
      button.title = positionLabel
        ? `${benchPlayer.name} は ${positionLabel} を守れません。`
        : `${benchPlayer.name} はこの守備位置を守れません。`;
    } else if (hint) {
      hint.textContent = '';
      hint.classList.add('hidden');
      button.title = '';
    } else {
      button.title = '';
    }
  });
}

export function applyDefenseSelectionHighlights() {
  const selection = stateCache.defenseSelection.first;
  const lineupTarget = selection && selection.type === 'lineup' ? selection.index : null;
  const benchTarget = selection && selection.type === 'bench' ? selection.index : null;

  if (elements.defenseField) {
    elements.defenseField.querySelectorAll('[data-role="lineup"]').forEach((button) => {
      const value = button.dataset.index ?? button.dataset.lineupIndex;
      const index = Number(value);
      const isSelected = Number.isInteger(index) && index === lineupTarget;
      button.classList.toggle('selected', isSelected);
    });
  }

  if (elements.defenseExtras) {
    elements.defenseExtras.querySelectorAll('[data-role="lineup"]').forEach((button) => {
      const value = button.dataset.index ?? button.dataset.lineupIndex;
      const index = Number(value);
      const isSelected = Number.isInteger(index) && index === lineupTarget;
      button.classList.toggle('selected', isSelected);
    });
  }

  if (elements.defenseBench) {
    elements.defenseBench.querySelectorAll('[data-role="bench"]').forEach((button) => {
      const value = button.dataset.index ?? button.dataset.benchIndex;
      const index = Number(value);
      const isSelected = Number.isInteger(index) && index === benchTarget;
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
  } else if (selection?.type === 'lineup') {
    const player = getPlanLineupPlayer(selection.index);
    message = player
      ? `${player.name} と入れ替える選手を選択してください。`
      : '守備交代を行う守備位置を選択してください。';
  } else if (selection?.type === 'bench') {
    const player = getPlanBenchPlayer(selection.index);
    message = player
      ? `${player.name} を投入する守備位置を選択してください。`
      : '守備交代を行う守備位置を選択してください。';
  } else if (operationsCount > 0) {
    message = `未適用の守備交代案が ${operationsCount} 件あります。適用ボタンで確定してください。`;
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

  applyDefenseSelectionHighlights();
}

export function handleDefensePlayerClick(event) {
  const button = event.target.closest('[data-role]');
  if (!button || button.disabled) return;

  const role = button.dataset.role;
  if (!role || (role !== 'lineup' && role !== 'bench')) return;

  const value =
    role === 'lineup'
      ? button.dataset.index ?? button.dataset.lineupIndex
      : button.dataset.index ?? button.dataset.benchIndex;
  const index = Number(value);
  if (!Number.isInteger(index) || index < 0) return;

  const currentSelection = stateCache.defenseSelection.first;

  if (currentSelection && currentSelection.type === role && currentSelection.index === index) {
    stateCache.defenseSelection.first = null;
    setDefenseFeedback(null);
    applyDefenseSelectionHighlights();
    updateDefenseSelectionInfo();
    return;
  }

  if (!currentSelection) {
    stateCache.defenseSelection.first = { type: role, index };
    setDefenseFeedback(null);
    applyDefenseSelectionHighlights();
    updateDefenseSelectionInfo();
    return;
  }

  const result = applyDefenseSwap(currentSelection, { type: role, index });
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
    selection.type === 'lineup' &&
    (!Array.isArray(defenseLineup) || selection.index < 0 || selection.index >= defenseLineup.length)
  ) {
    resetDefenseSelection();
  } else if (
    selection.type === 'bench' &&
    (!Array.isArray(defenseBenchPlayers) || selection.index < 0 || selection.index >= defenseBenchPlayers.length)
  ) {
    resetDefenseSelection();
  }
}

