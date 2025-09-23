import { FIELD_POSITIONS } from '../config.js';
import { elements } from '../dom.js';
import {
  stateCache,
  normalizePositionKey,
  getLineupPlayer,
  getBenchPlayer,
  getLineupPositionKey,
  canBenchPlayerCoverPosition,
  updateDefenseContext,
  resetDefenseSelection,
  isKnownFieldPosition,
} from '../state.js';
import { escapeHtml, renderPositionList, renderPositionToken } from '../utils.js';

export function renderDefensePanel(defenseTeam, gameState) {
  const lineup = defenseTeam?.lineup || [];
  const benchPlayers = defenseTeam?.bench || [];

  const lineupMap = {};
  lineup.forEach((player) => {
    lineupMap[player.index] = player;
  });
  const benchMap = {};
  benchPlayers.forEach((player) => {
    benchMap[player.index] = player;
  });

  updateDefenseContext(lineupMap, benchMap, stateCache.defenseContext.canSub);

  if (elements.defenseField) {
    elements.defenseField.innerHTML = '';
    const assigned = new Map();
    const extras = [];

    lineup.forEach((player) => {
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
      const labelHtml = renderPositionToken(slot.label, null, 'position-label');
      if (player) {
        button.dataset.lineupIndex = player.index;
        const eligibleHtml = renderPositionList(player.eligible || [], player.pitcher_type);
        button.innerHTML = `
          ${labelHtml}
          <strong>${escapeHtml(player.name ?? '-')}</strong>
          <span class="eligible-positions">${eligibleHtml}</span>
        `;
        button.disabled = !gameState.active;
      } else {
        button.dataset.lineupIndex = '';
        const emptyEligible = renderPositionList([], null);
        button.innerHTML = `
          ${labelHtml}
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
          button.dataset.lineupIndex = player.index;
          const currentPosition = player.position && player.position !== '-' ? player.position : '-';
          const currentPositionHtml = renderPositionToken(currentPosition, player.pitcher_type);
          const eligibleHtml = renderPositionList(player.eligible || [], player.pitcher_type);
          button.innerHTML = `
            <strong>${escapeHtml(player.name ?? '-')}</strong>
            <span class="eligible-label">現在: ${currentPositionHtml}</span>
            <span class="eligible-positions">適性: ${eligibleHtml}</span>
          `;
          button.disabled = !gameState.active;
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
        button.dataset.benchIndex = player.index;
        const eligibleHtml = renderPositionList(player.eligible || [], player.pitcher_type);
        button.innerHTML = `
          <strong>${escapeHtml(player.name ?? '-')}</strong>
          <span class="eligible-label">守備適性</span>
          <span class="eligible-positions">${eligibleHtml}</span>
        `;
        button.disabled = !(gameState.active && stateCache.defenseContext.canSub);
        elements.defenseBench.appendChild(button);
      });
    }
  }

  updateDefenseBenchAvailability();
  applyDefenseSelectionHighlights();
}

export function updateDefenseBenchAvailability() {
  if (!elements.defenseBench) return;

  const lineupPlayer = getLineupPlayer(stateCache.defenseSelection.lineupIndex);
  const lineupPositionKey = getLineupPositionKey(lineupPlayer);
  const positionLabel = lineupPlayer ? lineupPlayer.position || lineupPositionKey || '' : '';
  const canSubBase = stateCache.defenseContext.canSub;

  elements.defenseBench.querySelectorAll('[data-bench-index]').forEach((button) => {
    const benchValue = button.dataset.benchIndex;
    const benchIndex = Number(benchValue);
    const benchPlayer = Number.isInteger(benchIndex) ? getBenchPlayer(benchIndex) : null;

    const hasLineupSelection = Boolean(lineupPlayer);
    const hasBenchPlayer = Boolean(benchPlayer);
    let eligibleForPosition = false;
    if (canSubBase && hasLineupSelection && hasBenchPlayer) {
      if (lineupPositionKey) {
        eligibleForPosition = canBenchPlayerCoverPosition(benchPlayer, lineupPositionKey);
      } else {
        eligibleForPosition = true;
      }
    }

    const enableButton = canSubBase && hasLineupSelection && hasBenchPlayer && eligibleForPosition;
    const markIneligible =
      canSubBase && hasLineupSelection && hasBenchPlayer && Boolean(lineupPositionKey) && !eligibleForPosition;

    button.disabled = !enableButton;
    button.classList.toggle('ineligible', markIneligible);

    if (!enableButton && stateCache.defenseSelection.benchIndex === benchIndex) {
      stateCache.defenseSelection.benchIndex = null;
    }

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
  const { lineupIndex, benchIndex } = stateCache.defenseSelection;
  if (elements.defenseField) {
    elements.defenseField.querySelectorAll('[data-lineup-index]').forEach((button) => {
      const value = button.dataset.lineupIndex;
      const index = Number(value);
      const isSelected = value !== undefined && value !== '' && Number.isInteger(index) && index === lineupIndex;
      button.classList.toggle('selected', isSelected);
    });
  }
  if (elements.defenseExtras) {
    elements.defenseExtras.querySelectorAll('[data-lineup-index]').forEach((button) => {
      const value = button.dataset.lineupIndex;
      const index = Number(value);
      const isSelected = value !== undefined && value !== '' && Number.isInteger(index) && index === lineupIndex;
      button.classList.toggle('selected', isSelected);
    });
  }
  if (elements.defenseBench) {
    elements.defenseBench.querySelectorAll('[data-bench-index]').forEach((button) => {
      const value = button.dataset.benchIndex;
      const index = Number(value);
      const isSelected = value !== undefined && value !== '' && Number.isInteger(index) && index === benchIndex;
      button.classList.toggle('selected', isSelected);
    });
  }
}

export function updateDefenseSelectionInfo() {
  updateDefenseBenchAvailability();

  const infoEl = elements.defenseSelectionInfo;
  const { lineupIndex, benchIndex } = stateCache.defenseSelection;
  const lineupPlayer = getLineupPlayer(lineupIndex);
  const benchPlayer = getBenchPlayer(benchIndex);
  const lineupPositionKey = getLineupPositionKey(lineupPlayer);
  const positionLabel = lineupPlayer ? lineupPlayer.position || lineupPositionKey || '指定ポジション' : '';

  let benchEligible = true;
  if (lineupPlayer && benchPlayer && lineupPositionKey) {
    benchEligible = canBenchPlayerCoverPosition(benchPlayer, lineupPositionKey);
  }

  if (infoEl) {
    if (!stateCache.defenseContext.canSub) {
      infoEl.textContent = '守備交代を行える状況ではありません。';
    } else if (!lineupPlayer && !benchPlayer) {
      infoEl.textContent = '守備交代を行う守備位置とベンチ選手を選択してください。';
    } else if (!lineupPlayer) {
      infoEl.textContent = benchPlayer
        ? `${benchPlayer.name} を投入する守備位置を選択してください。`
        : '守備交代を行う守備位置を選択してください。';
    } else if (!benchPlayer) {
      infoEl.textContent = `${lineupPlayer.name} を交代させる選手を選択してください。`;
    } else if (lineupPositionKey && !benchEligible) {
      infoEl.textContent = `${benchPlayer.name} は ${positionLabel} を守れません。別の選手を選択してください。`;
    } else {
      infoEl.textContent = `${lineupPlayer.name} ↔ ${benchPlayer.name} の守備交代を実行できます。`;
    }
  }

  const canApply =
    stateCache.defenseContext.canSub && Boolean(lineupPlayer) && Boolean(benchPlayer) && (!lineupPositionKey || benchEligible);
  if (elements.defenseApplyButton) {
    elements.defenseApplyButton.disabled = !canApply;
  }

  applyDefenseSelectionHighlights();
}

export function handleDefenseFieldClick(event) {
  const button = event.target.closest('button[data-lineup-index]');
  if (!button || button.disabled) return;
  const value = button.dataset.lineupIndex;
  if (value === undefined || value === '') return;
  const index = Number(value);
  if (Number.isNaN(index)) return;
  stateCache.defenseSelection.lineupIndex = index;
  applyDefenseSelectionHighlights();
  updateDefenseSelectionInfo();
}

export function handleDefenseBenchClick(event) {
  const lineupButton = event.target.closest('button[data-lineup-index]');
  if (lineupButton && !lineupButton.disabled) {
    const value = lineupButton.dataset.lineupIndex;
    if (value !== undefined && value !== '') {
      const index = Number(value);
      if (!Number.isNaN(index)) {
        stateCache.defenseSelection.lineupIndex = index;
        applyDefenseSelectionHighlights();
        updateDefenseSelectionInfo();
        return;
      }
    }
  }

  const benchButton = event.target.closest('button[data-bench-index]');
  if (!benchButton || benchButton.disabled) return;
  const benchValue = benchButton.dataset.benchIndex;
  if (benchValue === undefined || benchValue === '') return;
  const benchIndex = Number(benchValue);
  if (Number.isNaN(benchIndex)) return;
  stateCache.defenseSelection.benchIndex = benchIndex;
  applyDefenseSelectionHighlights();
  updateDefenseSelectionInfo();
}

export function resetDefenseSelectionsIfUnavailable(defenseLineup, defenseBenchPlayers) {
  if (!defenseLineup.some((player) => player.index === stateCache.defenseSelection.lineupIndex)) {
    stateCache.defenseSelection.lineupIndex = null;
  }
  if (!defenseBenchPlayers.some((player) => player.index === stateCache.defenseSelection.benchIndex)) {
    stateCache.defenseSelection.benchIndex = null;
  }
}

