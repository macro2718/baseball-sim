const elements = {
  titleScreen: document.getElementById('title-screen'),
  gameScreen: document.getElementById('game-screen'),
  statusMessage: document.getElementById('status-message'),
  startButton: document.getElementById('start-game'),
  reloadTeams: document.getElementById('reload-teams'),
  restartButton: document.getElementById('restart-game'),
  returnTitle: document.getElementById('return-title'),
  clearLog: document.getElementById('clear-log'),
  swingButton: document.getElementById('swing-button'),
  buntButton: document.getElementById('bunt-button'),
  openOffenseButton: document.getElementById('open-offense-strategy'),
  openDefenseButton: document.getElementById('open-defense-strategy'),
  openStatsButton: document.getElementById('open-stats'),
  defenseMenu: document.getElementById('defense-strategy-menu'),
  defenseSubMenuButton: document.getElementById('open-defense-sub'),
  pitcherMenuButton: document.getElementById('open-pitcher-change-menu'),
  actionWarning: document.getElementById('action-warning'),
  titleHint: document.getElementById('title-hint'),
  logContainer: document.getElementById('log-entries'),
  scoreboard: document.getElementById('scoreboard'),
  situationText: document.getElementById('situation-text'),
  matchupText: document.getElementById('matchup-text'),
  halfIndicator: document.getElementById('half-indicator'),
  outsIndicator: document.getElementById('outs-indicator'),
  defenseErrors: document.getElementById('defense-errors'),
  offenseRoster: document.querySelector('#offense-roster tbody'),
  defenseRoster: document.querySelector('#defense-roster tbody'),
  offenseBench: document.getElementById('offense-bench'),
  defenseBenchList: document.getElementById('defense-bench'),
  homePitchers: document.querySelector('#home-pitchers ul'),
  awayPitchers: document.querySelector('#away-pitchers ul'),
  baseState: document.getElementById('base-state'),
  pinchPlayer: document.getElementById('pinch-player'),
  pinchButton: document.getElementById('pinch-hit-button'),
  pinchCurrentBatter: document.getElementById('pinch-current-batter'),
  offenseModal: document.getElementById('offense-modal'),
  defenseModal: document.getElementById('defense-modal'),
  pitcherModal: document.getElementById('pitcher-modal'),
  statsModal: document.getElementById('stats-modal'),
  modalCloseButtons: Array.from(document.querySelectorAll('.modal-close')),
  defenseApplyButton: document.getElementById('defense-sub-button'),
  defenseField: document.getElementById('defense-field'),
  defenseBench: document.getElementById('defense-bench-panel'),
  defenseExtras: document.getElementById('defense-extras'),
  defenseSelectionInfo: document.getElementById('defense-selection-info'),
  pitcherSelect: document.getElementById('pitcher-select'),
  pitcherButton: document.getElementById('change-pitcher-button'),
  statsTeamButtons: Array.from(document.querySelectorAll('[data-stats-team]')),
  statsTypeButtons: Array.from(document.querySelectorAll('[data-stats-type]')),
  statsTableHead: document.querySelector('#stats-table thead tr'),
  statsTableBody: document.querySelector('#stats-table tbody'),
  statsTitle: document.getElementById('stats-title'),
};

const FIELD_POSITIONS = [
  { key: 'CF', label: 'CF', className: 'pos-cf' },
  { key: 'LF', label: 'LF', className: 'pos-lf' },
  { key: 'RF', label: 'RF', className: 'pos-rf' },
  { key: '2B', label: '2B', className: 'pos-2b' },
  { key: 'SS', label: 'SS', className: 'pos-ss' },
  { key: '3B', label: '3B', className: 'pos-3b' },
  { key: '1B', label: '1B', className: 'pos-1b' },
  { key: 'P', label: 'P', className: 'pos-p' },
  { key: 'C', label: 'C', className: 'pos-c' },
  { key: 'DH', label: 'DH', className: 'pos-dh' },
];

const FIELD_POSITION_KEYS = new Set(FIELD_POSITIONS.map((slot) => slot.key));

function normalizePositionKey(position) {
  if (!position) return null;
  const upper = String(position).toUpperCase();
  if (upper === 'SP' || upper === 'RP') return 'P';
  return upper;
}

function getLineupPlayer(index) {
  const lineupMap = stateCache.defenseContext.lineup || {};
  if (!Number.isInteger(index)) return null;
  return Object.prototype.hasOwnProperty.call(lineupMap, index) ? lineupMap[index] : null;
}

function getBenchPlayer(index) {
  const benchMap = stateCache.defenseContext.bench || {};
  if (!Number.isInteger(index)) return null;
  return Object.prototype.hasOwnProperty.call(benchMap, index) ? benchMap[index] : null;
}

function getLineupPositionKey(lineupPlayer) {
  if (!lineupPlayer) return null;
  return normalizePositionKey(lineupPlayer.position_key || lineupPlayer.position);
}

function getEligiblePositionsAll(player) {
  if (!player) return [];
  const raw = Array.isArray(player.eligible_all) ? player.eligible_all : player.eligible;
  if (!raw) return [];
  return raw.map((pos) => String(pos).toUpperCase());
}

function canBenchPlayerCoverPosition(benchPlayer, positionKey) {
  if (!benchPlayer || !positionKey) return false;
  const eligiblePositions = getEligiblePositionsAll(benchPlayer);
  return eligiblePositions.includes(positionKey);
}

const BATTING_COLUMNS = [
  { key: 'name', label: '選手' },
  { key: 'ab', label: '打数' },
  { key: 'single', label: '単打' },
  { key: 'double', label: '二塁打' },
  { key: 'triple', label: '三塁打' },
  { key: 'hr', label: '本塁打' },
  { key: 'runs', label: '得点' },
  { key: 'rbi', label: '打点' },
  { key: 'bb', label: '四球' },
  { key: 'so', label: '三振' },
  { key: 'avg', label: '打率' },
];

const PITCHING_COLUMNS = [
  { key: 'name', label: '選手' },
  { key: 'ip', label: '投球回' },
  { key: 'h', label: '被安打' },
  { key: 'r', label: '失点' },
  { key: 'er', label: '自責点' },
  { key: 'bb', label: '四球' },
  { key: 'k', label: '奪三振' },
  { key: 'era', label: '防御率' },
  { key: 'whip', label: 'WHIP' },
];

const stateCache = {
  data: null,
  defenseSelection: { lineupIndex: null, benchIndex: null },
  defenseContext: { lineup: {}, bench: {}, canSub: false },
  currentBatterIndex: null,
  statsView: { team: 'away', type: 'batting' },
};

async function apiRequest(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  const payload = await response.json();
  if (!response.ok) {
    if (payload.state) {
      render(payload.state);
    }
    if (payload.error) {
      showStatus(payload.error, 'danger');
    }
    throw new Error(payload.error || 'Request failed');
  }
  return payload;
}

function resolveModal(target) {
  if (!target) return null;
  if (target instanceof HTMLElement) return target;
  if (typeof target === 'string') {
    if (target.endsWith('-modal')) {
      return document.getElementById(target);
    }
    const key = `${target}Modal`;
    if (Object.prototype.hasOwnProperty.call(elements, key)) {
      return elements[key];
    }
    return document.getElementById(`${target}-modal`);
  }
  return null;
}

function openModal(name) {
  const modal = resolveModal(name);
  if (!modal) return;
  hideDefenseMenu();
  modal.classList.remove('hidden');
  modal.setAttribute('aria-hidden', 'false');
  if (modal === elements.statsModal) {
    updateStatsPanel(stateCache.data);
  }
  if (modal === elements.defenseModal) {
    updateDefenseSelectionInfo();
  }
}

function closeModal(name) {
  const modal = resolveModal(name);
  if (!modal) return;
  modal.classList.add('hidden');
  modal.setAttribute('aria-hidden', 'true');
}

function hideDefenseMenu() {
  const { defenseMenu } = elements;
  if (!defenseMenu) return;
  if (!defenseMenu.classList.contains('hidden')) {
    defenseMenu.classList.add('hidden');
  }
  defenseMenu.setAttribute('aria-hidden', 'true');
}

function showDefenseMenu() {
  const { defenseMenu } = elements;
  if (!defenseMenu) return;
  defenseMenu.classList.remove('hidden');
  defenseMenu.setAttribute('aria-hidden', 'false');
}

function toggleDefenseMenu() {
  const { defenseMenu, openDefenseButton } = elements;
  if (!defenseMenu) return;
  if (openDefenseButton && openDefenseButton.disabled) {
    return;
  }
  if (defenseMenu.classList.contains('hidden')) {
    showDefenseMenu();
  } else {
    hideDefenseMenu();
  }
}

function initEventListeners() {
  elements.startButton.addEventListener('click', () => handleStart(false));
  elements.reloadTeams.addEventListener('click', handleReloadTeams);
  elements.restartButton.addEventListener('click', () => handleStart(true));
  elements.returnTitle.addEventListener('click', handleReturnToTitle);
  elements.clearLog.addEventListener('click', handleClearLog);
  elements.swingButton.addEventListener('click', handleSwing);
  elements.buntButton.addEventListener('click', handleBunt);
  if (elements.pinchButton) {
    elements.pinchButton.addEventListener('click', handlePinchHit);
  }
  if (elements.openOffenseButton) {
    elements.openOffenseButton.addEventListener('click', () => openModal('offense'));
  }
  if (elements.openDefenseButton) {
    elements.openDefenseButton.addEventListener('click', toggleDefenseMenu);
  }
  if (elements.defenseSubMenuButton) {
    elements.defenseSubMenuButton.addEventListener('click', () => openModal('defense'));
  }
  if (elements.pitcherMenuButton) {
    elements.pitcherMenuButton.addEventListener('click', () => openModal('pitcher'));
  }
  if (elements.openStatsButton) {
    elements.openStatsButton.addEventListener('click', () => openModal('stats'));
  }
  if (elements.defenseApplyButton) {
    elements.defenseApplyButton.addEventListener('click', handleDefenseSubstitution);
  }
  if (elements.pitcherButton) {
    elements.pitcherButton.addEventListener('click', handlePitcherChange);
  }
  elements.modalCloseButtons.forEach((button) => {
    const target = button.dataset.close;
    button.addEventListener('click', () => closeModal(target || button.closest('.modal')));
  });
  ['offense', 'defense', 'pitcher', 'stats'].forEach((name) => {
    const modal = resolveModal(name);
    if (modal) {
      modal.addEventListener('click', (event) => {
        if (event.target === modal) {
          closeModal(modal);
        }
      });
    }
  });
  if (elements.defenseMenu) {
    document.addEventListener('click', (event) => {
      if (elements.defenseMenu.classList.contains('hidden')) {
        return;
      }
      const clickedInsideMenu = elements.defenseMenu.contains(event.target);
      const clickedToggle =
        elements.openDefenseButton && elements.openDefenseButton.contains(event.target);
      if (!clickedInsideMenu && !clickedToggle) {
        hideDefenseMenu();
      }
    });
  }
  if (elements.defenseField) {
    elements.defenseField.addEventListener('click', handleDefenseFieldClick);
  }
  if (elements.defenseBench) {
    elements.defenseBench.addEventListener('click', handleDefenseBenchClick);
  }
  if (elements.defenseExtras) {
    elements.defenseExtras.addEventListener('click', handleDefenseBenchClick);
  }
  elements.statsTeamButtons.forEach((button) => {
    button.addEventListener('click', () => {
      if (button.disabled) return;
      const teamKey = button.dataset.statsTeam;
      if (!teamKey) return;
      stateCache.statsView.team = teamKey;
      updateStatsPanel(stateCache.data);
    });
  });
  elements.statsTypeButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const viewType = button.dataset.statsType;
      if (!viewType) return;
      stateCache.statsView.type = viewType;
      updateStatsPanel(stateCache.data);
    });
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      ['offense', 'defense', 'pitcher', 'stats'].forEach((name) => {
        const modal = resolveModal(name);
        if (modal && !modal.classList.contains('hidden')) {
          closeModal(modal);
        }
      });
    }
  });
}

async function handleStart(reload) {
  try {
    const payload = await apiRequest('/api/game/start', {
      method: 'POST',
      body: JSON.stringify({ reload }),
    });
    render(payload);
  } catch (err) {
    console.warn(err);
  }
}

async function handleReloadTeams() {
  try {
    const payload = await apiRequest('/api/teams/reload', { method: 'POST' });
    render(payload);
  } catch (err) {
    console.warn(err);
  }
}

async function handleReturnToTitle() {
  try {
    const payload = await apiRequest('/api/game/stop', { method: 'POST' });
    render(payload);
  } catch (err) {
    console.warn(err);
  }
}

async function handleClearLog() {
  try {
    const payload = await apiRequest('/api/log/clear', { method: 'POST' });
    render(payload);
  } catch (err) {
    console.warn(err);
  }
}

async function handleSwing() {
  try {
    const payload = await apiRequest('/api/game/swing', { method: 'POST' });
    render(payload);
  } catch (err) {
    console.warn(err);
  }
}

async function handleBunt() {
  try {
    const payload = await apiRequest('/api/game/bunt', { method: 'POST' });
    render(payload);
  } catch (err) {
    console.warn(err);
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
    const payload = await apiRequest('/api/strategy/pinch_hit', {
      method: 'POST',
      body: JSON.stringify({ lineup_index: lineupIndex, bench_index: benchIndex }),
    });
    render(payload);
  } catch (err) {
    console.warn(err);
  }
}

async function handleDefenseSubstitution() {
  const { lineupIndex, benchIndex } = stateCache.defenseSelection;
  if (!stateCache.defenseContext.canSub) {
    showStatus('守備交代は現在行えません。', 'danger');
    return;
  }
  if (!Number.isInteger(lineupIndex) || lineupIndex < 0) {
    showStatus('交代させる守備位置を選択してください。', 'danger');
    return;
  }
  if (!Number.isInteger(benchIndex) || benchIndex < 0) {
    showStatus('ベンチから交代させる選手を選択してください。', 'danger');
    return;
  }

  const lineupPlayer = getLineupPlayer(lineupIndex);
  if (!lineupPlayer) {
    showStatus('選択された守備位置が無効です。', 'danger');
    return;
  }
  const benchPlayer = getBenchPlayer(benchIndex);
  if (!benchPlayer) {
    showStatus('選択されたベンチ選手が見つかりません。', 'danger');
    return;
  }
  const targetPositionKey = getLineupPositionKey(lineupPlayer);
  if (targetPositionKey && !canBenchPlayerCoverPosition(benchPlayer, targetPositionKey)) {
    const positionLabel = lineupPlayer.position || targetPositionKey;
    showStatus(`${benchPlayer.name} は ${positionLabel} を守れません。別の選手を選択してください。`, 'danger');
    return;
  }

  try {
    const payload = await apiRequest('/api/strategy/defense_substitution', {
      method: 'POST',
      body: JSON.stringify({ lineup_index: lineupIndex, bench_index: benchIndex }),
    });
    stateCache.defenseSelection = { lineupIndex: null, benchIndex: null };
    render(payload);
  } catch (err) {
    console.warn(err);
  }
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
    const payload = await apiRequest('/api/strategy/change_pitcher', {
      method: 'POST',
      body: JSON.stringify({ pitcher_index: pitcherIndex }),
    });
    render(payload);
  } catch (err) {
    console.warn(err);
  }
}

function render(data) {
  stateCache.data = data;
  setStatusMessage(data.notification);
  renderTitle(data.title);
  renderGame(data.game, data.teams, data.log);
  updateStatsPanel(data);
}

function setStatusMessage(notification) {
  if (!notification) {
    elements.statusMessage.textContent = '';
    elements.statusMessage.classList.remove('danger', 'success', 'info');
    return;
  }
  elements.statusMessage.textContent = notification.message;
  elements.statusMessage.classList.remove('danger', 'success', 'info');
  elements.statusMessage.classList.add(notification.level || 'info');
}

function renderTitle(titleState) {
  const homeName = document.querySelector('.team-name[data-team="home"]');
  const awayName = document.querySelector('.team-name[data-team="away"]');
  const homeMessage = document.querySelector('.team-message[data-team="home"]');
  const awayMessage = document.querySelector('.team-message[data-team="away"]');
  const homeErrors = document.querySelector('.team-errors[data-team="home"]');
  const awayErrors = document.querySelector('.team-errors[data-team="away"]');

  if (titleState.home) {
    homeName.textContent = titleState.home.name;
    homeMessage.textContent = titleState.home.message;
    homeErrors.innerHTML = '';
    titleState.home.errors.forEach((err) => {
      const li = document.createElement('li');
      li.textContent = err;
      homeErrors.appendChild(li);
    });
  }

  if (titleState.away) {
    awayName.textContent = titleState.away.name;
    awayMessage.textContent = titleState.away.message;
    awayErrors.innerHTML = '';
    titleState.away.errors.forEach((err) => {
      const li = document.createElement('li');
      li.textContent = err;
      awayErrors.appendChild(li);
    });
  }

  elements.titleHint.textContent = titleState.hint || '';
  elements.startButton.disabled = !titleState.ready;
}

function renderGame(gameState, teams, log) {
  if (!gameState.active) {
    elements.gameScreen.classList.add('hidden');
    elements.titleScreen.classList.remove('hidden');
    updateScoreboard(gameState, teams);
    elements.actionWarning.textContent = '';
    elements.swingButton.disabled = true;
    elements.buntButton.disabled = true;
    updateRosters(elements.offenseRoster, []);
    updateRosters(elements.defenseRoster, []);
    updateBench(elements.offenseBench, [], 'ゲーム開始後に表示されます');
    updateBench(elements.defenseBenchList, [], 'ゲーム開始後に表示されます');
    updateLog(log || []);
    elements.defenseErrors.classList.add('hidden');
    elements.defenseErrors.textContent = '';
    updateStrategyControls(gameState, teams);
    return;
  }

  elements.titleScreen.classList.add('hidden');
  elements.gameScreen.classList.remove('hidden');

  updateScoreboard(gameState, teams);
  elements.situationText.textContent = gameState.situation || '';
  elements.halfIndicator.textContent = `${gameState.half_label} ${gameState.inning}`;
  const outsLabel = gameState.outs === 1 ? 'OUT' : 'OUTS';
  elements.outsIndicator.textContent = `${gameState.outs} ${outsLabel}`;
  elements.matchupText.textContent = gameState.matchup || '';
  updateBases(gameState.bases || []);

  const offenseTeam = gameState.offense ? teams[gameState.offense] : null;
  const defenseTeam = gameState.defense ? teams[gameState.defense] : null;

  updateRosters(elements.offenseRoster, offenseTeam?.lineup || []);
  updateBench(elements.offenseBench, offenseTeam?.bench || []);
  updateRosters(elements.defenseRoster, defenseTeam?.lineup || []);
  updateBench(elements.defenseBenchList, defenseTeam?.bench || []);

  updatePitchers(elements.homePitchers, teams.home?.pitchers || []);
  updatePitchers(elements.awayPitchers, teams.away?.pitchers || []);

  updateLog(log || []);

  elements.swingButton.disabled = !gameState.actions?.swing;
  elements.buntButton.disabled = !gameState.actions?.bunt;
  elements.actionWarning.textContent = gameState.action_block_reason || '';

  const errors = gameState.defensive_errors || [];
  if (errors.length) {
    elements.defenseErrors.classList.remove('hidden');
    elements.defenseErrors.innerHTML = errors.map((err) => `<p>${err}</p>`).join('');
  } else {
    elements.defenseErrors.classList.add('hidden');
    elements.defenseErrors.textContent = '';
  }

  updateStrategyControls(gameState, teams);
}

function updateScoreboard(gameState, teams) {
  if (!elements.scoreboard) return;

  if (!gameState.active) {
    elements.scoreboard.innerHTML = '<p>試合はまだ開始されていません。</p>';
    return;
  }

  const innings = Math.max(
    gameState.max_innings || 0,
    gameState.inning_scores.away.length,
    gameState.inning_scores.home.length,
    9,
  );
  const hits = gameState.hits || { home: 0, away: 0 };
  const errors = gameState.errors || { home: 0, away: 0 };

  let html = '<table class="score-table"><thead><tr><th class="team-col">Team</th>';
  for (let i = 0; i < innings; i += 1) {
    html += `<th>${i + 1}</th>`;
  }
  html += '<th>R</th><th>H</th><th>E</th></tr></thead><tbody>';

  const renderRow = (teamKey) => {
    const teamName = teams[teamKey]?.name || (teamKey === 'home' ? 'Home' : 'Away');
    const scores = gameState.inning_scores[teamKey] || [];
    const totalRuns = gameState.score?.[teamKey] ?? 0;
    const totalHits = hits?.[teamKey] ?? 0;
    const totalErrors = errors?.[teamKey] ?? 0;
    let row = `<tr><td class="team-name">${teamName}</td>`;
    for (let i = 0; i < innings; i += 1) {
      const value = scores[i];
      row += `<td>${value ?? ''}</td>`;
    }
    row += `<td>${totalRuns}</td><td>${totalHits}</td><td>${totalErrors}</td></tr>`;
    return row;
  };

  html += renderRow('away');
  html += renderRow('home');
  html += '</tbody></table>';

  elements.scoreboard.innerHTML = html;
}

function updateBases(bases) {
  if (!elements.baseState) return;
  const baseElements = elements.baseState.querySelectorAll('.base');
  baseElements.forEach((el) => {
    const baseIndex = Number(el.dataset.base);
    const info = bases[baseIndex] || {};
    const occupied = Boolean(info.occupied);
    el.classList.toggle('occupied', occupied);
    const span = el.querySelector('span');
    if (span) {
      span.textContent = occupied ? '●' : '';
    }
    el.title = info.runner || '';
  });
}

function updateRosters(tbody, players) {
  if (!tbody) return;
  tbody.innerHTML = '';
  players.forEach((player) => {
    const tr = document.createElement('tr');
    if (player.is_current_batter) {
      tr.classList.add('active');
    }
    tr.innerHTML = `
      <td>${player.order}</td>
      <td>${player.position || '-'}</td>
      <td>${player.name}</td>
      <td>${(player.eligible || []).join(', ')}</td>
    `;
    tbody.appendChild(tr);
  });
}

function updateBench(listEl, benchPlayers, emptyMessage = '利用可能なベンチ選手はいません') {
  if (!listEl) return;
  listEl.innerHTML = '';

  if (!benchPlayers || benchPlayers.length === 0) {
    const li = document.createElement('li');
    li.classList.add('empty');
    li.textContent = emptyMessage;
    listEl.appendChild(li);
    return;
  }

  benchPlayers.forEach((player) => {
    const li = document.createElement('li');
    const name = document.createElement('span');
    name.textContent = player.name;
    const eligible = document.createElement('span');
    const eligibleText = (player.eligible || []).join(', ');
    eligible.textContent = eligibleText || '-';
    li.appendChild(name);
    li.appendChild(eligible);
    listEl.appendChild(li);
  });
}

function updatePitchers(listEl, pitchers) {
  if (!listEl) return;
  listEl.innerHTML = '';
  pitchers.forEach((pitcher) => {
    const li = document.createElement('li');
    if (pitcher.is_current) {
      li.classList.add('current');
    }
    const stamina = pitcher.stamina != null ? `${pitcher.stamina}` : '-';
    li.innerHTML = `
      <span>${pitcher.name} (${pitcher.pitcher_type})</span>
      <span>${stamina}</span>
    `;
    listEl.appendChild(li);
  });
}

function updateLog(logEntries) {
  if (!elements.logContainer) return;
  elements.logContainer.innerHTML = '';
  logEntries.forEach((entry) => {
    const div = document.createElement('div');
    div.classList.add('log-entry');
    div.classList.add(entry.variant || 'info');
    div.textContent = entry.text;
    elements.logContainer.appendChild(div);
  });
  elements.logContainer.scrollTop = elements.logContainer.scrollHeight;
}

function populateSelect(selectEl, options, placeholder) {
  if (!selectEl) return;
  const previousValue = selectEl.value;
  selectEl.innerHTML = '';

  const placeholderOption = document.createElement('option');
  placeholderOption.value = '';
  placeholderOption.textContent = placeholder;
  placeholderOption.disabled = true;
  placeholderOption.selected = true;
  selectEl.appendChild(placeholderOption);

  options.forEach((option) => {
    const opt = document.createElement('option');
    opt.value = `${option.value}`;
    opt.textContent = option.label;
    selectEl.appendChild(opt);
  });

  if (options.length) {
    const exists = options.some((option) => `${option.value}` === previousValue);
    if (exists) {
      selectEl.value = previousValue;
      placeholderOption.selected = false;
    } else {
      selectEl.value = '';
    }
  } else {
    selectEl.value = '';
  }
}

function updateStrategyControls(gameState, teams) {
  const {
    pinchPlayer,
    pinchButton,
    pinchCurrentBatter,
    openOffenseButton,
    openDefenseButton,
    openStatsButton,
    defenseSubMenuButton,
    pitcherMenuButton,
    pitcherSelect,
    pitcherButton,
  } = elements;

  const isActive = Boolean(gameState.active);

  const offenseTeam = gameState.offense ? teams[gameState.offense] : null;
  const offenseLineup = offenseTeam?.lineup || [];
  const offenseBench = offenseTeam?.bench || [];
  const currentBatter = offenseLineup.find((player) => player.is_current_batter) || null;

  stateCache.currentBatterIndex =
    currentBatter && Number.isInteger(currentBatter.index) ? currentBatter.index : null;

  if (pinchCurrentBatter) {
    if (currentBatter) {
      const orderLabel = Number.isInteger(currentBatter.order)
        ? `${currentBatter.order}. `
        : '';
      const positionLabel = currentBatter.position ? `${currentBatter.position} ` : '';
      pinchCurrentBatter.textContent = `現在の打者: ${orderLabel}${positionLabel}${currentBatter.name}`;
    } else {
      pinchCurrentBatter.textContent = '現在の打者: -';
    }
  }

  if (pinchPlayer && pinchButton) {
    const benchPlaceholder = !currentBatter
      ? '現在の打者が見つかりません'
      : offenseBench.length
      ? 'ベンチ選手を選択'
      : '選択可能な選手がいません';

    populateSelect(
      pinchPlayer,
      offenseBench.map((player) => ({
        value: player.index,
        label: `${player.name} (${(player.eligible || []).join(', ') || '-'})`,
      })),
      benchPlaceholder,
    );

    const canPinch = isActive && Boolean(currentBatter) && offenseBench.length > 0;
    pinchButton.disabled = !canPinch;
    pinchPlayer.disabled = !canPinch;
    if (!canPinch) {
      pinchPlayer.value = '';
    }
  }

  if (openOffenseButton) {
    openOffenseButton.disabled = !isActive;
  }

  const defenseTeam = gameState.defense ? teams[gameState.defense] : null;
  const defenseLineup = defenseTeam?.lineup || [];
  const defenseBenchPlayers = defenseTeam?.bench || [];
  const pitcherOptions = defenseTeam?.pitcher_options || [];

  const canDefenseSub = isActive && defenseLineup.length > 0 && defenseBenchPlayers.length > 0;

  if (!defenseLineup.some((player) => player.index === stateCache.defenseSelection.lineupIndex)) {
    stateCache.defenseSelection.lineupIndex = null;
  }
  if (!defenseBenchPlayers.some((player) => player.index === stateCache.defenseSelection.benchIndex)) {
    stateCache.defenseSelection.benchIndex = null;
  }

  stateCache.defenseContext.canSub = canDefenseSub;
  updateDefensePanel(defenseTeam, gameState);
  updateDefenseSelectionInfo();

  if (openDefenseButton) {
    openDefenseButton.disabled = !isActive;
  }
  if (!isActive) {
    hideDefenseMenu();
  }
  if (defenseSubMenuButton) {
    defenseSubMenuButton.disabled = !canDefenseSub;
  }

  if (pitcherSelect && pitcherButton) {
    const pitcherPlaceholder = pitcherOptions.length
      ? '交代する投手を選択'
      : '交代可能な投手がいません';

    populateSelect(
      pitcherSelect,
      pitcherOptions.map((pitcher) => ({
        value: pitcher.index,
        label: `${pitcher.name} (${pitcher.pitcher_type || 'P'})`,
      })),
      pitcherPlaceholder,
    );

    const canChangePitcher = isActive && pitcherOptions.length > 0;
    pitcherButton.disabled = !canChangePitcher;
    pitcherSelect.disabled = !canChangePitcher;
    if (!canChangePitcher) {
      pitcherSelect.value = '';
    }
    if (pitcherMenuButton) {
      pitcherMenuButton.disabled = !canChangePitcher;
    }
  } else if (pitcherMenuButton) {
    pitcherMenuButton.disabled = true;
  }

  if (openStatsButton) {
    const homeStatsAvailable = Boolean(teams.home?.stats);
    const awayStatsAvailable = Boolean(teams.away?.stats);
    openStatsButton.disabled = !(homeStatsAvailable || awayStatsAvailable);
  }
}

function handleDefenseFieldClick(event) {
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

function handleDefenseBenchClick(event) {
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

function applyDefenseSelectionHighlights() {
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

function updateDefenseBenchAvailability() {
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
    const markIneligible = canSubBase
      && hasLineupSelection
      && hasBenchPlayer
      && Boolean(lineupPositionKey)
      && !eligibleForPosition;

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

function updateDefenseSelectionInfo() {
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
    stateCache.defenseContext.canSub
    && Boolean(lineupPlayer)
    && Boolean(benchPlayer)
    && (!lineupPositionKey || benchEligible);
  if (elements.defenseApplyButton) {
    elements.defenseApplyButton.disabled = !canApply;
  }

  applyDefenseSelectionHighlights();
}

function updateDefensePanel(defenseTeam, gameState) {
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
  stateCache.defenseContext.lineup = lineupMap;
  stateCache.defenseContext.bench = benchMap;

  if (elements.defenseField) {
    elements.defenseField.innerHTML = '';
    const assigned = new Map();
    const extras = [];

    lineup.forEach((player) => {
      const key = normalizePositionKey(player.position_key || player.position);
      if (key && FIELD_POSITION_KEYS.has(key) && !assigned.has(key)) {
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
      if (player) {
        button.dataset.lineupIndex = player.index;
        const eligibleText = (player.eligible || []).join(', ') || '-';
        button.innerHTML = `
          <span class="position-label">${slot.label}</span>
          <strong>${player.name}</strong>
          <span>${eligibleText}</span>
        `;
        button.disabled = !gameState.active;
      } else {
        button.dataset.lineupIndex = '';
        button.innerHTML = `
          <span class="position-label">${slot.label}</span>
          <strong>空席</strong>
          <span>-</span>
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
          const eligibleText = (player.eligible || []).join(', ') || '-';
          button.innerHTML = `
            <strong>${player.name}</strong>
            <span class="eligible-label">現在: ${player.position || '-'}</span>
            <span class="eligible-positions">適性: ${eligibleText}</span>
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
        const eligibleText = (player.eligible || []).join(', ') || '-';
        button.innerHTML = `
          <strong>${player.name}</strong>
          <span class="eligible-label">守備適性</span>
          <span class="eligible-positions">${eligibleText}</span>
        `;
        button.disabled = !(gameState.active && stateCache.defenseContext.canSub);
        elements.defenseBench.appendChild(button);
      });
    }
  }

  updateDefenseBenchAvailability();
  applyDefenseSelectionHighlights();
}

function updateStatsPanel(state) {
  if (!state) return;
  const teams = state.teams || {};
  const availableTeams = ['away', 'home'].filter((key) => teams[key]?.stats);

  elements.statsTeamButtons.forEach((button) => {
    button.disabled = !availableTeams.includes(button.dataset.statsTeam);
  });

  if (!availableTeams.length) {
    if (elements.statsTableHead) {
      elements.statsTableHead.innerHTML = '';
    }
    if (elements.statsTableBody) {
      elements.statsTableBody.innerHTML = '';
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = BATTING_COLUMNS.length;
      td.textContent = '成績データがありません。';
      td.classList.add('empty');
      tr.appendChild(td);
      elements.statsTableBody.appendChild(tr);
    }
    if (elements.statsTitle) {
      elements.statsTitle.textContent = '成績データがありません';
    }
    elements.statsTypeButtons.forEach((button) => button.classList.remove('active'));
    return;
  }

  if (!availableTeams.includes(stateCache.statsView.team)) {
    stateCache.statsView.team = availableTeams[0];
  }
  if (!['batting', 'pitching'].includes(stateCache.statsView.type)) {
    stateCache.statsView.type = 'batting';
  }

  const teamKey = stateCache.statsView.team;
  const viewType = stateCache.statsView.type;
  const columns = viewType === 'pitching' ? PITCHING_COLUMNS : BATTING_COLUMNS;
  const teamData = teams[teamKey] || {};
  const stats = teamData.stats || {};
  const rows = stats[viewType] || [];

  elements.statsTeamButtons.forEach((button) => {
    const isActive = button.dataset.statsTeam === teamKey;
    button.classList.toggle('active', isActive);
  });
  elements.statsTypeButtons.forEach((button) => {
    const isActive = button.dataset.statsType === viewType;
    button.classList.toggle('active', isActive);
  });

  if (elements.statsTableHead) {
    elements.statsTableHead.innerHTML = '';
    columns.forEach((column) => {
      const th = document.createElement('th');
      th.textContent = column.label;
      elements.statsTableHead.appendChild(th);
    });
  }

  if (elements.statsTableBody) {
    elements.statsTableBody.innerHTML = '';
    if (!rows.length) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = columns.length;
      td.textContent = '成績データがありません。';
      td.classList.add('empty');
      tr.appendChild(td);
      elements.statsTableBody.appendChild(tr);
    } else {
      rows.forEach((row) => {
        const tr = document.createElement('tr');
        columns.forEach((column) => {
          const td = document.createElement('td');
          const value = row[column.key];
          td.textContent = value != null && value !== '' ? value : '-';
          tr.appendChild(td);
        });
        elements.statsTableBody.appendChild(tr);
      });
    }
  }

  if (elements.statsTitle) {
    const teamName = teamData?.name || (teamKey === 'home' ? 'Home' : 'Away');
    const typeLabel = viewType === 'pitching' ? '投球成績' : '打撃成績';
    elements.statsTitle.textContent = `${teamName} の${typeLabel}`;
  }
}

function showStatus(message, level = 'danger') {
  elements.statusMessage.textContent = message;
  elements.statusMessage.classList.remove('danger', 'success', 'info');
  elements.statusMessage.classList.add(level);
}

async function bootstrap() {
  initEventListeners();
  try {
    const initialState = await apiRequest('/api/game/state');
    render(initialState);
  } catch (err) {
    console.error(err);
    showStatus('初期状態の取得に失敗しました。ページを再読み込みしてください。', 'danger');
  }
}

bootstrap();
