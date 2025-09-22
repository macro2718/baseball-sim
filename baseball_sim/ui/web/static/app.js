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
  defenseBench: document.getElementById('defense-bench'),
  homePitchers: document.querySelector('#home-pitchers ul'),
  awayPitchers: document.querySelector('#away-pitchers ul'),
  baseState: document.getElementById('base-state'),
  pinchTarget: document.getElementById('pinch-target'),
  pinchPlayer: document.getElementById('pinch-player'),
  pinchButton: document.getElementById('pinch-hit-button'),
  defenseTarget: document.getElementById('defense-target'),
  defensePlayer: document.getElementById('defense-player'),
  defenseButton: document.getElementById('defense-sub-button'),
  pitcherSelect: document.getElementById('pitcher-select'),
  pitcherButton: document.getElementById('change-pitcher-button'),
};

const stateCache = {
  data: null,
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
  if (elements.defenseButton) {
    elements.defenseButton.addEventListener('click', handleDefenseSubstitution);
  }
  if (elements.pitcherButton) {
    elements.pitcherButton.addEventListener('click', handlePitcherChange);
  }
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
  if (!elements.pinchTarget || !elements.pinchPlayer) return;
  const lineupValue = elements.pinchTarget.value;
  const benchValue = elements.pinchPlayer.value;
  if (!lineupValue || !benchValue) {
    showStatus('代打対象とベンチ選手を選択してください。', 'danger');
    return;
  }

  const lineupIndex = Number(lineupValue);
  const benchIndex = Number(benchValue);
  if (Number.isNaN(lineupIndex) || Number.isNaN(benchIndex)) {
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
  if (!elements.defenseTarget || !elements.defensePlayer) return;
  const lineupValue = elements.defenseTarget.value;
  const benchValue = elements.defensePlayer.value;
  if (!lineupValue || !benchValue) {
    showStatus('守備交代の対象と選手を選択してください。', 'danger');
    return;
  }

  const lineupIndex = Number(lineupValue);
  const benchIndex = Number(benchValue);
  if (Number.isNaN(lineupIndex) || Number.isNaN(benchIndex)) {
    showStatus('選択内容を解釈できませんでした。', 'danger');
    return;
  }

  try {
    const payload = await apiRequest('/api/strategy/defense_substitution', {
      method: 'POST',
      body: JSON.stringify({ lineup_index: lineupIndex, bench_index: benchIndex }),
    });
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
    updateBench(elements.defenseBench, [], 'ゲーム開始後に表示されます');
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
  updateBench(elements.defenseBench, defenseTeam?.bench || []);

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
    pinchTarget,
    pinchPlayer,
    pinchButton,
    defenseTarget,
    defensePlayer,
    defenseButton,
    pitcherSelect,
    pitcherButton,
  } = elements;

  const isActive = Boolean(gameState.active);

  const offenseTeam = gameState.offense ? teams[gameState.offense] : null;
  const offenseLineup = offenseTeam?.lineup || [];
  const offenseBench = offenseTeam?.bench || [];

  if (pinchTarget && pinchPlayer && pinchButton) {
    const lineupPlaceholder = offenseLineup.length
      ? '代打対象を選択'
      : '出場中の選手がいません';
    const benchPlaceholder = offenseBench.length
      ? 'ベンチ選手を選択'
      : '選択可能な選手がいません';

    populateSelect(
      pinchTarget,
      offenseLineup.map((player) => ({
        value: player.index,
        label: `${player.order}. ${player.position || '-'} ${player.name}`,
      })),
      lineupPlaceholder,
    );

    populateSelect(
      pinchPlayer,
      offenseBench.map((player) => ({
        value: player.index,
        label: `${player.name} (${(player.eligible || []).join(', ') || '-'})`,
      })),
      benchPlaceholder,
    );

    const canPinch = isActive && offenseLineup.length > 0 && offenseBench.length > 0;
    pinchButton.disabled = !canPinch;
    pinchTarget.disabled = !canPinch;
    pinchPlayer.disabled = !canPinch;
    if (!canPinch) {
      pinchTarget.value = '';
      pinchPlayer.value = '';
    }
  }

  const defenseTeam = gameState.defense ? teams[gameState.defense] : null;
  const defenseLineup = defenseTeam?.lineup || [];
  const defenseBenchPlayers = defenseTeam?.bench || [];
  const pitcherOptions = defenseTeam?.pitcher_options || [];

  if (defenseTarget && defensePlayer && defenseButton) {
    const lineupPlaceholder = defenseLineup.length
      ? '守備交代する選手を選択'
      : '守備につく選手がいません';
    const benchPlaceholder = defenseBenchPlayers.length
      ? 'ベンチ選手を選択'
      : '選択可能な選手がいません';

    populateSelect(
      defenseTarget,
      defenseLineup.map((player) => ({
        value: player.index,
        label: `${player.position || '-'} ${player.name}`,
      })),
      lineupPlaceholder,
    );

    populateSelect(
      defensePlayer,
      defenseBenchPlayers.map((player) => ({
        value: player.index,
        label: `${player.name} (${(player.eligible || []).join(', ') || '-'})`,
      })),
      benchPlaceholder,
    );

    const canSubDefense = isActive && defenseLineup.length > 0 && defenseBenchPlayers.length > 0;
    defenseButton.disabled = !canSubDefense;
    defenseTarget.disabled = !canSubDefense;
    defensePlayer.disabled = !canSubDefense;
    if (!canSubDefense) {
      defenseTarget.value = '';
      defensePlayer.value = '';
    }
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
