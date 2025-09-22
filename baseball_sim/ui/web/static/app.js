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
  homePitchers: document.querySelector('#home-pitchers ul'),
  awayPitchers: document.querySelector('#away-pitchers ul'),
  baseState: document.getElementById('base-state'),
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
    updateLog(log || []);
    elements.defenseErrors.classList.add('hidden');
    elements.defenseErrors.textContent = '';
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

  updateRosters(
    elements.offenseRoster,
    gameState.offense ? teams[gameState.offense]?.lineup || [] : [],
  );
  updateRosters(
    elements.defenseRoster,
    gameState.defense ? teams[gameState.defense]?.lineup || [] : [],
  );

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
}

function updateScoreboard(gameState, teams) {
  if (!elements.scoreboard) return;

  if (!gameState.active) {
    elements.scoreboard.innerHTML = '<p>試合はまだ開始されていません。</p>';
    return;
  }

  const innings = Math.max(
    gameState.inning_scores.away.length,
    gameState.inning_scores.home.length,
  );

  let html = '<table><thead><tr><th>Team</th>';
  for (let i = 0; i < innings; i += 1) {
    html += `<th>${i + 1}</th>`;
  }
  html += '<th>R</th></tr></thead><tbody>';

  const awayName = teams.away?.name || 'Away';
  html += `<tr><td class="team-name">${awayName}</td>`;
  for (let i = 0; i < innings; i += 1) {
    const value = gameState.inning_scores.away[i];
    html += `<td>${value ?? ''}</td>`;
  }
  html += `<td>${gameState.score.away}</td></tr>`;

  const homeName = teams.home?.name || 'Home';
  html += `<tr><td class="team-name">${homeName}</td>`;
  for (let i = 0; i < innings; i += 1) {
    const value = gameState.inning_scores.home[i];
    html += `<td>${value ?? ''}</td>`;
  }
  html += `<td>${gameState.score.home}</td></tr>`;
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
