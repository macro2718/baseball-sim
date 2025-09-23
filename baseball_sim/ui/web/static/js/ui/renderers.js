import { CONFIG, BATTING_COLUMNS, PITCHING_COLUMNS } from '../config.js';
import { elements } from '../dom.js';
import {
  stateCache,
  getLineupPlayer,
  getBenchPlayer,
  getLineupPositionKey,
  canBenchPlayerCoverPosition,
} from '../state.js';
import {
  escapeHtml,
  formatRosterStat,
  numberOrZero,
  renderPositionList,
  renderPositionToken,
  setInsightText,
} from '../utils.js';
import {
  renderDefensePanel,
  updateDefenseSelectionInfo,
  updateDefenseBenchAvailability,
  applyDefenseSelectionHighlights,
  resetDefenseSelectionsIfUnavailable,
} from './defensePanel.js';
import { setStatusMessage } from './status.js';

function setInsightsVisibility(visible) {
  const { insightGrid } = elements;
  if (!insightGrid) return;
  if (visible) {
    insightGrid.classList.remove('hidden');
  } else {
    insightGrid.classList.add('hidden');
  }
}

export function updateOutsIndicator(outs) {
  if (!elements.outsIndicator) return;
  const numericValue = Number(outs);
  const numeric = Number.isFinite(numericValue) ? numericValue : 0;
  const clamped = Math.max(0, Math.min(Math.trunc(numeric), CONFIG.maxOutsDisplay));
  const dots = [];
  for (let i = 0; i < CONFIG.maxOutsDisplay; i += 1) {
    const active = i < clamped;
    dots.push(`<span class="out-dot${active ? ' active' : ''}" aria-hidden="true"></span>`);
  }
  const announcement = `アウトカウント: ${clamped}`;
  elements.outsIndicator.innerHTML = `
    <div class="outs-dots" aria-hidden="true">${dots.join('')}</div>
    <span class="outs-label">OUT</span>
    <span class="visually-hidden">${announcement}</span>
  `;
  elements.outsIndicator.setAttribute('aria-label', announcement);
  elements.outsIndicator.dataset.outs = String(clamped);
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

function updateRosters(tbody, players) {
  if (!tbody) return;
  tbody.innerHTML = '';
  players.forEach((player) => {
    const tr = document.createElement('tr');
    if (player.is_current_batter) {
      tr.classList.add('active');
    }
    const orderLabel = escapeHtml(player.order ?? '');
    const positionHtml = renderPositionToken(player.position, player.pitcher_type);
    const nameHtml = escapeHtml(player.name ?? '-');
    const avgValue = formatRosterStat(player.avg, '-');
    const hrValue = formatRosterStat(player.hr, '-');
    const rbiValue = formatRosterStat(player.rbi, '-');
    tr.innerHTML = `
      <td>${orderLabel}</td>
      <td>${positionHtml}</td>
      <td class="player-name">${nameHtml}</td>
      <td class="stat-col">${avgValue}</td>
      <td class="stat-col">${hrValue}</td>
      <td class="stat-col">${rbiValue}</td>
    `;
    tbody.appendChild(tr);
  });
}

function updateBench(listEl, players) {
  if (!listEl) return;
  listEl.innerHTML = '';

  const section = listEl.closest('.bench-section');
  if (section) {
    section.classList.add('hidden');
    section.setAttribute('aria-hidden', 'true');
  }
}

function updatePitchers(listEl, pitchers) {
  if (!listEl) return;
  listEl.innerHTML = '';

  const visiblePitchers = Array.isArray(pitchers)
    ? pitchers.filter((pitcher) => {
        if (!pitcher) return false;
        if (pitcher.is_current) return true;
        if ('has_entered_game' in pitcher) return Boolean(pitcher.has_entered_game);
        if ('has_played' in pitcher) return Boolean(pitcher.has_played);
        return false;
      })
    : [];

  visiblePitchers.forEach((pitcher) => {
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

function updateScoreboard(gameState, teams) {
  if (!elements.scoreboard) return;

  if (!gameState || !gameState.active) {
    elements.scoreboard.innerHTML = '<p>試合はまだ開始されていません。</p>';
    return;
  }

  const inningScores = gameState.inning_scores || { home: [], away: [] };
  const innings = Math.max(
    gameState.max_innings || 0,
    inningScores.away.length,
    inningScores.home.length,
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
    const scores = inningScores[teamKey] || [];
    const totalRuns = gameState.score?.[teamKey] ?? 0;
    const totalHits = hits?.[teamKey] ?? 0;
    const totalErrors = errors?.[teamKey] ?? 0;
    let row = `<tr><td class="team-name">${teamName}</td>`;
    const isHomeTeam = teamKey === 'home';
    const isTopHalf = String(gameState.half || '').toLowerCase() === 'top';
    const inningNumber = Number(gameState.inning);
    const currentInningIndex = Number.isFinite(inningNumber) && inningNumber > 0 ? inningNumber - 1 : null;
    for (let i = 0; i < innings; i += 1) {
      const value = scores[i];
      let displayValue = value ?? '';
      if (
        isHomeTeam &&
        isTopHalf &&
        currentInningIndex !== null &&
        i === currentInningIndex &&
        (value === 0 || value === '0')
      ) {
        displayValue = '';
      }
      row += `<td>${displayValue}</td>`;
    }
    row += `<td>${totalRuns}</td><td>${totalHits}</td><td>${totalErrors}</td></tr>`;
    return row;
  };

  html += renderRow('away');
  html += renderRow('home');
  html += '</tbody></table>';

  elements.scoreboard.innerHTML = html;
}

function updateAnalyticsPanel(gameState) {
  if (!elements.insightRunRate) return;

  const resetInsights = () => {
    setInsightText(elements.insightRunRate, '--');
    setInsightText(elements.insightInningsSample, 'イニングサンプル: 0');
    if (elements.insightBasePressure) {
      setInsightText(elements.insightBasePressure, '--');
      elements.insightBasePressure.removeAttribute('data-intensity');
    }
    setInsightText(elements.insightBaseCount, '0 / 3');
    if (elements.insightRunDiff) {
      setInsightText(elements.insightRunDiff, '--');
      elements.insightRunDiff.dataset.trend = 'neutral';
    }
    if (elements.insightProgressFill) {
      elements.insightProgressFill.style.width = '0%';
    }
    if (elements.insightProgressLabel) {
      elements.insightProgressLabel.textContent = '0%';
    }
    if (elements.insightMeter) {
      elements.insightMeter.setAttribute('aria-label', 'ゲーム進行度 0%');
    }
  };

  if (!gameState || !gameState.active) {
    resetInsights();
    return;
  }

  const score = gameState.score || {};
  const homeRuns = numberOrZero(score.home);
  const awayRuns = numberOrZero(score.away);
  const totalRuns = homeRuns + awayRuns;

  const inningsHome = Array.isArray(gameState.inning_scores?.home)
    ? gameState.inning_scores.home.length
    : 0;
  const inningsAway = Array.isArray(gameState.inning_scores?.away)
    ? gameState.inning_scores.away.length
    : 0;
  const inningsSample = Math.max(inningsHome, inningsAway, 1);
  const runRate = totalRuns / inningsSample;

  setInsightText(elements.insightRunRate, runRate.toFixed(2));
  setInsightText(elements.insightInningsSample, `イニングサンプル: ${inningsSample}`);

  const bases = Array.isArray(gameState.bases) ? gameState.bases : [];
  const occupiedBases = bases.reduce(
    (count, base) => (base && base.occupied ? count + 1 : count),
    0,
  );
  const basePressure = Math.round((occupiedBases / 3) * 100);
  if (elements.insightBasePressure) {
    setInsightText(elements.insightBasePressure, `${basePressure}%`);
    let intensity = 'low';
    if (basePressure >= 67) {
      intensity = 'high';
    } else if (basePressure >= 34) {
      intensity = 'medium';
    }
    elements.insightBasePressure.dataset.intensity = intensity;
  }
  setInsightText(elements.insightBaseCount, `${occupiedBases} / 3`);

  if (elements.insightRunDiff) {
    const runDiff = homeRuns - awayRuns;
    const formattedDiff = runDiff > 0 ? `+${runDiff}` : runDiff < 0 ? `${runDiff}` : '±0';
    setInsightText(elements.insightRunDiff, formattedDiff);
    elements.insightRunDiff.dataset.trend =
      runDiff > 0 ? 'positive' : runDiff < 0 ? 'negative' : 'neutral';
  }

  const maxInnings = Math.max(numberOrZero(gameState.max_innings), 1);
  const inningNumber = Math.max(numberOrZero(gameState.inning), 1);
  const rawHalfLabel = String(gameState.half_label || '').toLowerCase();
  const rawHalf = rawHalfLabel || String(gameState.half || '').toLowerCase();
  let halfFraction = 0.5;
  if (/(裏|bottom|bot|後攻)/.test(rawHalf)) {
    halfFraction = 1;
  } else if (/(mid)/.test(rawHalf)) {
    halfFraction = 0.75;
  } else if (/(end)/.test(rawHalf)) {
    halfFraction = 1;
  } else if (/(表|top|先攻)/.test(rawHalf)) {
    halfFraction = 0.5;
  } else if (rawHalf.startsWith('b')) {
    halfFraction = 1;
  } else if (rawHalf.startsWith('t')) {
    halfFraction = 0.5;
  }

  const rawProgress = ((inningNumber - 1) + halfFraction) / maxInnings;
  const clampedProgress = gameState.game_over ? 1 : Math.min(Math.max(rawProgress, 0), 1);
  const progressPercentRaw = Math.round(Math.max(rawProgress, 0) * 100);
  const fillPercent = Math.round(clampedProgress * 100);

  if (elements.insightProgressFill) {
    elements.insightProgressFill.style.width = `${Math.min(Math.max(fillPercent, 0), 100)}%`;
  }
  if (elements.insightProgressLabel) {
    elements.insightProgressLabel.textContent = gameState.game_over ? '試合終了' : `${progressPercentRaw}%`;
  }
  if (elements.insightMeter) {
    elements.insightMeter.setAttribute(
      'aria-label',
      `ゲーム進行度 ${gameState.game_over ? '100%' : `${progressPercentRaw}%`}`,
    );
  }
}

function updateStrategyControls(gameState, teams) {
  const {
    pinchPlayer,
    pinchButton,
    pinchCurrentBatter,
    openOffenseButton,
    offensePinchMenuButton,
    openDefenseButton,
    openStatsButton,
    defenseSubMenuButton,
    pitcherMenuButton,
    pitcherSelect,
    pitcherButton,
  } = elements;

  const isActive = Boolean(gameState.active);
  const isGameOver = Boolean(gameState.game_over);

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
      let positionSegment = '';
      if (currentBatter.position && currentBatter.position !== '-') {
        const positionHtml = renderPositionToken(
          currentBatter.position,
          currentBatter.pitcher_type,
        );
        positionSegment = positionHtml ? `${positionHtml} ` : '';
      }
      pinchCurrentBatter.innerHTML = `現在の打者: ${orderLabel}${positionSegment}${escapeHtml(
        currentBatter.name ?? '-',
      )}`;
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

    const canPinch = isActive && !isGameOver && Boolean(currentBatter) && offenseBench.length > 0;
    pinchButton.disabled = !canPinch || isGameOver;
    pinchPlayer.disabled = !canPinch || isGameOver;
    if (!canPinch || isGameOver) {
      pinchPlayer.value = '';
    }
    if (isGameOver && pinchButton) {
      pinchButton.textContent = 'Game Over';
    } else if (pinchButton) {
      pinchButton.textContent = '代打';
    }
  }

  if (offensePinchMenuButton) {
    const canPinch =
      isActive && !isGameOver && Boolean(currentBatter) && (offenseBench?.length ?? 0) > 0;
    offensePinchMenuButton.disabled = !canPinch || isGameOver;
    offensePinchMenuButton.textContent = isGameOver ? 'Game Over' : '代打戦略';
  }

  if (openOffenseButton) {
    openOffenseButton.disabled = !isActive || isGameOver;
    openOffenseButton.textContent = isGameOver ? 'Game Over' : '攻撃戦略';
  }

  const defenseTeam = gameState.defense ? teams[gameState.defense] : null;
  const defenseLineup = defenseTeam?.lineup || [];
  const defenseBenchPlayers = defenseTeam?.bench || [];
  const pitcherOptions = defenseTeam?.pitcher_options || [];

  resetDefenseSelectionsIfUnavailable(defenseLineup, defenseBenchPlayers);

  const canDefenseSub =
    isActive && !isGameOver && defenseLineup.length > 0 && defenseBenchPlayers.length > 0;

  stateCache.defenseContext.canSub = canDefenseSub && !isGameOver;
  renderDefensePanel(defenseTeam, gameState);
  updateDefenseSelectionInfo();

  if (openDefenseButton) {
    openDefenseButton.disabled = !isActive || isGameOver;
    openDefenseButton.textContent = isGameOver ? 'Game Over' : '守備戦略';
  }
  if (!isActive || isGameOver) {
    updateDefenseBenchAvailability();
    applyDefenseSelectionHighlights();
  }
  if (defenseSubMenuButton) {
    defenseSubMenuButton.disabled = !canDefenseSub || isGameOver;
    defenseSubMenuButton.textContent = isGameOver ? 'Game Over' : '守備交代';
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

    const canChangePitcher = isActive && !isGameOver && pitcherOptions.length > 0;
    pitcherButton.disabled = !canChangePitcher || isGameOver;
    pitcherSelect.disabled = !canChangePitcher || isGameOver;
    if (!canChangePitcher || isGameOver) {
      pitcherSelect.value = '';
    }
    pitcherButton.textContent = isGameOver ? 'Game Over' : '投手交代';

    if (pitcherMenuButton) {
      pitcherMenuButton.disabled = !canChangePitcher || isGameOver;
      pitcherMenuButton.textContent = isGameOver ? 'Game Over' : '投手交代';
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

export function renderGame(gameState, teams, log) {
  updateAnalyticsPanel(gameState);
  const isActiveGame = Boolean(gameState && gameState.active);
  setInsightsVisibility(isActiveGame);

  if (!isActiveGame) {
    elements.gameScreen.classList.add('hidden');
    elements.titleScreen.classList.remove('hidden');
    updateScoreboard(gameState, teams);
    updateOutsIndicator(0);
    elements.actionWarning.textContent = '';
    elements.swingButton.disabled = true;
    elements.buntButton.disabled = true;
    updateRosters(elements.offenseRoster, []);
    updateRosters(elements.defenseRoster, []);
    updateBench(elements.offenseBench, []);
    updateBench(elements.defenseBenchList, []);
    updateLog(log || []);
    elements.defenseErrors.classList.add('hidden');
    elements.defenseErrors.textContent = '';
    updateStrategyControls(gameState || {}, teams || {});
    return;
  }

  elements.titleScreen.classList.add('hidden');
  elements.gameScreen.classList.remove('hidden');

  updateScoreboard(gameState, teams);
  elements.situationText.textContent = gameState.situation || '';
  elements.halfIndicator.textContent = `${gameState.half_label} ${gameState.inning}`;
  updateOutsIndicator(gameState.outs);
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

  if (gameState.game_over) {
    elements.swingButton.disabled = true;
    elements.buntButton.disabled = true;
    elements.swingButton.textContent = 'Game Over';
    elements.buntButton.textContent = 'Game Over';
    elements.actionWarning.textContent = 'ゲーム終了 - 新しい試合を開始するか、タイトルに戻ってください';
  } else {
    elements.swingButton.disabled = !gameState.actions?.swing;
    elements.buntButton.disabled = !gameState.actions?.bunt;
    elements.swingButton.textContent = '通常打撃';
    elements.buntButton.textContent = 'バント';
    elements.actionWarning.textContent = gameState.action_block_reason || '';
  }

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

export function renderTitle(titleState) {
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

export function updateStatsPanel(state) {
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

export function render(data) {
  stateCache.data = data;
  setStatusMessage(data.notification);
  renderTitle(data.title);
  renderGame(data.game, data.teams, data.log);
  updateStatsPanel(data);
}
