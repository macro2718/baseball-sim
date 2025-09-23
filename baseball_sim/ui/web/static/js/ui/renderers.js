import {
  CONFIG,
  BATTING_COLUMNS,
  PITCHING_COLUMNS,
  ABILITY_BATTING_COLUMNS,
  ABILITY_PITCHING_COLUMNS,
  FIELD_POSITIONS,
  FIELD_POSITION_KEYS,
} from '../config.js';
import { elements } from '../dom.js';
import {
  stateCache,
  getLineupPlayer,
  getBenchPlayer,
  getLineupPositionKey,
  canBenchPlayerCoverPosition,
  normalizePositionKey,
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

function hasAbilityData(team) {
  if (!team || !team.traits) return false;
  const { batting, pitching } = team.traits;
  const battingCount = Array.isArray(batting) ? batting.length : 0;
  const pitchingCount = Array.isArray(pitching) ? pitching.length : 0;
  return battingCount > 0 || pitchingCount > 0;
}

const FIELD_ALIGNMENT_SLOTS = FIELD_POSITIONS.filter((slot) => slot.key !== 'DH');

const BASE_LABELS = ['一塁', '二塁', '三塁'];

function formatRunnerSpeed(info) {
  if (!info) return null;
  const display = typeof info.speed_display === 'string' ? info.speed_display.trim() : '';
  if (display && display !== '-') {
    return display;
  }
  const raw = info.speed ?? info.runner_speed;
  const numeric = Number(raw);
  if (Number.isFinite(numeric)) {
    return `${numeric.toFixed(2)}秒`;
  }
  return null;
}

function parseFieldingValue(raw) {
  if (raw === null || raw === undefined) return null;
  if (typeof raw === 'number') {
    return Number.isFinite(raw) ? raw : null;
  }
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    const numeric = Number.parseFloat(trimmed);
    return Number.isFinite(numeric) ? numeric : null;
  }
  return null;
}

function getFieldingTier(value) {
  if (!Number.isFinite(value)) {
    return 'unknown';
  }
  if (value >= 85) return 'elite';
  if (value >= 70) return 'strong';
  if (value >= 55) return 'average';
  return 'developing';
}

function getFieldingMetrics(player) {
  if (!player) {
    return { label: '-', value: null };
  }
  const direct = parseFieldingValue(player.fielding_value);
  const fromSkill = parseFieldingValue(player.fielding_skill);
  const fromRating = parseFieldingValue(player.fielding_rating);
  const numeric = direct ?? fromSkill ?? fromRating;
  let label = typeof player.fielding_rating === 'string' ? player.fielding_rating.trim() : '';
  if (!label || label === '-') {
    if (Number.isFinite(numeric)) {
      label = Number.isInteger(numeric) ? String(Math.trunc(numeric)) : numeric.toFixed(1);
    } else {
      label = '-';
    }
  }
  return { label, value: numeric };
}

function updateDefenseAlignment(gameState, teams) {
  const container = elements.defenseAlignment;
  if (!container) return;

  container.innerHTML = '';
  container.classList.add('hidden');
  container.setAttribute('aria-hidden', 'true');
  container.setAttribute('aria-label', '守備配置');
  delete container.dataset.team;

  if (!gameState || !gameState.active) {
    return;
  }

  const defenseKey = gameState.defense;
  const defenseTeam = defenseKey && teams ? teams[defenseKey] : null;
  if (!defenseTeam) {
    return;
  }

  const lineup = Array.isArray(defenseTeam.lineup) ? defenseTeam.lineup : [];
  const assignments = new Map();

  lineup.forEach((player) => {
    if (!player) return;
    const key = normalizePositionKey(player.position_key || player.position);
    if (!key || key === 'DH') return;
    if (!FIELD_POSITION_KEYS.has(key)) return;
    if (assignments.has(key)) return;
    assignments.set(key, player);
  });

  if (!assignments.has('P')) {
    const pitcherFromTeam = Array.isArray(defenseTeam.pitchers)
      ? defenseTeam.pitchers.find((pitcher) => pitcher && pitcher.is_current)
      : null;
    const pitcherFromGame = gameState && gameState.current_pitcher ? gameState.current_pitcher : null;
    const currentPitcher = pitcherFromTeam || pitcherFromGame;
    if (currentPitcher) {
      const pitcherTypeRaw = currentPitcher.pitcher_type || currentPitcher.position || 'P';
      const pitcherType = typeof pitcherTypeRaw === 'string' ? pitcherTypeRaw.toUpperCase() : 'P';
      const staminaValue = currentPitcher.stamina ?? currentPitcher.current_stamina;
      assignments.set('P', {
        name: currentPitcher.name || '-',
        position: 'P',
        position_key: 'P',
        pitcher_type: pitcherType,
        stamina: staminaValue,
        current_stamina: staminaValue,
      });
    }
  }

  let rendered = 0;

  FIELD_ALIGNMENT_SLOTS.forEach((slot) => {
    const player = assignments.get(slot.key);
    if (!player) return;

    const slotEl = document.createElement('div');
    slotEl.className = `alignment-slot ${slot.className}`;
    slotEl.dataset.position = slot.key;

    const { label, value } = getFieldingMetrics(player);
    const tier = getFieldingTier(value);
    if (tier) {
      slotEl.dataset.tier = tier;
    }

    let ratingText = label && label !== '-' ? `守備 ${label}` : '守備 -';
    if (slot.key === 'P') {
      const typeRaw = player.pitcher_type || player.position || 'P';
      const typeLabel = typeof typeRaw === 'string' ? typeRaw.toUpperCase() : 'P';
      const staminaRaw = player.stamina ?? player.current_stamina;
      const staminaNumeric = Number(staminaRaw);
      const staminaText = Number.isFinite(staminaNumeric) ? staminaNumeric.toFixed(1) : null;
      ratingText = `タイプ ${typeLabel}`;
      if (staminaText) {
        ratingText += `｜体力 ${staminaText}`;
      }
    }

    slotEl.innerHTML = `
      <div class="player-chip">
        <span class="pos-tag">${slot.label}</span>
        <div class="player-info">
          <span class="player-name">${escapeHtml(player.name ?? '-')}</span>
          <span class="player-rating">${escapeHtml(ratingText)}</span>
        </div>
      </div>
    `;
    container.appendChild(slotEl);
    rendered += 1;
  });

  if (rendered > 0) {
    container.classList.remove('hidden');
    container.setAttribute('aria-hidden', 'false');
    const teamName = defenseTeam.name || '';
    container.setAttribute('aria-label', teamName ? `守備配置 (${teamName})` : '守備配置');
    if (teamName) {
      container.dataset.team = teamName;
    }
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
    const info = (Array.isArray(bases) ? bases[baseIndex] : null) || {};
    const hasRunnerName = typeof info.runner === 'string' && info.runner.trim() !== '';
    const isOccupied = Boolean(info.occupied);
    const showRunnerChip = isOccupied && hasRunnerName;
    el.classList.toggle('occupied', isOccupied);
    if (isOccupied) {
      if (el.tabIndex !== 0) {
        el.tabIndex = 0;
      }
    } else if (el.tabIndex !== -1) {
      el.tabIndex = -1;
    }
    const indicator = el.querySelector('.base-indicator');
    if (indicator) {
      indicator.textContent = isOccupied ? '●' : '';
    }
    const runnerChip = el.querySelector('.runner-chip');
    if (runnerChip) {
      const nameEl = runnerChip.querySelector('.runner-name');
      const speedEl = runnerChip.querySelector('.runner-speed');
      if (showRunnerChip && nameEl && speedEl) {
        const speedText = formatRunnerSpeed(info);
        nameEl.textContent = info.runner;
        speedEl.textContent = speedText ? `スピード ${speedText}` : 'スピード -';
        runnerChip.classList.add('active');
        runnerChip.setAttribute('aria-hidden', 'false');
      } else {
        if (nameEl) nameEl.textContent = '';
        if (speedEl) speedEl.textContent = '';
        runnerChip.classList.remove('active');
        runnerChip.setAttribute('aria-hidden', 'true');
      }
    }

    const baseLabel = BASE_LABELS[baseIndex] || '塁';
    let ariaLabel = `${baseLabel}: 走者なし`;
    if (isOccupied) {
      if (hasRunnerName) {
        const speedText = formatRunnerSpeed(info);
        ariaLabel = `${baseLabel}: ${info.runner}${speedText ? `（スピード ${speedText}）` : ''}`;
      } else {
        ariaLabel = `${baseLabel}: 走者あり`;
      }
    }
    el.setAttribute('aria-label', ariaLabel);
    el.removeAttribute('title');
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

function clampStaminaPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(numeric)));
}

function formatStaminaLabel(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '--';
  }
  return `${Math.round(numeric)}%`;
}

function updateCurrentPitcherCard(cardEl, pitcher) {
  if (!cardEl) return;
  if (!pitcher) {
    cardEl.innerHTML = '<p class="pitcher-card-empty">現在の投手情報が取得できません。</p>';
    return;
  }

  const staminaPercent = clampStaminaPercent(pitcher.stamina);
  const staminaLabel = formatStaminaLabel(pitcher.stamina);
  const typeRaw = pitcher.pitcher_type ?? 'P';
  const nameRaw = pitcher.name ?? '-';
  const throwsRaw = pitcher.throws ? String(pitcher.throws) : '';
  const typeLabel = escapeHtml(typeRaw);
  const nameLabel = escapeHtml(nameRaw);
  const throwsLabel = throwsRaw ? escapeHtml(throwsRaw) : '';
  const throwsBlock = throwsLabel
    ? `
        <div class="pitcher-meta-block">
          <span class="pitcher-meta-label">投球腕</span>
          <span class="pitcher-meta-value">${throwsLabel}</span>
        </div>
      `
    : '';

  cardEl.innerHTML = `
    <div class="current-pitcher-header">
      <span class="card-label">現在の投手</span>
      <span class="pitcher-role-badge">${typeLabel}</span>
    </div>
    <div class="current-pitcher-body">
      <h4 class="pitcher-name">${nameLabel}</h4>
      <div class="pitcher-meta">
        <div class="pitcher-meta-block">
          <span class="pitcher-meta-label">スタミナ</span>
          <div class="stamina-meter" role="presentation">
            <span class="stamina-fill" style="width: ${staminaPercent}%"></span>
          </div>
          <span class="pitcher-meta-value stamina-value">${staminaLabel}</span>
        </div>
        <div class="pitcher-meta-block">
          <span class="pitcher-meta-label">タイプ</span>
          <span class="pitcher-meta-value">${typeLabel}</span>
        </div>
        ${throwsBlock}
      </div>
    </div>
  `;
}

function formatBatsLabel(rawBats) {
  if (!rawBats) return '';
  const normalized = String(rawBats).trim().toUpperCase();
  if (!normalized) return '';
  if (normalized === 'L') return '左打';
  if (normalized === 'R') return '右打';
  if (normalized === 'S' || normalized === 'B') return '両打';
  return normalized;
}

function renderBatsBadge(rawBats) {
  const label = formatBatsLabel(rawBats);
  if (!label) return '';
  const ariaLabel = `打席: ${label}`;
  return `<span class="pitcher-throws-badge batter-bats-badge" aria-label="${escapeHtml(
    ariaLabel,
  )}">${escapeHtml(label)}</span>`;
}

function updateCurrentBatterCard(cardEl, batter) {
  if (!cardEl) return;
  if (!batter) {
    cardEl.innerHTML = '<p class="pitcher-card-empty">現在の打者情報が取得できません。</p>';
    return;
  }

  const orderValue = Number.isInteger(batter.order) ? `${batter.order}番` : '-';
  const positionHtml =
    batter.position && batter.position !== '-' ? renderPositionToken(batter.position, batter.pitcher_type) : '';
  const batsBadge = renderBatsBadge(batter.bats);
  const tags = [positionHtml, batsBadge].filter(Boolean).join('');

  const nameLabel = escapeHtml(batter.name ?? '-');
  const orderLabel = escapeHtml(orderValue);
  const avgValue = formatRosterStat(batter.avg, '-');
  const hrValue = formatRosterStat(batter.hr, '-');
  const rbiValue = formatRosterStat(batter.rbi, '-');

  cardEl.innerHTML = `
    <div class="current-pitcher-header current-batter-header">
      <span class="card-label">現在の打者</span>
      <div class="current-batter-tags">${tags || ''}</div>
    </div>
    <div class="current-pitcher-body current-batter-body">
      <h4 class="pitcher-name current-batter-name">${nameLabel}</h4>
      <div class="pitcher-meta batter-meta">
        <div class="pitcher-meta-block batter-meta-block">
          <span class="pitcher-meta-label batter-meta-label">打順</span>
          <span class="pitcher-meta-value batter-meta-value">${orderLabel}</span>
        </div>
        <div class="pitcher-meta-block batter-meta-block">
          <span class="pitcher-meta-label batter-meta-label">AVG</span>
          <span class="pitcher-meta-value batter-meta-value">${avgValue}</span>
        </div>
        <div class="pitcher-meta-block batter-meta-block">
          <span class="pitcher-meta-label batter-meta-label">HR</span>
          <span class="pitcher-meta-value batter-meta-value">${hrValue}</span>
        </div>
        <div class="pitcher-meta-block batter-meta-block">
          <span class="pitcher-meta-label batter-meta-label">RBI</span>
          <span class="pitcher-meta-value batter-meta-value">${rbiValue}</span>
        </div>
      </div>
    </div>
  `;
}

function highlightPitcherCards(gridEl, selectedValue) {
  if (!gridEl) return;
  const normalized = selectedValue != null ? String(selectedValue) : '';
  const cards = gridEl.querySelectorAll('.pitcher-card');
  cards.forEach((card) => {
    const value = card.dataset.value || '';
    const isSelected = normalized !== '' && value === normalized;
    if (isSelected) {
      card.classList.add('selected');
      card.setAttribute('aria-pressed', 'true');
    } else {
      card.classList.remove('selected');
      card.setAttribute('aria-pressed', 'false');
    }
  });
}

function updatePitcherOptionGrid(gridEl, options, selectEl, helperEl) {
  if (!gridEl) return;

  gridEl.innerHTML = '';
  const hasOptions = Array.isArray(options) && options.length > 0;
  const selectionDisabled = !selectEl || Boolean(selectEl.disabled);

  if (!hasOptions) {
    const emptyMessage = document.createElement('p');
    emptyMessage.className = 'pitcher-card-empty';
    emptyMessage.textContent = '交代可能な投手がいません。';
    gridEl.appendChild(emptyMessage);
    highlightPitcherCards(gridEl, '');
    if (helperEl) {
      helperEl.textContent = '現在ブルペンで準備できる投手がいません。';
    }
    return;
  }

  options.forEach((pitcher) => {
    const optionValue = pitcher.index != null ? String(pitcher.index) : '';
    const staminaPercent = clampStaminaPercent(pitcher.stamina);
    const staminaLabel = formatStaminaLabel(pitcher.stamina);
    const typeRaw = pitcher.pitcher_type ?? 'P';
    const nameRaw = pitcher.name ?? '-';
    const throwsRaw = pitcher.throws ? String(pitcher.throws) : '';
    const typeLabel = escapeHtml(typeRaw);
    const nameLabel = escapeHtml(nameRaw);
    const throwsLabel = throwsRaw ? escapeHtml(throwsRaw) : '';

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'pitcher-card';
    button.dataset.value = optionValue;
    button.setAttribute('aria-pressed', 'false');
    button.setAttribute('role', 'listitem');
    button.title = `${nameRaw} (${typeRaw}${throwsRaw ? `/${throwsRaw}` : ''})`;
    button.innerHTML = `
      <div class="pitcher-card-top">
        <span class="pitcher-role-badge">${typeLabel}</span>
        ${throwsLabel ? `<span class="pitcher-throws-badge" aria-label="投球腕">${throwsLabel}</span>` : ''}
      </div>
      <h4 class="pitcher-card-name">${nameLabel}</h4>
      <div class="pitcher-card-bottom">
        <div class="stamina-meter" role="presentation">
          <span class="stamina-fill" style="width: ${staminaPercent}%"></span>
        </div>
        <span class="stamina-value">${staminaLabel}</span>
      </div>
    `;

    button.addEventListener('click', () => {
      if (!selectEl) return;
      selectEl.value = optionValue;
      const changeEvent = new Event('change', { bubbles: true });
      selectEl.dispatchEvent(changeEvent);
      highlightPitcherCards(gridEl, optionValue);
    });

    if (selectionDisabled) {
      button.disabled = true;
    }

    gridEl.appendChild(button);
  });

  if (selectEl && !selectEl.dataset.pitcherCardListener) {
    selectEl.addEventListener('change', () => {
      highlightPitcherCards(gridEl, selectEl.value);
    });
    selectEl.dataset.pitcherCardListener = 'true';
  }

  highlightPitcherCards(gridEl, selectEl ? selectEl.value : '');

  if (helperEl) {
    helperEl.textContent = selectionDisabled
      ? '現在は投手交代ができません。'
      : 'カードを選択するとここに反映されます。';
  }
}

function highlightPinchCards(gridEl, selectedValue) {
  if (!gridEl) return;
  const normalized = selectedValue != null ? String(selectedValue) : '';
  const cards = gridEl.querySelectorAll('.pinch-card');
  cards.forEach((card) => {
    const value = card.dataset.value || '';
    const isSelected = normalized !== '' && value === normalized;
    if (isSelected) {
      card.classList.add('selected');
      card.setAttribute('aria-pressed', 'true');
    } else {
      card.classList.remove('selected');
      card.setAttribute('aria-pressed', 'false');
    }
  });
}

function updatePinchOptionGrid(gridEl, options, selectEl, helperEl, helperMessage) {
  if (!gridEl) return;

  gridEl.innerHTML = '';
  const hasOptions = Array.isArray(options) && options.length > 0;
  const selectionDisabled = !selectEl || Boolean(selectEl.disabled);

  if (!hasOptions) {
    const emptyMessage = document.createElement('p');
    emptyMessage.className = 'pitcher-card-empty pinch-card-empty';
    emptyMessage.textContent = '代打に使える選手がいません。';
    gridEl.appendChild(emptyMessage);
    highlightPinchCards(gridEl, '');
    if (helperEl) {
      helperEl.textContent = helperMessage || '現在は代打を選択できません。';
    }
    return;
  }

  options.forEach((player) => {
    const optionValue = player.index != null ? String(player.index) : '';
    const nameRaw = player.name ?? '-';
    const positionHtml =
      player.position && player.position !== '-' ? renderPositionToken(player.position, player.pitcher_type) : '';
    const positionsList = renderPositionList(player.eligible || [], player.pitcher_type);
    const batsBadge = renderBatsBadge(player.bats);
    const nameLabel = escapeHtml(nameRaw);
    const avgValue = formatRosterStat(player.avg, '-');
    const hrValue = formatRosterStat(player.hr, '-');
    const rbiValue = formatRosterStat(player.rbi, '-');

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'pitcher-card pinch-card';
    button.dataset.value = optionValue;
    button.setAttribute('aria-pressed', 'false');
    button.setAttribute('role', 'listitem');
    button.title = `${nameRaw} | AVG ${player.avg ?? '-'} / HR ${player.hr ?? '-'} / RBI ${player.rbi ?? '-'}`;
    button.innerHTML = `
      <div class="pinch-card-top">
        <div class="pinch-card-heading">
          ${positionHtml || ''}
          <h4 class="pinch-card-name">${nameLabel}</h4>
        </div>
        ${batsBadge || ''}
      </div>
      <div class="pinch-card-stats">
        <div class="pinch-stat-block">
          <span class="pinch-stat-label">AVG</span>
          <span class="pinch-stat-value">${avgValue}</span>
        </div>
        <div class="pinch-stat-block">
          <span class="pinch-stat-label">HR</span>
          <span class="pinch-stat-value">${hrValue}</span>
        </div>
        <div class="pinch-stat-block">
          <span class="pinch-stat-label">RBI</span>
          <span class="pinch-stat-value">${rbiValue}</span>
        </div>
      </div>
      <div class="pinch-card-positions">
        <span class="pinch-stat-label pinch-positions-label">守備適性</span>
        <div class="pinch-position-list">${positionsList}</div>
      </div>
    `;

    button.addEventListener('click', () => {
      if (!selectEl || selectEl.disabled) return;
      selectEl.value = optionValue;
      const changeEvent = new Event('change', { bubbles: true });
      selectEl.dispatchEvent(changeEvent);
      highlightPinchCards(gridEl, optionValue);
    });

    if (selectionDisabled) {
      button.disabled = true;
    }

    gridEl.appendChild(button);
  });

  if (selectEl && !selectEl.dataset.pinchCardListener) {
    selectEl.addEventListener('change', () => {
      highlightPinchCards(gridEl, selectEl.value);
    });
    selectEl.dataset.pinchCardListener = 'true';
  }

  highlightPinchCards(gridEl, selectEl ? selectEl.value : '');

  if (helperEl) {
    if (helperMessage) {
      helperEl.textContent = helperMessage;
    } else {
      helperEl.textContent = selectionDisabled
        ? '現在は代打を選択できません。'
        : 'カードを選択するとここに反映されます。';
    }
  }
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
  const inningNumber = Number(gameState.inning);
  const currentInningIndex = Number.isFinite(inningNumber) && inningNumber > 0 ? inningNumber - 1 : null;
  const half = String(gameState.half || '').toLowerCase();
  const isTopHalf = half === 'top';
  const gameOver = Boolean(gameState.game_over);

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
    for (let i = 0; i < innings; i += 1) {
      const value = scores[i];
      let displayValue = value ?? '';
      if (isHomeTeam && currentInningIndex !== null && i === currentInningIndex) {
        if (gameOver && isTopHalf) {
          displayValue = 'x';
        } else if (isTopHalf && (value === 0 || value === '0' || value == null)) {
          displayValue = '';
        }
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
  if (!elements.insightInningRunExpectancy) return;

  const resetInsights = () => {
    setInsightText(elements.insightInningRunExpectancy, '--');
    if (elements.insightInningContext) {
      setInsightText(elements.insightInningContext, 'アウト: -- / 走者: --');
    }
    if (elements.insightOneRunProbability) {
      setInsightText(elements.insightOneRunProbability, '--');
      elements.insightOneRunProbability.removeAttribute('data-intensity');
    }
    if (elements.insightOneRunContext) {
      setInsightText(elements.insightOneRunContext, '走者指数: --');
    }
    if (elements.insightWinProbability) {
      setInsightText(elements.insightWinProbability, '--');
      elements.insightWinProbability.dataset.trend = 'neutral';
    }
    if (elements.insightWinContext) {
      setInsightText(elements.insightWinContext, '得点差: ±0');
    }
    if (elements.insightProbabilityFill) {
      elements.insightProbabilityFill.style.width = '0%';
    }
    if (elements.insightProbabilityLabel) {
      elements.insightProbabilityLabel.textContent = '--';
    }
    if (elements.insightProbabilityMeter) {
      elements.insightProbabilityMeter.setAttribute('aria-label', '勝利確率 --');
    }
  };

  if (!gameState || !gameState.active) {
    resetInsights();
    return;
  }

  const bases = Array.isArray(gameState.bases) ? gameState.bases : [];
  const occupiedBases = bases.reduce(
    (count, base) => (base && base.occupied ? count + 1 : count),
    0,
  );
  const outs = Math.min(Math.max(numberOrZero(gameState.outs), 0), 3);
  const baseWeights = [0.55, 0.7, 0.9];
  const baseThreatScore = bases.reduce((scoreAcc, base, index) => {
    if (base && base.occupied) {
      return scoreAcc + (baseWeights[index] ?? 0.45);
    }
    return scoreAcc;
  }, 0);
  const remainingOutsFactor = Math.max(0, (3 - outs) / 3);
  const inningRunExpectancy = Math.max(0, 0.15 + baseThreatScore + remainingOutsFactor * 0.8);

  setInsightText(elements.insightInningRunExpectancy, inningRunExpectancy.toFixed(2));
  if (elements.insightInningContext) {
    setInsightText(
      elements.insightInningContext,
      `アウト: ${outs} / 3 ・走者: ${occupiedBases}`,
    );
  }

  if (elements.insightOneRunProbability) {
    const scoringPressure = Math.max(
      0,
      Math.min(1, 0.1 + baseThreatScore / 2.4 + remainingOutsFactor * 0.6),
    );
    const probabilityPercent = Math.round(scoringPressure * 100);
    setInsightText(elements.insightOneRunProbability, `${probabilityPercent}%`);
    let intensity = 'low';
    if (scoringPressure >= 0.67) {
      intensity = 'high';
    } else if (scoringPressure >= 0.34) {
      intensity = 'medium';
    }
    elements.insightOneRunProbability.dataset.intensity = intensity;
    if (elements.insightOneRunContext) {
      setInsightText(
        elements.insightOneRunContext,
        `走者指数: ${baseThreatScore.toFixed(2)}`,
      );
    }
  }

  const score = gameState.score || {};
  const homeRuns = numberOrZero(score.home);
  const awayRuns = numberOrZero(score.away);
  const runDiff = homeRuns - awayRuns;

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
  const clampedProgress = Math.min(Math.max(rawProgress, 0), 1);
  const leverage = Math.max(0.25, 1 - Math.abs(0.5 - clampedProgress) * 1.2);

  let winProbability;
  if (gameState.game_over) {
    if (homeRuns > awayRuns) {
      winProbability = 1;
    } else if (homeRuns < awayRuns) {
      winProbability = 0;
    } else {
      winProbability = 0.5;
    }
  } else {
    const runDiffImpact = Math.max(-3, Math.min(3, runDiff)) * 0.09 * leverage;
    const offenseAdjustment = gameState.offense === 'home' ? 0.03 : -0.03;
    winProbability = 0.5 + runDiffImpact + offenseAdjustment;
  }
  winProbability = Math.min(Math.max(winProbability, 0), 1);

  if (elements.insightWinProbability) {
    const winPercent = Math.round(winProbability * 100);
    setInsightText(elements.insightWinProbability, `${winPercent}%`);
    let trend = 'neutral';
    if (winProbability >= 0.55) {
      trend = 'positive';
    } else if (winProbability <= 0.45) {
      trend = 'negative';
    }
    elements.insightWinProbability.dataset.trend = trend;
  }
  if (elements.insightWinContext) {
    const formattedDiff = runDiff > 0 ? `+${runDiff}` : runDiff < 0 ? `${runDiff}` : '±0';
    setInsightText(
      elements.insightWinContext,
      `得点差: ${formattedDiff} ・進行度: ${Math.round(clampedProgress * 100)}%`,
    );
  }

  const winPercentLabel = Math.round(winProbability * 100);
  if (elements.insightProbabilityFill) {
    elements.insightProbabilityFill.style.width = `${winPercentLabel}%`;
  }
  if (elements.insightProbabilityLabel) {
    elements.insightProbabilityLabel.textContent = gameState.game_over
      ? `最終値 ${winPercentLabel}%`
      : `推定 ${winPercentLabel}%`;
  }
  if (elements.insightProbabilityMeter) {
    elements.insightProbabilityMeter.setAttribute(
      'aria-label',
      `勝利確率 ${winPercentLabel}%`,
    );
  }
}

function updateStrategyControls(gameState, teams) {
  const {
    pinchPlayer,
    pinchButton,
    pinchCurrentCard,
    pinchOptionGrid,
    pinchSelectHelper,
    openOffenseButton,
    offensePinchMenuButton,
    openDefenseButton,
    openStatsButton,
    openAbilitiesButton,
    defenseSubMenuButton,
    pitcherMenuButton,
    currentPitcherCard,
    pitcherOptionGrid,
    pitcherSelectHelper,
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

  if (pinchCurrentCard) {
    updateCurrentBatterCard(pinchCurrentCard, currentBatter);
  }

  const canPinch = isActive && !isGameOver && Boolean(currentBatter) && offenseBench.length > 0;

  if (pinchPlayer) {
    const benchPlaceholder = !currentBatter
      ? '現在の打者が見つかりません'
      : offenseBench.length
      ? 'カードまたはリストから選択'
      : '選択可能な選手がいません';

    populateSelect(
      pinchPlayer,
      offenseBench.map((player) => ({
        value: player.index,
        label: `${player.name} (AVG ${player.avg ?? '-'}, HR ${player.hr ?? '-'})`,
      })),
      benchPlaceholder,
    );

    pinchPlayer.disabled = !canPinch;
    if (!canPinch) {
      pinchPlayer.value = '';
    }
  }

  if (pinchButton) {
    pinchButton.disabled = !canPinch;
    pinchButton.textContent = isGameOver ? 'Game Over' : '代打を送る';
  }

  if (offensePinchMenuButton) {
    offensePinchMenuButton.disabled = !canPinch;
    offensePinchMenuButton.textContent = isGameOver ? 'Game Over' : '代打戦略';
  }

  if (pinchOptionGrid) {
    let pinchHelperMessage = 'カードを選択するとここに反映されます。';
    if (!isActive) {
      pinchHelperMessage = '試合開始後に代打が選択できます。';
    } else if (isGameOver) {
      pinchHelperMessage = '試合終了のため代打は行えません。';
    } else if (!currentBatter) {
      pinchHelperMessage = '現在の打者が確定するまで代打は選択できません。';
    } else if (!offenseBench.length) {
      pinchHelperMessage = '代打に使える選手がいません。';
    }

    updatePinchOptionGrid(pinchOptionGrid, offenseBench, pinchPlayer, pinchSelectHelper, pinchHelperMessage);
  }

  if (openOffenseButton) {
    openOffenseButton.disabled = !isActive || isGameOver;
    openOffenseButton.textContent = isGameOver ? 'Game Over' : '攻撃戦略';
  }

  const defenseTeam = gameState.defense ? teams[gameState.defense] : null;
  const defenseLineup = defenseTeam?.lineup || [];
  const defenseBenchPlayers = defenseTeam?.bench || [];
  const pitcherOptions = defenseTeam?.pitcher_options || [];

  const currentPitcher =
    (defenseTeam?.pitchers || []).find((pitcher) => pitcher && pitcher.is_current) || null;

  updateCurrentPitcherCard(currentPitcherCard, currentPitcher);

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

    updatePitcherOptionGrid(pitcherOptionGrid, pitcherOptions, pitcherSelect, pitcherSelectHelper);

    pitcherButton.textContent = isGameOver ? 'Game Over' : '投手交代';

    if (pitcherMenuButton) {
      pitcherMenuButton.disabled = !canChangePitcher || isGameOver;
      pitcherMenuButton.textContent = isGameOver ? 'Game Over' : '投手交代';
    }
  } else {
    updatePitcherOptionGrid(pitcherOptionGrid, pitcherOptions, null, pitcherSelectHelper);
    if (pitcherMenuButton) {
      pitcherMenuButton.disabled = true;
    }
  }

  if (openStatsButton) {
    const homeStatsAvailable = Boolean(teams.home?.stats);
    const awayStatsAvailable = Boolean(teams.away?.stats);
    openStatsButton.disabled = !(homeStatsAvailable || awayStatsAvailable);
  }
  if (openAbilitiesButton) {
    const homeAbilitiesAvailable = hasAbilityData(teams.home);
    const awayAbilitiesAvailable = hasAbilityData(teams.away);
    openAbilitiesButton.disabled = !(homeAbilitiesAvailable || awayAbilitiesAvailable);
  }
}

export function renderGame(gameState, teams, log) {
  updateAnalyticsPanel(gameState);
  updateDefenseAlignment(gameState, teams);
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

export function updateAbilitiesPanel(state) {
  if (!state) return;
  const teams = state.teams || {};
  const availableTeams = ['away', 'home'].filter((key) => hasAbilityData(teams[key]));
  const fallbackColumnCount = Math.max(ABILITY_BATTING_COLUMNS.length, ABILITY_PITCHING_COLUMNS.length);

  elements.abilitiesTeamButtons.forEach((button) => {
    const teamKey = button.dataset.abilitiesTeam;
    const isAvailable = availableTeams.includes(teamKey);
    button.disabled = !isAvailable;
    if (!isAvailable) {
      button.classList.remove('active');
    }
  });

  if (!availableTeams.length) {
    if (elements.abilitiesTableHead) {
      elements.abilitiesTableHead.innerHTML = '';
    }
    if (elements.abilitiesTableBody) {
      elements.abilitiesTableBody.innerHTML = '';
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = fallbackColumnCount;
      td.textContent = '能力データがありません。';
      td.classList.add('empty');
      tr.appendChild(td);
      elements.abilitiesTableBody.appendChild(tr);
    }
    if (elements.abilitiesTitle) {
      elements.abilitiesTitle.textContent = '能力データがありません';
    }
    elements.abilitiesTypeButtons.forEach((button) => {
      button.disabled = true;
      button.classList.remove('active');
    });
    return;
  }

  if (!availableTeams.includes(stateCache.abilitiesView.team)) {
    stateCache.abilitiesView.team = availableTeams[0];
  }

  let viewType = stateCache.abilitiesView.type;
  if (!['batting', 'pitching'].includes(viewType)) {
    viewType = 'batting';
    stateCache.abilitiesView.type = viewType;
  }

  const teamKey = stateCache.abilitiesView.team;
  const teamData = teams[teamKey] || {};
  const traits = teamData.traits || {};
  const battingRows = Array.isArray(traits.batting) ? traits.batting : [];
  const pitchingRows = Array.isArray(traits.pitching) ? traits.pitching : [];

  if (viewType === 'batting' && !battingRows.length && pitchingRows.length) {
    viewType = 'pitching';
    stateCache.abilitiesView.type = viewType;
  } else if (viewType === 'pitching' && !pitchingRows.length && battingRows.length) {
    viewType = 'batting';
    stateCache.abilitiesView.type = viewType;
  }

  const columns = viewType === 'pitching' ? ABILITY_PITCHING_COLUMNS : ABILITY_BATTING_COLUMNS;
  const rows = viewType === 'pitching' ? pitchingRows : battingRows;

  elements.abilitiesTeamButtons.forEach((button) => {
    const isActive = button.dataset.abilitiesTeam === teamKey;
    button.classList.toggle('active', isActive);
  });

  elements.abilitiesTypeButtons.forEach((button) => {
    const type = button.dataset.abilitiesType;
    const hasData = type === 'pitching' ? pitchingRows.length > 0 : battingRows.length > 0;
    button.disabled = !hasData;
    const isActive = type === viewType && hasData;
    button.classList.toggle('active', isActive);
    if (!hasData) {
      button.classList.remove('active');
    }
  });

  if (elements.abilitiesTableHead) {
    elements.abilitiesTableHead.innerHTML = '';
    columns.forEach((column) => {
      const th = document.createElement('th');
      th.textContent = column.label;
      elements.abilitiesTableHead.appendChild(th);
    });
  }

  if (elements.abilitiesTableBody) {
    elements.abilitiesTableBody.innerHTML = '';
    if (!rows.length) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = columns.length || 1;
      td.textContent = '選択したカテゴリーの能力データがありません。';
      td.classList.add('empty');
      tr.appendChild(td);
      elements.abilitiesTableBody.appendChild(tr);
    } else {
      rows.forEach((row) => {
        const tr = document.createElement('tr');
        columns.forEach((column) => {
          const td = document.createElement('td');
          const value = row[column.key];
          td.textContent = value != null && value !== '' ? value : '-';
          tr.appendChild(td);
        });
        elements.abilitiesTableBody.appendChild(tr);
      });
    }
  }

  if (elements.abilitiesTitle) {
    const teamName = teamData?.name || (teamKey === 'home' ? 'Home' : 'Away');
    const typeLabel = viewType === 'pitching' ? '投手特性' : '打者特性';
    elements.abilitiesTitle.textContent = `${teamName} の${typeLabel}`;
  }
}

export function render(data) {
  stateCache.data = data;
  setStatusMessage(data.notification);
  renderTitle(data.title);
  renderGame(data.game, data.teams, data.log);
  updateStatsPanel(data);
  updateAbilitiesPanel(data);
}
