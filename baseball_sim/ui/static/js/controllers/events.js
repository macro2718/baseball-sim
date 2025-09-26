import { elements } from '../dom.js';
import {
  stateCache,
  setUIView,
  setPinchRunSelectedBase,
  getPinchRunSelectedBase,
  setSimulationResultsView,
  setPlayersTeamView,
  setPlayersTypeView,
} from '../state.js';
import {
  hideDefenseMenu,
  hideOffenseMenu,
  toggleDefenseMenu,
  toggleLogPanel,
  toggleOffenseMenu,
} from '../ui/menus.js';
import { closeModal, openModal, resolveModal } from '../ui/modals.js';
import {
  updateStatsPanel,
  updateAbilitiesPanel,
  render,
  renderTitle,
  updateScreenVisibility,
} from '../ui/renderers.js';
import { showStatus } from '../ui/status.js';
import { handleDefensePlayerClick, updateDefenseSelectionInfo } from '../ui/defensePanel.js';
import {
  ensureTitleLineupPlan,
  getTitleLineupPlan,
  getTitleLineupSelection,
  setTitleLineupSelection,
  clearTitleLineupSelection,
  swapTitleLineupPlayers,
  swapTitleLineupPositions,
  moveBenchPlayerToLineup,
} from '../ui/titleLineup.js';
import { escapeHtml, renderPositionList, renderPositionToken } from '../utils.js';
import { applyAbilityColor, resetAbilityColor, ABILITY_COLOR_PRESETS } from '../ui/renderers.js';

const BATTER_POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF'];
const PITCHER_TYPES = ['SP', 'RP'];

const DEFAULT_PLAYER_TEMPLATES = {
  batter: {
    name: 'New Batter',
    eligible_positions: ['LF', 'CF', 'RF', 'DH'],
    bats: 'R',
    k_pct: 22.8,
    bb_pct: 8.5,
    hard_pct: 38.6,
    gb_pct: 44.6,
    speed: 100,
    fielding_skill: 100,
  },
  pitcher: {
    name: 'New Pitcher',
    pitcher_type: 'SP',
    throws: 'R',
    k_pct: 22.8,
    bb_pct: 8.5,
    hard_pct: 38.6,
    gb_pct: 44.6,
    stamina: 80,
  },
};

const LINEUP_SIZE = 9;
const DEFAULT_LINEUP_POSITIONS = ['CF', 'SS', '2B', '1B', '3B', 'LF', 'RF', 'C', 'DH'];
const POSITION_CHOICES = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH', 'P'];
const TEAM_BUILDER_DEFAULT_BENCH_SLOTS = 3;
const TEAM_BUILDER_DEFAULT_PITCHER_SLOTS = 5;

function clonePlayerTemplate(role = 'batter') {
  const template = DEFAULT_PLAYER_TEMPLATES[role] || DEFAULT_PLAYER_TEMPLATES.batter;
  return JSON.parse(JSON.stringify(template));
}

function setPlayerBuilderFeedback(message, level = 'info') {
  if (!elements.playerBuilderFeedback) return;
  const el = elements.playerBuilderFeedback;
  el.textContent = message || '';
  el.classList.remove('danger', 'success', 'info');
  if (message) {
    el.classList.add(level);
    el.dataset.level = level;
  } else {
    el.removeAttribute('data-level');
  }
}

function formatPercent(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return '-';
  }
  return `${num.toFixed(1)}%`;
}

function formatNumber(value, digits = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return '-';
  }
  if (digits <= 0) {
    return `${Math.round(num)}`;
  }
  return num.toFixed(digits);
}

function createAbilityChip(label, value, metricKey = null) {
  const safeLabel = escapeHtml(label);
  const safeValue = escapeHtml(value);
  const metricAttr = metricKey ? ` data-metric="${escapeHtml(metricKey)}"` : '';
  return `<span class="ability-chip"${metricAttr}><span class="label">${safeLabel}</span><span class="value">${safeValue}</span></span>`;
}

function getBatterAbilityChips(player) {
  if (!player) return [];
  const chips = [];
  if (player.bats) {
    chips.push(createAbilityChip('打席', player.bats));
  }
  const stats = player.stats || {};
  chips.push(createAbilityChip('K%', formatPercent(stats.k_pct), 'k_pct'));
  chips.push(createAbilityChip('BB%', formatPercent(stats.bb_pct), 'bb_pct'));
  chips.push(createAbilityChip('Hard%', formatPercent(stats.hard_pct), 'hard_pct'));
  chips.push(createAbilityChip('GB%', formatPercent(stats.gb_pct), 'gb_pct'));
  chips.push(createAbilityChip('Speed', formatNumber(stats.speed, 1), 'speed'));
  chips.push(createAbilityChip('Field', formatNumber(stats.fielding_skill, 0), 'fielding'));
  return chips;
}

function getPitcherAbilityChips(player) {
  if (!player) return [];
  const chips = [];
  if (player.pitcher_type) {
    chips.push(createAbilityChip('Type', player.pitcher_type));
  }
  if (player.throws) {
    chips.push(createAbilityChip('Throws', player.throws));
  }
  const stats = player.stats || {};
  chips.push(createAbilityChip('K%', formatPercent(stats.k_pct), 'k_pct'));
  chips.push(createAbilityChip('BB%', formatPercent(stats.bb_pct), 'bb_pct'));
  chips.push(createAbilityChip('Hard%', formatPercent(stats.hard_pct), 'hard_pct'));
  chips.push(createAbilityChip('GB%', formatPercent(stats.gb_pct), 'gb_pct'));
  chips.push(createAbilityChip('Sta', formatNumber(stats.stamina, 0), 'stamina'));
  return chips;
}

function colorizeAbilityChips(container, context = 'catalog') {
  if (!container) return;
  const chips = container.querySelectorAll('.ability-chip');
  chips.forEach((chip) => {
    const metric = chip.getAttribute('data-metric');
    const valueEl = chip.querySelector('.value');
    if (!valueEl) return;
    const text = valueEl.textContent ?? '';
    if (metric) {
      let invert = false;
      if (context === 'batter') {
        invert = metric === 'k_pct' || metric === 'gb_pct';
      } else if (context === 'pitcher') {
        invert = metric === 'bb_pct' || metric === 'hard_pct';
      }
      applyAbilityColor(valueEl, metric, text, { ...ABILITY_COLOR_PRESETS.table, invert });
    } else {
      resetAbilityColor(valueEl);
    }
  });
}

function updateChipToggleState(button, selected) {
  if (!button) return;
  if (selected) {
    button.classList.add('active');
    button.setAttribute('aria-pressed', 'true');
  } else {
    button.classList.remove('active');
    button.setAttribute('aria-pressed', 'false');
  }
}

function setSelectedPositions(positions) {
  const selectedSet = new Set((positions || []).map((pos) => String(pos).toUpperCase()));
  elements.playerPositionButtons?.forEach((button) => {
    const key = String(button.dataset.positionOption || '').toUpperCase();
    updateChipToggleState(button, selectedSet.has(key));
  });
}

function getSelectedPositions() {
  return (elements.playerPositionButtons || [])
    .filter((button) => button.classList.contains('active'))
    .map((button) => String(button.dataset.positionOption || '').toUpperCase())
    .filter(Boolean);
}

function setSelectedPitcherType(type) {
  const normalized = typeof type === 'string' ? type.toUpperCase() : '';
  let matched = false;
  elements.playerPitcherTypeButtons?.forEach((button, index) => {
    const key = String(button.dataset.pitcherType || '').toUpperCase();
    const shouldSelect = key === normalized;
    updateChipToggleState(button, shouldSelect);
    if (shouldSelect) {
      matched = true;
    }
    if (!normalized && index === 0 && !matched) {
      updateChipToggleState(button, true);
      matched = true;
    }
  });
  if (!matched && elements.playerPitcherTypeButtons?.length) {
    const [first] = elements.playerPitcherTypeButtons;
    updateChipToggleState(first, true);
    return String(first.dataset.pitcherType || 'SP').toUpperCase();
  }
  const active = elements.playerPitcherTypeButtons?.find((button) => button.classList.contains('active'));
  return active ? String(active.dataset.pitcherType || '').toUpperCase() : null;
}

function getSelectedPitcherType() {
  const active = elements.playerPitcherTypeButtons?.find((button) => button.classList.contains('active'));
  return active ? String(active.dataset.pitcherType || '').toUpperCase() : null;
}

function updatePlayerRoleUI(role = 'batter') {
  const normalized = role === 'pitcher' ? 'pitcher' : 'batter';
  if (elements.playerEditorRole) {
    elements.playerEditorRole.value = normalized;
  }
  elements.playerRoleButtons?.forEach((button) => {
    const value = String(button.dataset.roleChoice || '').toLowerCase();
    const isActive = value === normalized;
    updateChipToggleState(button, isActive);
  });
  elements.playerRoleSections?.forEach((section) => {
    const roles = (section.dataset.roleSection || '')
      .split(',')
      .map((value) => value.trim().toLowerCase())
      .filter(Boolean);
    const shouldShow = !roles.length || roles.includes(normalized);
    section.classList.toggle('hidden', !shouldShow);
    if (shouldShow) {
      section.removeAttribute('aria-hidden');
    } else {
      section.setAttribute('aria-hidden', 'true');
    }
  });
  const enablePositions = normalized === 'batter';
  elements.playerPositionButtons?.forEach((button) => {
    button.disabled = !enablePositions;
    if (!enablePositions) {
      updateChipToggleState(button, false);
    }
  });
  const enablePitcher = normalized === 'pitcher';
  elements.playerPitcherTypeButtons?.forEach((button) => {
    button.disabled = !enablePitcher;
    if (!enablePitcher) {
      updateChipToggleState(button, false);
    }
  });
  return normalized;
}

function clearPlayerForm(role = 'batter') {
  if (elements.playerEditorName) elements.playerEditorName.value = '';
  if (elements.playerEditorKPct) elements.playerEditorKPct.value = '';
  if (elements.playerEditorBBPct) elements.playerEditorBBPct.value = '';
  if (elements.playerEditorHardPct) elements.playerEditorHardPct.value = '';
  if (elements.playerEditorGBPct) elements.playerEditorGBPct.value = '';
  setSelectedPositions([]);
  const normalized = role === 'pitcher' ? 'pitcher' : 'batter';
  if (normalized === 'batter') {
    if (elements.playerEditorBats) elements.playerEditorBats.value = 'R';
    if (elements.playerEditorSpeed) elements.playerEditorSpeed.value = '';
    if (elements.playerEditorFielding) elements.playerEditorFielding.value = '';
  } else {
    if (elements.playerEditorThrows) elements.playerEditorThrows.value = 'R';
    if (elements.playerEditorStamina) elements.playerEditorStamina.value = '';
    setSelectedPitcherType('SP');
  }
}

function setInputValue(input, value) {
  if (!input) return;
  if (value === null || value === undefined || value === '') {
    input.value = '';
    return;
  }
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    input.value = String(numeric);
  } else {
    input.value = String(value);
  }
}

function applyPlayerFormData(player, role = 'batter') {
  const normalized = updatePlayerRoleUI(role);
  if (!player) {
    clearPlayerForm(normalized);
    return;
  }
  if (elements.playerEditorName) elements.playerEditorName.value = player.name || '';
  setInputValue(elements.playerEditorKPct, player.k_pct);
  setInputValue(elements.playerEditorBBPct, player.bb_pct);
  setInputValue(elements.playerEditorHardPct, player.hard_pct);
  setInputValue(elements.playerEditorGBPct, player.gb_pct);
  if (normalized === 'batter') {
    if (elements.playerEditorBats) {
      elements.playerEditorBats.value = player.bats || 'R';
    }
    setInputValue(elements.playerEditorSpeed, player.speed);
    setInputValue(elements.playerEditorFielding, player.fielding_skill);
    const eligible = Array.isArray(player.eligible_positions)
      ? player.eligible_positions
      : Array.isArray(player.eligible)
      ? player.eligible
      : [];
    const filtered = eligible
      .map((pos) => String(pos).toUpperCase())
      .filter((pos) => pos && pos !== 'DH');
    setSelectedPositions(filtered);
  } else {
    if (elements.playerEditorThrows) {
      elements.playerEditorThrows.value = player.throws || 'R';
    }
    setInputValue(elements.playerEditorStamina, player.stamina);
    const type = typeof player.pitcher_type === 'string' ? player.pitcher_type.toUpperCase() : 'SP';
    setSelectedPitcherType(PITCHER_TYPES.includes(type) ? type : 'SP');
  }
}

function readNumberField(input, label, { allowEmpty = false } = {}) {
  if (!input) return { value: null };
  const raw = String(input.value ?? '').trim();
  if (!raw) {
    if (allowEmpty) {
      return { value: null };
    }
    return { error: `${label}を入力してください。` };
  }
  const value = Number.parseFloat(raw);
  if (Number.isNaN(value)) {
    return { error: `${label}には数値を入力してください。` };
  }
  return { value };
}

function getPlayerFormData(role = 'batter') {
  const normalized = role === 'pitcher' ? 'pitcher' : 'batter';
  const name = elements.playerEditorName?.value?.trim() || '';
  if (!name) {
    setPlayerBuilderFeedback('名前を入力してください。', 'danger');
    return null;
  }

  const kPct = readNumberField(elements.playerEditorKPct, 'K%');
  if (kPct.error) {
    setPlayerBuilderFeedback(kPct.error, 'danger');
    return null;
  }
  const bbPct = readNumberField(elements.playerEditorBBPct, 'BB%');
  if (bbPct.error) {
    setPlayerBuilderFeedback(bbPct.error, 'danger');
    return null;
  }
  const hardPct = readNumberField(elements.playerEditorHardPct, 'Hard%');
  if (hardPct.error) {
    setPlayerBuilderFeedback(hardPct.error, 'danger');
    return null;
  }
  const gbPct = readNumberField(elements.playerEditorGBPct, 'GB%');
  if (gbPct.error) {
    setPlayerBuilderFeedback(gbPct.error, 'danger');
    return null;
  }

  const baseData = {
    name,
    k_pct: kPct.value,
    bb_pct: bbPct.value,
    hard_pct: hardPct.value,
    gb_pct: gbPct.value,
  };

  if (normalized === 'batter') {
    const speed = readNumberField(elements.playerEditorSpeed, 'Speed');
    if (speed.error) {
      setPlayerBuilderFeedback(speed.error, 'danger');
      return null;
    }
    const fielding = readNumberField(elements.playerEditorFielding, 'Fielding');
    if (fielding.error) {
      setPlayerBuilderFeedback(fielding.error, 'danger');
      return null;
    }
    const selectedPositions = getSelectedPositions();
    const uniquePositions = Array.from(
      new Set(
        selectedPositions.filter((pos) => BATTER_POSITIONS.includes(pos)).map((pos) => pos.toUpperCase()),
      ),
    );
    const positionsWithDh = Array.from(new Set([...uniquePositions, 'DH']));
    return {
      ...baseData,
      eligible_positions: positionsWithDh,
      bats: elements.playerEditorBats?.value || 'R',
      speed: speed.value,
      fielding_skill: fielding.value,
    };
  }

  const stamina = readNumberField(elements.playerEditorStamina, 'Stamina');
  if (stamina.error) {
    setPlayerBuilderFeedback(stamina.error, 'danger');
    return null;
  }
  const type = getSelectedPitcherType();
  if (!type) {
    setPlayerBuilderFeedback('投手タイプを選択してください。', 'danger');
    return null;
  }
  return {
    ...baseData,
    pitcher_type: type,
    throws: elements.playerEditorThrows?.value || 'R',
    stamina: stamina.value,
  };
}

function loadPlayerTemplate(role = 'batter') {
  const normalized = updatePlayerRoleUI(role);
  const template = clonePlayerTemplate(normalized);
  applyPlayerFormData(template, normalized);
  if (elements.playerEditorSelect) {
    elements.playerEditorSelect.value = '__new__';
  }
  setPlayerBuilderFeedback('テンプレートを読み込みました。', 'info');
}

function setTeamBuilderFeedback(message, level = 'info') {
  if (!elements.teamBuilderFeedback) return;
  const feedback = elements.teamBuilderFeedback;
  feedback.textContent = message || '';
  feedback.classList.remove('danger', 'success', 'info');
  if (message) {
    feedback.classList.add(level);
    feedback.dataset.level = level;
  } else {
    feedback.removeAttribute('data-level');
  }
}

function ensureTeamBuilderState() {
  if (!stateCache.teamBuilder.form) {
    stateCache.teamBuilder.form = createDefaultTeamForm();
    resetLineupPositionSelection();
  }
  if (!stateCache.teamBuilder.players) {
    stateCache.teamBuilder.players = {
      batters: [],
      pitchers: [],
      byId: {},
      byName: {},
      loaded: false,
      loading: false,
    };
  } else {
    const players = stateCache.teamBuilder.players;
    if (!Array.isArray(players.batters)) players.batters = [];
    if (!Array.isArray(players.pitchers)) players.pitchers = [];
    if (!players.byId) players.byId = {};
    if (!players.byName) players.byName = {};
    if (typeof players.loaded !== 'boolean') players.loaded = false;
    if (typeof players.loading !== 'boolean') players.loading = false;
  }
  if (!('playersLoadingPromise' in stateCache.teamBuilder)) {
    stateCache.teamBuilder.playersLoadingPromise = null;
  }
  if (!stateCache.teamBuilder.selection) {
    stateCache.teamBuilder.selection = { group: 'lineup', index: 0 };
  }
  if (!stateCache.teamBuilder.positionSwap) {
    stateCache.teamBuilder.positionSwap = { first: null };
  } else if (!Number.isInteger(stateCache.teamBuilder.positionSwap.first)) {
    stateCache.teamBuilder.positionSwap.first = null;
  }
  if (!stateCache.teamBuilder.playerSwap) {
    stateCache.teamBuilder.playerSwap = { source: null };
  } else if (
    stateCache.teamBuilder.playerSwap.source &&
    (!('group' in stateCache.teamBuilder.playerSwap.source) ||
      !Number.isInteger(stateCache.teamBuilder.playerSwap.source.index))
  ) {
    stateCache.teamBuilder.playerSwap.source = null;
  }
  if (!stateCache.teamBuilder.catalog) {
    stateCache.teamBuilder.catalog = 'batters';
  }
  if (stateCache.teamBuilder.searchTerm == null) {
    stateCache.teamBuilder.searchTerm = '';
  }
  if (!stateCache.teamBuilder.initialForm) {
    stateCache.teamBuilder.initialForm = cloneTeamForm(stateCache.teamBuilder.form);
  }
}

function createEmptyLineupSlot(order) {
  const defaultPosition = DEFAULT_LINEUP_POSITIONS[order] || 'DH';
  return {
    order,
    position: defaultPosition,
    playerId: null,
    playerName: '',
    playerRole: null,
    player: null,
    eligible: [defaultPosition],
  };
}

function createEmptyBenchEntry() {
  return { playerId: null, playerName: '', playerRole: null, player: null, eligible: [] };
}

function createEmptyPitcherEntry() {
  return { playerId: null, playerName: '', playerRole: 'pitcher', player: null, eligible: ['P'] };
}

function createDefaultTeamForm() {
  const lineup = Array.from({ length: LINEUP_SIZE }, (_, index) => createEmptyLineupSlot(index));
  const bench = Array.from({ length: TEAM_BUILDER_DEFAULT_BENCH_SLOTS }, () => createEmptyBenchEntry());
  const pitchers = Array.from({ length: TEAM_BUILDER_DEFAULT_PITCHER_SLOTS }, () => createEmptyPitcherEntry());
  return { name: '', lineup, bench, pitchers };
}

function cloneLineupSlotData(slot, index) {
  const defaultPosition = DEFAULT_LINEUP_POSITIONS[index] || 'DH';
  const position = slot?.position ? String(slot.position).toUpperCase() : defaultPosition;
  const eligibleSource = Array.isArray(slot?.eligible) && slot.eligible.length ? slot.eligible : [position];
  return {
    order: Number.isInteger(slot?.order) ? slot.order : index,
    position,
    playerId: slot?.playerId || null,
    playerName: slot?.playerName || '',
    playerRole: slot?.playerRole || null,
    player: slot?.player || null,
    eligible: eligibleSource.map((pos) => String(pos).toUpperCase()),
  };
}

function cloneBenchEntryData(entry) {
  return {
    playerId: entry?.playerId || null,
    playerName: entry?.playerName || '',
    playerRole: entry?.playerRole || null,
    player: entry?.player || null,
    eligible: Array.isArray(entry?.eligible) ? [...entry.eligible] : [],
  };
}

function clonePitcherEntryData(entry) {
  return {
    playerId: entry?.playerId || null,
    playerName: entry?.playerName || '',
    playerRole: entry?.playerRole || 'pitcher',
    player: entry?.player || null,
    eligible: Array.isArray(entry?.eligible) && entry.eligible.length ? [...entry.eligible] : ['P'],
  };
}

function cloneTeamForm(form) {
  if (!form) {
    return createDefaultTeamForm();
  }
  const lineup = Array.from({ length: LINEUP_SIZE }, (_, index) =>
    cloneLineupSlotData(form.lineup?.[index], index),
  );
  const bench = Array.isArray(form.bench)
    ? form.bench.map((entry) => cloneBenchEntryData(entry))
    : [];
  const pitchers = Array.isArray(form.pitchers)
    ? form.pitchers.map((entry) => clonePitcherEntryData(entry))
    : [];
  return { name: form.name || '', lineup, bench, pitchers };
}

function captureTeamBuilderSnapshot(form = null) {
  ensureTeamBuilderState();
  const source = form || stateCache.teamBuilder.form;
  if (!source) return;
  stateCache.teamBuilder.initialForm = cloneTeamForm(source);
}

function normalizePositionsList(list) {
  const out = [];
  if (Array.isArray(list)) {
    list.forEach((pos) => {
      if (typeof pos !== 'string') return;
      const token = pos.trim().toUpperCase();
      if (token && !out.includes(token)) {
        out.push(token);
      }
    });
  }
  return out;
}

function clearLineupPlayerOnly(slot) {
  if (!slot) return;
  const basePosition = slot.position || 'DH';
  slot.playerId = null;
  slot.playerName = '';
  slot.playerRole = null;
  slot.player = null;
  slot.eligible = [String(basePosition).toUpperCase()];
}

function setLineupSlotFromPlayerData(slot, data) {
  if (!slot) return;
  if (data && (data.player || data.playerName)) {
    if (data.player) {
      applyRecordToLineupSlot(slot, data.player, slot.position);
    } else {
      slot.playerId = data.playerId || null;
      slot.playerName = data.playerName || '';
      slot.playerRole = data.playerRole || null;
      slot.player = null;
      const fallback = slot.position || DEFAULT_LINEUP_POSITIONS[slot.order] || 'DH';
      if (Array.isArray(data.eligible) && data.eligible.length) {
        slot.eligible = [...data.eligible];
      } else if (slot.playerRole === 'pitcher') {
        slot.eligible = ['P'];
      } else {
        slot.eligible = [String(fallback).toUpperCase()];
      }
    }
  } else {
    clearLineupPlayerOnly(slot);
  }
}

function setBenchEntryFromData(entry, data) {
  if (!entry) return;
  if (data && (data.player || data.playerName)) {
    const record = data.player || null;
    entry.player = record;
    entry.playerId = record?.id || data.playerId || null;
    entry.playerName = record?.name || data.playerName || '';
    entry.playerRole = record?.role || data.playerRole || null;
    if (entry.playerRole === 'pitcher') {
      entry.eligible = ['P'];
    } else if (record && Array.isArray(record.eligible)) {
      entry.eligible = [...record.eligible];
    } else if (Array.isArray(data.eligible)) {
      entry.eligible = [...data.eligible];
    } else {
      entry.eligible = [];
    }
  } else {
    const empty = createEmptyBenchEntry();
    entry.player = empty.player;
    entry.playerId = empty.playerId;
    entry.playerName = empty.playerName;
    entry.playerRole = empty.playerRole;
    entry.eligible = empty.eligible;
  }
}

function clearPlayerSwapSelection() {
  ensureTeamBuilderState();
  if (!stateCache.teamBuilder.playerSwap) {
    stateCache.teamBuilder.playerSwap = { source: null };
    return;
  }
  stateCache.teamBuilder.playerSwap.source = null;
}

function clearPlayerSwapIfMatches(group, index) {
  const source = stateCache.teamBuilder.playerSwap?.source;
  if (source && source.group === group && source.index === index) {
    clearPlayerSwapSelection();
  }
}

function applyRecordToLineupSlot(slot, record, positionOverride = null) {
  if (!slot || !record) return;
  slot.playerId = record.id;
  slot.playerName = record.name;
  slot.player = record;
  slot.playerRole = record.role;
  if (record.role === 'batter') {
    const eligible = Array.isArray(record.eligible) ? [...record.eligible] : [];
    if (!eligible.includes('DH')) {
      eligible.push('DH');
    }
    // Keep eligibility as a hint only; preserve user's desired position even if ineligible
    slot.eligible = eligible;
    const desired = positionOverride
      ? String(positionOverride).toUpperCase()
      : String(slot.position || '').toUpperCase();
    slot.position = desired || slot.position || 'DH';
  } else {
    // For pitchers, keep eligibility hint but preserve desired position
    slot.eligible = ['P'];
    const desired = positionOverride
      ? String(positionOverride).toUpperCase()
      : String(slot.position || '').toUpperCase();
    slot.position = desired || slot.position || 'P';
  }
}

function applyPlayersCatalogData(catalog) {
  ensureTeamBuilderState();
  const battersRaw = Array.isArray(catalog?.batters) ? catalog.batters : [];
  const pitchersRaw = Array.isArray(catalog?.pitchers) ? catalog.pitchers : [];
  const players = {
    batters: [],
    pitchers: [],
    byId: {},
    byName: {},
    loaded: true,
    loading: false,
  };

  const addToNameIndex = (record) => {
    const key = record.name ? record.name.trim().toLowerCase() : '';
    if (!key) return;
    if (!players.byName[key]) {
      players.byName[key] = [];
    }
    players.byName[key].push(record);
  };

  battersRaw.forEach((raw) => {
    if (!raw || typeof raw !== 'object') return;
    const id = raw.id ? String(raw.id) : null;
    const name = raw.name ? String(raw.name) : null;
    if (!id || !name) return;
    const eligible = normalizePositionsList(raw.eligible_positions);
    if (!eligible.includes('DH')) {
      eligible.push('DH');
    }
    const record = {
      id,
      name,
      role: 'batter',
      bats: raw.bats ? String(raw.bats).toUpperCase() : null,
      eligible,
      stats: {
        k_pct: Number(raw.k_pct),
        bb_pct: Number(raw.bb_pct),
        hard_pct: Number(raw.hard_pct),
        gb_pct: Number(raw.gb_pct),
        speed: Number(raw.speed),
        fielding_skill: Number(raw.fielding_skill),
      },
    };
    players.batters.push(record);
    players.byId[id] = record;
    addToNameIndex(record);
  });

  pitchersRaw.forEach((raw) => {
    if (!raw || typeof raw !== 'object') return;
    const id = raw.id ? String(raw.id) : null;
    const name = raw.name ? String(raw.name) : null;
    if (!id || !name) return;
    const record = {
      id,
      name,
      role: 'pitcher',
      pitcher_type: raw.pitcher_type ? String(raw.pitcher_type).toUpperCase() : null,
      throws: raw.throws ? String(raw.throws).toUpperCase() : null,
      eligible: ['P'],
      stats: {
        k_pct: Number(raw.k_pct),
        bb_pct: Number(raw.bb_pct),
        hard_pct: Number(raw.hard_pct),
        gb_pct: Number(raw.gb_pct),
        stamina: Number(raw.stamina),
      },
    };
    players.pitchers.push(record);
    players.byId[id] = record;
    addToNameIndex(record);
  });

  stateCache.teamBuilder.players = players;
  relinkFormPlayers();
  if (stateCache.teamBuilder.initialForm) {
    relinkFormPlayers(stateCache.teamBuilder.initialForm);
  }
}

function getPlayerRecordById(playerId) {
  if (!playerId) return null;
  ensureTeamBuilderState();
  return stateCache.teamBuilder.players?.byId?.[playerId] || null;
}

function resolvePlayerByName(name, positionHint = null) {
  if (!name) return null;
  ensureTeamBuilderState();
  const byName = stateCache.teamBuilder.players?.byName || {};
  const key = String(name).trim().toLowerCase();
  const matches = byName[key];
  if (!matches || !matches.length) {
    return null;
  }
  const upperHint = positionHint ? String(positionHint).toUpperCase() : null;
  if (upperHint === 'P') {
    const pitcher = matches.find((record) => record.role === 'pitcher');
    if (pitcher) {
      return pitcher;
    }
  }
  const batter = matches.find((record) => record.role === 'batter');
  if (upperHint === 'P' && !batter) {
    return matches[0];
  }
  return batter || matches[0];
}

function relinkFormPlayers(targetForm = null) {
  ensureTeamBuilderState();
  const form = targetForm || stateCache.teamBuilder.form;
  if (!form) return;

  form.lineup = Array.isArray(form.lineup) ? form.lineup : [];
  form.lineup.forEach((slot, index) => {
    if (!slot) {
      form.lineup[index] = createEmptyLineupSlot(index);
      return;
    }
    const position = slot.position || DEFAULT_LINEUP_POSITIONS[index] || 'DH';
    if (slot.playerId) {
      const record = getPlayerRecordById(slot.playerId);
      if (record) {
        applyRecordToLineupSlot(slot, record, position);
        return;
      }
    }
    if (slot.playerName) {
      const record = resolvePlayerByName(slot.playerName, position);
      if (record) {
        applyRecordToLineupSlot(slot, record, position);
        return;
      }
    }
    slot.player = null;
    slot.playerRole = null;
    slot.playerId = null;
    slot.eligible = [position];
    slot.position = position;
  });

  form.bench = Array.isArray(form.bench) ? form.bench : [];
  form.bench.forEach((entry, index) => {
    if (!entry) {
      form.bench[index] = createEmptyBenchEntry();
      return;
    }
    if (entry.playerId) {
      const record = getPlayerRecordById(entry.playerId);
      if (record) {
        entry.player = record;
        entry.playerRole = record.role;
        entry.playerName = record.name;
        entry.eligible = record.role === 'pitcher' ? ['P'] : [...(record.eligible || [])];
        return;
      }
    }
    if (entry.playerName) {
      const record = resolvePlayerByName(entry.playerName);
      if (record) {
        entry.playerId = record.id;
        entry.player = record;
        entry.playerRole = record.role;
        entry.playerName = record.name;
        entry.eligible = record.role === 'pitcher' ? ['P'] : [...(record.eligible || [])];
        return;
      }
    }
    entry.player = null;
    entry.playerRole = null;
    entry.playerId = null;
    entry.eligible = [];
  });

  form.pitchers = Array.isArray(form.pitchers) ? form.pitchers : [];
  form.pitchers.forEach((entry, index) => {
    if (!entry) {
      form.pitchers[index] = createEmptyPitcherEntry();
      return;
    }
    if (entry.playerId) {
      const record = getPlayerRecordById(entry.playerId);
      if (record) {
        entry.player = record;
        entry.playerRole = record.role;
        entry.playerName = record.name;
        return;
      }
    }
    if (entry.playerName) {
      const record = resolvePlayerByName(entry.playerName, 'P');
      if (record) {
        entry.playerId = record.id;
        entry.player = record;
        entry.playerRole = record.role;
        entry.playerName = record.name;
        return;
      }
    }
    entry.player = null;
    entry.playerRole = null;
    entry.playerId = null;
  });
}

function getTeamBuilderForm() {
  ensureTeamBuilderState();
  return stateCache.teamBuilder.form;
}

function refreshView() {
  if (stateCache.data) {
    render(stateCache.data);
  } else {
    updateScreenVisibility();
  }
  if (stateCache.uiView === 'team-builder') {
    renderTeamBuilderView();
  }
}

function setSelectionAndCatalog(group, index) {
  ensureTeamBuilderState();
  const normalizedGroup = group === 'pitchers' ? 'pitchers' : group === 'bench' ? 'bench' : 'lineup';
  const normalizedIndex = Number.isInteger(index) ? index : 0;
  stateCache.teamBuilder.selection = { group: normalizedGroup, index: normalizedIndex };
  const desiredCatalog = normalizedGroup === 'pitchers' ? 'pitchers' : 'batters';
  if (stateCache.teamBuilder.catalog !== desiredCatalog) {
    stateCache.teamBuilder.catalog = desiredCatalog;
  }
}

function selectTeamBuilderSlot(group, index) {
  ensureTeamBuilderState();
  const form = stateCache.teamBuilder.form;
  let normalizedGroup = group === 'bench' || group === 'pitchers' ? group : 'lineup';
  let normalizedIndex = Number.isInteger(index) ? index : 0;
  if (normalizedGroup === 'bench') {
    const benchLength = Array.isArray(form.bench) ? form.bench.length : 0;
    if (!benchLength) {
      normalizedIndex = 0;
    } else if (normalizedIndex < 0) {
      normalizedIndex = 0;
    } else if (normalizedIndex >= benchLength) {
      normalizedIndex = benchLength - 1;
    }
  } else if (normalizedGroup === 'pitchers') {
    const pitcherLength = Array.isArray(form.pitchers) ? form.pitchers.length : 0;
    if (!pitcherLength) {
      normalizedIndex = 0;
    } else if (normalizedIndex < 0) {
      normalizedIndex = 0;
    } else if (normalizedIndex >= pitcherLength) {
      normalizedIndex = pitcherLength - 1;
    }
  } else {
    normalizedGroup = 'lineup';
    if (!Number.isInteger(normalizedIndex) || normalizedIndex < 0 || normalizedIndex >= form.lineup.length) {
      normalizedIndex = 0;
    }
  }
  setSelectionAndCatalog(normalizedGroup, normalizedIndex);
  renderTeamBuilderView();
}

function focusRosterForCatalog(catalog) {
  ensureTeamBuilderState();
  const target = catalog === 'pitchers' ? 'pitchers' : 'batters';
  const selection = stateCache.teamBuilder.selection || { group: 'lineup', index: 0 };
  if (target === 'pitchers') {
    const index =
      selection.group === 'pitchers' && Number.isInteger(selection.index) ? selection.index : 0;
    selectTeamBuilderSlot('pitchers', index);
    return;
  }
  if (selection.group === 'bench') {
    selectTeamBuilderSlot('bench', selection.index);
    return;
  }
  const index =
    selection.group === 'lineup' && Number.isInteger(selection.index) ? selection.index : 0;
  selectTeamBuilderSlot('lineup', index);
}

function getAvailablePositionsForSlot(slot) {
  if (!slot) return [...POSITION_CHOICES];
  if (slot.playerRole === 'pitcher') {
    return ['P'];
  }
  if (Array.isArray(slot.eligible) && slot.eligible.length) {
    const unique = [];
    slot.eligible.forEach((pos) => {
      const token = String(pos || '').toUpperCase();
      if (token && !unique.includes(token)) {
        unique.push(token);
      }
    });
    return unique.length ? unique : [...POSITION_CHOICES];
  }
  return [...POSITION_CHOICES];
}

function resetLineupPositionSelection() {
  if (!stateCache.teamBuilder.positionSwap) {
    stateCache.teamBuilder.positionSwap = { first: null };
    return;
  }
  stateCache.teamBuilder.positionSwap.first = null;
}

function swapLineupPositions(firstIndex, secondIndex) {
  const form = getTeamBuilderForm();
  const lineup = form?.lineup;
  if (!Array.isArray(lineup) || lineup.length < 2) {
    resetLineupPositionSelection();
    renderTeamBuilderView();
    return;
  }
  if (
    !Number.isInteger(firstIndex) ||
    !Number.isInteger(secondIndex) ||
    firstIndex === secondIndex
  ) {
    resetLineupPositionSelection();
    renderTeamBuilderView();
    return;
  }
  if (
    firstIndex < 0 ||
    secondIndex < 0 ||
    firstIndex >= lineup.length ||
    secondIndex >= lineup.length
  ) {
    resetLineupPositionSelection();
    renderTeamBuilderView();
    return;
  }

  const firstSlot = lineup[firstIndex];
  const secondSlot = lineup[secondIndex];
  if (!firstSlot || !secondSlot) {
    resetLineupPositionSelection();
    renderTeamBuilderView();
    return;
  }

  const firstPosition = (
    firstSlot.position ||
    DEFAULT_LINEUP_POSITIONS[firstIndex] ||
    'DH'
  ).toUpperCase();
  const secondPosition = (
    secondSlot.position ||
    DEFAULT_LINEUP_POSITIONS[secondIndex] ||
    'DH'
  ).toUpperCase();

  // Always allow swapping positions regardless of eligibility
  const firstHasPlayer = Boolean(firstSlot.player);
  const secondHasPlayer = Boolean(secondSlot.player);

  firstSlot.position = secondPosition;
  secondSlot.position = firstPosition;

  if (!firstHasPlayer) {
    firstSlot.eligible = [secondPosition];
  }
  if (!secondHasPlayer) {
    secondSlot.eligible = [firstPosition];
  }

  stateCache.teamBuilder.editorDirty = true;
  resetLineupPositionSelection();
  renderTeamBuilderView();
}

function toggleLineupPositionSelection(index) {
  ensureTeamBuilderState();
  const form = getTeamBuilderForm();
  const lineup = form?.lineup;
  if (!Array.isArray(lineup) || !lineup.length) {
    resetLineupPositionSelection();
    renderTeamBuilderView();
    return;
  }
  const normalizedIndex = Number.isInteger(index) ? index : 0;
  if (normalizedIndex < 0 || normalizedIndex >= lineup.length) {
    resetLineupPositionSelection();
    renderTeamBuilderView();
    return;
  }

  const currentSelection = stateCache.teamBuilder.positionSwap?.first;
  if (
    !Number.isInteger(currentSelection) ||
    currentSelection < 0 ||
    currentSelection >= lineup.length
  ) {
    stateCache.teamBuilder.positionSwap.first = normalizedIndex;
    renderTeamBuilderView();
    return;
  }

  if (currentSelection === normalizedIndex) {
    resetLineupPositionSelection();
    renderTeamBuilderView();
    return;
  }

  swapLineupPositions(currentSelection, normalizedIndex);
}

function clearLineupSlot(index) {
  const form = getTeamBuilderForm();
  const slot = form?.lineup?.[index];
  if (!slot) return;
  const defaultPosition = DEFAULT_LINEUP_POSITIONS[index] || 'DH';
  slot.playerId = null;
  slot.playerName = '';
  slot.playerRole = null;
  slot.player = null;
  slot.eligible = [defaultPosition];
  slot.position = defaultPosition;
  clearPlayerSwapIfMatches('lineup', index);
  stateCache.teamBuilder.editorDirty = true;
  renderTeamBuilderView();
}

function clearBenchEntry(index) {
  const form = getTeamBuilderForm();
  if (!form || !form.bench || index < 0 || index >= form.bench.length) return;
  form.bench[index] = createEmptyBenchEntry();
  clearPlayerSwapIfMatches('bench', index);
}

function clearPitcherEntry(index) {
  const form = getTeamBuilderForm();
  if (!form || !form.pitchers || index < 0 || index >= form.pitchers.length) return;
  form.pitchers[index] = createEmptyPitcherEntry();
}

function removeBenchSlot(index) {
  const form = getTeamBuilderForm();
  if (!form || !Array.isArray(form.bench) || index < 0 || index >= form.bench.length) return;
  clearPlayerSwapSelection();
  form.bench.splice(index, 1);
  stateCache.teamBuilder.editorDirty = true;
  if (stateCache.teamBuilder.selection?.group === 'bench') {
    if (!form.bench.length) {
      setSelectionAndCatalog('lineup', 0);
      renderTeamBuilderView();
      return;
    }
    const currentIndex = Number.isInteger(stateCache.teamBuilder.selection.index)
      ? stateCache.teamBuilder.selection.index
      : 0;
    const nextIndex = Math.min(Math.max(currentIndex, 0), form.bench.length - 1);
    selectTeamBuilderSlot('bench', nextIndex);
    return;
  }
  renderTeamBuilderView();
}

function removePitcherSlot(index) {
  const form = getTeamBuilderForm();
  if (!form || !Array.isArray(form.pitchers) || index < 0 || index >= form.pitchers.length) return;
  form.pitchers.splice(index, 1);
  stateCache.teamBuilder.editorDirty = true;
  if (stateCache.teamBuilder.selection?.group === 'pitchers') {
    const currentIndex = Number.isInteger(stateCache.teamBuilder.selection.index)
      ? stateCache.teamBuilder.selection.index
      : 0;
    const nextIndex = form.pitchers.length
      ? Math.min(Math.max(currentIndex, 0), form.pitchers.length - 1)
      : 0;
    selectTeamBuilderSlot('pitchers', nextIndex);
    return;
  }
  renderTeamBuilderView();
}

function addBenchSlot() {
  const form = getTeamBuilderForm();
  if (!form) return;
  if (!Array.isArray(form.bench)) {
    form.bench = [];
  }
  const index = form.bench.length;
  form.bench.push(createEmptyBenchEntry());
  stateCache.teamBuilder.editorDirty = true;
  selectTeamBuilderSlot('bench', index);
}

function addPitcherSlot() {
  const form = getTeamBuilderForm();
  if (!form) return;
  if (!Array.isArray(form.pitchers)) {
    form.pitchers = [];
  }
  const index = form.pitchers.length;
  form.pitchers.push(createEmptyPitcherEntry());
  stateCache.teamBuilder.editorDirty = true;
  selectTeamBuilderSlot('pitchers', index);
}

function findPlayerAssignment(playerId) {
  if (!playerId) return null;
  const form = getTeamBuilderForm();
  const lineupIndex = form.lineup.findIndex((slot) => slot?.playerId === playerId);
  if (lineupIndex >= 0) {
    return { group: 'lineup', index: lineupIndex };
  }
  const benchIndex = form.bench.findIndex((entry) => entry?.playerId === playerId);
  if (benchIndex >= 0) {
    return { group: 'bench', index: benchIndex };
  }
  const pitcherIndex = form.pitchers.findIndex((entry) => entry?.playerId === playerId);
  if (pitcherIndex >= 0) {
    return { group: 'pitchers', index: pitcherIndex };
  }
  return null;
}

function getLineupPlayerSnapshot(index) {
  const form = getTeamBuilderForm();
  return cloneLineupSlotData(form?.lineup?.[index], index);
}

function getBenchPlayerSnapshot(index) {
  const form = getTeamBuilderForm();
  return cloneBenchEntryData(form?.bench?.[index]);
}

function swapLineupPlayers(indexA, indexB) {
  const form = getTeamBuilderForm();
  if (!form || !Array.isArray(form.lineup)) return false;
  if (!Number.isInteger(indexA) || !Number.isInteger(indexB) || indexA === indexB) {
    return false;
  }
  if (indexA < 0 || indexB < 0 || indexA >= form.lineup.length || indexB >= form.lineup.length) {
    return false;
  }
  const lineup = form.lineup;
  const slotA = lineup[indexA];
  const slotB = lineup[indexB];
  const nameA = slotA?.playerName || '空き枠';
  const nameB = slotB?.playerName || '空き枠';
  lineup[indexA] = slotB;
  lineup[indexB] = slotA;
  if (lineup[indexA]) {
    lineup[indexA].order = indexA;
  }
  if (lineup[indexB]) {
    lineup[indexB].order = indexB;
  }
  stateCache.teamBuilder.editorDirty = true;
  setSelectionAndCatalog('lineup', indexB);
  clearPlayerSwapSelection();
  setTeamBuilderFeedback(`${nameA} と ${nameB} を入れ替えました。`, 'success');
  renderTeamBuilderView();
  return true;
}

function swapBenchPlayers(indexA, indexB) {
  const form = getTeamBuilderForm();
  if (!form || !Array.isArray(form.bench)) return false;
  if (!Number.isInteger(indexA) || !Number.isInteger(indexB) || indexA === indexB) {
    return false;
  }
  if (indexA < 0 || indexB < 0 || indexA >= form.bench.length || indexB >= form.bench.length) {
    return false;
  }
  const bench = form.bench;
  const entryA = bench[indexA];
  const entryB = bench[indexB];
  const nameA = entryA?.playerName || '空き枠';
  const nameB = entryB?.playerName || '空き枠';
  bench[indexA] = entryB;
  bench[indexB] = entryA;
  stateCache.teamBuilder.editorDirty = true;
  setSelectionAndCatalog('bench', indexB);
  clearPlayerSwapSelection();
  setTeamBuilderFeedback(`${nameA} と ${nameB} を入れ替えました。`, 'success');
  renderTeamBuilderView();
  return true;
}

function swapLineupAndBench(lineupIndex, benchIndex) {
  const form = getTeamBuilderForm();
  if (!form || !Array.isArray(form.lineup) || !Array.isArray(form.bench)) {
    return false;
  }
  if (
    !Number.isInteger(lineupIndex) ||
    !Number.isInteger(benchIndex) ||
    lineupIndex < 0 ||
    benchIndex < 0 ||
    lineupIndex >= form.lineup.length ||
    benchIndex >= form.bench.length
  ) {
    return false;
  }

  const lineupSlot = form.lineup[lineupIndex];
  if (!form.bench[benchIndex]) {
    form.bench[benchIndex] = createEmptyBenchEntry();
  }
  const benchEntry = form.bench[benchIndex];

  const lineupData = cloneLineupSlotData(lineupSlot, lineupIndex);
  const benchData = cloneBenchEntryData(benchEntry);

  if (benchData.player || benchData.playerName) {
    setLineupSlotFromPlayerData(lineupSlot, benchData);
  } else {
    clearLineupPlayerOnly(lineupSlot);
  }

  setBenchEntryFromData(benchEntry, lineupData);

  stateCache.teamBuilder.editorDirty = true;
  setSelectionAndCatalog('lineup', lineupIndex);
  clearPlayerSwapSelection();
  const lineupName = benchData.playerName || benchData.player?.name || '空き枠';
  const benchName = lineupData.playerName || lineupData.player?.name || '空き枠';
  setTeamBuilderFeedback(`${lineupName} と ${benchName} を入れ替えました。`, 'success');
  renderTeamBuilderView();
  return true;
}

function togglePlayerSwap(group, index) {
  ensureTeamBuilderState();
  const normalizedGroup = group === 'bench' ? 'bench' : 'lineup';
  const form = getTeamBuilderForm();
  if (!form) return;
  const source = stateCache.teamBuilder.playerSwap?.source;
  if (source && source.group === normalizedGroup && source.index === index) {
    clearPlayerSwapSelection();
    setTeamBuilderFeedback('入れ替えモードを解除しました。', 'info');
    renderTeamBuilderView();
    return;
  }

  if (normalizedGroup === 'lineup') {
    const slot = form.lineup?.[index];
    if (!slot || !slot.playerName) {
      setTeamBuilderFeedback('入れ替える選手が設定されていません。', 'warning');
      return;
    }
  } else if (normalizedGroup === 'bench') {
    const entry = form.bench?.[index];
    if (!entry || !entry.playerName) {
      setTeamBuilderFeedback('入れ替える選手が設定されていません。', 'warning');
      return;
    }
  }

  stateCache.teamBuilder.playerSwap = { source: { group: normalizedGroup, index } };
  setTeamBuilderFeedback('入れ替え先を選択してください。', 'info');
  renderTeamBuilderView();
}

function attemptPlayerSwapWithSlot(group, index) {
  const source = stateCache.teamBuilder.playerSwap?.source;
  if (!source) {
    return false;
  }
  const normalizedGroup = group === 'bench' ? 'bench' : group === 'lineup' ? 'lineup' : group;
  if (normalizedGroup === 'pitchers') {
    setTeamBuilderFeedback('投手枠とは入れ替えできません。', 'danger');
    clearPlayerSwapSelection();
    renderTeamBuilderView();
    return true;
  }
  if (source.group === normalizedGroup && source.index === index) {
    clearPlayerSwapSelection();
    renderTeamBuilderView();
    return true;
  }
  if (source.group === 'lineup' && normalizedGroup === 'lineup') {
    return swapLineupPlayers(source.index, index);
  }
  if (source.group === 'lineup' && normalizedGroup === 'bench') {
    return swapLineupAndBench(source.index, index);
  }
  if (source.group === 'bench' && normalizedGroup === 'lineup') {
    return swapLineupAndBench(index, source.index);
  }
  if (source.group === 'bench' && normalizedGroup === 'bench') {
    return swapBenchPlayers(source.index, index);
  }
  return false;
}

function attemptPlayerSwapWithRecord(record) {
  const source = stateCache.teamBuilder.playerSwap?.source;
  if (!source || !record) {
    return false;
  }
  const assignment = findPlayerAssignment(record.id);
  if (source.group === 'lineup') {
    if (assignment) {
      if (assignment.group === 'lineup') {
        return swapLineupPlayers(source.index, assignment.index);
      }
      if (assignment.group === 'bench') {
        return swapLineupAndBench(source.index, assignment.index);
      }
      if (assignment.group === 'pitchers') {
        setTeamBuilderFeedback('投手リストとは入れ替えできません。', 'danger');
        clearPlayerSwapSelection();
        renderTeamBuilderView();
        return true;
      }
    }
    const form = getTeamBuilderForm();
    const slot = form?.lineup?.[source.index];
    if (!slot) {
      clearPlayerSwapSelection();
      return true;
    }
    const previous = getLineupPlayerSnapshot(source.index);
    applyRecordToLineupSlot(slot, record, slot.position);
    stateCache.teamBuilder.editorDirty = true;
    clearPlayerSwapSelection();
    const prevName = previous.playerName || previous.player?.name || '空き枠';
    setSelectionAndCatalog('lineup', source.index);
    setTeamBuilderFeedback(`${record.name} と ${prevName} を入れ替えました。`, 'success');
    renderTeamBuilderView();
    return true;
  }

  if (source.group === 'bench') {
    if (assignment) {
      if (assignment.group === 'bench') {
        return swapBenchPlayers(source.index, assignment.index);
      }
      if (assignment.group === 'lineup') {
        return swapLineupAndBench(assignment.index, source.index);
      }
      if (assignment.group === 'pitchers') {
        setTeamBuilderFeedback('投手リストとは入れ替えできません。', 'danger');
        clearPlayerSwapSelection();
        renderTeamBuilderView();
        return true;
      }
    }
    const form = getTeamBuilderForm();
    if (!form) {
      clearPlayerSwapSelection();
      return true;
    }
    if (!form.bench[source.index]) {
      form.bench[source.index] = createEmptyBenchEntry();
    }
    const entry = form.bench[source.index];
    const previous = getBenchPlayerSnapshot(source.index);
    setBenchEntryFromData(entry, { player: record });
    stateCache.teamBuilder.editorDirty = true;
    clearPlayerSwapSelection();
    setSelectionAndCatalog('bench', source.index);
    const prevName = previous.playerName || previous.player?.name || '空き枠';
    setTeamBuilderFeedback(`${record.name} と ${prevName} を入れ替えました。`, 'success');
    renderTeamBuilderView();
    return true;
  }

  return false;
}

function clearAssignmentAt(group, index) {
  if (!group || !Number.isInteger(index) || index < 0) return;
  if (group === 'lineup') {
    clearLineupSlot(index);
  } else if (group === 'bench') {
    clearBenchEntry(index);
    stateCache.teamBuilder.editorDirty = true;
  } else if (group === 'pitchers') {
    clearPitcherEntry(index);
    stateCache.teamBuilder.editorDirty = true;
  }
}

function canAssignRecordToSelection(record, selection) {
  if (!record) return false;
  const target = selection?.group || 'lineup';
  if (target === 'pitchers') {
    return record.role === 'pitcher';
  }
  // For lineup and bench, allow any player regardless of eligibility or position
  return true;
}

function assignPlayerToLineup(index, record) {
  const form = getTeamBuilderForm();
  const slot = form?.lineup?.[index];
  if (!slot || !record) return;
  const existing = findPlayerAssignment(record.id);
  const isSameSlot = existing && existing.group === 'lineup' && existing.index === index;
  const previous = getLineupPlayerSnapshot(index);
  applyRecordToLineupSlot(slot, record, slot.position);

  if (existing && !isSameSlot) {
    if (existing.group === 'bench') {
      if (!form.bench[existing.index]) {
        form.bench[existing.index] = createEmptyBenchEntry();
      }
      setBenchEntryFromData(form.bench[existing.index], previous);
    } else if (existing.group === 'lineup') {
      const targetSlot = form.lineup[existing.index];
      setLineupSlotFromPlayerData(targetSlot, previous);
    } else if (existing.group === 'pitchers') {
      clearPitcherEntry(existing.index);
    }
  }

  setSelectionAndCatalog('lineup', index);
  stateCache.teamBuilder.editorDirty = true;
  clearPlayerSwapSelection();
  const prevName = previous.playerName || previous.player?.name || '空き枠';
  const swapped = existing && !isSameSlot && prevName !== record.name;
  if (swapped) {
    setTeamBuilderFeedback(`${record.name} と ${prevName} を入れ替えました。`, 'success');
  } else {
    setTeamBuilderFeedback(`${record.name} を${slot.position || 'DH'}に割り当てました。`, 'success');
  }
  renderTeamBuilderView();
}

function assignPlayerToBench(index, record) {
  const form = getTeamBuilderForm();
  if (!form || index < 0) return;
  if (!form.bench[index]) {
    form.bench[index] = createEmptyBenchEntry();
  }
  const existing = findPlayerAssignment(record.id);
  const isSameSlot = existing && existing.group === 'bench' && existing.index === index;
  const entry = form.bench[index];
  const previous = getBenchPlayerSnapshot(index);

  setBenchEntryFromData(entry, { player: record });

  if (existing && !isSameSlot) {
    if (existing.group === 'lineup') {
      const targetSlot = form.lineup[existing.index];
      setLineupSlotFromPlayerData(targetSlot, previous);
    } else if (existing.group === 'bench') {
      if (!form.bench[existing.index]) {
        form.bench[existing.index] = createEmptyBenchEntry();
      }
      setBenchEntryFromData(form.bench[existing.index], previous);
    } else if (existing.group === 'pitchers') {
      clearPitcherEntry(existing.index);
    }
  }

  setSelectionAndCatalog('bench', index);
  stateCache.teamBuilder.editorDirty = true;
  clearPlayerSwapSelection();
  const prevName = previous.playerName || previous.player?.name || '空き枠';
  const swapped = existing && !isSameSlot && prevName !== record.name;
  if (swapped) {
    setTeamBuilderFeedback(`${record.name} と ${prevName} を入れ替えました。`, 'success');
  } else {
    setTeamBuilderFeedback(`${record.name} をベンチ枠に追加しました。`, 'success');
  }
  renderTeamBuilderView();
}

function assignPlayerToPitchers(index, record) {
  if (!record || record.role !== 'pitcher') {
    setTeamBuilderFeedback('投手リストには投手のみ割り当てられます。', 'danger');
    return;
  }
  const form = getTeamBuilderForm();
  if (!form || index < 0) return;
  if (!form.pitchers[index]) {
    form.pitchers[index] = createEmptyPitcherEntry();
  }
  const existing = findPlayerAssignment(record.id);
  if (existing && (existing.group !== 'pitchers' || existing.index !== index)) {
    clearAssignmentAt(existing.group, existing.index);
  }
  const entry = form.pitchers[index];
  entry.playerId = record.id;
  entry.playerName = record.name;
  entry.player = record;
  entry.playerRole = record.role;
  setSelectionAndCatalog('pitchers', index);
  stateCache.teamBuilder.editorDirty = true;
  setTeamBuilderFeedback(`${record.name} を投手リストに追加しました。`, 'success');
  renderTeamBuilderView();
}

function assignPlayerToSelection(record) {
  ensureTeamBuilderState();
  const selection = stateCache.teamBuilder.selection || { group: 'lineup', index: 0 };
  if (!canAssignRecordToSelection(record, selection)) {
    setTeamBuilderFeedback('この枠には選択した選手を割り当てられません。', 'danger');
    return;
  }
  if (selection.group === 'bench') {
    assignPlayerToBench(selection.index, record);
    return;
  }
  if (selection.group === 'pitchers') {
    assignPlayerToPitchers(selection.index, record);
    return;
  }
  assignPlayerToLineup(selection.index, record);
}

function getSelectedRecord() {
  const selection = stateCache.teamBuilder.selection || { group: 'lineup', index: 0 };
  const form = getTeamBuilderForm();
  if (selection.group === 'bench') {
    return form.bench?.[selection.index]?.player || null;
  }
  if (selection.group === 'pitchers') {
    return form.pitchers?.[selection.index]?.player || null;
  }
  return form.lineup?.[selection.index]?.player || null;
}

function renderTeamBuilderLineup() {
  const container = elements.teamBuilderLineup;
  if (!container) return;
  container.innerHTML = '';
  const form = getTeamBuilderForm();
  const selection = stateCache.teamBuilder.selection || { group: 'lineup', index: 0 };
  const swapSelection = stateCache.teamBuilder.positionSwap;
  const playerSwapSource = stateCache.teamBuilder.playerSwap?.source;
  const selectedPositionIndexRaw = Number.isInteger(swapSelection?.first)
    ? swapSelection.first
    : null;
  const selectedPositionIndex =
    selectedPositionIndexRaw != null &&
    selectedPositionIndexRaw >= 0 &&
    selectedPositionIndexRaw < form.lineup.length
      ? selectedPositionIndexRaw
      : null;

  form.lineup.forEach((slot, index) => {
    const row = document.createElement('div');
    row.className = 'defense-lineup-row';
    if (selection.group === 'lineup' && selection.index === index) {
      row.classList.add('selected');
    }
    const isSwapSource = playerSwapSource?.group === 'lineup' && playerSwapSource.index === index;
    if (isSwapSource) {
      row.classList.add('swap-source');
    }

    const order = document.createElement('span');
    order.className = 'lineup-order';
    order.textContent = `${index + 1}.`;

    const positionButton = document.createElement('button');
    positionButton.type = 'button';
    positionButton.className = 'defense-action-button lineup-position-button';
    positionButton.dataset.builderAction = 'select-position';
    positionButton.dataset.group = 'lineup';
    positionButton.dataset.index = String(index);
    positionButton.innerHTML = renderPositionToken(slot.position || '-', slot.player?.pitcher_type, 'position-token');
    positionButton.setAttribute('aria-pressed', selectedPositionIndex === index ? 'true' : 'false');
    if (selectedPositionIndex === index) {
      positionButton.classList.add('selected');
    }

    const playerButton = document.createElement('button');
    playerButton.type = 'button';
    playerButton.className = 'defense-action-button lineup-player-button';
    playerButton.dataset.builderAction = 'select-slot';
    playerButton.dataset.group = 'lineup';
    playerButton.dataset.index = String(index);
    const displayName = slot.playerName ? escapeHtml(slot.playerName) : '選手を選択';
    const missing = slot.playerName && !slot.player ? '<small class="empty-hint">(未登録)</small>' : '';
    playerButton.innerHTML = `<span>${displayName}</span> ${missing}`;

    const meta = document.createElement('div');
    meta.className = 'lineup-meta';
    const eligibleLabel = document.createElement('span');
    eligibleLabel.className = 'eligible-label';
    eligibleLabel.textContent = '守備適性';
    const eligibleSpan = document.createElement('span');
    eligibleSpan.className = 'eligible-positions';
    eligibleSpan.innerHTML = renderPositionList(slot.eligible || [], slot.player?.pitcher_type);
    meta.append(eligibleLabel, eligibleSpan);

    const abilityRow = document.createElement('div');
    abilityRow.className = 'chip-row';
    const isPitcher = slot.playerRole === 'pitcher';
    const chips = isPitcher ? getPitcherAbilityChips(slot.player) : getBatterAbilityChips(slot.player);
    abilityRow.innerHTML = chips.join('');
    if (chips.length) {
      meta.appendChild(abilityRow);
      // Apply color coding to chip values
      colorizeAbilityChips(abilityRow, isPitcher ? 'pitcher' : 'batter');
    }

    const swapButton = document.createElement('button');
    swapButton.type = 'button';
    swapButton.className = 'swap-button';
    swapButton.dataset.builderAction = 'swap-slot';
    swapButton.dataset.group = 'lineup';
    swapButton.dataset.index = String(index);
    swapButton.textContent = '入れ替え';
    swapButton.setAttribute('aria-pressed', isSwapSource ? 'true' : 'false');
    if (!slot.playerName) {
      swapButton.disabled = true;
      swapButton.classList.add('disabled');
    }
    if (isSwapSource) {
      swapButton.classList.add('active');
    }
    meta.appendChild(swapButton);

    if (slot.playerName) {
      const clearButton = document.createElement('button');
      clearButton.type = 'button';
      clearButton.className = 'builder-clear-button';
      clearButton.dataset.builderAction = 'clear-lineup';
      clearButton.dataset.index = String(index);
      clearButton.textContent = 'クリア';
      meta.appendChild(clearButton);
    }

    row.append(order, positionButton, playerButton, meta);
    container.appendChild(row);
  });
}

function renderTeamBuilderBench() {
  const container = elements.teamBuilderBench;
  if (!container) return;
  container.innerHTML = '';
  const form = getTeamBuilderForm();
  const selection = stateCache.teamBuilder.selection || {};
  const playerSwapSource = stateCache.teamBuilder.playerSwap?.source;

  if (!form.bench.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-message';
    empty.textContent = 'ベンチ枠がありません。枠を追加してください。';
    container.appendChild(empty);
    return;
  }

  form.bench.forEach((entry, index) => {
    const slotDiv = document.createElement('div');
    slotDiv.className = 'builder-bench-slot';
    if (selection.group === 'bench' && selection.index === index) {
      slotDiv.classList.add('selected');
    }
    const isSwapSource = playerSwapSource?.group === 'bench' && playerSwapSource.index === index;
    if (isSwapSource) {
      slotDiv.classList.add('swap-source');
    }

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'defense-action-button bench-player-button';
    button.dataset.builderAction = 'select-slot';
    button.dataset.group = 'bench';
    button.dataset.index = String(index);
    const displayName = entry?.playerName ? escapeHtml(entry.playerName) : '空き枠';
    button.innerHTML = `<strong>${displayName}</strong>`;

    const meta = document.createElement('div');
    meta.className = 'bench-meta';
    if (entry?.playerName && !entry.player) {
      const hint = document.createElement('span');
      hint.className = 'empty-hint';
      hint.textContent = '選手データが見つかりません。';
      meta.appendChild(hint);
    }
    if (entry?.playerRole === 'pitcher') {
      const role = document.createElement('span');
      role.className = 'eligible-label';
      role.textContent = '投手';
      meta.appendChild(role);
    } else if (Array.isArray(entry?.eligible) && entry.eligible.length) {
      const eligibleLabel = document.createElement('span');
      eligibleLabel.className = 'eligible-label';
      eligibleLabel.textContent = '守備適性';
      const eligibleSpan = document.createElement('span');
      eligibleSpan.className = 'eligible-positions';
      eligibleSpan.innerHTML = renderPositionList(entry.eligible, entry.player?.pitcher_type);
      meta.append(eligibleLabel, eligibleSpan);
    }

    const abilityRow = document.createElement('div');
    abilityRow.className = 'chip-row';
    const isPitcher = entry?.playerRole === 'pitcher';
    const chips = isPitcher ? getPitcherAbilityChips(entry.player) : getBatterAbilityChips(entry.player);
    abilityRow.innerHTML = chips.join('');
    if (chips.length) {
      meta.appendChild(abilityRow);
      colorizeAbilityChips(abilityRow, isPitcher ? 'pitcher' : 'batter');
    }

    button.appendChild(meta);
    slotDiv.appendChild(button);

    const swapButton = document.createElement('button');
    swapButton.type = 'button';
    swapButton.className = 'swap-button';
    swapButton.dataset.builderAction = 'swap-slot';
    swapButton.dataset.group = 'bench';
    swapButton.dataset.index = String(index);
    swapButton.textContent = '入れ替え';
    swapButton.setAttribute('aria-pressed', isSwapSource ? 'true' : 'false');
    const hasPlayer = Boolean(entry?.playerName);
    if (!hasPlayer) {
      swapButton.disabled = true;
      swapButton.classList.add('disabled');
    }
    if (isSwapSource) {
      swapButton.classList.add('active');
    }
    slotDiv.appendChild(swapButton);

    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'remove-button';
    removeButton.dataset.builderAction = 'remove-bench-slot';
    removeButton.dataset.index = String(index);
    removeButton.textContent = '削除';
    slotDiv.appendChild(removeButton);

    container.appendChild(slotDiv);
  });
}

function renderTeamBuilderPitchers() {
  const container = elements.teamBuilderPitchers;
  if (!container) return;
  container.innerHTML = '';
  const form = getTeamBuilderForm();
  const selection = stateCache.teamBuilder.selection || {};

  if (!form.pitchers.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-message';
    empty.textContent = '投手枠がありません。枠を追加してください。';
    container.appendChild(empty);
    return;
  }

  form.pitchers.forEach((entry, index) => {
    const slotDiv = document.createElement('div');
    slotDiv.className = 'builder-pitcher-slot';
    if (selection.group === 'pitchers' && selection.index === index) {
      slotDiv.classList.add('selected');
    }

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'defense-action-button bench-player-button';
    button.dataset.builderAction = 'select-slot';
    button.dataset.group = 'pitchers';
    button.dataset.index = String(index);
    const displayName = entry?.playerName ? escapeHtml(entry.playerName) : '空き枠';
    button.innerHTML = `<strong>${displayName}</strong>`;

    const meta = document.createElement('div');
    meta.className = 'bench-meta';
    if (entry?.playerName && !entry.player) {
      const hint = document.createElement('span');
      hint.className = 'empty-hint';
      hint.textContent = '選手データが見つかりません。';
      meta.appendChild(hint);
    }

    const abilityRow = document.createElement('div');
    abilityRow.className = 'chip-row';
    const chips = getPitcherAbilityChips(entry.player);
    abilityRow.innerHTML = chips.join('');
    if (chips.length) {
      meta.appendChild(abilityRow);
      colorizeAbilityChips(abilityRow, 'pitcher');
    }

    button.appendChild(meta);
    slotDiv.appendChild(button);

    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'remove-button';
    removeButton.dataset.builderAction = 'remove-pitcher-slot';
    removeButton.dataset.index = String(index);
    removeButton.textContent = '削除';
    slotDiv.appendChild(removeButton);

    container.appendChild(slotDiv);
  });
}

function updateCatalogButtons() {
  const active = stateCache.teamBuilder.catalog === 'pitchers' ? 'pitchers' : 'batters';
  elements.teamBuilderCatalogButtons?.forEach((button) => {
    const value = button.dataset.builderCatalog === 'pitchers' ? 'pitchers' : 'batters';
    const isActive = value === active;
    button.classList.toggle('active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
}

function updateRosterPanels() {
  const active = stateCache.teamBuilder.catalog === 'pitchers' ? 'pitchers' : 'batters';
  elements.teamBuilderRosterPanels?.forEach((panel) => {
    if (!panel) return;
    const value = panel.dataset.rosterPanel === 'pitchers' ? 'pitchers' : 'batters';
    const isActive = value === active;
    panel.classList.toggle('hidden', !isActive);
    panel.setAttribute('aria-hidden', isActive ? 'false' : 'true');
  });
}

function renderTeamBuilderCatalog() {
  const panel = elements.teamBuilderPlayerPanel;
  if (!panel) return;
  const players = stateCache.teamBuilder.players || {};
  panel.innerHTML = '';
  if (players.loading) {
    const loading = document.createElement('p');
    loading.className = 'builder-empty';
    loading.textContent = '選手データを読み込み中です...';
    panel.appendChild(loading);
    return;
  }
  const catalog = stateCache.teamBuilder.catalog === 'pitchers' ? 'pitchers' : 'batters';
  let list = Array.isArray(players[catalog]) ? players[catalog] : [];
  // Exclude players already assigned anywhere (lineup, bench, pitchers) from the selectable list
  try {
    const form = getTeamBuilderForm();
    const takenIds = new Set(
      [
        ...((form?.lineup || []).map((slot) => (slot && slot.playerId ? String(slot.playerId) : null))),
        ...((form?.bench || []).map((entry) => (entry && entry.playerId ? String(entry.playerId) : null))),
        ...((form?.pitchers || []).map((entry) => (entry && entry.playerId ? String(entry.playerId) : null))),
      ].filter((id) => id)
    );
    if (takenIds.size > 0) {
      list = list.filter((record) => record && !takenIds.has(String(record.id)));
    }
  } catch (_) {
    // noop – fall back to unfiltered list if anything goes wrong
  }
  const term = (stateCache.teamBuilder.searchTerm || '').trim().toLowerCase();
  const filtered = term
    ? list.filter((record) => record.name && record.name.toLowerCase().includes(term))
    : list;
  if (!filtered.length) {
    const empty = document.createElement('p');
    empty.className = 'builder-empty';
    empty.textContent = '該当する選手が見つかりません。';
    panel.appendChild(empty);
    return;
  }
  const assignedSelection = getSelectedRecord();
  const playerSwapSource = stateCache.teamBuilder.playerSwap?.source;
  filtered.forEach((record) => {
    const assignment = findPlayerAssignment(record.id);
  // Already-assigned players are filtered above; keep guard for safety
  if (assignment) return;
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'builder-player-card';
    card.dataset.builderPlayerId = record.id;
    card.dataset.builderRole = record.role;
    if (assignment) {
      card.classList.add('assigned');
    }
    if (
      playerSwapSource &&
      assignment &&
      playerSwapSource.group === assignment.group &&
      playerSwapSource.index === assignment.index
    ) {
      card.classList.add('swap-source');
    }
    if (assignedSelection && assignedSelection.id === record.id) {
      card.classList.add('active');
    }
    const selection = stateCache.teamBuilder.selection || { group: 'lineup', index: 0 };
    if (!canAssignRecordToSelection(record, selection)) {
      card.disabled = true;
      card.classList.add('disabled');
    }

    const name = document.createElement('span');
    name.className = 'player-name';
    name.textContent = record.name;
    card.appendChild(name);

    const meta = document.createElement('div');
    meta.className = 'player-meta';
    if (record.role === 'batter') {
      const positions = document.createElement('div');
      positions.className = 'player-positions';
      positions.innerHTML = renderPositionList(record.eligible || [], null);
      meta.appendChild(positions);
      const abilityRow = document.createElement('div');
      abilityRow.className = 'chip-row';
  abilityRow.innerHTML = getBatterAbilityChips(record).join('');
      meta.appendChild(abilityRow);
  colorizeAbilityChips(abilityRow, 'batter');
    } else {
      const roleInfo = document.createElement('div');
      roleInfo.className = 'player-positions';
      roleInfo.textContent = `Type: ${record.pitcher_type || '-'} / Throws: ${record.throws || '-'}`;
      meta.appendChild(roleInfo);
      const abilityRow = document.createElement('div');
      abilityRow.className = 'chip-row';
  abilityRow.innerHTML = getPitcherAbilityChips(record).join('');
      meta.appendChild(abilityRow);
  colorizeAbilityChips(abilityRow, 'pitcher');
    }

    card.appendChild(meta);
    panel.appendChild(card);
  });
}

function renderTeamBuilderView() {
  ensureTeamBuilderState();
  if (stateCache.uiView !== 'team-builder') {
    return;
  }
  const lineupLength = Array.isArray(stateCache.teamBuilder.form?.lineup)
    ? stateCache.teamBuilder.form.lineup.length
    : 0;
  const selectedPositionIndex = stateCache.teamBuilder.positionSwap?.first;
  if (
    !Number.isInteger(selectedPositionIndex) ||
    selectedPositionIndex < 0 ||
    selectedPositionIndex >= lineupLength
  ) {
    resetLineupPositionSelection();
  }
  if (elements.teamBuilderName) {
    const desired = stateCache.teamBuilder.form?.name || '';
    if (document.activeElement !== elements.teamBuilderName || elements.teamBuilderName.value !== desired) {
      elements.teamBuilderName.value = desired;
    }
  }
  if (elements.teamBuilderSearch) {
    const desired = stateCache.teamBuilder.searchTerm || '';
    if (document.activeElement !== elements.teamBuilderSearch || elements.teamBuilderSearch.value !== desired) {
      elements.teamBuilderSearch.value = desired;
    }
  }
  updateCatalogButtons();
  updateRosterPanels();
  renderTeamBuilderLineup();
  renderTeamBuilderBench();
  renderTeamBuilderPitchers();
  renderTeamBuilderCatalog();
}

function buildTeamPayload() {
  const form = getTeamBuilderForm();
  const name = (form.name || '').trim();
  if (!name) {
    setTeamBuilderFeedback('チーム名を入力してください。', 'danger');
    return null;
  }

  const lineup = [];
  for (let index = 0; index < form.lineup.length; index += 1) {
    const slot = form.lineup[index];
    const playerName = slot?.playerName ? String(slot.playerName).trim() : '';
    const position = slot?.position ? String(slot.position).trim().toUpperCase() : '';
    if (!playerName) {
      setTeamBuilderFeedback(`${index + 1}番の選手が未設定です。`, 'danger');
      return null;
    }
    if (!position) {
      setTeamBuilderFeedback(`${playerName} の守備位置を選択してください。`, 'danger');
      return null;
    }
    lineup.push({ name: playerName, position });
  }

  const pitchers = (form.pitchers || [])
    .map((entry) => (entry?.playerName ? String(entry.playerName).trim() : ''))
    .filter((nameValue) => nameValue);
  if (!pitchers.length) {
    setTeamBuilderFeedback('少なくとも1人の投手を割り当ててください。', 'danger');
    return null;
  }

  const bench = (form.bench || [])
    .map((entry) => (entry?.playerName ? String(entry.playerName).trim() : ''))
    .filter((nameValue) => nameValue);

  return { name, lineup, pitchers, bench };
}

function findLineupEligibilityConflicts() {
  const form = getTeamBuilderForm();
  const issues = [];
  if (!form || !Array.isArray(form.lineup)) return issues;
  form.lineup.forEach((slot, index) => {
    if (!slot) return;
    const pos = (slot.position || '').toString().toUpperCase();
    const name = (slot.playerName || '').toString();
    const player = slot.player;
    if (!pos || !name || !player) return;
    if (player.role === 'pitcher') {
      if (pos !== 'P') {
        issues.push(`${index + 1}番 ${name}: ${pos} は投手の守備適性外です`);
      }
    } else {
      const eligible = Array.isArray(player.eligible)
        ? player.eligible.map((p) => String(p).toUpperCase())
        : [];
      if (!eligible.includes(pos)) {
        issues.push(`${index + 1}番 ${name}: ${pos} は守備適性外です`);
      }
    }
  });
  return issues;
}

function createNewTeamTemplate() {
  ensureTeamBuilderState();
  stateCache.teamBuilder.form = createDefaultTeamForm();
  stateCache.teamBuilder.currentTeamId = null;
  stateCache.teamBuilder.lastSavedId = '__new__';
  stateCache.teamBuilder.editorDirty = false;
  setSelectionAndCatalog('lineup', 0);
  resetLineupPositionSelection();
  clearPlayerSwapSelection();
  captureTeamBuilderSnapshot(stateCache.teamBuilder.form);
  setTeamBuilderFeedback('テンプレートを読み込みました。', 'info');
  renderTeamBuilderView();
}

function resetTeamBuilderFormToInitial(showFeedback = true) {
  ensureTeamBuilderState();
  const snapshot = stateCache.teamBuilder.initialForm;
  if (snapshot) {
    stateCache.teamBuilder.form = cloneTeamForm(snapshot);
  } else {
    stateCache.teamBuilder.form = createDefaultTeamForm();
    captureTeamBuilderSnapshot(stateCache.teamBuilder.form);
  }
  setSelectionAndCatalog('lineup', 0);
  stateCache.teamBuilder.editorDirty = false;
  resetLineupPositionSelection();
  clearPlayerSwapSelection();
  relinkFormPlayers();
  if (showFeedback) {
    setTeamBuilderFeedback('チームを初期状態に戻しました。', 'info');
  }
  renderTeamBuilderView();
}

async function ensureTeamBuilderPlayersLoaded(actions, forceReload = false) {
  ensureTeamBuilderState();
  const players = stateCache.teamBuilder.players;
  if (forceReload) {
    players.loaded = false;
    stateCache.teamBuilder.playersLoadingPromise = null;
  }
  if (players.loaded && !forceReload) {
    return players;
  }
  if (!forceReload && stateCache.teamBuilder.playersLoadingPromise) {
    return stateCache.teamBuilder.playersLoadingPromise;
  }

  const loadPromise = (async () => {
    stateCache.teamBuilder.players.loading = true;
    try {
      const catalog = await actions.fetchPlayersCatalog();
      applyPlayersCatalogData(catalog);
      stateCache.teamBuilder.players.loaded = true;
      return stateCache.teamBuilder.players;
    } catch (error) {
      stateCache.teamBuilder.players.loaded = false;
      throw error;
    } finally {
      stateCache.teamBuilder.players.loading = false;
      stateCache.teamBuilder.playersLoadingPromise = null;
    }
  })();

  stateCache.teamBuilder.playersLoadingPromise = loadPromise;
  return loadPromise;
}

function applyTeamDataToForm(teamData) {
  ensureTeamBuilderState();
  const form = createDefaultTeamForm();
  form.name = typeof teamData?.name === 'string' ? teamData.name : '';

  const lineupEntries = Array.isArray(teamData?.lineup) ? teamData.lineup : [];
  lineupEntries.forEach((entry, index) => {
    if (index >= form.lineup.length) return;
    const slot = form.lineup[index];
    const playerName = entry?.name ? String(entry.name).trim() : '';
    const position = entry?.position ? String(entry.position).trim().toUpperCase() : slot.position;
    slot.position = position || slot.position;
    if (playerName) {
      const record = resolvePlayerByName(playerName, position);
      if (record) {
        applyRecordToLineupSlot(slot, record, position);
      } else {
        slot.playerId = null;
        slot.playerName = playerName;
        slot.playerRole = null;
        slot.player = null;
        slot.eligible = [slot.position];
      }
    } else {
      slot.playerId = null;
      slot.playerName = '';
      slot.playerRole = null;
      slot.player = null;
      slot.eligible = [slot.position];
    }
  });

  const benchEntries = Array.isArray(teamData?.bench) ? teamData.bench : [];
  form.bench = benchEntries.map((nameValue) => {
    const entry = createEmptyBenchEntry();
    const playerName = nameValue ? String(nameValue).trim() : '';
    if (!playerName) {
      return entry;
    }
    const record = resolvePlayerByName(playerName);
    if (record) {
      entry.playerId = record.id;
      entry.playerName = record.name;
      entry.player = record;
      entry.playerRole = record.role;
      entry.eligible = record.role === 'pitcher' ? ['P'] : [...(record.eligible || [])];
    } else {
      entry.playerName = playerName;
    }
    return entry;
  });
  if (!form.bench.length) {
    form.bench = Array.from({ length: TEAM_BUILDER_DEFAULT_BENCH_SLOTS }, () => createEmptyBenchEntry());
  }

  const pitcherEntries = Array.isArray(teamData?.pitchers) ? teamData.pitchers : [];
  form.pitchers = pitcherEntries.map((nameValue) => {
    const entry = createEmptyPitcherEntry();
    const playerName = nameValue ? String(nameValue).trim() : '';
    if (!playerName) {
      return entry;
    }
    const record = resolvePlayerByName(playerName, 'P');
    if (record) {
      entry.playerId = record.id;
      entry.playerName = record.name;
      entry.player = record;
      entry.playerRole = record.role;
    } else {
      entry.playerName = playerName;
    }
    return entry;
  });
  if (!form.pitchers.length) {
    form.pitchers = Array.from({ length: TEAM_BUILDER_DEFAULT_PITCHER_SLOTS }, () => createEmptyPitcherEntry());
  }

  stateCache.teamBuilder.form = form;
  setSelectionAndCatalog('lineup', 0);
  stateCache.teamBuilder.editorDirty = false;
  resetLineupPositionSelection();
  clearPlayerSwapSelection();
  relinkFormPlayers(form);
  captureTeamBuilderSnapshot(form);
  renderTeamBuilderView();
}

function sanitizeSearchTerm(value) {
  if (value == null) return '';
  return String(value).trim();
}

export function initEventListeners(actions) {
  // --- Home: Delete dialogs helpers ---
  function setTeamDeleteFeedback(message, level = 'info') {
    const el = elements.teamDeleteFeedback;
    if (!el) return;
    el.textContent = message || '';
    el.classList.remove('danger', 'success', 'info');
    if (message) {
      el.classList.add(level);
      el.dataset.level = level;
    } else {
      el.removeAttribute('data-level');
    }
  }

  function setPlayerDeleteFeedback(message, level = 'info') {
    const el = elements.playerDeleteFeedback;
    if (!el) return;
    el.textContent = message || '';
    el.classList.remove('danger', 'success', 'info');
    if (message) {
      el.classList.add(level);
      el.dataset.level = level;
    } else {
      el.removeAttribute('data-level');
    }
  }

  function populateTeamDeleteOptions() {
    const select = elements.teamDeleteSelect;
    if (!select) return [];
    const teams = (stateCache.data?.team_library?.teams || []).map((t) => ({
      id: String(t?.id || ''),
      name: String(t?.name || '') || String(t?.id || ''),
    }));
    const prev = select.value;
    select.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'チームを選択';
    select.appendChild(placeholder);
    teams.forEach((team) => {
      if (!team.id) return;
      const opt = document.createElement('option');
      opt.value = team.id;
      opt.textContent = team.name;
      select.appendChild(opt);
    });
    if (teams.some((t) => t.id === prev)) {
      select.value = prev;
    } else {
      select.value = '';
    }
    return teams;
  }

  async function populatePlayerDeleteOptions(role) {
    const select = elements.playerDeleteSelect;
    if (!select) return [];
    const normalized = role === 'pitcher' ? 'pitcher' : 'batter';
    const players = await actions.fetchPlayersList(normalized);
    const prev = select.value;
    select.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = '選手を選択';
    select.appendChild(placeholder);
    players.forEach((p) => {
      if (!p?.id) return;
      const opt = document.createElement('option');
      opt.value = String(p.id);
      opt.textContent = String(p.name || p.id);
      select.appendChild(opt);
    });
    if (players.some((p) => String(p.id) === prev)) {
      select.value = prev;
    } else {
      select.value = '';
    }
    return players;
  }
  async function populatePlayerSelect(role, desiredValue) {
    const select = elements.playerEditorSelect;
    if (!select) return [];
    const players = await actions.fetchPlayersList(role);
    const prevValue = desiredValue ?? select.value;
    select.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = '選手を選択';
    select.appendChild(placeholder);
    const newOpt = document.createElement('option');
    newOpt.value = '__new__';
    newOpt.textContent = '新規選手を作成';
    select.appendChild(newOpt);
    players.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p.id; // use stable id
      opt.textContent = p.name;
      select.appendChild(opt);
    });
    const validValues = new Set(['', '__new__', ...players.map((p) => p.id)]);
    const targetValue = prevValue && validValues.has(prevValue) ? prevValue : '';
    select.value = targetValue;
    return players;
  }

  async function submitTitleLineup(teamKey, button) {
    const normalizedTeam = teamKey === 'home' ? 'home' : teamKey === 'away' ? 'away' : null;
    if (!normalizedTeam) {
      showStatus('スタメンを更新するチームを正しく指定してください。', 'danger');
      return;
    }

    const container = elements.titleScreen?.querySelector(`[data-title-lineup="${normalizedTeam}"]`);
    if (!container) {
      showStatus('スタメン編集エリアが見つかりません。', 'danger');
      return;
    }

    const teamsData = stateCache.data?.teams || {};
    const teamData = teamsData[normalizedTeam];
    ensureTitleLineupPlan(normalizedTeam, teamData, Boolean(teamData));
    const plan = getTitleLineupPlan(normalizedTeam);
    if (!plan) {
      showStatus('スタメンを編成できる状態ではありません。', 'danger');
      return;
    }

    const lineupEntries = [];
    const seenNames = new Set();
    for (const slot of plan.lineup) {
      const player = slot?.player;
      const playerName = player?.name ? String(player.name).trim() : '';
      const position = slot?.slotPositionKey ? String(slot.slotPositionKey).trim().toUpperCase() : '';
      if (!playerName) {
        showStatus('すべての打順に選手を割り当ててください。', 'danger');
        return;
      }
      if (!position) {
        showStatus('ポジション情報を取得できませんでした。', 'danger');
        return;
      }
      if (seenNames.has(playerName)) {
        showStatus(`${playerName} が複数の打順に選択されています。`, 'danger');
        return;
      }
      seenNames.add(playerName);
      lineupEntries.push({ name: playerName, position });
    }

    const originalText = button?.textContent || '';
    if (button) {
      button.disabled = true;
      button.textContent = '更新中…';
    }

    try {
      await actions.handleLineupUpdate(normalizedTeam, lineupEntries);
    } catch (error) {
      // Error feedback is handled within handleLineupUpdate via handleApiError.
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = originalText || 'スタメンを更新';
      }
    }
  }

  async function submitTitlePitcher(teamKey, button) {
    const normalizedTeam = teamKey === 'home' ? 'home' : teamKey === 'away' ? 'away' : null;
    if (!normalizedTeam) {
      showStatus('先発投手を設定するチームを正しく指定してください。', 'danger');
      return;
    }

    const select = elements.titleScreen?.querySelector(`.title-pitcher-select[data-team="${normalizedTeam}"]`);
    if (!select) {
      showStatus('先発投手の選択欄が見つかりません。', 'danger');
      return;
    }

    const pitcherName = String(select.value || '').trim();
    if (!pitcherName) {
      showStatus('先発投手を選択してください。', 'danger');
      return;
    }

    const originalText = button?.textContent || '';
    if (button) {
      button.disabled = true;
      button.textContent = '設定中…';
    }

    try {
      await actions.handleSetStartingPitcher(normalizedTeam, pitcherName);
    } catch (error) {
      // Feedback is displayed by handleSetStartingPitcher/handleApiError.
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = originalText || '先発を設定';
      }
    }
  }

  updatePlayerRoleUI(elements.playerEditorRole?.value || 'batter');

  elements.startButton.addEventListener('click', () => actions.handleStart(false));
  elements.reloadTeams.addEventListener('click', actions.handleReloadTeams);
  elements.restartButton.addEventListener('click', () => actions.handleStart(true));
  elements.returnTitle.addEventListener('click', actions.handleReturnToTitle);
  elements.clearLog.addEventListener('click', actions.handleClearLog);
  elements.swingButton.addEventListener('click', actions.handleSwing);
  elements.buntButton.addEventListener('click', actions.handleBunt);
  if (elements.stealButton) {
    elements.stealButton.addEventListener('click', actions.handleSteal);
  }

  if (elements.pinchButton) {
    elements.pinchButton.addEventListener('click', actions.handlePinchHit);
  }
  if (elements.pinchRunButton) {
    elements.pinchRunButton.addEventListener('click', actions.handlePinchRun);
  }
  if (elements.openOffenseButton) {
    elements.openOffenseButton.addEventListener('click', toggleOffenseMenu);
  }
  if (elements.offensePinchMenuButton) {
    elements.offensePinchMenuButton.addEventListener('click', () => openModal('offense'));
  }
  if (elements.offensePinchRunMenuButton) {
    elements.offensePinchRunMenuButton.addEventListener('click', () => openModal('pinch-run'));
  }
  if (elements.openDefenseButton) {
    elements.openDefenseButton.addEventListener('click', toggleDefenseMenu);
  }
  if (elements.defenseSubMenuButton) {
    elements.defenseSubMenuButton.addEventListener('click', () => {
      updateDefenseSelectionInfo();
      openModal('defense');
    });
  }
  if (elements.pitcherMenuButton) {
    elements.pitcherMenuButton.addEventListener('click', () => openModal('pitcher'));
  }
  if (elements.openStatsButton) {
    elements.openStatsButton.addEventListener('click', () => {
      updateStatsPanel(stateCache.data);
      openModal('stats');
    });
  }
  if (elements.openAbilitiesButton) {
    elements.openAbilitiesButton.addEventListener('click', () => {
      updateAbilitiesPanel(stateCache.data);
      openModal('abilities');
    });
  }
  if (elements.defenseResetButton) {
    elements.defenseResetButton.addEventListener('click', actions.handleDefenseReset);
  }
  if (elements.defenseApplyButton) {
    elements.defenseApplyButton.addEventListener('click', actions.handleDefenseSubstitution);
  }
  if (elements.pitcherButton) {
    elements.pitcherButton.addEventListener('click', actions.handlePitcherChange);
  }

  if (elements.enterTitleButton) {
    elements.enterTitleButton.addEventListener('click', async () => {
      if (!elements.lobbyHomeSelect || !elements.lobbyAwaySelect) {
        return;
      }
      const homeId = elements.lobbyHomeSelect.value;
      const awayId = elements.lobbyAwaySelect.value;
      if (!homeId || !awayId) {
        showStatus('ホーム・アウェイのチームを選択してください。', 'danger');
        return;
      }
      elements.enterTitleButton.disabled = true;
      try {
        await actions.handleTeamSelection(homeId, awayId);
      } catch (error) {
        // エラーハンドリングはhandleTeamSelection内で行われる
      } finally {
        elements.enterTitleButton.disabled = false;
      }
    });
  }

  if (elements.titleScreen) {
    elements.titleScreen.addEventListener('click', (event) => {
      const lineupButton = event.target.closest('[data-action="apply-lineup"]');
      if (lineupButton) {
        event.preventDefault();
        const teamKey = lineupButton.dataset.team || '';
        submitTitleLineup(teamKey, lineupButton);
        return;
      }

      const pitcherButton = event.target.closest('[data-action="apply-pitcher"]');
      if (pitcherButton) {
        event.preventDefault();
        const teamKey = pitcherButton.dataset.team || '';
        submitTitlePitcher(teamKey, pitcherButton);
        return;
      }

      const interactiveTarget = event.target.closest('[data-title-role]');
      if (interactiveTarget) {
        event.preventDefault();
        const role = interactiveTarget.dataset.titleRole;
        const rawTeam = interactiveTarget.dataset.team || '';
        const normalizedTeam = rawTeam === 'home' ? 'home' : rawTeam === 'away' ? 'away' : null;
        if (!normalizedTeam) {
          clearTitleLineupSelection();
          renderTitle(stateCache.data?.title || {});
          return;
        }

        const teamsData = stateCache.data?.teams || {};
        const teamData = teamsData[normalizedTeam];
        ensureTitleLineupPlan(normalizedTeam, teamData, Boolean(teamData));
        const plan = getTitleLineupPlan(normalizedTeam);
        if (!plan) {
          clearTitleLineupSelection();
          renderTitle(stateCache.data?.title || {});
          return;
        }

        const selection = getTitleLineupSelection();

        if (role === 'lineup') {
          const index = Number.parseInt(interactiveTarget.dataset.index || '', 10);
          if (!Number.isInteger(index)) {
            clearTitleLineupSelection();
            renderTitle(stateCache.data?.title || {});
            return;
          }

          const field = interactiveTarget.dataset.lineupField === 'position' ? 'position' : 'player';
          const selectionField =
            selection.field === 'position' ? 'position' : selection.field === 'player' ? 'player' : null;

          if (selection.team === normalizedTeam && selection.type === 'lineup') {
            if (selection.index === index && (selectionField === null || selectionField === field)) {
              clearTitleLineupSelection();
            } else if (selectionField === 'position' && field === 'position') {
              if (swapTitleLineupPositions(normalizedTeam, selection.index, index)) {
                clearTitleLineupSelection();
              } else {
                setTitleLineupSelection(normalizedTeam, 'lineup', index, field);
              }
            } else if (selectionField !== 'position' && field !== 'position') {
              if (swapTitleLineupPlayers(normalizedTeam, selection.index, index)) {
                clearTitleLineupSelection();
              } else {
                setTitleLineupSelection(normalizedTeam, 'lineup', index, field);
              }
            } else {
              setTitleLineupSelection(normalizedTeam, 'lineup', index, field);
            }
          } else if (selection.team === normalizedTeam && selection.type === 'bench') {
            if (moveBenchPlayerToLineup(normalizedTeam, selection.index, index)) {
              clearTitleLineupSelection();
            } else {
              setTitleLineupSelection(normalizedTeam, 'lineup', index, field);
            }
          } else {
            setTitleLineupSelection(normalizedTeam, 'lineup', index, field);
          }
        } else if (role === 'bench') {
          const index = Number.parseInt(interactiveTarget.dataset.index || '', 10);
          if (!Number.isInteger(index)) {
            clearTitleLineupSelection();
            renderTitle(stateCache.data?.title || {});
            return;
          }

          if (selection.team === normalizedTeam && selection.type === 'bench' && selection.index === index) {
            clearTitleLineupSelection();
          } else if (selection.team === normalizedTeam && selection.type === 'lineup') {
            if (moveBenchPlayerToLineup(normalizedTeam, index, selection.index)) {
              clearTitleLineupSelection();
            } else {
              setTitleLineupSelection(normalizedTeam, 'bench', index);
            }
          } else {
            setTitleLineupSelection(normalizedTeam, 'bench', index);
          }
        }

        renderTitle(stateCache.data?.title || {});
      }
    });
  }

  if (elements.openTeamBuilder) {
    elements.openTeamBuilder.addEventListener('click', async () => {
      setUIView('team-builder');
      ensureTeamBuilderState();
      const players = stateCache.teamBuilder.players || {};
      if (!players.loaded) {
        stateCache.teamBuilder.players.loading = true;
        setTeamBuilderFeedback('選手データを読み込み中です...', 'info');
      } else {
        setTeamBuilderFeedback('編集するチームを選択するか、新規作成してください。', 'info');
      }
      refreshView();
      try {
        await ensureTeamBuilderPlayersLoaded(actions);
        setTeamBuilderFeedback('編集するチームを選択するか、新規作成してください。', 'info');
      } catch (error) {
        const message =
          error instanceof Error ? error.message : '選手データの読み込みに失敗しました。';
        setTeamBuilderFeedback(message, 'danger');
      } finally {
        refreshView();
      }
    });
  }

  if (elements.openMatchButton) {
    elements.openMatchButton.addEventListener('click', () => {
      setUIView('team-select');
      refreshView();
    });
  }

  if (elements.openSimulationButton) {
    elements.openSimulationButton.addEventListener('click', () => {
      setUIView('simulation');
      refreshView();
    });
  }

  // --- Home: Team delete modal
  if (elements.openTeamDelete) {
    elements.openTeamDelete.addEventListener('click', () => {
      populateTeamDeleteOptions();
      setTeamDeleteFeedback('削除するチームを選択してください。', 'info');
      openModal('team-delete');
    });
  }
  if (elements.teamDeleteConfirm) {
    elements.teamDeleteConfirm.addEventListener('click', async () => {
      const select = elements.teamDeleteSelect;
      const teamId = select?.value || '';
      if (!teamId) {
        setTeamDeleteFeedback('削除するチームを選択してください。', 'danger');
        return;
      }
      elements.teamDeleteConfirm.disabled = true;
      try {
        await actions.handleTeamDelete(teamId);
        setTeamDeleteFeedback('チームを削除しました。', 'success');
        populateTeamDeleteOptions();
        closeModal('team-delete');
        // Stay on lobby and refresh view to reflect team list change
        setUIView('lobby');
        refreshView();
      } catch (error) {
        const message = error instanceof Error ? error.message : 'チームの削除に失敗しました。';
        setTeamDeleteFeedback(message, 'danger');
      } finally {
        elements.teamDeleteConfirm.disabled = false;
      }
    });
  }
  if (elements.teamDeleteHome) {
    elements.teamDeleteHome.addEventListener('click', () => {
      closeModal('team-delete');
      setUIView('lobby');
      refreshView();
    });
  }

  // --- Home: Player delete modal
  if (elements.openPlayerDelete) {
    elements.openPlayerDelete.addEventListener('click', async () => {
      if (elements.playerDeleteRole) {
        elements.playerDeleteRole.value = 'batter';
      }
      setPlayerDeleteFeedback('種別と選手を選択してください。', 'info');
      await populatePlayerDeleteOptions('batter');
      openModal('player-delete');
    });
  }
  if (elements.playerDeleteRole) {
    elements.playerDeleteRole.addEventListener('change', async (ev) => {
      const role = ev?.target?.value === 'pitcher' ? 'pitcher' : 'batter';
      await populatePlayerDeleteOptions(role);
      setPlayerDeleteFeedback('削除する選手を選択してください。', 'info');
    });
  }
  if (elements.playerDeleteConfirm) {
    elements.playerDeleteConfirm.addEventListener('click', async () => {
      const role = elements.playerDeleteRole?.value === 'pitcher' ? 'pitcher' : 'batter';
      const select = elements.playerDeleteSelect;
      const playerId = select?.value || '';
      if (!playerId) {
        setPlayerDeleteFeedback('削除する選手を選択してください。', 'danger');
        return;
      }
      elements.playerDeleteConfirm.disabled = true;
      try {
        await actions.handlePlayerDelete(playerId, role, null);
        setPlayerDeleteFeedback('選手を削除しました。', 'success');
        await populatePlayerDeleteOptions(role);
        closeModal('player-delete');
        setUIView('lobby');
        refreshView();
      } catch (error) {
        const message = error instanceof Error ? error.message : '選手の削除に失敗しました。';
        setPlayerDeleteFeedback(message, 'danger');
      } finally {
        elements.playerDeleteConfirm.disabled = false;
      }
    });
  }
  if (elements.playerDeleteHome) {
    elements.playerDeleteHome.addEventListener('click', () => {
      closeModal('player-delete');
      setUIView('lobby');
      refreshView();
    });
  }

  // シミュレーション結果タブ切り替え
  if (elements.simulationTabSummary) {
    elements.simulationTabSummary.addEventListener('click', () => {
      setSimulationResultsView('summary');
      refreshView();
    });
  }
  if (elements.simulationTabGames) {
    elements.simulationTabGames.addEventListener('click', () => {
      setSimulationResultsView('games');
      refreshView();
    });
  }
  if (elements.simulationTabPlayers) {
    elements.simulationTabPlayers.addEventListener('click', () => {
      setSimulationResultsView('players');
      setPlayersTypeView('batting');
      refreshView();
    });
  }

  if (elements.simulationGameCountInput) {
    elements.simulationGameCountInput.addEventListener('input', () => {
      elements.simulationGameCountInput.dataset.userModified = 'true';
    });
  }

  if (elements.simulationSetupCancel) {
    elements.simulationSetupCancel.addEventListener('click', () => {
      setUIView('lobby');
      refreshView();
    });
  }

  if (elements.simulationResultsHome) {
    elements.simulationResultsHome.addEventListener('click', () => {
      setUIView('lobby');
      refreshView();
    });
  }

  if (elements.simulationRunAgain) {
    elements.simulationRunAgain.addEventListener('click', () => {
      setUIView('simulation');
      refreshView();
    });
  }

  if (elements.simulationSetupForm) {
    elements.simulationSetupForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (!actions.startSimulation) return;

      const awayId = elements.simulationSetupAway?.value ?? '';
      const homeId = elements.simulationSetupHome?.value ?? '';
      const gamesValue = elements.simulationGameCountInput?.value ?? '';

      const button = elements.simulationStartButton;
      let originalText = '';
      if (button) {
        originalText = button.textContent || '';
        button.disabled = true;
        button.textContent = '実行中…';
      }

      try {
        await actions.startSimulation(homeId, awayId, gamesValue);
        // 実行直後は要約ビューをデフォルト表示
        setSimulationResultsView('summary');
        setPlayersTeamView('away');
        setPlayersTypeView('batting');
        if (elements.simulationGameCountInput) {
          elements.simulationGameCountInput.dataset.userModified = '';
        }
      } finally {
        if (button) {
          button.disabled = false;
          button.textContent = originalText || 'シミュレーション開始';
        }
      }
    });
  }

  // 個人成績用サブタブ（Away/Home）
  if (elements.simulationPlayersTabAway) {
    elements.simulationPlayersTabAway.addEventListener('click', () => {
      setPlayersTeamView('away');
      refreshView();
    });
  }
  if (elements.simulationPlayersTabHome) {
    elements.simulationPlayersTabHome.addEventListener('click', () => {
      setPlayersTeamView('home');
      refreshView();
    });
  }

  // 個人成績 種別切り替え（打者/投手）
  if (elements.simulationPlayersTypeBatting) {
    elements.simulationPlayersTypeBatting.addEventListener('click', () => {
      setPlayersTypeView('batting');
      refreshView();
    });
  }
  if (elements.simulationPlayersTypePitching) {
    elements.simulationPlayersTypePitching.addEventListener('click', () => {
      setPlayersTypeView('pitching');
      refreshView();
    });
  }

  if (elements.openPlayerBuilder) {
    elements.openPlayerBuilder.addEventListener('click', () => {
      setUIView('player-builder');
      refreshView();
      setPlayerBuilderFeedback('区分と選手を選択するか、新規作成してください。', 'info');
      const role = updatePlayerRoleUI(elements.playerEditorRole?.value || 'batter');
      clearPlayerForm(role);
      if (elements.playerEditorSelect) {
        elements.playerEditorSelect.value = '';
      }
      (async () => {
        await populatePlayerSelect(role);
      })();
    });
  }

  if (elements.backToLobby) {
    elements.backToLobby.addEventListener('click', () => {
      setUIView('lobby');
      refreshView();
    });
  }

  if (elements.backToTeamSelect) {
    elements.backToTeamSelect.addEventListener('click', () => {
      setUIView('team-select');
      refreshView();
    });
  }

  if (elements.teamBuilderBack) {
    elements.teamBuilderBack.addEventListener('click', () => {
      if (stateCache.teamBuilder.editorDirty) {
        const confirmed = window.confirm('変更が保存されていません。破棄しますか？');
        if (!confirmed) {
          setTeamBuilderFeedback('変更は破棄されませんでした。', 'info');
          return;
        }
        resetTeamBuilderFormToInitial(false);
      }
      setUIView('lobby');
      refreshView();
    });
  }

  if (elements.playerBuilderBack) {
    elements.playerBuilderBack.addEventListener('click', () => {
      setUIView('lobby');
      refreshView();
    });
  }

  function handleTeamBuilderRosterClick(event) {
    const button = event.target.closest('[data-builder-action]');
    if (!button || !event.currentTarget.contains(button)) {
      return;
    }
    event.preventDefault();
    const action = button.dataset.builderAction;
    const indexValue = Number.parseInt(button.dataset.index || '', 10);
    const index = Number.isNaN(indexValue) ? 0 : indexValue;
    if (action === 'select-position') {
      toggleLineupPositionSelection(index);
      return;
    }
    if (action === 'clear-lineup') {
      clearLineupSlot(index);
      return;
    }
    if (action === 'remove-bench-slot') {
      removeBenchSlot(index);
      return;
    }
    if (action === 'remove-pitcher-slot') {
      removePitcherSlot(index);
      return;
    }
    if (action === 'swap-slot') {
      const group = button.dataset.group || 'lineup';
      togglePlayerSwap(group, index);
      return;
    }
    if (action === 'select-slot') {
      const group = button.dataset.group || 'lineup';
      if (stateCache.teamBuilder.playerSwap?.source) {
        const handled = attemptPlayerSwapWithSlot(group, index);
        if (handled) {
          return;
        }
      }
      selectTeamBuilderSlot(group, index);
    }
  }

  if (elements.teamBuilderName) {
    elements.teamBuilderName.addEventListener('input', (event) => {
      const form = getTeamBuilderForm();
      if (!form) return;
      const input = event.target;
      const value = typeof input?.value === 'string' ? input.value : '';
      form.name = value;
      stateCache.teamBuilder.editorDirty = true;
    });
  }

  if (elements.teamBuilderAddBench) {
    elements.teamBuilderAddBench.addEventListener('click', () => {
      addBenchSlot();
    });
  }

  if (elements.teamBuilderAddPitcher) {
    elements.teamBuilderAddPitcher.addEventListener('click', () => {
      addPitcherSlot();
    });
  }

  if (elements.teamBuilderReset) {
    elements.teamBuilderReset.addEventListener('click', () => {
      resetTeamBuilderFormToInitial();
    });
  }

  if (elements.teamBuilderLineup) {
    elements.teamBuilderLineup.addEventListener('click', handleTeamBuilderRosterClick);
  }

  if (elements.teamBuilderBench) {
    elements.teamBuilderBench.addEventListener('click', handleTeamBuilderRosterClick);
  }

  if (elements.teamBuilderPitchers) {
    elements.teamBuilderPitchers.addEventListener('click', handleTeamBuilderRosterClick);
  }

  if (elements.teamBuilderPlayerPanel) {
    elements.teamBuilderPlayerPanel.addEventListener('click', (event) => {
      const card = event.target.closest('[data-builder-player-id]');
      if (!card || !elements.teamBuilderPlayerPanel.contains(card)) {
        return;
      }
      if (card.disabled) {
        return;
      }
    event.preventDefault();
    const playerId = card.dataset.builderPlayerId || '';
    const record = getPlayerRecordById(playerId);
    if (!record) {
      setTeamBuilderFeedback('選手データが見つかりません。', 'danger');
      return;
    }
    if (stateCache.teamBuilder.playerSwap?.source) {
      const handled = attemptPlayerSwapWithRecord(record);
      if (handled) {
        return;
      }
    }
    assignPlayerToSelection(record);
  });
}

  elements.teamBuilderCatalogButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const catalog = button.dataset.builderCatalog === 'pitchers' ? 'pitchers' : 'batters';
      focusRosterForCatalog(catalog);
    });
  });

  if (elements.teamBuilderSearch) {
    elements.teamBuilderSearch.addEventListener('input', (event) => {
      const input = event.target;
      const value = typeof input?.value === 'string' ? input.value : '';
      stateCache.teamBuilder.searchTerm = sanitizeSearchTerm(value);
      renderTeamBuilderCatalog();
    });
  }

  if (elements.teamBuilderNew) {
    elements.teamBuilderNew.addEventListener('click', () => {
      if (elements.teamEditorSelect) {
        elements.teamEditorSelect.value = '__new__';
      }
      createNewTeamTemplate();
    });
  }

  if (elements.playerBuilderNew) {
    elements.playerBuilderNew.addEventListener('click', () => {
      const role = elements.playerEditorRole?.value || 'batter';
      loadPlayerTemplate(role);
    });
  }

  if (elements.teamEditorSelect) {
    elements.teamEditorSelect.addEventListener('change', async (event) => {
      const selectValue = event.target.value;
      if (selectValue === '__new__') {
        createNewTeamTemplate();
        return;
      }
      if (!selectValue) {
        stateCache.teamBuilder.currentTeamId = null;
        stateCache.teamBuilder.lastSavedId = null;
        stateCache.teamBuilder.editorDirty = false;
        stateCache.teamBuilder.form = createDefaultTeamForm();
        setSelectionAndCatalog('lineup', 0);
        resetLineupPositionSelection();
        clearPlayerSwapSelection();
        captureTeamBuilderSnapshot(stateCache.teamBuilder.form);
        renderTeamBuilderView();
        setTeamBuilderFeedback('編集するチームを選択してください。', 'info');
        return;
      }
      setTeamBuilderFeedback('チームデータを読み込み中...', 'info');
      try {
        await ensureTeamBuilderPlayersLoaded(actions);
        const teamData = await actions.fetchTeamDefinition(selectValue);
        if (teamData) {
          applyTeamDataToForm(teamData);
          stateCache.teamBuilder.currentTeamId = selectValue;
          stateCache.teamBuilder.lastSavedId = selectValue;
          setTeamBuilderFeedback('チームデータを読み込みました。', 'info');
        } else {
          stateCache.teamBuilder.currentTeamId = null;
          stateCache.teamBuilder.lastSavedId = null;
          setTeamBuilderFeedback('チームデータの読み込みに失敗しました。', 'danger');
        }
      } catch (error) {
        const message =
          error instanceof Error ? error.message : 'チームデータの読み込みに失敗しました。';
        stateCache.teamBuilder.currentTeamId = null;
        stateCache.teamBuilder.lastSavedId = null;
        setTeamBuilderFeedback(message, 'danger');
      }
    });
  }

  if (elements.playerRoleButtons?.length) {
    elements.playerRoleButtons.forEach((button) => {
      button.addEventListener('click', () => {
        const value = String(button.dataset.roleChoice || '').toLowerCase();
        const normalized = value === 'pitcher' ? 'pitcher' : 'batter';
        if (elements.playerEditorRole?.value === normalized) {
          return;
        }
        updatePlayerRoleUI(normalized);
        clearPlayerForm(normalized);
        if (elements.playerEditorSelect) {
          elements.playerEditorSelect.value = '';
        }
        setPlayerBuilderFeedback('区分が変更されました。選手を選ぶかテンプレートを作成してください。', 'info');
        (async () => {
          await populatePlayerSelect(normalized);
        })();
      });
    });
  }

  if (elements.playerPositionButtons?.length) {
    elements.playerPositionButtons.forEach((button) => {
      button.addEventListener('click', () => {
        const role = elements.playerEditorRole?.value || 'batter';
        if (role !== 'batter' || button.disabled) {
          return;
        }
        const isActive = button.classList.contains('active');
        updateChipToggleState(button, !isActive);
      });
    });
  }

  if (elements.playerPitcherTypeButtons?.length) {
    elements.playerPitcherTypeButtons.forEach((button) => {
      button.addEventListener('click', () => {
        const role = elements.playerEditorRole?.value || 'batter';
        if (role !== 'pitcher' || button.disabled) {
          return;
        }
        elements.playerPitcherTypeButtons.forEach((btn) => {
          updateChipToggleState(btn, btn === button);
        });
      });
    });
  }

  if (elements.playerEditorSelect) {
    elements.playerEditorSelect.addEventListener('change', async (event) => {
      const selectValue = event.target.value;
      if (selectValue === '__new__') {
        const role = elements.playerEditorRole?.value || 'batter';
        loadPlayerTemplate(role);
        return;
      }
      if (!selectValue) {
        const role = elements.playerEditorRole?.value || 'batter';
        clearPlayerForm(role);
        setPlayerBuilderFeedback('編集する選手を選択してください。', 'info');
        return;
      }
      setPlayerBuilderFeedback('選手データを読み込み中...', 'info');
      const result = await actions.fetchPlayerDefinition(selectValue); // pass id
      const fetchedPlayer = result?.player || null;
      const fetchedRole = result?.role === 'pitcher' ? 'pitcher' : result?.role === 'batter' ? 'batter' : null;
      const currentRole = elements.playerEditorRole?.value || 'batter';
      const roleToUse = fetchedRole || currentRole;
      if (fetchedRole && fetchedRole !== currentRole) {
        updatePlayerRoleUI(fetchedRole);
        await populatePlayerSelect(fetchedRole, selectValue);
      }
      if (fetchedPlayer) {
        applyPlayerFormData(fetchedPlayer, roleToUse);
        // Disable delete if referenced by any team
        const hasRefs = Boolean(result?.hasReferences);
        if (elements.playerBuilderDelete) {
          elements.playerBuilderDelete.disabled = hasRefs;
        }
        if (hasRefs) {
          const refs = Array.isArray(result?.referencedBy) ? result.referencedBy : [];
          const list = refs.slice(0, 5).join(', ') + (refs.length > 5 ? ` 他${refs.length - 5}件` : '');
          setPlayerBuilderFeedback(`この選手は以下のチームに含まれているため削除できません: ${list}`, 'warning');
        } else {
          setPlayerBuilderFeedback('選手データを読み込みました。', 'info');
        }
      } else {
        clearPlayerForm(roleToUse);
        setPlayerBuilderFeedback('選手データの読み込みに失敗しました。', 'danger');
      }
    });
  }

  if (elements.playerBuilderSave) {
    elements.playerBuilderSave.addEventListener('click', async () => {
      const role = elements.playerEditorRole?.value || 'batter';
      const formData = getPlayerFormData(role);
      if (!formData) {
        return;
      }
      // If editing an existing player, capture the selected id and visible name
      const selectEl = elements.playerEditorSelect;
      const selectedValue = selectEl?.value || '';
      const selectedText = selectEl?.selectedOptions?.[0]?.textContent || null;
      const isEditing = selectedValue && selectedValue !== '__new__';
      const playerId = isEditing ? selectedValue : null;
      const originalName = isEditing ? selectedText : null;
      elements.playerBuilderSave.disabled = true;
      try {
        const saved = await actions.handlePlayerSave(formData, role, originalName, playerId);
        if (saved?.id) {
          await populatePlayerSelect(role, saved.id);
          await ensureTeamBuilderPlayersLoaded(actions, true);
          renderTeamBuilderView();
          setPlayerBuilderFeedback('選手を保存しました。', 'success');
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : '選手の保存に失敗しました。';
        setPlayerBuilderFeedback(message, 'danger');
      } finally {
        elements.playerBuilderSave.disabled = false;
      }
    });
  }

  if (elements.playerBuilderDelete) {
    elements.playerBuilderDelete.addEventListener('click', async () => {
      const role = elements.playerEditorRole?.value || 'batter';
      const selectEl = elements.playerEditorSelect;
      const idValue = selectEl?.value || '';
      const nameText = selectEl?.selectedOptions?.[0]?.textContent || '';
      if (!idValue || idValue === '__new__') {
        setPlayerBuilderFeedback('削除する選手を選択してください。', 'danger');
        return;
      }
      if (elements.playerBuilderDelete.disabled) {
        // Should not happen due to disabled state, but double-guard.
        setPlayerBuilderFeedback('この選手はチームで使用中のため削除できません。', 'warning');
        return;
      }
      const confirmed = window.confirm(`選手 '${nameText}' を削除します。よろしいですか？`);
      if (!confirmed) return;
      elements.playerBuilderDelete.disabled = true;
      try {
        await actions.handlePlayerDelete(idValue, role, nameText);
        await populatePlayerSelect(role);
        clearPlayerForm(role);
        setPlayerBuilderFeedback('選手を削除しました。', 'success');
      } catch (error) {
        const message = error instanceof Error ? error.message : '選手の削除に失敗しました。';
        setPlayerBuilderFeedback(message, 'danger');
      } finally {
        elements.playerBuilderDelete.disabled = false;
      }
    });
  }

  if (elements.teamBuilderSave) {
    elements.teamBuilderSave.addEventListener('click', async () => {
      // Warn if lineup has positional eligibility conflicts
      const conflicts = findLineupEligibilityConflicts();
      if (conflicts.length) {
        const list = conflicts.slice(0, 8).map((s) => `- ${s}`).join('\n');
        const more = conflicts.length > 8 ? `\n他${conflicts.length - 8}件` : '';
        const ok = window.confirm(
          `守備適性に不一致があります。このまま保存しますか？\n\n${list}${more}`,
        );
        if (!ok) {
          return;
        }
      }

      const teamPayload = buildTeamPayload();
      if (!teamPayload) {
        return;
      }

      elements.teamBuilderSave.disabled = true;
      try {
        const savedId = await actions.handleTeamSave(
          stateCache.teamBuilder.currentTeamId,
          teamPayload,
        );
        if (savedId) {
          stateCache.teamBuilder.currentTeamId = savedId;
          stateCache.teamBuilder.lastSavedId = savedId;
          stateCache.teamBuilder.editorDirty = false;
          const form = getTeamBuilderForm();
          if (form) {
            form.name = teamPayload.name;
            captureTeamBuilderSnapshot(form);
          }
          if (elements.teamEditorSelect) {
            elements.teamEditorSelect.value = savedId;
          }
          setTeamBuilderFeedback('チームを保存しました。', 'success');
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'チームの保存に失敗しました。';
        setTeamBuilderFeedback(message, 'danger');
      } finally {
        elements.teamBuilderSave.disabled = false;
        renderTeamBuilderView();
      }
    });
  }

  if (elements.teamBuilderDelete) {
    elements.teamBuilderDelete.addEventListener('click', async () => {
      const teamId = stateCache.teamBuilder.currentTeamId;
      if (!teamId) {
        setTeamBuilderFeedback('削除するチームを選択してください。', 'danger');
        return;
      }
      const confirmed = window.confirm(`チーム '${teamId}' を削除します。よろしいですか？`);
      if (!confirmed) return;
      elements.teamBuilderDelete.disabled = true;
      try {
        await actions.handleTeamDelete(teamId);
        stateCache.teamBuilder.currentTeamId = null;
        stateCache.teamBuilder.lastSavedId = null;
        stateCache.teamBuilder.editorDirty = false;
        stateCache.teamBuilder.form = createDefaultTeamForm();
        setSelectionAndCatalog('lineup', 0);
        resetLineupPositionSelection();
        clearPlayerSwapSelection();
        captureTeamBuilderSnapshot(stateCache.teamBuilder.form);
        if (elements.teamEditorSelect) {
          elements.teamEditorSelect.value = '';
        }
        renderTeamBuilderView();
        setTeamBuilderFeedback('チームを削除しました。', 'success');
        setUIView('team-builder');
        if (stateCache.data) {
          render(stateCache.data);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'チームの削除に失敗しました。';
        setTeamBuilderFeedback(message, 'danger');
      } finally {
        elements.teamBuilderDelete.disabled = false;
      }
    });
  }

  elements.modalCloseButtons.forEach((button) => {
    const target = button.dataset.close;
    button.addEventListener('click', () => closeModal(target || button.closest('.modal')));
  });

  ['offense', 'pinch-run', 'defense', 'pitcher', 'stats', 'abilities'].forEach((name) => {
    const modal = resolveModal(name);
    if (modal) {
      modal.addEventListener('click', (event) => {
        if (event.target === modal) {
          closeModal(modal);
        }
      });
    }
  });

  if (elements.pinchRunBase) {
    elements.pinchRunBase.addEventListener('change', (event) => {
      const rawValue = event.target.value;
      const parsed = rawValue === '' ? null : Number(rawValue);
      if (rawValue !== '' && Number.isInteger(parsed)) {
        setPinchRunSelectedBase(parsed);
      } else {
        setPinchRunSelectedBase(null);
      }
    });
  }

  if (elements.pinchRunModal) {
    elements.pinchRunModal.addEventListener('show', () => {
      const selected = getPinchRunSelectedBase();
      if (Number.isInteger(selected) && elements.pinchRunBase) {
        elements.pinchRunBase.value = String(selected);
        const changeEvent = new Event('change', { bubbles: true });
        elements.pinchRunBase.dispatchEvent(changeEvent);
      }
    });
  }

  if (elements.offenseMenu) {
    document.addEventListener('click', (event) => {
      if (elements.offenseMenu.classList.contains('hidden')) {
        return;
      }
      const clickedInsideMenu = elements.offenseMenu.contains(event.target);
      const clickedToggle =
        elements.openOffenseButton && elements.openOffenseButton.contains(event.target);
      if (!clickedInsideMenu && !clickedToggle) {
        hideOffenseMenu();
      }
    });
  }

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
    elements.defenseField.addEventListener('click', handleDefensePlayerClick);
  }
  if (elements.defenseBench) {
    elements.defenseBench.addEventListener('click', handleDefensePlayerClick);
  }
  if (elements.defenseExtras) {
    elements.defenseExtras.addEventListener('click', handleDefensePlayerClick);
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

  elements.abilitiesTeamButtons.forEach((button) => {
    button.addEventListener('click', () => {
      if (button.disabled) return;
      const teamKey = button.dataset.abilitiesTeam;
      if (!teamKey) return;
      stateCache.abilitiesView.team = teamKey;
      updateAbilitiesPanel(stateCache.data);
    });
  });

  elements.abilitiesTypeButtons.forEach((button) => {
    button.addEventListener('click', () => {
      if (button.disabled) return;
      const viewType = button.dataset.abilitiesType;
      if (!viewType) return;
      stateCache.abilitiesView.type = viewType;
      updateAbilitiesPanel(stateCache.data);
    });
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      hideOffenseMenu();
      hideDefenseMenu();
      ['offense', 'defense', 'pitcher', 'stats', 'abilities'].forEach((name) => {
        const modal = resolveModal(name);
        if (modal && !modal.classList.contains('hidden')) {
          closeModal(modal);
        }
      });
    }
    if (event.key === 'Tab' && !event.ctrlKey && !event.altKey && !event.shiftKey) {
      event.preventDefault();
      toggleLogPanel();
    }
  });

  updateDefenseSelectionInfo();
}
