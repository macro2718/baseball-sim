import { elements } from '../dom.js';
import { stateCache, setUIView } from '../state.js';
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
  updateScreenVisibility,
} from '../ui/renderers.js';
import { showStatus } from '../ui/status.js';
import { handleDefensePlayerClick, updateDefenseSelectionInfo } from '../ui/defensePanel.js';
import { escapeHtml, renderPositionList, renderPositionToken } from '../utils.js';

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
    speed: 4.3,
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

function createAbilityChip(label, value) {
  const safeLabel = escapeHtml(label);
  const safeValue = escapeHtml(value);
  return `<span class="ability-chip"><span class="label">${safeLabel}</span><span class="value">${safeValue}</span></span>`;
}

function getBatterAbilityChips(player) {
  if (!player) return [];
  const chips = [];
  if (player.bats) {
    chips.push(createAbilityChip('打席', player.bats));
  }
  const stats = player.stats || {};
  chips.push(createAbilityChip('K%', formatPercent(stats.k_pct)));
  chips.push(createAbilityChip('BB%', formatPercent(stats.bb_pct)));
  chips.push(createAbilityChip('Hard%', formatPercent(stats.hard_pct)));
  chips.push(createAbilityChip('GB%', formatPercent(stats.gb_pct)));
  chips.push(createAbilityChip('Speed', formatNumber(stats.speed, 2)));
  chips.push(createAbilityChip('Field', formatNumber(stats.fielding_skill, 0)));
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
  chips.push(createAbilityChip('K%', formatPercent(stats.k_pct)));
  chips.push(createAbilityChip('BB%', formatPercent(stats.bb_pct)));
  chips.push(createAbilityChip('Hard%', formatPercent(stats.hard_pct)));
  chips.push(createAbilityChip('GB%', formatPercent(stats.gb_pct)));
  chips.push(createAbilityChip('Sta', formatNumber(stats.stamina, 0)));
  return chips;
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
  if (!stateCache.teamBuilder.selection) {
    stateCache.teamBuilder.selection = { group: 'lineup', index: 0 };
  }
  if (!stateCache.teamBuilder.catalog) {
    stateCache.teamBuilder.catalog = 'batters';
  }
  if (stateCache.teamBuilder.searchTerm == null) {
    stateCache.teamBuilder.searchTerm = '';
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
    slot.eligible = eligible;
    const desired = positionOverride ? String(positionOverride).toUpperCase() : String(slot.position || '').toUpperCase();
    if (desired && eligible.includes(desired)) {
      slot.position = desired;
    } else if (eligible.length) {
      slot.position = eligible[0];
    } else {
      slot.position = desired || 'DH';
    }
  } else {
    slot.eligible = ['P'];
    slot.position = 'P';
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

function relinkFormPlayers() {
  ensureTeamBuilderState();
  const form = stateCache.teamBuilder.form;
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

function selectTeamBuilderSlot(group, index) {
  ensureTeamBuilderState();
  const form = stateCache.teamBuilder.form;
  let normalizedGroup = group;
  let normalizedIndex = index;
  if (group === 'bench') {
    if (!Array.isArray(form.bench) || !form.bench.length) {
      normalizedGroup = 'lineup';
      normalizedIndex = 0;
    }
  } else if (group === 'pitchers') {
    if (!Array.isArray(form.pitchers) || !form.pitchers.length) {
      normalizedGroup = 'lineup';
      normalizedIndex = 0;
    }
  }
  if (normalizedGroup === 'lineup') {
    if (!Number.isInteger(normalizedIndex) || normalizedIndex < 0 || normalizedIndex >= form.lineup.length) {
      normalizedIndex = 0;
    }
  } else if (normalizedGroup === 'bench') {
    if (!Number.isInteger(normalizedIndex) || normalizedIndex < 0 || normalizedIndex >= form.bench.length) {
      normalizedIndex = 0;
    }
  } else if (normalizedGroup === 'pitchers') {
    if (!Number.isInteger(normalizedIndex) || normalizedIndex < 0 || normalizedIndex >= form.pitchers.length) {
      normalizedIndex = 0;
    }
  }
  stateCache.teamBuilder.selection = { group: normalizedGroup, index: normalizedIndex };
  renderTeamBuilderView();
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

function cycleLineupPosition(index) {
  const form = getTeamBuilderForm();
  const slot = form?.lineup?.[index];
  if (!slot) return;
  const options = getAvailablePositionsForSlot(slot);
  if (!options.length) return;
  const current = String(slot.position || options[0]).toUpperCase();
  const currentIndex = options.indexOf(current);
  const next = options[(currentIndex + 1) % options.length];
  slot.position = next;
  stateCache.teamBuilder.editorDirty = true;
  renderTeamBuilderView();
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
  stateCache.teamBuilder.editorDirty = true;
  renderTeamBuilderView();
}

function clearBenchEntry(index) {
  const form = getTeamBuilderForm();
  if (!form || !form.bench || index < 0 || index >= form.bench.length) return;
  form.bench[index] = createEmptyBenchEntry();
}

function clearPitcherEntry(index) {
  const form = getTeamBuilderForm();
  if (!form || !form.pitchers || index < 0 || index >= form.pitchers.length) return;
  form.pitchers[index] = createEmptyPitcherEntry();
}

function removeBenchSlot(index) {
  const form = getTeamBuilderForm();
  if (!form || !Array.isArray(form.bench) || index < 0 || index >= form.bench.length) return;
  form.bench.splice(index, 1);
  stateCache.teamBuilder.editorDirty = true;
  if (stateCache.teamBuilder.selection?.group === 'bench') {
    if (!form.bench.length) {
      stateCache.teamBuilder.selection = { group: 'lineup', index: 0 };
    } else if (stateCache.teamBuilder.selection.index >= form.bench.length) {
      stateCache.teamBuilder.selection.index = form.bench.length - 1;
    }
  }
  renderTeamBuilderView();
}

function removePitcherSlot(index) {
  const form = getTeamBuilderForm();
  if (!form || !Array.isArray(form.pitchers) || index < 0 || index >= form.pitchers.length) return;
  form.pitchers.splice(index, 1);
  stateCache.teamBuilder.editorDirty = true;
  if (stateCache.teamBuilder.selection?.group === 'pitchers') {
    if (!form.pitchers.length) {
      stateCache.teamBuilder.selection = { group: 'lineup', index: 0 };
    } else if (stateCache.teamBuilder.selection.index >= form.pitchers.length) {
      stateCache.teamBuilder.selection.index = form.pitchers.length - 1;
    }
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
  if (target === 'lineup') {
    const form = getTeamBuilderForm();
    const slot = form?.lineup?.[selection.index];
    const position = slot?.position ? String(slot.position).toUpperCase() : null;
    if (position === 'P') {
      if (record.role === 'pitcher') return true;
      return Array.isArray(record.eligible) && record.eligible.includes('P');
    }
    return record.role === 'batter';
  }
  return true;
}

function assignPlayerToLineup(index, record) {
  const form = getTeamBuilderForm();
  const slot = form?.lineup?.[index];
  if (!slot || !record) return;
  if (record.role === 'pitcher' && slot.position !== 'P') {
    slot.position = 'P';
  }
  const existing = findPlayerAssignment(record.id);
  if (existing && (existing.group !== 'lineup' || existing.index !== index)) {
    clearAssignmentAt(existing.group, existing.index);
  }
  applyRecordToLineupSlot(slot, record, slot.position);
  stateCache.teamBuilder.selection = { group: 'lineup', index };
  stateCache.teamBuilder.editorDirty = true;
  setTeamBuilderFeedback(`${record.name} を${slot.position || 'DH'}に割り当てました。`, 'success');
  renderTeamBuilderView();
}

function assignPlayerToBench(index, record) {
  const form = getTeamBuilderForm();
  if (!form || index < 0) return;
  if (!form.bench[index]) {
    form.bench[index] = createEmptyBenchEntry();
  }
  const existing = findPlayerAssignment(record.id);
  if (existing && (existing.group !== 'bench' || existing.index !== index)) {
    clearAssignmentAt(existing.group, existing.index);
  }
  const entry = form.bench[index];
  entry.playerId = record.id;
  entry.playerName = record.name;
  entry.player = record;
  entry.playerRole = record.role;
  entry.eligible = record.role === 'pitcher' ? ['P'] : [...(record.eligible || [])];
  stateCache.teamBuilder.selection = { group: 'bench', index };
  stateCache.teamBuilder.editorDirty = true;
  setTeamBuilderFeedback(`${record.name} をベンチ枠に追加しました。`, 'success');
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
  stateCache.teamBuilder.selection = { group: 'pitchers', index };
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

  form.lineup.forEach((slot, index) => {
    const row = document.createElement('div');
    row.className = 'defense-lineup-row';
    if (selection.group === 'lineup' && selection.index === index) {
      row.classList.add('selected');
    }

    const order = document.createElement('span');
    order.className = 'lineup-order';
    order.textContent = `${index + 1}.`;

    const positionButton = document.createElement('button');
    positionButton.type = 'button';
    positionButton.className = 'defense-action-button lineup-position-button';
    positionButton.dataset.builderAction = 'cycle-position';
    positionButton.dataset.group = 'lineup';
    positionButton.dataset.index = String(index);
    positionButton.innerHTML = renderPositionToken(slot.position || '-', slot.player?.pitcher_type, 'position-token');

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
    const chips = slot.playerRole === 'pitcher' ? getPitcherAbilityChips(slot.player) : getBatterAbilityChips(slot.player);
    abilityRow.innerHTML = chips.join('');
    if (chips.length) {
      meta.appendChild(abilityRow);
    }

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
    const chips = entry?.playerRole === 'pitcher' ? getPitcherAbilityChips(entry.player) : getBatterAbilityChips(entry.player);
    abilityRow.innerHTML = chips.join('');
    if (chips.length) {
      meta.appendChild(abilityRow);
    }

    button.appendChild(meta);
    slotDiv.appendChild(button);

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
  const list = Array.isArray(players[catalog]) ? players[catalog] : [];
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
  filtered.forEach((record) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'builder-player-card';
    card.dataset.builderPlayerId = record.id;
    card.dataset.builderRole = record.role;
    const assigned = findPlayerAssignment(record.id);
    if (assigned) {
      card.classList.add('assigned');
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
    } else {
      const roleInfo = document.createElement('div');
      roleInfo.className = 'player-positions';
      roleInfo.textContent = `Type: ${record.pitcher_type || '-'} / Throws: ${record.throws || '-'}`;
      meta.appendChild(roleInfo);
      const abilityRow = document.createElement('div');
      abilityRow.className = 'chip-row';
      abilityRow.innerHTML = getPitcherAbilityChips(record).join('');
      meta.appendChild(abilityRow);
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

function createNewTeamTemplate() {
  ensureTeamBuilderState();
  stateCache.teamBuilder.form = createDefaultTeamForm();
  stateCache.teamBuilder.currentTeamId = null;
  stateCache.teamBuilder.lastSavedId = '__new__';
  stateCache.teamBuilder.editorDirty = false;
  stateCache.teamBuilder.selection = { group: 'lineup', index: 0 };
  setTeamBuilderFeedback('テンプレートを読み込みました。', 'info');
  renderTeamBuilderView();
}

async function ensureTeamBuilderPlayersLoaded(actions) {
  ensureTeamBuilderState();
  const players = stateCache.teamBuilder.players;
  if (players.loaded) {
    return players;
  }
  if (players.loading) {
    return players;
  }
  stateCache.teamBuilder.players.loading = true;
  try {
    const catalog = await actions.fetchPlayersCatalog();
    applyPlayersCatalogData(catalog);
    stateCache.teamBuilder.players.loaded = true;
  } catch (error) {
    stateCache.teamBuilder.players.loaded = false;
    throw error;
  } finally {
    stateCache.teamBuilder.players.loading = false;
  }
  return stateCache.teamBuilder.players;
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
  stateCache.teamBuilder.selection = { group: 'lineup', index: 0 };
  stateCache.teamBuilder.editorDirty = false;
  relinkFormPlayers();
  renderTeamBuilderView();
}

function sanitizeSearchTerm(value) {
  if (value == null) return '';
  return String(value).trim();
}

export function initEventListeners(actions) {
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

  updatePlayerRoleUI(elements.playerEditorRole?.value || 'batter');

  elements.startButton.addEventListener('click', () => actions.handleStart(false));
  elements.reloadTeams.addEventListener('click', actions.handleReloadTeams);
  elements.restartButton.addEventListener('click', () => actions.handleStart(true));
  elements.returnTitle.addEventListener('click', actions.handleReturnToTitle);
  elements.clearLog.addEventListener('click', actions.handleClearLog);
  elements.swingButton.addEventListener('click', actions.handleSwing);
  elements.buntButton.addEventListener('click', actions.handleBunt);

  if (elements.pinchButton) {
    elements.pinchButton.addEventListener('click', actions.handlePinchHit);
  }
  if (elements.openOffenseButton) {
    elements.openOffenseButton.addEventListener('click', toggleOffenseMenu);
  }
  if (elements.offensePinchMenuButton) {
    elements.offensePinchMenuButton.addEventListener('click', () => openModal('offense'));
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
    if (action === 'cycle-position') {
      cycleLineupPosition(index);
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
    if (action === 'select-slot') {
      const group = button.dataset.group || 'lineup';
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
      assignPlayerToSelection(record);
    });
  }

  elements.teamBuilderCatalogButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const catalog = button.dataset.builderCatalog === 'pitchers' ? 'pitchers' : 'batters';
      if (stateCache.teamBuilder.catalog === catalog) {
        return;
      }
      stateCache.teamBuilder.catalog = catalog;
      updateCatalogButtons();
      renderTeamBuilderCatalog();
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
        stateCache.teamBuilder.selection = { group: 'lineup', index: 0 };
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
        stateCache.teamBuilder.selection = { group: 'lineup', index: 0 };
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

  ['offense', 'defense', 'pitcher', 'stats', 'abilities'].forEach((name) => {
    const modal = resolveModal(name);
    if (modal) {
      modal.addEventListener('click', (event) => {
        if (event.target === modal) {
          closeModal(modal);
        }
      });
    }
  });

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
