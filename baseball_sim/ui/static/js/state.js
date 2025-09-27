// Shared mutable state that mirrors the current UI view.

import { FIELD_POSITION_KEYS } from './config.js';

export const stateCache = {
  data: null,
  defensePlan: null,
  defenseSelection: { first: null, feedback: null },
  defenseContext: { lineup: {}, bench: {}, canSub: false },
  currentBatterIndex: null,
  pinchRunContext: { bases: [], availableBaseIndexes: [], selectedBaseIndex: null },
  statsView: { team: 'away', type: 'batting' },
  abilitiesView: { team: 'away', type: 'batting' },
  uiView: 'lobby',
  // Navigation-reset flags
  resetTeamSelect: false,
  resetSimulationSelect: false,
  simulationResultsView: 'summary',
  playersSelectedTeamIndex: 0,
  playersTypeView: 'batting',
  matchSetup: { mode: 'manual', userTeam: 'home' },
  gameControl: {
    mode: 'manual',
    userTeam: null,
    cpuTeam: null,
    userTeamName: null,
    cpuTeamName: null,
    userIsOffense: true,
    userIsDefense: true,
    offenseAllowed: true,
    defenseAllowed: true,
    progressAvailable: false,
    instruction: '',
  },
  simulation: { running: false, defaultGames: 20, lastRun: null, log: [], limits: { min: 1, max: 200 } },
  simulationSetup: { leagueTeams: [], gamesPerCard: 3, cardsPerOpponent: 1, seed: null },
  teamLibrary: { teams: [], selection: { home: null, away: null }, ready: false, hint: '' },
  titleLineup: {
    plans: { home: null, away: null },
    selection: { team: null, type: null, index: null, field: null },
  },
  teamBuilder: {
    currentTeamId: null,
    lastSavedId: null,
    editorDirty: false,
    form: null,
    initialForm: null,
    players: { batters: [], pitchers: [], byId: {}, byName: {}, loaded: false, loading: false },
    playersLoadingPromise: null,
    selection: { group: 'lineup', index: 0 },
    positionSwap: { first: null },
    playerSwap: { source: null },
    catalog: 'batters',
    searchTerm: '',
    rotationSize: 5,
    folders: [],
    selectedFolder: '',
  },
  playerEditor: {
    searchTerm: '',
    folder: '',
  },
};

export function setUIView(view) {
  if (!view) return;
  stateCache.uiView = view;
}

export function setSimulationResultsView(view) {
  const allowed = new Set(['summary', 'games', 'players']);
  const next = allowed.has(view) ? view : 'summary';
  stateCache.simulationResultsView = next;
  return next;
}

export function getSimulationResultsView() {
  const v = stateCache.simulationResultsView;
  return v === 'games' || v === 'players' ? v : 'summary';
}

export function setPlayersSelectedTeamIndex(index) {
  const next = Number.isInteger(index) && index >= 0 ? index : 0;
  stateCache.playersSelectedTeamIndex = next;
  return next;
}

export function getPlayersSelectedTeamIndex() {
  const v = stateCache.playersSelectedTeamIndex;
  return Number.isInteger(v) && v >= 0 ? v : 0;
}

export function setPlayersTypeView(type) {
  const next = type === 'pitching' ? 'pitching' : 'batting';
  stateCache.playersTypeView = next;
  return next;
}

export function getPlayersTypeView() {
  return stateCache.playersTypeView === 'pitching' ? 'pitching' : 'batting';
}

export function getUIView() {
  return stateCache.uiView;
}

export function setPinchRunContext(bases, availableIndexes, selectedIndex) {
  const baseList = Array.isArray(bases) ? bases : [];
  const indexList = Array.isArray(availableIndexes)
    ? availableIndexes.filter((idx) => Number.isInteger(idx))
    : [];
  const selected = Number.isInteger(selectedIndex) && indexList.includes(selectedIndex)
    ? selectedIndex
    : null;
  stateCache.pinchRunContext = {
    bases: baseList,
    availableBaseIndexes: indexList,
    selectedBaseIndex: selected,
  };
}

export function setPinchRunSelectedBase(index) {
  const normalized = Number.isInteger(index) ? index : null;
  if (!stateCache.pinchRunContext) {
    stateCache.pinchRunContext = { bases: [], availableBaseIndexes: [], selectedBaseIndex: null };
  }
  if (
    normalized !== null &&
    Array.isArray(stateCache.pinchRunContext.availableBaseIndexes) &&
    stateCache.pinchRunContext.availableBaseIndexes.includes(normalized)
  ) {
    stateCache.pinchRunContext.selectedBaseIndex = normalized;
  } else {
    stateCache.pinchRunContext.selectedBaseIndex = null;
  }
}

export function getPinchRunSelectedBase() {
  const value = stateCache.pinchRunContext?.selectedBaseIndex;
  return Number.isInteger(value) ? value : null;
}

export function normalizePositionKey(position) {
  if (!position) return null;
  const upper = String(position).toUpperCase();
  if (upper === 'SP' || upper === 'RP') return 'P';
  return upper;
}

export function getLineupPlayer(index) {
  const { lineup } = stateCache.defenseContext;
  if (!Number.isInteger(index)) return null;
  return Object.prototype.hasOwnProperty.call(lineup, index) ? lineup[index] : null;
}

export function getBenchPlayer(index) {
  const { bench } = stateCache.defenseContext;
  if (!Number.isInteger(index)) return null;
  return Object.prototype.hasOwnProperty.call(bench, index) ? bench[index] : null;
}

export function getLineupPositionKey(lineupPlayer) {
  if (!lineupPlayer) return null;
  return normalizePositionKey(lineupPlayer.position_key || lineupPlayer.position);
}

export function getEligiblePositionsAll(player) {
  if (!player) return [];
  const raw = Array.isArray(player.eligible_all) ? player.eligible_all : player.eligible;
  if (!raw) return [];
  return raw.map((pos) => String(pos).toUpperCase());
}

export function canBenchPlayerCoverPosition(benchPlayer, positionKey) {
  if (!benchPlayer || !positionKey) return false;
  const eligiblePositions = getEligiblePositionsAll(benchPlayer);
  return eligiblePositions.includes(positionKey);
}

export function getDefensePlanInvalidAssignments(plan) {
  if (!plan || !Array.isArray(plan.lineup)) {
    return [];
  }

  const invalid = [];
  plan.lineup.forEach((player, index) => {
    if (!player) return;
    const positionKey = normalizePositionKey(player.position_key || player.position);
    if (!positionKey || positionKey === '-') {
      return;
    }

    if (!canBenchPlayerCoverPosition(player, positionKey)) {
      invalid.push({
        index,
        player,
        positionKey,
        positionLabel: player.position || positionKey,
      });
    }
  });

  return invalid;
}

export function updateDefenseContext(lineupMap, benchMap, canSub) {
  stateCache.defenseContext = {
    lineup: lineupMap,
    bench: benchMap,
    canSub: Boolean(canSub),
  };
}

export function resetDefenseSelection() {
  stateCache.defenseSelection = { first: null, feedback: null };
}

export function isKnownFieldPosition(positionKey) {
  return FIELD_POSITION_KEYS.has(positionKey);
}

function ensureSimulationSetupState() {
  if (!stateCache.simulationSetup) {
    stateCache.simulationSetup = {
      leagueTeams: [],
      gamesPerCard: 3,
      cardsPerOpponent: 1,
      seed: null,
    };
  }
}

function normalizeTeamIdList(teamIds) {
  if (!Array.isArray(teamIds)) {
    return [];
  }
  const normalized = [];
  teamIds.forEach((teamId) => {
    if (typeof teamId !== 'string') return;
    const trimmed = teamId.trim();
    if (!trimmed) return;
    normalized.push(trimmed);
  });
  return normalized;
}

export function getSimulationLeagueTeams() {
  ensureSimulationSetupState();
  return Array.isArray(stateCache.simulationSetup.leagueTeams)
    ? [...stateCache.simulationSetup.leagueTeams]
    : [];
}

export function setSimulationLeagueTeams(teamIds) {
  ensureSimulationSetupState();
  const normalized = normalizeTeamIdList(teamIds);
  stateCache.simulationSetup.leagueTeams = normalized;
  return normalized;
}

export function addSimulationLeagueTeam(teamId) {
  if (typeof teamId !== 'string' || !teamId.trim()) {
    return getSimulationLeagueTeams();
  }
  ensureSimulationSetupState();
  const normalized = teamId.trim();
  const current = Array.isArray(stateCache.simulationSetup.leagueTeams)
    ? [...stateCache.simulationSetup.leagueTeams]
    : [];
  current.push(normalized);
  stateCache.simulationSetup.leagueTeams = normalizeTeamIdList(current);
  return [...stateCache.simulationSetup.leagueTeams];
}

export function removeSimulationLeagueTeamAt(index) {
  ensureSimulationSetupState();
  const current = Array.isArray(stateCache.simulationSetup.leagueTeams)
    ? [...stateCache.simulationSetup.leagueTeams]
    : [];
  if (!Number.isInteger(index) || index < 0 || index >= current.length) {
    return current;
  }
  current.splice(index, 1);
  stateCache.simulationSetup.leagueTeams = normalizeTeamIdList(current);
  return [...stateCache.simulationSetup.leagueTeams];
}

export function clearSimulationLeagueTeams() {
  ensureSimulationSetupState();
  stateCache.simulationSetup.leagueTeams = [];
  return [];
}

export function primeSimulationSetup({ teams, gamesPerCard, cardsPerOpponent }) {
  ensureSimulationSetupState();

  const normalizedTeams = normalizeTeamIdList(Array.isArray(teams) ? teams : []);
  const normalizedGames = Number.isFinite(gamesPerCard) && gamesPerCard > 0 ? gamesPerCard : null;
  const normalizedCards =
    Number.isFinite(cardsPerOpponent) && cardsPerOpponent > 0 ? cardsPerOpponent : null;

  const seedPayload = {
    teams: normalizedTeams,
    gamesPerCard: normalizedGames,
    cardsPerOpponent: normalizedCards,
  };
  const seed = JSON.stringify(seedPayload);
  if (stateCache.simulationSetup.seed === seed) {
    return;
  }

  stateCache.simulationSetup.seed = seed;

  if (normalizedTeams.length) {
    stateCache.simulationSetup.leagueTeams = normalizedTeams;
  } else if (!stateCache.simulationSetup.leagueTeams.length) {
    stateCache.simulationSetup.leagueTeams = [];
  }

  if (normalizedGames) {
    stateCache.simulationSetup.gamesPerCard = normalizedGames;
  }
  if (normalizedCards) {
    stateCache.simulationSetup.cardsPerOpponent = normalizedCards;
  }
}

export function getSimulationScheduleDefaults() {
  ensureSimulationSetupState();
  const { gamesPerCard, cardsPerOpponent } = stateCache.simulationSetup;
  return {
    gamesPerCard: Number.isFinite(gamesPerCard) && gamesPerCard > 0 ? gamesPerCard : 3,
    cardsPerOpponent:
      Number.isFinite(cardsPerOpponent) && cardsPerOpponent > 0 ? cardsPerOpponent : 1,
  };
}

export function setSimulationScheduleDefaults({ gamesPerCard, cardsPerOpponent }) {
  ensureSimulationSetupState();
  if (Number.isFinite(gamesPerCard) && gamesPerCard > 0) {
    stateCache.simulationSetup.gamesPerCard = gamesPerCard;
  }
  if (Number.isFinite(cardsPerOpponent) && cardsPerOpponent > 0) {
    stateCache.simulationSetup.cardsPerOpponent = cardsPerOpponent;
  }
  return getSimulationScheduleDefaults();
}

export function updateSimulationScheduleField(field, value) {
  ensureSimulationSetupState();
  const numeric = Number.parseInt(value, 10);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return getSimulationScheduleDefaults();
  }
  if (field === 'gamesPerCard') {
    stateCache.simulationSetup.gamesPerCard = numeric;
  } else if (field === 'cardsPerOpponent') {
    stateCache.simulationSetup.cardsPerOpponent = numeric;
  }
  return getSimulationScheduleDefaults();
}

export function getSimulationSchedule() {
  ensureSimulationSetupState();
  return {
    gamesPerCard: stateCache.simulationSetup.gamesPerCard,
    cardsPerOpponent: stateCache.simulationSetup.cardsPerOpponent,
  };
}
