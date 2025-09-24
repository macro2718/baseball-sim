// Shared mutable state that mirrors the current UI view.

import { FIELD_POSITION_KEYS } from './config.js';

export const stateCache = {
  data: null,
  defensePlan: null,
  defenseSelection: { first: null, feedback: null },
  defenseContext: { lineup: {}, bench: {}, canSub: false },
  currentBatterIndex: null,
  statsView: { team: 'away', type: 'batting' },
  abilitiesView: { team: 'away', type: 'batting' },
  uiView: 'lobby',
  teamLibrary: { teams: [], selection: { home: null, away: null }, ready: false, hint: '' },
  teamBuilder: {
    currentTeamId: null,
    lastSavedId: null,
    editorDirty: false,
    form: null,
    players: { batters: [], pitchers: [], byId: {}, byName: {}, loaded: false, loading: false },
    selection: { group: 'lineup', index: 0 },
    catalog: 'batters',
    searchTerm: '',
  },
};

export function setUIView(view) {
  if (!view) return;
  stateCache.uiView = view;
}

export function getUIView() {
  return stateCache.uiView;
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
