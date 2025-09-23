// Shared mutable state that mirrors the current UI view.

import { FIELD_POSITION_KEYS } from './config.js';

export const stateCache = {
  data: null,
  defenseSelection: { lineupIndex: null, benchIndex: null },
  defenseContext: { lineup: {}, bench: {}, canSub: false },
  currentBatterIndex: null,
  statsView: { team: 'away', type: 'batting' },
};

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

export function updateDefenseContext(lineupMap, benchMap, canSub) {
  stateCache.defenseContext = {
    lineup: lineupMap,
    bench: benchMap,
    canSub: Boolean(canSub),
  };
}

export function resetDefenseSelection() {
  stateCache.defenseSelection = { lineupIndex: null, benchIndex: null };
}

export function isKnownFieldPosition(positionKey) {
  return FIELD_POSITION_KEYS.has(positionKey);
}
