import { stateCache, normalizePositionKey, canBenchPlayerCoverPosition } from '../state.js';

function computeTeamSignature(teamData) {
  if (!teamData) return '';
  const lineup = Array.isArray(teamData?.lineup) ? teamData.lineup : [];
  const bench = Array.isArray(teamData?.bench) ? teamData.bench : [];
  const lineupSig = lineup
    .map((player) => {
      const name = player?.name || '';
      const pos = normalizePositionKey(player?.position_key || player?.position) || '-';
      return `${name}:${pos}`;
    })
    .join('|');
  const benchSig = bench
    .map((player) => player?.name || '')
    .join('|');
  return `${teamData?.name || ''}::${lineupSig}::${benchSig}`;
}

function clonePlayer(player, source) {
  if (!player) return null;
  const eligibleAll = Array.isArray(player.eligible_all)
    ? player.eligible_all
    : Array.isArray(player.eligible)
    ? player.eligible
    : [];
  const normalizedEligibleAll = eligibleAll.map((pos) => String(pos || '').toUpperCase()).filter(Boolean);
  const normalizedEligible = Array.isArray(player.eligible)
    ? player.eligible.map((pos) => String(pos || '').toUpperCase()).filter(Boolean)
    : [];
  const clone = {
    name: player.name || '',
    position: player.position || player.position_key || '-',
    position_key: normalizePositionKey(player.position_key || player.position) || '-',
    eligible: normalizedEligible,
    eligible_all: normalizedEligibleAll,
    pitcher_type: player.pitcher_type || null,
    bats: player.bats || null,
    fielding_rating: player.fielding_rating || null,
    fielding_value: Number.isFinite(player.fielding_value) ? Number(player.fielding_value) : null,
    avg: player.avg || null,
    hr: player.hr || null,
    rbi: player.rbi || null,
    source,
    index: Number.isInteger(player.index) ? Number(player.index) : null,
  };
  return clone;
}

function updatePlayerForSlot(player, slot) {
  if (!player || !slot) return;
  player.position_key = slot.slotPositionKey || '-';
  player.position = slot.slotPositionLabel || player.position || player.position_key || '-';
  player.source = 'lineup';
  if (Number.isInteger(slot.index)) {
    player.index = slot.index;
  }
  player.order = slot.order;
}

function createPlan(teamKey, teamData, signature) {
  const lineupEntries = Array.isArray(teamData?.lineup) ? teamData.lineup : [];
  const benchEntries = Array.isArray(teamData?.bench) ? teamData.bench : [];
  const plan = {
    teamKey,
    teamName: teamData?.name || '',
    signature,
    lineup: lineupEntries.map((entry, index) => {
      const slotPositionKey = normalizePositionKey(entry?.position_key || entry?.position) || '-';
      const slotPositionLabel = entry?.position || entry?.position_key || '-';
      const slot = {
        index,
        order: Number.isInteger(entry?.order) ? entry.order : index + 1,
        slotPositionKey,
        slotPositionLabel,
        player: clonePlayer(entry, 'lineup'),
      };
      if (slot.player) {
        updatePlayerForSlot(slot.player, slot);
      }
      return slot;
    }),
    bench: benchEntries.map((entry, index) => {
      const clone = clonePlayer(entry, 'bench');
      if (clone) {
        clone.index = index;
        clone.source = 'bench';
      }
      return clone;
    }).filter(Boolean),
  };
  reindexPlan(plan);
  return plan;
}

function reindexPlan(plan) {
  if (!plan) return;
  plan.lineup.forEach((slot, index) => {
    if (!slot) return;
    slot.index = index;
    slot.order = index + 1;
    if (slot.player) {
      slot.player.index = index;
      slot.player.order = slot.order;
      updatePlayerForSlot(slot.player, slot);
    }
  });
  plan.bench.forEach((player, index) => {
    if (!player) return;
    player.index = index;
    player.source = 'bench';
  });
}

export function ensureTitleLineupPlan(teamKey, teamData, enabled) {
  const normalizedKey = teamKey === 'home' ? 'home' : teamKey === 'away' ? 'away' : null;
  if (!normalizedKey) {
    return null;
  }
  if (!enabled || !teamData) {
    stateCache.titleLineup.plans[normalizedKey] = null;
    if (stateCache.titleLineup.selection.team === normalizedKey) {
      clearTitleLineupSelection();
    }
    return null;
  }
  const signature = computeTeamSignature(teamData);
  const existing = stateCache.titleLineup.plans[normalizedKey];
  if (!existing || existing.signature !== signature) {
    stateCache.titleLineup.plans[normalizedKey] = createPlan(normalizedKey, teamData, signature);
    if (stateCache.titleLineup.selection.team === normalizedKey) {
      clearTitleLineupSelection();
    }
  }
  return stateCache.titleLineup.plans[normalizedKey];
}

export function getTitleLineupPlan(teamKey) {
  const normalizedKey = teamKey === 'home' ? 'home' : teamKey === 'away' ? 'away' : null;
  if (!normalizedKey) return null;
  return stateCache.titleLineup.plans[normalizedKey] || null;
}

export function getTitleLineupSelection() {
  return stateCache.titleLineup.selection || { team: null, type: null, index: null };
}

export function clearTitleLineupSelection() {
  stateCache.titleLineup.selection = { team: null, type: null, index: null };
}

export function setTitleLineupSelection(teamKey, type, index) {
  const normalizedKey = teamKey === 'home' ? 'home' : teamKey === 'away' ? 'away' : null;
  if (!normalizedKey) {
    clearTitleLineupSelection();
    return;
  }
  if (type !== 'lineup' && type !== 'bench') {
    clearTitleLineupSelection();
    return;
  }
  if (!Number.isInteger(index) || index < 0) {
    clearTitleLineupSelection();
    return;
  }
  const plan = getTitleLineupPlan(normalizedKey);
  if (!plan) {
    clearTitleLineupSelection();
    return;
  }
  if (type === 'lineup') {
    if (!plan.lineup[index]) {
      clearTitleLineupSelection();
      return;
    }
  } else if (type === 'bench') {
    if (!plan.bench[index]) {
      clearTitleLineupSelection();
      return;
    }
  }
  stateCache.titleLineup.selection = { team: normalizedKey, type, index };
}

export function swapTitleLineupPlayers(teamKey, indexA, indexB) {
  const plan = getTitleLineupPlan(teamKey);
  if (!plan) return false;
  if (!Number.isInteger(indexA) || !Number.isInteger(indexB)) return false;
  if (indexA === indexB) return false;
  const slotA = plan.lineup[indexA];
  const slotB = plan.lineup[indexB];
  if (!slotA || !slotB) return false;
  const temp = slotA.player;
  slotA.player = slotB.player || null;
  slotB.player = temp || null;
  if (slotA.player) {
    updatePlayerForSlot(slotA.player, slotA);
  }
  if (slotB.player) {
    updatePlayerForSlot(slotB.player, slotB);
  }
  reindexPlan(plan);
  return true;
}

export function moveBenchPlayerToLineup(teamKey, benchIndex, lineupIndex) {
  const plan = getTitleLineupPlan(teamKey);
  if (!plan) return false;
  if (!Number.isInteger(benchIndex) || !Number.isInteger(lineupIndex)) return false;
  const slot = plan.lineup[lineupIndex];
  if (!slot) return false;
  if (benchIndex < 0 || benchIndex >= plan.bench.length) return false;
  const incoming = plan.bench.splice(benchIndex, 1)[0];
  if (!incoming) return false;
  const outgoing = slot.player || null;
  slot.player = incoming;
  updatePlayerForSlot(slot.player, slot);
  if (outgoing) {
    outgoing.source = 'bench';
    outgoing.position = outgoing.position || outgoing.position_key || '-';
    outgoing.position_key = normalizePositionKey(outgoing.position_key || outgoing.position) || '-';
    plan.bench.push(outgoing);
  }
  reindexPlan(plan);
  return true;
}

export function getTitleLineupInvalidAssignments(plan) {
  if (!plan) return [];
  const invalid = [];
  plan.lineup.forEach((slot) => {
    if (!slot || !slot.player) return;
    const positionKey = slot.slotPositionKey || normalizePositionKey(slot.player.position_key || slot.player.position);
    if (!positionKey || positionKey === '-') return;
    if (!canBenchPlayerCoverPosition(slot.player, positionKey)) {
      invalid.push({ index: slot.index, positionKey, player: slot.player, slot });
    }
  });
  return invalid;
}

export function getBenchEligibilityForPosition(plan, positionKey) {
  if (!plan || !positionKey) return { eligible: new Set(), ineligible: new Set() };
  const eligible = new Set();
  const ineligible = new Set();
  plan.bench.forEach((player, index) => {
    if (!player) return;
    if (canBenchPlayerCoverPosition(player, positionKey)) {
      eligible.add(index);
    } else {
      ineligible.add(index);
    }
  });
  return { eligible, ineligible };
}

export function getLineupEligibilityForBenchPlayer(plan, benchIndex) {
  if (!plan) return { eligible: new Set(), ineligible: new Set() };
  const benchPlayer = plan.bench[benchIndex];
  if (!benchPlayer) return { eligible: new Set(), ineligible: new Set() };
  const eligiblePositions = new Set(
    Array.isArray(benchPlayer.eligible_all)
      ? benchPlayer.eligible_all.map((pos) => String(pos || '').toUpperCase()).filter(Boolean)
      : []
  );
  const eligible = new Set();
  const ineligible = new Set();
  plan.lineup.forEach((slot, index) => {
    if (!slot) return;
    const positionKey = slot.slotPositionKey || '-';
    if (positionKey && eligiblePositions.has(positionKey)) {
      eligible.add(index);
    } else {
      ineligible.add(index);
    }
  });
  return { eligible, ineligible };
}
