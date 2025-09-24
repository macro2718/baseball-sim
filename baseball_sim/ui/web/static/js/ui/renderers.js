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
import { triggerPlayAnimation, resetPlayAnimation } from './fieldAnimation.js';
import { updateFieldResultDisplay, resetFieldResultDisplay } from './fieldResultDisplay.js';

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

const ABILITY_METRIC_CONFIG = {
  k_pct: { mean: 22.8, variation: 0.15, min: 9 },
  bb_pct: { mean: 8.5, variation: 0.15, min: 2.5 },
  hard_pct: { mean: 38.6, variation: 0.15, min: 18 },
  gb_pct: { mean: 44.6, variation: 0.15, min: 20 },
  speed: { mean: 4.3, variation: 0.05, min: 3.3, max: 5.8 },
  fielding: { mean: 100, variation: 0.15, min: 40, max: 160 },
  stamina: { mean: 80, variation: 0.15, min: 30, max: 150 },
};

const ABILITY_COLOR_PRESETS = {
  table: Object.freeze({ emphasize: true, glowBase: 4.6, glowScale: 5.4 }),
  matchupPitcher: Object.freeze({ emphasize: true, glowBase: 3.1, glowScale: 4.6 }),
  matchupBatter: Object.freeze({ emphasize: false, glowBase: 2.4, glowScale: 3.8 }),
};

function parseAbilityNumeric(value) {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value !== 'string') {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed || trimmed === '-' || trimmed === '--') {
    return null;
  }
  const sanitized = trimmed.replace(/[,]/g, '');
  const match = sanitized.match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return null;
  }
  const numeric = Number.parseFloat(match[0]);
  return Number.isFinite(numeric) ? numeric : null;
}

function computeAbilityMetricStyling(metricKey, displayValue) {
  const config = ABILITY_METRIC_CONFIG[metricKey];
  if (!config) {
    return null;
  }

  const numeric = parseAbilityNumeric(displayValue);
  if (!Number.isFinite(numeric)) {
    return null;
  }

  const mean = config.mean ?? numeric;
  const variation = config.variation ?? 0.15;
  const spread = config.spread ?? Math.max(mean * variation * 4, 1e-6);
  const lowerBound =
    config.min != null ? config.min : Math.max(0, mean - spread);
  const upperBound =
    config.max != null ? config.max : Math.max(mean + spread, lowerBound + 1e-3);

  const ratio = (numeric - lowerBound) / (upperBound - lowerBound);
  const clamped = Math.min(1, Math.max(0, ratio));
  const distance = Math.abs(clamped - 0.5) * 2;

  const hue = 120 - clamped * 120;
  const saturationBase = config.saturationBase ?? 52;
  const saturationRange = config.saturationRange ?? 32;
  const saturation = Math.max(
    35,
    Math.min(90, saturationBase + saturationRange * distance),
  );
  const lightnessBase = config.lightnessBase ?? 67;
  const lightnessRange = config.lightnessRange ?? 15;
  const lightness = Math.max(
    42,
    Math.min(78, lightnessBase - lightnessRange * distance),
  );

  const hueValue = Math.round(hue);
  const saturationValue = Math.round(saturation);
  const lightnessValue = Math.round(lightness);

  return {
    color: `hsl(${hueValue}, ${saturationValue}%, ${lightnessValue}%)`,
    ratio: clamped,
    intensity: distance,
    numeric,
  };
}

function resetAbilityColor(element) {
  element.classList.remove('ability-colorized');
  element.style.removeProperty('--ability-color');
  element.style.removeProperty('--ability-intensity');
  element.style.removeProperty('color');
  element.style.removeProperty('text-shadow');
  element.style.removeProperty('font-weight');
  delete element.dataset.abilityMetric;
}

function applyAbilityColor(
  element,
  metricKey,
  displayValue,
  options = {},
) {
  if (!element) return;
  const config = metricKey ? ABILITY_METRIC_CONFIG[metricKey] : null;
  if (!config) {
    resetAbilityColor(element);
    return;
  }

  const styling = computeAbilityMetricStyling(metricKey, displayValue);
  if (!styling) {
    resetAbilityColor(element);
    return;
  }

  const { emphasize = false, glowBase = 3, glowScale = 4 } = options;
  const glow = glowBase + glowScale * styling.intensity;

  element.classList.add('ability-colorized');
  element.dataset.abilityMetric = metricKey;
  element.style.setProperty('--ability-color', styling.color);
  element.style.setProperty('--ability-intensity', styling.intensity.toFixed(3));
  element.style.color = styling.color;
  element.style.textShadow = `0 0 ${glow.toFixed(1)}px ${styling.color}`;

  if (emphasize) {
    let weight = '500';
    if (styling.intensity > 0.8) {
      weight = '700';
    } else if (styling.intensity > 0.45) {
      weight = '600';
    }
    element.style.fontWeight = weight;
  } else {
    element.style.removeProperty('font-weight');
  }
}

const FIELD_ALIGNMENT_SLOTS = FIELD_POSITIONS.filter((slot) => slot.key !== 'DH');

const BASE_LABELS = ['一塁', '二塁', '三塁'];

const HOME_START_INDEX = -1;
const HOME_BASE_INDEX = 3;
const RUNNER_SPEED_PX_PER_MS = 0.55;
const RUNNER_MIN_DURATION = 320;
const RUNNER_MAX_DURATION = 900;
const RUNNER_SEGMENT_DELAY = 120;
const HOME_HOLD_DURATION = 420;

let runnerAnimationToken = 0;

function normalizeRunnerName(name) {
  if (typeof name !== 'string') {
    return '';
  }
  return name.trim();
}

function cloneBaseInfo(info) {
  if (!info || typeof info !== 'object') {
    return null;
  }
  const clone = { occupied: Boolean(info.occupied) };
  if (typeof info.runner === 'string') {
    const trimmed = info.runner.trim();
    if (trimmed) {
      clone.runner = trimmed;
    }
  }
  if (info.speed !== undefined) clone.speed = info.speed;
  if (info.speed_display !== undefined) clone.speed_display = info.speed_display;
  if (info.runner_speed !== undefined) clone.runner_speed = info.runner_speed;
  return clone;
}

function buildDisplayInfo(info, fallbackName) {
  const clone = cloneBaseInfo(info) || {};
  clone.occupied = true;
  const normalizedName = normalizeRunnerName(clone.runner) || normalizeRunnerName(fallbackName);
  if (normalizedName) {
    clone.runner = normalizedName;
  } else if (typeof clone.runner === 'string') {
    clone.runner = clone.runner.trim();
  }
  return clone;
}

function createRunnerDescriptor(info, index) {
  if (!info || !info.occupied) {
    return null;
  }
  const descriptor = {
    info: cloneBaseInfo(info),
    baseIndex: index,
  };
  const name = normalizeRunnerName(info.runner);
  descriptor.name = name;
  const keyParts = [];
  if (name) {
    keyParts.push(name);
  }
  const speedDisplay =
    typeof info.speed_display === 'string' ? info.speed_display.trim() : '';
  if (speedDisplay) {
    keyParts.push(`spd:${speedDisplay}`);
  }
  const speedRaw = info.speed ?? info.runner_speed;
  if (speedRaw !== undefined && speedRaw !== null && speedRaw !== '') {
    keyParts.push(`sp:${speedRaw}`);
  }
  descriptor.key = keyParts.length ? keyParts.join('|') : `base-${index}`;
  return descriptor;
}

function extractRunnerDescriptors(bases) {
  if (!Array.isArray(bases)) {
    return [];
  }
  return bases
    .map((info, index) => createRunnerDescriptor(info, index))
    .filter((descriptor) => descriptor && descriptor.info);
}

function createSegments(start, end) {
  const segments = [];
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return segments;
  }
  for (let current = start; current < end; current += 1) {
    segments.push({ from: current, to: current + 1 });
  }
  return segments;
}

function calculateRunsScored(prevScore, nextScore, offenseKey) {
  if (!offenseKey) {
    return 0;
  }
  if (!prevScore || !nextScore) {
    return 0;
  }
  const prevValue = Number(prevScore[offenseKey]);
  const nextValue = Number(nextScore[offenseKey]);
  if (!Number.isFinite(prevValue) || !Number.isFinite(nextValue)) {
    return 0;
  }
  const diff = nextValue - prevValue;
  return diff > 0 ? diff : 0;
}

function createRunnerAnimationPlan({
  previousBases,
  nextBases,
  previousBatter,
  prevScore,
  nextScore,
  offense,
}) {
  const previousRunners = extractRunnerDescriptors(previousBases);
  const nextRunners = extractRunnerDescriptors(nextBases);

  const remainingNext = [...nextRunners];
  const unmatchedPrev = [];
  const movements = [];

  previousRunners.forEach((prevRunner) => {
    let matchIndex = -1;
    if (prevRunner.name) {
      matchIndex = remainingNext.findIndex((candidate) => candidate.name === prevRunner.name);
    }
    if (matchIndex === -1) {
      matchIndex = remainingNext.findIndex((candidate) => candidate.key === prevRunner.key);
    }
    if (matchIndex !== -1) {
      const nextRunner = remainingNext.splice(matchIndex, 1)[0];
      const diff = nextRunner.baseIndex - prevRunner.baseIndex;
      if (diff > 0) {
        const segments = createSegments(prevRunner.baseIndex, nextRunner.baseIndex);
        if (segments.length) {
          movements.push({
            runnerName: nextRunner.name || prevRunner.name || '',
            displayInfo: buildDisplayInfo(
              nextRunner.info || prevRunner.info,
              nextRunner.name || prevRunner.name,
            ),
            finalInfo: cloneBaseInfo(nextRunner.info),
            sourceInfo: cloneBaseInfo(prevRunner.info),
            segments,
          });
        }
      }
    } else {
      unmatchedPrev.push(prevRunner);
    }
  });

  remainingNext.forEach((runner) => {
    const segments = createSegments(HOME_START_INDEX, runner.baseIndex);
    if (segments.length) {
      movements.push({
        runnerName: runner.name || '',
        displayInfo: buildDisplayInfo(runner.info, runner.name),
        finalInfo: cloneBaseInfo(runner.info),
        sourceInfo: null,
        segments,
      });
    }
  });

  let runsScored = calculateRunsScored(prevScore, nextScore, offense);

  unmatchedPrev
    .sort((a, b) => b.baseIndex - a.baseIndex)
    .forEach((runner) => {
      if (runsScored <= 0) {
        return;
      }
      const segments = createSegments(runner.baseIndex, HOME_BASE_INDEX);
      if (segments.length) {
        movements.push({
          runnerName: runner.name || '',
          displayInfo: buildDisplayInfo(runner.info, runner.name),
          finalInfo: null,
          sourceInfo: cloneBaseInfo(runner.info),
          segments,
        });
        runsScored -= 1;
      }
    });

  if (runsScored > 0 && previousBatter) {
    const batterName = normalizeRunnerName(previousBatter.name);
    if (batterName) {
      const segments = createSegments(HOME_START_INDEX, HOME_BASE_INDEX);
      if (segments.length) {
        movements.push({
          runnerName: batterName,
          displayInfo: buildDisplayInfo(
            {
              occupied: true,
              runner: batterName,
              speed: previousBatter.speed ?? previousBatter.runner_speed ?? null,
              speed_display: previousBatter.speed_display ?? null,
              runner_speed: previousBatter.runner_speed ?? previousBatter.speed ?? null,
            },
            batterName,
          ),
          finalInfo: null,
          sourceInfo: null,
          segments,
        });
      }
    }
  }

  const arrivalSet = new Set();
  const startBases = new Set();

  movements.forEach((movement) => {
    if (!movement.displayInfo) {
      movement.displayInfo = buildDisplayInfo(movement.finalInfo, movement.runnerName);
    }
    movement.segments.forEach((segment, index) => {
      if (segment.from >= 0 && segment.from <= 2) {
        startBases.add(segment.from);
      }
      if (index === movement.segments.length - 1) {
        const destination = segment.to;
        if (destination >= 0 && destination <= 2) {
          arrivalSet.add(destination);
        }
      }
    });
  });

  return { movements, arrivalSet, startBases };
}

function wait(duration) {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, duration);
  });
}

async function playRunnerAnimations(movements, context) {
  await Promise.all(movements.map((movement) => playRunnerMovement(movement, context)));
}

async function playRunnerMovement(movement, context) {
  for (let i = 0; i < movement.segments.length; i += 1) {
    if (!context.isCurrent()) {
      return;
    }
    const segment = movement.segments[i];
    const isFinalSegment = i === movement.segments.length - 1;
    await playRunnerSegment(segment, movement, context, isFinalSegment);
    if (!context.isCurrent()) {
      return;
    }
    if (!isFinalSegment && context.segmentDelay > 0) {
      await wait(context.segmentDelay);
    }
  }
}

async function playRunnerSegment(segment, movement, context, isFinalSegment) {
  const { layer, getPoint, applyBaseDisplay, homeHighlight, isCurrent, speed, homeHoldDuration } =
    context;
  if (!layer) {
    return;
  }

  const fromPoint = getPoint(segment.from);
  const toPoint = getPoint(segment.to);
  if (!fromPoint || !toPoint) {
    return;
  }

  if (segment.from >= 0 && segment.from <= 2) {
    applyBaseDisplay(segment.from, null);
  }

  const dx = toPoint.x - fromPoint.x;
  const dy = toPoint.y - fromPoint.y;
  const distance = Math.hypot(dx, dy);
  if (distance <= 0) {
    if (segment.to >= 0 && segment.to <= 2) {
      const infoToApply =
        isFinalSegment && movement.finalInfo ? movement.finalInfo : movement.displayInfo;
      applyBaseDisplay(segment.to, infoToApply);
    } else if (segment.to === HOME_BASE_INDEX) {
      homeHighlight.enter(movement.displayInfo);
      await wait(homeHoldDuration);
      if (isCurrent()) {
        homeHighlight.leave();
      }
    }
    return;
  }

  const angle = Math.atan2(dy, dx);
  const duration = Math.max(
    RUNNER_MIN_DURATION,
    Math.min(distance / Math.max(speed, 1e-6), RUNNER_MAX_DURATION),
  );

  const segmentEl = document.createElement('div');
  segmentEl.className = 'runner-path';
  segmentEl.style.left = `${fromPoint.x}px`;
  segmentEl.style.top = `${fromPoint.y}px`;
  layer.appendChild(segmentEl);

  const keyframes = [
    {
      transform: `translate(-50%, -50%) rotate(${angle}rad) translateX(0px)`,
      opacity: 1,
    },
    {
      transform: `translate(-50%, -50%) rotate(${angle}rad) translateX(${distance}px)`,
      opacity: 0.4,
    },
  ];

  const animation = segmentEl.animate(keyframes, {
    duration,
    easing: 'ease-in-out',
  });

  try {
    await animation.finished;
  } catch (error) {
    // Ignore cancellation
  }
  segmentEl.remove();

  if (!isCurrent()) {
    return;
  }

  if (segment.to >= 0 && segment.to <= 2) {
    const infoToApply =
      isFinalSegment && movement.finalInfo ? movement.finalInfo : movement.displayInfo;
    applyBaseDisplay(segment.to, infoToApply);
  } else if (segment.to === HOME_BASE_INDEX) {
    homeHighlight.enter(movement.displayInfo);
    await wait(homeHoldDuration);
    if (isCurrent()) {
      homeHighlight.leave();
    }
  }
}

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
  
  // Prepare base ARIA defaults
  container.setAttribute('aria-label', '守備配置');

  const nextDefenseKey = gameState?.defense || null;
  const prevDefenseKey = container.dataset.teamKey || null;
  const isSwitchingDefense = Boolean(prevDefenseKey && nextDefenseKey && prevDefenseKey !== nextDefenseKey);
  const isAfterFade = container.dataset.justSwitched === '1';

  // Handle inactive or missing team state
  if (!gameState || !gameState.active || !nextDefenseKey || !teams) {
    container.innerHTML = '';
    container.classList.add('hidden');
    container.classList.remove('fade-hidden');
    container.setAttribute('aria-hidden', 'true');
    delete container.dataset.team;
    delete container.dataset.teamKey;
    delete container.dataset.pendingTeamKey;
    delete container.dataset.fading;
    delete container.dataset.justSwitched;
    return;
  }

  const defenseTeam = teams[nextDefenseKey] || null;
  if (!defenseTeam) {
    container.innerHTML = '';
    container.classList.add('hidden');
    container.classList.remove('fade-hidden');
    container.setAttribute('aria-hidden', 'true');
    delete container.dataset.team;
    delete container.dataset.teamKey;
    delete container.dataset.pendingTeamKey;
    delete container.dataset.fading;
    delete container.dataset.justSwitched;
    return;
  }

  // If switching and not already in a fade sequence, start fade-out and defer rendering
  if (isSwitchingDefense && !isAfterFade && !container.dataset.fading && container.children.length > 0) {
    container.dataset.fading = '1';
    container.dataset.pendingTeamKey = nextDefenseKey;
    container.classList.add('fade-hidden');
    // Wait for CSS transition (~240ms); add a small buffer
    globalThis.setTimeout(() => {
      // Mark as after-fade for the next invocation
      container.dataset.justSwitched = '1';
      delete container.dataset.fading;
      // Re-render now with the same state (will take the after-fade path)
      try {
        updateDefenseAlignment(gameState, teams);
      } catch (e) {
        // no-op
      }
    }, 260);
    return; // Defer actual content switch until after fade-out
  }

  // From here we will actually render the alignment
  container.innerHTML = '';

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
    const teamName = defenseTeam.name || '';
    container.classList.remove('hidden');
    container.setAttribute('aria-hidden', 'false');
    container.setAttribute('aria-label', teamName ? `守備配置 (${teamName})` : '守備配置');
    if (teamName) container.dataset.team = teamName;
    // Track current defense side by key for future comparisons
    container.dataset.teamKey = nextDefenseKey;
  } else {
    container.classList.add('hidden');
    container.setAttribute('aria-hidden', 'true');
    delete container.dataset.team;
  }

  // If we had faded out due to switching, schedule fade-in after layout
  if (isAfterFade || (container.classList.contains('fade-hidden') && rendered > 0)) {
    // Force reflow to ensure transition applies
    // eslint-disable-next-line no-unused-expressions
    void container.offsetWidth;
    // Small timeout to allow DOM paint
    globalThis.setTimeout(() => {
      container.classList.remove('fade-hidden');
      delete container.dataset.justSwitched;
      delete container.dataset.pendingTeamKey;
    }, 10);
  } else {
    // Ensure we are fully visible in normal updates
    container.classList.remove('fade-hidden');
  }
}

function updateBatterAlignment(gameState, teams) {
  const container = elements.batterAlignment;
  if (!container) return;

  // Reset/hide by default
  container.innerHTML = '';
  container.classList.add('hidden');
  container.setAttribute('aria-hidden', 'true');

  if (!gameState || !gameState.active) {
    return;
  }

  const offenseKey = gameState.offense;
  const offenseTeam = offenseKey && teams ? teams[offenseKey] : null;
  const lineup = Array.isArray(offenseTeam?.lineup) ? offenseTeam.lineup : [];
  const batter = lineup.find((p) => p && p.is_current_batter) || gameState.current_batter || null;
  if (!batter) return;

  // Determine handedness: "R" -> righty (stands left of plate from viewer), "L" -> lefty (right of plate)
  const batsRaw = (batter.bats || '').toString().trim().toUpperCase();
  let handedClass = 'righty';
  if (batsRaw === 'L') handedClass = 'lefty';
  else if (batsRaw === 'S' || batsRaw === 'B') {
    // For switch hitters, bias to opposite of pitcher throws if available, else default righty
    const defenseKey = gameState.defense;
    const defenseTeam = defenseKey && teams ? teams[defenseKey] : null;
    const currentPitcher = (defenseTeam?.pitchers || []).find((p) => p && p.is_current) || null;
    const throws = (currentPitcher?.throws || '').toString().trim().toUpperCase();
    if (throws === 'R') handedClass = 'lefty';
    else if (throws === 'L') handedClass = 'righty';
  }

  const batsLabel = formatBatsLabel(batter.bats) || '--';
  const nameLabel = escapeHtml(batter.name ?? '-');

  const slot = document.createElement('div');
  slot.className = `batter-slot ${handedClass}`;
  slot.innerHTML = `
    <div class="batter-chip">
      <span class="batter-pos">BAT</span>
      <span class="batter-name">${nameLabel}</span>
      <span class="batter-bats" aria-label="打席">${escapeHtml(batsLabel)}</span>
    </div>
  `;
  container.appendChild(slot);

  container.classList.remove('hidden');
  container.setAttribute('aria-hidden', 'false');
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

function updateBases(nextBases, options = {}) {
  if (!elements.baseState) return;

  const field = elements.baseState;
  const baseElements = Array.from(field.querySelectorAll('.base'));
  const baseMap = new Map();
  baseElements.forEach((el) => {
    const baseIndex = Number(el.dataset.base);
    if (Number.isFinite(baseIndex)) {
      baseMap.set(baseIndex, el);
    }
  });

  const runnerLayer = elements.runnerAnimationLayer || null;
  const homePlate = field.querySelector('.home-plate');

  runnerAnimationToken += 1;
  const animationToken = runnerAnimationToken;

  if (runnerLayer) {
    runnerLayer.innerHTML = '';
  }

  const homeHighlight = (() => {
    let activeCount = 0;
    return {
      enter(info) {
        if (!homePlate) return;
        activeCount += 1;
        homePlate.classList.add('occupied');
        homePlate.classList.add('runner-arrival');
        if (info && typeof info.runner === 'string' && info.runner.trim() !== '') {
          homePlate.dataset.runner = info.runner.trim();
        } else {
          homePlate.removeAttribute('data-runner');
        }
      },
      leave() {
        if (!homePlate) return;
        if (activeCount > 0) {
          activeCount -= 1;
        }
        if (activeCount <= 0) {
          homePlate.classList.remove('runner-arrival');
          homePlate.classList.remove('occupied');
          homePlate.removeAttribute('data-runner');
          activeCount = 0;
        }
      },
      reset() {
        if (!homePlate) return;
        activeCount = 0;
        homePlate.classList.remove('runner-arrival');
        homePlate.classList.remove('occupied');
        homePlate.removeAttribute('data-runner');
      },
    };
  })();

  homeHighlight.reset();

  const applyBaseDisplay = (baseIndex, info) => {
    const el = baseMap.get(baseIndex);
    if (!el) return;

    const baseInfo = info && typeof info === 'object' ? info : null;
    const isOccupied = Boolean(baseInfo?.occupied);
    const hasRunnerName =
      isOccupied && typeof baseInfo?.runner === 'string' && baseInfo.runner.trim() !== '';
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
      if (hasRunnerName && nameEl && speedEl) {
        nameEl.textContent = baseInfo.runner;
        const speedText = formatRunnerSpeed(baseInfo);
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
        const speedText = formatRunnerSpeed(baseInfo);
        ariaLabel = `${baseLabel}: ${baseInfo.runner}${speedText ? `（スピード ${speedText}）` : ''}`;
      } else {
        ariaLabel = `${baseLabel}: 走者あり`;
      }
    }
    el.setAttribute('aria-label', ariaLabel);
    el.removeAttribute('title');
  };

  const sanitizedBases = Array.isArray(nextBases) ? nextBases : [];
  const finalBaseInfos = new Map();
  baseMap.forEach((_, index) => {
    finalBaseInfos.set(index, sanitizedBases[index] || null);
  });

  const previousBases = Array.isArray(options.previousBases) ? options.previousBases : null;
  const sequenceChanged = Boolean(options.sequenceChanged);

  const canAttemptAnimation =
    sequenceChanged &&
    previousBases &&
    baseMap.size > 0 &&
    previousBases.length === baseMap.size &&
    runnerLayer;

  if (!canAttemptAnimation || !homePlate) {
    finalBaseInfos.forEach((info, index) => {
      applyBaseDisplay(index, info);
    });
    return;
  }

  const plan = createRunnerAnimationPlan({
    previousBases,
    nextBases: sanitizedBases,
    previousBatter: options.previousBatter || null,
    prevScore: options.previousScore || null,
    nextScore: options.currentScore || null,
    offense: options.previousOffense || null,
  });

  if (!plan.movements.length) {
    finalBaseInfos.forEach((info, index) => {
      applyBaseDisplay(index, info);
    });
    return;
  }

  const layerRect = runnerLayer.getBoundingClientRect();
  if (!layerRect || layerRect.width <= 0 || layerRect.height <= 0) {
    finalBaseInfos.forEach((info, index) => {
      applyBaseDisplay(index, info);
    });
    return;
  }

  const homeRect = homePlate.getBoundingClientRect();
  const homePoint = homeRect
    ? {
        x: homeRect.left + homeRect.width / 2 - layerRect.left,
        y: homeRect.top + homeRect.height / 2 - layerRect.top,
      }
    : null;

  const basePointCache = new Map();
  const getPoint = (position) => {
    if (position === HOME_START_INDEX || position === HOME_BASE_INDEX) {
      return homePoint;
    }
    if (basePointCache.has(position)) {
      return basePointCache.get(position);
    }
    const baseEl = baseMap.get(position);
    if (!baseEl) {
      return null;
    }
    const rect = baseEl.getBoundingClientRect();
    const point = {
      x: rect.left + rect.width / 2 - layerRect.left,
      y: rect.top + rect.height / 2 - layerRect.top,
    };
    basePointCache.set(position, point);
    return point;
  };

  finalBaseInfos.forEach((info, index) => {
    if (plan.startBases.has(index) || plan.arrivalSet.has(index)) {
      applyBaseDisplay(index, null);
    } else {
      applyBaseDisplay(index, info);
    }
  });

  const animationContext = {
    layer: runnerLayer,
    getPoint,
    applyBaseDisplay,
    homeHighlight,
    isCurrent: () => animationToken === runnerAnimationToken,
    speed: RUNNER_SPEED_PX_PER_MS,
    segmentDelay: RUNNER_SEGMENT_DELAY,
    homeHoldDuration: HOME_HOLD_DURATION,
  };

  playRunnerAnimations(plan.movements, animationContext)
    .catch(() => undefined)
    .finally(() => {
      if (animationToken !== runnerAnimationToken) {
        return;
      }
      plan.arrivalSet.forEach((index) => {
        applyBaseDisplay(index, finalBaseInfos.get(index) || null);
      });
      homeHighlight.reset();
      if (runnerLayer) {
        runnerLayer.innerHTML = '';
      }
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

  const section = listEl.closest('.pitcher-section');

  const visiblePitchers = Array.isArray(pitchers)
    ? pitchers.filter((pitcher) => {
        if (!pitcher) return false;
        if (pitcher.is_current) return true;
        if ('has_entered_game' in pitcher) return Boolean(pitcher.has_entered_game);
        if ('has_played' in pitcher) return Boolean(pitcher.has_played);
        return false;
      })
    : [];

  const hasPitchers = visiblePitchers.length > 0;

  if (section) {
    if (hasPitchers) {
      section.classList.remove('hidden');
      section.setAttribute('aria-hidden', 'false');
    } else {
      section.classList.add('hidden');
      section.setAttribute('aria-hidden', 'true');
    }
  }

  if (!hasPitchers) {
    return;
  }

  visiblePitchers.forEach((pitcher) => {
    const li = document.createElement('li');
    if (pitcher.is_current) {
      li.classList.add('current');
    }

    const nameSpan = document.createElement('span');
    const typeRaw = pitcher.pitcher_type != null ? String(pitcher.pitcher_type) : 'P';
    const typeLabel = typeRaw.trim() || 'P';
    const nameLabel = pitcher.name != null ? String(pitcher.name).trim() : '';
    nameSpan.textContent = `${nameLabel || '-'} (${typeLabel})`;

    const staminaSpan = document.createElement('span');
    const staminaValue = pitcher.stamina != null ? `${pitcher.stamina}` : '-';
    staminaSpan.textContent = staminaValue;

    li.appendChild(nameSpan);
    li.appendChild(staminaSpan);
    listEl.appendChild(li);
  });
}

function normalizeTraitValue(value) {
  if (value === null || value === undefined) {
    return '--';
  }
  const text = String(value).trim();
  if (!text || text === '-' || text === '--') {
    return '--';
  }
  return text;
}

function coalesceTraitValue(primary, fallback) {
  const normalizedPrimary = normalizeTraitValue(primary);
  if (normalizedPrimary !== '--') {
    return normalizedPrimary;
  }
  return normalizeTraitValue(fallback);
}

function setMatchupText(element, value, metricKey) {
  if (!element) return;
  const normalized = normalizeTraitValue(value);
  element.textContent = normalized;

  if (metricKey) {
    applyAbilityColor(
      element,
      metricKey,
      normalized,
      ABILITY_COLOR_PRESETS.matchupPitcher,
    );
  } else {
    resetAbilityColor(element);
  }
}

function buildPitcherDisplay(pitcher, defenseTeam) {
  const base = {
    name: normalizeTraitValue(pitcher?.name),
    pitcher_type: normalizeTraitValue(pitcher?.pitcher_type),
    throws: normalizeTraitValue(pitcher?.throws),
    k_pct: normalizeTraitValue(pitcher?.k_pct),
    bb_pct: normalizeTraitValue(pitcher?.bb_pct),
    hard_pct: normalizeTraitValue(pitcher?.hard_pct),
    gb_pct: normalizeTraitValue(pitcher?.gb_pct),
  };

  if (!defenseTeam) {
    return base;
  }

  const traitEntries = defenseTeam.traits?.pitching;
  if (!Array.isArray(traitEntries) || !pitcher?.name) {
    return base;
  }

  const trait = traitEntries.find((entry) => entry && entry.name === pitcher.name);
  if (!trait) {
    return base;
  }

  return {
    ...base,
    name: coalesceTraitValue(base.name, trait.name),
    pitcher_type: coalesceTraitValue(base.pitcher_type, trait.pitcher_type),
    throws: coalesceTraitValue(base.throws, trait.throws),
    k_pct: coalesceTraitValue(base.k_pct, trait.k_pct),
    bb_pct: coalesceTraitValue(base.bb_pct, trait.bb_pct),
    hard_pct: coalesceTraitValue(base.hard_pct, trait.hard_pct),
    gb_pct: coalesceTraitValue(base.gb_pct, trait.gb_pct),
  };
}

function buildUpcomingBatters(offenseTeam, gameState) {
  if (!offenseTeam || !Array.isArray(offenseTeam.lineup) || offenseTeam.lineup.length === 0) {
    return [];
  }

  const lineup = offenseTeam.lineup;
  let currentIndex = lineup.findIndex((player) => player && player.is_current_batter);
  if (currentIndex < 0 && gameState?.current_batter?.name) {
    const targetName = String(gameState.current_batter.name).trim();
    if (targetName) {
      currentIndex = lineup.findIndex((player) => player && player.name === targetName);
    }
  }
  if (currentIndex < 0) {
    currentIndex = 0;
  }

  const count = Math.min(3, lineup.length);
  const batters = [];
  for (let offset = 0; offset < count; offset += 1) {
    const index = (currentIndex + offset) % lineup.length;
    const player = lineup[index];
    if (!player) continue;
    batters.push({ player, isCurrent: offset === 0 });
  }
  return batters;
}

function renderUpcomingBatters(listEl, emptyEl, batters, battingTraits) {
  if (!listEl) return;

  listEl.innerHTML = '';

  const hasBatters = Array.isArray(batters) && batters.length > 0;
  if (emptyEl) {
    emptyEl.classList.toggle('hidden', hasBatters);
  }
  if (!hasBatters) {
    return;
  }

  const traitMap = new Map();
  if (Array.isArray(battingTraits)) {
    battingTraits.forEach((trait) => {
      if (!trait || !trait.name) return;
      traitMap.set(trait.name, trait);
    });
  }

  batters.forEach(({ player, isCurrent }) => {
    const trait = traitMap.get(player.name);
    const orderLabel = Number.isInteger(player.order) ? `${player.order}番` : '--';
    const nameLabel = escapeHtml(player.name ?? '--');
    const positionHtml = renderPositionToken(player.position, player.pitcher_type);
    const batsRaw = player.bats || (trait && trait.bats && trait.bats !== '-' ? trait.bats : null);
    const batsBadge = renderBatsBadge(batsRaw);
    const batsBadgeHtml = batsBadge || '<span class="batter-bats-placeholder">--</span>';
    const batsLabel = normalizeTraitValue(formatBatsLabel(batsRaw));

    const li = document.createElement('li');
    li.className = 'matchup-batter-item';
    if (isCurrent) {
      li.classList.add('current');
    }
    li.innerHTML = `
      <div class="matchup-batter-header">
        <div class="matchup-batter-title">
          <span class="matchup-batter-order">${escapeHtml(orderLabel)}</span>
          <span class="matchup-batter-name">${nameLabel}</span>
        </div>
        <div class="matchup-batter-badges">
          ${positionHtml}
          ${batsBadgeHtml}
        </div>
      </div>
      <div class="matchup-batter-stats"></div>
    `;

    const statsContainer = li.querySelector('.matchup-batter-stats');
    if (statsContainer) {
      const stats = [
        { label: 'K%', key: 'k_pct', value: coalesceTraitValue(player.k_pct, trait?.k_pct) },
        { label: 'BB%', key: 'bb_pct', value: coalesceTraitValue(player.bb_pct, trait?.bb_pct) },
        { label: 'Hard%', key: 'hard_pct', value: coalesceTraitValue(player.hard_pct, trait?.hard_pct) },
        { label: 'GB%', key: 'gb_pct', value: coalesceTraitValue(player.gb_pct, trait?.gb_pct) },
      ];

      stats.forEach((entry) => {
        const statWrap = document.createElement('span');
        statWrap.className = 'matchup-batter-stat';

        const labelEl = document.createElement('strong');
        labelEl.textContent = entry.label;

        const valueEl = document.createElement('span');
        valueEl.className = 'matchup-batter-stat-value';
        const displayValue = normalizeTraitValue(entry.value);
        valueEl.textContent = displayValue;

        applyAbilityColor(
          valueEl,
          entry.key,
          displayValue,
          ABILITY_COLOR_PRESETS.matchupBatter,
        );

        statWrap.appendChild(labelEl);
        statWrap.appendChild(valueEl);
        statsContainer.appendChild(statWrap);
      });
    }

    listEl.appendChild(li);
  });
}

function updateMatchupPanel(gameState, teams) {
  const {
    matchupPanel,
    matchupPitcherName,
    matchupPitcherType,
    matchupPitcherThrows,
    matchupPitcherK,
    matchupPitcherBB,
    matchupPitcherHard,
    matchupPitcherGB,
    upcomingBattersList,
    upcomingBattersEmpty,
  } = elements;

  if (!matchupPanel) return;

  const isActive = Boolean(gameState && gameState.active);
  matchupPanel.classList.toggle('inactive', !isActive);

  const defenseKey = gameState ? gameState.defense : null;
  const offenseKey = gameState ? gameState.offense : null;

  const defenseTeam = defenseKey ? teams?.[defenseKey] : null;
  const offenseTeam = offenseKey ? teams?.[offenseKey] : null;

  let pitcherSource = isActive ? gameState.current_pitcher || null : null;
  if (!pitcherSource && defenseTeam && Array.isArray(defenseTeam.pitchers)) {
    pitcherSource = defenseTeam.pitchers.find((pitcher) => pitcher && pitcher.is_current) || null;
  }

  const pitcherDisplay = buildPitcherDisplay(pitcherSource, defenseTeam);

  setMatchupText(matchupPitcherName, pitcherDisplay.name);
  setMatchupText(matchupPitcherType, pitcherDisplay.pitcher_type);
  setMatchupText(matchupPitcherThrows, pitcherDisplay.throws);
  setMatchupText(matchupPitcherK, pitcherDisplay.k_pct, 'k_pct');
  setMatchupText(matchupPitcherBB, pitcherDisplay.bb_pct, 'bb_pct');
  setMatchupText(matchupPitcherHard, pitcherDisplay.hard_pct, 'hard_pct');
  setMatchupText(matchupPitcherGB, pitcherDisplay.gb_pct, 'gb_pct');

  const batters = isActive ? buildUpcomingBatters(offenseTeam, gameState) : [];
  renderUpcomingBatters(
    upcomingBattersList,
    upcomingBattersEmpty,
    batters,
    offenseTeam?.traits?.batting || [],
  );
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

function populateSelectSimple(selectEl, options, placeholder) {
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

    populateSelectSimple(
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

    populateSelectSimple(
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

export function renderGame(gameState, teams, log, previousGameState = null) {
  updateAnalyticsPanel(gameState);
  updateDefenseAlignment(gameState, teams);
  updateBatterAlignment(gameState, teams);
  const isActiveGame = Boolean(gameState && gameState.active);
  const showGameView = stateCache.uiView === 'game';
  setInsightsVisibility(isActiveGame && showGameView);

  if (!isActiveGame) {
    resetPlayAnimation();
    resetFieldResultDisplay();
    updateScoreboard(gameState || {}, teams || {});
    updateOutsIndicator(gameState?.outs ?? 0);
    elements.actionWarning.textContent = '';
    elements.swingButton.disabled = true;
    elements.buntButton.disabled = true;
    if (elements.batterAlignment) {
      elements.batterAlignment.innerHTML = '';
      elements.batterAlignment.classList.add('hidden');
      elements.batterAlignment.setAttribute('aria-hidden', 'true');
    }
    updateRosters(elements.offenseRoster, []);
    updateRosters(elements.defenseRoster, []);
    updateBench(elements.offenseBench, []);
    updateBench(elements.defenseBenchList, []);
    updatePitchers(elements.offensePitchers, []);
    updatePitchers(elements.defensePitchers, []);
    updateMatchupPanel(gameState || {}, teams || {});
    updateLog(log || []);
    elements.defenseErrors.classList.add('hidden');
    elements.defenseErrors.textContent = '';
    updateStrategyControls(gameState || {}, teams || {});
    return;
  }

  updateScoreboard(gameState, teams);

  if (!showGameView) {
    updateMatchupPanel(gameState, teams);
    updateLog(log || []);
    return;
  }

  elements.situationText.textContent = gameState.situation || '';
  elements.halfIndicator.textContent = `${gameState.half_label} ${gameState.inning}`;
  updateOutsIndicator(gameState.outs);
  elements.matchupText.textContent = gameState.matchup || '';

  const currentSequence = Number(gameState.last_play?.sequence ?? 0);
  const previousSequence = Number(previousGameState?.last_play?.sequence ?? 0);
  const sequenceChanged = currentSequence !== previousSequence;

  updateBases(gameState.bases || [], {
    previousBases: previousGameState?.bases || null,
    previousBatter: previousGameState?.current_batter || null,
    previousScore: previousGameState?.score || null,
    currentScore: gameState.score || null,
    previousOffense: previousGameState?.offense || null,
    sequenceChanged,
  });
  triggerPlayAnimation(gameState.last_play || null, { isActive: true });
  updateFieldResultDisplay(gameState, previousGameState || null);

  const offenseTeam = gameState.offense ? teams[gameState.offense] : null;
  const defenseTeam = gameState.defense ? teams[gameState.defense] : null;

  updateRosters(elements.offenseRoster, offenseTeam?.lineup || []);
  updateBench(elements.offenseBench, offenseTeam?.bench || []);
  updateRosters(elements.defenseRoster, defenseTeam?.lineup || []);
  updateBench(elements.defenseBenchList, defenseTeam?.bench || []);

  updateMatchupPanel(gameState, teams);

  updatePitchers(elements.offensePitchers, offenseTeam?.pitchers || []);
  updatePitchers(elements.defensePitchers, defenseTeam?.pitchers || []);

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

function populateSelect(select, options, { placeholder, selected, fallback }) {
  if (!select) return;
  const previousValue = select.value;
  select.innerHTML = '';

  const placeholderOption = document.createElement('option');
  placeholderOption.value = '';
  placeholderOption.textContent = placeholder ?? '選択してください';
  select.appendChild(placeholderOption);

  options.forEach((option) => {
    const opt = document.createElement('option');
    opt.value = option.value;
    opt.textContent = option.label;
    select.appendChild(opt);
  });

  const availableValues = new Set(options.map((option) => option.value));
  let target = selected;
  if (!target || !availableValues.has(target)) {
    if (availableValues.has(previousValue)) {
      target = previousValue;
    } else if (fallback && availableValues.has(fallback)) {
      target = fallback;
    } else {
      target = '';
    }
  }
  select.value = target || '';
}

function renderLobby(teamLibraryState) {
  if (!teamLibraryState) {
    teamLibraryState = {};
  }
  const teams = Array.isArray(teamLibraryState.teams) ? teamLibraryState.teams : [];
  const selection = teamLibraryState.selection || {};
  const homeSelect = elements.lobbyHomeSelect;
  const awaySelect = elements.lobbyAwaySelect;
  const optionList = teams.map((team) => ({
    value: team.id,
    label: team.name || team.id,
  }));

  const fallbackHome = optionList.length > 0 ? optionList[0].value : '';
  if (homeSelect) {
    populateSelect(homeSelect, optionList, {
      placeholder: 'チームを選択',
      selected: selection.home,
      fallback: fallbackHome,
    });
  }

  if (awaySelect) {
    populateSelect(awaySelect, optionList, {
      placeholder: 'チームを選択',
      selected: selection.away,
      fallback: optionList.length > 1 ? optionList[1].value : fallbackHome,
    });
  }

  if (elements.lobbyHint) {
    elements.lobbyHint.textContent = teamLibraryState.hint || '';
  }

  if (elements.enterTitleButton) {
    const canStart = Boolean(homeSelect?.value && awaySelect?.value);
    elements.enterTitleButton.disabled = !canStart;
  }
}

function renderTeamBuilder(teamLibraryState) {
  const select = elements.teamEditorSelect;
  if (!select) return;

  const teams = Array.isArray(teamLibraryState?.teams) ? teamLibraryState.teams : [];
  const previousValue = select.value;
  let desiredValue = previousValue;
  if (!stateCache.teamBuilder.editorDirty) {
    desiredValue =
      stateCache.teamBuilder.lastSavedId || stateCache.teamBuilder.currentTeamId || previousValue;
  }

  select.innerHTML = '';

  const placeholderOption = document.createElement('option');
  placeholderOption.value = '';
  placeholderOption.textContent = 'チームを選択';
  select.appendChild(placeholderOption);

  const newOption = document.createElement('option');
  newOption.value = '__new__';
  newOption.textContent = '新規チームを作成';
  select.appendChild(newOption);

  teams.forEach((team) => {
    const option = document.createElement('option');
    option.value = team.id;
    option.textContent = team.name || team.id;
    select.appendChild(option);
  });

  const validValues = new Set(['', '__new__', ...teams.map((team) => team.id)]);
  if (!desiredValue || !validValues.has(desiredValue)) {
    desiredValue = '';
  }
  select.value = desiredValue;

  if (!stateCache.teamBuilder.editorDirty) {
    if (desiredValue && desiredValue !== '__new__') {
      stateCache.teamBuilder.currentTeamId = desiredValue;
    } else {
      stateCache.teamBuilder.currentTeamId = null;
    }
    stateCache.teamBuilder.lastSavedId = null;
  }
}

export function updateScreenVisibility() {
  const view = stateCache.uiView;
  const showLobby = view === 'lobby';
  const showBuilder = view === 'team-builder';
  const showTitle = view === 'title';
  const showGame = view === 'game';

  if (elements.lobbyScreen) {
    elements.lobbyScreen.classList.toggle('hidden', !showLobby);
  }
  if (elements.teamBuilderScreen) {
    elements.teamBuilderScreen.classList.toggle('hidden', !showBuilder);
  }
  if (elements.titleScreen) {
    elements.titleScreen.classList.toggle('hidden', !showTitle);
  }
  if (elements.gameScreen) {
    elements.gameScreen.classList.toggle('hidden', !showGame);
  }
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

  const columns =
    viewType === 'pitching' ? ABILITY_PITCHING_COLUMNS : ABILITY_BATTING_COLUMNS;
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
          const displayValue =
            value != null && value !== '' ? String(value) : '-';
          td.textContent = displayValue;

          if (ABILITY_METRIC_CONFIG[column.key]) {
            applyAbilityColor(
              td,
              column.key,
              displayValue,
              ABILITY_COLOR_PRESETS.table,
            );
          }

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
  const previousData = stateCache.data;
  stateCache.data = data;

  const teamLibraryState = data.team_library || {};
  stateCache.teamLibrary = {
    teams: Array.isArray(teamLibraryState.teams) ? teamLibraryState.teams : [],
    selection: teamLibraryState.selection || { home: null, away: null },
    ready: Boolean(teamLibraryState.ready),
    hint: teamLibraryState.hint || '',
    active: teamLibraryState.active || { home: null, away: null },
  };

  if (data.game?.active) {
    stateCache.uiView = 'game';
  } else if (stateCache.uiView === 'game') {
    stateCache.uiView = 'title';
  }

  setStatusMessage(data.notification);
  renderLobby(stateCache.teamLibrary);
  renderTeamBuilder(stateCache.teamLibrary);
  renderTitle(data.title);
  renderGame(data.game, data.teams, data.log, previousData?.game || null);
  updateStatsPanel(data);
  updateAbilitiesPanel(data);
  updateScreenVisibility();
}
