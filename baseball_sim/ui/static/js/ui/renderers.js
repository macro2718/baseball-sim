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
  getTitlePitcherSelection,
  setTitlePitcherSelection,
  clearTitlePitcherSelection,
  setPinchRunContext,
  setPinchRunSelectedBase,
  getPinchRunSelectedBase,
  setSimulationLeagueTeams,
  getSimulationSchedule,
  getSimulationScheduleDefaults,
  getSimulationLeagueTeams,
  primeSimulationSetup,
  getPlayersSelectedTeamIndex,
  setPlayersSelectedTeamIndex,
  getSimulationRankingsState,
  setSimulationRankingsScope,
  resetSimulationRankingsOverride,
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
import {
  ensureTitleLineupPlan,
  getTitleLineupSelection,
  getTitleLineupInvalidAssignments,
  getBenchEligibilityForPosition,
  getLineupEligibilityForBenchPlayer,
  isTitleEditorOpen,
  getTitleEditorTab,
  getTitleEditorView,
} from './titleLineup.js';
import { setStatusMessage } from './status.js';
import { triggerPlayAnimation, resetPlayAnimation } from './fieldAnimation.js';
import { updateFieldResultDisplay, resetFieldResultDisplay } from './fieldResultDisplay.js';
import { hideOffenseMenu, hideDefenseMenu } from './menus.js';
import { updateOverlayEvents } from './overlayEvents.js';

const JAPANESE_COLLATOR = new Intl.Collator('ja-JP', { numeric: true, sensitivity: 'base' });

function setInsightsVisibility(visible) {
  const { insightGrid } = elements;
  if (!insightGrid) return;
  if (visible) {
    insightGrid.classList.remove('hidden');
  } else {
    insightGrid.classList.add('hidden');
  }
}

function normalizeAnalyticsState(rawAnalytics) {
  const base = {
    running: false,
    samples: 0,
    sequence: null,
    offense: null,
    timestamp: null,
    result: null,
  };

  if (!rawAnalytics || typeof rawAnalytics !== 'object') {
    return base;
  }

  base.running = Boolean(rawAnalytics.running);
  const sampleValue = Number(rawAnalytics.samples);
  if (Number.isFinite(sampleValue) && sampleValue >= 0) {
    base.samples = sampleValue;
  }
  if (rawAnalytics.offense === 'home' || rawAnalytics.offense === 'away') {
    base.offense = rawAnalytics.offense;
  }
  const sequenceValue = Number(rawAnalytics.sequence);
  if (Number.isFinite(sequenceValue)) {
    base.sequence = sequenceValue;
  }
  if (rawAnalytics.timestamp !== undefined) {
    base.timestamp = rawAnalytics.timestamp;
  }
  if (rawAnalytics.result && typeof rawAnalytics.result === 'object') {
    const expectedRuns = Number(rawAnalytics.result.expected_runs);
    const scoreProbability = Number(rawAnalytics.result.score_probability);
    const winProbability = Number(rawAnalytics.result.home_win_probability);
    base.result = {
      expected_runs: Number.isFinite(expectedRuns) ? expectedRuns : null,
      score_probability: Number.isFinite(scoreProbability) ? scoreProbability : null,
      home_win_probability: Number.isFinite(winProbability) ? winProbability : null,
      home_wins: Number.isFinite(Number(rawAnalytics.result.home_wins))
        ? Number(rawAnalytics.result.home_wins)
        : 0,
      away_wins: Number.isFinite(Number(rawAnalytics.result.away_wins))
        ? Number(rawAnalytics.result.away_wins)
        : 0,
      ties: Number.isFinite(Number(rawAnalytics.result.ties))
        ? Number(rawAnalytics.result.ties)
        : 0,
    };
  }

  return base;
}

function formatSimulationTimestamp(timestamp) {
  if (!timestamp) return '';
  try {
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) {
      return '';
    }
    return date.toLocaleString('ja-JP', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch (error) {
    console.warn('Failed to format timestamp', error);
    return '';
  }
}

function formatPercent01(value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return '--';
  return `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%`;
}

function hasAbilityData(team) {
  if (!team || !team.traits) return false;
  const { batting, pitching } = team.traits;
  const battingCount = Array.isArray(batting) ? batting.length : 0;
  const pitchingCount = Array.isArray(pitching) ? pitching.length : 0;
  return battingCount > 0 || pitchingCount > 0;
}

const CONTROL_TEAM_KEYS = new Set(['home', 'away']);

function normalizeTeamKey(value) {
  if (!value) return null;
  const key = String(value).toLowerCase();
  return CONTROL_TEAM_KEYS.has(key) ? key : null;
}

function createDefaultControlState() {
  return {
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
  };
}

function normalizeControlState(rawControl) {
  const control = createDefaultControlState();
  if (!rawControl || typeof rawControl !== 'object') {
    return control;
  }

  const rawMode = typeof rawControl.mode === 'string' ? rawControl.mode.toLowerCase() : '';
  const mode = rawMode === 'cpu' ? 'cpu' : rawMode === 'auto' ? 'auto' : 'manual';
  control.mode = mode;

  const userTeamKey = normalizeTeamKey(rawControl.user_team ?? rawControl.userTeam);
  const cpuTeamKey = normalizeTeamKey(rawControl.cpu_team ?? rawControl.cpuTeam);

  control.userTeam = mode === 'cpu' ? userTeamKey : null;
  control.cpuTeam = mode === 'cpu' ? cpuTeamKey : null;

  const userName = rawControl.user_team_name ?? rawControl.userTeamName;
  const cpuName = rawControl.cpu_team_name ?? rawControl.cpuTeamName;

  control.userTeamName = typeof userName === 'string' && userName.trim() ? userName.trim() : null;
  control.cpuTeamName = typeof cpuName === 'string' && cpuName.trim() ? cpuName.trim() : null;

  if (mode === 'cpu') {
    control.userIsOffense = Boolean(rawControl.user_is_offense ?? rawControl.userIsOffense ?? false);
    control.userIsDefense = Boolean(rawControl.user_is_defense ?? rawControl.userIsDefense ?? false);
    control.offenseAllowed = Boolean(rawControl.offense_allowed ?? rawControl.offenseAllowed ?? false);
    control.defenseAllowed = Boolean(rawControl.defense_allowed ?? rawControl.defenseAllowed ?? false);
    control.progressAvailable = Boolean(
      rawControl.progress_available ?? rawControl.progressAvailable ?? false,
    );
    const instruction = rawControl.instruction ?? rawControl.control_instruction;
    control.instruction = typeof instruction === 'string' ? instruction.trim() : '';
  }

  return control;
}

const ABILITY_METRIC_CONFIG = {
  k_pct: { mean: 22.8, variation: 0.15, min: 9 },
  bb_pct: { mean: 8.5, variation: 0.15, min: 2.5 },
  hard_pct: { mean: 38.6, variation: 0.05, min: 18 },
  gb_pct: { mean: 44.6, variation: 0.01, min: 20, max: 60 },
  speed: { mean: 100, variation: 0.05, min: 74, max: 130 },
  fielding: { mean: 100, variation: 0.15, min: 40, max: 160 },
  stamina: { mean: 80, variation: 0.15, min: 30, max: 150 },
};

// Allow the app to update coloring baselines from server-side league averages.
export function setLeagueAveragesForColoring(averages) {
  if (!averages || typeof averages !== 'object') return;
  const { k_pct, bb_pct, hard_pct, gb_pct } = averages;
  if (Number.isFinite(Number(k_pct))) ABILITY_METRIC_CONFIG.k_pct.mean = Number(k_pct);
  if (Number.isFinite(Number(bb_pct))) ABILITY_METRIC_CONFIG.bb_pct.mean = Number(bb_pct);
  if (Number.isFinite(Number(hard_pct))) ABILITY_METRIC_CONFIG.hard_pct.mean = Number(hard_pct);
  if (Number.isFinite(Number(gb_pct))) ABILITY_METRIC_CONFIG.gb_pct.mean = Number(gb_pct);
}

export const ABILITY_COLOR_PRESETS = {
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

function computeAbilityMetricStyling(metricKey, displayValue, { invert = false } = {}) {
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
  const mapped = invert ? 1 - clamped : clamped;
  const distance = Math.abs(clamped - 0.5) * 2;

  // Default: low -> green (120), high -> red (0)
  // Invert: low -> red, high -> green
  const hue = 120 - mapped * 120;
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

export function resetAbilityColor(element) {
  if (!element || !element.classList || !element.style) {
    return;
  }
  element.classList.remove('ability-colorized');
  element.style.removeProperty?.('--ability-color');
  element.style.removeProperty?.('--ability-intensity');
  element.style.removeProperty?.('color');
  element.style.removeProperty?.('text-shadow');
  element.style.removeProperty?.('font-weight');
  if (element.dataset) {
    delete element.dataset.abilityMetric;
  }
}

export function applyAbilityColor(
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

  const styling = computeAbilityMetricStyling(metricKey, displayValue, { invert: Boolean(options.invert) });
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
    const rounded = Number(numeric.toFixed(1));
    if (Number.isInteger(rounded)) {
      return `${rounded}`;
    }
    return `${rounded.toFixed(1)}`;
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

    const posTagHtml = renderPositionToken(slot.label, player.pitcher_type, 'pos-tag');
    slotEl.innerHTML = `
      <div class="player-chip">
        ${posTagHtml}
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
    if (player.is_next_batter) {
      tr.classList.add('next');
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
    // Remove any existing dynamic pitcher table
    if (section) {
      const existing = section.querySelector('table.pitcher-table');
      if (existing) existing.remove();
    }
    return;
  }

  // Build or update a table just like the batter lineup style
  let table = section ? section.querySelector('table.pitcher-table') : null;
  if (!table) {
    table = document.createElement('table');
    table.className = 'pitcher-table';
    // Reuse roster table styling
    table.style.width = '100%';
    table.style.borderCollapse = 'collapse';

    const thead = document.createElement('thead');
    const tr = document.createElement('tr');
    ['Pos', '選手', 'ERA', 'IP', 'SO'].forEach((label, idx) => {
      const th = document.createElement('th');
      th.textContent = label;
      if (idx >= 2) th.className = 'stat-col';
      if (idx === 1) th.className = 'player-col';
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    table.appendChild(tbody);

    if (section) {
      // Insert after UL to keep layout
      listEl.insertAdjacentElement('afterend', table);
    }
  }

  const tbody = table.querySelector('tbody');
  if (tbody) tbody.innerHTML = '';

  visiblePitchers.forEach((pitcher) => {
    const tr = document.createElement('tr');
    if (pitcher.is_current) {
      tr.classList.add('active');
    }

    const typeRaw = pitcher.pitcher_type != null ? String(pitcher.pitcher_type).toUpperCase() : 'P';
    const typeLabel = typeRaw === 'SP' || typeRaw === 'RP' ? typeRaw : 'P';
    const positionHtml = renderPositionToken(typeLabel, typeLabel);
    const nameLabel = pitcher.name != null ? String(pitcher.name).trim() : '-';
    const eraText = pitcher.era != null && String(pitcher.era).trim() !== '' ? String(pitcher.era) : '-';
    const ipText = pitcher.ip != null && String(pitcher.ip).trim() !== '' ? String(pitcher.ip) : '-';
    const soRaw = pitcher.so != null ? pitcher.so : pitcher.strikeouts != null ? pitcher.strikeouts : null;
    const soText = soRaw != null && String(soRaw).trim() !== '' ? String(soRaw) : '-';

    tr.innerHTML = `
      <td>${positionHtml}</td>
      <td class="player-name">${escapeHtml(nameLabel)}</td>
      <td class="stat-col">${escapeHtml(eraText)}</td>
      <td class="stat-col">${escapeHtml(ipText)}</td>
      <td class="stat-col">${escapeHtml(soText)}</td>
    `;
    if (tbody) tbody.appendChild(tr);
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

function setMatchupText(element, value, metricKey, extraOptions = undefined) {
  if (!element) return;
  const normalized = normalizeTraitValue(value);
  element.textContent = normalized;

  if (metricKey) {
    applyAbilityColor(
      element,
      metricKey,
      normalized,
      { ...ABILITY_COLOR_PRESETS.matchupPitcher, ...(extraOptions || {}) },
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
        { label: 'K%', key: 'k_pct', value: coalesceTraitValue(player.k_pct, trait?.k_pct), invert: true },
        { label: 'BB%', key: 'bb_pct', value: coalesceTraitValue(player.bb_pct, trait?.bb_pct), invert: false },
        { label: 'Hard%', key: 'hard_pct', value: coalesceTraitValue(player.hard_pct, trait?.hard_pct), invert: false },
        { label: 'GB%', key: 'gb_pct', value: coalesceTraitValue(player.gb_pct, trait?.gb_pct), invert: true },
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
          { ...ABILITY_COLOR_PRESETS.matchupBatter, invert: Boolean(entry.invert) },
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
  setMatchupText(matchupPitcherThrows, formatThrowsLabel(pitcherDisplay.throws));

  // Apply color classes for SP/RP and throws
  if (matchupPitcherType) {
    const t = (pitcherDisplay.pitcher_type || '').toString().toUpperCase();
    matchupPitcherType.classList.remove('sp', 'rp');
    if (t === 'SP') matchupPitcherType.classList.add('sp');
    else if (t === 'RP') matchupPitcherType.classList.add('rp');
  }

  if (matchupPitcherThrows) {
    const th = (pitcherDisplay.throws || '').toString().trim();
    matchupPitcherThrows.classList.remove('throws-colored');
    if (th) matchupPitcherThrows.classList.add('throws-colored');
  }
  setMatchupText(matchupPitcherK, pitcherDisplay.k_pct, 'k_pct');
  setMatchupText(matchupPitcherBB, pitcherDisplay.bb_pct, 'bb_pct', { invert: true });
  setMatchupText(matchupPitcherHard, pitcherDisplay.hard_pct, 'hard_pct', { invert: true });
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
  const throwsLabel = throwsRaw ? escapeHtml(formatThrowsLabel(throwsRaw)) : '';
  const typeClass = (() => {
    const t = String(typeRaw).toUpperCase();
    if (t === 'SP') return 'sp';
    if (t === 'RP') return 'rp';
    return '';
  })();
  const throwsBlock = throwsLabel
    ? `
        <div class="pitcher-meta-block">
          <span class="pitcher-meta-label">投球腕</span>
          <span class="pitcher-meta-value"><span class="pitcher-throws-badge throws-colored">${throwsLabel}</span></span>
        </div>
      `
    : '';

  cardEl.innerHTML = `
    <div class="current-pitcher-header">
      <span class="card-label">現在の投手</span>
      <span class="pitcher-role-badge ${typeClass}">${typeLabel}</span>
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

function formatThrowsLabel(rawThrows) {
  if (!rawThrows) return '';
  const normalized = String(rawThrows).trim().toUpperCase();
  if (!normalized) return '';
  if (normalized === 'L') return '左投';
  if (normalized === 'R') return '右投';
  if (normalized === 'S' || normalized === 'B') return '両投';
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

function updatePinchRunCurrentCard(cardEl, baseOption) {
  if (!cardEl) return;
  if (!baseOption || !baseOption.occupied) {
    cardEl.innerHTML = '<p class="pitcher-card-empty pinch-card-empty">代走対象の走者がいません。</p>';
    return;
  }

  const baseLabel = baseOption.baseLabel || BASE_LABELS[baseOption.index] || '塁';
  const lineupData = baseOption.lineupData || {};
  const nameLabel = escapeHtml(baseOption.runnerName || lineupData.name || '-');
  const positionHtml =
    lineupData.position && lineupData.position !== '-'
      ? renderPositionToken(lineupData.position, lineupData.pitcher_type)
      : '';
  const batsBadge = renderBatsBadge(lineupData.bats);
  const tags = [positionHtml, batsBadge]
    .filter(Boolean)
    .join('');

  const orderValue = Number.isInteger(lineupData.order)
    ? `${lineupData.order}番`
    : lineupData.order
    ? String(lineupData.order)
    : '-';
  const speedValue = baseOption.speedDisplay || lineupData.speed || '-';

  cardEl.innerHTML = `
    <div class="current-pitcher-header current-batter-header">
      <span class="card-label">選択した走者</span>
      <div class="current-batter-tags">
        ${tags || ''}
        <span class="pitcher-throws-badge" aria-label="塁: ${escapeHtml(baseLabel)}">${escapeHtml(baseLabel)}</span>
      </div>
    </div>
    <div class="current-pitcher-body current-batter-body">
      <h4 class="pitcher-name current-batter-name">${nameLabel}</h4>
      <div class="pitcher-meta batter-meta">
        <div class="pitcher-meta-block batter-meta-block">
          <span class="pitcher-meta-label batter-meta-label">塁</span>
          <span class="pitcher-meta-value batter-meta-value">${escapeHtml(baseLabel)}</span>
        </div>
        <div class="pitcher-meta-block batter-meta-block">
          <span class="pitcher-meta-label batter-meta-label">打順</span>
          <span class="pitcher-meta-value batter-meta-value">${escapeHtml(orderValue)}</span>
        </div>
        <div class="pitcher-meta-block batter-meta-block">
          <span class="pitcher-meta-label batter-meta-label">Speed</span>
          <span class="pitcher-meta-value batter-meta-value">${escapeHtml(speedValue)}</span>
        </div>
      </div>
    </div>
  `;
}

function highlightPinchRunBases(gridEl, selectedValue) {
  if (!gridEl) return;
  const normalized = selectedValue != null ? String(selectedValue) : '';
  const cards = gridEl.querySelectorAll('.pinch-run-card');
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

function updatePinchRunBaseGrid(
  gridEl,
  baseOptions,
  selectEl,
  helperEl,
  helperMessage,
  disabledMessage,
  currentCardEl,
) {
  if (!gridEl) return;

  gridEl.innerHTML = '';
  const hasOptions = Array.isArray(baseOptions) && baseOptions.length > 0;
  const selectionDisabled = !selectEl || Boolean(selectEl.disabled);

  if (!hasOptions) {
    const emptyMessage = document.createElement('p');
    emptyMessage.className = 'pitcher-card-empty pinch-card-empty';
    emptyMessage.textContent = '走者がいないため代走を送れません。';
    gridEl.appendChild(emptyMessage);
    highlightPinchRunBases(gridEl, '');
    if (helperEl) {
      helperEl.textContent = helperMessage || '走者が出塁すると代走を選択できます。';
    }
    if (currentCardEl) {
      updatePinchRunCurrentCard(currentCardEl, null);
    }
    return;
  }

  baseOptions.forEach((baseOption) => {
    const optionValue = String(baseOption.index);
    const runnerName = baseOption.runnerName || baseOption.lineupData?.name || '-';
    const positionsList = renderPositionList(
      baseOption.lineupData?.eligible || baseOption.lineupData?.eligible_all,
      baseOption.lineupData?.pitcher_type,
    );
    const nameLabel = escapeHtml(runnerName);
    const baseLabel = escapeHtml(baseOption.baseLabel || BASE_LABELS[baseOption.index] || '塁');
    const speedLabel = escapeHtml(baseOption.speedDisplay || baseOption.lineupData?.speed || '-');
    const orderValue = baseOption.lineupData?.order;
    const orderLabel = Number.isInteger(orderValue) ? `${orderValue}番` : '-';

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'pitcher-card pinch-card pinch-run-card';
    button.dataset.value = optionValue;
    button.setAttribute('role', 'listitem');
    button.setAttribute('aria-pressed', 'false');
    button.title = `${baseLabel}走者: ${runnerName}`;
    button.innerHTML = `
      <div class="pinch-card-top pinch-run-card-top">
        <div class="pinch-card-heading">
          <span class="pinch-run-base-chip">${baseLabel}</span>
          <h4 class="pinch-card-name">${nameLabel}</h4>
        </div>
      </div>
      <div class="pinch-card-stats pinch-run-stats">
        <div class="pinch-stat-block">
          <span class="pinch-stat-label">打順</span>
          <span class="pinch-stat-value">${escapeHtml(orderLabel)}</span>
        </div>
        <div class="pinch-stat-block">
          <span class="pinch-stat-label">Speed</span>
          <span class="pinch-stat-value">${speedLabel}</span>
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
      highlightPinchRunBases(gridEl, optionValue);
      if (currentCardEl) {
        const selectedIndex = Number(optionValue);
        const selectedOption = stateCache.pinchRunContext?.bases?.find(
          (entry) => Number(entry.index) === selectedIndex,
        );
        updatePinchRunCurrentCard(currentCardEl, selectedOption);
      }
    });

    if (selectionDisabled) {
      button.disabled = true;
    }

    gridEl.appendChild(button);
  });

  if (selectEl && !selectEl.dataset.pinchRunListener) {
    selectEl.addEventListener('change', () => {
      highlightPinchRunBases(gridEl, selectEl.value);
      const selectedIndex = Number(selectEl.value);
      setPinchRunSelectedBase(selectedIndex);
      if (currentCardEl) {
        const selectedOption = stateCache.pinchRunContext?.bases?.find(
          (entry) => Number(entry.index) === selectedIndex,
        );
        updatePinchRunCurrentCard(currentCardEl, selectedOption);
      }
    });
    selectEl.dataset.pinchRunListener = 'true';
  }

  highlightPinchRunBases(gridEl, selectEl ? selectEl.value : '');

  if (helperEl) {
    const defaultMessage = 'カードを選択するとここに反映されます。';
    helperEl.textContent = helperMessage || (selectionDisabled ? disabledMessage : defaultMessage);
  }

  if (currentCardEl) {
    const selectedIndex = Number(selectEl ? selectEl.value : getPinchRunSelectedBase());
    const selectedOption = stateCache.pinchRunContext?.bases?.find(
      (entry) => Number(entry.index) === selectedIndex,
    );
    updatePinchRunCurrentCard(currentCardEl, selectedOption);
  }
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
    const throwsLabel = throwsRaw ? escapeHtml(formatThrowsLabel(throwsRaw)) : '';

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'pitcher-card';
    button.dataset.value = optionValue;
    button.setAttribute('aria-pressed', 'false');
    button.setAttribute('role', 'listitem');
    button.title = `${nameRaw} (${typeRaw}${throwsRaw ? `/${throwsRaw}` : ''})`;
    const typeClass = (() => {
      const t = String(typeRaw).toUpperCase();
      if (t === 'SP') return 'sp';
      if (t === 'RP') return 'rp';
      return '';
    })();

    button.innerHTML = `
      <div class="pitcher-card-top">
        <span class="pitcher-role-badge ${typeClass}">${typeLabel}</span>
        ${throwsLabel ? `<span class=\"pitcher-throws-badge throws-colored\" aria-label=\"投球腕\">${throwsLabel}</span>` : ''}
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

function updatePinchOptionGrid(
  gridEl,
  options,
  selectEl,
  helperEl,
  helperMessage,
  optionsConfig = {},
) {
  if (!gridEl) return;

  gridEl.innerHTML = '';
  const hasOptions = Array.isArray(options) && options.length > 0;
  const selectionDisabled = !selectEl || Boolean(selectEl.disabled);
  const config = optionsConfig || {};
  const emptyMessageText = config.emptyMessage || '代打に使える選手がいません。';
  const disabledMessageText = config.disabledMessage || '現在は代打を選択できません。';

  if (!hasOptions) {
    const emptyMessage = document.createElement('p');
    emptyMessage.className = 'pitcher-card-empty pinch-card-empty';
    emptyMessage.textContent = emptyMessageText;
    gridEl.appendChild(emptyMessage);
    highlightPinchCards(gridEl, '');
    if (helperEl) {
      helperEl.textContent = helperMessage || disabledMessageText;
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
        ? disabledMessageText
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
    const headerClasses = [];
    if (currentInningIndex !== null && i === currentInningIndex) {
      headerClasses.push('current-inning');
    }
    const classAttr = headerClasses.length ? ` class="${headerClasses.join(' ')}"` : '';
    html += `<th${classAttr}>${i + 1}</th>`;
  }
  html += '<th class="total-col">R</th><th class="total-col">H</th><th class="total-col">E</th></tr></thead><tbody>';

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
      const cellClasses = [];
      if (currentInningIndex !== null && i === currentInningIndex) {
        cellClasses.push('current-inning');
      }
      const classAttr = cellClasses.length ? ` class="${cellClasses.join(' ')}"` : '';
      row += `<td${classAttr}>${displayValue}</td>`;
    }
    row += `<td class="total-col">${totalRuns}</td><td class="total-col">${totalHits}</td><td class="total-col">${totalErrors}</td></tr>`;
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

  const analyticsRaw =
    (gameState.analytics && typeof gameState.analytics === 'object'
      ? gameState.analytics
      : null) || stateCache.analytics || {};
  const analytics = normalizeAnalyticsState(analyticsRaw);
  const analyticsRunning = Boolean(analytics.running) || Boolean(stateCache.analyticsPending);
  const samples = Number.isFinite(Number(analytics.samples)) ? Number(analytics.samples) : 0;
  const offenseKey = analytics.offense || gameState.offense || null;
  const offenseLabel = offenseKey === 'home' ? 'ホーム' : offenseKey === 'away' ? 'アウェイ' : '--';
  const result = analytics.result && typeof analytics.result === 'object' ? analytics.result : null;

  if (analyticsRunning) {
    setInsightText(elements.insightInningRunExpectancy, '計算中…');
    if (elements.insightOneRunProbability) {
      setInsightText(elements.insightOneRunProbability, '計算中…');
      elements.insightOneRunProbability.dataset.intensity = 'low';
    }
    if (elements.insightWinProbability) {
      setInsightText(elements.insightWinProbability, '計算中…');
      elements.insightWinProbability.dataset.trend = 'neutral';
    }
    if (elements.insightOneRunContext) {
      setInsightText(
        elements.insightOneRunContext,
        `サンプル: ${samples || '--'}試合 / 攻撃: ${offenseLabel}`,
      );
    }
    if (elements.insightWinContext) {
      const score = gameState.score || {};
      const runDiff = numberOrZero(score.home) - numberOrZero(score.away);
      const formattedDiff = runDiff > 0 ? `+${runDiff}` : runDiff < 0 ? `${runDiff}` : '±0';
      setInsightText(
        elements.insightWinContext,
        `得点差: ${formattedDiff} ・サンプル: ${samples || '--'}試合`,
      );
    }
    if (elements.insightProbabilityFill) {
      elements.insightProbabilityFill.style.width = '0%';
    }
    if (elements.insightProbabilityLabel) {
      elements.insightProbabilityLabel.textContent = '計算中…';
    }
    if (elements.insightProbabilityMeter) {
      elements.insightProbabilityMeter.setAttribute('aria-label', '勝利確率 計算中');
    }
    return;
  }

  if (!result) {
    resetInsights();
    if (elements.insightInningContext) {
      setInsightText(
        elements.insightInningContext,
        `アウト: ${outs} / 3 ・走者: ${occupiedBases}`,
      );
    }
    return;
  }

  const expectedRuns = Number.isFinite(Number(result.expected_runs))
    ? Number(result.expected_runs)
    : 0;
  setInsightText(elements.insightInningRunExpectancy, expectedRuns.toFixed(2));
  if (elements.insightInningContext) {
    setInsightText(
      elements.insightInningContext,
      `アウト: ${outs} / 3 ・走者: ${occupiedBases}`,
    );
  }

  if (elements.insightOneRunProbability) {
    const scoringProbability = Math.max(
      0,
      Math.min(1, Number.isFinite(Number(result.score_probability)) ? Number(result.score_probability) : 0),
    );
    const probabilityPercent = Math.round(scoringProbability * 100);
    setInsightText(elements.insightOneRunProbability, `${probabilityPercent}%`);
    let intensity = 'low';
    if (scoringProbability >= 0.67) {
      intensity = 'high';
    } else if (scoringProbability >= 0.34) {
      intensity = 'medium';
    }
    elements.insightOneRunProbability.dataset.intensity = intensity;
    if (elements.insightOneRunContext) {
      setInsightText(
        elements.insightOneRunContext,
        `サンプル: ${samples || '--'}試合 / 攻撃: ${offenseLabel}`,
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

  // Prefer simulation-based win probability when available.
  // Fallback to heuristic only if simulation result is unavailable.
  let winProbability;
  const simWin = result && Number.isFinite(Number(result.home_win_probability))
    ? Number(result.home_win_probability)
    : null;

  if (simWin !== null) {
    winProbability = Math.min(Math.max(simWin, 0), 1);
  } else if (gameState.game_over) {
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
    winProbability = Math.min(Math.max(winProbability, 0), 1);
  }

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
      `得点差: ${formattedDiff} ・サンプル: ${samples || '--'}試合`,
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

export function updateProbabilityPanel(state) {
  const {
    probCurrentExpected,
    probCurrentScoreProb,
    probCurrentWin,
    probCurrentMeta,
    probCurrentSamples,
    probCurrentOffense,
    probCurrentTimestamp,
    probTimelineEmpty,
    probCurrentBody,
    probTimelineBody,
    probTabCurrent,
    probTabTimeline,
    probTimelineChart,
    probTimelineChartLine,
    probTimelineChartArea,
    probTimelineChartDots,
  } = elements;

  // Guard missing DOM (non-game view)
  if (!probCurrentBody || !probTimelineBody) return;

  const analytics = stateCache.analytics || {};
  const result = analytics?.result || null;
  const samples = Number.isFinite(Number(analytics?.samples)) ? Number(analytics.samples) : 0;
  const offenseKey = analytics?.offense || null;
  const offenseLabel = offenseKey === 'home' ? 'ホーム' : offenseKey === 'away' ? 'アウェイ' : '--';

  // Current metrics
  const expectedRuns = Number.isFinite(Number(result?.expected_runs)) ? Number(result.expected_runs) : null;
  const scoreProb = Number.isFinite(Number(result?.score_probability)) ? Number(result.score_probability) : null;
  const winProb = Number.isFinite(Number(result?.home_win_probability)) ? Number(result.home_win_probability) : null;

  if (probCurrentExpected) probCurrentExpected.textContent = expectedRuns != null ? expectedRuns.toFixed(2) : '--';
  if (probCurrentScoreProb) probCurrentScoreProb.textContent = scoreProb != null ? formatPercent01(scoreProb) : '--';
  if (probCurrentWin) probCurrentWin.textContent = winProb != null ? formatPercent01(winProb) : '--';
  if (probCurrentSamples) probCurrentSamples.textContent = samples ? `${samples}試合` : '--';
  if (probCurrentOffense) probCurrentOffense.textContent = offenseLabel;
  if (probCurrentMeta) {
    const ts = analytics?.timestamp ? formatSimulationTimestamp(analytics.timestamp) : '';
    if (probCurrentTimestamp) probCurrentTimestamp.textContent = ts || '--';
    probCurrentMeta.textContent = `サンプル: ${samples || '--'}試合 ・ 攻撃: ${offenseLabel}${ts ? ' ・ 計算時刻: ' + ts : ''}`;
  }

  // Timeline chart
  const history = Array.isArray(stateCache.wpHistory) ? stateCache.wpHistory : [];
  const sanitizedHistory = history
    .map((entry) => ({
      sequence: entry.sequence,
      timestamp: entry.timestamp,
      winProb: Number.isFinite(Number(entry.winProb)) ? Number(entry.winProb) : null,
    }))
    .filter((entry) => entry.winProb != null);

  const hasHistory = sanitizedHistory.length > 0;

  if (probTimelineEmpty) {
    probTimelineEmpty.classList.toggle('hidden', hasHistory);
  }

  if (probTimelineChartLine && probTimelineChartArea && probTimelineChartDots) {
    if (!hasHistory) {
      probTimelineChartLine.setAttribute('d', '');
      probTimelineChartArea.setAttribute('d', '');
      probTimelineChartDots.innerHTML = '';
      if (probTimelineChart) {
        probTimelineChart.setAttribute('aria-hidden', 'true');
      }
    } else {
      if (probTimelineChart) {
        probTimelineChart.removeAttribute('aria-hidden');
      }

      const clamp01 = (value) => Math.min(1, Math.max(0, value));
      const pointCount = sanitizedHistory.length;
      const denominator = pointCount > 1 ? pointCount - 1 : 1;
      const rawPoints = sanitizedHistory.map((entry, index) => {
        const ratio = pointCount > 1 ? index / denominator : 0.5;
        const x = ratio * 100;
        const y = 100 - clamp01(entry.winProb) * 100;
        return { x, y, entry };
      });

      const linePoints = pointCount > 1
        ? rawPoints
        : [
            { x: 0, y: rawPoints[0].y, entry: rawPoints[0].entry },
            { x: 100, y: rawPoints[0].y, entry: rawPoints[0].entry },
          ];

      const linePath = linePoints
        .map((point, idx) => `${idx === 0 ? 'M' : 'L'} ${point.x.toFixed(3)} ${point.y.toFixed(3)}`)
        .join(' ');

      let areaPath = '';
      if (linePoints.length > 0) {
        const startX = linePoints[0].x.toFixed(3);
        const endX = linePoints[linePoints.length - 1].x.toFixed(3);
        const segments = linePoints
          .map((point) => `L ${point.x.toFixed(3)} ${point.y.toFixed(3)}`)
          .join(' ');
        areaPath = `M ${startX} 100 ${segments} L ${endX} 100 Z`;
      }

      probTimelineChartLine.setAttribute('d', linePath);
      probTimelineChartArea.setAttribute('d', areaPath);

      const svgNS = 'http://www.w3.org/2000/svg';
      probTimelineChartDots.innerHTML = '';
      rawPoints.forEach((point, idx) => {
        const circle = document.createElementNS(svgNS, 'circle');
        circle.setAttribute('cx', point.x.toFixed(3));
        circle.setAttribute('cy', point.y.toFixed(3));
        circle.setAttribute('r', pointCount > 30 ? '0.8' : '1.4');
        circle.dataset.sequence = String(sanitizedHistory[idx].sequence ?? '');
        circle.dataset.winProb = String(Math.round(clamp01(sanitizedHistory[idx].winProb) * 100));
        probTimelineChartDots.appendChild(circle);
      });
    }
  }

  // Ensure tab classes reflect state
  const active = stateCache.probTab === 'timeline' ? 'timeline' : 'current';
  if (probTabCurrent) probTabCurrent.classList.toggle('active', active === 'current');
  if (probTabTimeline) probTabTimeline.classList.toggle('active', active === 'timeline');
  probCurrentBody.classList.toggle('hidden', active !== 'current');
  probTimelineBody.classList.toggle('hidden', active !== 'timeline');
}

function updateStrategyControls(gameState, teams) {
  const {
    pinchPlayer,
    pinchButton,
    pinchCurrentCard,
    pinchOptionGrid,
    pinchSelectHelper,
    pinchRunBase,
    pinchRunBaseGrid,
    pinchRunBaseHelper,
    pinchRunPlayer,
    pinchRunOptionGrid,
    pinchRunSelectHelper,
    pinchRunButton,
    pinchRunCurrentCard,
    openOffenseButton,
    offensePinchMenuButton,
    offensePinchRunMenuButton,
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
  const controlState = stateCache.gameControl || createDefaultControlState();
  const analyticsState =
    (gameState && typeof gameState.analytics === 'object' ? gameState.analytics : null)
      || stateCache.analytics
      || {};
  const analyticsRunning = Boolean(analyticsState.running) || Boolean(stateCache.analyticsPending);
  const controlMode = controlState.mode || 'manual';
  const isCpuMode = controlMode === 'cpu';
  const isAutoMode = controlMode === 'auto';
  const offenseControlsAllowed = isAutoMode
    ? false
    : !isCpuMode || Boolean(controlState.offenseAllowed);
  const defenseControlsAllowed = isAutoMode
    ? false
    : !isCpuMode || Boolean(controlState.defenseAllowed);
  const offenseControlsEnabled =
    isActive && !isGameOver && offenseControlsAllowed && !analyticsRunning;
  const defenseControlsEnabled =
    isActive && !isGameOver && defenseControlsAllowed && !analyticsRunning;

  const offenseTeam = gameState.offense ? teams[gameState.offense] : null;
  const offenseLineup = offenseTeam?.lineup || [];
  const offenseBench = offenseTeam?.bench || [];
  const currentBatter = offenseLineup.find((player) => player.is_current_batter) || null;

  stateCache.currentBatterIndex =
    currentBatter && Number.isInteger(currentBatter.index) ? currentBatter.index : null;

  const basesState = Array.isArray(gameState.bases) ? gameState.bases : [];
  const previousSelectedBase = getPinchRunSelectedBase();
  const baseOptions = [];
  const availableBaseOptions = [];

  basesState.forEach((base, index) => {
    const occupied = Boolean(base && base.occupied);
    const baseLabel = BASE_LABELS[index] || `${index + 1}塁`;
    let lineupIndex = Number.isInteger(base?.lineup_index) ? base.lineup_index : null;
    let lineupEntry = null;

    if (Number.isInteger(lineupIndex)) {
      lineupEntry = offenseLineup.find(
        (player) => Number.isInteger(player.index) && player.index === lineupIndex,
      );
    }

    const runnerNameRaw = typeof base?.runner === 'string' ? base.runner : null;
    if (!lineupEntry && runnerNameRaw) {
      lineupEntry = offenseLineup.find((player) => player.name === runnerNameRaw) || null;
      if (lineupEntry && Number.isInteger(lineupEntry.index)) {
        lineupIndex = lineupEntry.index;
      }
    }

    const speedDisplay =
      (base && typeof base.speed_display === 'string' && base.speed_display.trim() !== '')
        ? base.speed_display
        : lineupEntry?.speed || (base?.speed != null ? String(base.speed) : '-');

    const option = {
      index,
      occupied,
      baseLabel,
      runnerName: runnerNameRaw || lineupEntry?.name || null,
      speedDisplay,
      lineupIndex: Number.isInteger(lineupIndex) ? lineupIndex : null,
      lineupData: lineupEntry || null,
    };

    baseOptions.push(option);

    if (occupied && Number.isInteger(option.lineupIndex) && option.lineupData) {
      availableBaseOptions.push(option);
    }
  });

  let selectedBaseIndex = Number.isInteger(previousSelectedBase) ? previousSelectedBase : null;
  if (!availableBaseOptions.some((option) => option.index === selectedBaseIndex)) {
    selectedBaseIndex = availableBaseOptions.length ? availableBaseOptions[0].index : null;
  }

  setPinchRunContext(
    baseOptions,
    availableBaseOptions.map((option) => option.index),
    selectedBaseIndex,
  );

  const sanitizedBaseIndex = stateCache.pinchRunContext?.selectedBaseIndex ?? null;
  const selectedBaseOption =
    Number.isInteger(sanitizedBaseIndex)
      ? baseOptions.find((option) => option.index === sanitizedBaseIndex)
      : null;
  if (!Number.isInteger(previousSelectedBase) || previousSelectedBase !== sanitizedBaseIndex) {
    setPinchRunSelectedBase(sanitizedBaseIndex);
  }

  if (pinchCurrentCard) {
    updateCurrentBatterCard(pinchCurrentCard, currentBatter);
  }

  const canPinch =
    offenseControlsEnabled && Boolean(currentBatter) && offenseBench.length > 0;
  const canPinchRun =
    offenseControlsEnabled && offenseBench.length > 0 && availableBaseOptions.length > 0;

  if (!offenseControlsAllowed) {
    hideOffenseMenu();
  }

  if (pinchPlayer) {
    let benchPlaceholder;
    if (!offenseControlsEnabled) {
      benchPlaceholder = 'CPUが攻撃中のため操作できません。';
    } else if (!currentBatter) {
      benchPlaceholder = '現在の打者が見つかりません';
    } else if (offenseBench.length) {
      benchPlaceholder = 'カードまたはリストから選択';
    } else {
      benchPlaceholder = '選択可能な選手がいません';
    }

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

  if (pinchRunBase) {
    let basePlaceholder;
    if (!offenseControlsEnabled) {
      basePlaceholder = 'CPUが攻撃中のため操作できません。';
    } else if (!availableBaseOptions.length) {
      basePlaceholder = '走者がいないため選択できません';
    } else {
      basePlaceholder = 'カードまたはリストから選択';
    }

    populateSelectSimple(
      pinchRunBase,
      availableBaseOptions.map((option) => ({
        value: option.index,
        label: `${option.baseLabel} ${option.runnerName ?? '-'}`.trim(),
      })),
      basePlaceholder,
    );

    pinchRunBase.disabled = !canPinchRun;
    if (Number.isInteger(sanitizedBaseIndex) && canPinchRun) {
      pinchRunBase.value = String(sanitizedBaseIndex);
    } else if (!canPinchRun) {
      pinchRunBase.value = '';
    }
  }

  if (pinchRunPlayer) {
    let benchPlaceholder;
    if (!offenseControlsEnabled) {
      benchPlaceholder = 'CPUが攻撃中のため操作できません。';
    } else if (!availableBaseOptions.length) {
      benchPlaceholder = '走者がいないため選択できません';
    } else if (offenseBench.length) {
      benchPlaceholder = 'カードまたはリストから選択';
    } else {
      benchPlaceholder = '代走に使える選手がいません';
    }

    populateSelectSimple(
      pinchRunPlayer,
      offenseBench.map((player) => ({
        value: player.index,
        label: `${player.name} (AVG ${player.avg ?? '-'}, HR ${player.hr ?? '-'})`,
      })),
      benchPlaceholder,
    );

    pinchRunPlayer.disabled = !canPinchRun;
    if (!canPinchRun) {
      pinchRunPlayer.value = '';
    }
  }

  if (pinchRunButton) {
    pinchRunButton.disabled = !canPinchRun;
    pinchRunButton.textContent = isGameOver ? 'Game Over' : '代走を送る';
  }

  if (offensePinchRunMenuButton) {
    offensePinchRunMenuButton.disabled = !canPinchRun;
    offensePinchRunMenuButton.textContent = isGameOver ? 'Game Over' : '代走戦略';
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
    } else if (!offenseControlsEnabled) {
      pinchHelperMessage = 'CPUが攻撃中のため代打は行えません。';
    }

    updatePinchOptionGrid(pinchOptionGrid, offenseBench, pinchPlayer, pinchSelectHelper, pinchHelperMessage);
  }

  if (pinchRunOptionGrid) {
    let pinchRunHelperMessage = 'カードを選択するとここに反映されます。';
    if (!isActive) {
      pinchRunHelperMessage = '試合開始後に代走が選択できます。';
    } else if (isGameOver) {
      pinchRunHelperMessage = '試合終了のため代走は行えません。';
    } else if (!availableBaseOptions.length) {
      pinchRunHelperMessage = '走者がいないため代走を送れません。';
    } else if (!offenseBench.length) {
      pinchRunHelperMessage = '代走に使える選手がいません。';
    } else if (!offenseControlsEnabled) {
      pinchRunHelperMessage = 'CPUが攻撃中のため代走は行えません。';
    }

    updatePinchOptionGrid(
      pinchRunOptionGrid,
      offenseBench,
      pinchRunPlayer,
      pinchRunSelectHelper,
      pinchRunHelperMessage,
      {
        emptyMessage: '代走に使える選手がいません。',
        disabledMessage: offenseControlsEnabled
          ? '現在は代走を選択できません。'
          : 'CPUが攻撃中のため代走は選択できません。',
      },
    );
  }

  if (pinchRunBaseGrid) {
    let baseHelperMessage = 'カードを選択するとここに反映されます。';
    if (!isActive) {
      baseHelperMessage = '試合開始後に代走が選択できます。';
    } else if (isGameOver) {
      baseHelperMessage = '試合終了のため代走は行えません。';
    } else if (!availableBaseOptions.length) {
      baseHelperMessage = '走者がいないため代走を送れません。';
    } else if (!offenseControlsEnabled) {
      baseHelperMessage = 'CPUが攻撃中のため代走は指定できません。';
    }

    updatePinchRunBaseGrid(
      pinchRunBaseGrid,
      availableBaseOptions,
      pinchRunBase,
      pinchRunBaseHelper,
      baseHelperMessage,
      offenseControlsEnabled
        ? '現在は代走を選択できません。'
        : 'CPUが攻撃中のため代走は指定できません。',
      pinchRunCurrentCard,
    );
  } else if (pinchRunCurrentCard) {
    updatePinchRunCurrentCard(pinchRunCurrentCard, selectedBaseOption);
  }

  if (openOffenseButton) {
    openOffenseButton.disabled = !offenseControlsEnabled;
    const offenseLabel = !isGameOver && !offenseControlsAllowed ? '攻撃戦略 (CPU)' : '攻撃戦略';
    openOffenseButton.textContent = isGameOver ? 'Game Over' : offenseLabel;
  }

  const defenseTeam = gameState.defense ? teams[gameState.defense] : null;
  const defenseLineup = defenseTeam?.lineup || [];
  const defenseBenchPlayers = defenseTeam?.bench || [];
  const pitcherOptions = defenseTeam?.pitcher_options || [];

  const currentPitcher =
    (defenseTeam?.pitchers || []).find((pitcher) => pitcher && pitcher.is_current) || null;

  updateCurrentPitcherCard(currentPitcherCard, currentPitcher);

  resetDefenseSelectionsIfUnavailable(defenseLineup, defenseBenchPlayers);

  if (!defenseControlsAllowed) {
    hideDefenseMenu();
  }

  const canDefenseSub =
    defenseControlsEnabled && defenseLineup.length > 0 && defenseBenchPlayers.length > 0;

  stateCache.defenseContext.canSub = canDefenseSub;
  renderDefensePanel(defenseTeam, gameState);
  updateDefenseSelectionInfo();

  if (openDefenseButton) {
    openDefenseButton.disabled = !defenseControlsEnabled;
    const defenseLabel = !isGameOver && !defenseControlsAllowed ? '守備戦略 (CPU)' : '守備戦略';
    openDefenseButton.textContent = isGameOver ? 'Game Over' : defenseLabel;
  }
  if (!defenseControlsEnabled) {
    updateDefenseBenchAvailability();
    applyDefenseSelectionHighlights();
  }
  if (defenseSubMenuButton) {
    defenseSubMenuButton.disabled = !canDefenseSub;
    const defenseSubLabel = !isGameOver && !defenseControlsAllowed ? '守備交代 (CPU)' : '守備交代';
    defenseSubMenuButton.textContent = isGameOver ? 'Game Over' : defenseSubLabel;
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

    const canChangePitcher = defenseControlsEnabled && pitcherOptions.length > 0;
    pitcherButton.disabled = !canChangePitcher;
    pitcherSelect.disabled = !canChangePitcher;
    if (!canChangePitcher) {
      pitcherSelect.value = '';
    }

    updatePitcherOptionGrid(pitcherOptionGrid, pitcherOptions, pitcherSelect, pitcherSelectHelper);

    pitcherButton.textContent = isGameOver ? 'Game Over' : '投手交代';

    if (pitcherMenuButton) {
      pitcherMenuButton.disabled = !canChangePitcher;
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
  const controlInfo = stateCache.gameControl || createDefaultControlState();
  const analyticsState =
    (gameState && typeof gameState.analytics === 'object' ? gameState.analytics : null)
      || stateCache.analytics
      || {};
  const analyticsRunning = Boolean(analyticsState.running) || Boolean(stateCache.analyticsPending);
  setInsightsVisibility(isActiveGame && showGameView);

  if (!isActiveGame) {
    resetPlayAnimation();
    resetFieldResultDisplay();
    updateScoreboard(gameState || {}, teams || {});
    updateOutsIndicator(gameState?.outs ?? 0);
    elements.actionWarning.textContent = '';
    elements.swingButton.disabled = true;
    elements.buntButton.disabled = true;
    if (elements.squeezeButton) {
      elements.squeezeButton.disabled = true;
      elements.squeezeButton.textContent = 'スクイズ';
      elements.squeezeButton.classList.remove('hidden');
    }
    if (elements.stealButton) {
      elements.stealButton.disabled = true;
      elements.stealButton.textContent = '盗塁';
    }
    if (elements.progressButton) {
      elements.progressButton.classList.add('hidden');
      elements.progressButton.disabled = true;
    }
    elements.swingButton.classList.remove('hidden');
    elements.buntButton.classList.remove('hidden');
    if (elements.squeezeButton) {
      elements.squeezeButton.classList.remove('hidden');
    }
    if (elements.stealButton) {
      elements.stealButton.classList.remove('hidden');
    }
    if (elements.batterAlignment) {
      elements.batterAlignment.innerHTML = '';
      elements.batterAlignment.classList.add('hidden');
      elements.batterAlignment.setAttribute('aria-hidden', 'true');
    }
    updateRosters(elements.offenseRoster, []);
    updateRosters(elements.defenseRoster, []);
    if (elements.offenseTeamName) elements.offenseTeamName.textContent = '';
    if (elements.defenseTeamName) elements.defenseTeamName.textContent = '';
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
  if (Array.isArray(gameState.overlays)) {
    try {
      updateOverlayEvents(gameState.overlays);
    } catch (_) {
      // no-op if overlay handler not loaded
    }
  }

  const offenseTeam = gameState.offense ? teams[gameState.offense] : null;
  const defenseTeam = gameState.defense ? teams[gameState.defense] : null;

  // Update roster section headings with team names
  if (elements.offenseTeamName) {
    elements.offenseTeamName.textContent = offenseTeam?.name ? String(offenseTeam.name) : '';
  }
  if (elements.defenseTeamName) {
    elements.defenseTeamName.textContent = defenseTeam?.name ? String(defenseTeam.name) : '';
  }

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
    if (elements.squeezeButton) {
      elements.squeezeButton.disabled = true;
      elements.squeezeButton.textContent = 'Game Over';
    }
    if (elements.stealButton) {
      elements.stealButton.disabled = true;
      elements.stealButton.textContent = 'Game Over';
    }
    if (elements.progressButton) {
      elements.progressButton.classList.add('hidden');
      elements.progressButton.disabled = true;
    }
    elements.actionWarning.textContent = 'ゲーム終了 - 新しい試合を開始するか、タイトルに戻ってください';
  } else {
    elements.swingButton.disabled = !(gameState.actions?.swing && !analyticsRunning);
    const showBunt = Boolean(gameState.actions?.show_bunt ?? true);
    const buntAllowed = Boolean(gameState.actions?.bunt && !analyticsRunning);
    elements.buntButton.disabled = !(buntAllowed && showBunt);
    elements.swingButton.textContent = '通常打撃';
    elements.buntButton.textContent = 'バント';
    if (elements.squeezeButton) {
      const squeezeAllowed = Boolean(gameState.actions?.squeeze && !analyticsRunning);
      elements.squeezeButton.disabled = !squeezeAllowed;
      elements.squeezeButton.textContent = 'スクイズ';
    }
    if (elements.stealButton) {
      elements.stealButton.disabled = !(gameState.actions?.steal && !analyticsRunning);
      elements.stealButton.textContent = '盗塁';
    }
    const progressAllowed = Boolean(gameState.actions?.progress && !analyticsRunning);
    if (elements.progressButton) {
      elements.progressButton.classList.toggle('hidden', !progressAllowed);
      elements.progressButton.disabled = !progressAllowed;
    }
    const controlMode = controlInfo.mode;
    const hideOffenseActions =
      controlMode === 'auto' || (progressAllowed && controlMode === 'cpu');
    elements.swingButton.classList.toggle('hidden', hideOffenseActions);
    const buntVisible = showBunt && !hideOffenseActions;
    elements.buntButton.classList.toggle('hidden', !buntVisible);
    if (elements.squeezeButton) {
      const squeezeVisible = Boolean(gameState.actions?.squeeze) && !hideOffenseActions;
      elements.squeezeButton.classList.toggle('hidden', !squeezeVisible);
    }
    if (elements.stealButton) {
      elements.stealButton.classList.toggle('hidden', hideOffenseActions);
    }
    if (!hideOffenseActions && elements.progressButton) {
      elements.progressButton.classList.add('hidden');
    }
    const actionMessages = [];
    if (gameState.action_block_reason) {
      actionMessages.push(gameState.action_block_reason);
    }
    if (controlInfo.instruction) {
      actionMessages.push(controlInfo.instruction);
    }
    if (analyticsRunning) {
      actionMessages.push('統計計算中です。CPU解析が完了するまでお待ちください。');
    }
    elements.actionWarning.textContent = actionMessages.join(' / ');
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

function renderTitleLineup(teamKey, teamData, enabled) {
  const editorContainer = document.querySelector(`[data-title-lineup="${teamKey}"]`);
  const summaryContainer = document.querySelector(`[data-title-lineup-summary="${teamKey}"]`);
  const note = document.querySelector(`[data-title-lineup-note="${teamKey}"]`);
  const applyButton = document.querySelector(`[data-action="apply-lineup"][data-team="${teamKey}"]`);

  if (summaryContainer) {
    summaryContainer.innerHTML = '';
  }
  if (editorContainer) {
    editorContainer.innerHTML = '';
  }
  if (note) {
    note.textContent = '';
  }
  if (applyButton) {
    applyButton.disabled = true;
  }

  if (!enabled) {
    if (summaryContainer) {
      const message = document.createElement('p');
      message.className = 'title-summary-empty';
      message.textContent = 'チームが読み込まれていません。';
      summaryContainer.appendChild(message);
    }
    if (editorContainer) {
      const message = document.createElement('p');
      message.className = 'title-bench-empty';
      message.textContent = 'チームが読み込まれていません。';
      editorContainer.appendChild(message);
    }
    return;
  }

  const plan = ensureTitleLineupPlan(teamKey, teamData, enabled);
  if (!plan || !plan.lineup.length) {
    if (summaryContainer) {
      const message = document.createElement('p');
      message.className = 'title-summary-empty';
      message.textContent = 'スタメン情報がありません。';
      summaryContainer.appendChild(message);
    }
    if (note) {
      note.textContent = 'スタメン情報が不足しています。チーム編成を確認してください。';
    }
    if (editorContainer) {
      const message = document.createElement('p');
      message.className = 'title-bench-empty';
      message.textContent = 'スタメン情報がありません。';
      editorContainer.appendChild(message);
    }
    if (applyButton) {
      applyButton.disabled = true;
    }
    return;
  }

  const formatValue = (value) => {
    if (value === null || value === undefined) return null;
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) return null;
      return String(value);
    }
    const text = String(value).trim();
    if (!text || text === '-' || text === '--') return null;
    return text;
  };

  const createAbilityChip = (label, value) => {
    const formatted = formatValue(value);
    if (!formatted) return null;
    const chip = document.createElement('span');
    chip.className = 'title-lineup-chip';
    chip.textContent = `${label} ${formatted}`;
    return chip;
  };

  const selection = getTitleLineupSelection();
  const selectionMatchesTeam = selection.team === plan.teamKey;
  const invalidAssignments = getTitleLineupInvalidAssignments(plan);
  const invalidIndexes = new Set(invalidAssignments.map((entry) => entry.index));
  const editorOpen = isTitleEditorOpen(teamKey);
  const activeTab = getTitleEditorTab(teamKey);
  const editingLineup = editorOpen && activeTab === 'lineup';
  let helperMessage = '';

  let lineupEligibility = { eligible: new Set(), ineligible: new Set() };

  if (selectionMatchesTeam && selection.type === 'lineup') {
    const slot = plan.lineup[selection.index];
    if (slot) {
      helperMessage = `${slot.order}番 ${slot.slotPositionLabel || slot.slotPositionKey} の交代先をベンチから選択してください。`;
    }
  } else if (selectionMatchesTeam && selection.type === 'bench') {
    lineupEligibility = getLineupEligibilityForBenchPlayer(plan, selection.index);
    const benchPlayer = plan.bench[selection.index];
    if (benchPlayer) {
      helperMessage = `${benchPlayer.name} を出場させる守備位置を選択してください。`;
    }
  }

  if (summaryContainer) {
    plan.lineup.forEach((slot) => {
      if (!slot) return;
      const row = document.createElement('div');
      row.className = 'title-lineup-summary-row';
      if (invalidIndexes.has(slot.index)) {
        row.classList.add('invalid');
      }
      if (
        editingLineup &&
        selectionMatchesTeam &&
        selection.type === 'lineup' &&
        selection.index === slot.index
      ) {
        row.classList.add('selected');
      }

      const order = document.createElement('span');
      order.className = 'title-lineup-summary-order';
      order.textContent = String(slot.order);
      row.appendChild(order);

      const main = document.createElement('div');
      main.className = 'title-lineup-summary-main';

      const header = document.createElement('div');
      header.className = 'title-lineup-summary-header';

      const position = document.createElement('span');
      position.className = 'title-lineup-summary-position';
      position.textContent = slot.slotPositionLabel || slot.slotPositionKey || '-';
      header.appendChild(position);

      const name = document.createElement('span');
      name.className = 'title-lineup-summary-name';
      name.textContent = slot.player?.name || '選手未設定';
      header.appendChild(name);

      if (slot.player?.bats) {
        const bats = document.createElement('span');
        bats.className = 'title-lineup-summary-bats';
        bats.textContent = slot.player.bats;
        header.appendChild(bats);
      }

      main.appendChild(header);

      const abilities = document.createElement('div');
      abilities.className = 'title-lineup-summary-abilities';

      if (slot.player) {
        const chips = [
          createAbilityChip('AVG', slot.player.avg),
          createAbilityChip('HR', slot.player.hr),
          createAbilityChip('BB%', slot.player.bb_pct),
          createAbilityChip('K%', slot.player.k_pct),
          createAbilityChip('走力', slot.player.speed),
          createAbilityChip('守備', slot.player.fielding_rating),
        ].filter(Boolean);

        if (chips.length) {
          chips.forEach((chip) => abilities.appendChild(chip));
        } else {
          abilities.classList.add('empty');
          abilities.textContent = '能力データがありません。';
        }
      } else {
        abilities.classList.add('empty');
        abilities.textContent = '選手を割り当ててください。';
      }

      main.appendChild(abilities);
      row.appendChild(main);
      summaryContainer.appendChild(row);
    });
  }

  if (editorContainer) {
    plan.lineup.forEach((slot) => {
      if (!slot) return;
      const row = document.createElement('div');
      row.className = 'title-lineup-row';
      row.dataset.team = plan.teamKey;
      row.dataset.titleRole = 'lineup';
      row.dataset.index = String(slot.index);

      if (selectionMatchesTeam && selection.type === 'lineup' && selection.index === slot.index) {
        row.classList.add('selected');
      }
      if (invalidIndexes.has(slot.index)) {
        row.classList.add('invalid');
      }
      if (selectionMatchesTeam && selection.type === 'bench') {
        if (lineupEligibility.eligible.has(slot.index)) {
          row.classList.add('eligible');
        } else if (lineupEligibility.ineligible.has(slot.index)) {
          row.classList.add('ineligible');
        }
      }

      const order = document.createElement('div');
      order.className = 'title-lineup-order';
      order.textContent = `${slot.order}`;

      const positionLabel = slot.slotPositionLabel || slot.slotPositionKey || '-';

      const positionButton = document.createElement('button');
      positionButton.type = 'button';
      positionButton.className = 'title-lineup-position';
      positionButton.textContent = positionLabel;
      positionButton.dataset.team = plan.teamKey;
      positionButton.dataset.titleRole = 'lineup';
      positionButton.dataset.index = String(slot.index);
      positionButton.dataset.lineupField = 'position';
      positionButton.setAttribute('aria-label', `${slot.order}番 ${positionLabel} の守備位置を選択`);
      positionButton.setAttribute(
        'aria-pressed',
        selectionMatchesTeam &&
          selection.type === 'lineup' &&
          selection.index === slot.index &&
          (!selection.field || selection.field === 'position')
          ? 'true'
          : 'false',
      );

      const playerButton = document.createElement('button');
      playerButton.type = 'button';
      playerButton.className = 'title-lineup-player';
      playerButton.dataset.team = plan.teamKey;
      playerButton.dataset.titleRole = 'lineup';
      playerButton.dataset.index = String(slot.index);
      playerButton.dataset.lineupField = 'player';
      playerButton.setAttribute(
        'aria-pressed',
        selectionMatchesTeam &&
          selection.type === 'lineup' &&
          selection.index === slot.index &&
          (!selection.field || selection.field === 'player')
          ? 'true'
          : 'false',
      );

      const name = document.createElement('span');
      name.className = 'title-lineup-player-name';
      name.textContent = slot.player?.name || '選手を選択';
      playerButton.appendChild(name);

      if (!slot.player) {
        playerButton.classList.add('empty');
      } else {
        const metaParts = [];
        if (slot.player.bats) {
          metaParts.push(`Bats ${slot.player.bats}`);
        }
        if (slot.player.fielding_rating) {
          metaParts.push(`守備 ${slot.player.fielding_rating}`);
        }
        if (slot.player.avg) {
          metaParts.push(`AVG ${slot.player.avg}`);
        }
        if (metaParts.length) {
          const meta = document.createElement('span');
          meta.className = 'title-lineup-player-meta';
          meta.textContent = metaParts.join(' · ');
          playerButton.appendChild(meta);
        }
      }

      row.appendChild(order);
      row.appendChild(positionButton);
      row.appendChild(playerButton);
      editorContainer.appendChild(row);
    });

    if (applyButton) {
      const hasEmptySlot = plan.lineup.some((slot) => !slot.player || !slot.player.name);
      applyButton.disabled = hasEmptySlot;
    }
  }

  if (note) {
    if (helperMessage) {
      note.textContent = helperMessage;
    } else if (invalidAssignments.length) {
      const labels = invalidAssignments
        .map((entry) => `${entry.slot.slotPositionLabel || entry.positionKey}`)
        .join('、');
      note.textContent = `${labels} を守れる選手が見つかりません。`;
    } else {
      note.textContent = '守備位置または選手名をクリックして、ベンチから交代させる選手を選択できます。';
    }
  }
}

function renderTitleBench(teamKey, teamData, enabled) {
  const container = document.querySelector(`[data-title-bench="${teamKey}"]`);
  if (!container) return;

  container.innerHTML = '';
  if (!enabled) {
    const message = document.createElement('p');
    message.className = 'title-bench-empty';
    message.textContent = 'チームが読み込まれていません。';
    container.appendChild(message);
    return;
  }

  const plan = ensureTitleLineupPlan(teamKey, teamData, enabled);
  if (!plan) {
    const message = document.createElement('p');
    message.className = 'title-bench-empty';
    message.textContent = 'ベンチ情報が読み込めませんでした。';
    container.appendChild(message);
    return;
  }

  const benchPlayers = plan.bench || [];
  const selection = getTitleLineupSelection();
  const selectionMatchesTeam = selection.team === plan.teamKey;
  let benchHighlight = { eligible: new Set(), ineligible: new Set() };

  if (selectionMatchesTeam && selection.type === 'lineup') {
    const slot = plan.lineup[selection.index];
    if (slot) {
      benchHighlight = getBenchEligibilityForPosition(plan, slot.slotPositionKey);
    }
  }

  if (!benchPlayers.length) {
    const message = document.createElement('p');
    message.className = 'title-bench-empty';
    message.textContent = 'ベンチ登録がありません。';
    container.appendChild(message);
    return;
  }

  benchPlayers.forEach((player, index) => {
    if (!player || !player.name) return;
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'title-bench-chip';
    chip.dataset.team = plan.teamKey;
    chip.dataset.titleRole = 'bench';
    chip.dataset.index = String(index);
    let label = player.name;
    const metaParts = [];
    if (player.pitcher_type && player.pitcher_type !== 'P') {
      metaParts.push(player.pitcher_type);
    } else if (player.position && player.position !== '-' && player.position !== 'P') {
      metaParts.push(player.position);
    }
    if (player.bats) {
      metaParts.push(player.bats);
    }
    if (metaParts.length) {
      label += ` (${metaParts.join('/')})`;
    }
    chip.textContent = label;

    if (selectionMatchesTeam && selection.type === 'bench' && selection.index === index) {
      chip.classList.add('selected');
    }
    if (benchHighlight.eligible.has(index)) {
      chip.classList.add('eligible');
    } else if (benchHighlight.ineligible.has(index)) {
      chip.classList.add('ineligible');
    }

    container.appendChild(chip);
  });
}

function renderTitlePitcher(teamKey, teamData, enabled) {
  const list = document.querySelector(`[data-title-pitcher-list="${teamKey}"]`);
  const summary = document.querySelector(`[data-title-starter="${teamKey}"]`);

  if (summary) {
    summary.innerHTML = '';
  }
  if (list) {
    list.innerHTML = '';
  }

  const appendMessage = (container, className, text) => {
    if (!container) return;
    const message = document.createElement('p');
    message.className = className;
    message.textContent = text;
    container.appendChild(message);
  };

  if (!enabled) {
    clearTitlePitcherSelection(teamKey);
    if (summary) {
      appendMessage(summary, 'title-starter-empty', 'チームが読み込まれていません。');
    }
    if (list) {
      appendMessage(list, 'title-pitcher-empty', '投手データがありません。');
    }
    return;
  }

  const pitchers = Array.isArray(teamData?.pitchers) ? teamData.pitchers : [];
  const traitRows = Array.isArray(teamData?.traits?.pitching) ? teamData.traits.pitching : [];
  const traitMap = new Map(
    traitRows.map((row) => [String(row?.name || '').trim(), row]),
  );

  const hasPitchers = pitchers.length > 0;
  const currentPitcher = pitchers.find((pitcher) => pitcher?.is_current) || null;

  let selectedName = getTitlePitcherSelection(teamKey) || '';
  if (selectedName) {
    const exists = pitchers.some((pitcher) => String(pitcher?.name || '').trim() === selectedName);
    if (!exists) {
      selectedName = '';
    }
  }
  if (!selectedName && currentPitcher?.name) {
    selectedName = String(currentPitcher.name || '').trim();
  }
  setTitlePitcherSelection(teamKey, selectedName);

  const formatAbilityValue = (value) => {
    if (value === null || value === undefined) return null;
    const text = String(value).trim();
    if (!text || text === '-' || text === '--') return null;
    return text;
  };

  const formatStatValue = (value) => {
    if (value === null || value === undefined || value === '') return null;
    return String(value);
  };

  const createChip = (label, value, className = 'title-pitcher-chip') => {
    const formatted = typeof value === 'function' ? value() : value;
    if (!formatted) return null;
    const chip = document.createElement('span');
    chip.className = className;
    chip.textContent = `${label} ${formatted}`;
    return chip;
  };

  if (summary) {
    if (!currentPitcher) {
      appendMessage(summary, 'title-starter-empty', '先発が設定されていません。');
    } else {
      const normalizedName = String(currentPitcher.name || '').trim();
      const traits = traitMap.get(normalizedName);

      const header = document.createElement('div');
      header.className = 'title-starter-header';

      const name = document.createElement('div');
      name.className = 'title-starter-name';
      name.textContent = normalizedName || '先発投手未設定';
      header.appendChild(name);

      const metaParts = [];
      if (currentPitcher.pitcher_type) {
        metaParts.push(currentPitcher.pitcher_type);
      }
      if (currentPitcher.throws) {
        metaParts.push(currentPitcher.throws);
      }
      if (currentPitcher.stamina != null && currentPitcher.stamina !== '') {
        metaParts.push(`St ${currentPitcher.stamina}`);
      }
      if (metaParts.length) {
        const meta = document.createElement('div');
        meta.className = 'title-starter-meta';
        meta.textContent = metaParts.join(' / ');
        header.appendChild(meta);
      }

      summary.appendChild(header);

      const statsRow = document.createElement('div');
      statsRow.className = 'title-starter-stats';

      const eraChip = createChip('ERA', () => formatStatValue(currentPitcher.era));
      const ipChip = createChip('IP', () => formatStatValue(currentPitcher.ip));
      const soChip = createChip('SO', () => formatStatValue(currentPitcher.so));
      [eraChip, ipChip, soChip].forEach((chip) => {
        if (chip) statsRow.appendChild(chip);
      });

      if (!statsRow.children.length) {
        const fallback = document.createElement('span');
        fallback.className = 'title-pitcher-chip empty';
        fallback.textContent = '投球成績データなし';
        statsRow.appendChild(fallback);
      }

      summary.appendChild(statsRow);

      const abilityRow = document.createElement('div');
      abilityRow.className = 'title-starter-abilities';

      const abilityChips = [
        createChip('K%', () => formatAbilityValue(traits?.k_pct)),
        createChip('BB%', () => formatAbilityValue(traits?.bb_pct)),
        createChip('強打%', () => formatAbilityValue(traits?.hard_pct)),
        createChip('ゴロ%', () => formatAbilityValue(traits?.gb_pct)),
        createChip('スタミナ', () => formatAbilityValue(traits?.stamina ?? currentPitcher.stamina)),
      ].filter(Boolean);

      if (abilityChips.length) {
        abilityChips.forEach((chip) => abilityRow.appendChild(chip));
      } else {
        const fallback = document.createElement('span');
        fallback.className = 'title-pitcher-chip empty';
        fallback.textContent = '能力データがありません。';
        abilityRow.appendChild(fallback);
      }

      summary.appendChild(abilityRow);
    }
  }

  if (list) {
    if (!hasPitchers) {
      appendMessage(list, 'title-pitcher-empty', '先発候補が登録されていません。');
    } else {
      pitchers.forEach((pitcher) => {
        if (!pitcher || !pitcher.name) return;
        const normalizedName = String(pitcher.name).trim();
        const traits = traitMap.get(normalizedName);
        const isCurrent = Boolean(pitcher.is_current);
        const isSelected = selectedName === normalizedName;

        const option = document.createElement('button');
        option.type = 'button';
        option.className = 'title-pitcher-option';
        option.dataset.titlePitcherOption = 'true';
        option.dataset.team = teamKey;
        option.dataset.pitcher = normalizedName;
        option.setAttribute('aria-pressed', isSelected ? 'true' : 'false');

        if (isSelected) {
          option.classList.add('selected');
        }
        if (isCurrent) {
          option.classList.add('current');
        }

        const header = document.createElement('div');
        header.className = 'title-pitcher-option-header';

        const name = document.createElement('div');
        name.className = 'title-pitcher-option-name';
        name.textContent = normalizedName || '投手';
        header.appendChild(name);

        const metaParts = [];
        if (pitcher.pitcher_type) {
          metaParts.push(pitcher.pitcher_type);
        }
        if (pitcher.throws) {
          metaParts.push(pitcher.throws);
        }
        if (metaParts.length) {
          const meta = document.createElement('div');
          meta.className = 'title-pitcher-option-meta';
          meta.textContent = metaParts.join(' / ');
          header.appendChild(meta);
        }

        if (isCurrent) {
          const badge = document.createElement('span');
          badge.className = 'title-pitcher-option-badge';
          badge.textContent = '現在の先発';
          header.appendChild(badge);
        } else if (isSelected) {
          const badge = document.createElement('span');
          badge.className = 'title-pitcher-option-badge';
          badge.textContent = '選択中';
          header.appendChild(badge);
        }

        option.appendChild(header);

        const statsRow = document.createElement('div');
        statsRow.className = 'title-pitcher-option-stats';

        const eraChip = createChip('ERA', () => formatStatValue(pitcher.era));
        const ipChip = createChip('IP', () => formatStatValue(pitcher.ip));
        const soChip = createChip('SO', () => formatStatValue(pitcher.so));
        [eraChip, ipChip, soChip].forEach((chip) => {
          if (chip) statsRow.appendChild(chip);
        });

        if (!statsRow.children.length) {
          const fallback = document.createElement('span');
          fallback.className = 'title-pitcher-chip empty';
          fallback.textContent = '成績データなし';
          statsRow.appendChild(fallback);
        }

        option.appendChild(statsRow);

        const abilityRow = document.createElement('div');
        abilityRow.className = 'title-pitcher-option-abilities';

        const abilityChips = [
          createChip('K%', () => formatAbilityValue(traits?.k_pct)),
          createChip('BB%', () => formatAbilityValue(traits?.bb_pct)),
          createChip('強打%', () => formatAbilityValue(traits?.hard_pct)),
          createChip('ゴロ%', () => formatAbilityValue(traits?.gb_pct)),
          createChip('スタミナ', () => formatAbilityValue(traits?.stamina ?? pitcher.stamina)),
        ].filter(Boolean);

        if (abilityChips.length) {
          abilityChips.forEach((chip) => abilityRow.appendChild(chip));
        } else {
          const fallback = document.createElement('span');
          fallback.className = 'title-pitcher-chip empty';
          fallback.textContent = '能力データなし';
          abilityRow.appendChild(fallback);
        }

        option.appendChild(abilityRow);

        const actionsRow = document.createElement('div');
        actionsRow.className = 'title-pitcher-option-actions';

        const actionButton = document.createElement('button');
        actionButton.type = 'button';
        actionButton.className = 'title-pitcher-option-button primary';
        actionButton.dataset.action = 'review-title-pitcher';
        actionButton.dataset.team = teamKey;
        actionButton.dataset.pitcher = normalizedName;
        if (isCurrent) {
          actionButton.textContent = '先発に設定済み';
          actionButton.disabled = true;
        } else if (isSelected) {
          actionButton.textContent = '設定を確認';
        } else {
          actionButton.textContent = '先発に設定';
        }

        actionsRow.appendChild(actionButton);
        option.appendChild(actionsRow);

        list.appendChild(option);
      });
    }
  }
}

function updateTitleEditorUI(teamKey) {
  const { team: activeTeam, view } = getTitleEditorView();
  const card = document.querySelector(`[data-team-card="${teamKey}"]`);
  const openButtons = document.querySelectorAll(
    `[data-action="open-lineup"][data-team="${teamKey}"], [data-action="open-pitcher"][data-team="${teamKey}"]`,
  );
  const isOpen = isTitleEditorOpen(teamKey);
  const isLineupActive = isOpen && activeTeam === teamKey && view === 'lineup';
  const isPitcherActive = isOpen && activeTeam === teamKey && view === 'pitcher';

  if (card) {
    card.classList.toggle('editor-open', isOpen);
  }

  openButtons.forEach((button) => {
    const action = button.dataset.action;
    if (action === 'open-lineup') {
      const expanded = isLineupActive;
      button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
      button.classList.toggle('active', expanded);
    } else if (action === 'open-pitcher') {
      const expanded = isPitcherActive;
      button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
      button.classList.toggle('active', expanded);
    } else {
      button.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      button.classList.toggle('active', isOpen);
    }
  });

  const lineupPanel = document.querySelector(`#lineup-editor-screen [data-title-editor="${teamKey}"]`);
  if (lineupPanel) {
    const visible = isLineupActive;
    lineupPanel.classList.toggle('is-open', visible);
    lineupPanel.setAttribute('aria-hidden', visible ? 'false' : 'true');
    if (visible) {
      lineupPanel.removeAttribute('hidden');
    } else {
      lineupPanel.setAttribute('hidden', '');
    }
  }

  const pitcherPanel = document.querySelector(`#pitcher-editor-screen [data-title-editor="${teamKey}"]`);
  if (pitcherPanel) {
    const visible = isPitcherActive;
    pitcherPanel.classList.toggle('is-open', visible);
    pitcherPanel.setAttribute('aria-hidden', visible ? 'false' : 'true');
    if (visible) {
      pitcherPanel.removeAttribute('hidden');
    } else {
      pitcherPanel.setAttribute('hidden', '');
    }
  }
}

export function renderTitle(titleState) {
  const homeName = document.querySelector('.team-name[data-team="home"]');
  const awayName = document.querySelector('.team-name[data-team="away"]');
  const homeMessage = document.querySelector('.team-message[data-team="home"]');
  const awayMessage = document.querySelector('.team-message[data-team="away"]');
  const homeErrors = document.querySelector('.team-errors[data-team="home"]');
  const awayErrors = document.querySelector('.team-errors[data-team="away"]');
  const homeControl = elements.teamControlHome;
  const awayControl = elements.teamControlAway;

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

  const teamsData = stateCache.data?.teams || {};
  renderTitleLineup('home', teamsData.home, Boolean(teamsData?.home));
  renderTitleLineup('away', teamsData.away, Boolean(teamsData?.away));
  renderTitleBench('home', teamsData.home, Boolean(teamsData?.home));
  renderTitleBench('away', teamsData.away, Boolean(teamsData?.away));
  renderTitlePitcher('home', teamsData.home, Boolean(teamsData?.home));
  renderTitlePitcher('away', teamsData.away, Boolean(teamsData?.away));
  updateTitleEditorUI('home');
  updateTitleEditorUI('away');

  const { team: editorTeam, view: editorView } = getTitleEditorView();
  if (elements.lineupEditorTeamLabel) {
    if (editorTeam && editorView === 'lineup') {
      const name = editorTeam === 'home' ? titleState.home?.name : titleState.away?.name;
      elements.lineupEditorTeamLabel.textContent = name ? `編集中: ${name}` : '';
    } else {
      elements.lineupEditorTeamLabel.textContent = '';
    }
  }
  if (elements.pitcherEditorTeamLabel) {
    if (editorTeam && editorView === 'pitcher') {
      const name = editorTeam === 'home' ? titleState.home?.name : titleState.away?.name;
      elements.pitcherEditorTeamLabel.textContent = name ? `編集中: ${name}` : '';
    } else {
      elements.pitcherEditorTeamLabel.textContent = '';
    }
  }

  if (homeControl) {
    if (teamsData.home) {
      const label = teamsData.home.control_label || 'あなた';
      homeControl.textContent = `操作: ${label}`;
      homeControl.classList.remove('hidden');
    } else {
      homeControl.textContent = '';
      homeControl.classList.add('hidden');
    }
  }
  if (awayControl) {
    if (teamsData.away) {
      const label = teamsData.away.control_label || 'あなた';
      awayControl.textContent = `操作: ${label}`;
      awayControl.classList.remove('hidden');
    } else {
      awayControl.textContent = '';
      awayControl.classList.add('hidden');
    }
  }

  elements.titleHint.textContent = titleState.hint || '';
  elements.startButton.disabled = !titleState.ready;

  if (elements.titleControlHint) {
    let hintMode = stateCache.gameControl?.mode || 'manual';
    let hintUserTeam = stateCache.gameControl?.userTeam || null;
    let hintCpuTeam = stateCache.gameControl?.cpuTeam || null;
    let hintUserName = stateCache.gameControl?.userTeamName || null;
    let hintCpuName = stateCache.gameControl?.cpuTeamName || null;
    let hintHomeName = null;
    let hintAwayName = null;

    const upcomingSetup = stateCache.matchSetup || { mode: 'manual', userTeam: 'home' };
    const gameActive = Boolean(stateCache.data?.game?.active);
    const teamLibraryTeams = Array.isArray(stateCache.teamLibrary?.teams)
      ? stateCache.teamLibrary.teams
      : [];
    const teamLookup = new Map(teamLibraryTeams.map((team) => [team.id, team]));

    if (!gameActive) {
      if (upcomingSetup.mode === 'cpu') {
        hintMode = 'cpu';
        hintUserTeam = upcomingSetup.userTeam === 'away' ? 'away' : 'home';
        hintCpuTeam = hintUserTeam === 'home' ? 'away' : 'home';
        const homeTeam = teamLookup.get(stateCache.teamLibrary?.selection?.home);
        const awayTeam = teamLookup.get(stateCache.teamLibrary?.selection?.away);
        hintUserName = hintUserTeam === 'home' ? homeTeam?.name : awayTeam?.name;
        hintCpuName = hintCpuTeam === 'home' ? homeTeam?.name : awayTeam?.name;
      } else if (upcomingSetup.mode === 'auto') {
        hintMode = 'auto';
        const homeTeam = teamLookup.get(stateCache.teamLibrary?.selection?.home);
        const awayTeam = teamLookup.get(stateCache.teamLibrary?.selection?.away);
        hintHomeName = homeTeam?.name || null;
        hintAwayName = awayTeam?.name || null;
      }
    }

    if (hintMode === 'cpu') {
      if (!hintUserTeam) {
        elements.titleControlHint.textContent = 'CPU対戦: 自操作チームを選択してください。';
      } else {
        const userLabel = hintUserName
          || (hintUserTeam === 'away' ? 'アウェイチーム' : 'ホームチーム');
        const cpuLabel = hintCpuName
          || (hintCpuTeam === 'away' ? 'アウェイチーム' : 'ホームチーム');
        elements.titleControlHint.textContent = `${userLabel} を操作します。${cpuLabel} はCPUが操作します。`;
      }
    } else if (hintMode === 'auto') {
      const homeName = hintHomeName || teamsData.home?.name || 'ホームチーム';
      const awayName = hintAwayName || teamsData.away?.name || 'アウェイチーム';
      elements.titleControlHint.textContent = `${awayName} と ${homeName} の試合は全自動CPUモードです。進行ボタンで試合を進めてください。`;
    } else {
      elements.titleControlHint.textContent = '全操作対戦: 両チームを自分で操作します。';
    }
  }
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
  const setup = stateCache.matchSetup || { mode: 'manual', userTeam: 'home' };
  const matchMode =
    setup.mode === 'cpu' ? 'cpu' : setup.mode === 'auto' ? 'auto' : 'manual';
  const selectedControlTeam = matchMode === 'cpu' && setup.userTeam === 'away' ? 'away' : 'home';

  (elements.matchModeRadios || []).forEach((radio) => {
    if (!radio) return;
    const rawValue = typeof radio.value === 'string' ? radio.value : '';
    const value = rawValue === 'cpu' ? 'cpu' : rawValue === 'auto' ? 'auto' : 'manual';
    radio.checked = value === matchMode;
  });

  if (elements.controlTeamField) {
    const hidden = matchMode !== 'cpu';
    elements.controlTeamField.classList.toggle('hidden', hidden);
    elements.controlTeamField.setAttribute('aria-hidden', hidden ? 'true' : 'false');
  }

  // Defer updating control team label until after team selects are populated
  const homeSelect = elements.lobbyHomeSelect;
  const awaySelect = elements.lobbyAwaySelect;
  const optionList = teams.map((team) => ({
    value: team.id,
    label: team.name || team.id,
  }));

  // Do not auto-select a fallback for either side.
  // Keep placeholder until the user explicitly chooses.
  if (homeSelect) {
    populateSelect(homeSelect, optionList, {
      placeholder: 'チームを選択する',
      selected: selection.home,
      fallback: undefined,
    });
    // If reset is requested for this view, force placeholder selection
    if (stateCache.resetTeamSelect === true) {
      homeSelect.value = '';
    }
  }

  if (awaySelect) {
    populateSelect(awaySelect, optionList, {
      placeholder: 'チームを選択する',
      selected: selection.away,
      fallback: undefined,
    });
    if (stateCache.resetTeamSelect === true) {
      awaySelect.value = '';
    }
  }

  // Clear the one-shot reset flag after applying
  if (stateCache.resetTeamSelect === true) {
    stateCache.resetTeamSelect = false;
  }

  // Now update the control team select labels.
  // Show team names only when both home/away are explicitly chosen in the UI.
  if (elements.controlTeamSelect) {
    const select = elements.controlTeamSelect;
    const teamLookup = new Map(teams.map((team) => [team.id, team]));
    const awayTeam = teamLookup.get(selection.away);
    const homeTeam = teamLookup.get(selection.home);
    const homeOption = select.querySelector('option[value="home"]');
    const awayOption = select.querySelector('option[value="away"]');

    const bothChosen = Boolean(homeSelect?.value && awaySelect?.value);

    if (homeOption) {
      homeOption.textContent = bothChosen && homeTeam?.name ? `ホーム (${homeTeam.name})` : 'ホーム';
    }
    if (awayOption) {
      awayOption.textContent = bothChosen && awayTeam?.name ? `アウェイ (${awayTeam.name})` : 'アウェイ';
    }

    select.value = selectedControlTeam;
    select.disabled = matchMode !== 'cpu';
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
  const showTeamSelect = view === 'team-select';
  const showBuilder = view === 'team-builder';
  const showPlayerBuilder = view === 'player-builder';
  const showTitle = view === 'title';
  const showLineupEditor = view === 'title-lineup-editor';
  const showPitcherEditor = view === 'title-pitcher-editor';
  const showGame = view === 'game';
  const showSimulationSetup = view === 'simulation';
  const showSimulationResults = view === 'simulation-results';
  const showSimulationMatch = view === 'simulation-match';

  if (elements.lobbyScreen) {
    elements.lobbyScreen.classList.toggle('hidden', !showLobby);
  }
  if (elements.teamSelectScreen) {
    elements.teamSelectScreen.classList.toggle('hidden', !showTeamSelect);
  }
  if (elements.teamBuilderScreen) {
    elements.teamBuilderScreen.classList.toggle('hidden', !showBuilder);
  }
  if (elements.playerBuilderScreen) {
    elements.playerBuilderScreen.classList.toggle('hidden', !showPlayerBuilder);
  }
  if (elements.titleScreen) {
    elements.titleScreen.classList.toggle('hidden', !showTitle);
  }
  if (elements.lineupEditorScreen) {
    elements.lineupEditorScreen.classList.toggle('hidden', !showLineupEditor);
  }
  if (elements.pitcherEditorScreen) {
    elements.pitcherEditorScreen.classList.toggle('hidden', !showPitcherEditor);
  }
  if (elements.gameScreen) {
    elements.gameScreen.classList.toggle('hidden', !showGame);
  }
  if (elements.simulationSetupScreen) {
    elements.simulationSetupScreen.classList.toggle('hidden', !showSimulationSetup);
  }
  if (elements.simulationResultsScreen) {
    elements.simulationResultsScreen.classList.toggle('hidden', !showSimulationResults);
  }
  if (elements.simulationMatchScreen) {
    elements.simulationMatchScreen.classList.toggle('hidden', !showSimulationMatch);
  }
}

function formatAverageDisplay(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '-';
  const formatted = num.toFixed(3);
  return formatted.startsWith('0') ? formatted.slice(1) : formatted;
}

function formatNumberDisplay(value, digits = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '-';
  return digits > 0 ? num.toFixed(digits) : String(Math.round(num));
}

function formatSignedNumber(value, digits = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '-';
  const formatted = digits > 0 ? num.toFixed(digits) : String(Math.round(num));
  if (num > 0) {
    return `+${formatted}`;
  }
  return formatted;
}

function formatInningsDisplay(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '-';
  const outs = Math.round(num * 3);
  const innings = Math.floor(outs / 3);
  const remainder = outs % 3;
  return `${innings}.${remainder}`;
}

function getTeamGamesPlayed(teamEntry) {
  if (!teamEntry) return 0;
  const record = teamEntry.record || {};
  const games = Number(record.games);
  if (Number.isFinite(games) && games > 0) {
    return games;
  }
  const wins = Number(record.wins);
  const losses = Number(record.losses);
  const draws = Number(record.draws);
  const total =
    (Number.isFinite(wins) ? wins : 0) +
    (Number.isFinite(losses) ? losses : 0) +
    (Number.isFinite(draws) ? draws : 0);
  return total > 0 ? total : 0;
}

function formatQualificationDisplay(reached, applicable) {
  if (!applicable) return '-';
  return reached ? '◯' : '×';
}

function formatPercentageDisplay(value, digits = 1) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '-';
  return `${num.toFixed(digits)}%`;
}

const BATTING_RANKING_COLUMNS = [
  { key: 'rank', label: '順位', sortable: false },
  { key: 'name', label: '選手', sortable: true, sortKey: 'name', defaultDir: 'asc', type: 'string', cellClass: 'player-col' },
  { key: 'team', label: 'チーム', sortable: true, sortKey: 'team', defaultDir: 'asc', type: 'string', cellClass: 'player-col' },
  { key: 'pa', label: 'PA', sortable: true, sortKey: 'pa', defaultDir: 'desc', type: 'number' },
  { key: 'ab', label: 'AB', sortable: true, sortKey: 'ab', defaultDir: 'desc', type: 'number' },
  { key: 'hits', label: 'H', sortable: true, sortKey: 'hits', defaultDir: 'desc', type: 'number' },
  { key: 'avg', label: 'AVG', sortable: true, sortKey: 'avg', defaultDir: 'desc', type: 'number', formatter: (row) => formatAverageDisplay(row.avg) },
  { key: 'obp', label: 'OBP', sortable: true, sortKey: 'obp', defaultDir: 'desc', type: 'number', formatter: (row) => formatAverageDisplay(row.obp) },
  { key: 'slg', label: 'SLG', sortable: true, sortKey: 'slg', defaultDir: 'desc', type: 'number', formatter: (row) => formatAverageDisplay(row.slg) },
  { key: 'ops', label: 'OPS', sortable: true, sortKey: 'ops', defaultDir: 'desc', type: 'number', formatter: (row) => formatAverageDisplay(row.ops) },
  { key: 'homeRuns', label: 'HR', sortable: true, sortKey: 'homeRuns', defaultDir: 'desc', type: 'number' },
  { key: 'rbi', label: 'RBI', sortable: true, sortKey: 'rbi', defaultDir: 'desc', type: 'number' },
  { key: 'runs', label: 'R', sortable: true, sortKey: 'runs', defaultDir: 'desc', type: 'number' },
  { key: 'walks', label: 'BB', sortable: true, sortKey: 'walks', defaultDir: 'desc', type: 'number' },
  { key: 'strikeouts', label: 'SO', sortable: true, sortKey: 'strikeouts', defaultDir: 'desc', type: 'number' },
  { key: 'k_pct', label: 'K%', sortable: true, sortKey: 'k_pct', defaultDir: 'desc', type: 'number', formatter: (row) => formatPercentageDisplay(row.k_pct) },
  { key: 'bb_pct', label: 'BB%', sortable: true, sortKey: 'bb_pct', defaultDir: 'desc', type: 'number', formatter: (row) => formatPercentageDisplay(row.bb_pct) },
  {
    key: 'qualified',
    label: '規定',
    sortable: true,
    sortKey: 'qualified',
    defaultDir: 'desc',
    type: 'boolean',
    formatter: (row) => formatQualificationDisplay(row.qualified && row.qualApplicable, row.qualApplicable),
  },
];

const PITCHING_RANKING_COLUMNS = [
  { key: 'rank', label: '順位', sortable: false },
  { key: 'name', label: '選手', sortable: true, sortKey: 'name', defaultDir: 'asc', type: 'string', cellClass: 'player-col' },
  { key: 'team', label: 'チーム', sortable: true, sortKey: 'team', defaultDir: 'asc', type: 'string', cellClass: 'player-col' },
  {
    key: 'appearances',
    label: '登板',
    sortable: true,
    sortKey: 'appearances',
    defaultDir: 'desc',
    type: 'number',
  },
  {
    key: 'ip',
    label: '投球回',
    sortable: true,
    sortKey: 'ip',
    defaultDir: 'desc',
    type: 'number',
    formatter: (row) => formatInningsDisplay(row.ip),
  },
  {
    key: 'era',
    label: 'ERA',
    sortable: true,
    sortKey: 'era',
    defaultDir: 'asc',
    type: 'number',
    formatter: (row) => formatNumberDisplay(row.era, 2),
  },
  {
    key: 'whip',
    label: 'WHIP',
    sortable: true,
    sortKey: 'whip',
    defaultDir: 'asc',
    type: 'number',
    formatter: (row) => formatNumberDisplay(row.whip, 2),
  },
  {
    key: 'kPer9',
    label: 'K/9',
    sortable: true,
    sortKey: 'kPer9',
    defaultDir: 'desc',
    type: 'number',
    formatter: (row) => formatNumberDisplay(row.kPer9, 2),
  },
  {
    key: 'bbPer9',
    label: 'BB/9',
    sortable: true,
    sortKey: 'bbPer9',
    defaultDir: 'asc',
    type: 'number',
    formatter: (row) => formatNumberDisplay(row.bbPer9, 2),
  },
  {
    key: 'strikeouts',
    label: '奪三振',
    sortable: true,
    sortKey: 'strikeouts',
    defaultDir: 'desc',
    type: 'number',
  },
  {
    key: 'walks',
    label: '与四球',
    sortable: true,
    sortKey: 'walks',
    defaultDir: 'asc',
    type: 'number',
  },
  {
    key: 'hits',
    label: '被安打',
    sortable: true,
    sortKey: 'hits',
    defaultDir: 'asc',
    type: 'number',
  },
  {
    key: 'runs',
    label: '失点',
    sortable: true,
    sortKey: 'runs',
    defaultDir: 'asc',
    type: 'number',
  },
  {
    key: 'earnedRuns',
    label: '自責',
    sortable: true,
    sortKey: 'earnedRuns',
    defaultDir: 'asc',
    type: 'number',
  },
  {
    key: 'homeRuns',
    label: '被本塁打',
    sortable: true,
    sortKey: 'homeRuns',
    defaultDir: 'asc',
    type: 'number',
  },
  {
    key: 'qualified',
    label: '規定',
    sortable: true,
    sortKey: 'qualified',
    defaultDir: 'desc',
    type: 'boolean',
    formatter: (row) => formatQualificationDisplay(row.qualified && row.qualApplicable, row.qualApplicable),
  },
];

const RANKING_ZERO_PARTICIPATION_COLUMNS = {
  batting: new Set(['avg', 'obp', 'slg', 'ops', 'k_pct', 'bb_pct']),
  pitching: new Set(['era', 'whip', 'kPer9', 'bbPer9']),
};

function renderSimulationSetup(teamLibraryState, simulationState) {
  const {
    simulationSetupForm,
    simulationLeagueSelect,
    simulationLeagueAdd,
    simulationLeagueClear,
    simulationLeagueList,
    simulationLeagueBalance,
    simulationLeagueCount,
    simulationSeriesStructure,
    simulationGamesPerCardSummary,
    simulationGamesPerTeamSummary,
    simulationTotalGamesSummary,
    simulationGamesPerCardInput,
    simulationCardsPerOpponentInput,
    simulationStartButton,
    simulationSetupFeedback,
  } = elements;

  if (!simulationSetupForm) {
    return;
  }

  const teams = Array.isArray(teamLibraryState?.teams) ? teamLibraryState.teams : [];
  const options = teams.map((team) => ({ value: team.id, label: team.name || team.id }));
  const optionMap = new Map(options.map((entry) => [entry.value, entry.label]));
  const running = Boolean(simulationState?.running);

  const leagueDefaults = simulationState?.league || {};
  const defaultTeams = Array.isArray(leagueDefaults.teams) ? leagueDefaults.teams : [];
  const sanitizedDefaults = defaultTeams
    .filter((teamId) => typeof teamId === 'string' && teamId.trim())
    .map((teamId) => teamId.trim())
    .filter((teamId) => optionMap.has(teamId));

  primeSimulationSetup({
    teams: sanitizedDefaults,
    gamesPerCard:
      typeof leagueDefaults.gamesPerCard === 'number' && leagueDefaults.gamesPerCard > 0
        ? leagueDefaults.gamesPerCard
        : null,
    cardsPerOpponent:
      typeof leagueDefaults.cardsPerOpponent === 'number' && leagueDefaults.cardsPerOpponent > 0
        ? leagueDefaults.cardsPerOpponent
        : null,
  });

  const schedule = getSimulationSchedule();
  const scheduleDefaults = getSimulationScheduleDefaults();

  const gamesPerCardValue =
    Number.isFinite(schedule.gamesPerCard) && schedule.gamesPerCard > 0
      ? schedule.gamesPerCard
      : scheduleDefaults.gamesPerCard || 1;
  const cardsPerOpponentValue =
    Number.isFinite(schedule.cardsPerOpponent) && schedule.cardsPerOpponent > 0
      ? schedule.cardsPerOpponent
      : scheduleDefaults.cardsPerOpponent || 1;

  if (simulationLeagueSelect) {
    populateSelect(simulationLeagueSelect, options, {
      placeholder: 'チームを選択する',
      selected: simulationLeagueSelect.value,
      fallback: options.length ? options[0].value : '',
    });
    simulationLeagueSelect.disabled = running || options.length === 0;
  }

  if (simulationLeagueAdd) {
    simulationLeagueAdd.disabled = running || options.length === 0;
  }
  if (simulationLeagueClear) {
    simulationLeagueClear.disabled = running;
  }

  const currentTeams = getSimulationLeagueTeams().filter((teamId) => optionMap.has(teamId));
  setSimulationLeagueTeams(currentTeams);

  const teamCount = currentTeams.length;
  const hasTeams = teamCount >= 2;
  const evenTeamCount = teamCount % 2 === 0;
  const opponentsPerTeam = Math.max(teamCount - 1, 0);
  const pairingCount = teamCount >= 2 ? (teamCount * opponentsPerTeam) / 2 : 0;
  const gamesPerOpponent = gamesPerCardValue * cardsPerOpponentValue;
  const totalCards = pairingCount * cardsPerOpponentValue;
  const gamesPerTeam = opponentsPerTeam > 0 ? gamesPerOpponent * opponentsPerTeam : 0;
  const totalGames = pairingCount > 0 ? gamesPerOpponent * pairingCount : 0;
  const formatCount = (value, suffix = '') =>
    Number.isFinite(value) && value > 0
      ? `${Math.round(value).toLocaleString('ja-JP')}${suffix}`
      : '--';

  if (simulationLeagueCount) {
    simulationLeagueCount.textContent = teamCount.toLocaleString('ja-JP');
  }
  if (simulationSeriesStructure) {
    simulationSeriesStructure.textContent = formatCount(totalCards, 'カード');
  }
  if (simulationGamesPerCardSummary) {
    simulationGamesPerCardSummary.textContent = `${gamesPerCardValue.toLocaleString('ja-JP')}試合 × ${cardsPerOpponentValue.toLocaleString('ja-JP')}カード`;
  }
  if (simulationGamesPerTeamSummary) {
    simulationGamesPerTeamSummary.textContent = formatCount(gamesPerTeam, '試合');
  }
  if (simulationTotalGamesSummary) {
    simulationTotalGamesSummary.textContent = formatCount(totalGames, '試合');
  }
  if (simulationLeagueBalance) {
    let balanceMessage = '0チーム';
    let balanceState = 'neutral';
    if (teamCount === 1) {
      balanceMessage = '1チーム（追加してください）';
      balanceState = 'odd';
    } else if (teamCount >= 2 && evenTeamCount) {
      balanceMessage = '偶数チームでバランス良好';
      balanceState = 'even';
    } else if (teamCount >= 2) {
      balanceMessage = '奇数チーム（1チーム追加が必要）';
      balanceState = 'odd';
    }
    simulationLeagueBalance.textContent = balanceMessage;
    simulationLeagueBalance.dataset.state = balanceState;
  }

  if (simulationLeagueList) {
    simulationLeagueList.innerHTML = '';
    if (!currentTeams.length) {
      const empty = document.createElement('li');
      empty.className = 'league-team-empty';
      empty.textContent = 'チームを追加してください。';
      simulationLeagueList.appendChild(empty);
    } else {
      currentTeams.forEach((teamId, index) => {
        const li = document.createElement('li');
        li.className = 'league-team-item';

        const order = document.createElement('span');
        order.className = 'league-team-order';
        order.textContent = `${index + 1}.`;

        const name = document.createElement('span');
        name.className = 'league-team-name';
        name.textContent = optionMap.get(teamId) || teamId;

        const removeButton = document.createElement('button');
        removeButton.type = 'button';
        removeButton.className = 'ghost danger';
        removeButton.dataset.action = 'remove';
        removeButton.dataset.teamId = teamId;
        removeButton.dataset.index = String(index);
        removeButton.textContent = '削除';

        li.appendChild(order);
        li.appendChild(name);
        li.appendChild(removeButton);
        simulationLeagueList.appendChild(li);
      });
    }
  }

  if (simulationGamesPerCardInput) {
    simulationGamesPerCardInput.min = '1';
    simulationGamesPerCardInput.max = '20';
    const focused = document.activeElement === simulationGamesPerCardInput;
    const value = gamesPerCardValue || 1;
    if (!focused) {
      simulationGamesPerCardInput.value = String(value);
    }
    simulationGamesPerCardInput.disabled = running;
  }

  if (simulationCardsPerOpponentInput) {
    simulationCardsPerOpponentInput.min = '1';
    simulationCardsPerOpponentInput.max = '100';
    const focused = document.activeElement === simulationCardsPerOpponentInput;
    const value = cardsPerOpponentValue || 1;
    if (!focused) {
      simulationCardsPerOpponentInput.value = String(value);
    }
    simulationCardsPerOpponentInput.disabled = running;
  }

  const validSchedule =
    Number.isFinite(schedule.gamesPerCard) &&
    schedule.gamesPerCard > 0 &&
    Number.isFinite(schedule.cardsPerOpponent) &&
    schedule.cardsPerOpponent > 0;
  const canStart = hasTeams && evenTeamCount && validSchedule && !running;

  if (simulationStartButton) {
    simulationStartButton.disabled = !canStart;
  }

  if (simulationSetupFeedback) {
    let message = '';
    let level = 'info';
    if (running) {
      message = 'シミュレーションを実行中です…';
    } else if (!options.length) {
      message = 'チームデータを読み込めませんでした。チーム管理を確認してください。';
      level = 'danger';
    } else if (!hasTeams) {
      message = 'リーグに参加させるチームを2チーム以上追加してください。';
    } else if (!evenTeamCount) {
      message = 'リーグ参加チーム数は偶数で指定してください。';
      level = 'danger';
    } else if (!validSchedule) {
      message = 'カード設定は1以上の数値で指定してください。';
      level = 'danger';
    } else if (teamLibraryState?.hint) {
      message = teamLibraryState.hint;
    } else {
      message = '準備ができたら「シミュレーション開始」を押してください。';
    }

    simulationSetupFeedback.textContent = message;
    simulationSetupFeedback.classList.remove('danger', 'success', 'info');
    simulationSetupFeedback.classList.add(level);
  }
}

function renderTeamRecordRow(tbody, teamEntry) {
  if (!tbody || !teamEntry) return;
  const row = document.createElement('tr');
  const nameCell = document.createElement('th');
  nameCell.scope = 'row';
  const rank = Number.isFinite(Number(teamEntry.rank)) ? Number(teamEntry.rank) : null;
  const displayName = teamEntry.name || '-';
  nameCell.textContent = rank ? `${rank}. ${displayName}` : displayName;
  row.appendChild(nameCell);

  const record = teamEntry.record || {};
  const cells = [
    Number.isFinite(record.wins) ? record.wins : 0,
    Number.isFinite(record.losses) ? record.losses : 0,
    Number.isFinite(record.draws) ? record.draws : 0,
    Number.isFinite(record.winPct) ? record.winPct.toFixed(3) : '-',
    Number.isFinite(record.runsScored) ? record.runsScored : 0,
    Number.isFinite(record.runsAllowed) ? record.runsAllowed : 0,
    Number.isFinite(record.runDiff) ? formatSignedNumber(record.runDiff) : '-',
  ];

  cells.forEach((value, index) => {
    const cell = document.createElement('td');
    cell.textContent = typeof value === 'number' ? String(value) : value;
    if (index === 3 && typeof value === 'string' && value !== '-') {
      cell.classList.add('simulation-winrate');
    }
    if (index === 6) {
      if (typeof value === 'string' && value.startsWith('+')) {
        cell.classList.add('positive');
      } else if (typeof value === 'string' && value.startsWith('-')) {
        cell.classList.add('negative');
      }
    }
    row.appendChild(cell);
  });

  tbody.appendChild(row);
}

function describeTeamTotals(teamEntry) {
  if (!teamEntry) return '';
  const record = teamEntry.record || {};
  const batting = teamEntry.batting || {};
  const pitching = teamEntry.pitching || {};

  const pieces = [];
  const wins = Number.isFinite(record.wins) ? record.wins : 0;
  const losses = Number.isFinite(record.losses) ? record.losses : 0;
  const draws = Number.isFinite(record.draws) ? record.draws : 0;
  pieces.push(`勝:${wins} 敗:${losses} 分:${draws}`);
  if (Number.isFinite(record.runDiff)) {
    pieces.push(`得失点差 ${formatSignedNumber(record.runDiff)}`);
  }
  if (Number.isFinite(batting.ops)) {
    pieces.push(`OPS ${formatAverageDisplay(batting.ops)}`);
  }
  if (Number.isFinite(pitching.era)) {
    pieces.push(`防御率 ${formatNumberDisplay(pitching.era, 2)}`);
  }
  return pieces.join(' / ');
}

function renderBattingTable(tbody, teamEntry) {
  if (!tbody) return;
  tbody.innerHTML = '';
  const rows = Array.isArray(teamEntry?.batters) ? teamEntry.batters : [];
  if (!rows.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 19;
    td.textContent = 'データがありません。';
    td.classList.add('empty');
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  const gamesPlayed = getTeamGamesPlayed(teamEntry);
  const paRequirement = gamesPlayed > 0 ? Math.max(0, Math.round(gamesPlayed * 3)) : 0;
  const qualificationApplicable = paRequirement > 0;

  rows.forEach((row) => {
    const tr = document.createElement('tr');
    const cells = [
      row.name || '-',
      row.pa ?? 0,
      row.ab ?? 0,
      row.hits ?? 0,
      row.singles ?? 0,
      row.doubles ?? 0,
      row.triples ?? 0,
      row.homeRuns ?? 0,
      row.runs ?? 0,
      row.rbi ?? 0,
      row.walks ?? 0,
      row.strikeouts ?? 0,
      // K% / BB% are numeric (e.g., 12.3); display with one decimal and a % sign
      (() => {
        const v = Number(row.k_pct);
        return Number.isFinite(v) ? `${v.toFixed(1)}%` : '-';
      })(),
      (() => {
        const v = Number(row.bb_pct);
        return Number.isFinite(v) ? `${v.toFixed(1)}%` : '-';
      })(),
      formatAverageDisplay(row.avg),
      formatAverageDisplay(row.obp),
      formatAverageDisplay(row.slg),
      formatAverageDisplay(row.ops),
      (() => {
        const pa = Number(row.pa);
        if (!Number.isFinite(pa)) {
          return formatQualificationDisplay(false, qualificationApplicable);
        }
        const reached = qualificationApplicable ? pa >= paRequirement : false;
        return formatQualificationDisplay(reached, qualificationApplicable);
      })(),
    ];
    cells.forEach((value) => {
      const td = document.createElement('td');
      td.textContent = value;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function renderPitchingTable(tbody, teamEntry) {
  if (!tbody) return;
  tbody.innerHTML = '';
  const rows = Array.isArray(teamEntry?.pitchers) ? teamEntry.pitchers : [];
  if (!rows.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 14;
    td.textContent = 'データがありません。';
    td.classList.add('empty');
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  const gamesPlayed = getTeamGamesPlayed(teamEntry);
  const ipRequirementOuts = gamesPlayed > 0 ? Math.max(0, Math.round(gamesPlayed * 3)) : 0;
  const qualificationApplicable = ipRequirementOuts > 0;

  rows.forEach((row) => {
    const tr = document.createElement('tr');
    const cells = [
      row.name || '-',
      row.appearances ?? 0,
      formatInningsDisplay(row.ip),
      row.hits ?? 0,
      row.runs ?? 0,
      row.earnedRuns ?? 0,
      row.walks ?? 0,
      row.strikeouts ?? 0,
      row.homeRuns ?? 0,
      formatNumberDisplay(row.era, 2),
      formatNumberDisplay(row.whip, 2),
      formatNumberDisplay(row.kPer9, 2),
      formatNumberDisplay(row.bbPer9, 2),
      (() => {
        const ip = Number(row.ip);
        if (!Number.isFinite(ip)) {
          return formatQualificationDisplay(false, qualificationApplicable);
        }
        const outs = Math.round(ip * 3);
        const reached = qualificationApplicable ? outs >= ipRequirementOuts : false;
        return formatQualificationDisplay(reached, qualificationApplicable);
      })(),
    ];
    cells.forEach((value) => {
      const td = document.createElement('td');
      td.textContent = value;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function setTabActive(button, active) {
  if (!button) return;
  if (active) {
    button.classList.add('active');
    button.setAttribute('aria-pressed', 'true');
  } else {
    button.classList.remove('active');
    button.setAttribute('aria-pressed', 'false');
  }
}

function updateSimulationResultsViewUI() {
  const {
    simulationSummarySection,
    simulationSeriesSection,
    simulationMatchupsSection,
    simulationTeamStatsSection,
    simulationGamesSection,
    simulationPlayersSection,
    simulationRankingsSection,
    simulationPlayersTabsRow,
    simulationPlayersTypeBatting,
    simulationPlayersTypePitching,
    simulationSelectedBattingTable,
    simulationSelectedPitchingTable,
  } = elements;
  const {
    simulationTabSummary,
    simulationTabSeries,
    simulationTabMatchups,
    simulationTabTeamStats,
    simulationTabGames,
    simulationTabPlayers,
    simulationTabRankings,
  } = elements;
  const view = stateCache.simulationResultsView || 'summary';
  const showSummary = view === 'summary';
  const showSeries = view === 'series';
  const showMatchups = view === 'matchups';
  const showTeamStats = view === 'teamStats';
  const showGames = view === 'games';
  const showPlayers = view === 'players';
  const showRankings = view === 'rankings';

  if (simulationSummarySection) {
    simulationSummarySection.classList.toggle('hidden', !showSummary);
    simulationSummarySection.setAttribute('aria-hidden', showSummary ? 'false' : 'true');
  }
  if (simulationSeriesSection) {
    simulationSeriesSection.classList.toggle('hidden', !showSeries);
    simulationSeriesSection.setAttribute('aria-hidden', showSeries ? 'false' : 'true');
  }
  if (simulationMatchupsSection) {
    simulationMatchupsSection.classList.toggle('hidden', !showMatchups);
    simulationMatchupsSection.setAttribute('aria-hidden', showMatchups ? 'false' : 'true');
  }
  if (simulationTeamStatsSection) {
    simulationTeamStatsSection.classList.toggle('hidden', !showTeamStats);
    simulationTeamStatsSection.setAttribute('aria-hidden', showTeamStats ? 'false' : 'true');
  }
  if (simulationGamesSection) {
    simulationGamesSection.classList.toggle('hidden', !showGames);
    simulationGamesSection.setAttribute('aria-hidden', showGames ? 'false' : 'true');
  }
  if (simulationPlayersSection) {
    simulationPlayersSection.classList.toggle('hidden', !showPlayers);
    simulationPlayersSection.setAttribute('aria-hidden', showPlayers ? 'false' : 'true');
  }
  if (simulationRankingsSection) {
    simulationRankingsSection.classList.toggle('hidden', !showRankings);
    simulationRankingsSection.setAttribute('aria-hidden', showRankings ? 'false' : 'true');
  }

  setTabActive(simulationTabSummary, showSummary);
  setTabActive(simulationTabSeries, showSeries);
  setTabActive(simulationTabMatchups, showMatchups);
  setTabActive(simulationTabTeamStats, showTeamStats);
  setTabActive(simulationTabGames, showGames);
  setTabActive(simulationTabPlayers, showPlayers);
  setTabActive(simulationTabRankings, showRankings);

  // 個人成績サブタブの表示/切替
  if (simulationPlayersTabsRow) {
    simulationPlayersTabsRow.classList.toggle('hidden', !showPlayers);
    simulationPlayersTabsRow.setAttribute('aria-hidden', showPlayers ? 'false' : 'true');
  }
  if (showPlayers) {
    const typeView = stateCache.playersTypeView === 'pitching' ? 'pitching' : 'batting';

    // 種別タブのActive表示
    setTabActive(simulationPlayersTypeBatting, typeView === 'batting');
    setTabActive(simulationPlayersTypePitching, typeView === 'pitching');

    // テーブルの表示切替（打者/投手）
    const showBatting = typeView === 'batting';
    const showPitching = typeView === 'pitching';
    if (simulationSelectedBattingTable)
      simulationSelectedBattingTable.classList.toggle('hidden', !showBatting);
    if (simulationSelectedPitchingTable)
      simulationSelectedPitchingTable.classList.toggle('hidden', !showPitching);
  }
}

function pickLeader(candidates, key, opts = {}) {
  const { minKey = null, minValue = 0, reverse = false } = opts;
  const filtered = candidates.filter((p) => {
    if (minKey) {
      const threshold = typeof minValue === 'function' ? Number(minValue(p)) : Number(minValue);
      const required = Number.isFinite(threshold) && threshold > 0 ? threshold : 0;
      const v = Number(p[minKey]);
      if (!Number.isFinite(v) || v < required) return false;
    }
    const val = Number(p[key]);
    return Number.isFinite(val);
  });
  if (!filtered.length) return null;
  filtered.sort((a, b) => {
    const va = Number(a[key]);
    const vb = Number(b[key]);
    return reverse ? va - vb : vb - va;
  });
  return filtered[0];
}

function renderSimulationLeaders(lastRun) {
  const { simulationLeadersList } = elements;
  if (!simulationLeadersList) return;

  simulationLeadersList.innerHTML = '';
  if (!lastRun) return;

  const teams = Array.isArray(lastRun.teams) ? lastRun.teams : [];
  const batters = teams.flatMap((t) => (Array.isArray(t.batters) ? t.batters.map((b) => ({ ...b, team: t.name })) : []));
  const pitchers = teams.flatMap((t) => (Array.isArray(t.pitchers) ? t.pitchers.map((p) => ({ ...p, team: t.name })) : []));

  const gamesByTeam = new Map();
  teams.forEach((team) => {
    const games = Number(team?.record?.games);
    gamesByTeam.set(team.name, Number.isFinite(games) && games > 0 ? games : 0);
  });

  const paRequirement = (player) => {
    const games = gamesByTeam.get(player.team) || 0;
    return games > 0 ? games * 3 : 0;
  };
  const ipRequirement = (player) => {
    const games = gamesByTeam.get(player.team) || 0;
    return games > 0 ? games : 0;
  };

  const hitterAvg = pickLeader(batters, 'avg', { minKey: 'pa', minValue: paRequirement });
  const hitterOPS = pickLeader(batters, 'ops', { minKey: 'pa', minValue: paRequirement });
  const hrKing = pickLeader(batters, 'homeRuns');
  const rbiKing = pickLeader(batters, 'rbi');

  const eraBest = pickLeader(pitchers, 'era', { minKey: 'ip', minValue: ipRequirement, reverse: true });
  const k9Best = pickLeader(pitchers, 'kPer9', { minKey: 'ip', minValue: ipRequirement });
  const whipBest = pickLeader(pitchers, 'whip', { minKey: 'ip', minValue: ipRequirement, reverse: true });

  const fmtName = (p) => `${p.name || '-'}（${p.team || '-'}）`;
  const addCard = (kind, icon, label, valueText, nameText) => {
    const li = document.createElement('li');
    li.className = `leader-card kind-${kind}`;
    const title = document.createElement('div');
    title.className = 'leader-title';
    const iconEl = document.createElement('span');
    iconEl.className = 'leader-icon';
    iconEl.textContent = icon;
    const labelEl = document.createElement('span');
    labelEl.textContent = label;
    const valueEl = document.createElement('div');
    valueEl.className = 'leader-value';
    valueEl.textContent = valueText;
    const nameEl = document.createElement('div');
    nameEl.className = 'leader-name';
    nameEl.textContent = nameText;
    title.appendChild(iconEl);
    title.appendChild(labelEl);
    li.appendChild(title);
    li.appendChild(valueEl);
    li.appendChild(nameEl);
    simulationLeadersList.appendChild(li);
  };

  let added = 0;
  if (hitterAvg) {
    addCard('batting', '🥇', '首位打者', `AVG ${formatAverageDisplay(hitterAvg.avg)}`, fmtName(hitterAvg));
    added += 1;
  }
  if (hrKing) {
    addCard('batting', '⚾', '本塁打王', `HR ${Number(hrKing.homeRuns) || 0}`, fmtName(hrKing));
    added += 1;
  }
  if (rbiKing) {
    addCard('batting', '🎯', '打点王', `RBI ${Number(rbiKing.rbi) || 0}`, fmtName(rbiKing));
    added += 1;
  }
  if (hitterOPS) {
    addCard('batting', '🔥', 'OPS 1位', `OPS ${formatAverageDisplay(hitterOPS.ops)}`, fmtName(hitterOPS));
    added += 1;
  }
  if (eraBest) {
    addCard('pitching', '🧱', '最優秀防御率', `ERA ${formatNumberDisplay(eraBest.era, 2)}`, fmtName(eraBest));
    added += 1;
  }
  if (k9Best) {
    addCard('pitching', '⚡', '最多奪三振率', `K/9 ${formatNumberDisplay(k9Best.kPer9, 2)}`, fmtName(k9Best));
    added += 1;
  }
  if (whipBest) {
    addCard('pitching', '🧪', '最優秀WHIP', `WHIP ${formatNumberDisplay(whipBest.whip, 2)}`, fmtName(whipBest));
    added += 1;
  }

  if (added === 0) {
    const li = document.createElement('li');
    li.className = 'leader-card';
    const title = document.createElement('div');
    title.className = 'leader-title';
    title.textContent = '個人タイトル';
    const valueEl = document.createElement('div');
    valueEl.className = 'leader-value';
    valueEl.textContent = '--';
    const nameEl = document.createElement('div');
    nameEl.className = 'leader-name';
    nameEl.textContent = '対象者がいません。';
    li.appendChild(title);
    li.appendChild(valueEl);
    li.appendChild(nameEl);
    simulationLeadersList.appendChild(li);
  }
}

function renderSimulationTeamStatsTable(tbody, teams) {
  if (!tbody) return;
  tbody.innerHTML = '';
  const rows = Array.isArray(teams) ? teams : [];
  if (!rows.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 10;
    td.textContent = 'チーム別統計はまだありません。';
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  rows.forEach((team) => {
    const record = team.record || {};
    const batting = team.batting || {};
    const pitching = team.pitching || {};
    const games = Number.isFinite(Number(record.games)) ? Number(record.games) : 0;
    const runsPerGame = games > 0 ? Number(record.runsScored || 0) / games : 0;

    const tr = document.createElement('tr');
    const cells = [
      team.name || '-',
      formatAverageDisplay(batting.avg),
      formatAverageDisplay(batting.obp),
      formatAverageDisplay(batting.slg),
      formatAverageDisplay(batting.ops),
      formatNumberDisplay(runsPerGame, 2),
      formatNumberDisplay(pitching.era, 2),
      formatNumberDisplay(pitching.whip, 2),
      formatNumberDisplay(pitching.kPer9, 2),
      formatNumberDisplay(pitching.bbPer9, 2),
    ];

    cells.forEach((value, index) => {
      const isNameCell = index === 0;
      const cell = document.createElement(isNameCell ? 'th' : 'td');
      cell.textContent = typeof value === 'number' ? String(value) : value;
      if (isNameCell) {
        cell.scope = 'row';
        cell.style.fontWeight = '600';
      }
      tr.appendChild(cell);
    });
    tbody.appendChild(tr);
  });
}

function renderSimulationMatchupsTable(tbody, games, aliases) {
  if (!tbody) return;
  const table = elements.simulationMatchupsTable;
  const thead = table ? table.querySelector('thead') : null;
  // reset
  if (thead) thead.innerHTML = '';
  tbody.innerHTML = '';

  const schedule = Array.isArray(games) ? games : [];
  const aliasMap = aliases && typeof aliases === 'object' ? aliases : {};
  const normalizeName = (name) => {
    if (!name || typeof name !== 'string') return '';
    const trimmed = name.trim();
    return aliasMap[trimmed] || trimmed;
  };

  // Collect team names
  const teamSet = new Set();
  schedule.forEach((g) => {
    const h = normalizeName(g.homeTeam ?? g.home_team);
    const a = normalizeName(g.awayTeam ?? g.away_team);
    if (h) teamSet.add(h);
    if (a) teamSet.add(a);
  });
  const teams = Array.from(teamSet).sort((a, b) => a.localeCompare(b));

  if (!teams.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.textContent = '対戦成績はまだありません。';
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  // Initialize matrix stats: stats[row][col] = {w,l,d}
  const stats = new Map();
  teams.forEach((r) => {
    const inner = new Map();
    teams.forEach((c) => {
      if (r !== c) inner.set(c, { w: 0, l: 0, d: 0 });
    });
    stats.set(r, inner);
  });

  // Tally results
  schedule.forEach((game) => {
    const home = normalizeName(game.homeTeam ?? game.home_team);
    const away = normalizeName(game.awayTeam ?? game.away_team);
    if (!home || !away) return;
    const winner = game.winner === 'home' || game.winner === 'away' ? game.winner : 'draw';
    const rowHome = stats.get(home);
    const rowAway = stats.get(away);
    if (!rowHome || !rowAway) return;
    const homeVsAway = rowHome.get(away);
    const awayVsHome = rowAway.get(home);
    if (!homeVsAway || !awayVsHome) return;
    if (winner === 'home') {
      homeVsAway.w += 1;
      awayVsHome.l += 1;
    } else if (winner === 'away') {
      awayVsHome.w += 1;
      homeVsAway.l += 1;
    } else {
      homeVsAway.d += 1;
      awayVsHome.d += 1;
    }
  });

  // Build header
  if (thead) {
    const hr = document.createElement('tr');
    const corner = document.createElement('th');
    corner.scope = 'col';
    corner.textContent = 'チーム';
    hr.appendChild(corner);
    teams.forEach((name) => {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = name;
      hr.appendChild(th);
    });
    thead.appendChild(hr);
  }

  // Build body rows
  teams.forEach((rowTeam) => {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.scope = 'row';
    th.textContent = rowTeam;
    th.style.fontWeight = '600';
    tr.appendChild(th);

    teams.forEach((colTeam) => {
      const td = document.createElement('td');
      if (rowTeam === colTeam) {
        td.textContent = '—';
        td.style.textAlign = 'center';
      } else {
        const rec = stats.get(rowTeam)?.get(colTeam) || { w: 0, l: 0, d: 0 };
        const label = rec.d ? `${rec.w}-${rec.l}-${rec.d}` : `${rec.w}-${rec.l}`;
        td.textContent = label;
        td.style.textAlign = 'center';
        td.title = `${rowTeam} vs ${colTeam}: ${label}`;
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function getNumericValue(source, ...keys) {
  if (!source) return null;
  for (let i = 0; i < keys.length; i += 1) {
    const key = keys[i];
    if (key && Object.prototype.hasOwnProperty.call(source, key)) {
      const value = Number(source[key]);
      if (Number.isFinite(value)) {
        return value;
      }
    }
  }
  return null;
}

function buildBattingRankingRows(teams) {
  const rows = [];
  teams.forEach((team) => {
    if (!team) return;
    const teamName = team.name || team.key || 'Team';
    const gamesPlayed = getTeamGamesPlayed(team);
    const paRequirement = gamesPlayed > 0 ? Math.max(0, Math.round(gamesPlayed * 3)) : 0;
    const qualificationApplicable = paRequirement > 0;
    const batters = Array.isArray(team.batters) ? team.batters : [];
    batters.forEach((player) => {
      if (!player) return;
      const pa = getNumericValue(player, 'pa', 'PA') ?? 0;
      const row = {
        name: player.name || '',
        team: teamName,
        pa,
        ab: getNumericValue(player, 'ab', 'AB') ?? 0,
        hits: getNumericValue(player, 'hits', 'H') ?? 0,
        homeRuns: getNumericValue(player, 'homeRuns', 'home_runs', 'HR') ?? 0,
        runs: getNumericValue(player, 'runs', 'R') ?? 0,
        rbi: getNumericValue(player, 'rbi', 'RBI') ?? 0,
        walks: getNumericValue(player, 'walks', 'BB') ?? 0,
        strikeouts: getNumericValue(player, 'strikeouts', 'SO', 'K') ?? 0,
        avg: getNumericValue(player, 'avg') ?? null,
        obp: getNumericValue(player, 'obp') ?? null,
        slg: getNumericValue(player, 'slg') ?? null,
        ops: getNumericValue(player, 'ops') ?? null,
        k_pct: getNumericValue(player, 'k_pct', 'kPct') ?? null,
        bb_pct: getNumericValue(player, 'bb_pct', 'bbPct') ?? null,
      };
      row.qualApplicable = qualificationApplicable;
      row.qualified = qualificationApplicable ? pa >= paRequirement : true;
      row.workload = pa;
      rows.push(row);
    });
  });
  return rows;
}

function buildPitchingRankingRows(teams) {
  const rows = [];
  teams.forEach((team) => {
    if (!team) return;
    const teamName = team.name || team.key || 'Team';
    const gamesPlayed = getTeamGamesPlayed(team);
    const ipRequirementOuts = gamesPlayed > 0 ? Math.max(0, Math.round(gamesPlayed * 3)) : 0;
    const qualificationApplicable = ipRequirementOuts > 0;
    const pitchers = Array.isArray(team.pitchers) ? team.pitchers : [];
    pitchers.forEach((player) => {
      if (!player) return;
      const ip = getNumericValue(player, 'ip', 'IP') ?? 0;
      const outs = Math.round(ip * 3);
      const row = {
        name: player.name || '',
        team: teamName,
        appearances: getNumericValue(player, 'appearances', 'G') ?? 0,
        ip,
        hits: getNumericValue(player, 'hits', 'H') ?? 0,
        runs: getNumericValue(player, 'runs', 'R') ?? 0,
        earnedRuns: getNumericValue(player, 'earnedRuns', 'earned_runs', 'ER') ?? 0,
        walks: getNumericValue(player, 'walks', 'BB') ?? 0,
        strikeouts: getNumericValue(player, 'strikeouts', 'SO', 'K') ?? 0,
        homeRuns: getNumericValue(player, 'homeRuns', 'home_runs', 'HR') ?? 0,
        era: getNumericValue(player, 'era') ?? null,
        whip: getNumericValue(player, 'whip') ?? null,
        kPer9: getNumericValue(player, 'kPer9', 'k_per_9') ?? null,
        bbPer9: getNumericValue(player, 'bbPer9', 'bb_per_9') ?? null,
      };
      row.qualApplicable = qualificationApplicable;
      row.qualified = qualificationApplicable ? outs >= ipRequirementOuts : true;
      row.workload = outs;
      rows.push(row);
    });
  });
  return rows;
}

function sortRankingRows(rows, column, sortKey, direction, type, scope) {
  if (!rows.length || !sortKey) return;
  const columnType = column?.type || 'number';
  const zeroSet = RANKING_ZERO_PARTICIPATION_COLUMNS[type] || new Set();
  const dir = direction === 'asc' ? 'asc' : 'desc';
  const multiplier = dir === 'asc' ? 1 : -1;

  rows.sort((a, b) => {
    if (scope === 'all' && zeroSet.has(sortKey)) {
      const aWork = Number(a.workload);
      const bWork = Number(b.workload);
      const aZero = !Number.isFinite(aWork) || aWork <= 0;
      const bZero = !Number.isFinite(bWork) || bWork <= 0;
      if (aZero !== bZero) {
        return aZero ? 1 : -1;
      }
    }

    let comparison = 0;
    if (columnType === 'string') {
      const aValue = typeof a[sortKey] === 'string' ? a[sortKey] : '';
      const bValue = typeof b[sortKey] === 'string' ? b[sortKey] : '';
      comparison = JAPANESE_COLLATOR.compare(aValue, bValue);
    } else if (columnType === 'boolean') {
      const aValue = a[sortKey] ? 1 : 0;
      const bValue = b[sortKey] ? 1 : 0;
      comparison = aValue - bValue;
    } else {
      const aRaw = Number(a[sortKey]);
      const bRaw = Number(b[sortKey]);
      const fallback = dir === 'asc' ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
      const aValue = Number.isFinite(aRaw) ? aRaw : fallback;
      const bValue = Number.isFinite(bRaw) ? bRaw : fallback;
      comparison = aValue - bValue;
    }

    if (comparison !== 0) {
      return comparison * multiplier;
    }

    const teamCompare = JAPANESE_COLLATOR.compare(a.team || '', b.team || '');
    if (teamCompare !== 0) {
      return teamCompare;
    }
    return JAPANESE_COLLATOR.compare(a.name || '', b.name || '');
  });
}

function renderSimulationRankings(lastRun) {
  const {
    simulationRankingsHead,
    simulationRankingsBody,
    simulationRankingsEmpty,
    simulationRankingsTypeButtons,
    simulationRankingsScopeButtons,
  } = elements;

  if (!simulationRankingsHead || !simulationRankingsBody) {
    return;
  }

  let state = getSimulationRankingsState();
  const columns = state.type === 'pitching' ? PITCHING_RANKING_COLUMNS : BATTING_RANKING_COLUMNS;
  const sortableColumns = columns.filter((column) => column.sortable);
  const availableSortKeys = new Set(sortableColumns.map((column) => column.sortKey));
  let sortKey = availableSortKeys.has(state.sortKey)
    ? state.sortKey
    : sortableColumns[0]?.sortKey || null;
  let sortDir = state.sortDir === 'asc' || state.sortDir === 'desc'
    ? state.sortDir
    : sortableColumns.find((column) => column.sortKey === sortKey)?.defaultDir || 'desc';

  simulationRankingsHead.innerHTML = '';
  columns.forEach((column) => {
    const th = document.createElement('th');
    if (column.sortable) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'ranking-sort-button';
      btn.dataset.rankingsSort = column.sortKey;
      btn.dataset.rankingsDefault = column.defaultDir || 'desc';
      btn.textContent = column.label;
      th.appendChild(btn);
    } else {
      th.textContent = column.label;
    }
    simulationRankingsHead.appendChild(th);
  });

  const teams = Array.isArray(lastRun?.teams) ? lastRun.teams : [];
  let rows = [];
  if (teams.length) {
    rows = state.type === 'pitching' ? buildPitchingRankingRows(teams) : buildBattingRankingRows(teams);
  }

  let filteredRows = state.scope === 'qualified' ? rows.filter((row) => row.qualified) : rows.slice();

  const shouldFallbackToAll =
    state.scope === 'qualified' && !state.userOverride && filteredRows.length === 0 && rows.length > 0;

  if (shouldFallbackToAll) {
    setSimulationRankingsScope('all', false);
    state = getSimulationRankingsState();
    filteredRows = rows.slice();
    sortKey = availableSortKeys.has(state.sortKey)
      ? state.sortKey
      : sortableColumns[0]?.sortKey || null;
    sortDir = state.sortDir === 'asc' || state.sortDir === 'desc'
      ? state.sortDir
      : sortableColumns.find((column) => column.sortKey === sortKey)?.defaultDir || 'desc';
  }

  (simulationRankingsTypeButtons || []).forEach((button) => {
    const type = button.dataset.rankingsType === 'pitching' ? 'pitching' : 'batting';
    const active = type === state.type;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
  });

  (simulationRankingsScopeButtons || []).forEach((button) => {
    const scope = button.dataset.rankingsScope === 'all' ? 'all' : 'qualified';
    const active = scope === state.scope;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
  });

  if (sortKey) {
    const sortColumn = columns.find((column) => column.sortKey === sortKey);
    if (sortColumn) {
      const fallbackDir = sortColumn.defaultDir || 'desc';
      const direction = sortDir === 'asc' || sortDir === 'desc' ? sortDir : fallbackDir;
      sortRankingRows(filteredRows, sortColumn, sortKey, direction, state.type, state.scope);
      filteredRows.forEach((row, index) => {
        row.rank = index + 1;
      });
      Array.from(simulationRankingsHead.children).forEach((th) => {
        const btn = th.querySelector('button[data-rankings-sort]');
        if (!btn) {
          th.removeAttribute('aria-sort');
          return;
        }
        const key = btn.dataset.rankingsSort;
        if (key === sortKey) {
          btn.classList.add('active');
          btn.dataset.sortDirection = direction;
          th.setAttribute('aria-sort', direction === 'asc' ? 'ascending' : 'descending');
        } else {
          btn.classList.remove('active');
          btn.removeAttribute('data-sort-direction');
          th.removeAttribute('aria-sort');
        }
      });
    } else {
      filteredRows.forEach((row, index) => {
        row.rank = index + 1;
      });
    }
  } else {
    filteredRows.forEach((row, index) => {
      row.rank = index + 1;
    });
    Array.from(simulationRankingsHead.children).forEach((th) => {
      const btn = th.querySelector('button[data-rankings-sort]');
      if (btn) {
        btn.classList.remove('active');
        btn.removeAttribute('data-sort-direction');
      }
      th.removeAttribute('aria-sort');
    });
  }

  simulationRankingsBody.innerHTML = '';
  if (filteredRows.length) {
    filteredRows.forEach((row) => {
      const tr = document.createElement('tr');
      columns.forEach((column) => {
        const cell = document.createElement('td');
        let display = '-';
        if (column.key === 'rank') {
          display = row.rank != null ? String(row.rank) : '-';
        } else if (typeof column.formatter === 'function') {
          display = column.formatter(row);
        } else {
          const value = row[column.key];
          if (value !== null && value !== undefined && value !== '') {
            display = String(value);
          }
        }
        if (column.cellClass) {
          cell.classList.add(column.cellClass);
        } else if (column.key === 'name' || column.key === 'team') {
          cell.classList.add('player-col');
        }
        cell.textContent = display;
        tr.appendChild(cell);
      });
      simulationRankingsBody.appendChild(tr);
    });
  }

  if (simulationRankingsEmpty) {
    let message = '表示できる選手がいません。';
    if (!lastRun) {
      message = 'シミュレーション結果がまだありません。';
    } else if (rows.length && !filteredRows.length && state.scope === 'qualified') {
      message = '規定到達者がいません。';
    }
    if (shouldFallbackToAll) {
      message = '規定到達者がいなかったため、全選手を表示しています。';
    }
    simulationRankingsEmpty.textContent = message;
    const hideMessage = filteredRows.length > 0 && !shouldFallbackToAll;
    simulationRankingsEmpty.classList.toggle('hidden', hideMessage);
  }
}

function renderSimulationResults(simulationState) {
  const {
    simulationResultsSummary,
    simulationResultsMeta,
    simulationResultsHighlights,
    simulationResultsTableBody,
    simulationGamesTableBody,
    simulationGamesStats,
    simulationSelectedTeamName,
    simulationSelectedTeamSummary,
    simulationSelectedBattingBody,
    simulationSelectedPitchingBody,
    simulationTeamStatsBody,
    simulationMatchupsBody,
    simulationPlayersTeamSelect,
    simulationSelectedBattingTable,
    simulationSelectedPitchingTable,
    simulationPlayMatch,
  } = elements;

  const lastRun = simulationState?.lastRun || null;
  const playableTeams = Array.isArray(simulationState?.playable?.teams)
    ? simulationState.playable.teams
    : [];
  const canPlayMatch = Boolean(lastRun && playableTeams.length >= 2);

  if (simulationPlayMatch) {
    simulationPlayMatch.classList.toggle('hidden', !lastRun);
    simulationPlayMatch.disabled = !canPlayMatch;
    if (!canPlayMatch && lastRun) {
      simulationPlayMatch.title = '試合を行うには異なる2チームを選択してください。';
    } else {
      simulationPlayMatch.removeAttribute('title');
    }
  }

  if (!lastRun) {
    if (simulationResultsSummary) {
      simulationResultsSummary.textContent = 'まだシミュレーション結果がありません。';
    }
    if (simulationResultsMeta) {
      simulationResultsMeta.textContent = '';
    }
    if (simulationSelectedTeamName) simulationSelectedTeamName.textContent = 'チーム';
    if (simulationResultsTableBody) {
      simulationResultsTableBody.innerHTML = '';
    }
    if (simulationGamesTableBody) {
      simulationGamesTableBody.innerHTML = '';
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 6;
      td.textContent = 'シミュレーション結果が表示されるとここに試合一覧が表示されます。';
      tr.appendChild(td);
      simulationGamesTableBody.appendChild(tr);
    }
    if (simulationGamesStats) {
      simulationGamesStats.innerHTML = '';
      const emptyStats = document.createElement('p');
      emptyStats.className = 'simulation-empty-message';
      emptyStats.textContent = '集計可能な試合がまだありません。';
      simulationGamesStats.appendChild(emptyStats);
    }
    if (simulationResultsHighlights) {
      simulationResultsHighlights.innerHTML = '';
      const emptyHighlights = document.createElement('p');
      emptyHighlights.className = 'simulation-empty-message';
      emptyHighlights.textContent = 'シミュレーションを実行するとハイライトが表示されます。';
      simulationResultsHighlights.appendChild(emptyHighlights);
    }
    if (simulationSelectedTeamSummary) simulationSelectedTeamSummary.textContent = '';
    if (simulationSelectedBattingBody) simulationSelectedBattingBody.innerHTML = '';
    if (simulationSelectedPitchingBody) simulationSelectedPitchingBody.innerHTML = '';
    if (simulationPlayersTeamSelect) simulationPlayersTeamSelect.innerHTML = '';
    renderSimulationTeamStatsTable(simulationTeamStatsBody, []);
    renderSimulationMatchupsTable(simulationMatchupsBody, [], {});
    renderSimulationRankings(null);
    updateSimulationResultsViewUI();
    return;
  }

  const teams = Array.isArray(lastRun.teams) ? lastRun.teams : [];
  const leagueInfo = lastRun.league || {};
  const mode = typeof lastRun.mode === 'string' ? lastRun.mode.toLowerCase() : '';
  const isLeague = mode === 'league' || teams.length > 2;
  const games = Array.isArray(lastRun.games) ? lastRun.games : [];

  const totalGamesCount = Number.isFinite(Number(lastRun.totalGames))
    ? Number(lastRun.totalGames)
    : games.length;

  const gameTotals = {
    total: games.length,
    homeWins: 0,
    awayWins: 0,
    draws: 0,
    runsHome: 0,
    runsAway: 0,
  };

  games.forEach((g) => {
    if (g.winner === 'home') gameTotals.homeWins += 1;
    else if (g.winner === 'away') gameTotals.awayWins += 1;
    else gameTotals.draws += 1;
    gameTotals.runsHome += Number(g.homeScore) || 0;
    gameTotals.runsAway += Number(g.awayScore) || 0;
  });

  const combinedRuns = gameTotals.runsHome + gameTotals.runsAway;
  const averageTotalRuns = gameTotals.total ? combinedRuns / gameTotals.total : null;

  const rolesMap = new Map();
  teams.forEach((team) => {
    if (Array.isArray(team.roles)) {
      team.roles.forEach((role) => {
        if (typeof role === 'string' && !rolesMap.has(role)) {
          rolesMap.set(role, team);
        }
      });
    }
  });

  // 個人成績: チーム選択を生成
  if (simulationPlayersTeamSelect) {
    simulationPlayersTeamSelect.innerHTML = '';
    teams.forEach((team, idx) => {
      const opt = document.createElement('option');
      opt.value = String(idx);
      opt.textContent = team?.name || `チーム${idx + 1}`;
      simulationPlayersTeamSelect.appendChild(opt);
    });
  }

  let selectedIndex = getPlayersSelectedTeamIndex ? getPlayersSelectedTeamIndex() : 0;
  if (!Number.isInteger(selectedIndex) || selectedIndex < 0 || selectedIndex >= teams.length) {
    selectedIndex = 0;
    if (setPlayersSelectedTeamIndex) setPlayersSelectedTeamIndex(0);
  }
  const selectedTeam = teams[selectedIndex] || null;
  if (simulationPlayersTeamSelect) {
    simulationPlayersTeamSelect.value = String(selectedIndex);
  }
  if (simulationSelectedTeamName) {
    simulationSelectedTeamName.textContent = selectedTeam?.name || 'チーム';
  }

  if (simulationResultsSummary) {
    if (isLeague) {
      const totalTeams = Number.isFinite(Number(leagueInfo.totalTeams))
        ? Number(leagueInfo.totalTeams)
        : teams.length;
      simulationResultsSummary.textContent = `${totalTeams || teams.length}チームによるリーグシミュレーション`;
    } else {
      // 旧: 対戦形式。サマリにはAway/Homeを表示
      const awayName = rolesMap.get('away')?.name || 'Away';
      const homeName = rolesMap.get('home')?.name || 'Home';
      simulationResultsSummary.textContent = `${awayName} vs ${homeName}`;
    }
  }

  if (simulationResultsMeta) {
    const timestamp = formatSimulationTimestamp(lastRun.timestamp);
    const parts = [];
    if (isLeague) {
      const totalTeams = Number.isFinite(Number(leagueInfo.totalTeams))
        ? Number(leagueInfo.totalTeams)
        : teams.length;
      if (totalTeams > 0) {
        parts.push(`${totalTeams}チーム参加`);
      }
      const gamesPerCard = Number.isFinite(Number(leagueInfo.gamesPerCard))
        ? Number(leagueInfo.gamesPerCard)
        : null;
      const cardsPerOpponent = Number.isFinite(Number(leagueInfo.cardsPerOpponent))
        ? Number(leagueInfo.cardsPerOpponent)
        : null;
      if (gamesPerCard && cardsPerOpponent) {
        parts.push(`1カード${gamesPerCard}試合 × ${cardsPerOpponent}カード`);
      }
      const completedGames = Number.isFinite(Number(leagueInfo.completedGames))
        ? Number(leagueInfo.completedGames)
        : totalGamesCount;
      const scheduledGames = Number.isFinite(Number(leagueInfo.scheduledGames))
        ? Number(leagueInfo.scheduledGames)
        : null;
      if (completedGames || scheduledGames) {
        parts.push(
          `総試合数: ${completedGames}${scheduledGames ? ` / ${scheduledGames}` : ''}`,
        );
      }
      const completedDays = Number.isFinite(Number(leagueInfo.completedDays))
        ? Number(leagueInfo.completedDays)
        : null;
      const totalDays = Number.isFinite(Number(leagueInfo.totalDays))
        ? Number(leagueInfo.totalDays)
        : null;
      if (completedDays || totalDays) {
        parts.push(`日程: ${completedDays ?? '-'}${totalDays ? ` / ${totalDays}` : ''}`);
      }
    } else {
      if (totalGamesCount > 0) {
        parts.push(`${totalGamesCount}試合をシミュレーションしました`);
      }
    }
    if (timestamp) {
      parts.push(`最終実行: ${timestamp}`);
    }
    simulationResultsMeta.textContent = parts.join(' / ');
  }

  if (simulationResultsHighlights) {
    simulationResultsHighlights.innerHTML = '';
    const highlightItems = [];
    const addHighlight = (label, value, meta = '') => {
      if (!label) return;
      const valueText = value != null ? String(value).trim() : '';
      if (!valueText) return;
      const entry = {
        label,
        value: valueText,
        meta: typeof meta === 'string' ? meta : '',
      };
      highlightItems.push(entry);
      return entry;
    };

    if (!isLeague && gameTotals.total > 0) {
      const awayTeamEntry = rolesMap.get('away') || null;
      const homeTeamEntry = rolesMap.get('home') || null;
      const awayName = awayTeamEntry?.name || 'Away';
      const homeName = homeTeamEntry?.name || 'Home';
      const drawLabel = gameTotals.draws ? ` / 引き分け ${gameTotals.draws}` : '';
      addHighlight(
        'シリーズ結果',
        `${gameTotals.awayWins} - ${gameTotals.homeWins}`,
        `${awayName} vs ${homeName}${drawLabel}`,
      );
    }

    if (teams.length) {
      const teamsByWinPct = [...teams].sort((a, b) => {
        const recordA = a?.record || {};
        const recordB = b?.record || {};
        const winDiff = (recordB.winPct ?? 0) - (recordA.winPct ?? 0);
        if (winDiff !== 0) return winDiff;
        const winsDiff = (recordB.wins ?? 0) - (recordA.wins ?? 0);
        if (winsDiff !== 0) return winsDiff;
        const runDiffA = Number.isFinite(recordA.runDiff)
          ? recordA.runDiff
          : Number.NEGATIVE_INFINITY;
        const runDiffB = Number.isFinite(recordB.runDiff)
          ? recordB.runDiff
          : Number.NEGATIVE_INFINITY;
        if (runDiffA !== runDiffB) return runDiffB - runDiffA;
        return (recordB.runsScored ?? 0) - (recordA.runsScored ?? 0);
      });
      const bestTeam = teamsByWinPct[0] || null;
      if (bestTeam) {
        const record = bestTeam.record || {};
        const wins = Number.isFinite(record.wins) ? record.wins : 0;
        const losses = Number.isFinite(record.losses) ? record.losses : 0;
        const draws = Number.isFinite(record.draws) ? record.draws : 0;
        const winPctDisplay = Number.isFinite(record.winPct)
          ? formatAverageDisplay(record.winPct)
          : null;
        const metaParts = [`勝${wins}`, `敗${losses}`];
        if (draws) metaParts.push(`分${draws}`);
        if (winPctDisplay && winPctDisplay !== '-') {
          metaParts.push(`勝率 ${winPctDisplay}`);
        }
        addHighlight('勝率トップ', bestTeam.name || '-', metaParts.join(' / '));
      }

      const runDiffLeader = [...teams]
        .filter((team) => Number.isFinite(team?.record?.runDiff))
        .sort((a, b) => (b.record.runDiff ?? Number.NEGATIVE_INFINITY)
          - (a.record.runDiff ?? Number.NEGATIVE_INFINITY))[0] || null;
      if (
        runDiffLeader &&
        Number.isFinite(runDiffLeader.record?.runDiff) &&
        runDiffLeader.record.runDiff !== 0
      ) {
        addHighlight(
          '得失点差トップ',
          formatSignedNumber(runDiffLeader.record.runDiff),
          runDiffLeader.name || '',
        );
      }

      const offenseLeader = [...teams]
        .map((team) => {
          const record = team.record || {};
          const wins = Number.isFinite(record.wins) ? record.wins : 0;
          const losses = Number.isFinite(record.losses) ? record.losses : 0;
          const draws = Number.isFinite(record.draws) ? record.draws : 0;
          const gamesPlayed = Number.isFinite(record.games) && record.games > 0
            ? record.games
            : wins + losses + draws;
          const runsScored = Number.isFinite(record.runsScored) ? record.runsScored : 0;
          if (!gamesPlayed) return null;
          return {
            team,
            runsScored,
            perGame: runsScored / gamesPlayed,
          };
        })
        .filter(Boolean)
        .sort((a, b) => b.perGame - a.perGame)[0] || null;

      if (offenseLeader && offenseLeader.perGame > 0) {
        const perGameDisplay = formatNumberDisplay(offenseLeader.perGame, 1);
        const metaParts = [];
        const leaderName = offenseLeader.team.name || '';
        if (leaderName) metaParts.push(leaderName);
        if (Number.isFinite(offenseLeader.runsScored)) {
          metaParts.push(`総得点 ${offenseLeader.runsScored}`);
        }
        addHighlight('平均得点トップ', perGameDisplay, metaParts.join(' / '));
      }
    }

    let totalGamesHighlight = null;
    if (totalGamesCount > 0) {
      const avgDisplay = Number.isFinite(averageTotalRuns)
        ? formatNumberDisplay(averageTotalRuns, 1)
        : null;
      const metaParts = [];
      if (avgDisplay && avgDisplay !== '-') {
        metaParts.push(`平均合計得点 ${avgDisplay}`);
      }
      if (isLeague) {
        const scheduled = Number.isFinite(Number(leagueInfo.scheduledGames))
          ? Number(leagueInfo.scheduledGames)
          : null;
        if (scheduled && scheduled > 0) {
          const progress = Math.round((totalGamesCount / scheduled) * 100);
          if (Number.isFinite(progress)) {
            metaParts.push(`進捗 ${Math.max(0, Math.min(progress, 999))}%`);
          }
        }
      }
      const metaText = metaParts.length
        ? metaParts.join(' / ')
        : isLeague
        ? 'リーグ全体'
        : 'シリーズ合計';
      totalGamesHighlight = addHighlight('総試合数', `${totalGamesCount}`, metaText);
    }

    let itemsToRender = highlightItems.slice(0, 4);
    if (
      totalGamesHighlight &&
      itemsToRender.length === 4 &&
      !itemsToRender.includes(totalGamesHighlight)
    ) {
      itemsToRender = [...itemsToRender.slice(0, 3), totalGamesHighlight];
    }
    if (!itemsToRender.length) {
      const emptyHighlights = document.createElement('p');
      emptyHighlights.className = 'simulation-empty-message';
      emptyHighlights.textContent = 'シミュレーションを実行するとハイライトが表示されます。';
      simulationResultsHighlights.appendChild(emptyHighlights);
    } else {
      itemsToRender.forEach((item) => {
        const card = document.createElement('div');
        card.className = 'simulation-highlight-card';
        const labelEl = document.createElement('div');
        labelEl.className = 'label';
        labelEl.textContent = item.label;
        const valueEl = document.createElement('div');
        valueEl.className = 'value';
        valueEl.textContent = item.value;
        card.appendChild(labelEl);
        card.appendChild(valueEl);
        if (item.meta) {
          const metaEl = document.createElement('div');
          metaEl.className = 'meta';
          metaEl.textContent = item.meta;
          card.appendChild(metaEl);
        }
        simulationResultsHighlights.appendChild(card);
      });
    }
  }

  if (simulationResultsTableBody) {
    simulationResultsTableBody.innerHTML = '';
    if (!teams.length) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 8;
      td.textContent = 'チーム成績がありません。';
      tr.appendChild(td);
      simulationResultsTableBody.appendChild(tr);
    } else {
      teams.forEach((team) => {
        renderTeamRecordRow(simulationResultsTableBody, team);
      });
    }
  }

  if (simulationGamesTableBody) {
    simulationGamesTableBody.innerHTML = '';
    if (!games.length) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 6;
      td.textContent = '試合結果はまだありません。';
      tr.appendChild(td);
      simulationGamesTableBody.appendChild(tr);
    } else {
      games.forEach((game) => {
        const tr = document.createElement('tr');
        const winnerLabel =
          game.winner === 'home' ? 'ホーム勝利' : game.winner === 'away' ? 'アウェイ勝利' : '引き分け';
        const row = [
          game.index,
          game.awayTeam ?? 'Away',
          game.homeTeam ?? 'Home',
          `${game.awayScore} - ${game.homeScore}`,
          Number.isFinite(game.innings) && game.innings > 0 ? `${game.innings}` : '-',
          winnerLabel,
        ];
        row.forEach((cell, i) => {
          const td = document.createElement('td');
          td.textContent = String(cell);
          if (i === 3) td.style.textAlign = 'center';
          tr.appendChild(td);
        });
        simulationGamesTableBody.appendChild(tr);
      });
    }
  }

  if (simulationGamesStats) {
    simulationGamesStats.innerHTML = '';
    if (!games.length) {
      const emptyStats = document.createElement('p');
      emptyStats.className = 'simulation-empty-message';
      emptyStats.textContent = '集計可能な試合がまだありません。';
      simulationGamesStats.appendChild(emptyStats);
    } else {
      const total = gameTotals.total;
      const homeWins = gameTotals.homeWins;
      const awayWins = gameTotals.awayWins;
      const draws = gameTotals.draws;
      const avgHome = total ? (gameTotals.runsHome / total).toFixed(2) : '0.00';
      const avgAway = total ? (gameTotals.runsAway / total).toFixed(2) : '0.00';
      const avgTotal = total
        ? ((gameTotals.runsHome + gameTotals.runsAway) / total).toFixed(2)
        : '0.00';

      const addCard = (label, value) => {
        const card = document.createElement('div');
        card.className = 'simulation-stats-card';
        const labelEl = document.createElement('div');
        labelEl.className = 'label';
        labelEl.textContent = label;
        const valueEl = document.createElement('div');
        valueEl.className = 'value';
        valueEl.textContent = String(value);
        card.appendChild(labelEl);
        card.appendChild(valueEl);
        simulationGamesStats.appendChild(card);
      };

      addCard('総試合数', total);
      addCard('ホーム勝利', homeWins);
      addCard('アウェイ勝利', awayWins);
      if (draws) addCard('引き分け', draws);
      addCard('平均得点 (ホーム)', avgHome);
      addCard('平均得点 (アウェイ)', avgAway);
      addCard('平均合計得点', avgTotal);
    }
  }

  if (simulationSelectedTeamSummary) {
    simulationSelectedTeamSummary.textContent = describeTeamTotals(selectedTeam);
  }

  renderBattingTable(simulationSelectedBattingBody, selectedTeam);
  renderPitchingTable(simulationSelectedPitchingBody, selectedTeam);

  // 要約用の個人タイトル表示
  renderSimulationLeaders(lastRun);
  renderSimulationTeamStatsTable(simulationTeamStatsBody, teams);
  renderSimulationMatchupsTable(simulationMatchupsBody, games, lastRun.aliases || {});
  renderSimulationRankings(lastRun);
  // タブ/セクション表示更新
  updateSimulationResultsViewUI();
}

function renderSimulationMatch(simulationState) {
  const {
    simulationMatchAwaySelect,
    simulationMatchHomeSelect,
    simulationMatchHint,
    simulationMatchStart,
    simulationMatchModeRadios,
    simulationMatchControlField,
    simulationMatchControlSelect,
  } = elements;

  const matchState = stateCache.simulationMatch || {};
  const teams = Array.isArray(matchState.teams) ? matchState.teams : [];
  const defaults = matchState.defaults || { home: null, away: null };
  let selection = matchState.selection || { home: null, away: null };

  const validIds = new Set(teams.map((team) => team?.id).filter(Boolean));
  let selectedAway = validIds.has(selection.away) ? selection.away : null;
  let selectedHome = validIds.has(selection.home) ? selection.home : null;

  if (stateCache.resetSimulationSelect) {
    selectedAway = validIds.has(defaults.away) ? defaults.away : null;
    selectedHome = validIds.has(defaults.home) ? defaults.home : null;
  }

  if (!selectedAway && validIds.has(defaults.away)) {
    selectedAway = defaults.away;
  }
  if ((!selectedHome || selectedHome === selectedAway) && validIds.has(defaults.home)) {
    selectedHome = defaults.home !== selectedAway ? defaults.home : null;
  }

  if (!validIds.has(selectedAway)) {
    selectedAway = null;
  }
  if (!validIds.has(selectedHome) || selectedHome === selectedAway) {
    selectedHome = null;
  }

  selection = { home: selectedHome, away: selectedAway };
  stateCache.simulationMatch.selection = selection;

  const optionList = teams.map((team) => {
    if (!team || !team.id) {
      return null;
    }
    const name = team.name || team.id;
    const summary = typeof team.summary === 'string' && team.summary ? ` (${team.summary})` : '';
    return { value: team.id, label: `${name}${summary}` };
  }).filter(Boolean);

  if (simulationMatchAwaySelect) {
    populateSelect(simulationMatchAwaySelect, optionList, {
      placeholder: 'チームを選択する',
      selected: selectedAway,
      fallback: undefined,
    });
    if (!selectedAway) {
      simulationMatchAwaySelect.value = '';
    }
  }

  if (simulationMatchHomeSelect) {
    populateSelect(simulationMatchHomeSelect, optionList, {
      placeholder: 'チームを選択する',
      selected: selectedHome,
      fallback: undefined,
    });
    if (!selectedHome) {
      simulationMatchHomeSelect.value = '';
    }
  }

  const teamsReady = teams.length >= 2;
  const selectionReady = Boolean(selectedAway && selectedHome && selectedAway !== selectedHome);
  if (simulationMatchStart) {
    simulationMatchStart.disabled = !(teamsReady && selectionReady);
  }

  if (simulationMatchHint) {
    let hint = '';
    if (!teams.length) {
      hint = 'シミュレーション結果がありません。まずはシミュレーションを実行してください。';
    } else if (teams.length < 2) {
      hint = '試合を開始するにはもう1チーム必要です。';
    } else if (!selectionReady) {
      hint = '異なる2チームをホームとアウェイに選択してください。';
    } else {
      hint = `${teams.length}チームから選択できます。シミュレーション成績がそのまま適用されます。`;
    }
    simulationMatchHint.textContent = hint;
  }

  const setup = stateCache.matchSetup || { mode: 'manual', userTeam: 'home' };
  const matchMode =
    setup.mode === 'cpu' ? 'cpu' : setup.mode === 'auto' ? 'auto' : 'manual';
  const selectedControlTeam = matchMode === 'cpu' && setup.userTeam === 'away' ? 'away' : 'home';

  (simulationMatchModeRadios || []).forEach((radio) => {
    if (!radio) return;
    const rawValue = typeof radio.value === 'string' ? radio.value : '';
    const value = rawValue === 'cpu' ? 'cpu' : rawValue === 'auto' ? 'auto' : 'manual';
    radio.checked = value === matchMode;
  });

  if (simulationMatchControlField) {
    const hidden = matchMode !== 'cpu';
    simulationMatchControlField.classList.toggle('hidden', hidden);
    simulationMatchControlField.setAttribute('aria-hidden', hidden ? 'true' : 'false');
  }

  if (simulationMatchControlSelect) {
    const teamLookup = new Map(teams.map((team) => [team.id, team]));
    const awayTeam = teamLookup.get(selectedAway || undefined);
    const homeTeam = teamLookup.get(selectedHome || undefined);
    const awayOption = simulationMatchControlSelect.querySelector('option[value="away"]');
    const homeOption = simulationMatchControlSelect.querySelector('option[value="home"]');
    const bothChosen = Boolean(selectedAway && selectedHome);
    if (homeOption) {
      homeOption.textContent = bothChosen && homeTeam?.name ? `ホーム (${homeTeam.name})` : 'ホーム';
    }
    if (awayOption) {
      awayOption.textContent = bothChosen && awayTeam?.name ? `アウェイ (${awayTeam.name})` : 'アウェイ';
    }
    simulationMatchControlSelect.value = selectedControlTeam;
    simulationMatchControlSelect.disabled = matchMode !== 'cpu';
  }

  if (stateCache.resetSimulationSelect) {
    stateCache.resetSimulationSelect = false;
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
          const key = column.key;
          const value = row[key];
          if (key === 'name') {
            const nameText = value != null && value !== '' ? String(value) : '-';
            if (row.retired) {
              td.innerHTML = `${escapeHtml(nameText)} <span class="retired-mark" title="退いた">×</span>`;
            } else {
              td.textContent = nameText;
            }
          } else if (key === 'position') {
            td.innerHTML = renderPositionToken(value, row.pitcher_type);
            td.classList.add('player-col');
          } else if (key === 'eligible') {
            const list = Array.isArray(value) ? value : [];
            const filtered = list.filter((pos) => String(pos || '').toUpperCase() !== 'DH');
            td.innerHTML = renderPositionList(filtered);
            td.classList.add('player-col', 'position-list');
          } else if (viewType === 'pitching' && key === 'pitcher_type') {
            td.innerHTML = renderPositionToken(value, value);
          } else {
            td.textContent = value != null && value !== '' ? value : '-';
          }
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
          const key = column.key;
          const value = row[key];
          const displayValue = value != null && value !== '' ? String(value) : '-';

          if (viewType === 'batting' && key === 'position') {
            td.innerHTML = renderPositionToken(displayValue, row.pitcher_type);
          } else if (viewType === 'batting' && key === 'eligible') {
            const list = Array.isArray(value) ? value : [];
            const filtered = list.filter((pos) => String(pos || '').toUpperCase() !== 'DH');
            td.innerHTML = renderPositionList(filtered);
            td.classList.add('player-col', 'position-list');
          } else if (viewType === 'pitching' && key === 'pitcher_type') {
            td.innerHTML = renderPositionToken(displayValue, displayValue);
          } else {
            td.textContent = displayValue;
          }

          if (ABILITY_METRIC_CONFIG[key]) {
            const invert = (
              (viewType === 'batting' && (column.key === 'k_pct' || column.key === 'gb_pct')) ||
              (viewType === 'pitching' && (column.key === 'bb_pct' || column.key === 'hard_pct'))
            );
            applyAbilityColor(
              td,
              key,
              displayValue,
              { ...ABILITY_COLOR_PRESETS.table, invert },
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

export function updateGameLogPanel(state) {
  const container = elements.gameLogBody;
  if (!container) return;
  const log = (state && Array.isArray(state.log)) ? state.log : [];
  container.innerHTML = '';
  if (!log.length) {
    const p = document.createElement('p');
    p.classList.add('empty');
    p.textContent = 'ログはまだありません。';
    container.appendChild(p);
    return;
  }
  log.forEach((entry) => {
    const div = document.createElement('div');
    div.classList.add('log-entry');
    div.classList.add(entry?.variant || 'info');
    div.textContent = entry?.text || '';
    container.appendChild(div);
  });
  container.scrollTop = container.scrollHeight;
}

export function render(data) {
  const previousData = stateCache.data;
  stateCache.data = data;
  stateCache.analytics = normalizeAnalyticsState(data?.game?.analytics);

  // Track win probability history per play sequence using simulation output when available
  try {
    const seq = Number(stateCache.analytics?.sequence);
    const simWin = Number(stateCache.analytics?.result?.home_win_probability);
    const isWinFinite = Number.isFinite(simWin);
    if (Number.isFinite(seq) && seq >= 0 && isWinFinite) {
      const last = Number.isFinite(Number(stateCache.wpLastSeq)) ? Number(stateCache.wpLastSeq) : null;
      if (last === null || seq > last) {
        if (!Array.isArray(stateCache.wpHistory)) stateCache.wpHistory = [];
        stateCache.wpHistory.push({ sequence: seq, timestamp: stateCache.analytics?.timestamp || Date.now(), winProb: Math.max(0, Math.min(1, simWin)) });
        stateCache.wpLastSeq = seq;
      } else if (seq < last) {
        // New game or reset detected
        stateCache.wpHistory = [{ sequence: seq, timestamp: stateCache.analytics?.timestamp || Date.now(), winProb: Math.max(0, Math.min(1, simWin)) }];
        stateCache.wpLastSeq = seq;
      }
    }
  } catch (e) {
    // Non-fatal: history tracking optional
  }
  const previousSimulation = stateCache.simulation || {};

  const controlState = normalizeControlState(data?.game?.control);
  stateCache.gameControl = controlState;

  const existingSetup = stateCache.matchSetup || { mode: 'manual', userTeam: 'home' };
  const existingMode = typeof existingSetup.mode === 'string' ? existingSetup.mode : 'manual';
  let normalizedMode = ['manual', 'cpu', 'auto'].includes(existingMode) ? existingMode : 'manual';
  let normalizedUserTeam = existingSetup.userTeam === 'away' ? 'away' : 'home';

  if (normalizedMode === 'cpu' && !CONTROL_TEAM_KEYS.has(normalizedUserTeam)) {
    normalizedUserTeam = 'home';
  }
  if (normalizedMode !== 'cpu') {
    normalizedUserTeam = 'home';
  }

  if (data?.game?.active) {
    if (controlState.mode === 'cpu') {
      normalizedMode = 'cpu';
      normalizedUserTeam = controlState.userTeam || 'home';
    } else if (controlState.mode === 'auto') {
      normalizedMode = 'auto';
      normalizedUserTeam = 'home';
    } else {
      normalizedMode = 'manual';
      normalizedUserTeam = 'home';
    }
  }

  stateCache.matchSetup = {
    mode: normalizedMode,
    userTeam: normalizedUserTeam,
  };

  const teamLibraryState = data.team_library || {};
  stateCache.teamLibrary = {
    teams: Array.isArray(teamLibraryState.teams) ? teamLibraryState.teams : [],
    selection: teamLibraryState.selection || { home: null, away: null },
    ready: Boolean(teamLibraryState.ready),
    hint: teamLibraryState.hint || '',
    active: teamLibraryState.active || { home: null, away: null },
  };

  const rawSimulation = data.simulation || {};
  const rawLimits = rawSimulation.limits || {};
  const simulationLog = Array.isArray(rawSimulation.log)
    ? rawSimulation.log.filter((entry) => typeof entry === 'string').map((entry) => entry.trim()).filter(Boolean)
    : [];

  let lastRun = null;
  if (rawSimulation.last_run && typeof rawSimulation.last_run === 'object') {
    const rawLastRun = rawSimulation.last_run;
    const rawTeams = Array.isArray(rawLastRun.teams) ? rawLastRun.teams : [];
    const teamEntries = rawTeams.map((team, index) => {
      const record = team.record || {};
      const batting = team.batting || {};
      const pitching = team.pitching || {};
      const batters = Array.isArray(team.batters)
        ? team.batters.map((batter) => ({
            name: batter.name || '',
            pa: Number.isFinite(Number(batter.pa)) ? Number(batter.pa) : 0,
            ab: Number.isFinite(Number(batter.ab)) ? Number(batter.ab) : 0,
            hits: Number.isFinite(Number(batter.hits)) ? Number(batter.hits) : 0,
            singles: Number.isFinite(Number(batter.singles)) ? Number(batter.singles) : 0,
            doubles: Number.isFinite(Number(batter.doubles)) ? Number(batter.doubles) : 0,
            triples: Number.isFinite(Number(batter.triples)) ? Number(batter.triples) : 0,
            homeRuns: Number.isFinite(Number(batter.home_runs)) ? Number(batter.home_runs) : 0,
            runs: Number.isFinite(Number(batter.runs)) ? Number(batter.runs) : 0,
            rbi: Number.isFinite(Number(batter.rbi)) ? Number(batter.rbi) : 0,
            walks: Number.isFinite(Number(batter.walks)) ? Number(batter.walks) : 0,
            strikeouts: Number.isFinite(Number(batter.strikeouts)) ? Number(batter.strikeouts) : 0,
            k_pct: Number.isFinite(Number(batter.k_pct)) ? Number(batter.k_pct) : null,
            bb_pct: Number.isFinite(Number(batter.bb_pct)) ? Number(batter.bb_pct) : null,
            avg: Number.isFinite(Number(batter.avg)) ? Number(batter.avg) : 0,
            obp: Number.isFinite(Number(batter.obp)) ? Number(batter.obp) : 0,
            slg: Number.isFinite(Number(batter.slg)) ? Number(batter.slg) : 0,
            ops: Number.isFinite(Number(batter.ops)) ? Number(batter.ops) : 0,
          }))
        : [];
      const pitchers = Array.isArray(team.pitchers)
        ? team.pitchers.map((pitcher) => ({
            name: pitcher.name || '',
            appearances: Number.isFinite(Number(pitcher.appearances)) ? Number(pitcher.appearances) : 0,
            ip: Number.isFinite(Number(pitcher.ip)) ? Number(pitcher.ip) : 0,
            hits: Number.isFinite(Number(pitcher.hits)) ? Number(pitcher.hits) : 0,
            runs: Number.isFinite(Number(pitcher.runs)) ? Number(pitcher.runs) : 0,
            earnedRuns: Number.isFinite(Number(pitcher.earned_runs)) ? Number(pitcher.earned_runs) : 0,
            walks: Number.isFinite(Number(pitcher.walks)) ? Number(pitcher.walks) : 0,
            strikeouts: Number.isFinite(Number(pitcher.strikeouts)) ? Number(pitcher.strikeouts) : 0,
            homeRuns: Number.isFinite(Number(pitcher.home_runs)) ? Number(pitcher.home_runs) : 0,
            era: Number.isFinite(Number(pitcher.era)) ? Number(pitcher.era) : 0,
            whip: Number.isFinite(Number(pitcher.whip)) ? Number(pitcher.whip) : 0,
            kPer9: Number.isFinite(Number(pitcher.k_per_9)) ? Number(pitcher.k_per_9) : 0,
            bbPer9: Number.isFinite(Number(pitcher.bb_per_9)) ? Number(pitcher.bb_per_9) : 0,
          }))
        : [];

      const rankValue = Number.isFinite(Number(team.rank)) ? Number(team.rank) : index + 1;
      const aliases = Array.isArray(team.aliases)
        ? team.aliases.filter((alias) => typeof alias === 'string' && alias.trim())
        : [];
      const roles = Array.isArray(team.roles)
        ? team.roles.filter((role) => typeof role === 'string' && role.trim())
        : [];

      return {
        key: team.key || null,
        id: team.id || team.team_id || null,
        name: team.name || '',
        rank: rankValue,
        aliases,
        roles,
        record: {
          wins: Number.isFinite(Number(record.wins)) ? Number(record.wins) : 0,
          losses: Number.isFinite(Number(record.losses)) ? Number(record.losses) : 0,
          draws: Number.isFinite(Number(record.draws)) ? Number(record.draws) : 0,
          winPct: Number.isFinite(Number(record.win_pct)) ? Number(record.win_pct) : 0,
          runsScored: Number.isFinite(Number(record.runs_scored)) ? Number(record.runs_scored) : 0,
          runsAllowed: Number.isFinite(Number(record.runs_allowed)) ? Number(record.runs_allowed) : 0,
          runDiff: Number.isFinite(Number(record.run_diff)) ? Number(record.run_diff) : 0,
          games: Number.isFinite(Number(record.games)) ? Number(record.games) : null,
        },
        batting: {
          pa: Number.isFinite(Number(batting.pa)) ? Number(batting.pa) : 0,
          ab: Number.isFinite(Number(batting.ab)) ? Number(batting.ab) : 0,
          singles: Number.isFinite(Number(batting.singles)) ? Number(batting.singles) : 0,
          doubles: Number.isFinite(Number(batting.doubles)) ? Number(batting.doubles) : 0,
          triples: Number.isFinite(Number(batting.triples)) ? Number(batting.triples) : 0,
          homeRuns: Number.isFinite(Number(batting.home_runs)) ? Number(batting.home_runs) : 0,
          walks: Number.isFinite(Number(batting.walks)) ? Number(batting.walks) : 0,
          strikeouts: Number.isFinite(Number(batting.strikeouts)) ? Number(batting.strikeouts) : 0,
          hits: Number.isFinite(Number(batting.hits)) ? Number(batting.hits) : 0,
          avg: Number.isFinite(Number(batting.avg)) ? Number(batting.avg) : 0,
          obp: Number.isFinite(Number(batting.obp)) ? Number(batting.obp) : 0,
          slg: Number.isFinite(Number(batting.slg)) ? Number(batting.slg) : 0,
          ops: Number.isFinite(Number(batting.ops)) ? Number(batting.ops) : 0,
        },
        pitching: {
          ip: Number.isFinite(Number(pitching.ip)) ? Number(pitching.ip) : 0,
          hits: Number.isFinite(Number(pitching.hits_allowed)) ? Number(pitching.hits_allowed) : 0,
          runs: Number.isFinite(Number(pitching.runs_allowed)) ? Number(pitching.runs_allowed) : 0,
          earnedRuns: Number.isFinite(Number(pitching.earned_runs)) ? Number(pitching.earned_runs) : 0,
          walks: Number.isFinite(Number(pitching.walks)) ? Number(pitching.walks) : 0,
          strikeouts: Number.isFinite(Number(pitching.strikeouts)) ? Number(pitching.strikeouts) : 0,
          homeRuns: Number.isFinite(Number(pitching.home_runs)) ? Number(pitching.home_runs) : 0,
          era: Number.isFinite(Number(pitching.era)) ? Number(pitching.era) : 0,
          whip: Number.isFinite(Number(pitching.whip)) ? Number(pitching.whip) : 0,
          kPer9: Number.isFinite(Number(pitching.k_per_9)) ? Number(pitching.k_per_9) : 0,
          bbPer9: Number.isFinite(Number(pitching.bb_per_9)) ? Number(pitching.bb_per_9) : 0,
        },
        batters,
        pitchers,
      };
    });

    const games = Array.isArray(rawLastRun.games)
      ? rawLastRun.games.map((game, index) => {
          const idx = Number.isFinite(Number(game.index)) ? Number(game.index) : index + 1;
          const winner = typeof game.winner === 'string' ? game.winner : 'draw';
          return {
            index: idx,
            homeTeam: game.home_team || game.homeTeam || 'Home',
            awayTeam: game.away_team || game.awayTeam || 'Away',
            homeScore: Number.isFinite(Number(game.home_score)) ? Number(game.home_score) : 0,
            awayScore: Number.isFinite(Number(game.away_score)) ? Number(game.away_score) : 0,
            innings: Number.isFinite(Number(game.innings)) ? Number(game.innings) : 0,
            winner: ['home', 'away', 'draw'].includes(winner) ? winner : 'draw',
            day: Number.isFinite(Number(game.day)) ? Number(game.day) : null,
            card: Number.isFinite(Number(game.card)) ? Number(game.card) : null,
            round: Number.isFinite(Number(game.round)) ? Number(game.round) : null,
            cardGame: Number.isFinite(Number(game.card_game)) ? Number(game.card_game) : null,
          };
        })
      : [];

    const totalGamesValue = Number(rawLastRun.total_games);
    const computedTotalGames = Array.isArray(rawLastRun.games) ? rawLastRun.games.length : 0;

    const rawLeague = rawLastRun.league || {};
    const league = {
      totalTeams: Number.isFinite(Number(rawLeague.total_teams)) ? Number(rawLeague.total_teams) : null,
      gamesPerCard: Number.isFinite(Number(rawLeague.games_per_card))
        ? Number(rawLeague.games_per_card)
        : null,
      cardsPerOpponent: Number.isFinite(Number(rawLeague.cards_per_opponent))
        ? Number(rawLeague.cards_per_opponent)
        : null,
      completedDays: Number.isFinite(Number(rawLeague.completed_days))
        ? Number(rawLeague.completed_days)
        : null,
      totalDays: Number.isFinite(Number(rawLeague.total_days)) ? Number(rawLeague.total_days) : null,
      completedGames: Number.isFinite(Number(rawLeague.completed_games))
        ? Number(rawLeague.completed_games)
        : null,
      scheduledGames: Number.isFinite(Number(rawLeague.scheduled_games))
        ? Number(rawLeague.scheduled_games)
        : null,
    };
    const aliases = rawLastRun.aliases && typeof rawLastRun.aliases === 'object' ? rawLastRun.aliases : {};
    const roles = rawLastRun.roles && typeof rawLastRun.roles === 'object' ? rawLastRun.roles : {};

    lastRun = {
      totalGames:
        Number.isFinite(totalGamesValue) && totalGamesValue > 0 ? totalGamesValue : computedTotalGames,
      timestamp: rawLastRun.timestamp || '',
      teams: teamEntries,
      games,
      recentGames: games.slice(-5),
      mode: typeof rawLastRun.mode === 'string' ? rawLastRun.mode : null,
      league,
      aliases,
      roles,
    };
  }

  const previousLastRunTimestamp = previousSimulation?.lastRun?.timestamp || null;
  const currentLastRunTimestamp = lastRun?.timestamp || null;
  if (previousLastRunTimestamp !== currentLastRunTimestamp) {
    resetSimulationRankingsOverride();
  }

  const rawLeagueState = rawSimulation.league || {};
  const leagueState = {
    teams: Array.isArray(rawLeagueState.teams) ? rawLeagueState.teams : [],
    gamesPerCard: Number.isFinite(Number(rawLeagueState.games_per_card))
      ? Number(rawLeagueState.games_per_card)
      : null,
    cardsPerOpponent: Number.isFinite(Number(rawLeagueState.cards_per_opponent))
      ? Number(rawLeagueState.cards_per_opponent)
      : null,
  };

  const rawPlayable = rawSimulation.playable || {};
  const playableTeamsRaw = Array.isArray(rawPlayable.teams) ? rawPlayable.teams : [];
  const normalizedPlayableTeams = playableTeamsRaw
    .map((team) => {
      if (!team || typeof team !== 'object') return null;
      const id = typeof team.id === 'string' ? team.id : null;
      if (!id) return null;
      const name = typeof team.name === 'string' && team.name ? team.name : id;
      const summary = typeof team.summary === 'string' ? team.summary : '';
      const record = team.record && typeof team.record === 'object' ? team.record : {};
      const roles = Array.isArray(team.roles) ? [...team.roles] : [];
      return { id, name, summary, record, roles };
    })
    .filter(Boolean);

  const rawPlayableSelection =
    rawPlayable.selection && typeof rawPlayable.selection === 'object'
      ? rawPlayable.selection
      : {};
  const playableSelection = {
    home:
      typeof rawPlayableSelection.home === 'string' && rawPlayableSelection.home.trim()
        ? rawPlayableSelection.home.trim()
        : null,
    away:
      typeof rawPlayableSelection.away === 'string' && rawPlayableSelection.away.trim()
        ? rawPlayableSelection.away.trim()
        : null,
  };

  const previousTimestamp = stateCache.simulationMatch?.timestamp || null;
  const currentTimestamp = lastRun?.timestamp || null;
  const selectionShouldReset = stateCache.resetSimulationSelect || previousTimestamp !== currentTimestamp;

  const existingSelection = stateCache.simulationMatch?.selection || { home: null, away: null };
  const validIds = new Set(normalizedPlayableTeams.map((team) => team.id));
  let nextSelection;
  if (selectionShouldReset) {
    nextSelection = { ...playableSelection };
  } else {
    nextSelection = {
      home: validIds.has(existingSelection.home) ? existingSelection.home : playableSelection.home,
      away: validIds.has(existingSelection.away) ? existingSelection.away : playableSelection.away,
    };
  }

  stateCache.simulationMatch = {
    teams: normalizedPlayableTeams,
    selection: nextSelection,
    defaults: playableSelection,
    timestamp: currentTimestamp,
  };

  if (selectionShouldReset) {
    stateCache.resetSimulationSelect = true;
  }

  stateCache.simulation = {
    enabled: Boolean(rawSimulation.enabled),
    running: Boolean(rawSimulation.running),
    defaultGames: Number.isFinite(Number(rawSimulation.default_games))
      ? Number(rawSimulation.default_games)
      : 20,
    limits: {
      min: Number.isFinite(Number(rawLimits.min_games)) ? Number(rawLimits.min_games) : 1,
      max: Number.isFinite(Number(rawLimits.max_games)) ? Number(rawLimits.max_games) : 200,
    },
    lastRun,
    log: simulationLog.slice(-12),
    league: leagueState,
    playable: {
      teams: normalizedPlayableTeams,
      selection: playableSelection,
    },
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
  renderSimulationSetup(stateCache.teamLibrary, stateCache.simulation);
  renderSimulationResults(stateCache.simulation);
  renderSimulationMatch(stateCache.simulation);
  updateStatsPanel(data);
  updateAbilitiesPanel(data);
  updateScreenVisibility();
}
