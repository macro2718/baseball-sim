import { elements } from '../dom.js';

const HIT_RESULTS = new Set(['single', 'double', 'triple', 'bunt_single']);
const WALK_RESULTS = new Set(['walk']);
const STRIKEOUT_RESULTS = new Set(['strikeout']);
const HOMERUN_RESULTS = new Set(['home_run']);
const OUT_RESULTS = new Set([
  'groundout',
  'ground_out',
  'fly_out',
  'outfield_flyout',
  'infield_flyout',
  'sacrifice',
  'sacrifice_bunt',
  'bunt_out',
  'bunt_failed',
]);

const LABEL_OVERRIDES = Object.freeze({
  home_run: 'HOME RUN',
  single: 'Single',
  double: 'Double',
  triple: 'Triple',
  walk: 'Walk',
  strikeout: 'Strikeout',
  groundout: 'Out',
  ground_out: 'Out',
  fly_out: 'Out',
  outfield_flyout: 'Out',
  infield_flyout: 'Out',
  sacrifice: 'Out',
  sacrifice_bunt: 'Sacrifice Bunt',
  bunt_out: 'Bunt Out',
  bunt_failed: 'Bunt Failed',
  bunt_single: 'Bunt Single',
});

const DISPLAY_DURATION = 2400;
const FADE_DURATION = 650;

let hideTimeoutId = null;
let cleanupTimeoutId = null;
let lastSequenceDisplayed = 0;

function clearTimers() {
  if (hideTimeoutId !== null) {
    window.clearTimeout(hideTimeoutId);
    hideTimeoutId = null;
  }
  if (cleanupTimeoutId !== null) {
    window.clearTimeout(cleanupTimeoutId);
    cleanupTimeoutId = null;
  }
}

function hideDisplay() {
  const container = elements.fieldResultDisplay;
  clearTimers();
  if (!container) return;
  container.classList.remove('is-visible', 'is-fading-out');
  container.innerHTML = '';
  delete container.dataset.category;
}

export function resetFieldResultDisplay() {
  lastSequenceDisplayed = 0;
  hideDisplay();
}

function toLowerKey(value) {
  if (typeof value === 'string') {
    return value.toLowerCase();
  }
  return value == null ? '' : String(value).toLowerCase();
}

function determineCategory(resultKey) {
  if (!resultKey) return 'neutral';
  if (HOMERUN_RESULTS.has(resultKey)) return 'homerun';
  if (STRIKEOUT_RESULTS.has(resultKey)) return 'strikeout';
  if (WALK_RESULTS.has(resultKey)) return 'walk';
  if (HIT_RESULTS.has(resultKey)) return 'hit';
  if (OUT_RESULTS.has(resultKey)) return 'out';
  return 'neutral';
}

function formatLabel(resultKey) {
  if (!resultKey) return '';
  if (LABEL_OVERRIDES[resultKey]) {
    return LABEL_OVERRIDES[resultKey];
  }
  return resultKey
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function toScoreValue(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const numeric = Number.parseFloat(trimmed);
    return Number.isFinite(numeric) ? numeric : null;
  }
  return null;
}

function computeRunsScored(gameState, previousGameState, offenseHint) {
  if (!gameState || !gameState.score) return 0;
  const offenseKey = offenseHint || gameState.offense || null;
  if (!offenseKey) return 0;

  const currentScore = toScoreValue(gameState.score?.[offenseKey]);
  const previousScore = toScoreValue(previousGameState?.score?.[offenseKey]);
  if (currentScore == null || previousScore == null) {
    return 0;
  }

  const diff = currentScore - previousScore;
  return diff > 0 ? diff : 0;
}

function setDisplayContent(container, label, runsScored) {
  container.innerHTML = '';
  const wrapper = document.createElement('div');
  wrapper.className = 'field-result-text';

  const labelEl = document.createElement('span');
  labelEl.className = 'field-result-label';
  labelEl.textContent = label;
  wrapper.appendChild(labelEl);

  if (runsScored > 0) {
    const runsEl = document.createElement('span');
    runsEl.className = 'field-result-runs';
    runsEl.textContent = `${runsScored} ${runsScored === 1 ? 'Run' : 'Runs'}`;
    wrapper.appendChild(runsEl);
  }

  container.appendChild(wrapper);
}

function showDisplay(container, category, label, runsScored) {
  clearTimers();
  container.classList.remove('is-visible', 'is-fading-out');
  container.dataset.category = category;
  setDisplayContent(container, label, runsScored);

  // Force reflow so animations retrigger reliably
  // eslint-disable-next-line no-unused-expressions
  void container.offsetWidth;

  container.classList.add('is-visible');
  hideTimeoutId = window.setTimeout(() => {
    container.classList.add('is-fading-out');
    cleanupTimeoutId = window.setTimeout(() => {
      container.classList.remove('is-visible', 'is-fading-out');
      container.innerHTML = '';
      delete container.dataset.category;
    }, FADE_DURATION);
  }, DISPLAY_DURATION);
}

export function updateFieldResultDisplay(gameState, previousGameState) {
  const container = elements.fieldResultDisplay;
  if (!container) {
    if (gameState?.last_play?.sequence) {
      lastSequenceDisplayed = Number(gameState.last_play.sequence) || lastSequenceDisplayed;
    }
    return;
  }

  if (!gameState || !gameState.active) {
    resetFieldResultDisplay();
    return;
  }

  const lastPlay = gameState.last_play || null;
  if (!lastPlay || !lastPlay.result) {
    hideDisplay();
    return;
  }

  const sequence = Number(lastPlay.sequence) || 0;
  if (!sequence || sequence === lastSequenceDisplayed) {
    return;
  }

  const resultKey = toLowerKey(lastPlay.result);
  const category = determineCategory(resultKey);
  const label = formatLabel(resultKey);

  if (!label) {
    lastSequenceDisplayed = sequence;
    return;
  }

  const runsScored = computeRunsScored(gameState, previousGameState, previousGameState?.offense);
  showDisplay(container, category, label, runsScored);
  lastSequenceDisplayed = sequence;
}
