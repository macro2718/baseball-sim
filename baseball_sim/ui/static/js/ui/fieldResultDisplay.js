import { elements } from '../dom.js';

const HIT_RESULTS = new Set(['single', 'double', 'triple', 'bunt_single']);
const STEAL_SUCCESS_RESULTS = new Set(['stolen_base']);
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
  'caught_stealing',
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
  stolen_base: 'Stolen Base',
  caught_stealing: 'Caught Stealing',
  // Substitution overlays
  pinch_run: 'Pinch Runner',
  pinch_hit: 'Pinch Hitter',
  defense_sub: 'Defensive Sub',
  pitching_change: 'Pitching Change',
});

const DISPLAY_DURATION = 1000;
const FADE_DURATION = 650;

let hideTimeoutId = null;
let cleanupTimeoutId = null;
let lastSequenceDisplayed = 0;

function toOrdinal(n) {
  const num = Math.trunc(Number(n) || 0);
  const mod100 = num % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${num}th`;
  switch (num % 10) {
    case 1:
      return `${num}st`;
    case 2:
      return `${num}nd`;
    case 3:
      return `${num}rd`;
    default:
      return `${num}th`;
  }
}

function normalizeHalfLabel(raw) {
  const text = (raw || '').toString().trim().toLowerCase();
  if (!text) return null;
  if (/^(表|top|t)/.test(text)) return 'Top';
  if (/^(裏|bottom|bot|b)/.test(text)) return 'Bottom';
  if (/^mid/.test(text)) return 'Mid';
  if (/^end/.test(text)) return 'End';
  return null;
}

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
  if (STEAL_SUCCESS_RESULTS.has(resultKey)) return 'walk';
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

function setDisplayContent(container, label, extras = []) {
  container.innerHTML = '';
  const wrapper = document.createElement('div');
  wrapper.className = 'field-result-text';

  const labelEl = document.createElement('span');
  labelEl.className = 'field-result-label';
  labelEl.textContent = label;
  wrapper.appendChild(labelEl);

  if (Array.isArray(extras) && extras.length > 0) {
    const extrasWrap = document.createElement('div');
    extrasWrap.className = 'field-result-extras';
    extras.forEach((ex) => {
      if (!ex || !ex.text) return;
      const badge = document.createElement('div');
      const type = ex.type || 'info';
      badge.className = `field-extra-banner field-extra--${type}`;
      badge.textContent = ex.text;
      extrasWrap.appendChild(badge);
    });
    wrapper.appendChild(extrasWrap);
  }

  container.appendChild(wrapper);
}

function showDisplay(container, category, label, extras) {
  clearTimers();
  container.classList.remove('is-visible', 'is-fading-out');
  container.dataset.category = category;
  setDisplayContent(container, label, extras);

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

  // Determine if the half-inning ended because of this play
  const prev = previousGameState || null;
  const curr = gameState || null;
  let inningEnded = false;
  if (prev && curr) {
    const prevOuts = Number.isFinite(prev.outs) ? Number(prev.outs) : null;
    const currOuts = Number.isFinite(curr.outs) ? Number(curr.outs) : null;
    const prevOffense = prev.offense || null;
    const currOffense = curr.offense || null;
    const prevHalf = (prev.half_label || prev.half || '').toString().toLowerCase();
    const currHalf = (curr.half_label || curr.half || '').toString().toLowerCase();
    const prevInning = Number(prev.inning) || null;
    const currInning = Number(curr.inning) || null;

    // Case 1: Outs reached 3 on this play
    if (prevOuts != null && currOuts === 3 && prevOuts < 3) {
      inningEnded = true;
    }
    // Case 2: Half toggled or offense switched (outs likely reset)
    if (!inningEnded) {
      const halfChanged = prevHalf && currHalf && prevHalf !== currHalf;
      const offenseChanged = prevOffense && currOffense && prevOffense !== currOffense;
      const inningAdvanced = prevInning != null && currInning != null && currInning !== prevInning;
      // Only treat as inning/half end if we advanced context compared to previous
      if (halfChanged || offenseChanged || inningAdvanced) {
        // Guard against initial render where prevOuts could be null
        if (prevOuts == null || prevOuts < 3) {
          inningEnded = true;
        }
      }
    }
  }

  const extras = [];
  if (runsScored > 0) {
    const text = `${runsScored} ${runsScored === 1 ? 'Run!' : 'Runs!'}`;
    extras.push({ type: 'score', text });
  }
  if (inningEnded) {
    let endedHalfText = null;
    let endedInning = null;
    if (prev) {
      endedHalfText = normalizeHalfLabel(prev.half_label || prev.half) || null;
      const prevInningNum = Number(prev.inning);
      endedInning = Number.isFinite(prevInningNum) ? prevInningNum : null;
    }

    if (endedHalfText && endedInning != null) {
      const ordinal = toOrdinal(endedInning);
      const text = `End of ${endedHalfText} ${ordinal}`;
      extras.push({ type: 'inning-end', text });
    } else {
      extras.push({ type: 'inning-end', text: 'Side Retired' });
    }
  }

  showDisplay(container, category, label, extras);
  lastSequenceDisplayed = sequence;
}
