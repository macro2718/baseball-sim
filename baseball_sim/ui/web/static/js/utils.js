// General formatting helpers shared across modules.

export function escapeHtml(value) {
  if (value == null) {
    return '';
  }
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function numberOrZero(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function formatRosterStat(value, placeholder = '-') {
  if (value === null || value === undefined || value === '') {
    return escapeHtml(String(placeholder));
  }
  return escapeHtml(value);
}

export function getPositionColorClass(position, pitcherType = null) {
  if (!position) return null;
  const key = String(position).toUpperCase();
  if (key === 'SP') return 'pos-color-sp';
  if (key === 'RP') return 'pos-color-rp';
  if (key === 'C') return 'pos-color-c';
  if (['1B', '2B', '3B', 'SS'].includes(key)) return 'pos-color-infield';
  if (['LF', 'CF', 'RF'].includes(key)) return 'pos-color-outfield';
  if (key === 'P') {
    const type = pitcherType ? String(pitcherType).toUpperCase() : null;
    if (type === 'RP') return 'pos-color-rp';
    if (type === 'SP') return 'pos-color-sp';
    return 'pos-color-sp';
  }
  return null;
}

export function renderPositionToken(position, pitcherType = null, extraClass = '') {
  const classes = ['position-token'];
  if (extraClass) {
    extraClass
      .split(' ')
      .map((cls) => cls.trim())
      .filter(Boolean)
      .forEach((cls) => classes.push(cls));
  }

  if (position == null) {
    classes.push('position-token-empty');
    return `<span class="${classes.join(' ')}">-</span>`;
  }

  const token = String(position).trim();
  if (!token || token === '-') {
    classes.push('position-token-empty');
    return `<span class="${classes.join(' ')}">-</span>`;
  }

  const colorClass = getPositionColorClass(token, pitcherType);
  if (colorClass) {
    classes.push(colorClass);
  }
  return `<span class="${classes.join(' ')}">${escapeHtml(token)}</span>`;
}

export function renderPositionList(positions, pitcherType = null) {
  const tokens = [];
  if (Array.isArray(positions)) {
    positions.forEach((pos) => {
      const raw = pos == null ? '' : String(pos).trim();
      if (!raw) return;
      tokens.push(renderPositionToken(raw, pitcherType));
    });
  }

  if (!tokens.length) {
    return renderPositionToken('-', null);
  }

  return tokens.join('<span class="position-separator">, </span>');
}

export function setInsightText(element, value, fallback = '--') {
  if (!element) return;
  const textValue =
    value == null || (typeof value === 'string' && value.trim() === '') ? fallback : value;
  element.textContent = String(textValue);
}
