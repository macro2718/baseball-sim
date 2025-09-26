import { elements } from '../dom.js';

const SHOW_MS = 1800;
const FADE_MS = 500;

let seq = 0;

function makeBadge(text, type) {
  const div = document.createElement('div');
  div.className = 'overlay-event field-extra-banner';
  if (type === 'pitching_change') {
    div.classList.add('field-extra--inning-end');
  }
  div.textContent = text || '';
  return div;
}

export function updateOverlayEvents(events) {
  const container = elements.overlayEvents;
  if (!container || !Array.isArray(events) || events.length === 0) return;

  events.forEach((evt) => {
    const text = (evt && evt.text) || '';
    const type = (evt && evt.type) || 'info';
    const badge = makeBadge(text, type);
    const id = `ov-${Date.now()}-${seq++}`;
    badge.dataset.overlayId = id;
    container.appendChild(badge);

    // Force reflow for transition
    // eslint-disable-next-line no-unused-expressions
    void badge.offsetWidth;
    badge.classList.add('is-visible');

    window.setTimeout(() => {
      badge.classList.add('is-fading-out');
      window.setTimeout(() => {
        if (badge.parentNode) badge.parentNode.removeChild(badge);
      }, FADE_MS);
    }, SHOW_MS);
  });
}

