import { elements } from '../dom.js';
import { hideDefenseMenu, hideOffenseMenu } from './menus.js';

export function resolveModal(target) {
  if (!target) return null;
  if (target instanceof HTMLElement) return target;
  if (typeof target === 'string') {
    if (target.endsWith('-modal')) {
      return document.getElementById(target);
    }
    const key = `${target}Modal`;
    if (Object.prototype.hasOwnProperty.call(elements, key)) {
      return elements[key];
    }
    return document.getElementById(`${target}-modal`);
  }
  return null;
}

export function openModal(name) {
  const modal = resolveModal(name);
  if (!modal) return;
  hideDefenseMenu();
  hideOffenseMenu();
  modal.classList.remove('hidden');
  modal.setAttribute('aria-hidden', 'false');
  modal.dispatchEvent(new CustomEvent('show', { detail: { name } }));
}

export function closeModal(name) {
  const modal = resolveModal(name);
  if (!modal) return;
  modal.classList.add('hidden');
  modal.setAttribute('aria-hidden', 'true');
}
