import { elements } from '../dom.js';

export function setStatusMessage(notification) {
  if (!notification) {
    elements.statusMessage.textContent = '';
    elements.statusMessage.classList.remove('danger', 'success', 'info', 'warning');
    elements.statusMessage.removeAttribute('data-level');
    return;
  }
  const level = notification.level || 'info';
  const message = notification.message ?? '';
  elements.statusMessage.textContent = message;
  elements.statusMessage.classList.remove('danger', 'success', 'info', 'warning');
  elements.statusMessage.classList.add(level);
  elements.statusMessage.dataset.level = level;
}

export function showStatus(message, level = 'danger') {
  const resolvedMessage = message ?? '';
  elements.statusMessage.textContent = resolvedMessage;
  elements.statusMessage.classList.remove('danger', 'success', 'info', 'warning');
  elements.statusMessage.classList.add(level);
  elements.statusMessage.dataset.level = level;
}
