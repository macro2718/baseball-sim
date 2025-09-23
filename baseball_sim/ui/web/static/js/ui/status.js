import { elements } from '../dom.js';

export function setStatusMessage(notification) {
  if (!notification) {
    elements.statusMessage.textContent = '';
    elements.statusMessage.classList.remove('danger', 'success', 'info');
    return;
  }
  elements.statusMessage.textContent = notification.message;
  elements.statusMessage.classList.remove('danger', 'success', 'info');
  elements.statusMessage.classList.add(notification.level || 'info');
}

export function showStatus(message, level = 'danger') {
  elements.statusMessage.textContent = message;
  elements.statusMessage.classList.remove('danger', 'success', 'info');
  elements.statusMessage.classList.add(level);
}
