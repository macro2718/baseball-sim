import { elements } from '../dom.js';

export function hideOffenseMenu() {
  const { offenseMenu } = elements;
  if (!offenseMenu) return;
  if (!offenseMenu.classList.contains('hidden')) {
    offenseMenu.classList.add('hidden');
  }
  offenseMenu.setAttribute('aria-hidden', 'true');
}

export function showOffenseMenu() {
  const { offenseMenu } = elements;
  if (!offenseMenu) return;
  offenseMenu.classList.remove('hidden');
  offenseMenu.setAttribute('aria-hidden', 'false');
}

export function toggleOffenseMenu() {
  const { offenseMenu, openOffenseButton } = elements;
  if (!offenseMenu) return;
  if (openOffenseButton && openOffenseButton.disabled) {
    return;
  }
  if (offenseMenu.classList.contains('hidden')) {
    hideDefenseMenu();
    showOffenseMenu();
  } else {
    hideOffenseMenu();
  }
}

export function hideDefenseMenu() {
  const { defenseMenu } = elements;
  if (!defenseMenu) return;
  if (!defenseMenu.classList.contains('hidden')) {
    defenseMenu.classList.add('hidden');
  }
  defenseMenu.setAttribute('aria-hidden', 'true');
}

export function showDefenseMenu() {
  const { defenseMenu } = elements;
  if (!defenseMenu) return;
  defenseMenu.classList.remove('hidden');
  defenseMenu.setAttribute('aria-hidden', 'false');
}

export function toggleDefenseMenu() {
  const { defenseMenu, openDefenseButton } = elements;
  if (!defenseMenu) return;
  if (openDefenseButton && openDefenseButton.disabled) {
    return;
  }
  if (defenseMenu.classList.contains('hidden')) {
    hideOffenseMenu();
    showDefenseMenu();
  } else {
    hideDefenseMenu();
  }
}

export function toggleLogPanel() {
  const { logPanel } = elements;
  if (!logPanel) return;

  if (logPanel.classList.contains('hidden')) {
    logPanel.classList.remove('hidden');
    console.log('📝 開発者ログパネルが表示されました (Tabキーで非表示にできます)');
  } else {
    logPanel.classList.add('hidden');
    console.log('📝 開発者ログパネルが非表示になりました');
  }
}
