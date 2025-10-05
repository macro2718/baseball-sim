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
    hideGameDataMenu();
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
    console.log('📝 試合ログパネルが表示されました (試合データ→試合ログで開閉できます)');
  } else {
    logPanel.classList.add('hidden');
    console.log('📝 試合ログパネルが非表示になりました');
  }
}

// --- Game Data menu (same UX as strategy menus)
export function hideGameDataMenu() {
  const { gameDataMenu } = elements;
  if (!gameDataMenu) return;
  if (!gameDataMenu.classList.contains('hidden')) {
    gameDataMenu.classList.add('hidden');
  }
  gameDataMenu.setAttribute('aria-hidden', 'true');
}

export function showGameDataMenu() {
  const { gameDataMenu } = elements;
  if (!gameDataMenu) return;
  gameDataMenu.classList.remove('hidden');
  gameDataMenu.setAttribute('aria-hidden', 'false');
}

export function toggleGameDataMenu() {
  const { gameDataMenu, openGameDataButton } = elements;
  if (!gameDataMenu) return;
  if (openGameDataButton && openGameDataButton.disabled) {
    return;
  }
  if (gameDataMenu.classList.contains('hidden')) {
    hideOffenseMenu();
    hideDefenseMenu();
    showGameDataMenu();
  } else {
    hideGameDataMenu();
  }
}
