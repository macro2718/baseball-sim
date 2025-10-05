import { createGameActions } from './controllers/actions.js';
import { initEventListeners } from './controllers/events.js';
import { render, setLeagueAveragesForColoring } from './ui/renderers.js';
import { initButtonClickSound } from './ui/sound.js';
import { CONFIG } from './config.js';
import { apiRequest } from './services/apiClient.js';

async function bootstrap() {
  // Initialize UI click sound effects
  initButtonClickSound();
  const actions = createGameActions(render);
  initEventListeners(actions);

  console.log('%c🏟️ Baseball Simulation', 'color: #f97316; font-size: 16px; font-weight: bold;');
  console.log('%c試合データ→試合ログで試合ログパネルを開閉できます', 'color: #60a5fa; font-size: 14px;');

  try {
    const payload = await apiRequest(CONFIG.api.endpoints.leagueAverages);
    const avgs = (payload && payload.averages) || payload || null;
    if (avgs) setLeagueAveragesForColoring(avgs);
  } catch (e) {
    // Non-fatal: keep defaults if fetch fails
  }

  await actions.loadInitialState();
}

bootstrap();
