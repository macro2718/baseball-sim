import { createGameActions } from './controllers/actions.js';
import { initEventListeners } from './controllers/events.js';
import { render } from './ui/renderers.js';
import { initButtonClickSound } from './ui/sound.js';

async function bootstrap() {
  // Initialize UI click sound effects
  initButtonClickSound();
  const actions = createGameActions(render);
  initEventListeners(actions);

  console.log('%c🏟️ Baseball Simulation - Developer Mode', 'color: #f97316; font-size: 16px; font-weight: bold;');
  console.log('%cTabキーを押すと開発者用ログパネルを表示/表示できます', 'color: #60a5fa; font-size: 14px;');

  await actions.loadInitialState();
}

bootstrap();
