import { elements } from '../dom.js';

const RESULT_CATEGORY_MAP = {
  groundout: 'groundout',
  ground_out: 'groundout',
  sacrifice_bunt: 'groundout',
  bunt_out: 'groundout',
  bunt_failed: 'groundout',
  infield_flyout: 'infieldFly',
  fly_out: 'outfieldFly',
  outfield_flyout: 'outfieldFly',
  single: 'single',
  bunt_single: 'single',
  double: 'extraBase',
  triple: 'extraBase',
  home_run: 'homeRun',
};

const CATEGORY_SETTINGS = {
  groundout: {
    className: 'play-ball--groundout',
    distanceFactor: 0.42,
    angleRange: [38, 142],
    finalOpacity: 0.35,
    jitter: 0.08,
  },
  infieldFly: {
    className: 'play-ball--infield',
    distanceFactor: 0.55,
    angleRange: [46, 134],
    finalOpacity: 0.35,
    jitter: 0.12,
  },
  outfieldFly: {
    className: 'play-ball--outfield',
    distanceFactor: 0.9,
    angleRange: [34, 146],
    finalOpacity: 0.25,
    jitter: 0.14,
  },
  single: {
    className: 'play-ball--single',
    distanceFactor: 0.68,
    angleRange: [36, 144],
    finalOpacity: 0.32,
    jitter: 0.12,
  },
  extraBase: {
    className: 'play-ball--extra-base',
    distanceFactor: 0.98,
    angleRange: [32, 148],
    finalOpacity: 0.26,
    jitter: 0.15,
  },
  homeRun: {
    className: 'play-ball--home-run',
    distanceFactor: 1.05,
    angleRange: [28, 152],
    finalOpacity: 0.18,
    jitter: 0.18,
    spark: true,
  },
};

const BASE_SPEED = 0.52; // pixels per millisecond
const MIN_DURATION = 360;
const MAX_DURATION = 1400;
let lastSequence = 0;

function toNumber(value, fallback) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}

function clampDistance(layerRect, startX, startY, dx, dy, marginFactor) {
  const margin = Math.min(layerRect.width, layerRect.height) * (marginFactor ?? 0.06);
  const targetX = startX + dx;
  const targetY = startY + dy;
  let scale = 1;

  if (targetX < margin) {
    const candidate = (startX - margin) / (startX - targetX);
    if (Number.isFinite(candidate) && candidate > 0) {
      scale = Math.min(scale, candidate);
    }
  }
  if (targetX > layerRect.width - margin) {
    const candidate = (layerRect.width - margin - startX) / (targetX - startX);
    if (Number.isFinite(candidate) && candidate > 0) {
      scale = Math.min(scale, candidate);
    }
  }
  if (targetY < margin) {
    const candidate = (startY - margin) / (startY - targetY);
    if (Number.isFinite(candidate) && candidate > 0) {
      scale = Math.min(scale, candidate);
    }
  }

  if (!Number.isFinite(scale) || scale <= 0) {
    return { dx, dy };
  }

  if (scale < 1) {
    return { dx: dx * scale, dy: dy * scale };
  }
  return { dx, dy };
}

function createSpark(layer, x, y) {
  const spark = document.createElement('div');
  spark.className = 'play-ball-spark';
  spark.style.left = `${x}px`;
  spark.style.top = `${y}px`;
  layer.appendChild(spark);

  const animation = spark.animate(
    [
      { transform: 'translate(-50%, -50%) scale(0.4)', opacity: 0.85 },
      { transform: 'translate(-50%, -50%) scale(1.3)', opacity: 0 },
    ],
    { duration: 460, easing: 'ease-out' },
  );

  animation.finished
    .catch(() => undefined)
    .finally(() => {
      spark.remove();
    });
}

function performAnimation(settings) {
  const layer = elements.playAnimationLayer;
  const field = elements.baseState;
  if (!layer || !field) {
    return;
  }

  const homePlate = field.querySelector('.home-plate');
  const fieldRect = field.getBoundingClientRect();
  const layerRect = layer.getBoundingClientRect();

  let startX = layerRect.width / 2;
  let startY = layerRect.height * 0.8;
  if (homePlate) {
    const homeRect = homePlate.getBoundingClientRect();
    startX = homeRect.left + homeRect.width / 2 - layerRect.left;
    startY = homeRect.top + homeRect.height / 2 - layerRect.top;
  } else if (fieldRect.width && fieldRect.height) {
    startX = fieldRect.width / 2;
    startY = fieldRect.height * 0.8;
  }

  const fieldSize = Math.min(layerRect.width, layerRect.height);
  const jitter = settings.jitter ?? 0.1;
  let distance = fieldSize * settings.distanceFactor;
  distance += distance * (Math.random() - 0.5) * jitter;
  distance = Math.max(distance, fieldSize * 0.2);

  const [minAngle, maxAngle] = settings.angleRange;
  const angleDeg = minAngle + Math.random() * Math.max(0, maxAngle - minAngle);
  const angleRad = (angleDeg * Math.PI) / 180;

  let dx = Math.cos(angleRad) * distance;
  let dy = -Math.sin(angleRad) * distance;

  const adjusted = clampDistance(layerRect, startX, startY, dx, dy, settings.marginFactor);
  dx = adjusted.dx;
  dy = adjusted.dy;

  const endX = startX + dx;
  const endY = startY + dy;
  const actualDistance = Math.hypot(dx, dy);

  const durationRaw = actualDistance / (settings.speed ?? BASE_SPEED);
  const duration = Math.max(MIN_DURATION, Math.min(durationRaw, MAX_DURATION));
  const finalOpacity = settings.finalOpacity ?? 0.3;

  layer.innerHTML = '';
  const ball = document.createElement('div');
  const className = settings.className ? `play-ball ${settings.className}` : 'play-ball';
  ball.className = className;
  ball.style.left = `${startX}px`;
  ball.style.top = `${startY}px`;
  const trailAngle = (Math.atan2(dy, dx) * 180) / Math.PI;
  ball.style.setProperty('--trail-angle', `${trailAngle}deg`);

  layer.appendChild(ball);

  const animation = ball.animate(
    [
      { transform: 'translate(-50%, -50%)', opacity: 1 },
      {
        transform: `translate(-50%, -50%) translate(${dx}px, ${dy}px)`,
        opacity: finalOpacity,
      },
    ],
    { duration, easing: 'linear' },
  );

  animation.finished
    .then(() => {
      if (settings.spark) {
        createSpark(layer, endX, endY);
      }
    })
    .catch(() => undefined)
    .finally(() => {
      ball.remove();
    });
}

export function resetPlayAnimation() {
  lastSequence = 0;
  const layer = elements.playAnimationLayer;
  if (layer) {
    layer.innerHTML = '';
  }
}

export function triggerPlayAnimation(lastPlay, { isActive = true } = {}) {
  const layer = elements.playAnimationLayer;
  const field = elements.baseState;
  if (!layer || !field) {
    return;
  }

  if (!isActive) {
    resetPlayAnimation();
    return;
  }

  if (!lastPlay) {
    return;
  }

  const sequence = toNumber(lastPlay.sequence, null);
  const result = lastPlay.result;
  if (sequence === null || sequence <= lastSequence) {
    return;
  }
  lastSequence = sequence;

  if (!result) {
    return;
  }

  const categoryKey = RESULT_CATEGORY_MAP[result];
  if (!categoryKey) {
    return;
  }

  const settings = CATEGORY_SETTINGS[categoryKey];
  if (!settings) {
    return;
  }

  performAnimation(settings);
}
