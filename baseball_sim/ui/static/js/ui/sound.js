// Simple sound manager for UI SFX using Web Audio API.
// Generates a short click/tick sound on demand without external assets.

const DEFAULT_VOLUME = 0.3; // 0.0 - 1.0

let audioCtx = null;
let masterGain = null;

function ensureContext() {
  if (audioCtx) return audioCtx;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  audioCtx = new Ctx();
  masterGain = audioCtx.createGain();
  masterGain.gain.value = DEFAULT_VOLUME;
  masterGain.connect(audioCtx.destination);
  return audioCtx;
}

function resumeIfNeeded() {
  if (!audioCtx) return;
  if (audioCtx.state === 'suspended') {
    // Best-effort resume on a user gesture
    audioCtx.resume().catch(() => {});
  }
}

function playClickTone() {
  const ctx = ensureContext();
  if (!ctx) return;
  resumeIfNeeded();

  // Short percussive blip with slight randomization
  const now = ctx.currentTime;
  const duration = 0.06; // 60ms
  const baseFreq = 650;
  const freqJitter = (Math.random() - 0.5) * 80; // ±40Hz
  const freq = Math.max(220, baseFreq + freqJitter);

  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  osc.type = 'triangle';
  osc.frequency.setValueAtTime(freq, now);
  // Quick downward pitch sweep for clickiness
  osc.frequency.exponentialRampToValueAtTime(Math.max(180, freq * 0.6), now + duration);

  // Fast attack/decay envelope
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(1.0, now + 0.008);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

  osc.connect(gain);
  gain.connect(masterGain);

  osc.start(now);
  osc.stop(now + duration + 0.01);
}

export function initButtonClickSound({ selector = 'button, [role="button"]', filter } = {}) {
  // Global delegated listener so dynamically created buttons also click.
  document.addEventListener(
    'click',
    (ev) => {
      const target = ev.target instanceof Element ? ev.target : null;
      if (!target) return;
      const el = target.closest(selector);
      if (!el) return;
      // Skip disabled or aria-disabled
      if (el.matches(':disabled') || el.getAttribute('aria-disabled') === 'true') return;
      if (typeof filter === 'function' && !filter(el)) return;
      try {
        playClickTone();
      } catch (_) {
        // no-op
      }
    },
    { capture: true },
  );
}

