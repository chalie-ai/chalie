/**
 * Animated background — drifting gradient orbs on canvas.
 *
 * Port of frontend/interface/ambient_canvas.js — EXACT behavioural parity:
 *  - Hardcoded orb RGB arrays (NOT from CSS custom properties).
 *  - Theme-aware background colour and alphaMul read from data-theme attribute.
 *  - prefers-reduced-motion → single static frame + redraw on theme change.
 *  - t increments by 0.0012 per rAF.
 *
 * Improvement over legacy: cleans up the rAF handle and event listeners
 * on unmount (legacy was a never-unmounted singleton and leaked them).
 */
import type { Ref } from 'vue';

interface Orb {
  cx: number;
  cy: number;
  r: number;
  color: [number, number, number];
  alpha: number;
  dx: number;
  dy: number;
  sx: number;
  sy: number;
  rBreath: number;
  phase: number;
}

const ORBS: Orb[] = [
  // Large violet field — dominates top-left
  { cx: 0.22, cy: 0.20, r: 0.70, color: [100, 60, 220], alpha: 0.08,
    dx: 0.07, dy: 0.06, sx: 1,    sy: 0.75, rBreath: 0.06, phase: 0    },
  // Magenta — lower-right — warm human counterpoint
  { cx: 0.78, cy: 0.65, r: 0.55, color: [180, 30, 140], alpha: 0.06,
    dx: 0.06, dy: 0.07, sx: 0.85, sy: 1.1,  rBreath: 0.06, phase: 2.4  },
  // Cyan accent — top-right — "technology" colour
  { cx: 0.80, cy: 0.15, r: 0.30, color: [0, 180, 220],  alpha: 0.05,
    dx: 0.08, dy: 0.05, sx: 0.95, sy: 1.05, rBreath: 0.05, phase: 1.5  },
  // Deep indigo anchor — bottom — grounds the composition
  { cx: 0.40, cy: 0.90, r: 0.50, color: [60, 40, 140],  alpha: 0.06,
    dx: 0.05, dy: 0.08, sx: 1.1,  sy: 0.9,  rBreath: 0.06, phase: 4.2  },
];

function drawFrame(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, t: number): void {
  const w = canvas.width;
  const h = canvas.height;
  const m = Math.min(w, h);

  // Theme-aware base colour — hardcoded exactly as legacy does it.
  const isLight = document.documentElement.dataset.theme === 'light';
  ctx.fillStyle = isLight ? '#F6F4F1' : '#06080e';
  ctx.fillRect(0, 0, w, h);
  const alphaMul = isLight ? 0.55 : 1;

  for (const orb of ORBS) {
    const x = w * (orb.cx + orb.dx * Math.sin(t * orb.sx + orb.phase));
    const y = h * (orb.cy + orb.dy * Math.cos(t * orb.sy + orb.phase * 0.7));
    const r = m * orb.r * (1 + orb.rBreath * Math.sin(t * 0.5 + orb.phase));
    const a = orb.alpha * (0.75 + 0.25 * Math.cos(t * 0.3 + orb.phase)) * alphaMul;

    const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
    const [cr, cg, cb] = orb.color;
    grad.addColorStop(0,   `rgba(${cr},${cg},${cb},${a})`);
    grad.addColorStop(0.4, `rgba(${cr},${cg},${cb},${(a * 0.35).toFixed(4)})`);
    grad.addColorStop(1,   `rgba(${cr},${cg},${cb},0)`);

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
  }
}

/**
 * Takes a canvas ref and starts/stops the ambient animation.
 * Returns a `stop` function — call it in `onBeforeUnmount`.
 */
export function useAmbientCanvas(canvasRef: Ref<HTMLCanvasElement | null>): () => void {
  let rafId = 0;
  let resizeHandler: (() => void) | null = null;
  let themeHandler: (() => void) | null = null;

  function start(): void {
    const canvas = canvasRef.value;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    resizeHandler = () => {
      canvas.width = globalThis.innerWidth;
      canvas.height = globalThis.innerHeight;
    };
    resizeHandler();
    globalThis.addEventListener('resize', resizeHandler);

    const prefersReduced = globalThis.matchMedia('(prefers-reduced-motion: reduce)');
    if (prefersReduced.matches) {
      drawFrame(ctx, canvas, 0);
      themeHandler = () => drawFrame(ctx, canvas, 0);
      document.addEventListener('chalie:theme-changed', themeHandler);
      return;
    }

    let t = 0;
    const animate = () => {
      drawFrame(ctx, canvas, t);
      t += 0.0012;
      rafId = requestAnimationFrame(animate);
    };
    rafId = requestAnimationFrame(animate);
  }

  function stop(): void {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    if (resizeHandler) globalThis.removeEventListener('resize', resizeHandler);
    resizeHandler = null;
    if (themeHandler) document.removeEventListener('chalie:theme-changed', themeHandler);
    themeHandler = null;
  }

  start();

  return stop;
}
