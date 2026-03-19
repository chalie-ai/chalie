/**
 * Animated background — drifting gradient orbs on canvas.
 * Pure visual, zero interaction with other modules.
 */
export class AmbientCanvas {
  constructor() {
    this._canvas = document.getElementById('ambientCanvas');
    this._ctx = this._canvas ? this._canvas.getContext('2d') : null;
  }

  init() {
    const canvas = this._canvas;
    const ctx = this._ctx;
    if (!canvas || !ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // Check for reduced motion preference
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (prefersReduced.matches) {
      // Draw a single static frame
      this._drawFrame(ctx, canvas, 0);
      return;
    }

    // Radiant drifting gradient blobs
    let t = 0;
    const animate = () => {
      this._drawFrame(ctx, canvas, t);
      t += 0.0012;
      requestAnimationFrame(animate);
    };
    animate();
  }

  _drawFrame(ctx, canvas, t) {
    const w = canvas.width;
    const h = canvas.height;
    const m = Math.min(w, h);

    // Near-black base. The orbs are the only light source.
    ctx.fillStyle = '#06080e';
    ctx.fillRect(0, 0, w, h);

    // Restrained orbs — think distant nebulae, not lava lamps.
    // Two warm (violet / magenta) and one cool (cyan) for contrast.
    // Low alpha keeps it atmospheric, not decorative.
    const orbs = [
      // Large violet field — dominates top-left — this IS the brand color
      { cx: 0.22, cy: 0.20, r: 0.70, color: [100, 60, 220], alpha: 0.08,
        dx: 0.07, dy: 0.06, sx: 1.0,  sy: 0.75, rBreath: 0.06, phase: 0.0  },
      // Magenta — lower-right — warm human counterpoint
      { cx: 0.78, cy: 0.65, r: 0.55, color: [180, 30, 140], alpha: 0.06,
        dx: 0.06, dy: 0.07, sx: 0.85, sy: 1.1,  rBreath: 0.06, phase: 2.4  },
      // Cyan accent — top-right — the "technology" color, small and precise
      { cx: 0.80, cy: 0.15, r: 0.30, color: [0, 180, 220],  alpha: 0.05,
        dx: 0.08, dy: 0.05, sx: 0.95, sy: 1.05, rBreath: 0.05, phase: 1.5  },
      // Deep indigo anchor — bottom — grounds the composition
      { cx: 0.40, cy: 0.90, r: 0.50, color: [60, 40, 140],  alpha: 0.06,
        dx: 0.05, dy: 0.08, sx: 1.10, sy: 0.90, rBreath: 0.06, phase: 4.2  },
    ];

    for (const orb of orbs) {
      const x = w * (orb.cx + orb.dx * Math.sin(t * orb.sx + orb.phase));
      const y = h * (orb.cy + orb.dy * Math.cos(t * orb.sy + orb.phase * 0.7));
      const r = m * orb.r * (1 + orb.rBreath * Math.sin(t * 0.5 + orb.phase));
      const a = orb.alpha * (0.75 + 0.25 * Math.cos(t * 0.3 + orb.phase));

      const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
      const [cr, cg, cb] = orb.color;
      grad.addColorStop(0,   `rgba(${cr},${cg},${cb},${a})`);
      grad.addColorStop(0.4, `rgba(${cr},${cg},${cb},${(a * 0.35).toFixed(4)})`);
      grad.addColorStop(1,   `rgba(${cr},${cg},${cb},0)`);

      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);
    }
  }
}
