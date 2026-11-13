// solvOR gallery: pick an example, poke at it, press Solve and watch the solver get there one step at a time.
// Each example mirrors what the matching solvOR function does, so the code under the picture is the real call.
(function () {
  'use strict';

  const STEP_MS = 500;
  const W = 360, H = 260;
  const SVG_NS = 'http://www.w3.org/2000/svg';

  const reducedMotion = () => !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const fmt = (v, d = 2) => {
    if (!isFinite(v)) return '∞';
    const s = v.toFixed(d).replace(/\.?0+$/, '');
    return s === '-0' ? '0' : s;
  };
  const pyFloat = (v) => (Number.isInteger(v) ? v.toFixed(1) : String(v));
  const round2 = (v) => Math.round(v * 100) / 100;

  function svgEl(tag, attrs, parent) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  // Pointer position in viewBox coordinates.
  function svgPoint(svg, evt) {
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    const m = svg.getScreenCTM();
    return m ? pt.matrixTransform(m.inverse()) : { x: 0, y: 0 };
  }

  // ---------------------------------------------------------------------------------------------
  // Tiny Python highlighter: emits the same Pygments classes the docs theme already colours.

  const PY_NS = new Set(['from', 'import']);
  const PY_KW = new Set(['def', 'return', 'for', 'in', 'if', 'else', 'lambda', 'and', 'or', 'not', 'while']);
  const PY_KC = new Set(['None', 'True', 'False']);
  const PY_BI = new Set(['print', 'len', 'round', 'range', 'list', 'abs', 'min', 'max', 'sum']);
  const escHtml = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  function highlight(src) {
    const re = /(#[^\n]*)|("""[\s\S]*?"""|"[^"\n]*"|'[^'\n]*')|([A-Za-z_]\w*)|(\d+\.\d*(?:e-?\d+)?|\d+e-?\d+|\d+)|(\*\*|==|<=|>=|[-+*/%=<>])|([()[\]{},.:])|(\s+)|(.)/g;
    let out = '', last = '';
    let m;
    while ((m = re.exec(src)) !== null) {
      const [tok, com, str, name, num, op, punc, ws] = m;
      let cls = '';
      if (com) cls = 'c1';
      else if (str) cls = str.startsWith('"""') ? 'sd' : 's2';
      else if (name) {
        if (PY_NS.has(name)) cls = 'kn';
        else if (PY_KC.has(name)) cls = 'kc';
        else if (PY_KW.has(name)) cls = 'k';
        else if (last === 'def') cls = 'nf';
        else if (last === 'from') cls = 'nn';
        else if (PY_BI.has(name)) cls = 'nb';
        else cls = 'n';
        last = name;
      } else if (num) cls = /[.e]/.test(num) ? 'mf' : 'mi';
      else if (op) cls = 'o';
      else if (punc) cls = 'p';
      if (!name && !ws) last = tok;
      out += cls ? `<span class="${cls}">${escHtml(tok)}</span>` : escHtml(tok);
    }
    return `<div class="language-python highlight"><pre><span></span><code>${out}</code></pre></div>`;
  }

  // ---------------------------------------------------------------------------------------------
  // Glossary: [[term]] or [[shown text|term]] in any text becomes a dotted word with a tooltip.

  const GLOSSARY = {
    // Linear programming
    'objective': 'The thing being maximised, here a·x + b·y. The slider picks a and b.',
    'feasible region': 'Every (x, y) that satisfies all the constraints. For an LP it is always a convex polygon (in more dimensions: a polytope).',
    'level line': 'All points with the same objective value. Moving it in the arrow’s direction increases the objective; the optimum is where it last touches the region.',
    'slack': 'The room left in a constraint: s1 = 4 − x − y. Slack 0 means you are standing on that constraint’s edge (it is tight).',
    'basis': 'The variables allowed to be non-zero at the current corner. With 2 constraints there are 2 of them; the rest sit at 0.',
    'pivot': 'One simplex step: one variable enters the basis, one leaves, and you slide along an edge to the neighbouring corner.',
    'enter': 'The entering variable goes from 0 to positive because raising it improves the objective.',
    'bland': 'Pick the lowest-numbered variable that improves the objective, not the one that improves it most. Sometimes a detour, but it can never cycle forever.',
    'reduced cost': 'How much the objective would change per unit of a variable that is currently 0. When none of them improves things, the corner is optimal.',
    'analytic centre': 'The point that maximises the product of all slacks: as far from every wall as possible.',
    'barrier': 'A penalty, −μ·log(slack), that grows to infinity at the walls. It keeps the point strictly inside the region.',
    'mu': 'The barrier weight. Big μ: stay away from the walls. μ → 0: only the objective matters. Each step shrinks it.',
    'central path': 'The curve of best points for every value of μ. Interior point methods follow it from the centre to the optimum.',
    'gap': 'The duality gap: how far the current objective can still be from the true optimum. It shrinks with μ (here: 4·μ).',
    'solvers': 'HiGHS (open source), Gurobi and CPLEX are the industrial LP solvers. They run simplex and interior point, often both at once.',
    // A*
    'frontier': 'Cells that have been seen but not yet expanded, waiting in a priority queue sorted by f.',
    'expanded': 'Taken off the frontier and finished: its neighbours have been checked and its distance is final.',
    'f': 'g = steps from S so far, h = guess of the steps left (Manhattan distance), f = g + h = estimated total. Lowest f goes first.',
    'admissible': 'A heuristic that never overestimates the remaining cost. That is what guarantees A* still finds the shortest path.',
    'Dijkstra': 'A* with h = 0: no guess, so it spreads out evenly in all directions. Same answer, more work.',
    // Tabu search
    'greedy': 'Take the best-looking option right now and never reconsider. Fast, rarely optimal.',
    '2-opt': 'Remove two edges and reconnect the two pieces the other way round, which reverses a stretch of the tour. Removes a crossing every time.',
    'local optimum': 'No single move makes it better, but a better solution may still exist further away.',
    'tabu': 'Forbidden for a while. Recently used moves go on a tabu list (here for 10 iterations), so the search cannot undo them and walk in circles. A tabu move is still allowed if it beats the best ever.',
    'tour length': 'Sum of all edge lengths, in screen pixels. Shorter is better.',
    // Nelder-Mead
    'nm-simplex': 'A triangle in 2D, n + 1 points in n dimensions. Nothing to do with the simplex method for LPs, despite the name.',
    'reflect': 'Mirror the worst corner through the middle of the other two.',
    'expand': 'The reflected point was the best so far, so try going twice as far in that direction.',
    'contract': 'The reflected point was only so-so: settle for a point halfway between the middle and the reflection.',
    'contract-inside': 'The reflected point was worse than the worst corner: move the worst corner halfway toward the middle instead.',
    'shrink': 'Every move failed, so pull all corners halfway toward the best one and try again on a smaller scale.',
    'tol': 'Stop when the best and worst corner values differ by less than this (tol = 1e-4).',
    'himmelblau': 'f(x, y) = (x² + y − 11)² + (x + y² − 7)². A classic test function with four minima, all with f = 0.',
    'contour': 'Like height lines on a map: f is the same all along one line. Rings close together mean a steep slope.',
  };

  function rich(text) {
    return String(text).split(/(\[\[[^\]]+\]\])/).map((part) => {
      const m = /^\[\[([^|\]]+)(?:\|([^\]]+))?\]\]$/.exec(part);
      if (!m) return escHtml(part);
      const key = m[2] || m[1];
      return GLOSSARY[key] ? `<span class="sg-term" tabindex="0" data-term="${key}">${escHtml(m[1])}</span>` : escHtml(m[1]);
    }).join('');
  }

  // ---------------------------------------------------------------------------------------------
  // Example 1: linear programming, simplex or interior point.
  // maximize c.x  subject to  x + y <= 4,  x + 3y <= 6,  x, y >= 0

  function lpExample() {
    const A = [[1, 1], [1, 3]], B = [4, 6];
    const CORNERS = [[0, 0], [4, 0], [3, 1], [0, 2]];
    const ROWS_TXT = ['x + y ≤ 4', 'x + 3y ≤ 6'];
    const S = 66, OX = 30, OY = H - 30; // x in [0, 4.6], y in [0, 3.3]: the region is wide and flat
    const px = ([x, y]) => [OX + x * S, OY - y * S];
    const DEFAULT_ANGLE = 60;
    const state = { angle: DEFAULT_ANGLE, method: 'simplex' };
    let g = null;
    const pt3 = (p) => `(${p[0].toFixed(4)}, ${p[1].toFixed(4)})`;

    const objective = () => {
      const t = (state.angle * Math.PI) / 180;
      return [round2(Math.cos(t)), round2(Math.sin(t))];
    };
    // Corners that share the best objective value: more than one means a whole edge is optimal.
    const bestCorners = (c) => {
      const zs = CORNERS.map((v) => c[0] * v[0] + c[1] * v[1]), top = Math.max(...zs);
      return CORNERS.filter((_, i) => top - zs[i] < 1e-6 + 1e-3 * Math.abs(top));
    };

    // Tableau simplex with Bland's rule, the same pivoting rule solve_lp uses.
    function simplexTrace(c) {
      const names = ['x', 'y', 's1', 's2'];
      const T = A.map((row, i) => [row[0], row[1], i === 0 ? 1 : 0, i === 1 ? 1 : 0, B[i]]);
      const r = [-c[0], -c[1], 0, 0];
      const basis = [2, 3];
      const point = () => { const p = [0, 0]; basis.forEach((b, i) => { if (b < 2) p[b] = T[i][4]; }); return p; };
      const frames = [{ pt: [0, 0], z: 0, msg: 'Start at the origin, x = y = 0. All the room sits in the [[slacks|slack]] s1 and s2, which form the [[basis]]. Boring, but it is a corner.' }];
      for (let pivot = 1; pivot <= 10; pivot++) {
        const j = r.findIndex((v) => v < -1e-9);
        if (j < 0) break;
        let row = -1, best = Infinity;
        T.forEach((tr, i) => {
          if (tr[j] > 1e-9) {
            const ratio = tr[4] / tr[j];
            if (ratio < best - 1e-12 || (Math.abs(ratio - best) <= 1e-12 && basis[i] < basis[row])) { best = ratio; row = i; }
          }
        });
        if (row < 0) break;
        const leaving = basis[row], pv = T[row][j];
        T[row] = T[row].map((v) => v / pv);
        T.forEach((tr, i) => { if (i !== row) { const f = tr[j]; T[i] = tr.map((v, k) => v - f * T[row][k]); } });
        const f = r[j];
        for (let k = 0; k < 4; k++) r[k] -= f * T[row][k];
        basis[row] = j;
        const p = point();
        const why = leaving >= 2 ? `it hits 0: ${ROWS_TXT[leaving - 2]} is now tight` : 'it hits 0';
        frames.push({ pt: p, z: c[0] * p[0] + c[1] * p[1],
          msg: `[[Pivot|pivot]] ${pivot}: ${names[j]} [[enters|enter]] (the first variable that still pays, by [[Bland’s rule|bland]]) and ${names[leaving]} leaves (${why}). Walk along the edge to (${fmt(p[0])}, ${fmt(p[1])}).` });
      }
      const end = frames[frames.length - 1];
      const ties = bestCorners(c);
      const n = frames.length - 1, pivots = `${n} pivot${n === 1 ? '' : 's'}`;
      frames.push({ pt: end.pt, z: end.z, done: true, msg: ties.length > 1
        ? `Nothing uphill left, and the corners (${ties.map((v) => v.join(', ')).join(') and (')}) tie: the whole edge between them is optimal. Simplex reports the corner it reached. ${pivots}.`
        : `No neighbouring corner is better (every [[reduced cost]] says "don't"), so (${fmt(end.pt[0])}, ${fmt(end.pt[1])}) is optimal. ${pivots}, done.` });
      return frames;
    }

    // Log-barrier path following: maximize t*c.x + sum(log slack), raising t each step.
    // solve_lp_interior uses the primal-dual version of this idea; both follow the central path.
    function interiorTrace(c) {
      const G = [[1, 1], [1, 3], [-1, 0], [0, -1]], h = [4, 6, 0, 0];
      const slacks = (x) => G.map((gi, i) => h[i] - gi[0] * x[0] - gi[1] * x[1]);
      const phi = (x, t) => { const s = slacks(x); return s.some((v) => v <= 0) ? -Infinity : t * (c[0] * x[0] + c[1] * x[1]) + s.reduce((a, v) => a + Math.log(v), 0); };
      function centre(x, t) {
        for (let it = 0; it < 60; it++) {
          const s = slacks(x);
          const gr = [t * c[0], t * c[1]], M = [0, 0, 0];
          G.forEach((gi, i) => {
            gr[0] -= gi[0] / s[i]; gr[1] -= gi[1] / s[i];
            const w = 1 / (s[i] * s[i]);
            M[0] += w * gi[0] * gi[0]; M[1] += w * gi[0] * gi[1]; M[2] += w * gi[1] * gi[1];
          });
          const det = M[0] * M[2] - M[1] * M[1];
          const dx = [(M[2] * gr[0] - M[1] * gr[1]) / det, (M[0] * gr[1] - M[1] * gr[0]) / det];
          let a = 1;
          const f0 = phi(x, t);
          while (a > 1e-12 && phi([x[0] + a * dx[0], x[1] + a * dx[1]], t) < f0) a /= 2;
          x = [x[0] + a * dx[0], x[1] + a * dx[1]];
          if (Math.hypot(dx[0], dx[1]) * a < 1e-12) break;
        }
        return x;
      }
      let x = centre([1, 0.5], 0);
      const frames = [{ pt: x, z: c[0] * x[0] + c[1] * x[1], mu: Infinity,
        msg: `Start at the [[analytic centre]] ${pt3(x)}, as far from every wall as it gets.` }];
      for (let t = 0.5; ; t *= 2.5) {
        x = centre(x, t);
        const gap = 4 / t;
        const z = c[0] * x[0] + c[1] * x[1];
        if (gap < 1e-3) {
          const ties = bestCorners(c);
          frames.push({ pt: x, z, mu: 1 / t, gap, done: true, msg: ties.length > 1
            ? `[[Gap|gap]] below 0.001: done. Two corners tie, so it settles in the middle of the optimal edge at ${pt3(x)}, not on a corner.`
            : `[[Gap|gap]] below 0.001: close enough. It stops at ${pt3(x)}: a hair inside the region, never exactly on the corner.` });
          break;
        }
        const sig = (v) => (v < 0.01 ? v.toPrecision(2) : fmt(v, 3));
        frames.push({ pt: x, z, mu: 1 / t, gap, msg: `[[μ|mu]] = ${sig(1 / t)}: the [[barrier]] relaxes and the point slides along the [[central path]]. Gap now ${sig(gap)}.` });
      }
      return frames;
    }

    return {
      tab: 'Simplex',
      get title() { return state.method === 'simplex' ? 'Simplex: walk the edges' : 'Interior point: cut through the middle'; },
      get hint() {
        return state.method === 'simplex'
          ? 'Turn the objective with the slider, then press Solve. Simplex starts in a corner and climbs from corner to corner until nothing is uphill anymore.'
          : 'Same problem, other route. Simplex takes the stairs (corner to corner along the edges), interior point takes the elevator (straight through the middle).';
      },
      get note() {
        return state.method === 'simplex'
          ? 'The optimum of a linear program always sits on a corner, so simplex never bothers with the inside. The dashed [[level line]] is every point with the same objective value: slide it in the arrow’s direction and the optimum is the last corner it touches. Try 45°, where it lies flat along an edge.'
          : 'A [[barrier]] term pushes the point away from the walls. Each step weakens it ([[μ|mu]] shrinks) and the point drifts toward the optimum along the [[central path]]. Production solvers like [[HiGHS and Gurobi|solvers]] do this on millions of variables.';
      },
      aria: 'Feasible region of a two-variable linear program with the solver path drawn on it',
      legend: [
        ['rect', 'sg-region', '[[feasible region]]'],
        ['line', 'sg-level', '[[level line]]'],
        ['line', 'sg-trail', 'path so far'],
        ['dot', 'sg-corner', 'corner'],
        ['dot', 'sg-corner is-best', 'optimum'],
      ],
      restore() { state.angle = DEFAULT_ANGLE; },

      mount({ svg, controls, changed }) {
        const grid = svgEl('g', { class: 'sg-grid' }, svg);
        for (let i = 1; i <= 4; i++) svgEl('line', { x1: px([i, 0])[0], y1: px([0, 0])[1], x2: px([i, 0])[0], y2: px([0, 3.2])[1] }, grid);
        for (let i = 1; i <= 3; i++) svgEl('line', { x1: px([0, 0])[0], y1: px([0, i])[1], x2: px([4.5, 0])[0], y2: px([0, i])[1] }, grid);
        for (let i = 1; i <= 4; i++) svgEl('text', { class: 'sg-tick', x: px([i, 0])[0], y: OY + 14, 'text-anchor': 'middle' }, svg).textContent = i;
        for (let i = 1; i <= 3; i++) svgEl('text', { class: 'sg-tick', x: OX - 8, y: px([0, i])[1] + 3.5, 'text-anchor': 'end' }, svg).textContent = i;
        svgEl('polygon', { class: 'sg-region', points: CORNERS.map((v) => px(v).join(',')).join(' ') }, svg);
        const lab1 = px([3.72, 0.6]), lab2 = px([0.45, 2.08]);
        svgEl('text', { class: 'sg-label', x: lab1[0], y: lab1[1],
          'data-tip': '<strong>x + y ≤ 4</strong> Its slack s1 = 4 − x − y is the room left. On this edge s1 = 0: the constraint is tight.' }, svg).textContent = 'x + y ≤ 4';
        svgEl('text', { class: 'sg-label', x: lab2[0], y: lab2[1],
          'data-tip': '<strong>x + 3y ≤ 6</strong> Its slack s2 = 6 − x − 3y is the room left. On this edge s2 = 0: the constraint is tight.' }, svg).textContent = 'x + 3y ≤ 6';
        const marker = svgEl('marker', { id: 'sg-arrowhead', viewBox: '0 0 8 8', refX: 6, refY: 4, markerWidth: 8, markerHeight: 8, orient: 'auto' }, svgEl('defs', {}, svg));
        svgEl('path', { class: 'sg-arrowhead', d: 'M0,0L8,4L0,8z' }, marker);
        g = {
          level: svgEl('line', { class: 'sg-level' }, svg),
          arrow: svgEl('line', { class: 'sg-arrow', 'marker-end': 'url(#sg-arrowhead)',
            'data-tip': '<strong>Objective direction</strong> The way the objective grows. The slider turns it.' }, svg),
          path: svgEl('polyline', { class: 'sg-trail' }, svg),
          dots: svgEl('g', {}, svg),
          corners: CORNERS.map((v, i) => svgEl('circle', { class: 'sg-corner', 'data-i': i, cx: px(v)[0], cy: px(v)[1], r: 5 }, svg)),
          here: svgEl('circle', { class: 'sg-here', r: 7 }, svg),
        };
        svgEl('text', { class: 'sg-axis', x: px([4.6, 0])[0], y: px([0, 0])[1] + 4 }, svg).textContent = 'x';
        svgEl('text', { class: 'sg-axis', x: px([0, 0])[0] - 4, y: px([0, 3.3])[1] }, svg).textContent = 'y';

        controls.innerHTML = `
          <div class="sg-seg" role="group" aria-label="Method">
            <button type="button" data-m="simplex">Simplex</button><button type="button" data-m="interior">Interior point</button>
          </div>
          <label class="sg-field">Objective direction <span class="sg-angle"></span>
            <input class="sg-range" type="range" min="0" max="90" step="1" value="${state.angle}" aria-label="Objective direction in degrees">
          </label>
          <div class="sg-ends" aria-hidden="true"><span>0°: maximise x</span><span>90°: maximise y</span></div>`;
        const seg = controls.querySelectorAll('.sg-seg button');
        const syncSeg = () => seg.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.m === state.method)));
        seg.forEach((b) => b.addEventListener('click', () => {
          if (state.method !== b.dataset.m) { state.method = b.dataset.m; syncSeg(); changed({ text: true, restart: true }); }
        }));
        syncSeg();
        const angle = controls.querySelector('.sg-angle');
        const range = controls.querySelector('.sg-range');
        const syncAngle = () => { angle.textContent = `${state.angle}°`; range.value = state.angle; };
        range.addEventListener('input', (e) => { state.angle = +e.target.value; syncAngle(); changed(); });
        syncAngle();
      },

      trace() { const c = objective(); return state.method === 'simplex' ? simplexTrace(c) : interiorTrace(c); },

      draw(frames, k) {
        const c = objective(), f = frames[k];
        const [bx, by] = px(f.pt);
        const n = Math.hypot(c[0], c[1]) || 1, d = [-c[1] / n, c[0] / n], L = 3.4 * S;
        Object.entries({ x1: bx - d[0] * L, y1: by + d[1] * L, x2: bx + d[0] * L, y2: by - d[1] * L }).forEach(([a, v]) => g.level.setAttribute(a, v));
        g.level.setAttribute('data-tip', `<strong>Level line</strong> Every point on it has objective ${fmt(f.z, 2)}. Further in the arrow’s direction means better.`);
        const [ox, oy] = px([0, 0]);
        Object.entries({ x1: ox, y1: oy, x2: ox + (c[0] / n) * 1.3 * S, y2: oy - (c[1] / n) * 1.3 * S }).forEach(([a, v]) => g.arrow.setAttribute(a, v));
        g.path.setAttribute('points', frames.slice(0, k + 1).map((fr) => px(fr.pt).join(',')).join(' '));
        g.dots.innerHTML = '';
        if (state.method === 'interior') frames.slice(0, k).forEach((fr) => { const p = px(fr.pt); svgEl('circle', { class: 'sg-dot', cx: p[0], cy: p[1], r: 2.5 }, g.dots); });
        g.here.setAttribute('cx', bx); g.here.setAttribute('cy', by);
        g.here.classList.toggle('is-done', !!f.done);
        CORNERS.forEach((v, i) => g.corners[i].classList.toggle('is-best', !!f.done && state.method === 'simplex' && Math.hypot(v[0] - f.pt[0], v[1] - f.pt[1]) < 1e-6));
      },

      stats(frames, k) {
        const f = frames[k], c = objective();
        const at = state.method === 'interior' ? pt3(f.pt) : `(${fmt(f.pt[0])}, ${fmt(f.pt[1])})`;
        const rows = [['maximise', `${fmt(c[0])}x + ${fmt(c[1])}y`], ['at', at], ['[[objective]]', fmt(f.z, 3)]];
        if (state.method === 'interior') rows.push(['[[μ|mu]]', k === 0 ? '∞' : f.mu < 0.01 ? f.mu.toPrecision(2) : fmt(f.mu, 3)]);
        return rows;
      },

      tipAt(target, pt) {
        const c = objective();
        const ci = target.dataset && target.dataset.i;
        const r1 = (v) => Math.round(v * 10) / 10 || 0;
        const [x, y] = ci !== undefined ? CORNERS[+ci] : [r1((pt.x - OX) / S), r1((OY - pt.y) / S)];
        if (x < -0.15 || y < -0.15 || x > 4.7 || y > 3.4) return null;
        const z = c[0] * x + c[1] * y;
        const broken = [];
        if (x + y > 4 + 1e-9) broken.push('x + y ≤ 4');
        if (x + 3 * y > 6 + 1e-9) broken.push('x + 3y ≤ 6');
        if (x < -1e-9 || y < -1e-9) broken.push('x, y ≥ 0');
        const head = ci !== undefined ? `<strong>Corner (${fmt(x)}, ${fmt(y)})</strong>` : `<strong>(${fmt(x, 1)}, ${fmt(y, 1)})</strong>`;
        const val = `objective ${fmt(c[0])}·${fmt(x, 1)} + ${fmt(c[1])}·${fmt(y, 1)} = ${fmt(z, 2)}`;
        const s1 = 4 - x - y, s2 = 6 - x - 3 * y;
        return broken.length ? `${head} outside: breaks ${broken.join(' and ')}<br>${val}`
          : `${head} feasible<br>${val}<br><span class="sg-tip-k">slacks: s1 = ${fmt(s1, 1)}, s2 = ${fmt(s2, 1)}</span>`;
      },

      code(frames, solved) {
        const c = objective(), end = frames[frames.length - 1];
        const fn = state.method === 'simplex' ? 'solve_lp' : 'solve_lp_interior';
        // When two corners tie, a whole edge is optimal and interior point may stop anywhere on it.
        const tied = bestCorners(c).length > 1;
        const sol = state.method === 'simplex' ? `x = ${fmt(end.pt[0])}, y = ${fmt(end.pt[1])}`
          : tied ? 'somewhere on the optimal edge: two corners tie' : `about x = ${fmt(end.pt[0])}, y = ${fmt(end.pt[1])}`;
        return [
          `from solvor import ${fn}`,
          '',
          `# maximize ${fmt(c[0])}x + ${fmt(c[1])}y  subject to  x + y <= 4,  x + 3y <= 6,  x, y >= 0`,
          `result = ${fn}([${c.join(', ')}], [[1, 1], [1, 3]], [4, 6], minimize=False)`,
          `print(result.solution)${solved ? `   # ${sol}` : ''}`,
          `print(round(result.objective, 2))${solved ? `  # ${fmt(end.z, 2)}` : ''}`,
        ].join('\n');
      },
    };
  }

  // ---------------------------------------------------------------------------------------------
  // Example 2: A* on a grid, mirroring astar_grid (4 directions, Manhattan heuristic).

  function astarExample() {
    const ROWS = 13, COLS = 18, CELL = 20;
    const key = (r, c) => r * COLS + c;
    // Two walls with gaps at opposite ends: the straight line is blocked, so the guess has to be wrong for a while.
    const state = {};
    function restore() {
      state.start = [6, 1]; state.goal = [6, 16]; state.walls = new Set();
      for (let r = 2; r <= 12; r++) state.walls.add(key(r, 5));
      for (let r = 0; r <= 10; r++) state.walls.add(key(r, 12));
      for (let c = 6; c <= 9; c++) state.walls.add(key(7, c));
    }
    restore();
    let g = null;
    const DIRS = [[-1, 0], [0, -1], [0, 1], [1, 0]]; // the order astar_grid generates neighbours in

    // guided = false turns the heuristic off, which makes this Dijkstra: used for the comparison count.
    function search(guided = true) {
      const [gr, gc] = state.goal;
      const h = (r, c) => (guided ? Math.abs(r - gr) + Math.abs(c - gc) : 0);
      const gScore = new Map([[key(...state.start), 0]]);
      const parent = new Map();
      const closed = new Set();
      const heap = [[h(...state.start), -0, 0, state.start]];
      let counter = 1;
      const order = [], found = [];
      const less = (a, b) => a[0] !== b[0] ? a[0] < b[0] : a[1] !== b[1] ? a[1] < b[1] : a[2] < b[2];
      while (heap.length) {
        let bi = 0;
        for (let i = 1; i < heap.length; i++) if (less(heap[i], heap[bi])) bi = i;
        const [, , , cur] = heap.splice(bi, 1)[0];
        const ck = key(...cur);
        if (closed.has(ck)) continue;
        closed.add(ck);
        const discovered = [];
        order.push(ck); found.push(discovered);
        if (cur[0] === gr && cur[1] === gc) {
          const path = [ck];
          let p = ck;
          while (parent.has(p)) { p = parent.get(p); path.push(p); }
          return { order, found, path: path.reverse(), cost: gScore.get(ck) };
        }
        for (const [dr, dc] of DIRS) {
          const nr = cur[0] + dr, nc = cur[1] + dc;
          if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) continue;
          const nk = key(nr, nc);
          if (state.walls.has(nk) || closed.has(nk)) continue;
          const tg = gScore.get(ck) + 1;
          if (tg < (gScore.has(nk) ? gScore.get(nk) : Infinity)) {
            gScore.set(nk, tg); parent.set(nk, ck);
            heap.push([tg + h(nr, nc), -tg, counter++, [nr, nc]]);
            discovered.push([nk, tg]); // the cell and the g it got, for the hover tooltip
          }
        }
      }
      return { order, found, path: null, cost: Infinity };
    }

    return {
      tab: 'A*',
      title: 'A*: search with a sense of direction',
      hint: 'Click or drag on empty cells to draw walls, click a wall to remove it, drag S or G to move them. Then press Solve.',
      note: 'Dijkstra spreads out like ripples in a pond. A* sorts its [[frontier]] by [[f = g + h|f]]: steps taken so far plus a guess of the steps left. The guess pulls the search toward G, and because the guess never overestimates ([[admissible]]), the path is still the shortest one. See the boxed-in bay above the short wall arm? From there G looks close, so the guess is too optimistic and A* fills that dead end before it believes the detour.',
      legend: [
        ['rect', 'sg-cell is-wall', 'wall'],
        ['rect', 'sg-cell is-closed', '[[expanded]]'],
        ['rect', 'sg-cell is-open', '[[frontier]]'],
        ['line', 'sg-route', 'shortest path'],
      ],
      restore,
      aria: 'Grid maze with the A* search frontier and path',

      mount({ svg, changed }) {
        const cells = [];
        const layer = svgEl('g', {}, svg);
        for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
          cells.push(svgEl('rect', { class: 'sg-cell', x: c * CELL + 0.5, y: r * CELL + 0.5, width: CELL - 1, height: CELL - 1 }, layer));
        }
        g = {
          cells,
          path: svgEl('polyline', { class: 'sg-route' }, svg),
          s: svgEl('g', { class: 'sg-marker' }, svg),
          g: svgEl('g', { class: 'sg-marker' }, svg),
        };
        [[g.s, 'S'], [g.g, 'G']].forEach(([m, label]) => { svgEl('circle', { r: 8 }, m); svgEl('text', { y: 4.5 }, m).textContent = label; });

        let mode = null;
        const cellAt = (evt) => {
          const p = svgPoint(svg, evt);
          const r = Math.floor(p.y / CELL), c = Math.floor(p.x / CELL);
          return r >= 0 && r < ROWS && c >= 0 && c < COLS ? [r, c] : null;
        };
        const same = (a, b) => a[0] === b[0] && a[1] === b[1];
        const apply = (rc) => {
          if (!rc) return;
          const k = key(...rc);
          if (mode === 'start' || mode === 'goal') {
            const other = mode === 'start' ? state.goal : state.start;
            if (state.walls.has(k) || same(rc, other) || same(rc, state[mode])) return;
            state[mode] = rc;
          } else {
            if (same(rc, state.start) || same(rc, state.goal)) return;
            if (mode === 'add' ? state.walls.has(k) : !state.walls.has(k)) return;
            if (mode === 'add') state.walls.add(k); else state.walls.delete(k);
          }
          changed();
        };
        svg.addEventListener('pointerdown', (evt) => {
          const rc = cellAt(evt);
          if (!rc) return;
          mode = same(rc, state.start) ? 'start' : same(rc, state.goal) ? 'goal' : state.walls.has(key(...rc)) ? 'erase' : 'add';
          svg.setPointerCapture(evt.pointerId);
          evt.preventDefault();
          apply(rc);
        });
        svg.addEventListener('pointermove', (evt) => { if (mode) apply(cellAt(evt)); });
        const end = () => { mode = null; };
        svg.addEventListener('pointerup', end);
        svg.addEventListener('pointercancel', end);
      },

      trace() {
        const res = search();
        const n = res.order.length, chunk = Math.max(1, Math.ceil(n / 14));
        res.dijkstra = search(false).order.length;
        const per = chunk === 1 ? 'one cell per step' : `${chunk} cells per step here`;
        const frames = [{ expanded: 0, res, msg: 'The [[frontier]] holds just S. Nothing explored yet.' }];
        for (let e = chunk; e < n; e += chunk) {
          frames.push({ expanded: e, res, msg: e === chunk
            ? `[[Expanded|expanded]] ${e} cells (${per}), always the frontier cell with the lowest [[f = g + h|f]] first.`
            : `[[Expanded|expanded]] ${e} cells. Lowest [[f]] first, so the search leans toward G until a wall says otherwise.` });
        }
        frames.push({ expanded: n, res, done: true,
          msg: res.path ? `G came off the [[frontier]] after ${n} expansions, so its path is final: ${res.cost} steps. [[Dijkstra]] would have expanded ${res.dijkstra}.`
            : `Explored all ${n} reachable cells. No way through: the walls cut G off.` });
        return frames;
      },

      draw(frames, k) {
        const f = frames[k], { order, found, path } = f.res;
        const closed = new Set(order.slice(0, f.expanded));
        const frontier = new Set(f.expanded === 0 ? [key(...state.start)] : []);
        for (let i = 0; i < f.expanded; i++) found[i].forEach(([c]) => frontier.add(c));
        g.cells.forEach((rect, i) => {
          rect.setAttribute('class', state.walls.has(i) ? 'sg-cell is-wall' : closed.has(i) ? 'sg-cell is-closed' : frontier.has(i) ? 'sg-cell is-open' : 'sg-cell');
        });
        const centre = (i) => `${(i % COLS) * CELL + CELL / 2},${Math.floor(i / COLS) * CELL + CELL / 2}`;
        g.path.setAttribute('points', f.done && path ? path.map(centre).join(' ') : '');
        [[g.s, state.start], [g.g, state.goal]].forEach(([m, [r, c]]) => m.setAttribute('transform', `translate(${c * CELL + CELL / 2},${r * CELL + CELL / 2})`));
      },

      stats(frames, k) {
        const f = frames[k];
        return [['[[expanded]]', `${f.expanded}`], ['path', f.done ? (f.res.path ? `${f.res.cost} steps` : 'none') : '…'],
          ['[[Dijkstra]]', f.done ? `${f.res.dijkstra} expanded` : '…']];
      },

      tipAt(target, pt, frames, k) {
        const r = Math.floor(pt.y / CELL), c = Math.floor(pt.x / CELL);
        if (r < 0 || r >= ROWS || c < 0 || c >= COLS) return null;
        const i = key(r, c), f = frames[k], { order, found } = f.res;
        const h = Math.abs(r - state.goal[0]) + Math.abs(c - state.goal[1]);
        if (state.walls.has(i)) return 'Wall. Click it to knock it down.';
        let gv = r === state.start[0] && c === state.start[1] ? 0 : null;
        for (let e = 0; e < f.expanded; e++) for (const [cell, gg] of found[e]) if (cell === i) gv = gg;
        const isS = r === state.start[0] && c === state.start[1], isG = r === state.goal[0] && c === state.goal[1];
        const where = isS ? 'S, the start' : isG ? 'G, the goal' : `row ${r}, column ${c}`;
        const closed = order.slice(0, f.expanded).includes(i);
        const status = closed ? 'expanded: done, never looked at again' : gv !== null ? 'on the frontier: waiting its turn' : 'not reached yet';
        const nums = gv !== null ? `g = ${gv} <span class="sg-tip-k">(steps from S)</span><br>h = ${h} <span class="sg-tip-k">(guess to G)</span><br>f = g + h = ${gv + h}`
          : `h = ${h} <span class="sg-tip-k">(Manhattan distance to G)</span>`;
        return `<strong>${where}</strong> ${status}<br>${nums}${isS || isG ? '<br><span class="sg-tip-k">Drag to move it.</span>' : ''}`;
      },

      code(frames, solved) {
        const res = frames[0].res;
        const rows = [];
        for (let r = 0; r < ROWS; r++) {
          let line = '';
          for (let c = 0; c < COLS; c++) {
            line += r === state.start[0] && c === state.start[1] ? 'S' : r === state.goal[0] && c === state.goal[1] ? 'G' : state.walls.has(key(r, c)) ? '#' : '.';
          }
          rows.push(line);
        }
        return [
          'from solvor import astar_grid',
          '',
          'maze = """',
          ...rows,
          '"""',
          'grid = [[1 if ch == "#" else 0 for ch in row] for row in maze.split()]',
          `result = astar_grid(grid, (${state.start.join(', ')}), (${state.goal.join(', ')}))  # (row, column)`,
          `print(result.objective)${solved ? `   # ${res.path ? pyFloat(res.cost) + ' steps' : 'inf, no path'}` : ''}`,
          `print(result.iterations)${solved ? `  # ${res.order.length} cells expanded` : ''}`,
        ].join('\n');
      },
    };
  }

  // ---------------------------------------------------------------------------------------------
  // Example 3: TSP exactly the way solve_tsp does it: a nearest-neighbour tour, then tabu search
  // over 2-opt moves (cooldown 10, stop after 100 iterations without a new best).

  function tspExample() {
    const DEFAULT = [[32, 71], [107, 14], [264, 50], [334, 73], [229, 120], [308, 212], [203, 193], [111, 178], [57, 231], [157, 94]]; // tabu beats 2-opt here, and no city sits on an edge
    const COOLDOWN = 10, MAX_NO_IMPROVE = 100, MAX_ITER = 1000, SHOW_TABU = 6;
    const state = { cities: DEFAULT.map((p) => p.slice()) };
    let g = null;
    const tourLen = (tour, D) => { let s = 0; for (let i = 0; i < tour.length; i++) s += D[tour[i]][tour[(i + 1) % tour.length]]; return s; };
    const r0 = (v) => Math.round(v);

    // tabu_search from solvor/tabu.py, with the 2-opt neighbourhood solve_tsp gives it.
    function tabu(start, D) {
      const n = start.length;
      let sol = start, obj = tourLen(start, D);
      let best = sol, bestObj = obj, bestIter = 0;
      const tabuList = [], tabuSet = new Set(), steps = [];
      let iteration;
      for (iteration = 1; iteration <= MAX_ITER; iteration++) {
        let bm = null, bn = null, bo = Infinity;
        for (let i = 0; i < n - 1; i++) for (let j = i + 2; j < n; j++) {
          if (i === 0 && j === n - 1) continue;
          const nb = sol.slice(0, i + 1).concat(sol.slice(i + 1, j + 1).reverse(), sol.slice(j + 1));
          const no = tourLen(nb, D), m = `${i},${j}`;
          if (tabuSet.has(m) && no >= bestObj) continue; // tabu, unless it beats the best ever (aspiration)
          if (no < bo) { bo = no; bn = nb; bm = [i, j]; }
        }
        if (!bn) break;
        const [i, j] = bm, prev = sol, prevObj = obj;
        sol = bn; obj = bo;
        const m = `${i},${j}`;
        if (tabuList.length === COOLDOWN) tabuSet.delete(tabuList[0]);
        tabuList.push(m); if (tabuList.length > COOLDOWN) tabuList.shift();
        tabuSet.add(m);
        const newBest = obj < bestObj;
        if (newBest) { best = sol; bestObj = obj; bestIter = iteration; }
        steps.push({ iteration, tour: sol, len: obj, prevLen: prevObj, bestLen: bestObj, newBest,
          cut: [[prev[i], prev[i + 1]], [prev[j], prev[(j + 1) % n]]], fresh: [[prev[i], prev[j]], [prev[i + 1], prev[(j + 1) % n]]] });
        if (iteration - bestIter >= MAX_NO_IMPROVE) break;
      }
      return { steps, best, bestObj, bestIter, iterations: Math.min(iteration, MAX_ITER) };
    }

    return {
      tab: 'Tabu search',
      title: 'Tabu search: untangle the tour',
      hint: 'Drag the cities around, then press Solve. A greedy tour comes first, 2-opt moves undo its crossings, and then tabu search keeps going past the point where plain 2-opt gives up.',
      note: 'A [[2-opt]] move cuts two edges and reconnects the tour the other way round, which is exactly how a crossing disappears. Plain 2-opt stops at the first [[local optimum]]. Tabu search takes the best move that is not [[tabu]], even a worse one, and remembers its recent moves so it cannot walk straight back. solve_tsp stops after 100 moves without a new best.',
      aria: 'Cities and the current travelling salesman tour',
      legend: [
        ['line', 'sg-edge', 'tour'],
        ['line', 'sg-edge is-fresh', 'new edges this step'],
        ['line', 'sg-edge is-cut', 'edges just cut'],
        ['line', 'sg-edge is-done', 'best tour found'],
      ],
      restore() { state.cities = DEFAULT.map((p) => p.slice()); },

      mount({ svg, changed }) {
        g = {
          edges: svgEl('g', {}, svg),
          cities: state.cities.map((p, i) => {
            const grp = svgEl('g', { class: 'sg-city' }, svg);
            svgEl('circle', { r: 7 }, grp);
            svgEl('text', { y: 3.5 }, grp).textContent = String(i);
            return grp;
          }),
        };
        let drag = -1;
        svg.addEventListener('pointerdown', (evt) => {
          const p = svgPoint(svg, evt);
          let best = -1, bd = 18;
          state.cities.forEach(([x, y], i) => { const d = Math.hypot(x - p.x, y - p.y); if (d < bd) { bd = d; best = i; } });
          if (best < 0) return;
          drag = best;
          svg.setPointerCapture(evt.pointerId);
          evt.preventDefault();
        });
        svg.addEventListener('pointermove', (evt) => {
          if (drag < 0) return;
          const p = svgPoint(svg, evt);
          const np = [Math.round(Math.min(W - 10, Math.max(10, p.x))), Math.round(Math.min(H - 10, Math.max(10, p.y)))];
          const old = state.cities[drag];
          if (np[0] === old[0] && np[1] === old[1]) return;
          state.cities[drag] = np;
          changed();
        });
        const end = () => { drag = -1; };
        svg.addEventListener('pointerup', end);
        svg.addEventListener('pointercancel', end);
      },

      trace() {
        const P = state.cities, n = P.length;
        const D = P.map((a) => P.map((b) => Math.hypot(a[0] - b[0], a[1] - b[1])));
        const frames = [{ tour: [0], closed: false, msg: 'Start at city 0. [[Greedy|greedy]] rule: always go to the nearest city not visited yet.' }];
        const tour = [0], left = new Set(P.map((_, i) => i).slice(1));
        while (left.size) {
          let best = -1;
          left.forEach((c) => { if (best < 0 || D[tour[tour.length - 1]][c] < D[tour[tour.length - 1]][best]) best = c; });
          tour.push(best); left.delete(best);
          if (left.size) frames.push({ tour: tour.slice(), closed: false, msg: `Nearest unvisited city is ${best}.` });
        }
        const start = tourLen(tour, D);
        frames.push({ tour: tour.slice(), closed: true, len: start, bestLen: start,
          msg: `Close the loop: greedy tour of length ${r0(start)}. Fast, but usually tangled. Now tabu search takes over.` });

        const run = tabu(tour, D);
        let shown = 0, stuck = false, localOpt = Infinity;
        for (const st of run.steps) {
          const [[a, b], [c, d]] = st.cut;
          const edges = `cut ${a}–${b} and ${c}–${d}, reconnect ${a}–${c} and ${b}–${d}`;
          let msg;
          if (!stuck && st.len < st.prevLen) msg = `Tabu picks the best [[2-opt]] move: ${edges}. Length ${r0(st.prevLen)} → ${r0(st.len)}.`;
          else if (!stuck) {
            stuck = true; localOpt = st.prevLen;
            msg = `[[Local optimum|local optimum]]: no move makes it shorter, so plain 2-opt would stop here. Tabu takes the best allowed move anyway, ${r0(st.prevLen)} → ${r0(st.len)}, and marks it [[tabu]] for 10 iterations.`;
          } else if (st.newBest && st.len < localOpt - 1e-6) msg = `New best: ${r0(st.len)}. Taking a worse move first paid off.`;
          else msg = `Iteration ${st.iteration}: ${r0(st.prevLen)} → ${r0(st.len)}. Best so far stays ${r0(st.bestLen)}.`;
          if (stuck && ++shown > SHOW_TABU) break;
          frames.push({ ...st, closed: true, msg });
        }
        if (!stuck) localOpt = run.bestObj;
        const better = run.bestObj < localOpt - 1e-6;
        frames.push({ tour: run.best, closed: true, len: run.bestObj, bestLen: run.bestObj, iteration: run.iterations, done: true,
          msg: `Fast-forward: tabu ran ${run.iterations} iterations and stopped after ${MAX_NO_IMPROVE} without a new best. ` + (better
            ? `Best tour: ${r0(run.bestObj)}, shorter than the 2-opt local optimum of ${r0(localOpt)}. That is what the detours were for.`
            : `Nothing beat the 2-opt local optimum of ${r0(localOpt)} this time; drag a city or two and it sometimes does.`) });
        frames.run = run;
        return frames;
      },

      draw(frames, k) {
        const f = frames[k], P = state.cities;
        g.edges.innerHTML = '';
        const t = f.tour, m = f.closed ? t.length : t.length - 1;
        const has = (list, a, b) => (list || []).some(([u, v]) => (u === a && v === b) || (u === b && v === a));
        for (const [a, b] of f.done ? [] : f.cut || []) {
          svgEl('line', { class: 'sg-edge is-cut', 'data-a': a, 'data-b': b, 'data-cut': '1', x1: P[a][0], y1: P[a][1], x2: P[b][0], y2: P[b][1] }, g.edges);
        }
        for (let i = 0; i < m; i++) {
          const a = t[i], b = t[(i + 1) % t.length];
          svgEl('line', { class: f.done ? 'sg-edge is-done' : has(f.fresh, a, b) ? 'sg-edge is-fresh' : 'sg-edge',
            'data-a': a, 'data-b': b, x1: P[a][0], y1: P[a][1], x2: P[b][0], y2: P[b][1] }, g.edges);
        }
        g.cities.forEach((grp, i) => {
          grp.setAttribute('transform', `translate(${P[i][0]},${P[i][1]})`);
          grp.classList.toggle('is-here', !f.closed && i === t[t.length - 1]);
        });
      },

      stats(frames, k) {
        const f = frames[k];
        return [['[[tour length]]', f.closed ? fmt(f.len, 1) : '…'], ['best so far', f.closed ? fmt(f.bestLen, 1) : '…'],
          ['iteration', f.iteration ? `${f.iteration}` : '…']];
      },

      tipAt(target, pt, frames, k) {
        const P = state.cities, t = frames[k].tour;
        const grp = target.closest && target.closest('.sg-city');
        if (grp) {
          const i = g.cities.indexOf(grp), pos = t.indexOf(i);
          return `<strong>City ${i}</strong> at (${P[i][0]}, ${P[i][1]})<br>${pos < 0 ? 'not visited yet' : `stop ${pos + 1} of ${P.length} on the tour`}<br><span class="sg-tip-k">Drag to move it.</span>`;
        }
        if (target.dataset && target.dataset.a !== undefined) {
          const a = +target.dataset.a, b = +target.dataset.b;
          const len = fmt(Math.hypot(P[a][0] - P[b][0], P[a][1] - P[b][1]), 1);
          return `<strong>Edge ${a}–${b}</strong> length ${len}${target.dataset.cut ? '<br>Cut by this move.' : ''}`;
        }
        return null;
      },

      code(frames, solved) {
        const run = frames.run;
        const pts = state.cities.map(([x, y]) => `(${x}, ${y})`);
        return [
          'from math import dist',
          'from solvor import solve_tsp',
          '',
          `cities = [${pts.slice(0, 5).join(', ')},`,
          `          ${pts.slice(5).join(', ')}]`,
          'matrix = [[dist(a, b) for b in cities] for a in cities]',
          'result = solve_tsp(matrix, seed=0)',
          `print(result.solution)${solved ? '          # best tour found, starting at city 0' : ''}`,
          `print(round(result.objective))${solved ? `  # ${r0(run.bestObj)}` : ''}`,
        ].join('\n');
      },
    };
  }

  // ---------------------------------------------------------------------------------------------
  // Example 4: Nelder-Mead on Himmelblau's function, mirroring solvor.nelder_mead step for step.

  function nelderMeadExample() {
    const SC = 36, X0 = W / 2, Y0 = H / 2; // x in [-5, 5], y in [-3.6, 3.6]
    const px = ([x, y]) => [X0 + x * SC, Y0 - y * SC];
    const fn = (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2;
    const MINIMA = [[3, 2], [-2.805118, 3.131312], [-3.77931, -3.283186], [3.584428, -1.848126]];
    const STEP = 0.5, TOL = 1e-4;
    const DEFAULT_X0 = [1, -3];
    const state = { x0: DEFAULT_X0.slice() };
    const small = (v) => (v < 0.01 ? v.toPrecision(2) : fmt(v, 3));
    let g = null, contours = null;

    function contourPaths() {
      const levels = [0.5, 3, 10, 25, 50, 90, 150, 230, 330];
      const N = 4, cols = Math.ceil(W / N), rows = Math.ceil(H / N);
      const val = [];
      for (let j = 0; j <= rows; j++) { val.push([]); for (let i = 0; i <= cols; i++) val[j].push(fn((i * N - X0) / SC, (Y0 - j * N) / SC)); }
      return levels.map((lv) => {
        let d = '';
        for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
          const v = [val[j][i], val[j][i + 1], val[j + 1][i + 1], val[j + 1][i]];
          const pts = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
          const cross = [];
          for (let e = 0; e < 4; e++) {
            const a = v[e], b = v[(e + 1) % 4];
            if ((a < lv) !== (b < lv)) {
              const t = (lv - a) / (b - a), p = pts[e], q = pts[(e + 1) % 4];
              cross.push([(p[0] + t * (q[0] - p[0])) * N, (p[1] + t * (q[1] - p[1])) * N]);
            }
          }
          for (let s = 0; s + 1 < cross.length; s += 2) d += `M${cross[s][0].toFixed(1)},${cross[s][1].toFixed(1)}L${cross[s + 1][0].toFixed(1)},${cross[s + 1][1].toFixed(1)}`;
        }
        return d;
      });
    }

    function run() {
      const n = 2, alpha = 1, gamma = 2, rho = 0.5, sigma = 0.5;
      const x0 = state.x0;
      let simplex = [x0.slice()];
      for (let i = 0; i < n; i++) { const v = x0.slice(); v[i] += x0[i] === 0 ? STEP : STEP * Math.abs(x0[i]); simplex.push(v); }
      const f = (p) => fn(p[0], p[1]);
      let values = simplex.map(f);
      const snap = () => simplex.map((p) => p.slice());
      const b0 = values.indexOf(Math.min(...values));
      const frames = [{ simplex: snap(), best: simplex[b0].slice(), bestVal: values[b0], msg: 'Start with a triangle: the start point plus one corner nudged along x and one along y (initial_step = 0.5 means half of each coordinate). Evaluate f at all three. The triangle is the [[simplex|nm-simplex]].' }];
      let iteration = 0;
      for (iteration = 1; iteration <= 1000; iteration++) {
        const order = [0, 1, 2].sort((a, b) => values[a] - values[b]);
        simplex = order.map((i) => simplex[i]); values = order.map((i) => values[i]);
        const best = values[0], worst = values[n], second = values[n - 1];
        if (Math.abs(worst - best) < TOL) break;
        const centroid = [0, 0];
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) centroid[j] += simplex[i][j];
        for (let j = 0; j < n; j++) centroid[j] /= n;
        const w = simplex[n];
        const refl = centroid.map((cj, j) => cj + alpha * (cj - w[j]));
        const rv = f(refl);
        let op;
        const shrink = () => {
          for (let i = 1; i <= n; i++) { for (let j = 0; j < n; j++) simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]); values[i] = f(simplex[i]); }
          op = '[[shrink]]: nothing worked, so the whole triangle pulls in toward the best corner';
        };
        if (best <= rv && rv < second) { simplex[n] = refl; values[n] = rv; op = '[[reflect]]: flip the worst corner through the other two'; }
        else if (rv < best) {
          const exp = centroid.map((cj, j) => cj + gamma * (refl[j] - cj));
          const ev = f(exp);
          if (ev < rv) { simplex[n] = exp; values[n] = ev; op = '[[expand]]: the flip went so well, go twice as far'; }
          else { simplex[n] = refl; values[n] = rv; op = '[[reflect]]: flip the worst corner through the other two'; }
        } else if (rv < worst) {
          const con = centroid.map((cj, j) => cj + rho * (refl[j] - cj));
          const cv = f(con);
          if (cv <= rv) { simplex[n] = con; values[n] = cv; op = '[[contract]]: the flip landed somewhere better than the worst corner but not great, so stop halfway'; } else shrink();
        } else {
          const con = centroid.map((cj, j) => cj + rho * (w[j] - cj));
          const cv = f(con);
          if (cv < worst) { simplex[n] = con; values[n] = cv; op = '[[contract inside|contract-inside]]: the flip made things worse, so pull the worst corner toward the middle instead'; } else shrink();
        }
        const bi = values.indexOf(Math.min(...values));
        frames.push({ simplex: snap(), best: simplex[bi].slice(), bestVal: values[bi], msg: `Iteration ${iteration}, ${op}. Best f = ${small(values[bi])}.` });
      }
      const bi = values.indexOf(Math.min(...values));
      frames.push({ simplex: snap(), best: simplex[bi].slice(), bestVal: values[bi], iterations: iteration, done: true,
        msg: `The three corner values now differ by less than ${TOL} ([[tol]]), so it stops at (${fmt(simplex[bi][0])}, ${fmt(simplex[bi][1])}).` });
      return frames;
    }

    return {
      tab: 'Nelder–Mead',
      title: 'Nelder–Mead: feel your way downhill',
      hint: 'Click the map to pick a start point, then press Solve. [[Himmelblau’s function|himmelblau]] has four valleys; where you start decides where you land, and a lucky stretch can carry the triangle past the nearest one. Hover the map to read f.',
      note: 'No gradients needed, only function values. The worst corner of the triangle gets flipped through the others, stretched when that pays off, pulled in when it does not, and as a last resort the whole triangle shrinks toward its best corner. Yes, the triangle is also called a simplex: same word as the first tab, unrelated method.',
      legend: [
        ['line', 'sg-contour-key', '[[contour lines|contour]], lower inside'],
        ['plus', 'sg-min', 'minimum, f = 0'],
        ['rect', 'sg-simplex', 'the triangle'],
        ['line', 'sg-trail', 'best corner so far'],
      ],
      restore() { state.x0 = DEFAULT_X0.slice(); },
      aria: 'Contour map of Himmelblau’s function with the Nelder–Mead triangle',

      mount({ svg, changed }) {
        contours = contours || contourPaths();
        const layer = svgEl('g', { class: 'sg-contours' }, svg);
        contours.forEach((d, i) => svgEl('path', { d, 'stroke-opacity': (1 - i * 0.08).toFixed(2) }, layer));
        [-4, -2, 0, 2, 4].forEach((v) => { svgEl('text', { class: 'sg-tick', x: px([v, 0])[0], y: H - 4, 'text-anchor': 'middle' }, svg).textContent = v; });
        [-2, 0, 2].forEach((v) => { svgEl('text', { class: 'sg-tick', x: 4, y: px([0, v])[1] + 3.5 }, svg).textContent = v; });
        MINIMA.forEach((m) => { const [x, y] = px(m); svgEl('path', { class: 'sg-min', d: `M${x - 4},${y}h8M${x},${y - 4}v8` }, svg); });
        g = {
          trail: svgEl('polyline', { class: 'sg-trail' }, svg),
          tri: svgEl('polygon', { class: 'sg-simplex' }, svg),
          start: svgEl('circle', { class: 'sg-start', r: 4 }, svg),
          best: svgEl('circle', { class: 'sg-here', r: 6 }, svg),
        };
        let down = false;
        const place = (evt) => {
          const p = svgPoint(svg, evt);
          const x = Math.max(-4.5, Math.min(4.5, Math.round(((p.x - X0) / SC) * 2) / 2));
          const y = Math.max(-3.5, Math.min(3.5, Math.round(((Y0 - p.y) / SC) * 2) / 2));
          if (x === state.x0[0] && y === state.x0[1]) return;
          state.x0 = [x, y];
          changed({ restart: true });
        };
        svg.addEventListener('pointerdown', (evt) => { down = true; svg.setPointerCapture(evt.pointerId); evt.preventDefault(); place(evt); });
        svg.addEventListener('pointermove', (evt) => { if (down) place(evt); });
        const end = () => { down = false; };
        svg.addEventListener('pointerup', end);
        svg.addEventListener('pointercancel', end);
      },

      trace: run,

      draw(frames, k) {
        const f = frames[k];
        g.tri.setAttribute('points', f.simplex.map((p) => px(p).join(',')).join(' '));
        g.trail.setAttribute('points', frames.slice(0, k + 1).map((fr) => px(fr.best).join(',')).join(' '));
        const s = px(state.x0); g.start.setAttribute('cx', s[0]); g.start.setAttribute('cy', s[1]);
        const b = px(f.best); g.best.setAttribute('cx', b[0]); g.best.setAttribute('cy', b[1]);
        g.best.classList.toggle('is-done', !!f.done);
      },

      stats(frames, k) {
        const f = frames[k];
        return [['start', `(${fmt(state.x0[0])}, ${fmt(state.x0[1])})`], ['best', `(${fmt(f.best[0])}, ${fmt(f.best[1])})`], ['f(best)', small(f.bestVal)]];
      },

      tipAt(target, pt, frames, k) {
        const fv = (v) => (v < 1e-3 ? v.toExponential(1) : fmt(v, v < 10 ? 3 : 1));
        const corners = frames[k].simplex.map((p) => ({ p, v: fn(p[0], p[1]) })).sort((a, b) => a.v - b.v);
        const near = corners.findIndex(({ p }) => { const q = px(p); return Math.hypot(q[0] - pt.x, q[1] - pt.y) < 7; });
        if (near >= 0) {
          const { p, v } = corners[near];
          return `<strong>${['Best', 'Middle', 'Worst'][near]} corner</strong> (${fmt(p[0])}, ${fmt(p[1])})<br>f = ${fv(v)}${near === 2 ? '<br><span class="sg-tip-k">This one moves next.</span>' : ''}`;
        }
        const x = (pt.x - X0) / SC, y = (Y0 - pt.y) / SC;
        const m = MINIMA.find(([a, b]) => Math.hypot(a - x, b - y) < 0.15);
        if (m) return `<strong>A minimum</strong> (${fmt(m[0])}, ${fmt(m[1])}), f = 0<br><span class="sg-tip-k">Himmelblau has four of these.</span>`;
        const sx = Math.max(-4.5, Math.min(4.5, Math.round(x * 2) / 2)), sy = Math.max(-3.5, Math.min(3.5, Math.round(y * 2) / 2));
        return `<strong>f(${fmt(x, 1)}, ${fmt(y, 1)})</strong> = ${fv(fn(x, y))}<br><span class="sg-tip-k">Click to start at (${fmt(sx)}, ${fmt(sy)}): starts snap to halves.</span>`;
      },

      code(frames, solved) {
        const end = frames[frames.length - 1];
        return [
          'from solvor import nelder_mead',
          '',
          'def himmelblau(p):',
          '    x, y = p',
          '    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2',
          '',
          `result = nelder_mead(himmelblau, [${state.x0.map(pyFloat).join(', ')}], initial_step=0.5, tol=1e-4)`,
          `print(result.solution)${solved ? `    # about [${fmt(end.best[0])}, ${fmt(end.best[1])}]` : ''}`,
          `print(result.iterations)${solved ? `  # ${end.iterations}` : ''}`,
        ].join('\n');
      },
    };
  }

  const EXAMPLES = [lpExample, astarExample, tspExample, nelderMeadExample];

  // ---------------------------------------------------------------------------------------------
  // The gallery shell: tabs, stage, step controls and the code block.

  let uid = 0;
  function mountGallery(root) {
    if (root.dataset.ready) return;
    root.dataset.ready = '1';
    const id = `sg${++uid}`;
    const instances = [];
    root.innerHTML = `
      <div class="sg-tabs" role="tablist" aria-label="Examples">
        ${EXAMPLES.map((_, i) => `<button type="button" role="tab" id="${id}-tab${i}" aria-controls="${id}-panel"></button>`).join('')}
      </div>
      <div class="sg-body" role="tabpanel" id="${id}-panel">
        <div class="sg-stage">
          <div class="sg-figure">
            <svg class="sg-svg" viewBox="0 0 ${W} ${H}" role="img"></svg>
            <ul class="sg-legend"></ul>
            <p class="sg-restore-row"><button type="button" class="sg-restore">Restore the example</button></p>
          </div>
          <div class="sg-panel">
            <p class="sg-title"></p>
            <p class="sg-hint"></p>
            <div class="sg-controls"></div>
            <table class="sg-stats"><tbody></tbody></table>
            <div class="sg-actions">
              <button type="button" class="sg-btn sg-btn--primary" data-act="solve" title="Play every step, two per second">Solve</button>
              <button type="button" class="sg-btn" data-act="step" title="One step forward">Step</button>
              <button type="button" class="sg-btn" data-act="reset" title="Back to step 0. Your edits stay.">Reset</button>
              <span class="sg-count"></span>
            </div>
            <p class="sg-log" aria-live="polite"></p>
          </div>
        </div>
        <p class="sg-note"></p>
        <div class="sg-code"></div>
      </div>
      <div class="sg-tip" role="tooltip" hidden></div>`;

    const $ = (sel) => root.querySelector(sel);
    const tabs = [...root.querySelectorAll('[role="tab"]')];
    const panel = $('[role="tabpanel"]');
    let svg = $('.sg-svg');
    const controls = $('.sg-controls');
    const btnSolve = $('[data-act="solve"]'), btnStep = $('[data-act="step"]');
    let cur = null, frames = [], k = 0, timer = null, active = -1;

    // One floating tooltip per gallery: glossary terms in the text, and data under the pointer in the picture.
    const tip = $('.sg-tip');
    let tipFor = null;
    function showTip(html, x, y, below) {
      tip.innerHTML = html;
      tip.hidden = false;
      const box = root.getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight;
      let left = x - box.left + (below ? -tw / 2 : 14), top = y - box.top + (below ? 8 : 14);
      left = Math.max(4, Math.min(box.width - tw - 4, left));
      if (!below && top + th > box.height - 4) top = y - box.top - th - 10;
      tip.style.left = `${left}px`; tip.style.top = `${top}px`;
    }
    const hideTip = () => { tip.hidden = true; tipFor = null; };
    const termTip = (el) => {
      const text = GLOSSARY[el.dataset.term];
      if (!text) return;
      const r = el.getBoundingClientRect();
      tipFor = el;
      showTip(`<strong>${el.textContent}</strong> ${text}`, r.left + r.width / 2, r.bottom, true);
    };
    root.addEventListener('pointerover', (e) => { const t = e.target.closest && e.target.closest('.sg-term'); if (t && e.pointerType !== 'touch') termTip(t); });
    root.addEventListener('pointerout', (e) => { if (e.pointerType !== 'touch' && tipFor && e.target.closest && e.target.closest('.sg-term') === tipFor) hideTip(); });
    root.addEventListener('focusin', (e) => { if (e.target.classList.contains('sg-term')) termTip(e.target); });
    root.addEventListener('focusout', (e) => { if (e.target === tipFor) hideTip(); });
    root.addEventListener('click', (e) => {
      const t = e.target.closest && e.target.closest('.sg-term');
      if (t) termTip(t); else if (tipFor && tipFor !== svg) hideTip();
    });
    root.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideTip(); });

    function watchSvg() {
      svg.addEventListener('pointermove', (e) => {
        if (e.pointerType === 'touch' || e.buttons) { if (!tipFor || tipFor === svg) hideTip(); return; }
        const marked = e.target.closest && e.target.closest('[data-tip]');
        const html = marked ? marked.getAttribute('data-tip') : cur.tipAt ? cur.tipAt(e.target, svgPoint(svg, e), frames, k) : null;
        if (html) { tipFor = svg; showTip(html, e.clientX, e.clientY, false); } else if (tipFor === svg) hideTip();
      });
      svg.addEventListener('pointerleave', () => { if (tipFor === svg) hideTip(); });
      svg.addEventListener('pointerdown', () => { if (tipFor === svg) hideTip(); });
    }

    const last = () => frames.length - 1;
    const stop = () => { if (timer) clearInterval(timer); timer = null; btnSolve.textContent = 'Solve'; };

    function paint() {
      cur.draw(frames, k);
      $('.sg-stats tbody').innerHTML = cur.stats(frames, k).map(([a, b]) => `<tr><th>${rich(a)}</th><td>${b}</td></tr>`).join('');
      $('.sg-log').innerHTML = rich(frames[k].msg);
      if (tipFor && !root.contains(tipFor)) hideTip();
      $('.sg-count').textContent = `step ${k} / ${last()}`;
      btnStep.disabled = k >= last();
      $('.sg-code').innerHTML = highlight(cur.code(frames, k >= last()));
    }

    function texts() {
      $('.sg-title').textContent = cur.title;
      $('.sg-hint').innerHTML = rich(cur.hint);
      $('.sg-note').innerHTML = rich(cur.note);
      const shape = { rect: (c) => `<rect class="${c}" x="1" y="1" width="12" height="12"/>`,
        line: (c) => `<line class="${c}" x1="0" y1="7" x2="14" y2="7"/>`,
        dot: (c) => `<circle class="${c}" cx="7" cy="7" r="4"/>`,
        plus: (c) => `<path class="${c}" d="M3,7h8M7,3v8"/>` };
      $('.sg-legend').innerHTML = (cur.legend || []).map(([kind, cls, label]) =>
        `<li><svg class="sg-key" viewBox="0 0 14 14" aria-hidden="true">${shape[kind](cls)}</svg><span>${rich(label)}</span></li>`).join('');
      $('.sg-restore-row').hidden = !cur.restore;
      svg.setAttribute('aria-label', cur.aria);
    }

    function recompute() {
      const wasDone = frames.length > 1 && k >= last();
      frames = cur.trace();
      k = wasDone ? last() : 0;
      paint();
    }

    // Edits keep a solved picture solved (so dragging shows the new answer live); a restart goes back to step 0.
    function changed(opts) {
      stop();
      if (opts && opts.text) texts();
      if (opts && opts.restart) frames = [];
      recompute();
    }

    function mountCurrent() {
      hideTip();
      const fresh = svg.cloneNode(false);
      svg.replaceWith(fresh);
      svg = fresh;
      controls.innerHTML = '';
      cur.mount({ svg, controls, changed });
      watchSvg();
      texts();
    }

    function select(i, focus) {
      if (i === active) return;
      stop();
      active = i;
      cur = instances[i];
      tabs.forEach((t, j) => { t.setAttribute('aria-selected', String(j === i)); t.tabIndex = j === i ? 0 : -1; });
      panel.setAttribute('aria-labelledby', tabs[i].id);
      // Fresh svg per example, so the previous example's pointer listeners go with the old one.
      mountCurrent();
      frames = [];
      recompute();
      if (focus) tabs[i].focus();
    }

    EXAMPLES.forEach((make) => instances.push(make()));
    tabs.forEach((t, i) => {
      t.textContent = instances[i].tab;
      t.addEventListener('click', () => select(i));
      t.addEventListener('keydown', (e) => {
        const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
        if (d) { e.preventDefault(); select((active + d + tabs.length) % tabs.length, true); }
        else if (e.key === 'Home') { e.preventDefault(); select(0, true); }
        else if (e.key === 'End') { e.preventDefault(); select(tabs.length - 1, true); }
      });
    });

    btnSolve.addEventListener('click', () => {
      if (timer) { stop(); return; }
      if (k >= last()) k = 0;
      if (reducedMotion()) { k = last(); paint(); return; }
      paint();
      btnSolve.textContent = 'Pause';
      timer = setInterval(() => {
        if (!root.isConnected) { stop(); return; }
        k = Math.min(k + 1, last());
        paint();
        if (k >= last()) stop();
      }, STEP_MS);
    });
    btnStep.addEventListener('click', () => { stop(); if (k < last()) { k++; paint(); } });
    $('[data-act="reset"]').addEventListener('click', () => { stop(); k = 0; paint(); });
    root.addEventListener('click', (e) => {
      if (!e.target.closest || !e.target.closest('.sg-restore')) return;
      stop(); cur.restore(); mountCurrent(); frames = []; recompute();
    });

    select(0);
  }

  const init = () => document.querySelectorAll('.solvor-gallery').forEach(mountGallery);
  if (window.document$ && typeof window.document$.subscribe === 'function') window.document$.subscribe(init);
  else if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
