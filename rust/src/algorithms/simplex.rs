//! Bounded-variable dense-tableau simplex: the arithmetic kernel of the LP engine.
//!
//! Operation for operation the same as `BoundedSimplex` in solvor/lp_engine.py: the
//! same loops in the same order and the same float expressions (no fused multiply-add,
//! explicit comparisons where Python uses `max()`), so both backends produce identical
//! tableaux, iteration counts and solutions, bit for bit. The warm-start policy,
//! presolve and branch and bound stay in Python; this module only does the arithmetic.
//!
//! Every nonbasic column sits at one of its bounds. A column is written as t = y - lo
//! (at its lower bound) or t = hi - y (at its upper bound, "flipped"), so every
//! nonbasic t is 0 and the right-hand side holds the basic t values. Bland's rule
//! (smallest index on ties) keeps both simplex variants deterministic and cycle-free.

use crate::types::Status;

const INF: f64 = f64::INFINITY;
const NONE: usize = usize::MAX;

/// Dense tableau for: rows . y + s = rhs, lo <= y <= hi, slack s >= 0 (s = 0 on '=' rows).
///
/// Columns are [structural | slack | artificial | slacks of added rows], the
/// right-hand side sits at index `width`, and the objective row is `t[m]`.
pub struct BoundedSimplex {
    pub(crate) m: usize,
    pub(crate) width: usize,
    pub(crate) eps: f64,
    pub(crate) art_start: usize,
    pub(crate) art_end: usize,
    /// Pivots since construction (the caller may reset it).
    pub(crate) pivots: u64,
    pub(crate) art_tol: Vec<f64>,
    /// False until phase 1 has found a feasible basis.
    pub(crate) phase1_done: bool,
    pub(crate) lo: Vec<f64>,
    pub(crate) hi: Vec<f64>,
    pub(crate) span: Vec<f64>,
    pub(crate) flipped: Vec<bool>,
    pub(crate) is_basic: Vec<bool>,
    pub(crate) basis: Vec<usize>,
    pub(crate) row_of: Vec<usize>,
    pub(crate) t: Vec<Vec<f64>>,
}

impl BoundedSimplex {
    /// Tableau for `rows[i] . y (<= or =) rhs[i]` with `0 <= y <= upper`.
    ///
    /// Rows hold (column, coefficient) pairs with columns below `upper.len()`;
    /// `rhs` and `is_eq` have one entry per row.
    pub fn new(
        rows: &[Vec<(usize, f64)>],
        rhs: &[f64],
        is_eq: &[bool],
        upper: &[f64],
        eps: f64,
    ) -> Self {
        let (n, m) = (upper.len(), rows.len());
        let needs_art: Vec<bool> = rhs
            .iter()
            .zip(is_eq)
            .map(|(&r, &eq)| r < -eps || (eq && r > eps))
            .collect();
        let width = n + m + needs_art.iter().filter(|&&a| a).count();
        // Phase 1 is feasible when every artificial is zero up to a tolerance taken from its own row
        let art_tol = rhs
            .iter()
            .zip(&needs_art)
            .filter(|&(_, &needs)| needs)
            .map(|(&r, _)| 1e-9 * (1.0 + r.abs()))
            .collect();

        let mut hi = upper.to_vec();
        hi.extend(is_eq.iter().map(|&eq| if eq { 0.0 } else { INF }));
        hi.resize(width, INF);
        let span = hi.clone();

        let mut lp = BoundedSimplex {
            m,
            width,
            eps,
            art_start: n + m,
            art_end: width,
            pivots: 0,
            art_tol,
            phase1_done: width == n + m,
            lo: vec![0.0; width],
            hi,
            span,
            flipped: vec![false; width],
            is_basic: vec![false; width],
            basis: vec![0; m],
            row_of: vec![NONE; width],
            t: Vec::with_capacity(m + 1),
        };

        let mut art = lp.art_start;
        for i in 0..m {
            let mut row = vec![0.0; width + 1];
            for &(j, a) in &rows[i] {
                row[j] = a;
            }
            row[n + i] = 1.0;
            row[width] = rhs[i];
            let mut basic = n + i;
            if needs_art[i] {
                if rhs[i] < 0.0 {
                    for &(j, _) in &rows[i] {
                        row[j] = -row[j];
                    }
                    row[n + i] = -1.0;
                    row[width] = -rhs[i];
                }
                row[art] = 1.0;
                basic = art;
                art += 1;
            }
            lp.set_basic(i, basic);
            lp.t.push(row);
        }
        lp.t.push(vec![0.0; width + 1]);
        lp
    }

    // Cold solve

    /// Phase 1 (only if artificials exist), then phase 2 on `cost` (one entry per structural column).
    pub fn solve(&mut self, cost: &[f64], max_iter: i64) -> (Status, i64) {
        let mut iters = 0;
        if self.art_end > self.art_start {
            let mut phase1 = vec![0.0; self.width];
            phase1[self.art_start..self.art_end].fill(1.0);
            self.set_objective(&phase1);
            let (status, it) = self.primal(max_iter);
            iters = it;
            if status == Status::MaxIter {
                return (status, iters);
            }
            let w = self.width;
            for i in 0..self.m {
                let b = self.basis[i];
                if self.art_start <= b
                    && b < self.art_end
                    && self.t[i][w] > self.art_tol[b - self.art_start]
                {
                    return (Status::Infeasible, iters);
                }
            }
            self.drive_out_artificials();
            self.phase1_done = true;
        }

        self.set_objective(cost);
        let (status, more) = self.primal(max_iter.saturating_sub(iters));
        (status, iters + more)
    }

    /// Value of every column in y (not in the flipped representation).
    pub fn column_values(&self) -> Vec<f64> {
        let w = self.width;
        let mut tv = vec![0.0; w];
        for i in 0..self.m {
            tv[self.basis[i]] = self.t[i][w];
        }
        (0..w)
            .map(|j| {
                if self.flipped[j] {
                    self.hi[j] - tv[j]
                } else {
                    self.lo[j] + tv[j]
                }
            })
            .collect()
    }

    /// Objective row = reduced costs of `cost` for the current basis (missing entries are 0).
    pub fn set_objective(&mut self, cost: &[f64]) {
        let (m, w) = (self.m, self.width);
        let mut obj = vec![0.0; w + 1];
        for (j, &cj) in cost.iter().enumerate().take(w) {
            if cj != 0.0 {
                obj[j] = if self.flipped[j] { -cj } else { cj };
            }
        }
        for i in 0..m {
            let cb = obj[self.basis[i]];
            if cb != 0.0 {
                for (o, &a) in obj.iter_mut().zip(&self.t[i]) {
                    if a != 0.0 {
                        *o -= cb * a;
                    }
                }
            }
        }
        self.t[m] = obj;
    }

    /// Primal simplex from a primal feasible basis.
    pub fn primal(&mut self, max_iter: i64) -> (Status, i64) {
        let (m, w, eps) = (self.m, self.width, self.eps);
        let limit = max_iter.max(0);
        for iteration in 0..limit {
            // Bland's rule for entering: smallest index with negative reduced cost (fixed columns never enter)
            let obj = &self.t[m];
            let Some(enter) =
                (0..w).find(|&j| obj[j] < -eps && !self.is_basic[j] && self.span[j] > eps)
            else {
                return (Status::Optimal, iteration);
            };

            // Bounded ratio test: a basic column hits 0 or its span, or the entering column
            // reaches its own other bound first (bound flip). Ties go to the smallest basis index.
            let (mut best, mut leave, mut to_upper) = (self.span[enter], NONE, false);
            for i in 0..m {
                let a = self.t[i][enter];
                let (mut ratio, up) = if a > eps {
                    (self.t[i][w] / a, false)
                } else {
                    let span_basic = self.span[self.basis[i]];
                    if a < -eps && span_basic < INF {
                        ((span_basic - self.t[i][w]) / -a, true)
                    } else {
                        continue;
                    }
                };
                if ratio < 0.0 {
                    ratio = 0.0;
                }
                if ratio < best - eps
                    || (leave != NONE
                        && (ratio - best).abs() <= eps
                        && self.basis[i] < self.basis[leave])
                {
                    (best, leave, to_upper) = (ratio, i, up);
                }
            }

            if leave == NONE {
                if best == INF {
                    return (Status::Unbounded, iteration);
                }
                self.complement(enter); // bound flip, no basis change
                continue;
            }

            let left = self.basis[leave];
            self.pivot(leave, enter);
            if to_upper {
                self.complement(left);
            }
        }
        (Status::MaxIter, limit)
    }

    // Warm operations

    /// Move boxed nonbasic columns to the bound their reduced cost prefers.
    ///
    /// Returns false if a column without a finite span has the wrong sign
    /// (dual simplex cannot start; the caller rebuilds instead).
    pub fn make_dual_feasible(&mut self) -> bool {
        let m = self.m;
        for j in 0..self.width {
            if self.is_basic[j] || self.span[j] <= self.eps || self.t[m][j] >= -self.eps {
                continue;
            }
            if self.span[j] == INF {
                return false;
            }
            self.complement(j);
        }
        true
    }

    /// Dual simplex: restore primal feasibility while keeping the objective row dual feasible.
    pub fn dual(&mut self, max_iter: i64) -> (Status, i64) {
        let (eps, w) = (self.eps, self.width);
        let limit = max_iter.max(0);
        for iteration in 0..limit {
            let m = self.m;
            // Leaving row: the most violated basic column (ties: smallest row)
            let (mut leave, mut worst, mut above) = (NONE, eps, false);
            for i in 0..m {
                let v = self.t[i][w];
                if -v > worst {
                    (leave, worst, above) = (i, -v, false);
                } else {
                    let s = self.span[self.basis[i]];
                    if s < INF && v - s > worst {
                        (leave, worst, above) = (i, v - s, true);
                    }
                }
            }
            if leave == NONE {
                return (Status::Optimal, iteration);
            }
            if above {
                self.flip_basic(leave); // now below its (other) bound: t' = span - t < 0
            }

            // Entering column: smallest ratio obj[j] / -T[r][j] over columns that raise the row
            let (row, obj) = (&self.t[leave], &self.t[m]);
            let (mut enter, mut best) = (NONE, INF);
            for j in 0..w {
                let a = row[j];
                if a < -eps && !self.is_basic[j] && self.span[j] > eps {
                    // Python's max(obj[j], 0.0): 0.0 only if it is strictly greater
                    let o = if 0.0 > obj[j] { 0.0 } else { obj[j] };
                    let ratio = o / -a;
                    if ratio < best - eps {
                        (enter, best) = (j, ratio);
                    }
                }
            }
            if enter == NONE {
                return (Status::Infeasible, iteration);
            }
            self.pivot(leave, enter);
        }
        (Status::MaxIter, limit)
    }

    /// Change column j's bounds in place (lo must be finite). The basis may become primal infeasible.
    pub fn set_bounds(&mut self, j: usize, lo: f64, hi: f64) {
        let (old_lo, old_hi, w) = (self.lo[j], self.hi[j], self.width);
        if self.is_basic[j] {
            let r = self.row_of[j];
            if !self.flipped[j] {
                // t = y - lo
                self.t[r][w] -= lo - old_lo;
            } else if hi < INF {
                // t = hi - y
                self.t[r][w] += hi - old_hi;
            } else {
                // t = old_hi - y can no longer be used: switch to t' = y - lo = (old_hi - lo) - t
                self.negate_row_except(r, j);
                self.t[r][w] = (old_hi - lo) - self.t[r][w];
                self.flipped[j] = false;
            }
        } else if !self.flipped[j] {
            // sits at lo
            if lo != old_lo {
                self.shift(j, lo - old_lo);
            }
        } else if hi < INF {
            // sits at hi, t = hi - y
            if hi != old_hi {
                self.shift(j, old_hi - hi);
            }
        } else {
            // sits at hi, which disappears: move to lo
            self.shift(j, old_hi - lo);
            self.negate_column(j);
            self.flipped[j] = false;
        }
        (self.lo[j], self.hi[j], self.span[j]) = (lo, hi, hi - lo);
    }

    /// Append `coefs . y <= rhs` (or `=`) with a new basic slack, expressed in the current basis.
    pub fn add_row(&mut self, coefs: &[(usize, f64)], rhs: f64, is_eq: bool) {
        let w = self.width;
        for row in &mut self.t {
            row.insert(w, 0.0); // new slack column, just before the rhs
        }
        let mut new = vec![0.0; w + 2];
        let mut r = rhs;
        for &(j, a) in coefs {
            if self.flipped[j] {
                new[j] -= a;
                r -= a * self.hi[j];
            } else {
                new[j] += a;
                r -= a * self.lo[j];
            }
        }
        new[w + 1] = r;
        for i in 0..self.m {
            // eliminate basic columns
            let e = new[self.basis[i]];
            if e != 0.0 {
                for (k, &s) in self.t[i].iter().enumerate() {
                    if s != 0.0 {
                        new[k] -= e * s;
                    }
                }
            }
        }
        new[w] = 1.0;
        self.t.insert(self.m, new);

        let bound = if is_eq { 0.0 } else { INF };
        self.lo.push(0.0);
        self.hi.push(bound);
        self.span.push(bound);
        self.flipped.push(false);
        self.is_basic.push(false);
        self.row_of.push(NONE);
        self.basis.push(0);
        self.width = w + 1;
        self.m += 1;
        self.set_basic(self.m - 1, w);
    }

    // Tableau primitives

    fn set_basic(&mut self, r: usize, q: usize) {
        self.basis[r] = q;
        self.is_basic[q] = true;
        self.row_of[q] = r;
    }

    fn pivot(&mut self, r: usize, q: usize) {
        let prow = &mut self.t[r];
        let inv = 1.0 / prow[q];
        // Only the pivot row's nonzeros change other rows
        let nz: Vec<usize> = (0..prow.len()).filter(|&j| prow[j] != 0.0).collect();
        for &j in &nz {
            prow[j] *= inv;
        }
        let values: Vec<f64> = nz.iter().map(|&j| prow[j]).collect();
        for (i, row) in self.t.iter_mut().enumerate() {
            if i == r {
                continue;
            }
            let f = row[q];
            if f != 0.0 {
                for (&j, &p) in nz.iter().zip(&values) {
                    row[j] -= f * p;
                }
                row[q] = 0.0;
            }
        }
        let old = self.basis[r];
        self.is_basic[old] = false;
        self.row_of[old] = NONE;
        self.set_basic(r, q);
        self.pivots += 1;
    }

    /// Move nonbasic column j to its other bound (t' = span - t), which sits at 0.
    fn complement(&mut self, j: usize) {
        let (u, w) = (self.span[j], self.width);
        for row in &mut self.t {
            let a = row[j];
            if a != 0.0 {
                row[w] -= a * u;
                row[j] = -a;
            }
        }
        self.flipped[j] = !self.flipped[j];
    }

    /// Rewrite row r for its basic column measured from the other bound: t' = span - t.
    fn flip_basic(&mut self, r: usize) {
        let (b, w) = (self.basis[r], self.width);
        self.negate_row_except(r, b);
        self.t[r][w] = self.span[b] - self.t[r][w];
        self.flipped[b] = !self.flipped[b];
    }

    fn negate_row_except(&mut self, r: usize, keep: usize) {
        let w = self.width;
        for (k, v) in self.t[r][..w].iter_mut().enumerate() {
            if k != keep && *v != 0.0 {
                *v = -*v;
            }
        }
    }

    /// Nonbasic column j's t changes by dt: update every row's right-hand side.
    fn shift(&mut self, j: usize, dt: f64) {
        let w = self.width;
        for row in &mut self.t {
            let a = row[j];
            if a != 0.0 {
                row[w] -= a * dt;
            }
        }
    }

    fn negate_column(&mut self, j: usize) {
        for row in &mut self.t {
            if row[j] != 0.0 {
                row[j] = -row[j];
            }
        }
    }

    /// Pivot basic artificials (all at 0 now) out where possible, then pin every artificial to 0.
    fn drive_out_artificials(&mut self) {
        for i in 0..self.m {
            let b = self.basis[i];
            if !(self.art_start <= b && b < self.art_end) {
                continue;
            }
            let found =
                (0..self.art_start).find(|&j| !self.is_basic[j] && self.t[i][j].abs() > self.eps);
            if let Some(j) = found {
                self.pivot(i, j);
            }
        }
        for a in self.art_start..self.art_end {
            self.hi[a] = 0.0;
            self.span[a] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(entries: &[(usize, f64)]) -> Vec<(usize, f64)> {
        entries.to_vec()
    }

    #[test]
    fn test_maximize_two_variables() {
        // max 3a + 2b (min -3a - 2b) s.t. a + b <= 4, a + 3b <= 6, a <= 3
        let rows = vec![row(&[(0, 1.0), (1, 1.0)]), row(&[(0, 1.0), (1, 3.0)])];
        let mut lp = BoundedSimplex::new(&rows, &[4.0, 6.0], &[false, false], &[3.0, INF], 1e-9);
        let (status, _) = lp.solve(&[-3.0, -2.0], 100);
        assert_eq!(status, Status::Optimal);
        let y = lp.column_values();
        assert_eq!((y[0], y[1]), (3.0, 1.0));
    }

    #[test]
    fn test_phase_one_detects_infeasible() {
        // a + b = 5 with a, b <= 2
        let rows = vec![row(&[(0, 1.0), (1, 1.0)])];
        let mut lp = BoundedSimplex::new(&rows, &[5.0], &[true], &[2.0, 2.0], 1e-9);
        assert_eq!(lp.solve(&[1.0, 1.0], 100).0, Status::Infeasible);
        assert!(!lp.phase1_done);
    }

    #[test]
    fn test_unbounded() {
        // min -a with a - b <= 1, both unbounded above
        let rows = vec![row(&[(0, 1.0), (1, -1.0)])];
        let mut lp = BoundedSimplex::new(&rows, &[1.0], &[false], &[INF, INF], 1e-9);
        assert_eq!(lp.solve(&[-1.0, 0.0], 100).0, Status::Unbounded);
    }

    #[test]
    fn test_warm_bound_change_and_added_row() {
        // max a + b s.t. a + 2b <= 4, a, b in [0, 3]: optimum a = 3, b = 0.5
        let rows = vec![row(&[(0, 1.0), (1, 2.0)])];
        let mut lp = BoundedSimplex::new(&rows, &[4.0], &[false], &[3.0, 3.0], 1e-9);
        assert_eq!(lp.solve(&[-1.0, -1.0], 100).0, Status::Optimal);
        assert_eq!(lp.column_values()[..2], [3.0, 0.5]);

        // Branch b <= 0: dual simplex restores feasibility
        lp.set_bounds(1, 0.0, 0.0);
        assert!(lp.make_dual_feasible());
        assert_eq!(lp.dual(100).0, Status::Optimal);
        assert_eq!(lp.primal(100).0, Status::Optimal);
        assert_eq!(lp.column_values()[..2], [3.0, 0.0]);

        // Cut a <= 2 as a new row
        lp.add_row(&[(0, 1.0)], 2.0, false);
        assert_eq!(lp.dual(100).0, Status::Optimal);
        assert_eq!(lp.column_values()[..2], [2.0, 0.0]);
        assert!(lp.pivots > 0);
    }

    #[test]
    fn test_iteration_limit() {
        let rows = vec![row(&[(0, 1.0), (1, 1.0)]), row(&[(0, 1.0), (1, 3.0)])];
        let mut lp = BoundedSimplex::new(&rows, &[4.0, 6.0], &[false, false], &[3.0, INF], 1e-9);
        assert_eq!(lp.solve(&[-3.0, -2.0], 0), (Status::MaxIter, 0));
    }
}
