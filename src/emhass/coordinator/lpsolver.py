"""A small numpy LP solver for the Dantzig-Wolfe master.

`linprog`: primal-dual interior point (Mehrotra predictor-corrector) for
min c.x s.t. A x = b, lb <= x <= ub (ub may be +inf). It returns the equality
duals with HiGHS' sign convention (`eqlin.marginals`, d objective / d b):
they are the meter prices, so their accuracy matters as much as x.

The master has a structure a general solver does not use: blocks of rows
(each battery's) that touch each other only through a border (the meter and
convexity rows). The normal-equations matrix A D A^T is assembled from
coordinate triplets directly and factored by that structure. Its prices are
the reference the coordinator's results and tests were built on; the
recovery MILP (one plan per on/off participant) is scipy's HiGHS.
"""

# The predictor-corrector closures are defined and used within one iteration.
# ruff: noqa: B023

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class Triplets:
    """A write-only sparse matrix: `A[rows, cols] = values`, like scipy's lil.

    Accepts ints, slices, index arrays and broadcastable values - the
    assignment patterns the master builder uses - and records entries.
    """

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self._r: list[np.ndarray] = []
        self._c: list[np.ndarray] = []
        self._v: list[np.ndarray] = []

    def _idx(self, k: int | slice | npt.ArrayLike, size: int) -> np.ndarray:
        if isinstance(k, slice):
            return np.arange(size)[k]
        return np.atleast_1d(np.asarray(k, dtype=int))

    def __setitem__(
        self,
        key: tuple[int | slice | npt.ArrayLike, int | slice | npt.ArrayLike],
        value: npt.ArrayLike,
    ) -> None:
        r, c = key
        r = self._idx(r, self.shape[0])
        c = self._idx(c, self.shape[1])
        v = np.asarray(value, dtype=float)
        if r.size > 1 and c.size == 1:
            c = np.full(r.size, c[0])
        elif c.size > 1 and r.size == 1:
            r = np.full(c.size, r[0])
        v = np.broadcast_to(v.reshape(-1) if v.ndim > 1 else v, r.shape).astype(float)
        keep = v != 0.0
        self._r.append(r[keep])
        self._c.append(c[keep])
        self._v.append(v[keep])

    def coo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._r:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        return np.concatenate(self._r), np.concatenate(self._c), np.concatenate(self._v)


@dataclass
class LPResult:
    x: np.ndarray
    fun: float
    y: np.ndarray  # equality duals, d(fun)/d(b)
    status: int  # 0 optimal, 1 iteration limit, 2 numerical trouble
    iterations: int
    message: str


class _NormalEquations:
    """Assembles M = A diag(d) A^T from triplets, fast for our column mix.

    With `border` rows given (the meter and convexity rows of the master),
    the remaining rows split into independent blocks - one per battery - that
    touch each other only through the border. M is then
    block-arrow shaped and is solved by eliminating each block separately and
    factoring the small border Schur complement: about 100x cheaper than a
    dense factorisation for a 6-battery, 48 h master.
    """

    def __init__(
        self, m: int, n: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, border: int = 0
    ) -> None:
        self.m, self.n = m, n
        nnz = np.bincount(cols, minlength=n)
        dense_cols = np.where(nnz > 3)[0]  # the plan columns
        self.dense_cols = dense_cols
        is_dense = np.zeros(n, bool)
        is_dense[dense_cols] = True
        # sparse part: for every column, all (row_a, row_b) pairs of its nonzeros
        sel = ~is_dense[cols]
        r, c, v = rows[sel], cols[sel], vals[sel]
        order = np.argsort(c, kind="stable")
        r, c, v = r[order], c[order], v[order]
        pa, pb, pv, pc = [], [], [], []
        starts = np.searchsorted(c, np.arange(n))
        ends = np.searchsorted(c, np.arange(n), side="right")
        cnt = ends - starts
        for k in (1, 2, 3):
            js = np.where(cnt == k)[0]
            if js.size == 0:
                continue
            idx = starts[js][:, None] + np.arange(k)[None, :]
            R, V = r[idx], v[idx]
            for a in range(k):
                for b in range(k):
                    pa.append(R[:, a])
                    pb.append(R[:, b])
                    pv.append(V[:, a] * V[:, b])
                    pc.append(js)
        self.flat = (np.concatenate(pa) * m + np.concatenate(pb)) if pa else np.zeros(0, int)
        self.pv = np.concatenate(pv) if pv else np.zeros(0)
        self.pc = np.concatenate(pc) if pc else np.zeros(0, int)
        # dense part as a dense (m x k) block
        self.Ad = np.zeros((m, dense_cols.size))
        if dense_cols.size:
            pos = {j: i for i, j in enumerate(dense_cols)}
            dsel = is_dense[cols]
            self.Ad[rows[dsel], [pos[j] for j in cols[dsel]]] = vals[dsel]
        # A as dense-free matvecs
        self.rows, self.cols, self.vals = rows, cols, vals
        self.blocks = self._blocks(m, border, rows, cols, is_dense) if border else None

    @staticmethod
    def _blocks(
        m: int, border: int, rows: np.ndarray, cols: np.ndarray, is_dense: np.ndarray
    ) -> list[np.ndarray] | None:
        """Connected groups of non-border rows (rows sharing a column)."""
        parent = np.arange(m)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        order = np.argsort(cols, kind="stable")
        r_s, c_s = rows[order], cols[order]
        cut = np.flatnonzero(np.diff(c_s)) + 1
        for grp in np.split(r_s, cut):
            inner = grp[grp >= border]
            for a in inner[1:]:
                ra, rb = find(inner[0]), find(a)
                if ra != rb:
                    parent[rb] = ra
        roots = np.array([find(i) for i in range(border, m)])
        return [np.arange(border, m)[roots == rt] for rt in np.unique(roots)] or None

    def matrix(self, d: np.ndarray) -> np.ndarray:
        M = (
            np.bincount(self.flat, weights=self.pv * d[self.pc], minlength=self.m * self.m)
            .astype(float)
            .reshape(self.m, self.m)
        )
        if self.dense_cols.size:
            Ad = self.Ad
            M += (Ad * d[self.dense_cols]) @ Ad.T
        return M

    def Ax(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.rows, weights=self.vals * x[self.cols], minlength=self.m).astype(
            float
        )

    def ATy(self, y: np.ndarray) -> np.ndarray:
        return np.bincount(self.cols, weights=self.vals * y[self.rows], minlength=self.n).astype(
            float
        )


# A best iterate this accurate (relative residuals and duality gap) is
# accepted when the method cannot get closer. Late in a degenerate solve the
# normal equations are too ill-conditioned for 1e-8, and the gap can level off
# near 1e-5 with residuals at 1e-9 - an answer that matches HiGHS to 2e-5
# relative. The DW loop itself stops at a relative gap of 1e-3, so failing such
# an answer (and the plan with it) would be the worse outcome.
ACCEPT = 1e-4


def linprog(
    c: np.ndarray,
    A: Triplets,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 100,
    border: int = 0,
) -> LPResult:
    """min c.x  s.t.  A x = b,  lb <= x <= ub.  lb must be finite; ub may be inf.

    Variables with ub == lb (curtailment at night, plans excluded from a
    branch) are fixed: they are taken out before the interior point starts,
    which needs every bounded variable to have room strictly inside.
    """
    c = np.asarray(c, float)
    b = np.asarray(b, float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    m, n_all = A.shape
    rows, cols, vals = A.coo()
    if not np.all(np.isfinite(lb)):
        raise ValueError("lower bounds must be finite")
    fixed = ub - lb <= 1e-12
    if fixed.all():
        # nothing left to solve: x is its bounds, feasible if it meets A x = b
        x = lb.copy()
        residual = b - np.bincount(rows, weights=vals * x[cols], minlength=m)
        ok = bool(np.all(np.abs(residual) <= 1e-9 * (1.0 + np.abs(b))))
        return LPResult(
            x,
            float(c @ x),
            np.zeros(m),
            0 if ok else 2,
            0,
            "optimal (every variable fixed)" if ok else "infeasible (every variable fixed)",
        )
    if fixed.any():
        b = b - np.bincount(rows, weights=vals * np.where(fixed, lb, 0.0)[cols], minlength=m)
        keep = ~fixed
        new_index = np.cumsum(keep) - 1
        sel = keep[cols]
        rows, cols, vals = rows[sel], new_index[cols[sel]], vals[sel]
        res = _solve(
            c[keep],
            rows,
            cols,
            vals,
            m,
            int(keep.sum()),
            b,
            lb[keep],
            ub[keep],
            tol,
            max_iter,
            border,
        )
        x = lb.copy()
        x[keep] = res.x
        return LPResult(x, float(c @ x), res.y, res.status, res.iterations, res.message)
    return _solve(c, rows, cols, vals, m, n_all, b, lb, ub, tol, max_iter, border)


def _factor(M: np.ndarray, ne: _NormalEquations) -> Callable[[np.ndarray], np.ndarray]:
    """A solve(r) for M r' = r: block-arrow elimination when the structure is
    there, a dense Cholesky otherwise."""
    M[np.diag_indices_from(M)] += 1e-12 * max(1.0, np.abs(np.diag(M)).max())
    blocks = ne.blocks
    if not blocks:
        Li = np.linalg.inv(np.linalg.cholesky(M))
        return lambda r: Li.T @ (Li @ r)
    B = np.arange(min(int(bl.min()) for bl in blocks))  # the border rows
    S = M[np.ix_(B, B)].copy()
    facts = []
    for P in blocks:
        Li = np.linalg.inv(np.linalg.cholesky(M[np.ix_(P, P)]))
        MPB = M[np.ix_(P, B)]
        Y = Li.T @ (Li @ MPB)  # M_PP^-1 M_PB
        S -= MPB.T @ Y  # Schur complement
        facts.append((P, Li, MPB, Y))
    SLi = np.linalg.inv(np.linalg.cholesky(S))

    def solve(r: np.ndarray) -> np.ndarray:
        rB = r[B].copy()
        ts = []
        for P, Li, MPB, _Y in facts:
            tP = Li.T @ (Li @ r[P])
            rB -= MPB.T @ tP
            ts.append(tP)
        xB = SLi.T @ (SLi @ rB)
        out = np.empty_like(r)
        out[B] = xB
        for (P, _Li, _MPB, Y), tP in zip(facts, ts):
            out[P] = tP - Y @ xB
        return out

    return solve


def _solve(
    c: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    m: int,
    n: int,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    tol: float,
    max_iter: int,
    border: int = 0,
) -> LPResult:
    ne = _NormalEquations(m, n, rows, cols, vals, border)

    # shift to 0 <= x <= u
    b0 = b - ne.Ax(lb)
    u = ub - lb
    U = np.isfinite(u)
    uU = u[U]

    # scale-aware starting point, strictly inside the bounds
    x = np.where(U, u / 2.0, np.maximum(1.0, np.abs(b0).max() if b0.size else 1.0))
    w = uU - x[U]
    z = np.ones(n) * max(1.0, np.abs(c).max())
    s = np.ones(U.sum()) * max(1.0, np.abs(c).max())
    y = np.zeros(m)
    bn, cn = 1.0 + np.linalg.norm(b0), 1.0 + np.linalg.norm(c)

    def full(vU: np.ndarray) -> np.ndarray:
        out = np.zeros(n)
        out[U] = vU
        return out

    status, iters, msg = 1, 0, "iteration limit"
    best = (np.inf, x, y)
    for iters in range(1, max_iter + 1):  # noqa: B007 - reported after the loop
        rb = b0 - ne.Ax(x)
        ru = uU - x[U] - w
        rc = c - ne.ATy(y) - z + full(s)
        mu = (x @ z + w @ s) / (n + U.sum())
        pobj = c @ x
        dobj = b0 @ y - uU @ s
        err = max(
            np.linalg.norm(rb) / bn,
            np.linalg.norm(rc) / cn,
            np.linalg.norm(ru) / (1 + np.linalg.norm(uU)) if U.any() else 0.0,
            abs(pobj - dobj) / (1 + abs(pobj)),
        )
        # Keep the best iterate: once mu is tiny the normal equations become
        # numerically singular and further steps can only lose accuracy.
        if err < best[0]:
            best = (err, x.copy(), y.copy())
        if err < tol:
            status, msg = 0, "optimal"
            break
        if mu < 1e-13 * (1 + abs(pobj)):
            status = 0 if best[0] < ACCEPT else 2
            msg = "optimal (complementarity exhausted)" if status == 0 else "stalled"
            break

        dinv_raw = z / x + full(s / w)
        D = 1.0 / dinv_raw
        M = ne.matrix(D)
        # _factor regularises M in place, by a multiple of its LARGEST
        # diagonal. Late in the solve D spans ~1e12, and that shift swamps
        # the rows with small diagonals: the steps stop satisfying A dx = rb,
        # the primal residual grows back and the solve stalls a hair short of
        # tolerance. Refining against the unregularised M removes it.
        M_exact = M.copy()
        try:
            # One factorisation per step, reused by predictor and corrector.
            # numpy has no triangular solve, so the Cholesky factor is
            # inverted once (m^3/3) - four general solves cost ~4x that.
            solve = _factor(M, ne)
        except np.linalg.LinAlgError:
            Mi = np.linalg.pinv(M)
            solve = lambda r: Mi @ r  # noqa: E731

        def direction(
            rxz: np.ndarray, rws: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            rhat = rc - rxz / x + full((rws - s * ru) / w)
            r_dy = rb + ne.Ax(D * rhat)
            dy = solve(r_dy)
            for _ in range(2):  # iterative refinement
                dy = dy + solve(r_dy - M_exact @ dy)
            dx = D * (ne.ATy(dy) - rhat)
            dz = (rxz - z * dx) / x
            dw = ru - dx[U]
            ds = (rws - s * dw) / w
            return dx, dy, dz, dw, ds

        def step(v: np.ndarray, dv: np.ndarray) -> float:
            neg = dv < 0
            return min(1.0, float(np.min(-v[neg] / dv[neg]))) if neg.any() else 1.0

        # predictor
        dx, dy, dz, dw, ds = direction(-x * z, -w * s)
        ap = min(step(x, dx), step(w, dw) if U.any() else 1.0)
        ad = min(step(z, dz), step(s, ds) if U.any() else 1.0)
        mu_aff = ((x + ap * dx) @ (z + ad * dz) + (w + ap * dw) @ (s + ad * ds)) / (n + U.sum())
        sigma = (mu_aff / mu) ** 3
        # corrector
        dx, dy, dz, dw, ds = direction(sigma * mu - x * z - dx * dz, sigma * mu - w * s - dw * ds)
        ap = 0.995 * min(step(x, dx), step(w, dw) if U.any() else 1.0)
        ad = 0.995 * min(step(z, dz), step(s, ds) if U.any() else 1.0)
        x = x + ap * dx
        w = w + ap * dw
        y = y + ad * dy
        z = z + ad * dz
        s = s + ad * ds
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            status, msg = 2, "numerical trouble"
            break

    if status != 0 and best[0] < ACCEPT:
        status, msg = 0, "optimal (best iterate)"
    x, y = best[1], best[2]
    xs = x + lb
    return LPResult(xs, float(c @ xs), y, status, iters, msg)
