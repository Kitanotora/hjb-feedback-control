"""State-space grids with multilinear interpolation.

Each axis of a rectangular grid is either
  * clamped   -- linear interpolation inside, constant extrapolation outside, or
  * periodic  -- the axis wraps around (e.g. an angle theta in [-pi, pi)).
"""
from __future__ import annotations

from itertools import product

import numpy as np


class Grid:
    def __init__(self, lo, hi, n, periodic=()):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.n = np.asarray(n, dtype=int)
        self.dim = self.lo.size
        self.periodic = tuple(periodic)
        if self.lo.shape != self.hi.shape or self.lo.shape != self.n.shape:
            raise ValueError("lo, hi, n must have the same length")

        # Periodic axes tile [lo, hi) with n points; clamped axes include both ends.
        per = np.array([a in self.periodic for a in range(self.dim)])
        self.dx = np.where(per, (self.hi - self.lo) / self.n, (self.hi - self.lo) / (self.n - 1))
        self.shape = tuple(int(k) for k in self.n)
        self.size = int(np.prod(self.n))

        self._strides = np.ones(self.dim, dtype=np.int64)
        for a in range(self.dim - 2, -1, -1):
            self._strides[a] = self._strides[a + 1] * self.n[a + 1]
        self._corners = np.array(list(product((0, 1), repeat=self.dim)), dtype=np.int64)
        self._points = None

    def axes(self):
        return [self.lo[a] + self.dx[a] * np.arange(self.n[a]) for a in range(self.dim)]

    def points(self):
        """All grid points as an (N, d) array in C order (cached)."""
        if self._points is None:
            mesh = np.meshgrid(*self.axes(), indexing="ij")
            self._points = np.stack([x.ravel() for x in mesh], axis=1)
        return self._points

    def _frac_index(self, x):
        s = (x - self.lo) / self.dx
        i0 = np.floor(s).astype(np.int64)
        frac = s - i0  # in [0, 1)
        for a in range(self.dim):
            if a in self.periodic:
                i0[:, a] %= self.n[a]
            else:
                # Recompute frac against the *clipped* base index, otherwise
                # points beyond the boundary land on the wrong stencil.
                i0[:, a] = np.clip(i0[:, a], 0, self.n[a] - 2)
                frac[:, a] = np.clip(s[:, a] - i0[:, a], 0.0, 1.0)
        return i0, frac

    def interp_indices(self, pts):
        """Interpolation stencils at pts (N, d): flat corner indices (N, 2^d), weights (N, 2^d)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        i0, frac = self._frac_index(pts)
        N = i0.shape[0]
        C = self._corners.shape[0]
        idx = np.empty((N, C), dtype=np.int64)
        w = np.empty((N, C))
        for c, bits in enumerate(self._corners):
            flat = np.zeros(N, dtype=np.int64)
            weight = np.ones(N)
            for a in range(self.dim):
                ia = i0[:, a] + int(bits[a])
                if a in self.periodic:
                    ia %= self.n[a]
                else:
                    ia = np.minimum(ia, self.n[a] - 1)
                flat += ia * self._strides[a]
                f = frac[:, a]
                weight = weight * (f if bits[a] else 1.0 - f)
            idx[:, c] = flat
            w[:, c] = weight
        return idx, w

    def interp(self, V, pts):
        """Evaluate the grid function V (any shape) at points pts (N, d)."""
        Vf = np.asarray(V, dtype=float).ravel()
        idx, w = self.interp_indices(pts)
        return np.sum(w * Vf[idx], axis=1)
