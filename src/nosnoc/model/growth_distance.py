from typing import List
from dataclasses import dataclass

import casadi as ca
import numpy as np


class Polytope2D:
    """Bounded convex polytope {p | G p <= h} in body frame, rows of G are normalized."""
    def __init__(self, G: np.ndarray, h: np.ndarray):
        G = np.atleast_2d(np.asarray(G, dtype=float))
        h = np.asarray(h, dtype=float).flatten()
        if G.shape[1] != 2 or G.shape[0] != h.shape[0]:
            raise RuntimeError(f"G must be n_h x 2 and h of length n_h, got {G.shape} and {h.shape}.")
        norms = np.linalg.norm(G, axis=1)
        self.G = G/norms[:, None]
        self.h = h/norms
        if np.any(self.h <= 0):
            raise RuntimeError("The origin must lie in the interior of the polytope, i.e. h > 0.")
        angles = np.sort(np.arctan2(self.G[:, 1], self.G[:, 0]))
        if np.max(np.diff(np.append(angles, angles[0] + 2*np.pi))) >= np.pi:
            raise RuntimeError("The polytope is unbounded.")
        self.n_h = self.G.shape[0]

    @classmethod
    def box(cls, width: float, height: float) -> "Polytope2D":
        G = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        h = np.array([width, width, height, height])/2
        return cls(G, h)

    @classmethod
    def regular(cls, n: int, circumradius: float, angle_offset: float = 0.0) -> "Polytope2D":
        normal_angles = angle_offset + np.pi*(2*np.arange(n) + 1)/n
        G = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
        return cls(G, circumradius*np.cos(np.pi/n)*np.ones(n))

    def vertices(self) -> np.ndarray:
        order = np.argsort(np.arctan2(self.G[:, 1], self.G[:, 0]))
        G, h = self.G[order], self.h[order]
        return np.array([np.linalg.solve(G[[k, (k+1) % self.n_h]], h[[k, (k+1) % self.n_h]])
                         for k in range(self.n_h)])


def rotation_2d(theta):
    return ca.vertcat(ca.horzcat(ca.cos(theta), -ca.sin(theta)),
                      ca.horzcat(ca.sin(theta), ca.cos(theta)))


class GrowthDistancePair2D:
    """Growth distance between P_i and P_j, embedded via relaxed LP KKT conditions in z = (p, alpha, lambda, y)."""
    def __init__(self,
                 P_i: Polytope2D, rho_i, theta_i,
                 P_j: Polytope2D, rho_j, theta_j,
                 q: ca.SX,
                 tau: ca.SX|float,
                 name: str = "gd"):
        self.n_g = P_i.n_h + P_j.n_h

        GR_i = P_i.G@rotation_2d(theta_i).T
        GR_j = P_j.G@rotation_2d(theta_j).T
        A = ca.vertcat(ca.horzcat(GR_i, -np.ones((P_i.n_h, 1))),
                       ca.horzcat(GR_j, -np.ones((P_j.n_h, 1))))
        b = ca.vertcat(P_i.h + GR_i@rho_i, P_j.h + GR_j@rho_j)
        self.c = np.array([0.0, 0.0, 2.0])

        p = ca.SX.sym(f"{name}_p", 2)
        alpha = ca.SX.sym(f"{name}_alpha")
        lam = ca.SX.sym(f"{name}_lambda", self.n_g)
        y = ca.SX.sym(f"{name}_y", self.n_g)
        z_lp = ca.vertcat(p, alpha)
        self.z = ca.vertcat(z_lp, lam, y)

        self.g_eq = ca.vertcat(self.c + A.T@lam, y - b + A@z_lp)
        self.g_ineq = lam*y - tau
        self.lbz = np.concatenate([-np.inf*np.ones(3), np.zeros(2*self.n_g)])
        self.ubz = np.inf*np.ones(3 + 2*self.n_g)

        self.f_c = 2*alpha
        self.J_normal = ca.jacobian(A@z_lp - b, q).T@lam

        self.A_b_fun = ca.Function(f"{name}_A_b", [q], [A, b])

    def initial_guess(self, q0: np.ndarray, tau_val: float) -> np.ndarray:
        """Central path point at q0, i.e. lambda*y = tau_val."""
        A, b = (np.array(M) for M in self.A_b_fun(q0))
        b = b.flatten()
        z_lp = _barrier_central_path(A, b, self.c, tau_val)
        y = b - A@z_lp
        return np.concatenate([z_lp, tau_val/y, y])


@dataclass
class GrowthDistances2D:
    pairs: List[GrowthDistancePair2D]
    z: ca.SX
    g_z: ca.SX
    lbg_z: np.ndarray
    ubg_z: np.ndarray
    lbz: np.ndarray
    ubz: np.ndarray
    f_c: ca.SX
    J_normal: ca.SX

    def initial_guess(self, q0: np.ndarray, tau_val: float) -> np.ndarray:
        return np.concatenate([pair.initial_guess(q0, tau_val) for pair in self.pairs])


def stack_growth_distances(pairs: List[GrowthDistancePair2D], relaxed: bool = True) -> GrowthDistances2D:
    """relaxed: lambda*y <= tau (Scholtes relaxation), otherwise lambda*y = tau (central path)."""
    n_eq = [pair.g_eq.size1() for pair in pairs]
    lb_lam_y = -np.inf if relaxed else 0.0
    return GrowthDistances2D(
        pairs=pairs,
        z=ca.vertcat(*[pair.z for pair in pairs]),
        g_z=ca.vertcat(*[ca.vertcat(pair.g_eq, pair.g_ineq) for pair in pairs]),
        lbg_z=np.concatenate([np.r_[np.zeros(n), lb_lam_y*np.ones(pair.n_g)] for n, pair in zip(n_eq, pairs)]),
        ubg_z=np.concatenate([np.zeros(n + pair.n_g) for n, pair in zip(n_eq, pairs)]),
        lbz=np.concatenate([pair.lbz for pair in pairs]),
        ubz=np.concatenate([pair.ubz for pair in pairs]),
        f_c=ca.vertcat(*[pair.f_c for pair in pairs]),
        J_normal=ca.horzcat(*[pair.J_normal for pair in pairs]),
    )


def _barrier_central_path(A, b, c, tau, tol=1e-10, max_iter=200):
    # alpha has coefficient -1 in every row, so a large alpha is strictly feasible.
    z = np.zeros(A.shape[1])
    z[-1] = np.max(-b) + 1.0
    tau_k = max(tau, 1.0)
    while True:
        z = _barrier_newton(A, b, c, tau_k, z, tol, max_iter)
        if tau_k <= tau:
            return z
        tau_k = max(0.1*tau_k, tau)


def _barrier_newton(A, b, c, tau, z, tol, max_iter):
    def f(z):
        return c@z - tau*np.sum(np.log(b - A@z))

    for _ in range(max_iter):
        s = b - A@z
        grad = c + tau*A.T@(1/s)
        if np.max(np.abs(grad)) <= tol:
            return z
        # QR of the weighted rows instead of forming the Hessian A^T diag(tau/s^2) A, which squares the condition number.
        R = np.linalg.qr((np.sqrt(tau)/s)[:, None]*A, mode="r")
        w = np.linalg.solve(R.T, grad)
        dz = -np.linalg.solve(R, w)
        decrement = w@w
        # Small slacks are computed as differences of O(1) numbers, so the gradient stagnates at roundoff for small tau.
        if decrement/tau <= 1e-12:
            return z
        Adz = A@dz
        t = min(1.0, 0.99*np.min(s[Adz > 0]/Adz[Adz > 0])) if np.any(Adz > 0) else 1.0
        # f/tau is self-concordant: full steps converge quadratically once its decrement is below 1/4.
        if decrement/tau >= 0.25:
            f0 = f(z)
            while f(z + t*dz) > f0 - 0.25*t*decrement:
                t *= 0.5
        z = z + t*dz
    raise RuntimeError(f"Central path computation did not converge (tau = {tau}).")
