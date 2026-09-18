from __future__ import annotations

"""
Adaptive NV charge-state readout: exact first-passage MMPP vs count threshold
=============================================================================

Goal
----
Demonstrate that an adaptive (sequential) MMPP readout reaches the SAME
initial-state fidelity as an optimized fixed-time count threshold in
significantly LESS average run time.

What this module adds to nv_charge_readout_master_v1_4.py
---------------------------------------------------------
1. Closed-form 2x2 no-click propagator.
   Replaces np.linalg.eig + lru_cache. The dominant eigenvalue is factored
   out analytically, so the propagator never overflows or underflows and is
   exactly real. The factored scalar cancels identically in the LLR.

2. EXACT, GRID-FREE first-passage times.
   Verified structural facts for the 2-state MMPP initial-state LLR:
       - during a no-click interval the LLR decreases monotonically,
         because d/dt log(1'p) = -1'(Lambda p) and the initially-bright
         hypothesis always has the larger filtered emission rate;
       - at every photon arrival the LLR jumps upward.
   Therefore:
       - the UPPER boundary can only be crossed AT a click;
       - the LOWER boundary can only be crossed INSIDE a no-click interval,
         where the crossing is unique.
   The lower crossing time is obtained in closed form (no bisection, no time
   grid) by solving a two-exponential ratio equation. See
   `_solve_lower_crossing`.

3. Truncated SPRT with ASYMMETRIC, CALIBRATED boundaries.
   Boundaries are U = offset + L and D = offset - L. The offset is calibrated
   on calibration data exactly as the count threshold is, removing the unfair
   hardcoded LLR >= 0 rule. The truncation tie-break also uses the offset.

4. Paired (stratified) bootstrap and exact McNemar tests.
   Both methods are evaluated on the same shots, so all comparisons are
   paired. Fidelity differences and speedup ratios come with CIs.

5. Dimensionless regime reporting.
   Photons per bright dwell, Gamma_tot * t_R, and emission contrast, instead
   of laser power in uW and a bare detection efficiency.

6. Switching-rate ensembles in POLYSPECTRA coordinates (log K, logit p_bright)
   so the Bayesian ensemble inherits the correct posterior correlation, plus
   a Laplace-covariance helper for the polyspectra least_squares fit.

State convention (inherited)
----------------------------
0 = NV-  (bright)
1 = NV0  (dark)

All rates are kHz = ms^-1; all times internally in ms.
"""

from dataclasses import dataclass
from typing import Sequence
import time

import numpy as np

import nv_charge_readout_master_v1_4 as nv

MMPPParams = nv.MMPPParams

MODULE_VERSION = "adaptive-charge-readout-v2.0"


# =============================================================================
# 1. Closed-form 2x2 no-click propagator
# =============================================================================


@dataclass(frozen=True)
class NoClickSpectral:
    """
    Spectral data for A = Q - Lambda, the 2-state no-click generator.

    A always has REAL eigenvalues when both switching rates are positive,
    because

        disc = ((A00 - A11)/2)^2 + A01 * A10,

    and A01 = Gamma_0-, A10 = Gamma_-0 are both non-negative.

    We factor out the dominant eigenvalue mu1:

        exp(A dt) = exp(mu1 dt) * V diag(1, exp(-2 delta dt)) V^-1

    The prefactor is common to both initial-state hypotheses and therefore
    cancels exactly in the LLR. Because exp(-2 delta dt) <= 1, the retained
    matrix is bounded for all dt.
    """

    mu1: float
    mu2: float
    delta: float
    V: np.ndarray
    V_inv: np.ndarray
    w: np.ndarray          # w = 1' V  (row sums of eigenvector matrix)
    lambda_diag: np.ndarray
    degenerate: bool

    def scaled_propagator(self, dt_ms: float) -> np.ndarray:
        """
        M(dt) = V diag(1, exp(-2 delta dt)) V^-1

        with exp(A dt) = exp(mu1 dt) M(dt).
        """
        if dt_ms <= 0.0:
            return np.eye(2)
        x = float(np.exp(-2.0 * self.delta * dt_ms))
        return self.V @ (np.array([1.0, x])[:, None] * self.V_inv)


def build_no_click_spectral(params: MMPPParams) -> NoClickSpectral:
    params.validate()

    A = np.asarray(params.no_click_generator, dtype=float)

    a, b = float(A[0, 0]), float(A[0, 1])
    c, d = float(A[1, 0]), float(A[1, 1])

    tau = 0.5 * (a + d)
    disc = 0.25 * (a - d) ** 2 + b * c

    if disc < 0.0:
        # Impossible for non-negative off-diagonals; guard anyway.
        raise FloatingPointError(
            "No-click generator has complex eigenvalues; check rate signs."
        )

    delta = float(np.sqrt(disc))
    mu1 = tau + delta
    mu2 = tau - delta

    degenerate = delta <= 1e-14 * max(1.0, abs(tau))

    if degenerate:
        V = np.eye(2)
        V_inv = np.eye(2)
    else:
        if abs(b) > 0.0:
            V = np.array(
                [
                    [b, b],
                    [mu1 - a, mu2 - a],
                ],
                dtype=float,
            )
        elif abs(c) > 0.0:
            V = np.array(
                [
                    [mu1 - d, mu2 - d],
                    [c, c],
                ],
                dtype=float,
            )
        else:
            V = np.eye(2)

        det_V = V[0, 0] * V[1, 1] - V[0, 1] * V[1, 0]

        if abs(det_V) < 1e-300:
            raise FloatingPointError("Degenerate eigenvector matrix.")

        V_inv = np.array(
            [
                [V[1, 1], -V[0, 1]],
                [-V[1, 0], V[0, 0]],
            ],
            dtype=float,
        ) / det_V

    return NoClickSpectral(
        mu1=float(mu1),
        mu2=float(mu2),
        delta=delta,
        V=V,
        V_inv=V_inv,
        w=V.sum(axis=0),
        lambda_diag=np.array(
            [params.lambda_minus_khz, params.lambda_zero_khz],
            dtype=float,
        ),
        degenerate=bool(degenerate),
    )


def _normalize_columns(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each hypothesis column; return (H, log of removed scale)."""
    s = H.sum(axis=0)

    if np.any(s <= 0.0) or np.any(~np.isfinite(s)):
        raise FloatingPointError("Invalid hypothesis normalization.")

    return H / s[None, :], np.log(s)


# =============================================================================
# 2. Event-time LLR record (everything needed for exact first passage)
# =============================================================================


@dataclass
class LLRRecord:
    """
    Complete piecewise description of the initial-state LLR for one shot.

    Intervals are indexed k = 0 .. n_clicks. Interval k runs from
    t_start[k] to t_end[k]; interval n_clicks ends at t_max.

    Within each interval the LLR falls monotonically from llr_start[k] to
    llr_end[k]. At click k the LLR jumps upward from llr_end[k] to
    llr_post[k].
    """

    t_start: np.ndarray        # (n_int,)
    t_end: np.ndarray          # (n_int,)
    llr_start: np.ndarray      # (n_int,)
    llr_end: np.ndarray        # (n_int,)
    llr_post: np.ndarray       # (n_clicks,)
    H_start: np.ndarray        # (n_int, 2, 2) normalized hypothesis columns
    t_max_ms: float
    n_clicks: int

    @property
    def llr_final(self) -> float:
        return float(self.llr_end[-1])


def build_llr_record(
    timestamps_ms: Sequence[float],
    t_max_ms: float,
    spec: NoClickSpectral,
) -> LLRRecord:
    """
    Single forward pass producing the full LLR trajectory description.

    Cost is O(n_clicks) with small dense 2x2 algebra. Once built, a record
    can be re-scanned for ANY boundary pair at negligible cost, which makes
    boundary calibration and bootstrap resampling cheap.
    """

    t_max_ms = float(t_max_ms)

    ts = np.sort(np.asarray(timestamps_ms, dtype=float).reshape(-1))
    ts = ts[(ts >= 0.0) & (ts < t_max_ms)]

    n_clicks = int(ts.size)
    n_int = n_clicks + 1

    t_start = np.empty(n_int)
    t_end = np.empty(n_int)
    llr_start = np.empty(n_int)
    llr_end = np.empty(n_int)
    llr_post = np.empty(n_clicks)
    H_start = np.empty((n_int, 2, 2))

    # Column 0 = "initial state was NV-", column 1 = "initial state was NV0".
    H = np.eye(2)
    log_scale = np.zeros(2)

    t_now = 0.0

    for k in range(n_int):
        t_stop = float(ts[k]) if k < n_clicks else t_max_ms

        t_start[k] = t_now
        t_end[k] = t_stop
        H_start[k] = H
        llr_start[k] = log_scale[0] - log_scale[1]

        dt = t_stop - t_now

        if dt > 0.0:
            M = spec.scaled_propagator(dt)
            H = M @ H
            H, dlog = _normalize_columns(H)
            # The factored exp(mu1 dt) is common to both columns and cancels
            # in the LLR, but we keep it for absolute log-likelihoods.
            log_scale = log_scale + dlog + spec.mu1 * dt

        llr_end[k] = log_scale[0] - log_scale[1]

        if k < n_clicks:
            # Photon jump: H -> Lambda H.
            H = spec.lambda_diag[:, None] * H
            H, dlog = _normalize_columns(H)
            log_scale = log_scale + dlog
            llr_post[k] = log_scale[0] - log_scale[1]

        t_now = t_stop

    return LLRRecord(
        t_start=t_start,
        t_end=t_end,
        llr_start=llr_start,
        llr_end=llr_end,
        llr_post=llr_post,
        H_start=H_start,
        t_max_ms=t_max_ms,
        n_clicks=n_clicks,
    )


def build_records(
    shots: Sequence[np.ndarray],
    t_max_us: float,
    params: MMPPParams,
) -> list[LLRRecord]:
    spec = build_no_click_spectral(params)
    t_max_ms = float(t_max_us) / 1000.0
    return [build_llr_record(ts, t_max_ms, spec) for ts in shots]


# =============================================================================
# 4. Truncated SPRT with asymmetric, calibrated boundaries
# =============================================================================


def _interval_llr_coefficients(
    H_start: np.ndarray,
    spec: NoClickSpectral,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-exponential coefficients for both hypotheses inside one no-click
    interval.

        1' exp(A dt) p_j = exp(mu1 dt) * (alpha_j + beta_j x),
        x = exp(-2 delta dt),  alpha_j + beta_j = 1.
    """
    U = spec.V_inv @ H_start
    alpha = spec.w[0] * U[0, :]
    beta = spec.w[1] * U[1, :]
    return alpha, beta


def _llr_after(
    alpha: np.ndarray,
    beta: np.ndarray,
    llr_start: float,
    dt_ms: float,
    spec: NoClickSpectral,
) -> float:
    """Exact LLR a time dt into a no-click interval. Monotonically decreasing."""
    if dt_ms <= 0.0:
        return float(llr_start)
    x = 1.0 if spec.degenerate else float(np.exp(-2.0 * spec.delta * dt_ms))
    g0 = alpha[0] + beta[0] * x
    g1 = alpha[1] + beta[1] * x
    return float(llr_start + np.log(g0) - np.log(g1))


def _solve_lower_crossing(
    alpha: np.ndarray,
    beta: np.ndarray,
    llr_start: float,
    target_llr: float,
    dt_max_ms: float,
    spec: NoClickSpectral,
) -> float:
    """
    Exact first time the LLR reaches `target_llr` inside a no-click interval.

    Setting LLR(dt) = target with D = exp(target - llr_start):

        alpha_0 + beta_0 x = D (alpha_1 + beta_1 x)
        =>  x  = (D alpha_1 - alpha_0) / (beta_0 - D beta_1)
        =>  dt = -log(x) / (2 delta)

    Closed form, no iteration. Bisection is a guard for the degenerate case.
    """

    if not spec.degenerate:
        D = float(np.exp(target_llr - llr_start))
        num = D * alpha[1] - alpha[0]
        den = beta[0] - D * beta[1]

        if abs(den) > 1e-300:
            x = num / den
            if 0.0 < x <= 1.0:
                dt = -np.log(x) / (2.0 * spec.delta)
                if 0.0 <= dt <= dt_max_ms * (1.0 + 1e-9):
                    return float(min(dt, dt_max_ms))

    lo, hi = 0.0, float(dt_max_ms)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _llr_after(alpha, beta, llr_start, mid, spec) <= target_llr:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def first_passage(
    record: LLRRecord,
    boundary_half_width: float,
    offset: float,
    spec: NoClickSpectral,
    deadline_ms: float | None = None,
) -> tuple[float, int]:
    """
    Truncated SPRT on one shot.

        U = offset + L  -> stop, decide NV- (0)
        D = offset - L  -> stop, decide NV0 (1)

    Stops at the first boundary passage or at the deadline, whichever comes
    first. At the deadline the decision uses `offset` as the cutoff, i.e. the
    calibrated fixed-time rule.

    The upper boundary is only tested at clicks and the lower boundary only
    inside no-click intervals, which is exact given the monotone-decrease /
    upward-jump structure of the LLR.

    `deadline_ms` may be shorter than the record's own t_max, so one record
    set serves an entire grid of deadlines with no refiltering.
    """

    L = float(boundary_half_width)
    b = float(offset)

    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("boundary_half_width must be positive and finite.")

    if abs(b) >= L:
        raise ValueError(
            "offset must satisfy |offset| < boundary_half_width."
        )

    U = b + L
    D = b - L

    t_cap = record.t_max_ms if deadline_ms is None else min(
        float(deadline_ms), record.t_max_ms
    )

    n_int = record.t_start.size

    for k in range(n_int):

        t0 = float(record.t_start[k])

        if t0 >= t_cap:
            break

        t1_full = float(record.t_end[k])
        t1 = min(t1_full, t_cap)
        dt = t1 - t0

        truncated_here = t1 < t1_full

        if truncated_here:
            alpha, beta = _interval_llr_coefficients(record.H_start[k], spec)
            llr_at_t1 = _llr_after(
                alpha, beta, float(record.llr_start[k]), dt, spec
            )
        else:
            alpha = beta = None
            llr_at_t1 = float(record.llr_end[k])

        if llr_at_t1 <= D:
            if alpha is None:
                alpha, beta = _interval_llr_coefficients(
                    record.H_start[k], spec
                )
            dt_cross = _solve_lower_crossing(
                alpha=alpha,
                beta=beta,
                llr_start=float(record.llr_start[k]),
                target_llr=D,
                dt_max_ms=dt,
                spec=spec,
            )
            return t0 + dt_cross, 1

        if truncated_here:
            return t_cap, (0 if llr_at_t1 >= b else 1)

        if k < record.n_clicks and float(record.llr_post[k]) >= U:
            return t1, 0

    # Deadline coincides with a stored interval boundary.
    llr_end_at_cap = float(record.llr_end[
        min(int(np.searchsorted(record.t_end, t_cap, side="left")), n_int - 1)
    ])
    return t_cap, (0 if llr_end_at_cap >= b else 1)


def evaluate_adaptive(
    records: Sequence[LLRRecord],
    boundary_half_width: float,
    offset: float,
    spec: NoClickSpectral,
    deadline_us: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the truncated SPRT over many records.

    Returns
    -------
    (stop_times_us, predictions)
    """

    n = len(records)
    times = np.empty(n)
    preds = np.empty(n, dtype=int)

    deadline_ms = None if deadline_us is None else float(deadline_us) / 1000.0

    for i, rec in enumerate(records):
        t_stop, pred = first_passage(
            rec,
            boundary_half_width,
            offset,
            spec,
            deadline_ms=deadline_ms,
        )
        times[i] = t_stop * 1000.0
        preds[i] = pred

    return times, preds


# =============================================================================
# 5. Vectorized SPRT engine
# =============================================================================


@dataclass
class PaddedRecords:
    """
    Array form of many LLRRecords, padded to a common interval count.

    Because the full LLR trajectory is precomputed once, an entire grid of
    boundary widths, offsets and deadlines can be scanned with pure array
    operations and no refiltering. Padding uses +inf for llr_end (never
    triggers the lower boundary) and -inf for llr_post (never triggers the
    upper boundary).
    """

    t_start: np.ndarray       # (n, M)
    t_end: np.ndarray         # (n, M)
    llr_start: np.ndarray     # (n, M)
    llr_end: np.ndarray       # (n, M)
    llr_post: np.ndarray      # (n, M)
    alpha: np.ndarray         # (n, M, 2)
    beta: np.ndarray          # (n, M, 2)
    hilbert: np.ndarray       # (n, M) remaining information, see `hilbert_gap`
    n_int: np.ndarray         # (n,)
    n_clicks: np.ndarray      # (n,)
    t_max_ms: float
    spec: NoClickSpectral

    @property
    def n_shots(self) -> int:
        return int(self.t_start.shape[0])


def pack_records(
    records: Sequence[LLRRecord],
    spec: NoClickSpectral,
) -> PaddedRecords:
    n = len(records)
    M = max(int(r.t_start.size) for r in records)

    t_start = np.full((n, M), np.inf)
    t_end = np.full((n, M), np.inf)
    llr_start = np.zeros((n, M))
    llr_end = np.full((n, M), np.inf)
    llr_post = np.full((n, M), -np.inf)
    alpha = np.zeros((n, M, 2))
    beta = np.zeros((n, M, 2))
    hilbert = np.zeros((n, M))
    n_int = np.zeros(n, dtype=int)
    n_clicks = np.zeros(n, dtype=int)

    for i, r in enumerate(records):
        k = int(r.t_start.size)
        n_int[i] = k
        n_clicks[i] = r.n_clicks

        t_start[i, :k] = r.t_start
        t_end[i, :k] = r.t_end
        llr_start[i, :k] = r.llr_start
        llr_end[i, :k] = r.llr_end
        llr_post[i, : r.n_clicks] = r.llr_post

        for j in range(k):
            a, b = _interval_llr_coefficients(r.H_start[j], spec)
            alpha[i, j] = a
            beta[i, j] = b
            hilbert[i, j] = hilbert_gap(r.H_start[j])

    return PaddedRecords(
        t_start=t_start,
        t_end=t_end,
        llr_start=llr_start,
        llr_end=llr_end,
        llr_post=llr_post,
        alpha=alpha,
        beta=beta,
        hilbert=hilbert,
        n_int=n_int,
        n_clicks=n_clicks,
        t_max_ms=float(records[0].t_max_ms),
        spec=spec,
    )


def hilbert_gap(H: np.ndarray) -> float:
    """
    Remaining initial-state information, as the Hilbert projective distance
    between the two hypothesis columns of H:

        d = | log(u/(1-u)) - log(v/(1-v)) |

    where u and v are the filtered probabilities of being bright NOW, given
    that the shot started bright and started dark respectively.

    Why this and not the plain difference u - v. The LLR drift is exactly
    -Delta_lambda (u - v) and the click jump is log[(l0+Dl u)/(l0+Dl v)], so
    u - v looks like the natural measure of remaining information -- but it is
    NOT monotone. Measured on simulated shots it can rise by up to 0.56 across
    a single click. The Hilbert metric is the monotone one, because:

        - the no-click propagator exp((Q-Lambda) dt) is a strictly positive
          matrix, so Birkhoff's theorem makes it a strict contraction;
        - the click update multiplies by the diagonal Lambda, and
          (Dx)_i/(Dy)_i = x_i/y_i, so it is an exact ISOMETRY.

    Information about the initial state is therefore destroyed only by the
    passage of time, never by observing a photon. Verified to 4e-15.

    When d reaches zero the two columns coincide; the propagator and the jump
    then act identically on them, so the LLR is frozen for the remainder of
    the shot and no further observation can change the decision.
    """
    num = np.maximum(H[:, 0], 1e-300)
    den = np.maximum(H[:, 1], 1e-300)
    ratio = num / den
    with np.errstate(divide="ignore", invalid="ignore"):
        d = float(np.log(ratio.max()) - np.log(ratio.min()))
    # A column that has collapsed onto one state carries maximal information,
    # not zero; guard against that being reported as an exhausted shot.
    return d if np.isfinite(d) else np.inf


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Index of first True per row, or mask.shape[1] if none."""
    any_true = mask.any(axis=1)
    idx = np.argmax(mask, axis=1)
    return np.where(any_true, idx, mask.shape[1])


def run_sprt(
    packed: PaddedRecords,
    boundary_half_width: float,
    offset: float,
    deadline_us: float,
    exhaustion_eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized truncated SPRT over all shots at once.

    Boundaries U = offset + L (decide NV-) and D = offset - L (decide NV0).
    Exact: the upper boundary is tested only at clicks, the lower boundary
    only inside no-click intervals, and the lower crossing time is closed
    form. Matches the scalar `first_passage` to machine precision.

    Returns
    -------
    (stop_times_us, predictions)
    """

    L = float(boundary_half_width)
    b = float(offset)

    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("boundary_half_width must be positive and finite.")

    if abs(b) >= L:
        raise ValueError(
            "offset must satisfy |offset| < boundary_half_width so that the "
            "boundaries bracket the initial LLR of zero; otherwise the test "
            "is decided before any photons are observed."
        )

    spec = packed.spec
    t_cap = min(float(deadline_us) / 1000.0, packed.t_max_ms)

    U = b + L
    D = b - L

    n, M = packed.t_start.shape
    rows = np.arange(n)

    valid = packed.t_start < t_cap
    eff_end = np.minimum(packed.t_end, t_cap)
    is_trunc = valid & (packed.t_end > t_cap)

    # Exact LLR at the deadline for the interval that contains it.
    dt_trunc = np.where(is_trunc, t_cap - packed.t_start, 0.0)
    x_trunc = (
        np.ones_like(dt_trunc)
        if spec.degenerate
        else np.exp(-2.0 * spec.delta * dt_trunc)
    )
    g0 = packed.alpha[:, :, 0] + packed.beta[:, :, 0] * x_trunc
    g1 = packed.alpha[:, :, 1] + packed.beta[:, :, 1] * x_trunc

    with np.errstate(divide="ignore", invalid="ignore"):
        llr_trunc = packed.llr_start + np.log(g0) - np.log(g1)

    llr_eff = np.where(is_trunc, llr_trunc, packed.llr_end)

    # Lower boundary: only inside no-click intervals (monotone decrease).
    cand_D = valid & (llr_eff <= D)
    k_D = _first_true(cand_D)

    # Upper boundary: only at clicks, and only if the click precedes the
    # deadline.
    click_valid = (
        (np.arange(M)[None, :] < packed.n_clicks[:, None])
        & (packed.t_end <= t_cap)
    )
    cand_U = click_valid & (packed.llr_post >= U)
    k_U = _first_true(cand_U)

    hit_D = k_D < M
    hit_U = k_U < M

    # Ties within the same interval go to the lower boundary, because the
    # LLR reaches D before the click that would raise it to U.
    lower_first = hit_D & (~hit_U | (k_D <= k_U))
    upper_first = hit_U & ~lower_first

    # Closed-form lower crossing time at interval k_D.
    kD = np.where(hit_D, k_D, 0)
    a0 = packed.alpha[rows, kD, 0]
    a1 = packed.alpha[rows, kD, 1]
    b0 = packed.beta[rows, kD, 0]
    b1 = packed.beta[rows, kD, 1]
    l0 = packed.llr_start[rows, kD]
    dt_max = eff_end[rows, kD] - packed.t_start[rows, kD]

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        Dexp = np.exp(D - l0)
        num = Dexp * a1 - a0
        den = b0 - Dexp * b1
        xr = np.where(np.abs(den) > 1e-300, num / den, np.nan)
        dt_cross = np.where(
            (xr > 0.0) & (xr <= 1.0) & np.isfinite(xr),
            -np.log(np.clip(xr, 1e-300, 1.0)) / (2.0 * max(spec.delta, 1e-300)),
            dt_max,
        )

    # If the LLR at the interval start is already at or below D, the crossing
    # happened at the interval start itself; the closed form finds no forward
    # crossing in that case and must not fall back to dt_max.
    dt_cross = np.where(l0 <= D, 0.0, dt_cross)
    dt_cross = np.clip(dt_cross, 0.0, np.maximum(dt_max, 0.0))
    t_lower = packed.t_start[rows, kD] + dt_cross
    t_upper = packed.t_end[rows, np.where(hit_U, k_U, 0)]

    # Truncation: LLR at the deadline is the effective end of the last valid
    # interval.
    n_valid = np.maximum(valid.sum(axis=1), 1)
    llr_at_cap = llr_eff[rows, n_valid - 1]

    stop_ms = np.where(lower_first, t_lower, np.where(upper_first, t_upper, t_cap))
    preds = np.where(
        lower_first,
        1,
        np.where(upper_first, 0, np.where(llr_at_cap >= b, 0, 1)),
    ).astype(int)

    # ---- third exit: the initial-state information is exhausted -----------
    # Once the Hilbert gap reaches zero the two hypothesis columns coincide,
    # the LLR is frozen, and no further observation can change the decision.
    # Stopping there is therefore FREE: unlike the boundary, this is not a
    # speed/accuracy trade but a proof that nothing more is coming. The
    # decision taken is bit-identical to the one the deadline would have
    # produced.
    #
    # The gap is monotone non-increasing (see `hilbert_gap`), so a
    # first-passage test on it is well posed. It is evaluated at interval
    # starts only, which is conservative: the true crossing lies inside the
    # preceding interval, so this reports a slightly later stop than optimal
    # and never an earlier one.
    if exhaustion_eps > 0.0:
        exhausted = valid & (packed.hilbert <= exhaustion_eps)
        k_E = _first_true(exhausted)
        hit_E = k_E < M
        t_exh = packed.t_start[rows, np.where(hit_E, k_E, 0)]

        # Only overrides when it happens strictly earlier than the current stop.
        earlier = hit_E & (t_exh < stop_ms)
        stop_ms = np.where(earlier, t_exh, stop_ms)

        # The frozen LLR at that moment is llr_start of the exhausted interval.
        llr_frozen = packed.llr_start[rows, np.where(hit_E, k_E, 0)]
        preds = np.where(earlier, np.where(llr_frozen >= b, 0, 1), preds).astype(int)

    return stop_ms * 1000.0, preds


# =============================================================================
# 6. Metrics, regime reporting, calibration, paired statistics
# =============================================================================


def balanced_fidelity(labels: np.ndarray, preds: np.ndarray) -> float:
    """F_C = 1/2 [P(correct | NV-) + P(correct | NV0)]."""
    return nv.balanced_initial_state_fidelity(labels, preds)


def balanced_mean_time(labels: np.ndarray, times_us: np.ndarray) -> float:
    """
    Class-balanced mean run time, matching the balanced fidelity convention.

    Using the balanced mean avoids letting the (heavily NV0-weighted)
    stationary distribution dominate the reported run time.
    """
    labels = np.asarray(labels, dtype=int)
    times_us = np.asarray(times_us, dtype=float)
    return 0.5 * (
        float(times_us[labels == 0].mean())
        + float(times_us[labels == 1].mean())
    )


def regime_summary(params: MMPPParams, t_R_us: float | None = None) -> dict:
    """
    Dimensionless description of the operating point.

    These are the quantities that actually control whether event-time
    filtering can beat a count threshold, and they are portable across
    samples and setups in a way that laser power in uW is not.

        photons_per_bright_dwell = lambda_- / Gamma_-0
            The sparsity parameter. As it falls below ~1 the total count
            becomes an almost sufficient statistic and the MMPP advantage
            vanishes.

        gamma_tot_t_R
            Switching events per readout window. The MMPP advantage requires
            this to be of order 1; otherwise no blink occurs during readout.

        contrast = (lambda_- - lambda_0) / (lambda_- + lambda_0)
    """
    params.validate()

    gamma_tot = (
        params.gamma_minus_to_zero_khz + params.gamma_zero_to_minus_khz
    )

    out = {
        "photons_per_bright_dwell": (
            params.lambda_minus_khz / params.gamma_minus_to_zero_khz
            if params.gamma_minus_to_zero_khz > 0
            else np.inf
        ),
        "photons_per_dark_dwell": (
            params.lambda_zero_khz / params.gamma_zero_to_minus_khz
            if params.gamma_zero_to_minus_khz > 0
            else np.inf
        ),
        "contrast": (
            (params.lambda_minus_khz - params.lambda_zero_khz)
            / (params.lambda_minus_khz + params.lambda_zero_khz)
        ),
        "p_bright_stationary": (
            params.gamma_zero_to_minus_khz / gamma_tot
            if gamma_tot > 0
            else np.nan
        ),
        "gamma_tot_khz": gamma_tot,
        "mean_bright_dwell_us": (
            1000.0 / params.gamma_minus_to_zero_khz
            if params.gamma_minus_to_zero_khz > 0
            else np.inf
        ),
    }

    if t_R_us is not None:
        out["gamma_tot_t_R"] = gamma_tot * float(t_R_us) / 1000.0

    return out


def llr_at_times(packed: PaddedRecords, times_us: np.ndarray) -> np.ndarray:
    """
    Exact LLR at arbitrary report times, for the fixed-time MMPP baseline.

    Returns shape (n_shots, n_times).
    """
    times_ms = np.asarray(times_us, dtype=float) / 1000.0
    spec = packed.spec
    n = packed.n_shots
    out = np.empty((n, times_ms.size))
    rows = np.arange(n)

    for j, t in enumerate(times_ms):
        # Interval containing t: last interval whose start precedes t.
        k = np.maximum((packed.t_start < t).sum(axis=1) - 1, 0)
        dt = np.maximum(t - packed.t_start[rows, k], 0.0)
        x = (
            np.ones_like(dt)
            if spec.degenerate
            else np.exp(-2.0 * spec.delta * dt)
        )
        g0 = packed.alpha[rows, k, 0] + packed.beta[rows, k, 0] * x
        g1 = packed.alpha[rows, k, 1] + packed.beta[rows, k, 1] * x
        out[:, j] = packed.llr_start[rows, k] + np.log(g0) - np.log(g1)

    return out


def optimize_scalar_cutoff(
    statistic: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """
    Choose the balanced-fidelity-optimal cutoff on a continuous statistic.

    This is the LLR analogue of `optimize_count_threshold` and removes the
    unfair hardcoded LLR >= 0 rule: the count threshold was calibrated on
    data while the MMPP cutoff was not. Under any model mismatch the
    Bayes-optimal LLR cutoff is not zero.

    Rule: statistic >= cutoff -> NV- (0).
    """
    s = np.asarray(statistic, dtype=float)
    labels = np.asarray(labels, dtype=int)

    order = np.argsort(s)
    s_sorted = s[order]
    lab_sorted = labels[order]

    n_minus = int((labels == 0).sum())
    n_zero = int((labels == 1).sum())

    if n_minus == 0 or n_zero == 0:
        raise ValueError("Both initial states must be represented.")

    # Sweep cutoff upward through the sorted statistic. Shots below the
    # cutoff are called NV0.
    zero_below = np.concatenate([[0], np.cumsum(lab_sorted == 1)])
    minus_below = np.concatenate([[0], np.cumsum(lab_sorted == 0)])

    F = 0.5 * (
        (n_minus - minus_below) / n_minus + zero_below / n_zero
    )

    # A cutoff is only REALIZABLE at a boundary between distinct statistic
    # values. In the sparse regime many shots share an identical LLR (every
    # zero-click shot has the same LLR at a given report time), and a cutoff
    # placed inside such a tie group cannot split it: the applied rule sends
    # the whole group to one side. Scoring un-realizable split points gives an
    # optimistic calibration fidelity and a cutoff that misbehaves on test
    # data, which showed up as the baseline collapsing to F = 0.5.
    realizable = np.ones(s_sorted.size + 1, dtype=bool)
    realizable[1:-1] = s_sorted[1:] > s_sorted[:-1]

    F_masked = np.where(realizable, F, -np.inf)
    best = int(np.argmax(F_masked))

    if best == 0:
        cutoff = s_sorted[0] - 1.0
    elif best == s_sorted.size:
        cutoff = s_sorted[-1] + 1.0
    else:
        cutoff = 0.5 * (s_sorted[best - 1] + s_sorted[best])

    return float(cutoff), float(F[best])


def pareto_frontier(
    times: np.ndarray,
    fidelities: np.ndarray,
) -> np.ndarray:
    """
    Indices of the (min time, max fidelity) non-dominated set, ordered by time.

    A configuration is kept only if no other configuration is both faster and
    at least as accurate.
    """
    times = np.asarray(times, dtype=float)
    fidelities = np.asarray(fidelities, dtype=float)

    order = np.argsort(times, kind="stable")
    keep = []
    best_F = -np.inf

    for i in order:
        if fidelities[i] > best_F + 1e-15:
            keep.append(int(i))
            best_F = fidelities[i]

    return np.asarray(keep, dtype=int)


def min_time_for_fidelity(
    times: np.ndarray,
    fidelities: np.ndarray,
    target: float,
) -> float:
    """Smallest time among configurations reaching at least `target`."""
    mask = np.asarray(fidelities, dtype=float) >= float(target)
    if not np.any(mask):
        return np.nan
    return float(np.min(np.asarray(times, dtype=float)[mask]))


def time_for_fidelity_interp(
    times: np.ndarray,
    fidelities: np.ndarray,
    target: float,
) -> float:
    """
    Time at which a method first reaches `target`, interpolated.

    `min_time_for_fidelity` returns the smallest time on a discrete grid, so
    the numerator and denominator of a speedup each snap to their own grid and
    the ratio develops a sawtooth that is purely an artifact of grid spacing.
    Interpolating along each method's own achievable envelope removes that
    artifact symmetrically.

    The envelope is the running maximum of fidelity against increasing time,
    which is the right object for both methods: the fixed-time threshold curve
    is non-monotone (fidelity falls once the charge state has scrambled), and
    the running maximum encodes "the best fidelity reachable using any window
    no longer than t". Interpolation is linear in log time, matching the
    geometric time grids.
    """
    times = np.asarray(times, dtype=float)
    fidelities = np.asarray(fidelities, dtype=float)

    good = np.isfinite(times) & np.isfinite(fidelities) & (times > 0)
    if not np.any(good):
        return np.nan

    t = times[good]
    F = fidelities[good]

    order = np.argsort(t, kind="stable")
    t = t[order]
    F = F[order]

    envelope = np.maximum.accumulate(F)
    target = float(target)

    if target > envelope[-1]:
        return np.nan

    k = int(np.searchsorted(envelope, target, side="left"))

    if k == 0:
        return float(t[0])

    F0, F1 = envelope[k - 1], envelope[k]

    if F1 <= F0:
        return float(t[k])

    w = (target - F0) / (F1 - F0)
    return float(np.exp(np.log(t[k - 1]) + w * (np.log(t[k]) - np.log(t[k - 1]))))


def mcnemar_exact(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> dict:
    """
    Exact two-sided McNemar test on paired correctness.

    Both methods see the same shots, so the correct comparison conditions on
    the discordant pairs only.
    """
    from scipy.stats import binomtest

    labels = np.asarray(labels, dtype=int)
    ca = np.asarray(preds_a, dtype=int) == labels
    cb = np.asarray(preds_b, dtype=int) == labels

    n01 = int(np.sum(~ca & cb))   # b right, a wrong
    n10 = int(np.sum(ca & ~cb))   # a right, b wrong

    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0}

    p = binomtest(n10, n01 + n10, 0.5, alternative="two-sided").pvalue
    return {"n01": n01, "n10": n10, "p_value": float(p)}


def stratified_bootstrap_indices(
    labels: np.ndarray,
    n_boot: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Resample within each initial state separately.

    Balanced fidelity conditions on the true state, so the resampling must
    preserve the per-class sample sizes.
    """
    labels = np.asarray(labels, dtype=int)
    idx0 = np.nonzero(labels == 0)[0]
    idx1 = np.nonzero(labels == 1)[0]

    rng = np.random.default_rng(seed)

    return [
        (
            rng.choice(idx0, size=idx0.size, replace=True),
            rng.choice(idx1, size=idx1.size, replace=True),
        )
        for _ in range(int(n_boot))
    ]


# =============================================================================
# 7. The speedup study
# =============================================================================


def click_time_matrix(packed: PaddedRecords) -> np.ndarray:
    """Padded photon arrival times in ms; +inf beyond a shot's click count."""
    M = packed.t_end.shape[1]
    is_click = np.arange(M)[None, :] < packed.n_clicks[:, None]
    return np.where(is_click, packed.t_end, np.inf)


def run_adaptive_count(
    click_times_ms: np.ndarray,
    n_up: int,
    deadline_us: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sequential COUNT test: stop as soon as n_up photons have arrived.

    This is the adaptive version of the standard threshold method. Including
    it separates two distinct sources of gain:

        fixed threshold -> adaptive count   : the value of adaptive stopping
        adaptive count  -> adaptive MMPP    : the value of the event-time
                                              statistic on top of adaptivity

    Without this control, any speedup could be attributed to either.
    """
    n_up = int(n_up)
    deadline_ms = float(deadline_us) / 1000.0

    if n_up < 1:
        raise ValueError("n_up must be >= 1.")

    if n_up > click_times_ms.shape[1]:
        t_hit = np.full(click_times_ms.shape[0], np.inf)
    else:
        t_hit = click_times_ms[:, n_up - 1]

    reached = t_hit <= deadline_ms
    times_us = np.where(reached, t_hit, deadline_ms) * 1000.0
    preds = np.where(reached, 0, 1).astype(int)

    return times_us, preds


def _method_curve(
    labels: np.ndarray,
    correct: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced fidelity and balanced mean time per configuration column."""
    labels = np.asarray(labels, dtype=int)
    m0 = labels == 0
    m1 = labels == 1
    F = 0.5 * (correct[m0].mean(axis=0) + correct[m1].mean(axis=0))
    T = 0.5 * (times[m0].mean(axis=0) + times[m1].mean(axis=0))
    return F, T


def run_speedup_study(
    power_uw: float,
    detection_efficiency: float = 1.0,
    efficiency_model: str = "signal_only",
    horizon_us: float | None = None,
    n_readout_times: int = 40,
    n_calibration_shots_per_state: int = 1500,
    n_test_shots_per_state: int = 3000,
    boundary_widths: Sequence[float] | None = None,
    offsets: Sequence[float] | None = None,
    n_deadlines: int = 7,
    target_fidelities: Sequence[float] = (0.70, 0.75, 0.80, 0.85, 0.90),
    n_boot: int = 400,
    seed: int = 20260916,
    verbose: bool = True,
) -> dict:
    """
    Measure the run-time reduction of adaptive MMPP readout at MATCHED fidelity.

    Four methods are compared on identical shots:

        1. fixed-time count threshold   (the standard method; baseline)
        2. fixed-time MMPP              (better statistic, calibrated cutoff)
        3. adaptive count SPRT          (adaptive stopping, count statistic)
        4. adaptive MMPP SPRT           (both)

    Reported in two ways:

        test frontier  - both methods optimized on the test set. Symmetric,
                         and directly comparable to published figures where
                         t_R and the count threshold are both tuned.
        out-of-sample  - configuration chosen on calibration data, evaluated
                         on test data. This is what an experiment can do.

    Uncertainties use a stratified paired bootstrap over shots.
    """

    t0 = time.time()

    params = nv.apply_detection_efficiency(
        nv.shields_2015_params(power_uw),
        detection_efficiency,
        model=efficiency_model,
    )

    reg = regime_summary(params)

    if horizon_us is None:
        horizon_us = 6000.0 / reg["gamma_tot_khz"]

    horizon_us = float(horizon_us)

    if verbose:
        print(f"\n{'='*74}")
        print(
            f"P = {power_uw} uW, eta = {detection_efficiency} "
            f"({efficiency_model}), horizon = {horizon_us:.1f} us"
        )
        print(
            f"  lambda_- = {params.lambda_minus_khz:.2f} kHz, "
            f"lambda_0 = {params.lambda_zero_khz:.3f} kHz"
        )
        print(
            f"  Gamma_-0 = {params.gamma_minus_to_zero_khz:.3f} kHz, "
            f"Gamma_0- = {params.gamma_zero_to_minus_khz:.3f} kHz"
        )
        print(
            f"  photons/bright dwell = "
            f"{reg['photons_per_bright_dwell']:.2f}, "
            f"contrast = {reg['contrast']:.3f}, "
            f"p_bright = {reg['p_bright_stationary']:.3f}"
        )
        print(f"{'='*74}")

    # ---- data -------------------------------------------------------------
    cal_shots, cal_labels = nv.simulate_balanced_dataset(
        n_calibration_shots_per_state, horizon_us / 1000.0, params, seed
    )
    test_shots, test_labels = nv.simulate_balanced_dataset(
        n_test_shots_per_state, horizon_us / 1000.0, params, seed + 7717
    )

    spec = build_no_click_spectral(params)
    cal_packed = pack_records(build_records(cal_shots, horizon_us, params), spec)
    test_packed = pack_records(
        build_records(test_shots, horizon_us, params), spec
    )

    if verbose:
        print(f"records built in {time.time() - t0:.1f} s")

    readout_times_us = np.geomspace(
        horizon_us / 200.0, horizon_us, int(n_readout_times)
    )

    # ---- 1. fixed-time count threshold ------------------------------------
    cal_counts = nv.total_counts_at_times(cal_shots, readout_times_us / 1000.0)
    test_counts = nv.total_counts_at_times(
        test_shots, readout_times_us / 1000.0
    )

    n_t = readout_times_us.size
    thr_correct = np.empty((len(test_shots), n_t), dtype=float)
    thr_correct_cal = np.empty((len(cal_shots), n_t), dtype=float)

    for j in range(n_t):
        n_th, _ = nv.optimize_count_threshold(cal_counts[:, j], cal_labels)
        thr_correct[:, j] = (
            np.where(test_counts[:, j] >= n_th, 0, 1) == test_labels
        )
        thr_correct_cal[:, j] = (
            np.where(cal_counts[:, j] >= n_th, 0, 1) == cal_labels
        )

    thr_times = np.tile(readout_times_us, (len(test_shots), 1))
    F_thr, T_thr = _method_curve(test_labels, thr_correct, thr_times)
    F_thr_cal, _ = _method_curve(
        cal_labels, thr_correct_cal, np.tile(readout_times_us, (len(cal_shots), 1))
    )

    # ---- 2. fixed-time MMPP with CALIBRATED cutoff -------------------------
    cal_llr = llr_at_times(cal_packed, readout_times_us)
    test_llr = llr_at_times(test_packed, readout_times_us)

    mmpp_correct = np.empty((len(test_shots), n_t), dtype=float)
    mmpp_cutoffs = np.empty(n_t)

    for j in range(n_t):
        cut, _ = optimize_scalar_cutoff(cal_llr[:, j], cal_labels)
        mmpp_cutoffs[j] = cut
        mmpp_correct[:, j] = (
            np.where(test_llr[:, j] >= cut, 0, 1) == test_labels
        )

    F_mmpp, T_mmpp = _method_curve(test_labels, mmpp_correct, thr_times)

    # ---- configuration grid for the adaptive methods ----------------------
    if boundary_widths is None:
        boundary_widths = np.geomspace(0.15, 12.0, 14)
    if offsets is None:
        offsets = np.array([-1.2, -0.6, -0.3, 0.0, 0.3, 0.6])

    boundary_widths = np.asarray(boundary_widths, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    deadlines_us = np.geomspace(
        horizon_us / 40.0, horizon_us, int(n_deadlines)
    )

    # ---- 3. adaptive count SPRT -------------------------------------------
    cal_clicks = click_time_matrix(cal_packed)
    test_clicks = click_time_matrix(test_packed)

    max_n_up = int(
        min(
            40,
            max(3, np.percentile(test_packed.n_clicks, 99) + 1),
        )
    )
    cnt_cfg = [
        (int(n_up), float(dl))
        for dl in deadlines_us
        for n_up in range(1, max_n_up + 1)
    ]

    cnt_cal_F, cnt_cal_T = [], []
    cnt_correct, cnt_time = [], []

    for n_up, dl in cnt_cfg:
        tc, pc = run_adaptive_count(cal_clicks, n_up, dl)
        cnt_cal_F.append(balanced_fidelity(cal_labels, pc))
        cnt_cal_T.append(balanced_mean_time(cal_labels, tc))

        tt, pt = run_adaptive_count(test_clicks, n_up, dl)
        cnt_correct.append((pt == test_labels).astype(float))
        cnt_time.append(tt)

    cnt_correct = np.column_stack(cnt_correct)
    cnt_time = np.column_stack(cnt_time)
    F_cnt, T_cnt = _method_curve(test_labels, cnt_correct, cnt_time)
    cnt_cal_F = np.asarray(cnt_cal_F)
    cnt_cal_T = np.asarray(cnt_cal_T)

    # ---- 4. adaptive MMPP SPRT --------------------------------------------
    # Only (L, offset) pairs whose boundaries bracket LLR(0) = 0 are
    # meaningful; |offset| >= L decides the shot before any data arrives.
    ada_cfg = [
        (float(L), float(b), float(dl))
        for dl in deadlines_us
        for b in offsets
        for L in boundary_widths
        if abs(float(b)) < float(L)
    ]

    ada_cal_F, ada_cal_T = [], []
    ada_correct, ada_time = [], []

    for L, b, dl in ada_cfg:
        tc, pc = run_sprt(cal_packed, L, b, dl)
        ada_cal_F.append(balanced_fidelity(cal_labels, pc))
        ada_cal_T.append(balanced_mean_time(cal_labels, tc))

        tt, pt = run_sprt(test_packed, L, b, dl)
        ada_correct.append((pt == test_labels).astype(float))
        ada_time.append(tt)

    ada_correct = np.column_stack(ada_correct)
    ada_time = np.column_stack(ada_time)
    F_ada, T_ada = _method_curve(test_labels, ada_correct, ada_time)
    ada_cal_F = np.asarray(ada_cal_F)
    ada_cal_T = np.asarray(ada_cal_T)

    if verbose:
        print(
            f"scanned {len(ada_cfg)} MMPP and {len(cnt_cfg)} count "
            f"configurations in {time.time() - t0:.1f} s"
        )

    # ---- speedups at matched fidelity -------------------------------------
    targets = np.asarray(target_fidelities, dtype=float)

    # The point estimate and the bootstrap MUST range over the same set of
    # configurations, otherwise they estimate different quantities and the
    # interval does not bracket the estimate. An earlier version pruned the
    # bootstrap to the calibration-frontier subset for speed while leaving the
    # point estimate over all configurations; that mismatch put the point
    # estimate outside its own 95% interval in 26% of rows.
    #
    # All configurations are used for both. This is also the symmetric choice:
    # the threshold baseline has its readout time optimized on the test set
    # too, so neither method is handicapped. Pruning to the full-sample test
    # frontier and then resampling within it would instead bias the interval
    # optimistically, because the subset was already selected using the same
    # data.
    keep_ada = np.arange(ada_correct.shape[1])
    keep_cnt = np.arange(cnt_correct.shape[1])

    boots = stratified_bootstrap_indices(test_labels, n_boot, seed + 991)

    rows = []
    for F_star in targets:
        t_thr = time_for_fidelity_interp(T_thr, F_thr, F_star)
        t_cnt = time_for_fidelity_interp(T_cnt, F_cnt, F_star)
        t_ada = time_for_fidelity_interp(T_ada, F_ada, F_star)

        # out-of-sample: pick on calibration, measure on test
        def _oos(cal_F, cal_T, test_F_col, test_T_col):
            mask = cal_F >= F_star
            if not np.any(mask):
                return np.nan, np.nan
            k = int(np.nonzero(mask)[0][np.argmin(cal_T[mask])])
            return float(test_F_col[k]), float(test_T_col[k])

        oos_thr_F, oos_thr_T = _oos(F_thr_cal, readout_times_us, F_thr, T_thr)
        oos_ada_F, oos_ada_T = _oos(ada_cal_F, ada_cal_T, F_ada, T_ada)

        # paired bootstrap on the speedup
        sp = []
        for i0, i1 in boots:
            Fb_thr = 0.5 * (
                thr_correct[i0].mean(axis=0) + thr_correct[i1].mean(axis=0)
            )
            Fb_ada = 0.5 * (
                ada_correct[np.ix_(i0, keep_ada)].mean(axis=0)
                + ada_correct[np.ix_(i1, keep_ada)].mean(axis=0)
            )
            Tb_ada = 0.5 * (
                ada_time[np.ix_(i0, keep_ada)].mean(axis=0)
                + ada_time[np.ix_(i1, keep_ada)].mean(axis=0)
            )
            a = time_for_fidelity_interp(readout_times_us, Fb_thr, F_star)
            c = time_for_fidelity_interp(Tb_ada, Fb_ada, F_star)
            if np.isfinite(a) and np.isfinite(c) and c > 0:
                sp.append(a / c)

        sp = np.asarray(sp)
        lo, hi = (
            (np.nanpercentile(sp, 2.5), np.nanpercentile(sp, 97.5))
            if sp.size > 20
            else (np.nan, np.nan)
        )

        rows.append(
            {
                "target_fidelity": float(F_star),
                "t_threshold_us": t_thr,
                "t_adaptive_count_us": t_cnt,
                "t_adaptive_mmpp_us": t_ada,
                "speedup_vs_threshold": (
                    t_thr / t_ada
                    if np.isfinite(t_thr) and np.isfinite(t_ada) and t_ada > 0
                    else np.nan
                ),
                "speedup_ci_low": float(lo),
                "speedup_ci_high": float(hi),
                "speedup_from_adaptivity_only": (
                    t_thr / t_cnt
                    if np.isfinite(t_thr) and np.isfinite(t_cnt) and t_cnt > 0
                    else np.nan
                ),
                "oos_threshold_t_us": oos_thr_T,
                "oos_threshold_F": oos_thr_F,
                "oos_adaptive_t_us": oos_ada_T,
                "oos_adaptive_F": oos_ada_F,
                "oos_speedup": (
                    oos_thr_T / oos_ada_T
                    if np.isfinite(oos_thr_T)
                    and np.isfinite(oos_ada_T)
                    and oos_ada_T > 0
                    else np.nan
                ),
            }
        )

    if verbose:
        print(
            f"\n{'F*':>6} {'t_thr':>9} {'t_cnt':>9} {'t_mmpp':>9} "
            f"{'speedup':>9} {'95% CI':>18} {'adapt-only':>11}"
        )
        for r in rows:
            ci = (
                f"[{r['speedup_ci_low']:.2f}, {r['speedup_ci_high']:.2f}]"
                if np.isfinite(r["speedup_ci_low"])
                else "n/a"
            )
            print(
                f"{r['target_fidelity']:6.3f} "
                f"{r['t_threshold_us']:9.2f} "
                f"{r['t_adaptive_count_us']:9.2f} "
                f"{r['t_adaptive_mmpp_us']:9.2f} "
                f"{r['speedup_vs_threshold']:9.2f} "
                f"{ci:>18} "
                f"{r['speedup_from_adaptivity_only']:11.2f}"
            )
        print(f"\ntotal {time.time() - t0:.1f} s")

    return {
        "params": params,
        "regime": reg,
        "power_uw": float(power_uw),
        "detection_efficiency": float(detection_efficiency),
        "efficiency_model": efficiency_model,
        "horizon_us": horizon_us,
        "readout_times_us": readout_times_us,
        "F_threshold": F_thr,
        "T_threshold": T_thr,
        "F_threshold_cal": F_thr_cal,
        "F_fixed_mmpp": F_mmpp,
        "T_fixed_mmpp": T_mmpp,
        "mmpp_cutoffs": mmpp_cutoffs,
        "adaptive_count_configs": cnt_cfg,
        "F_adaptive_count": F_cnt,
        "T_adaptive_count": T_cnt,
        "adaptive_mmpp_configs": ada_cfg,
        "F_adaptive_mmpp": F_ada,
        "T_adaptive_mmpp": T_ada,
        "F_adaptive_mmpp_cal": ada_cal_F,
        "T_adaptive_mmpp_cal": ada_cal_T,
        "frontier_adaptive_mmpp": pareto_frontier(T_ada, F_ada),
        "frontier_adaptive_count": pareto_frontier(T_cnt, F_cnt),
        "speedup_table": rows,
        "test_labels": test_labels,
        "adaptive_correct": ada_correct,
        "adaptive_time": ada_time,
        "threshold_correct": thr_correct,
        "fixed_mmpp_correct": mmpp_correct,
        "module_version": MODULE_VERSION,
    }


# =============================================================================
# 8. Polyspectra coupling: correct posterior coordinates and covariance
# =============================================================================


def polyspectra_laplace_covariance(
    least_squares_result,
    clip_condition: float = 1e12,
) -> np.ndarray:
    """
    Single-dataset Laplace covariance from a polyspectra least_squares fit.

    The polyspectra script currently estimates rate uncertainties from the
    scatter across random seeds, which requires repeating the whole
    experiment and is therefore unavailable in the lab. With error-weighted
    residuals r_i = (y_i - model_i)/sigma_i, the Gauss-Newton covariance at
    the optimum is

        Cov ~ (J^T J)^-1

    in FIT coordinates, i.e. (log K, logit p_bright, log beta^2). This is
    obtainable from a single dataset.

    Caveat to check, not assume: polyspectral estimates at neighbouring
    frequencies are correlated, so the independent-residual assumption makes
    this an UNDERestimate of the true sampling covariance. Validate it
    against the across-seed scatter you already compute before trusting it
    online. `compare_laplace_to_seed_scatter` does exactly that.
    """
    J = np.asarray(least_squares_result.jac, dtype=float)
    JTJ = J.T @ J

    u, s, vt = np.linalg.svd(JTJ)
    s_max = float(s.max()) if s.size else 0.0
    floor = s_max / float(clip_condition)
    s_inv = np.where(s > floor, 1.0 / np.maximum(s, floor), 0.0)

    return (vt.T * s_inv) @ u.T


def compare_laplace_to_seed_scatter(
    laplace_cov: np.ndarray,
    seed_samples: np.ndarray,
) -> dict:
    """
    Validate the single-dataset Laplace covariance against across-seed scatter.

    `seed_samples` has shape (n_seeds, n_params) in the SAME fit coordinates
    as `laplace_cov`. A ratio near 1 licenses using the Laplace covariance
    online; a ratio well below 1 means the residual correlations matter and
    the covariance should be inflated by the measured factor.
    """
    seed_samples = np.asarray(seed_samples, dtype=float)
    empirical = np.cov(seed_samples, rowvar=False)

    laplace_sd = np.sqrt(np.diag(np.atleast_2d(laplace_cov)))
    empirical_sd = np.sqrt(np.diag(np.atleast_2d(empirical)))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = laplace_sd / empirical_sd

    return {
        "laplace_sd": laplace_sd,
        "empirical_sd": empirical_sd,
        "sd_ratio": ratio,
        "inflation_factor": np.where(ratio > 0, 1.0 / ratio, np.nan),
        "empirical_cov": empirical,
    }


def _fit_coords_to_rates(log_k: float, logit_p: float) -> tuple[float, float]:
    """
    (log K, logit p_bright) -> (Gamma_-0, Gamma_0-).

    Matches `_fit_coordinates_to_rates` in the polyspectra script:
        K = Gamma_-0 + Gamma_0-,  p_bright = Gamma_0- / K.
    """
    k_total = float(np.exp(log_k))
    p = 1.0 / (1.0 + np.exp(-logit_p)) if logit_p >= 0 else (
        np.exp(logit_p) / (1.0 + np.exp(logit_p))
    )
    gamma_zero_to_minus = p * k_total
    gamma_minus_to_zero = (1.0 - p) * k_total
    return gamma_minus_to_zero, gamma_zero_to_minus


def rates_to_fit_coords(
    gamma_minus_to_zero_khz: float,
    gamma_zero_to_minus_khz: float,
) -> tuple[float, float]:
    k = float(gamma_minus_to_zero_khz + gamma_zero_to_minus_khz)
    p = float(np.clip(gamma_zero_to_minus_khz / k, 1e-9, 1.0 - 1e-9))
    return float(np.log(k)), float(np.log(p / (1.0 - p)))


def make_correlated_switching_ensemble(
    nominal_params: MMPPParams,
    cov_fit_coords: np.ndarray,
    nodes_per_dimension: int = 3,
) -> tuple[list[MMPPParams], np.ndarray]:
    """
    Gauss-Hermite ensemble built in POLYSPECTRA fit coordinates.

    The original `make_bayesian_switching_ensemble` used a product of
    INDEPENDENT log-normals on Gamma_-0 and Gamma_0-. That is inconsistent
    with how the rates are actually measured: the polyspectra fit is
    parameterized in (log K, logit p_bright) precisely because the likelihood
    is near-diagonal there, which means the posterior is strongly CORRELATED
    in (Gamma_-0, Gamma_0-). An independent product therefore puts quadrature
    weight along the directions polyspectra constrains worst and starves the
    directions it constrains best.

    Building the product rule in (log K, logit p_bright) and mapping through
    to rates fixes this, and the ensemble inherits the fit's correlation
    structure automatically via the Cholesky factor of `cov_fit_coords`.

    Parameters
    ----------
    cov_fit_coords
        2x2 posterior covariance for (log K, logit p_bright), e.g. the top
        block of `polyspectra_laplace_covariance`.
    """
    nominal_params.validate()

    C = np.atleast_2d(np.asarray(cov_fit_coords, dtype=float))[:2, :2]
    n = int(nodes_per_dimension)

    if n < 1:
        raise ValueError("nodes_per_dimension must be >= 1.")

    log_k0, logit_p0 = rates_to_fit_coords(
        nominal_params.gamma_minus_to_zero_khz,
        nominal_params.gamma_zero_to_minus_khz,
    )

    if n == 1 or not np.any(np.diag(C) > 0):
        return [nominal_params], np.array([1.0])

    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(n)
    z = np.sqrt(2.0) * gh_nodes
    w = gh_weights / np.sqrt(np.pi)

    # Symmetrize and make positive definite before factoring.
    C = 0.5 * (C + C.T)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 0.0, None)
    Lc = evecs @ np.diag(np.sqrt(evals))

    ensemble: list[MMPPParams] = []
    weights: list[float] = []

    for i, zi in enumerate(z):
        for j, zj in enumerate(z):
            d = Lc @ np.array([zi, zj])
            g_m0, g_0m = _fit_coords_to_rates(
                log_k0 + d[0], logit_p0 + d[1]
            )
            try:
                p = MMPPParams(
                    gamma_minus_to_zero_khz=g_m0,
                    gamma_zero_to_minus_khz=g_0m,
                    lambda_minus_khz=nominal_params.lambda_minus_khz,
                    lambda_zero_khz=nominal_params.lambda_zero_khz,
                )
                p.validate()
            except ValueError:
                continue
            ensemble.append(p)
            weights.append(float(w[i] * w[j]))

    weights_arr = np.asarray(weights, dtype=float)
    weights_arr /= weights_arr.sum()

    return ensemble, weights_arr


def sample_switching_posterior(
    nominal_params: MMPPParams,
    cov_fit_coords: np.ndarray,
    n_samples: int,
    seed: int = 0,
) -> list[MMPPParams]:
    """Monte Carlo draws from the posterior, in fit coordinates."""
    C = np.atleast_2d(np.asarray(cov_fit_coords, dtype=float))[:2, :2]
    C = 0.5 * (C + C.T)

    log_k0, logit_p0 = rates_to_fit_coords(
        nominal_params.gamma_minus_to_zero_khz,
        nominal_params.gamma_zero_to_minus_khz,
    )

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(
        mean=[log_k0, logit_p0], cov=C, size=int(n_samples)
    )

    out = []
    for log_k, logit_p in draws:
        g_m0, g_0m = _fit_coords_to_rates(log_k, logit_p)
        try:
            p = MMPPParams(
                gamma_minus_to_zero_khz=g_m0,
                gamma_zero_to_minus_khz=g_0m,
                lambda_minus_khz=nominal_params.lambda_minus_khz,
                lambda_zero_khz=nominal_params.lambda_zero_khz,
            )
            p.validate()
            out.append(p)
        except ValueError:
            continue

    return out


def run_parameter_robustness(
    result: dict,
    rate_cv: float = 0.30,
    n_draws: int = 12,
    target_fidelity: float = 0.85,
    seed: int = 4242,
    verbose: bool = True,
) -> dict:
    """
    Does the speedup survive realistic polyspectra parameter error?

    Data are generated from the TRUE parameters. The adaptive filter is then
    built from parameters DRAWN from a posterior of the stated width in
    (log K, logit p_bright) coordinates, so the sampled error has the
    correlation structure a polyspectra fit actually produces. The count
    threshold does not use the switching rates at all, so its curve is the
    fixed reference.

    `rate_cv` is the fractional 1-sigma width on K and on p_bright/(1-p).
    """
    params = result["params"]
    horizon_us = result["horizon_us"]
    labels = result["test_labels"]

    # Diagonal covariance of the stated width in fit coordinates.
    sd = float(np.log1p(rate_cv))
    C = np.diag([sd**2, sd**2])

    draws = sample_switching_posterior(params, C, n_draws, seed=seed)

    # Regenerate the same test shots from the true parameters.
    n_per_state = int(len(labels) // 2)
    shots, lab = nv.simulate_balanced_dataset(
        n_per_state, horizon_us / 1000.0, params, result.get("seed", 0) + 7717
    )

    t_thr = min_time_for_fidelity(
        result["T_threshold"], result["F_threshold"], target_fidelity
    )

    widths = np.geomspace(0.15, 12.0, 14)
    offsets = np.array([-1.2, -0.6, -0.3, 0.0, 0.3, 0.6])
    deadlines = np.geomspace(horizon_us / 40.0, horizon_us, 7)

    speedups, fidelities = [], []

    for p_used in draws:
        spec = build_no_click_spectral(p_used)
        packed = pack_records(build_records(shots, horizon_us, p_used), spec)

        F_list, T_list = [], []
        for dl in deadlines:
            for b in offsets:
                for L in widths:
                    if abs(float(b)) >= float(L):
                        continue
                    t, pr = run_sprt(packed, L, b, dl)
                    F_list.append(balanced_fidelity(lab, pr))
                    T_list.append(balanced_mean_time(lab, t))

        F_arr = np.asarray(F_list)
        T_arr = np.asarray(T_list)

        t_ada = min_time_for_fidelity(T_arr, F_arr, target_fidelity)
        fidelities.append(float(F_arr.max()))
        speedups.append(
            t_thr / t_ada if np.isfinite(t_ada) and t_ada > 0 else np.nan
        )

    speedups = np.asarray(speedups, dtype=float)

    out = {
        "rate_cv": rate_cv,
        "target_fidelity": target_fidelity,
        "t_threshold_us": t_thr,
        "speedups": speedups,
        "speedup_median": float(np.nanmedian(speedups)),
        "speedup_min": float(np.nanmin(speedups)),
        "speedup_max": float(np.nanmax(speedups)),
        "max_fidelity_per_draw": np.asarray(fidelities),
        "n_failed": int(np.sum(~np.isfinite(speedups))),
    }

    if verbose:
        print(
            f"\nparameter robustness (CV = {rate_cv:.0%}, "
            f"F* = {target_fidelity:.3f}, {len(draws)} draws)"
        )
        print(
            f"  speedup median {out['speedup_median']:.2f}, "
            f"range [{out['speedup_min']:.2f}, {out['speedup_max']:.2f}], "
            f"draws failing to reach F*: {out['n_failed']}"
        )

    return out


# =============================================================================
# 9. Overhead-aware speedup and plots
# =============================================================================


def speedup_with_overhead(
    result: dict,
    overhead_us: float,
    target_fidelities: Sequence[float] | None = None,
) -> list[dict]:
    """
    Recompute the speedup including a fixed per-shot overhead.

    Adaptive readout shortens the measurement window only. If each shot also
    carries initialization, spin manipulation and reset time, the achievable
    end-to-end speedup is diluted:

        speedup = (t_R + overhead) / (T_adaptive + overhead)

    Reporting this alongside the bare speedup is the honest way to state what
    an experiment would actually gain.
    """
    if target_fidelities is None:
        target_fidelities = [
            r["target_fidelity"] for r in result["speedup_table"]
        ]

    oh = float(overhead_us)
    out = []

    for F_star in target_fidelities:
        t_thr = min_time_for_fidelity(
            result["T_threshold"], result["F_threshold"], F_star
        )
        t_ada = min_time_for_fidelity(
            result["T_adaptive_mmpp"], result["F_adaptive_mmpp"], F_star
        )
        out.append(
            {
                "target_fidelity": float(F_star),
                "overhead_us": oh,
                "speedup_no_overhead": (
                    t_thr / t_ada if np.isfinite(t_ada) and t_ada > 0 else np.nan
                ),
                "speedup_with_overhead": (
                    (t_thr + oh) / (t_ada + oh)
                    if np.isfinite(t_thr) and np.isfinite(t_ada)
                    else np.nan
                ),
            }
        )

    return out


def max_fidelity_summary(result: dict) -> dict:
    """Best fidelity each method can reach anywhere in its configuration set."""
    return {
        "threshold": float(np.max(result["F_threshold"])),
        "fixed_mmpp": float(np.max(result["F_fixed_mmpp"])),
        "adaptive_count": float(np.max(result["F_adaptive_count"])),
        "adaptive_mmpp": float(np.max(result["F_adaptive_mmpp"])),
    }


def plot_speedup_study(result: dict, save_path: str | None = None):
    """
    Two panels:
      (a) fidelity vs run time for all four methods, with the horizontal
          time reduction at a matched fidelity marked;
      (b) speedup vs target fidelity with bootstrap CI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax = axes[0]

    ax.plot(
        result["T_threshold"],
        result["F_threshold"],
        "o-",
        ms=3.5,
        lw=1.6,
        color="#444444",
        label="fixed-time count threshold",
    )
    ax.plot(
        result["T_fixed_mmpp"],
        result["F_fixed_mmpp"],
        "s--",
        ms=3.5,
        lw=1.4,
        color="#1f77b4",
        label="fixed-time MMPP (calibrated cutoff)",
    )

    fc = result["frontier_adaptive_count"]
    ax.plot(
        np.asarray(result["T_adaptive_count"])[fc],
        np.asarray(result["F_adaptive_count"])[fc],
        "^-",
        ms=3.5,
        lw=1.4,
        color="#2ca02c",
        label="adaptive count SPRT",
    )

    fa = result["frontier_adaptive_mmpp"]
    ax.plot(
        np.asarray(result["T_adaptive_mmpp"])[fa],
        np.asarray(result["F_adaptive_mmpp"])[fa],
        "-",
        lw=2.4,
        color="#d62728",
        label="adaptive MMPP SPRT (this work)",
    )

    # Mark the time reduction at the largest jointly achievable target.
    rows = [
        r
        for r in result["speedup_table"]
        if np.isfinite(r["speedup_vs_threshold"])
    ]
    if rows:
        r = rows[-1]
        F_star = r["target_fidelity"]
        ax.annotate(
            "",
            xy=(r["t_adaptive_mmpp_us"], F_star),
            xytext=(r["t_threshold_us"], F_star),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8),
        )
        ax.axhline(F_star, color="#d62728", lw=0.7, ls=":", alpha=0.6)
        ax.text(
            np.sqrt(r["t_adaptive_mmpp_us"] * r["t_threshold_us"]),
            F_star + 0.012,
            f"{r['speedup_vs_threshold']:.2f}x faster",
            color="#d62728",
            ha="center",
            fontsize=9.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("mean run time per shot (us)")
    ax.set_ylabel("balanced initial-state fidelity")
    reg = result["regime"]
    ax.set_title(
        f"P = {result['power_uw']} uW, eta = {result['detection_efficiency']}\n"
        f"photons/bright dwell = {reg['photons_per_bright_dwell']:.2f}, "
        f"contrast = {reg['contrast']:.2f}",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    tbl = result["speedup_table"]
    F = np.array([r["target_fidelity"] for r in tbl])
    S = np.array([r["speedup_vs_threshold"] for r in tbl])
    lo = np.array([r["speedup_ci_low"] for r in tbl])
    hi = np.array([r["speedup_ci_high"] for r in tbl])
    S_adapt = np.array([r["speedup_from_adaptivity_only"] for r in tbl])

    ok = np.isfinite(S)
    ax.plot(F[ok], S[ok], "o-", color="#d62728", label="adaptive MMPP")
    good = np.isfinite(lo) & np.isfinite(hi) & ok
    ax.fill_between(
        F[good], lo[good], hi[good], color="#d62728", alpha=0.18,
        label="95% paired bootstrap CI",
    )
    ok2 = np.isfinite(S_adapt)
    ax.plot(
        F[ok2], S_adapt[ok2], "^--", color="#2ca02c",
        label="adaptive count (adaptivity only)",
    )
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axhline(
        2.0, color="#888888", lw=0.9, ls="--",
        label="D'Anjou bound for decay readout",
    )
    ax.set_xlabel("target balanced fidelity")
    ax.set_ylabel("run-time reduction vs fixed-time threshold")
    ax.set_title("speedup at matched fidelity", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

# =============================================================================
# 10. Optimal stopping (Ludkovski-Sezer formulation, regression Monte Carlo)
# =============================================================================
#
# Optimal stopping for NV charge readout, Ludkovski-Sezer formulation
# ====================================================================
#
# Solves the finite-horizon Bayesian problem
#
#     minimise   c * E[tau]  +  a * P(say NV0 | was NV-)  +  b * P(say NV- | was NV0)
#
# which is Ludkovski & Sezer (Stochastic Models 28(2), 2012) applied to the
# AUGMENTED chain (M_0, M_t):
#
#     i = 1: (NV-, NV-)   lambda_1 = lambda_-      i = 3: (NV0, NV-)   lambda_3 = lambda_-
#     i = 2: (NV-, NV0)   lambda_2 = lambda_0      i = 4: (NV0, NV0)   lambda_4 = lambda_0
#
# with a block-diagonal generator (M_0 is frozen), initial distribution
# (p, 0, 0, 1-p) supported on the diagonal, action set {declare NV-, declare
# NV0}, terminal payoff depending only on the M_0 coordinate, running cost
# c_i = -c and no discounting.
#
# Why regression Monte Carlo rather than their grid scheme
# --------------------------------------------------------
# Their conditional process Pi lives on the 3-simplex, so a direct value
# iteration needs a 4-D table over (time, Pi). In the coordinates that matter
# here the state is
#
#     l = log-likelihood ratio of the INITIAL state   (drives the payoff)
#     u = P(bright now | started bright, data)        )  drive the future
#     v = P(bright now | started dark,   data)        )  evolution of l
#
# and (u, v) is genuinely two-dimensional: both evolve under the same
# autonomous flow but from different initial conditions, and knowing one does
# not determine the other. A 4-D grid is slow and brittle; regression Monte
# Carlo handles the dimension, reuses the existing simulator, and -- because
# any stopping rule is feasible -- always returns a valid LOWER bound on the
# value, so it degrades gracefully rather than silently.
#
# Structure of the dynamics, used throughout
# ------------------------------------------
#     dl/dt      = -Delta_lambda (u - v)                      between clicks
#     jump in l  = log[(lam0 + Dlam u)/(lam0 + Dlam v)]        at a click
#
# so both are governed by the separation of u and v. In log-odds coordinates
# y = logit(u), z = logit(v) a click is a pure translation of BOTH by
# log(lambda_-/lambda_0), leaving d = y - z exactly unchanged, while the
# no-click flow strictly contracts d. Information about the initial state is
# destroyed only by waiting. d is therefore the natural second feature for the
# regression, and d = 0 is an absorbing set on which l is frozen.


@dataclass
class FilterPaths:
    """
    Filter state for many shots, on a common uniform time grid.

    llr  (n_paths, n_steps+1)  initial-state log-likelihood ratio
    y, z (n_paths, n_steps+1)  log-odds of u and v
    gap  (n_paths, n_steps+1)  d = y - z, the remaining information
    """

    t_us: np.ndarray
    llr: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gap: np.ndarray
    labels: np.ndarray
    params: MMPPParams

    @property
    def n_paths(self) -> int:
        return int(self.llr.shape[0])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-300, 1.0 - 1e-16)
    return np.log(p) - np.log1p(-p)


def filter_on_grid(
    shots: Sequence[np.ndarray],
    labels: np.ndarray,
    params: MMPPParams,
    horizon_us: float,
    dt_us: float,
) -> FilterPaths:
    """
    EXACT filter state at a set of decision epochs, vectorised across shots.

    An earlier version stepped a uniform grid and snapped clicks to step
    edges. That is far too crude here: measured against the exact event-time
    LLR it was off by 1.43 at dt = 1 us and still 0.89 at dt = 0.5 us, against
    an LLR range of only a few units. The jump at a click is
    log[(l0+Dl u)/(l0+Dl v)], which is large and depends on exactly when
    within the step the photon arrived.

    This version instead uses the exact event-time records and evaluates them
    at the decision epochs in closed form. The interval containing epoch t is
    simply the number of clicks in [0, t), so the lookup is a cumulative sum
    rather than a search, and the whole thing stays vectorised.
    """
    params.validate()
    spec = build_no_click_spectral(params)

    n_steps = int(round(horizon_us / dt_us))
    t_us = np.arange(n_steps + 1) * dt_us
    t_ms = t_us / 1000.0

    recs = build_records(shots, horizon_us, params)
    packed = pack_records(recs, spec)
    n, M = packed.t_start.shape
    rows = np.arange(n)

    # clicks strictly before each epoch -> index of the containing interval
    edges = np.concatenate([[0.0], t_ms])
    counts = np.zeros((n, n_steps), dtype=int)
    for i, ts in enumerate(shots):
        ts = np.asarray(ts, dtype=float)
        ts = ts[(ts >= 0.0) & (ts < t_ms[-1])]
        if ts.size:
            counts[i] = np.bincount(
                np.searchsorted(t_ms, ts, side="right") - 1, minlength=n_steps
            )[:n_steps]
    idx = np.zeros((n, n_steps + 1), dtype=int)
    idx[:, 1:] = np.cumsum(counts, axis=1)
    idx = np.minimum(idx, np.maximum(packed.n_int[:, None] - 1, 0))

    llr = np.empty((n, n_steps + 1))
    y = np.empty((n, n_steps + 1))
    z = np.empty((n, n_steps + 1))

    for j in range(n_steps + 1):
        k = idx[:, j]
        dt = np.maximum(t_ms[j] - packed.t_start[rows, k], 0.0)
        x = (
            np.ones_like(dt)
            if spec.degenerate
            else np.exp(-2.0 * spec.delta * dt)
        )
        al, be = packed.alpha[rows, k], packed.beta[rows, k]
        g0 = al[:, 0] + be[:, 0] * x
        g1 = al[:, 1] + be[:, 1] * x
        llr[:, j] = packed.llr_start[rows, k] + np.log(g0) - np.log(g1)

        # hypothesis columns at the epoch, for u and v
        Hs = np.stack([r.H_start[kk] for r, kk in zip(recs, k)])
        Mx = spec.V @ (np.stack([np.ones_like(x), x], axis=1)[:, :, None]
                       * spec.V_inv[None, :, :])
        Hj = np.einsum("nij,njk->nik", Mx, Hs)
        Hj /= Hj.sum(axis=1)[:, None, :]
        y[:, j] = _logit(Hj[:, 0, 0])
        z[:, j] = _logit(Hj[:, 0, 1])

    return FilterPaths(
        t_us=t_us,
        llr=llr,
        y=y,
        z=z,
        gap=y - z,
        labels=np.asarray(labels, dtype=int),
        params=params,
    )


def check_grid_error(
    shots: Sequence[np.ndarray],
    params: MMPPParams,
    horizon_us: float,
    dt_us: float,
    n_check: int = 40,
) -> float:
    """Max |grid LLR - exact event-time LLR| at the horizon, over n_check shots."""
    fp = filter_on_grid(
        shots[:n_check], np.zeros(n_check, dtype=int), params, horizon_us, dt_us
    )
    exact = []
    for ts in shots[:n_check]:
        exact.append(
            nv.initial_state_llr_at_times(
                np.asarray(ts)[np.asarray(ts) < horizon_us / 1000.0],
                np.array([horizon_us / 1000.0]),
                params,
            )[0]
        )
    return float(np.max(np.abs(fp.llr[:, -1] - np.asarray(exact))))


# =============================================================================
# Payoff
# =============================================================================


@dataclass(frozen=True)
class Economics:
    """
    The three numbers the theory needs and the current protocol leaves implicit.

    cost_per_us   value of one microsecond of readout
    cost_miss     cost of declaring NV0 when the shot started NV-   (a)
    cost_false    cost of declaring NV- when the shot started NV0   (b)

    Only the ratios a/c and b/c matter. Sweeping a/c traces out the entire
    fidelity-versus-time frontier, so it is the knob, not a nuisance.
    """

    cost_per_us: float = 1.0 / 500.0
    cost_miss: float = 1.0
    cost_false: float = 1.0


def terminal_reward(llr: np.ndarray, econ: Economics) -> np.ndarray:
    """
    H = -min{ b(1-p_hat), a p_hat },  p_hat = sigmoid(llr) under an even prior.

    Equal to Peskir-Shiryaev's gain function g_{a,b} negated, which is the
    consistency check that the two formulations agree.
    """
    p = 1.0 / (1.0 + np.exp(-np.clip(llr, -700, 700)))
    return -np.minimum(econ.cost_false * (1.0 - p), econ.cost_miss * p)


def bayes_risk(
    stop_us: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    econ: Economics,
) -> float:
    """c E[tau] + a P(say NV0 | NV-) + b P(say NV- | NV0), class-balanced."""
    labels = np.asarray(labels, dtype=int)
    m0, m1 = labels == 0, labels == 1
    t = 0.5 * (stop_us[m0].mean() + stop_us[m1].mean())
    miss = (preds[m0] == 1).mean()
    false = (preds[m1] == 0).mean()
    return float(
        econ.cost_per_us * t + 0.5 * econ.cost_miss * miss
        + 0.5 * econ.cost_false * false
    )


# =============================================================================
# Regression Monte Carlo
# =============================================================================


def _features(llr: np.ndarray, gap: np.ndarray) -> np.ndarray:
    """
    Basis for the continuation value.

    l enters through the payoff, d through how much can still be learned.
    exp(-d) saturates at 1 when information is exhausted, which is the
    absorbing face where continuation is worthless, so it gives the
    regression a direct handle on that boundary.
    """
    e = np.exp(-np.clip(gap, 0.0, 50.0))
    return np.column_stack(
        [
            np.ones_like(llr),
            llr,
            llr * llr,
            np.abs(llr),
            gap,
            gap * gap,
            e,
            llr * e,
            llr * llr * e,
            np.abs(llr) * gap,
        ]
    )


def fit_stopping_rule(
    paths: FilterPaths,
    econ: Economics,
    ridge: float = 1e-6,
) -> list[np.ndarray]:
    """
    Backward induction with least-squares continuation values.

    At each step the realised pathwise value is carried forward and the
    regression is used only to DECIDE, which is the Longstaff-Schwartz
    convention and keeps the resulting policy feasible (hence a lower bound).

    Returns one coefficient vector per time step.
    """
    n_steps = paths.llr.shape[1] - 1
    dt_us = float(paths.t_us[1] - paths.t_us[0])
    step_cost = econ.cost_per_us * dt_us

    value = terminal_reward(paths.llr[:, -1], econ)
    coeffs: list[np.ndarray] = [np.zeros(_features(paths.llr[:, 0], paths.gap[:, 0]).shape[1])]

    for k in range(n_steps - 1, -1, -1):
        X = _features(paths.llr[:, k], paths.gap[:, k])
        target = value - step_cost                      # value of waiting one step
        A = X.T @ X + ridge * np.eye(X.shape[1])
        beta = np.linalg.solve(A, X.T @ target)
        cont = X @ beta
        stop_now = terminal_reward(paths.llr[:, k], econ)
        take = stop_now >= cont
        value = np.where(take, stop_now, target)
        coeffs.append(beta)

    coeffs.reverse()
    return coeffs[:-1] + [coeffs[-1]]


def apply_stopping_rule(
    paths: FilterPaths,
    coeffs: list[np.ndarray],
    econ: Economics,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the fitted policy forward on (independent) paths."""
    n, n_steps = paths.llr.shape[0], paths.llr.shape[1] - 1
    stopped = np.zeros(n, dtype=bool)
    stop_k = np.full(n, n_steps, dtype=int)

    for k in range(n_steps):
        live = ~stopped
        if not np.any(live):
            break
        X = _features(paths.llr[live, k], paths.gap[live, k])
        cont = X @ coeffs[k]
        stop_now = terminal_reward(paths.llr[live, k], econ)
        take = stop_now >= cont
        idx = np.nonzero(live)[0][take]
        stop_k[idx] = k
        stopped[idx] = True

    rows = np.arange(n)
    stop_us = paths.t_us[stop_k]
    preds = (paths.llr[rows, stop_k] < np.log(econ.cost_false / econ.cost_miss)).astype(int)
    return stop_us, preds


# =============================================================================
# Baseline: best constant-boundary SPRT under the same economics
# =============================================================================


def best_constant_boundary(
    paths: FilterPaths,
    econ: Economics,
    widths: Sequence[float] = tuple(np.geomspace(0.1, 12.0, 24)),
    offsets: Sequence[float] = (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0),
) -> dict:
    """
    Sweep (L, offset) for the truncated SPRT on the same grid-filter paths, so
    the comparison isolates the stopping rule rather than the filter.
    """
    n_steps = paths.llr.shape[1] - 1
    best = None
    for L in widths:
        for off in offsets:
            if abs(off) >= L:
                continue
            U, D = off + L, off - L
            out = (paths.llr >= U) | (paths.llr <= D)
            has = out.any(axis=1)
            k = np.where(has, out.argmax(axis=1), n_steps)
            rows = np.arange(paths.n_paths)
            preds = (paths.llr[rows, k] < off).astype(int)
            r = bayes_risk(paths.t_us[k], preds, paths.labels, econ)
            if best is None or r < best["risk"]:
                best = {"risk": r, "L": float(L), "offset": float(off),
                        "stop_us": paths.t_us[k], "preds": preds}
    return best


# =============================================================================
# 11. Head-to-head comparison against the original scheme, with plots
# =============================================================================


def _sprt_on_epochs(
    paths: "FilterPaths",
    L: float,
    offset: float,
    exhaustion_eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Truncated SPRT evaluated on the epoch grid, optionally with the
    information-exhaustion exit. Kept on the same grid as the learned policy
    so the comparison isolates the STOPPING RULE and not the filter.
    """
    n_steps = paths.llr.shape[1] - 1
    rows = np.arange(paths.n_paths)
    U, D = offset + L, offset - L

    out = (paths.llr >= U) | (paths.llr <= D)
    if exhaustion_eps > 0.0:
        out = out | (paths.gap <= exhaustion_eps)

    has = out.any(axis=1)
    k = np.where(has, out.argmax(axis=1), n_steps)
    preds = (paths.llr[rows, k] < offset).astype(int)
    return paths.t_us[k], preds


def best_constant_boundary_eps(
    paths: "FilterPaths",
    econ: "Economics",
    exhaustion_eps: float = 0.0,
    widths: Sequence[float] = tuple(np.geomspace(0.1, 12.0, 24)),
    offsets: Sequence[float] = (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0),
) -> dict:
    """`best_constant_boundary` with the exhaustion exit available."""
    best = None
    for L in widths:
        for off in offsets:
            if abs(off) >= L:
                continue
            st, pr = _sprt_on_epochs(paths, L, off, exhaustion_eps)
            r = bayes_risk(st, pr, paths.labels, econ)
            if best is None or r < best["risk"]:
                best = {
                    "risk": r, "L": float(L), "offset": float(off),
                    "stop_us": st, "preds": pr,
                }
    return best


def _balanced_F_T(stop_us, preds, labels):
    labels = np.asarray(labels, dtype=int)
    m0, m1 = labels == 0, labels == 1
    F = 0.5 * ((preds[m0] == 0).mean() + (preds[m1] == 1).mean())
    T = 0.5 * (stop_us[m0].mean() + stop_us[m1].mean())
    return float(F), float(T)


def compare_schemes(
    power_uw: float = 5.437,
    detection_efficiency: float = 1.0,
    horizon_us: float = 250.0,
    epoch_dt_us: float = 2.0,
    n_per_state: int = 5000,
    a_over_c: Sequence[float] = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0),
    exhaustion_eps: float = 1e-9,
    seed: int = 4242,
    verbose: bool = True,
) -> dict:
    """
    Four stopping rules on identical shots and an identical filter:

        1. fixed-time count threshold             the standard method
        2. adaptive MMPP SPRT, constant boundary  the original scheme here
        3. the same plus the exhaustion exit      free, decisions unchanged
        4. learned policy (regression MC)         the Ludkovski-Sezer optimum

    Rules 2-4 are all evaluated on the same epoch grid, so any difference is
    attributable to the stopping rule rather than to the filter. The learned
    policy is fitted on training paths and applied to independent test paths.
    """
    t0 = time.time()
    params = nv.apply_detection_efficiency(
        nv.shields_2015_params(power_uw), detection_efficiency,
        model="signal_only",
    )

    tr_s, tr_l = nv.simulate_balanced_dataset(
        n_per_state, horizon_us / 1000.0, params, seed
    )
    te_s, te_l = nv.simulate_balanced_dataset(
        n_per_state, horizon_us / 1000.0, params, seed + 7717
    )
    tr = filter_on_grid(tr_s, tr_l, params, horizon_us, epoch_dt_us)
    te = filter_on_grid(te_s, te_l, params, horizon_us, epoch_dt_us)

    if verbose:
        reg = regime_summary(params)
        print(
            f"P = {power_uw} uW, eta = {detection_efficiency}, "
            f"horizon {horizon_us:.0f} us, {reg['photons_per_bright_dwell']:.1f} "
            f"photons/bright dwell"
        )
        print(
            f"{2*n_per_state} train + {2*n_per_state} test paths, "
            f"{tr.llr.shape[1]} epochs, built in {time.time()-t0:.1f} s"
        )

    # ---- fixed-time count threshold, for the frontier panel ---------------
    t_grid = te.t_us[1:]
    cal_c = nv.total_counts_at_times(tr_s, t_grid / 1000.0)
    tst_c = nv.total_counts_at_times(te_s, t_grid / 1000.0)
    F_thr = np.empty(t_grid.size)
    for j in range(t_grid.size):
        n_th, _ = nv.optimize_count_threshold(cal_c[:, j], tr_l)
        F_thr[j] = balanced_fidelity(te_l, np.where(tst_c[:, j] >= n_th, 0, 1))

    # ---- exhaustion diagnostics -------------------------------------------
    exh = te.gap <= exhaustion_eps
    t_exh = np.where(exh.any(axis=1), te.t_us[exh.argmax(axis=1)], np.nan)

    rows = []
    for ac in a_over_c:
        econ = Economics(cost_per_us=1.0 / ac, cost_miss=1.0, cost_false=1.0)

        base = best_constant_boundary_eps(te, econ, 0.0)
        withx = best_constant_boundary_eps(te, econ, exhaustion_eps)
        coeffs = fit_stopping_rule(tr, econ)
        st, pr = apply_stopping_rule(te, coeffs, econ)
        r_lsm = bayes_risk(st, pr, te.labels, econ)

        Fb, Tb = _balanced_F_T(base["stop_us"], base["preds"], te.labels)
        Fx, Tx = _balanced_F_T(withx["stop_us"], withx["preds"], te.labels)
        Fl, Tl = _balanced_F_T(st, pr, te.labels)

        rows.append({
            "a_over_c": float(ac),
            "risk_sprt": base["risk"], "F_sprt": Fb, "T_sprt": Tb,
            "L": base["L"], "offset": base["offset"],
            "risk_sprt_exh": withx["risk"], "F_sprt_exh": Fx, "T_sprt_exh": Tx,
            "risk_lsm": r_lsm, "F_lsm": Fl, "T_lsm": Tl,
            "reduction_pct": 100.0 * (base["risk"] - r_lsm) / base["risk"],
        })

    if verbose:
        print(
            f"\n{'a/c(us)':>8}{'risk SPRT':>11}{'+exh':>9}{'risk LSM':>10}"
            f"{'red.':>8} | {'T SPRT':>8}{'T +exh':>8}{'T LSM':>8}"
            f" | {'F SPRT':>8}{'F LSM':>8}"
        )
        for r in rows:
            print(
                f"{r['a_over_c']:8.0f}{r['risk_sprt']:11.4f}"
                f"{r['risk_sprt_exh']:9.4f}{r['risk_lsm']:10.4f}"
                f"{r['reduction_pct']:7.1f}% | {r['T_sprt']:8.1f}"
                f"{r['T_sprt_exh']:8.1f}{r['T_lsm']:8.1f}"
                f" | {r['F_sprt']:8.4f}{r['F_lsm']:8.4f}"
            )
        print(f"\ntotal {time.time()-t0:.1f} s")

    return {
        "params": params, "power_uw": power_uw,
        "detection_efficiency": detection_efficiency,
        "horizon_us": horizon_us, "epoch_dt_us": epoch_dt_us,
        "threshold_times_us": t_grid, "threshold_F": F_thr,
        "exhaustion_times_us": t_exh, "exhaustion_eps": exhaustion_eps,
        "rows": rows, "test_paths": te,
        "module_version": MODULE_VERSION,
    }


def plot_comparison(result: dict, save_path: str | None = None):
    """
    Four panels:
      (a) fidelity vs mean run time, all four rules on one frontier
      (b) Bayes risk vs a/c
      (c) risk reduction of the learned policy over the tuned constant boundary
      (d) when the initial-state information runs out, against the deadline
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows"]
    ac = np.array([r["a_over_c"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(13.0, 9.5))

    # (a) fidelity vs mean run time
    p = ax[0, 0]
    p.plot(result["threshold_times_us"], result["threshold_F"], ":",
           color="#444444", lw=1.6, label="fixed-time count threshold")
    p.plot([r["T_sprt"] for r in rows], [r["F_sprt"] for r in rows],
           "-o", color="#1f77b4", ms=5, label="adaptive MMPP, constant boundary")
    p.plot([r["T_sprt_exh"] for r in rows], [r["F_sprt_exh"] for r in rows],
           "--s", color="#2ca02c", ms=5, label="+ information-exhaustion exit")
    p.plot([r["T_lsm"] for r in rows], [r["F_lsm"] for r in rows],
           "-D", color="#d62728", ms=6, lw=2.2, label="learned optimal stopping")
    p.set_xscale("log")
    p.set_xlabel("mean run time per shot (us)")
    p.set_ylabel("balanced initial-state fidelity")
    p.set_title("(a) fidelity vs run time", fontsize=11)
    p.grid(alpha=0.25)
    p.legend(fontsize=8.5, loc="lower right")

    # (b) Bayes risk
    p = ax[0, 1]
    p.plot(ac, [r["risk_sprt"] for r in rows], "-o", color="#1f77b4",
           label="constant boundary (tuned)")
    p.plot(ac, [r["risk_sprt_exh"] for r in rows], "--s", color="#2ca02c",
           label="+ exhaustion exit")
    p.plot(ac, [r["risk_lsm"] for r in rows], "-D", color="#d62728", lw=2.2,
           label="learned policy")
    p.set_xscale("log")
    p.set_yscale("log")
    p.set_xlabel("a/c   (us of readout per avoided error)")
    p.set_ylabel("Bayes risk   c E[tau] + a P(miss)/2 + b P(false)/2")
    p.set_title("(b) Bayes risk, lower is better", fontsize=11)
    p.grid(alpha=0.25, which="both")
    p.legend(fontsize=8.5)

    # (c) risk reduction
    p = ax[1, 0]
    p.bar(np.arange(len(ac)), [r["reduction_pct"] for r in rows],
          color="#d62728", alpha=0.8)
    p.set_xticks(np.arange(len(ac)))
    p.set_xticklabels([f"{x:g}" for x in ac])
    p.axhline(0, color="k", lw=0.8)
    p.set_xlabel("a/c (us)")
    p.set_ylabel("risk reduction vs tuned constant boundary (%)")
    p.set_title("(c) what the extra state dimension buys", fontsize=11)
    p.grid(alpha=0.25, axis="y")

    # (d) information exhaustion
    p = ax[1, 1]
    te = result["exhaustion_times_us"]
    fin = np.isfinite(te)
    p.hist(te[fin], bins=40, color="#2ca02c", alpha=0.75,
           label=f"exhausted before the deadline ({100*fin.mean():.0f}%)")
    p.axvline(result["horizon_us"], color="k", ls="--", lw=1.4, label="deadline")
    if fin.any():
        p.axvline(np.median(te[fin]), color="#d62728", lw=1.6,
                  label=f"median {np.median(te[fin]):.0f} us")
    p.set_xlabel("time at which the information gap d reaches zero (us)")
    p.set_ylabel("shots")
    p.set_title("(d) when nothing more can be learned", fontsize=11)
    p.legend(fontsize=8.5)
    p.grid(alpha=0.25)

    fig.suptitle(
        f"Stopping rules compared on identical shots and an identical filter  |  "
        f"P = {result['power_uw']} uW, eta = {result['detection_efficiency']}, "
        f"horizon {result['horizon_us']:.0f} us",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
