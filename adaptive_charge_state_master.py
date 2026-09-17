#!/usr/bin/env python3
"""
Adaptive NV charge-state readout: one master file
=================================================

This single file merges what used to be four scripts:

    adaptive_charge_readout.py      -> sections 2-7 (filter engine, statistics)
    run_speedup_demo.py             -> the ``demo`` experiment
    power_sweep_speedup.py          -> the ``power`` experiment
    validate_adaptive_readout.py    -> the ``validate`` command

and adds what was asked for on top:

    * an optional DETECTOR MODEL with dead time and afterpulsing, which can be
      switched on and off (section 1b, ``--detector``);
    * three new sweeps: switching-rate RATIO, emission CONTRAST, and detector
      EFFICIENCY (section 9).

Goal of the physics
-------------------
Show that an adaptive (sequential) MMPP readout reaches the SAME initial-state
charge fidelity as an optimized fixed-time count threshold in significantly
LESS average run time.

Four methods are always compared on identical shots:

    1. fixed-time count threshold   (the standard method; the baseline)
    2. adaptive count SPRT          (fixed-count stopping, count statistic)
    3. fixed-count MMPP             (fixed-count stopping, event-time statistic)
    4. adaptive MMPP SPRT           (LLR-boundary stopping, event-time statistic)

Methods 2 and 3 are the controls that factor the gain along the two axes of the
design, stopping rule and decision statistic:

    stopping rule   decision statistic   method
    -------------   ------------------   ---------------------------
    fixed time      photon count         1  fixed-time threshold
    fixed time      MMPP LLR                fixed-time MMPP (optional)
    fixed count     count (one bit)      2  adaptive count SPRT
    fixed count     MMPP LLR             3  fixed-count MMPP
    LLR boundary    MMPP LLR             4  adaptive MMPP SPRT

Methods 2 and 3 share their stopping rule exactly -- identical per-shot stop
times, hence identical mean run time -- so 2 -> 3 isolates the value of the
event-time statistic with stopping held fixed, and 3 -> 4 isolates the value of
the LLR as a stopping rule with the statistic held fixed. Without both controls,
a speedup could be attributed to either axis. A fifth method, the fixed-time
MMPP with a calibrated cutoff, is added when ``include_fixed_mmpp`` is set (the
``demo`` experiment uses it; the sweeps deliberately omit it).

What the engine does that a grid filter does not
------------------------------------------------
1. Closed-form 2x2 no-click propagator. The dominant eigenvalue is factored
   out analytically, so the propagator never overflows or underflows and is
   exactly real. The factored scalar cancels identically in the LLR.

2. EXACT, GRID-FREE first-passage times. For the 2-state MMPP initial-state
   LLR: during a no-click interval the LLR decreases monotonically, because
   d/dt log(1'p) = -1'(Lambda p) and the initially-bright hypothesis always has
   the larger filtered emission rate; at every photon arrival the LLR jumps
   upward. Therefore the UPPER boundary can only be crossed AT a click and the
   LOWER boundary only INSIDE a no-click interval, where the crossing is
   unique and available in closed form (``_solve_lower_crossing``).

3. Truncated SPRT with ASYMMETRIC, CALIBRATED boundaries U = offset + L and
   D = offset - L. The offset is calibrated on calibration data exactly as the
   count threshold is, removing the unfair hardcoded LLR >= 0 rule.

4. Paired (stratified) bootstrap and exact McNemar tests. Both methods see the
   same shots, so every comparison is paired.

5. Dimensionless regime reporting: photons per bright dwell, Gamma_tot * t_R,
   and emission contrast, instead of laser power in uW and a bare efficiency.

6. Switching-rate ensembles in POLYSPECTRA coordinates (log K, logit p_bright)
   so a Bayesian ensemble inherits the correct posterior correlation.

State convention
----------------
0 = NV-  (bright)
1 = NV0  (dark)

All rates are kHz = ms^-1; all times internally in ms, reported in us.

Dependency note
---------------
The original four scripts imported ``nv_charge_readout_master_v1_4`` for the
physics primitives (rate model, MMPP simulator, count threshold). If that
module is importable it is used, so the rate model and the simulator are the
same ones the original scripts ran on. If it is not, section 1 below provides
a self-contained REFERENCE implementation so this file runs standalone; its
constants are pinned to the same operating point but are a documented stand-in,
not a re-measurement. ``PHYSICS_LAYER`` reports which one is active, and every
saved result records it.

Two caveats on comparing against the original scripts even with the real module
present. The dataset loop is re-implemented here so the detector model can hook
into it, so random draws are consumed in a different order and results are
statistically equivalent rather than bit-identical. And matched-fidelity times
now always come from ``time_for_fidelity_interp`` rather than
``min_time_for_fidelity``, which removes a grid-spacing sawtooth in the speedup
ratio; ``run_speedup_demo.py`` used the un-interpolated version.

Usage
-----
    python adaptive_charge_state_master.py list
    python adaptive_charge_state_master.py run power
    python adaptive_charge_state_master.py run power --point 0
    python adaptive_charge_state_master.py run ratio --detector
    python adaptive_charge_state_master.py run efficiency --detector-preset stress
    python adaptive_charge_state_master.py plot contrast
    python adaptive_charge_state_master.py summary power
    python adaptive_charge_state_master.py validate

Experiments: demo, power, ratio, contrast, efficiency.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import pickle
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

MODULE_VERSION = "adaptive-charge-state-master-v2.0"


# =============================================================================
# 1a. Physics layer: the real module if available, else a reference build
# =============================================================================

try:  # pragma: no cover - depends on the user's environment
    import nv_charge_readout_master_v1_4 as _nv

    PHYSICS_LAYER = "nv_charge_readout_master_v1_4"
except ModuleNotFoundError:
    _nv = None
    PHYSICS_LAYER = "builtin-reference"


if _nv is not None:
    MMPPParams = _nv.MMPPParams
    shields_2015_params = _nv.shields_2015_params
    apply_detection_efficiency = _nv.apply_detection_efficiency
    total_counts_at_times = _nv.total_counts_at_times
    optimize_count_threshold = _nv.optimize_count_threshold
    balanced_initial_state_fidelity = _nv.balanced_initial_state_fidelity
    initial_state_llr_at_times = _nv.initial_state_llr_at_times
    make_bayesian_switching_ensemble = _nv.make_bayesian_switching_ensemble
    _nv_simulate_mmpp_shot = _nv.simulate_mmpp_shot

    def _replace_params(params, **changes):
        """
        Copy of `params` with some rates changed.

        `dataclasses.replace` is the fast path; the fallback covers a physics
        layer whose MMPPParams happens not to be a dataclass, so nothing here
        depends on that implementation detail.
        """
        try:
            return replace(params, **changes)
        except TypeError:
            fields = dict(
                gamma_minus_to_zero_khz=params.gamma_minus_to_zero_khz,
                gamma_zero_to_minus_khz=params.gamma_zero_to_minus_khz,
                lambda_minus_khz=params.lambda_minus_khz,
                lambda_zero_khz=params.lambda_zero_khz,
            )
            fields.update(changes)
            return MMPPParams(**fields)

else:

    def _replace_params(params, **changes):
        """See the docstring on the other branch."""
        return replace(params, **changes)

    # -------------------------------------------------------------------------
    # REFERENCE IMPLEMENTATION
    #
    # Used only when nv_charge_readout_master_v1_4 is absent. The functional
    # forms follow Shields et al., PRL 114, 136402 (2015): the 594-nm charge
    # switching rates are two-photon processes, hence P^2/(1 + P/P_sat), while
    # the NV- fluorescence rate is a one-photon saturating process, hence
    # P/(P + P_sat). The prefactors below are pinned so that the operating
    # point used throughout the original scripts, P = 5.437 uW, reproduces
    #
    #     Gamma_tot = 9.42 kHz,   p_bright = Gamma_0- / Gamma_tot = 0.118,
    #
    # which are the values the polyspectra fit coordinates in the original
    # validation suite were built around. Substitute the real module for
    # publication numbers; these constants are a faithful stand-in, not a
    # re-measurement.
    # -------------------------------------------------------------------------

    _ANCHOR_POWER_UW = 5.437
    _ANCHOR_GAMMA_TOT_KHZ = 9.42
    _ANCHOR_P_BRIGHT = 0.118

    _P_SAT_IONIZATION_UW = 12.0     # saturation of NV- -> NV0 (594 nm)
    _P_SAT_RECOMBINATION_UW = 25.0  # saturation of NV0 -> NV-
    _P_SAT_EMISSION_UW = 4.0        # saturation of the NV- fluorescence
    _LAMBDA_MINUS_SAT_KHZ = 150.0   # saturated NV- detected photon rate
    _NV0_EMISSION_FRACTION = 0.02   # residual NV0 brightness at 594 nm

    def _two_photon_shape(power_uw: float, p_sat_uw: float) -> float:
        p = float(power_uw)
        return p * p / (1.0 + p / p_sat_uw)

    _A_IONIZATION = (
        (1.0 - _ANCHOR_P_BRIGHT)
        * _ANCHOR_GAMMA_TOT_KHZ
        / _two_photon_shape(_ANCHOR_POWER_UW, _P_SAT_IONIZATION_UW)
    )
    _A_RECOMBINATION = (
        _ANCHOR_P_BRIGHT
        * _ANCHOR_GAMMA_TOT_KHZ
        / _two_photon_shape(_ANCHOR_POWER_UW, _P_SAT_RECOMBINATION_UW)
    )

    @dataclass(frozen=True)
    class MMPPParams:  # type: ignore[no-redef]
        """
        Two-state Markov-modulated Poisson process for NV charge readout.

        gamma_minus_to_zero_khz : ionization rate, NV- -> NV0
        gamma_zero_to_minus_khz : recombination rate, NV0 -> NV-
        lambda_minus_khz        : detected photon rate in NV- (bright)
        lambda_zero_khz         : detected photon rate in NV0 (dark)
        """

        gamma_minus_to_zero_khz: float
        gamma_zero_to_minus_khz: float
        lambda_minus_khz: float
        lambda_zero_khz: float

        def validate(self) -> None:
            if not (self.gamma_minus_to_zero_khz > 0.0):
                raise ValueError("gamma_minus_to_zero_khz must be positive.")
            if not (self.gamma_zero_to_minus_khz > 0.0):
                raise ValueError("gamma_zero_to_minus_khz must be positive.")
            if self.lambda_zero_khz < 0.0:
                raise ValueError("lambda_zero_khz must be non-negative.")
            if not (self.lambda_minus_khz > self.lambda_zero_khz):
                # The monotone-decrease structural fact the exact first-passage
                # scan relies on requires the bright state to be the brighter
                # one. Without it the LLR is not monotone in no-click
                # intervals and the upper/lower boundary asymmetry breaks.
                raise ValueError(
                    "lambda_minus_khz must exceed lambda_zero_khz."
                )
            if not all(
                np.isfinite(v)
                for v in (
                    self.gamma_minus_to_zero_khz,
                    self.gamma_zero_to_minus_khz,
                    self.lambda_minus_khz,
                    self.lambda_zero_khz,
                )
            ):
                raise ValueError("All parameters must be finite.")

        @property
        def switching_generator(self) -> np.ndarray:
            """Q with dp/dt = Q p for p = (p_NV-, p_NV0)'."""
            return np.array(
                [
                    [-self.gamma_minus_to_zero_khz, self.gamma_zero_to_minus_khz],
                    [self.gamma_minus_to_zero_khz, -self.gamma_zero_to_minus_khz],
                ],
                dtype=float,
            )

        @property
        def emission_matrix(self) -> np.ndarray:
            return np.diag([self.lambda_minus_khz, self.lambda_zero_khz])

        @property
        def no_click_generator(self) -> np.ndarray:
            """A = Q - Lambda, the generator of the unnormalized no-click flow."""
            return self.switching_generator - self.emission_matrix

    def shields_2015_params(power_uw: float) -> MMPPParams:  # type: ignore[no-redef]
        """594-nm charge-readout parameters at a given laser power."""
        p = float(power_uw)
        if not (p > 0.0):
            raise ValueError("power_uw must be positive.")

        gamma_m0 = _A_IONIZATION * _two_photon_shape(p, _P_SAT_IONIZATION_UW)
        gamma_0m = _A_RECOMBINATION * _two_photon_shape(
            p, _P_SAT_RECOMBINATION_UW
        )
        lam_minus = _LAMBDA_MINUS_SAT_KHZ * p / (p + _P_SAT_EMISSION_UW)

        return MMPPParams(
            gamma_minus_to_zero_khz=gamma_m0,
            gamma_zero_to_minus_khz=gamma_0m,
            lambda_minus_khz=lam_minus,
            lambda_zero_khz=_NV0_EMISSION_FRACTION * lam_minus,
        )

    def apply_detection_efficiency(  # type: ignore[no-redef]
        params: "MMPPParams",
        detection_efficiency: float,
        model: str = "signal_only",
        background_khz: float = 0.0,
    ) -> "MMPPParams":
        """
        Thin the detected photon rates by eta, leaving the switching rates
        untouched (collection efficiency does not change the charge dynamics).

        model
            "signal_only"     : both rates scale by eta.
            "with_background" : eta scales the signal, then a state-independent
                                background rate is added to both.
        """
        eta = float(detection_efficiency)
        if not (0.0 < eta <= 1.0):
            raise ValueError("detection_efficiency must lie in (0, 1].")

        if model == "signal_only":
            bg = 0.0
        elif model == "with_background":
            bg = float(background_khz)
        else:
            raise ValueError(f"Unknown efficiency model {model!r}.")

        return replace(
            params,
            lambda_minus_khz=eta * params.lambda_minus_khz + bg,
            lambda_zero_khz=eta * params.lambda_zero_khz + bg,
        )

    def balanced_initial_state_fidelity(  # type: ignore[no-redef]
        labels: np.ndarray,
        preds: np.ndarray,
    ) -> float:
        """F_C = 1/2 [P(correct | NV-) + P(correct | NV0)]."""
        labels = np.asarray(labels, dtype=int)
        preds = np.asarray(preds, dtype=int)
        out = 0.0
        for s in (0, 1):
            m = labels == s
            if not np.any(m):
                raise ValueError("Both initial states must be represented.")
            out += float((preds[m] == s).mean())
        return 0.5 * out

    def total_counts_at_times(  # type: ignore[no-redef]
        shots: Sequence[np.ndarray],
        times_ms: np.ndarray,
    ) -> np.ndarray:
        """Cumulative click count at each report time; shape (n_shots, n_times)."""
        times = np.asarray(times_ms, dtype=float).reshape(-1)
        out = np.empty((len(shots), times.size), dtype=int)
        for i, ts in enumerate(shots):
            t = np.sort(np.asarray(ts, dtype=float).reshape(-1))
            out[i] = np.searchsorted(t, times, side="right")
        return out

    def optimize_count_threshold(  # type: ignore[no-redef]
        counts: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[int, float]:
        """
        Balanced-fidelity-optimal integer count threshold.

        Rule: counts >= n_th -> NV- (0). Evaluated by cumulative histogram so
        the whole threshold range costs one pass.
        """
        counts = np.asarray(counts, dtype=int).reshape(-1)
        labels = np.asarray(labels, dtype=int).reshape(-1)

        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())
        if n0 == 0 or n1 == 0:
            raise ValueError("Both initial states must be represented.")

        m = int(counts.max()) if counts.size else 0
        h0 = np.bincount(counts[labels == 0], minlength=m + 1)
        h1 = np.bincount(counts[labels == 1], minlength=m + 1)

        # below[k] = #(counts < k), for k = 0 .. m+1
        below0 = np.concatenate([[0], np.cumsum(h0)])
        below1 = np.concatenate([[0], np.cumsum(h1)])

        F = 0.5 * ((n0 - below0) / n0 + below1 / n1)
        k = int(np.argmax(F))
        return k, float(F[k])

    def initial_state_llr_at_times(  # type: ignore[no-redef]
        timestamps_ms: Sequence[float],
        times_ms: Sequence[float],
        params: "MMPPParams",
    ) -> np.ndarray:
        """
        Independent reference LLR, computed with a dense matrix exponential.

        Deliberately naive: this is the object the closed-form engine in
        section 3 is validated against, so it must not share any of its
        machinery. Clicks strictly before the report time are used.
        """
        from scipy.linalg import expm

        params.validate()

        A = np.asarray(params.no_click_generator, dtype=float)
        Lam = np.asarray(params.emission_matrix, dtype=float)

        ts_all = np.sort(np.asarray(timestamps_ms, dtype=float).reshape(-1))
        times = np.asarray(times_ms, dtype=float).reshape(-1)
        out = np.empty(times.size)

        for j, t in enumerate(times):
            ts = ts_all[ts_all < t]
            H = np.eye(2)
            log_scale = np.zeros(2)
            t_now = 0.0

            for tc in list(ts) + [float(t)]:
                dt = float(tc) - t_now
                if dt > 0.0:
                    H = expm(A * dt) @ H
                    s = H.sum(axis=0)
                    H = H / s[None, :]
                    log_scale = log_scale + np.log(s)
                if tc < t:
                    H = Lam @ H
                    s = H.sum(axis=0)
                    H = H / s[None, :]
                    log_scale = log_scale + np.log(s)
                t_now = float(tc)

            out[j] = log_scale[0] - log_scale[1]

        return out

    def make_bayesian_switching_ensemble(  # type: ignore[no-redef]
        nominal_params: "MMPPParams",
        cv_minus_to_zero: float,
        cv_zero_to_minus: float,
        nodes_per_dimension: int = 3,
    ) -> tuple[list["MMPPParams"], np.ndarray]:
        """
        Gauss-Hermite product rule over INDEPENDENT log-normals on the two
        switching rates.

        Kept because it is the object ``make_correlated_switching_ensemble``
        improves on: an independent product puts quadrature weight along the
        directions polyspectra constrains worst. See section 8.
        """
        nominal_params.validate()

        n = int(nodes_per_dimension)
        if n < 1:
            raise ValueError("nodes_per_dimension must be >= 1.")
        if n == 1:
            return [nominal_params], np.array([1.0])

        sd0 = float(np.log1p(cv_minus_to_zero))
        sd1 = float(np.log1p(cv_zero_to_minus))

        gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(n)
        z = np.sqrt(2.0) * gh_nodes
        w = gh_weights / np.sqrt(np.pi)

        ensemble: list[MMPPParams] = []
        weights: list[float] = []

        for i, zi in enumerate(z):
            for j, zj in enumerate(z):
                p = replace(
                    nominal_params,
                    gamma_minus_to_zero_khz=(
                        nominal_params.gamma_minus_to_zero_khz
                        * float(np.exp(sd0 * zi))
                    ),
                    gamma_zero_to_minus_khz=(
                        nominal_params.gamma_zero_to_minus_khz
                        * float(np.exp(sd1 * zj))
                    ),
                )
                try:
                    p.validate()
                except ValueError:
                    continue
                ensemble.append(p)
                weights.append(float(w[i] * w[j]))

        weights_arr = np.asarray(weights, dtype=float)
        weights_arr /= weights_arr.sum()
        return ensemble, weights_arr

    def _nv_simulate_mmpp_shot(
        t_max_ms: float,
        initial_state: int,
        params: "MMPPParams",
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Exact MMPP shot: alternate exponential dwells, Poisson photons within
        each dwell placed uniformly. Returns sorted arrival times in ms.
        """
        params.validate()

        t_max_ms = float(t_max_ms)
        state = int(initial_state)
        if state not in (0, 1):
            raise ValueError("initial_state must be 0 (NV-) or 1 (NV0).")

        rate_out = (
            params.gamma_minus_to_zero_khz,
            params.gamma_zero_to_minus_khz,
        )
        rate_emit = (params.lambda_minus_khz, params.lambda_zero_khz)

        t = 0.0
        chunks: list[np.ndarray] = []

        while t < t_max_ms:
            dwell = float(rng.exponential(1.0 / rate_out[state]))
            t_end = min(t + dwell, t_max_ms)
            span = t_end - t

            lam = rate_emit[state]
            if span > 0.0 and lam > 0.0:
                k = int(rng.poisson(lam * span))
                if k:
                    chunks.append(t + span * rng.random(k))

            t = t + dwell
            state = 1 - state

        if not chunks:
            return np.empty(0, dtype=float)

        return np.sort(np.concatenate(chunks))


# =============================================================================
# 1b. Shot simulation with an optional detector response
# =============================================================================


@dataclass(frozen=True)
class DetectorModel:
    """
    Optional single-photon-detector imperfections, off by default.

    enabled
        Master switch. When False every other field is ignored and the
        simulator returns the ideal MMPP arrival times, reproducing the
        original scripts exactly.

    dead_time_ns
        After a recorded click the detector is blind for this long.

    paralyzable
        False (default) = non-paralyzable / non-extending dead time: arrivals
        during the dead window are simply lost. True = paralyzable: every
        arrival, recorded or not, restarts the dead window, so the recorded
        rate is non-monotone in the true rate (lambda exp(-lambda tau)).

    afterpulse_probability
        Probability that a recorded click spawns a spurious click. Afterpulses
        are themselves subject to dead time and can cascade, as in a real
        detector.

    afterpulse_time_constant_ns
        Mean delay of the afterpulse after its parent click (exponential). Note
        that the two imperfections interact: an afterpulse whose delay is
        shorter than the dead time is swallowed by its own parent's dead
        window, so the two knobs are not independent. Set this longer than
        ``dead_time_ns`` if you want afterpulsing to be visible at all.

    correct_filter_rates
        Whether the MMPP filter is built from dead-time/afterpulse-CORRECTED
        emission rates (see ``effective_emission_rates``) instead of the true
        ones. This is the honest choice: an experimenter calibrates the rate
        the detector actually records. Set False to measure how much the
        adaptive method loses under pure model mismatch.

    Caveat worth stating: with the detector on, the photon stream is no longer
    a Markov-modulated POISSON process. The MMPP likelihood is then
    approximate, so the SPRT boundaries and the calibrated cutoffs absorb part
    of the mismatch and the reported LLR is a score rather than an exact
    log-likelihood ratio. The count threshold is unaffected as a method, since
    it only ever uses recorded counts.
    """

    enabled: bool = False
    dead_time_ns: float = 50.0
    paralyzable: bool = False
    afterpulse_probability: float = 0.01
    afterpulse_time_constant_ns: float = 100.0
    correct_filter_rates: bool = True

    def validate(self) -> None:
        if self.dead_time_ns < 0.0:
            raise ValueError("dead_time_ns must be non-negative.")
        if not (0.0 <= self.afterpulse_probability < 1.0):
            raise ValueError(
                "afterpulse_probability must lie in [0, 1); at 1 the "
                "afterpulse cascade never terminates."
            )
        if self.afterpulse_probability > 0.0 and not (
            self.afterpulse_time_constant_ns > 0.0
        ):
            raise ValueError(
                "afterpulse_time_constant_ns must be positive when "
                "afterpulsing is enabled."
            )

    @property
    def is_ideal(self) -> bool:
        return (not self.enabled) or (
            self.dead_time_ns <= 0.0 and self.afterpulse_probability <= 0.0
        )

    @property
    def dead_time_ms(self) -> float:
        return 1e-6 * float(self.dead_time_ns)

    @property
    def afterpulse_tau_ms(self) -> float:
        return 1e-6 * float(self.afterpulse_time_constant_ns)

    def tag(self) -> str:
        """Short filesystem-safe descriptor, so runs never overwrite each other."""
        if not self.enabled:
            return "ideal"
        kind = "par" if self.paralyzable else "nonpar"
        return (
            f"det_{kind}_dt{self.dead_time_ns:g}ns"
            f"_ap{self.afterpulse_probability:g}"
            f"_tau{self.afterpulse_time_constant_ns:g}ns"
            f"{'' if self.correct_filter_rates else '_uncorr'}"
        )

    def describe(self) -> str:
        if not self.enabled:
            return "ideal detector (no dead time, no afterpulsing)"
        return (
            f"{'paralyzable' if self.paralyzable else 'non-paralyzable'} "
            f"dead time {self.dead_time_ns:g} ns, afterpulse "
            f"p = {self.afterpulse_probability:g} with tau = "
            f"{self.afterpulse_time_constant_ns:g} ns, filter rates "
            f"{'corrected' if self.correct_filter_rates else 'UNcorrected'}"
        )


DETECTOR_OFF = DetectorModel(enabled=False)

# Two named presets. "realistic" is a decent silicon SPAD; at the photon rates
# of interest (tens of kHz) its effect is a sub-percent rate error, which is
# the point -- the toggle should show that the conclusion is not an artifact.
# "stress" is deliberately abusive: a dead time comparable to the bright-state
# photon spacing and 5% afterpulsing, which is where a count threshold and an
# event-time filter can genuinely diverge. Its afterpulse time constant is set
# LONGER than its dead time on purpose; otherwise the dead window swallows
# essentially every afterpulse and the second knob does nothing.
DETECTOR_PRESETS: dict[str, DetectorModel] = {
    "off": DETECTOR_OFF,
    "realistic": DetectorModel(
        enabled=True,
        dead_time_ns=50.0,
        paralyzable=False,
        afterpulse_probability=0.01,
        afterpulse_time_constant_ns=100.0,
    ),
    "stress": DetectorModel(
        enabled=True,
        dead_time_ns=2000.0,
        paralyzable=False,
        afterpulse_probability=0.05,
        afterpulse_time_constant_ns=5000.0,
    ),
    "paralyzable": DetectorModel(
        enabled=True,
        dead_time_ns=2000.0,
        paralyzable=True,
        afterpulse_probability=0.05,
        afterpulse_time_constant_ns=5000.0,
    ),
}


def apply_detector_response(
    arrival_times_ms: np.ndarray,
    detector: DetectorModel,
    rng: np.random.Generator,
    t_max_ms: float | None = None,
) -> np.ndarray:
    """
    Map ideal photon arrivals to RECORDED click times.

    Single time-ordered pass over a merged stream of primary photons and
    afterpulses. The merge has to be dynamic rather than a two-stage
    post-process, because an afterpulse can arrive before the next primary
    photon, occupy the detector, and thereby block it -- and because an
    afterpulse can only be spawned by a click that was actually recorded.

    A min-heap holds the pending afterpulses; primaries are already sorted, so
    the merge is O(n log n) in the number of afterpulses only.
    """
    primary = np.sort(np.asarray(arrival_times_ms, dtype=float).reshape(-1))

    if detector.is_ideal:
        if t_max_ms is None:
            return primary
        return primary[primary < float(t_max_ms)]

    detector.validate()

    dead_ms = detector.dead_time_ms
    tau_ms = detector.afterpulse_tau_ms
    p_ap = float(detector.afterpulse_probability)

    pending: list[float] = []
    recorded: list[float] = []

    last_recorded = -np.inf   # non-paralyzable reference
    last_arrival = -np.inf    # paralyzable reference
    i = 0
    n = primary.size

    while i < n or pending:
        if pending and (i >= n or pending[0] <= primary[i]):
            t = heapq.heappop(pending)
        else:
            t = float(primary[i])
            i += 1

        if t_max_ms is not None and t >= float(t_max_ms):
            # Both streams are time-ordered from here on, so nothing left can
            # land before the horizon.
            break

        ref = last_arrival if detector.paralyzable else last_recorded
        if t - ref < dead_ms:
            if detector.paralyzable:
                last_arrival = t
            continue

        recorded.append(t)
        last_recorded = t
        last_arrival = t

        if p_ap > 0.0 and rng.random() < p_ap:
            heapq.heappush(pending, t + float(rng.exponential(tau_ms)))

    return np.asarray(recorded, dtype=float)


def effective_emission_rates(
    params: MMPPParams,
    detector: DetectorModel,
) -> MMPPParams:
    """
    Emission rates a real detector RECORDS, to leading order.

    Dead time throttles the recorded rate:

        non-paralyzable : r = lambda / (1 + lambda tau_d)
        paralyzable     : r = lambda exp(-lambda tau_d)

    Afterpulsing then inflates it, but only through the afterpulses that
    actually escape the dead window. An exponential delay outlives the dead
    time with probability exp(-tau_d / tau_ap), so the surviving afterpulse
    probability is

        p_surv = p_ap exp(-tau_d / tau_ap),    r -> r / (1 - p_surv)

    where the geometric factor accounts for afterpulses of afterpulses. Naively
    using p_ap here overestimates the recorded rate badly whenever the
    afterpulse time constant is shorter than the dead time, which is the usual
    case: the detector is still blind when its own afterpulse arrives.

    These are the standard stationary-Poisson results, so they are accurate
    while tau_d is short compared with the charge dwell times -- the regime
    every operating point here is in. They are what the filter should be built
    from when ``detector.correct_filter_rates`` is set: an experimenter
    calibrates the rate the detector reports, not the rate the emitter emits.
    """
    if detector.is_ideal:
        return params

    tau = detector.dead_time_ms
    tau_ap = detector.afterpulse_tau_ms
    p_ap = float(detector.afterpulse_probability)

    if p_ap > 0.0 and tau_ap > 0.0:
        p_surv = p_ap * float(np.exp(-tau / tau_ap))
    else:
        p_surv = 0.0

    def eff(lam: float) -> float:
        if lam <= 0.0:
            return 0.0
        if detector.paralyzable:
            r = lam * float(np.exp(-lam * tau))
        else:
            r = lam / (1.0 + lam * tau)
        return r / (1.0 - p_surv)

    return _replace_params(
        params,
        lambda_minus_khz=eff(params.lambda_minus_khz),
        lambda_zero_khz=eff(params.lambda_zero_khz),
    )


def simulate_mmpp_shot(
    t_max_ms: float,
    initial_state: int,
    params: MMPPParams,
    rng: np.random.Generator,
    detector: DetectorModel = DETECTOR_OFF,
) -> np.ndarray:
    """One shot of recorded click times (ms), starting in `initial_state`."""
    arrivals = _nv_simulate_mmpp_shot(t_max_ms, initial_state, params, rng)
    return apply_detector_response(arrivals, detector, rng, t_max_ms=t_max_ms)


def simulate_balanced_dataset(
    n_shots_per_state: int,
    t_max_ms: float,
    params: MMPPParams,
    seed: int,
    detector: DetectorModel = DETECTOR_OFF,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Equal numbers of NV- and NV0 initial states, in interleaved order.

    Balancing is what makes the balanced fidelity an unweighted average of two
    conditional accuracies; the stationary distribution is heavily NV0-weighted
    and would otherwise dominate every reported number.
    """
    n = int(n_shots_per_state)
    if n < 1:
        raise ValueError("n_shots_per_state must be >= 1.")

    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n)
    shots = [
        simulate_mmpp_shot(t_max_ms, int(s), params, rng, detector)
        for s in labels
    ]
    return shots, labels


# =============================================================================
# 2. Closed-form 2x2 no-click propagator
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
    w: np.ndarray          # w = 1' V  (column sums of eigenvector matrix)
    lambda_diag: np.ndarray
    degenerate: bool

    def scaled_propagator(self, dt_ms: float) -> np.ndarray:
        """M(dt) = V diag(1, exp(-2 delta dt)) V^-1, with exp(A dt) = e^{mu1 dt} M."""
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
            V = np.array([[b, b], [mu1 - a, mu2 - a]], dtype=float)
        elif abs(c) > 0.0:
            V = np.array([[mu1 - d, mu2 - d], [c, c]], dtype=float)
        else:
            V = np.eye(2)

        det_V = V[0, 0] * V[1, 1] - V[0, 1] * V[1, 0]

        if abs(det_V) < 1e-300:
            raise FloatingPointError("Degenerate eigenvector matrix.")

        V_inv = np.array(
            [[V[1, 1], -V[0, 1]], [-V[1, 0], V[0, 0]]], dtype=float
        ) / det_V

    return NoClickSpectral(
        mu1=float(mu1),
        mu2=float(mu2),
        delta=delta,
        V=V,
        V_inv=V_inv,
        w=V.sum(axis=0),
        lambda_diag=np.array(
            [params.lambda_minus_khz, params.lambda_zero_khz], dtype=float
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
# 3. Event-time LLR record and exact first passage
# =============================================================================


@dataclass
class LLRRecord:
    """
    Complete piecewise description of the initial-state LLR for one shot.

    Intervals are indexed k = 0 .. n_clicks. Interval k runs from t_start[k] to
    t_end[k]; interval n_clicks ends at t_max. Within each interval the LLR
    falls monotonically from llr_start[k] to llr_end[k]. At click k the LLR
    jumps upward from llr_end[k] to llr_post[k].
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

    Cost is O(n_clicks) with small dense 2x2 algebra. Once built, a record can
    be re-scanned for ANY boundary pair at negligible cost, which is what makes
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
    spec: NoClickSpectral | None = None,
) -> list[LLRRecord]:
    if spec is None:
        spec = build_no_click_spectral(params)
    t_max_ms = float(t_max_us) / 1000.0
    return [build_llr_record(ts, t_max_ms, spec) for ts in shots]


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

    `deadline_ms` may be shorter than the record's own t_max, so one record set
    serves an entire grid of deadlines with no refiltering.
    """
    L = float(boundary_half_width)
    b = float(offset)

    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("boundary_half_width must be positive and finite.")

    if abs(b) >= L:
        raise ValueError("offset must satisfy |offset| < boundary_half_width.")

    U = b + L
    D = b - L

    t_cap = (
        record.t_max_ms
        if deadline_ms is None
        else min(float(deadline_ms), record.t_max_ms)
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
                alpha, beta = _interval_llr_coefficients(record.H_start[k], spec)
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
    llr_end_at_cap = float(
        record.llr_end[
            min(int(np.searchsorted(record.t_end, t_cap, side="left")), n_int - 1)
        ]
    )
    return t_cap, (0 if llr_end_at_cap >= b else 1)


def evaluate_adaptive(
    records: Sequence[LLRRecord],
    boundary_half_width: float,
    offset: float,
    spec: NoClickSpectral,
    deadline_us: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar-loop truncated SPRT over many records. Returns (times_us, preds)."""
    n = len(records)
    times = np.empty(n)
    preds = np.empty(n, dtype=int)

    deadline_ms = None if deadline_us is None else float(deadline_us) / 1000.0

    for i, rec in enumerate(records):
        t_stop, pred = first_passage(
            rec, boundary_half_width, offset, spec, deadline_ms=deadline_ms
        )
        times[i] = t_stop * 1000.0
        preds[i] = pred

    return times, preds


# =============================================================================
# 4. Vectorized SPRT engine
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

    return PaddedRecords(
        t_start=t_start,
        t_end=t_end,
        llr_start=llr_start,
        llr_end=llr_end,
        llr_post=llr_post,
        alpha=alpha,
        beta=beta,
        n_int=n_int,
        n_clicks=n_clicks,
        t_max_ms=float(records[0].t_max_ms),
        spec=spec,
    )


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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized truncated SPRT over all shots at once.

    Boundaries U = offset + L (decide NV-) and D = offset - L (decide NV0).
    Exact: the upper boundary is tested only at clicks, the lower boundary only
    inside no-click intervals, and the lower crossing time is closed form.
    Matches the scalar `first_passage` to machine precision.
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

    # Ties within the same interval go to the lower boundary, because the LLR
    # reaches D before the click that would raise it to U.
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

    return stop_ms * 1000.0, preds


# =============================================================================
# 5. Metrics, regime reporting, calibration, paired statistics
# =============================================================================


def balanced_fidelity(labels: np.ndarray, preds: np.ndarray) -> float:
    """F_C = 1/2 [P(correct | NV-) + P(correct | NV0)]."""
    return balanced_initial_state_fidelity(labels, preds)


def balanced_mean_time(labels: np.ndarray, times_us: np.ndarray) -> float:
    """
    Class-balanced mean run time, matching the balanced fidelity convention.

    Using the balanced mean avoids letting the (heavily NV0-weighted)
    stationary distribution dominate the reported run time.
    """
    labels = np.asarray(labels, dtype=int)
    times_us = np.asarray(times_us, dtype=float)
    return 0.5 * (
        float(times_us[labels == 0].mean()) + float(times_us[labels == 1].mean())
    )


def regime_summary(params: MMPPParams, t_R_us: float | None = None) -> dict:
    """
    Dimensionless description of the operating point.

    These are the quantities that actually control whether event-time
    filtering can beat a count threshold, and they are portable across samples
    and setups in a way that laser power in uW is not.

        photons_per_bright_dwell = lambda_- / Gamma_-0
            The sparsity parameter. As it falls below ~1 the total count
            becomes an almost sufficient statistic and the MMPP advantage
            vanishes.

        gamma_tot_t_R
            Switching events per readout window. The MMPP advantage requires
            this to be of order 1; otherwise no blink occurs during readout.

        contrast = (lambda_- - lambda_0) / (lambda_- + lambda_0)

        switching_ratio = Gamma_-0 / Gamma_0-
            Ionization to recombination asymmetry; equals
            (1 - p_bright)/p_bright.
    """
    params.validate()

    gamma_tot = params.gamma_minus_to_zero_khz + params.gamma_zero_to_minus_khz

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
            params.gamma_zero_to_minus_khz / gamma_tot if gamma_tot > 0 else np.nan
        ),
        "switching_ratio": (
            params.gamma_minus_to_zero_khz / params.gamma_zero_to_minus_khz
            if params.gamma_zero_to_minus_khz > 0
            else np.inf
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
    unfair hardcoded LLR >= 0 rule: the count threshold was calibrated on data
    while the MMPP cutoff was not. Under any model mismatch -- and the detector
    model of section 1b guarantees some -- the Bayes-optimal LLR cutoff is not
    zero.

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

    # Sweep cutoff upward through the sorted statistic. Shots below the cutoff
    # are called NV0.
    zero_below = np.concatenate([[0], np.cumsum(lab_sorted == 1)])
    minus_below = np.concatenate([[0], np.cumsum(lab_sorted == 0)])

    F = 0.5 * ((n_minus - minus_below) / n_minus + zero_below / n_zero)

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


def pareto_frontier(times: np.ndarray, fidelities: np.ndarray) -> np.ndarray:
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

    This is the adaptive version of the standard threshold method. Including it
    separates two distinct sources of gain:

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


def llr_at_stop_times(
    packed: PaddedRecords,
    stop_times_us: np.ndarray,
) -> np.ndarray:
    """
    Exact LLR at a PER-SHOT stop time. Returns shape (n_shots,).

    `llr_at_times` evaluates a common time grid for every shot; this evaluates
    each shot at its own stopping instant, which is what any data-dependent
    stopping rule needs.

    Boundary convention: the value returned is the LLR an instant BEFORE any
    photon landing exactly at `stop_times_us`, because the interval search uses
    a strict `t_start < t`. Callers that stop AT a photon and have therefore
    observed it must use that photon's post-jump value instead -- see
    `fixed_count_mmpp_statistic`, which does exactly that.
    """
    t = np.asarray(stop_times_us, dtype=float).reshape(-1) / 1000.0
    spec = packed.spec
    n = packed.n_shots

    if t.size != n:
        raise ValueError("stop_times_us must have one entry per shot.")

    rows = np.arange(n)
    k = np.maximum((packed.t_start < t[:, None]).sum(axis=1) - 1, 0)
    dt = np.maximum(t - packed.t_start[rows, k], 0.0)
    x = (
        np.ones_like(dt)
        if spec.degenerate
        else np.exp(-2.0 * spec.delta * dt)
    )
    g0 = packed.alpha[rows, k, 0] + packed.beta[rows, k, 0] * x
    g1 = packed.alpha[rows, k, 1] + packed.beta[rows, k, 1] * x
    return packed.llr_start[rows, k] + np.log(g0) - np.log(g1)


def fixed_count_mmpp_statistic(
    packed: PaddedRecords,
    click_times_ms: np.ndarray,
    n_up: int,
    deadline_us: float,
    llr_at_deadline: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    FIXED-COUNT MMPP: stop at the n_up-th photon, decide with the MMPP LLR.

    This completes the 2x2 of stopping rule against decision statistic:

        stopping rule   decision statistic   method
        -------------   ------------------   ------------------------
        fixed time      photon count         fixed-time threshold (1)
        fixed time      MMPP LLR             fixed-time MMPP (2)
        fixed count     count (trivial)      adaptive count SPRT (3)
        fixed count     MMPP LLR             fixed-count MMPP (this)
        LLR boundary    MMPP LLR             adaptive MMPP SPRT (4)

    It shares its stopping rule EXACTLY with `run_adaptive_count` -- same
    per-shot stop times, hence the same mean run time -- so comparing the two
    isolates the value of the decision statistic with the stopping rule held
    fixed. Comparing it with the adaptive MMPP SPRT then isolates the value of
    the LLR as a STOPPING rule, with the statistic held fixed. Method 3 is the
    degenerate case where the statistic carries only one bit, "did n_up photons
    arrive before the deadline".

    Returns (stop_times_us, llr_at_stop). The caller calibrates a cutoff on the
    returned statistic, exactly as the count threshold is calibrated.

    A shot that reaches n_up has OBSERVED that photon, so its statistic is the
    post-jump LLR at that arrival, not the pre-jump limit. Using the pre-jump
    value would silently discard one photon of evidence and understate the
    method.

    `llr_at_deadline` may be passed in to avoid recomputing it for every n_up
    that shares a deadline.
    """
    n_up = int(n_up)
    if n_up < 1:
        raise ValueError("n_up must be >= 1.")

    deadline_ms = float(deadline_us) / 1000.0
    n = packed.n_shots

    if n_up > click_times_ms.shape[1]:
        t_hit = np.full(n, np.inf)
    else:
        t_hit = click_times_ms[:, n_up - 1]

    reached = t_hit <= deadline_ms
    stop_ms = np.where(reached, t_hit, deadline_ms)

    if llr_at_deadline is None:
        llr_at_deadline = llr_at_times(packed, np.array([deadline_us]))[:, 0]

    # Post-jump LLR at the n_up-th photon for the shots that got there.
    if n_up > packed.llr_post.shape[1]:
        llr_reached = np.full(n, -np.inf)
    else:
        llr_reached = packed.llr_post[:, n_up - 1]

    llr = np.where(reached, llr_reached, llr_at_deadline)

    return stop_ms * 1000.0, llr


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


# =============================================================================
# 6. One operating point: threshold vs adaptive count vs adaptive MMPP
# =============================================================================


@dataclass
class RunConfig:
    """Everything that is not the physics of the operating point."""

    n_cal: int = 700
    n_test: int = 1800
    n_boot: int = 300
    n_readout_times: int = 60
    n_deadlines: int = 8
    max_n_up: int = 120
    seed: int = 20260916
    detector: DetectorModel = DETECTOR_OFF
    include_fixed_mmpp: bool = False
    efficiency_model: str = "signal_only"
    verbose: bool = True

    boundary_widths: np.ndarray = field(
        default_factory=lambda: np.geomspace(0.15, 12.0, 10)
    )
    offsets: np.ndarray = field(
        default_factory=lambda: np.array([-1.2, -0.6, -0.3, 0.0, 0.3, 0.6])
    )
    # Common absolute fidelity grid. Curves terminate where a method's ceiling
    # is reached, which is itself informative: high power, low contrast and low
    # efficiency all cannot reach high fidelity at any readout time.
    target_fidelities: np.ndarray = field(
        default_factory=lambda: np.round(np.arange(0.55, 0.995, 0.01), 4)
    )

    def quick(self) -> "RunConfig":
        """Small, fast version for smoke tests and code changes."""
        return replace(
            self,
            n_cal=120,
            n_test=240,
            n_boot=40,
            n_readout_times=18,
            n_deadlines=4,
            max_n_up=24,
            boundary_widths=np.geomspace(0.3, 8.0, 5),
            offsets=np.array([-0.6, 0.0, 0.3]),
            target_fidelities=np.round(np.arange(0.55, 0.99, 0.05), 4),
        )


@dataclass
class OperatingPoint:
    """A physical operating point plus how to label it in a sweep."""

    name: str
    label: str
    params: MMPPParams
    sweep_value: float
    horizon_us: float | None = None
    power_uw: float | None = None
    detection_efficiency: float = 1.0
    seed_offset: int = 0


def choose_horizon_us(
    params: MMPPParams,
    n_switching_times: float = 6.0,
    max_expected_clicks: float = 500.0,
) -> float:
    """
    Simulation horizon: long enough that the fidelity optimum is interior,
    short enough that the padded click arrays stay compact.

    The binding constraint is normally `n_switching_times / Gamma_tot`; the
    click cap only matters at low power and high efficiency, where dwell times
    are long and the emitter is bright throughout.
    """
    reg = regime_summary(params)
    by_switching = 1000.0 * n_switching_times / reg["gamma_tot_khz"]

    p_b = reg["p_bright_stationary"]
    mean_rate_khz = (
        p_b * params.lambda_minus_khz + (1.0 - p_b) * params.lambda_zero_khz
    )
    by_clicks = 1000.0 * max_expected_clicks / max(mean_rate_khz, 1e-9)

    return float(min(by_switching, by_clicks))


def time_grid_floor_us(params: MMPPParams, horizon_us: float) -> float:
    """
    Shortest run time any method is allowed to use.

    This MUST NOT be derived from the simulation horizon. At low power the
    horizon is tens of ms, so a horizon/200 floor forbids the fixed-time
    threshold from using windows shorter than ~100 us while the adaptive rule
    stops freely at ~15 us. That asymmetry alone manufactured an apparent 8x
    speedup at 0.875 uW. Tying the floor to the bright-state photon interval
    instead gives every method the same accessible time range.
    """
    floor = 100.0 / params.lambda_minus_khz          # ~0.1 bright photon periods
    return float(min(max(floor, 1e-3), horizon_us / 5000.0 + floor))


def _filter_params_for(
    params: MMPPParams,
    detector: DetectorModel,
) -> MMPPParams:
    """
    Parameters the MMPP filter is built from.

    With the detector off this is the truth. With it on and
    ``correct_filter_rates`` set, it is the rate the detector actually records,
    which is what a calibration measurement returns.
    """
    if not detector.correct_filter_rates:
        return params

    eff = effective_emission_rates(params, detector)

    try:
        eff.validate()
    except ValueError as exc:
        raise ValueError(
            "Dead-time-corrected emission rates are not a valid MMPP "
            f"({exc}). With a paralyzable dead time this happens once "
            "lambda * tau_d exceeds ~1, where the recorded rate stops being "
            "monotone in the true rate and the bright state can record fewer "
            "clicks than the dark one. Shorten dead_time_ns, lower the "
            "photon rate, or run with correct_filter_rates=False."
        ) from exc

    return eff


def run_operating_point(point: OperatingPoint, cfg: RunConfig) -> dict:
    """
    Threshold vs adaptive count vs adaptive MMPP at one operating point.

    Returns the fidelity-vs-time curves for every method plus the
    matched-fidelity speedup table with paired bootstrap CIs. The fixed-time
    MMPP baseline is added when ``cfg.include_fixed_mmpp`` is set.
    """
    t0 = time.time()

    params = point.params
    params.validate()

    reg = regime_summary(params)
    horizon_us = (
        float(point.horizon_us)
        if point.horizon_us is not None
        else choose_horizon_us(params)
    )

    detector = cfg.detector
    filter_params = _filter_params_for(params, detector)
    seed = int(cfg.seed) + int(point.seed_offset)

    if cfg.verbose:
        print(f"\n{'=' * 78}")
        print(f"{point.label}  |  horizon = {horizon_us:.1f} us")
        print(f"  detector: {detector.describe()}")
        print(
            f"  lambda_- = {params.lambda_minus_khz:8.2f} kHz   "
            f"Gamma_-0 = {params.gamma_minus_to_zero_khz:8.4f} kHz"
        )
        print(
            f"  lambda_0 = {params.lambda_zero_khz:8.3f} kHz   "
            f"Gamma_0- = {params.gamma_zero_to_minus_khz:8.4f} kHz"
        )
        print(
            f"  photons/bright dwell = {reg['photons_per_bright_dwell']:8.2f}   "
            f"contrast = {reg['contrast']:.3f}   "
            f"Gamma_-0/Gamma_0- = {reg['switching_ratio']:.2f}"
        )
        if filter_params is not params:
            print(
                f"  filter rates (recorded): lambda_- = "
                f"{filter_params.lambda_minus_khz:.2f} kHz, lambda_0 = "
                f"{filter_params.lambda_zero_khz:.3f} kHz"
            )
        print(f"{'=' * 78}")

    # ---- data --------------------------------------------------------------
    cal_shots, cal_labels = simulate_balanced_dataset(
        cfg.n_cal, horizon_us / 1000.0, params, seed, detector
    )
    test_shots, test_labels = simulate_balanced_dataset(
        cfg.n_test, horizon_us / 1000.0, params, seed + 7717, detector
    )

    spec = build_no_click_spectral(filter_params)
    cal_packed = pack_records(
        build_records(cal_shots, horizon_us, filter_params, spec), spec
    )
    test_packed = pack_records(
        build_records(test_shots, horizon_us, filter_params, spec), spec
    )

    if cfg.verbose:
        print(
            f"  records built in {time.time() - t0:.1f} s "
            f"(max clicks/shot = {int(test_packed.n_clicks.max())})"
        )

    t_floor_us = time_grid_floor_us(params, horizon_us)
    readout_times_us = np.geomspace(t_floor_us, horizon_us, cfg.n_readout_times)

    # ---- 1. fixed-time count threshold -------------------------------------
    cal_counts = total_counts_at_times(cal_shots, readout_times_us / 1000.0)
    test_counts = total_counts_at_times(test_shots, readout_times_us / 1000.0)

    n_t = readout_times_us.size
    thr_correct = np.empty((len(test_shots), n_t))
    thr_correct_cal = np.empty((len(cal_shots), n_t))
    thresholds = np.empty(n_t, dtype=int)

    for j in range(n_t):
        n_th, _ = optimize_count_threshold(cal_counts[:, j], cal_labels)
        thresholds[j] = n_th
        thr_correct[:, j] = np.where(test_counts[:, j] >= n_th, 0, 1) == test_labels
        thr_correct_cal[:, j] = (
            np.where(cal_counts[:, j] >= n_th, 0, 1) == cal_labels
        )

    thr_times = np.tile(readout_times_us, (len(test_shots), 1))
    F_thr, T_thr = _method_curve(test_labels, thr_correct, thr_times)
    F_thr_cal, _ = _method_curve(
        cal_labels, thr_correct_cal, np.tile(readout_times_us, (len(cal_shots), 1))
    )

    # ---- 2. fixed-time MMPP with a CALIBRATED cutoff (optional) ------------
    F_mmpp = T_mmpp = mmpp_cutoffs = mmpp_correct = None
    mcnemar = None

    if cfg.include_fixed_mmpp:
        cal_llr = llr_at_times(cal_packed, readout_times_us)
        test_llr = llr_at_times(test_packed, readout_times_us)

        mmpp_correct = np.empty((len(test_shots), n_t))
        mmpp_cutoffs = np.empty(n_t)

        for j in range(n_t):
            cut, _ = optimize_scalar_cutoff(cal_llr[:, j], cal_labels)
            mmpp_cutoffs[j] = cut
            mmpp_correct[:, j] = (
                np.where(test_llr[:, j] >= cut, 0, 1) == test_labels
            )

        F_mmpp, T_mmpp = _method_curve(test_labels, mmpp_correct, thr_times)

        # Paired McNemar at the threshold's own best readout time: same shots,
        # same t_R, count threshold vs event-time statistic.
        j_best = int(np.argmax(F_thr))

        def _preds(correct_col):
            c = np.asarray(correct_col) > 0.5
            return np.where(c, test_labels, 1 - test_labels)

        mcnemar = mcnemar_exact(
            test_labels,
            _preds(thr_correct[:, j_best]),
            _preds(mmpp_correct[:, j_best]),
        )
        mcnemar["readout_time_us"] = float(readout_times_us[j_best])

    # Deadlines span the SAME range as the threshold readout times, so the
    # adaptive methods are neither helped nor handicapped by their grid.
    deadlines_us = np.geomspace(4.0 * t_floor_us, horizon_us, cfg.n_deadlines)

    # ---- 3. adaptive count SPRT --------------------------------------------
    cal_clicks = click_time_matrix(cal_packed)
    test_clicks = click_time_matrix(test_packed)

    max_n_up = int(
        min(cfg.max_n_up, max(3, np.percentile(test_packed.n_clicks, 99) + 1))
    )
    cnt_cfg = [
        (int(n_up), float(dl))
        for dl in deadlines_us
        for n_up in range(1, max_n_up + 1)
    ]

    cnt_correct, cnt_time, cnt_cal_F, cnt_cal_T = [], [], [], []
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

    # ---- 3b. fixed-count MMPP ----------------------------------------------
    # Same stopping rule as method 3, but the decision uses the calibrated MMPP
    # LLR at the stopping instant. The cutoff is fitted on calibration data and
    # applied to test, exactly as the count threshold is.
    cal_llr_deadline = {
        float(dl): llr_at_times(cal_packed, np.array([dl]))[:, 0]
        for dl in deadlines_us
    }
    test_llr_deadline = {
        float(dl): llr_at_times(test_packed, np.array([dl]))[:, 0]
        for dl in deadlines_us
    }

    fcm_correct, fcm_time, fcm_cal_F, fcm_cal_T = [], [], [], []
    fcm_cutoffs = []
    for n_up, dl in cnt_cfg:
        tc, sc = fixed_count_mmpp_statistic(
            cal_packed, cal_clicks, n_up, dl, cal_llr_deadline[float(dl)]
        )
        cut, _ = optimize_scalar_cutoff(sc, cal_labels)
        fcm_cutoffs.append(float(cut))
        pc = np.where(sc >= cut, 0, 1)
        fcm_cal_F.append(balanced_fidelity(cal_labels, pc))
        fcm_cal_T.append(balanced_mean_time(cal_labels, tc))

        tt, st = fixed_count_mmpp_statistic(
            test_packed, test_clicks, n_up, dl, test_llr_deadline[float(dl)]
        )
        pt = np.where(st >= cut, 0, 1)
        fcm_correct.append((pt == test_labels).astype(float))
        fcm_time.append(tt)

    fcm_correct = np.column_stack(fcm_correct)
    fcm_time = np.column_stack(fcm_time)
    F_fcm, T_fcm = _method_curve(test_labels, fcm_correct, fcm_time)

    # ---- 4. adaptive MMPP SPRT ---------------------------------------------
    # Only (L, offset) pairs whose boundaries bracket LLR(0) = 0 are
    # meaningful; |offset| >= L decides the shot before any data arrives.
    ada_cfg = [
        (float(L), float(b), float(dl))
        for dl in deadlines_us
        for b in np.asarray(cfg.offsets, dtype=float)
        for L in np.asarray(cfg.boundary_widths, dtype=float)
        if abs(float(b)) < float(L)
    ]

    ada_correct, ada_time, ada_cal_F, ada_cal_T = [], [], [], []
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

    if cfg.verbose:
        print(
            f"  scanned {len(ada_cfg)} MMPP and {len(cnt_cfg)} count "
            f"configurations in {time.time() - t0:.1f} s"
        )

    # ---- matched-fidelity speedups + paired bootstrap ----------------------
    keep_ada = pareto_frontier(np.asarray(ada_cal_T), np.asarray(ada_cal_F))
    keep_cnt = pareto_frontier(np.asarray(cnt_cal_T), np.asarray(cnt_cal_F))
    keep_fcm = pareto_frontier(np.asarray(fcm_cal_T), np.asarray(fcm_cal_F))

    boots = stratified_bootstrap_indices(test_labels, cfg.n_boot, seed + 991)

    # Precompute per-resample curves once, then reuse across all targets.
    boot_curves = []
    for i0, i1 in boots:
        boot_curves.append(
            (
                0.5 * (thr_correct[i0].mean(axis=0) + thr_correct[i1].mean(axis=0)),
                0.5
                * (
                    ada_correct[np.ix_(i0, keep_ada)].mean(axis=0)
                    + ada_correct[np.ix_(i1, keep_ada)].mean(axis=0)
                ),
                0.5
                * (
                    ada_time[np.ix_(i0, keep_ada)].mean(axis=0)
                    + ada_time[np.ix_(i1, keep_ada)].mean(axis=0)
                ),
                0.5
                * (
                    cnt_correct[np.ix_(i0, keep_cnt)].mean(axis=0)
                    + cnt_correct[np.ix_(i1, keep_cnt)].mean(axis=0)
                ),
                0.5
                * (
                    cnt_time[np.ix_(i0, keep_cnt)].mean(axis=0)
                    + cnt_time[np.ix_(i1, keep_cnt)].mean(axis=0)
                ),
                0.5
                * (
                    fcm_correct[np.ix_(i0, keep_fcm)].mean(axis=0)
                    + fcm_correct[np.ix_(i1, keep_fcm)].mean(axis=0)
                ),
                0.5
                * (
                    fcm_time[np.ix_(i0, keep_fcm)].mean(axis=0)
                    + fcm_time[np.ix_(i1, keep_fcm)].mean(axis=0)
                ),
            )
        )

    def _ci(v) -> tuple[float, float]:
        v = np.asarray(v, dtype=float)
        if v.size < 20:
            return np.nan, np.nan
        return (
            float(np.nanpercentile(v, 2.5)),
            float(np.nanpercentile(v, 97.5)),
        )

    rows = []
    for F_star in np.asarray(cfg.target_fidelities, dtype=float):
        t_thr = time_for_fidelity_interp(T_thr, F_thr, F_star)
        t_cnt = time_for_fidelity_interp(T_cnt, F_cnt, F_star)
        t_ada = time_for_fidelity_interp(T_ada, F_ada, F_star)
        t_fcm = time_for_fidelity_interp(T_fcm, F_fcm, F_star)

        sp_m, sp_c, sp_f = [], [], []
        for Fb_thr, Fb_a, Tb_a, Fb_c, Tb_c, Fb_f, Tb_f in boot_curves:
            base = time_for_fidelity_interp(readout_times_us, Fb_thr, F_star)
            if not np.isfinite(base):
                continue
            ta = time_for_fidelity_interp(Tb_a, Fb_a, F_star)
            tc = time_for_fidelity_interp(Tb_c, Fb_c, F_star)
            tf = time_for_fidelity_interp(Tb_f, Fb_f, F_star)
            if np.isfinite(ta) and ta > 0:
                sp_m.append(base / ta)
            if np.isfinite(tc) and tc > 0:
                sp_c.append(base / tc)
            if np.isfinite(tf) and tf > 0:
                sp_f.append(base / tf)

        lo_m, hi_m = _ci(sp_m)
        lo_c, hi_c = _ci(sp_c)
        lo_f, hi_f = _ci(sp_f)

        sp_mmpp = (
            t_thr / t_ada
            if np.isfinite(t_thr) and np.isfinite(t_ada) and t_ada > 0
            else np.nan
        )
        sp_count = (
            t_thr / t_cnt
            if np.isfinite(t_thr) and np.isfinite(t_cnt) and t_cnt > 0
            else np.nan
        )
        sp_fcm = (
            t_thr / t_fcm
            if np.isfinite(t_thr) and np.isfinite(t_fcm) and t_fcm > 0
            else np.nan
        )

        rows.append(
            {
                "target_fidelity": float(F_star),
                "t_threshold_us": t_thr,
                "t_adaptive_count_us": t_cnt,
                "t_fixed_count_mmpp_us": t_fcm,
                "t_adaptive_mmpp_us": t_ada,
                "speedup_mmpp": sp_mmpp,
                "speedup_mmpp_ci_low": lo_m,
                "speedup_mmpp_ci_high": hi_m,
                "speedup_count": sp_count,
                "speedup_count_ci_low": lo_c,
                "speedup_count_ci_high": hi_c,
                "speedup_fixed_count_mmpp": sp_fcm,
                "speedup_fixed_count_mmpp_ci_low": lo_f,
                "speedup_fixed_count_mmpp_ci_high": hi_f,
                # Aliases kept so code written against the original
                # run_speedup_study result dict still works.
                "speedup_vs_threshold": sp_mmpp,
                "speedup_ci_low": lo_m,
                "speedup_ci_high": hi_m,
                "speedup_from_adaptivity_only": sp_count,
            }
        )

    if cfg.verbose:
        print(
            f"\n  {'F*':>6} {'t_thr':>10} {'t_cnt':>10} {'t_fcm':>10} "
            f"{'t_mmpp':>10} {'sp_mmpp':>8} {'sp_fcm':>8} {'sp_cnt':>8}"
        )
        for r in rows:
            if not np.isfinite(r["speedup_mmpp"]):
                continue
            print(
                f"  {r['target_fidelity']:6.2f} {r['t_threshold_us']:10.2f} "
                f"{r['t_adaptive_count_us']:10.2f} "
                f"{r['t_fixed_count_mmpp_us']:10.2f} "
                f"{r['t_adaptive_mmpp_us']:10.2f} "
                f"{r['speedup_mmpp']:8.2f} {r['speedup_fixed_count_mmpp']:8.2f} "
                f"{r['speedup_count']:8.2f}"
            )
        ceil = f"threshold {F_thr.max():.4f}"
        if F_mmpp is not None:
            ceil += f", fixed MMPP {F_mmpp.max():.4f}"
        ceil += (
            f", adaptive count {F_cnt.max():.4f}, "
            f"fixed-count MMPP {F_fcm.max():.4f}, "
            f"adaptive MMPP {F_ada.max():.4f}"
        )
        print(f"\n  ceilings: {ceil}")
        if mcnemar is not None:
            print(
                f"  McNemar at t_R = {mcnemar['readout_time_us']:.1f} us "
                f"(threshold vs fixed-time MMPP): threshold-only-right "
                f"{mcnemar['n10']}, MMPP-only-right {mcnemar['n01']}, "
                f"p = {mcnemar['p_value']:.3g}"
            )
        print(f"  total {time.time() - t0:.1f} s")

    result = {
        "name": point.name,
        "label": point.label,
        "sweep_value": float(point.sweep_value),
        "power_uw": point.power_uw,
        "detection_efficiency": float(point.detection_efficiency),
        "efficiency_model": cfg.efficiency_model,
        "params": params,
        "filter_params": filter_params,
        "detector": detector,
        "regime": reg,
        "horizon_us": horizon_us,
        "seed": seed,
        "readout_times_us": readout_times_us,
        "time_grid_floor_us": t_floor_us,
        "deadlines_us": deadlines_us,
        "count_thresholds": thresholds,
        "F_threshold": F_thr,
        "T_threshold": T_thr,
        "F_threshold_cal": F_thr_cal,
        "F_fixed_mmpp": F_mmpp,
        "T_fixed_mmpp": T_mmpp,
        "mmpp_cutoffs": mmpp_cutoffs,
        "mcnemar_threshold_vs_fixed_mmpp": mcnemar,
        "adaptive_count_configs": cnt_cfg,
        "F_adaptive_count": F_cnt,
        "T_adaptive_count": T_cnt,
        "frontier_adaptive_count": pareto_frontier(T_cnt, F_cnt),
        # Fixed-count MMPP shares cnt_cfg: same (n_up, deadline) grid, same
        # stopping rule, different decision statistic.
        "F_fixed_count_mmpp": F_fcm,
        "T_fixed_count_mmpp": T_fcm,
        "frontier_fixed_count_mmpp": pareto_frontier(T_fcm, F_fcm),
        "fixed_count_mmpp_cutoffs": np.asarray(fcm_cutoffs),
        "adaptive_mmpp_configs": ada_cfg,
        "F_adaptive_mmpp": F_ada,
        "T_adaptive_mmpp": T_ada,
        "F_adaptive_mmpp_cal": np.asarray(ada_cal_F),
        "T_adaptive_mmpp_cal": np.asarray(ada_cal_T),
        "frontier_adaptive_mmpp": pareto_frontier(T_ada, F_ada),
        "speedup_table": rows,
        "n_cal_per_state": cfg.n_cal,
        "n_test_per_state": cfg.n_test,
        "n_boot": cfg.n_boot,
        "module_version": MODULE_VERSION,
        "physics_layer": PHYSICS_LAYER,
        # Stored compactly so the speedup analysis and bootstrap can be redone
        # without re-simulating: correctness as bool, times as float32.
        "test_labels": test_labels,
        "thr_correct": thr_correct.astype(bool),
        "ada_correct": ada_correct.astype(bool),
        "ada_time": ada_time.astype(np.float32),
        "cnt_correct": cnt_correct.astype(bool),
        "cnt_time": cnt_time.astype(np.float32),
        "fcm_correct": fcm_correct.astype(bool),
        "fcm_time": fcm_time.astype(np.float32),
        "keep_ada": keep_ada,
        "keep_cnt": keep_cnt,
        "keep_fcm": keep_fcm,
    }

    if mmpp_correct is not None:
        result["fixed_mmpp_correct"] = mmpp_correct.astype(bool)

    return result


def max_fidelity_summary(result: dict) -> dict:
    """Best fidelity each method can reach anywhere in its configuration set."""
    out = {"threshold": float(np.max(result["F_threshold"]))}
    if result.get("F_fixed_mmpp") is not None:
        out["fixed_mmpp"] = float(np.max(result["F_fixed_mmpp"]))
    out["adaptive_count"] = float(np.max(result["F_adaptive_count"]))
    if result.get("F_fixed_count_mmpp") is not None:
        out["fixed_count_mmpp"] = float(np.max(result["F_fixed_count_mmpp"]))
    out["adaptive_mmpp"] = float(np.max(result["F_adaptive_mmpp"]))
    return out


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
        target_fidelities = [r["target_fidelity"] for r in result["speedup_table"]]

    oh = float(overhead_us)
    out = []

    for F_star in target_fidelities:
        t_thr = time_for_fidelity_interp(
            result["T_threshold"], result["F_threshold"], F_star
        )
        t_ada = time_for_fidelity_interp(
            result["T_adaptive_mmpp"], result["F_adaptive_mmpp"], F_star
        )
        out.append(
            {
                "target_fidelity": float(F_star),
                "overhead_us": oh,
                "speedup_no_overhead": (
                    t_thr / t_ada
                    if np.isfinite(t_thr) and np.isfinite(t_ada) and t_ada > 0
                    else np.nan
                ),
                "speedup_with_overhead": (
                    (t_thr + oh) / (t_ada + oh)
                    if np.isfinite(t_thr) and np.isfinite(t_ada)
                    else np.nan
                ),
            }
        )

    return out


# =============================================================================
# 7. The experiments
# =============================================================================

# Validated range of the Shields 594-nm rate fits.
#
# Shields et al., PRL 114, 136402 (2015) measured the 594-nm charge-state rates
# g_{0-,-0} and gamma_{0,-} "for powers between 0.875 and 15 uW", and it is
# those fits that `shields_2015_params` implements. Swept powers therefore stay
# inside [0.875, 15] uW: outside that interval the P^2/(1 + P/P_sat) forms are
# extrapolation, not measurement.
#
# For reference, the same paper used 594 nm at 820 nW for its long
# charge-verification readout (t_R = 2.24 ms) and 11 uW for a 900 ns probe
# pulse, so this interval brackets the powers actually used for 594-nm charge
# readout: sub-uW for slow high-fidelity verification, several uW to ~15 uW for
# fast readout.
SHIELDS_FIT_RANGE_UW = (0.875, 15.0)

# Reference operating point for the sweeps that do not sweep power.
BASE_POWER_UW = 5.437

POWERS_UW = [0.875, 2.0, 4.0, 8.0, 15.0]

# Gamma_-0 / Gamma_0-, i.e. ionization over recombination. Equivalent to
# p_bright = 1/(1 + ratio): 0.5 -> 0.67 bright, 100 -> 0.0099 bright. The base
# point sits at about 7, near the middle of this grid.
SWITCHING_RATIOS = [0.5, 1.88, 7.07, 26.6, 100.0]

# (lambda_- - lambda_0)/(lambda_- + lambda_0). The base point is about 0.96.
CONTRASTS = [0.50, 0.70, 0.85, 0.95, 0.99]

# Detected fraction of emitted photons.
EFFICIENCIES = [0.02, 0.05, 0.15, 0.40, 1.00]

SWEEP_COLORS = ["#08306b", "#2171b5", "#6baed6", "#fd8d3c", "#a50f15"]


def check_power(power_uw: float) -> None:
    lo, hi = SHIELDS_FIT_RANGE_UW
    if not (lo - 1e-9 <= power_uw <= hi + 1e-9):
        raise ValueError(
            f"P = {power_uw} uW lies outside the range [{lo}, {hi}] uW over "
            "which the Shields 594-nm rate fits were measured. Extrapolating "
            "the P^2/(1+P/P_sat) forms beyond that interval is not supported."
        )


def params_with_switching_ratio(
    base: MMPPParams,
    ratio: float,
) -> MMPPParams:
    """
    Re-split the switching rates at a given Gamma_-0 / Gamma_0- while holding
    Gamma_tot fixed.

    Holding the total fixed is what makes this a clean one-parameter sweep: the
    number of blinks per readout window, Gamma_tot * t_R, is unchanged, so only
    the bright/dark ASYMMETRY varies. Changing the total as well would confound
    the asymmetry with the blink rate.
    """
    r = float(ratio)
    if not (r > 0.0):
        raise ValueError("ratio must be positive.")

    gamma_tot = base.gamma_minus_to_zero_khz + base.gamma_zero_to_minus_khz
    return _replace_params(
        base,
        gamma_minus_to_zero_khz=gamma_tot * r / (1.0 + r),
        gamma_zero_to_minus_khz=gamma_tot / (1.0 + r),
    )


def params_with_contrast(base: MMPPParams, contrast: float) -> MMPPParams:
    """
    Set the emission contrast C = (lambda_- - lambda_0)/(lambda_- + lambda_0)
    by adjusting the DARK rate only:

        lambda_0 = lambda_- (1 - C) / (1 + C).

    lambda_- is held fixed so the bright-state photon budget per dwell,
    lambda_- / Gamma_-0, does not move; only the dark-state leakage varies.
    Rescaling both rates instead would change the sparsity parameter at the
    same time and confound the two effects.
    """
    c = float(contrast)
    if not (0.0 < c <= 1.0):
        raise ValueError("contrast must lie in (0, 1].")

    return _replace_params(
        base,
        lambda_zero_khz=base.lambda_minus_khz * (1.0 - c) / (1.0 + c),
    )


DEMO_POINTS = [
    dict(
        name="high_flux",
        label="High flux (many photons per blink)",
        power_uw=15.0,
        detection_efficiency=1.0,
        horizon_us=100.0,
    ),
    dict(
        name="moderate",
        label="Moderate flux",
        power_uw=BASE_POWER_UW,
        detection_efficiency=1.0,
        horizon_us=250.0,
    ),
    dict(
        name="sparse",
        label="Sparse photons (eta = 0.10)",
        power_uw=BASE_POWER_UW,
        detection_efficiency=0.10,
        horizon_us=800.0,
    ),
]


def _base_params() -> MMPPParams:
    """Reference point the ratio, contrast and efficiency sweeps perturb."""
    return shields_2015_params(BASE_POWER_UW)


def _demo_points(cfg: RunConfig) -> list[OperatingPoint]:
    pts = []
    for i, p in enumerate(DEMO_POINTS):
        params = apply_detection_efficiency(
            shields_2015_params(p["power_uw"]),
            p["detection_efficiency"],
            model=cfg.efficiency_model,
        )
        pts.append(
            OperatingPoint(
                name=p["name"],
                label=p["label"],
                params=params,
                sweep_value=regime_summary(params)["photons_per_bright_dwell"],
                horizon_us=p["horizon_us"],
                power_uw=p["power_uw"],
                detection_efficiency=p["detection_efficiency"],
                seed_offset=1000 * i,
            )
        )
    return pts


def _power_points(cfg: RunConfig) -> list[OperatingPoint]:
    pts = []
    for P in POWERS_UW:
        check_power(P)
        params = shields_2015_params(P)
        pts.append(
            OperatingPoint(
                name=f"P{P:.3f}",
                label=f"{P:g} uW",
                params=params,
                sweep_value=float(P),
                power_uw=float(P),
                detection_efficiency=1.0,
            )
        )
    return pts


def _ratio_points(cfg: RunConfig) -> list[OperatingPoint]:
    base = _base_params()
    pts = []
    for r in SWITCHING_RATIOS:
        params = params_with_switching_ratio(base, r)
        pts.append(
            OperatingPoint(
                name=f"ratio{r:g}",
                label=f"$\\Gamma_{{-0}}/\\Gamma_{{0-}}$ = {r:g}",
                params=params,
                sweep_value=float(r),
                power_uw=BASE_POWER_UW,
                detection_efficiency=1.0,
            )
        )
    return pts


def _contrast_points(cfg: RunConfig) -> list[OperatingPoint]:
    base = _base_params()
    pts = []
    for c in CONTRASTS:
        params = params_with_contrast(base, c)
        pts.append(
            OperatingPoint(
                name=f"contrast{c:g}",
                label=f"C = {c:g}",
                params=params,
                sweep_value=float(c),
                power_uw=BASE_POWER_UW,
                detection_efficiency=1.0,
            )
        )
    return pts


def _efficiency_points(cfg: RunConfig) -> list[OperatingPoint]:
    base = _base_params()
    pts = []
    for eta in EFFICIENCIES:
        params = apply_detection_efficiency(base, eta, model=cfg.efficiency_model)
        pts.append(
            OperatingPoint(
                name=f"eta{eta:g}",
                label=f"$\\eta$ = {eta:g}",
                params=params,
                sweep_value=float(eta),
                power_uw=BASE_POWER_UW,
                detection_efficiency=float(eta),
            )
        )
    return pts


@dataclass(frozen=True)
class SweepSpec:
    key: str
    title: str
    xlabel: str
    legend_title: str
    build_points: Callable[[RunConfig], list[OperatingPoint]]
    xscale: str = "log"
    include_fixed_mmpp: bool = False
    fixed_targets: tuple[float, ...] = (0.70, 0.80, 0.90, 0.95)
    note: str = ""


EXPERIMENTS: dict[str, SweepSpec] = {
    "demo": SweepSpec(
        key="demo",
        title="Headline demonstration at three operating points",
        xlabel="photons per bright dwell",
        legend_title="operating point",
        build_points=_demo_points,
        include_fixed_mmpp=True,
        note=(
            "Three regimes described by their dimensionless sparsity rather "
            "than by laser power alone. This is the only experiment that also "
            "runs the fixed-time MMPP baseline and the paired McNemar test."
        ),
    ),
    "power": SweepSpec(
        key="power",
        title="594 nm laser-power sweep",
        xlabel="594 nm readout power (uW)",
        legend_title="594 nm power",
        build_points=_power_points,
        note=(
            "Powers stay inside the [0.875, 15] uW range over which the "
            "Shields et al. (2015) rate fits were measured."
        ),
    ),
    "ratio": SweepSpec(
        key="ratio",
        title="Switching-rate ratio sweep at fixed Gamma_tot",
        xlabel=r"$\Gamma_{-0}\,/\,\Gamma_{0-}$  (ionization / recombination)",
        legend_title="rate ratio",
        build_points=_ratio_points,
        note=(
            "Gamma_tot is held fixed, so blinks per window are constant and "
            "only the bright/dark asymmetry (p_bright = 1/(1+ratio)) varies."
        ),
    ),
    "contrast": SweepSpec(
        key="contrast",
        title="Emission-contrast sweep at fixed bright rate",
        xlabel=r"contrast $(\lambda_- - \lambda_0)/(\lambda_- + \lambda_0)$",
        legend_title="contrast",
        build_points=_contrast_points,
        xscale="linear",
        note=(
            "lambda_- is held fixed and lambda_0 is raised, so photons per "
            "bright dwell stay constant while the two states become harder "
            "to tell apart."
        ),
    ),
    "efficiency": SweepSpec(
        key="efficiency",
        title="Detection-efficiency sweep",
        xlabel=r"detection efficiency $\eta$",
        legend_title="efficiency",
        build_points=_efficiency_points,
        note=(
            "eta thins both emission rates, so contrast is unchanged but the "
            "sparsity parameter lambda_-/Gamma_-0 falls with eta. This is the "
            "sweep that shows where the event-time statistic stops paying."
        ),
    ),
}


# =============================================================================
# 8. Storage
# =============================================================================

DEFAULT_OUT_ROOT = Path("out_master")


def run_directory(
    spec: SweepSpec,
    out_root: Path,
    detector: DetectorModel,
) -> Path:
    """One directory per (experiment, detector setting) so runs never collide."""
    d = Path(out_root) / spec.key / detector.tag()
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_result(result: dict, index: int, directory: Path) -> Path:
    # Stored so later commands can address a point by its SWEEP position even
    # when only some points have been run.
    result["point_index"] = int(index)
    fp = directory / f"point_{index:02d}_{result['name']}.pkl"
    with open(fp, "wb") as f:
        pickle.dump(result, f)
    return fp


def load_results(
    spec: SweepSpec,
    out_root: Path,
    detector: DetectorModel,
    cfg: RunConfig | None = None,
) -> list[dict]:
    """Load whatever points have been run, in sweep order."""
    directory = run_directory(spec, out_root, detector)
    points = spec.build_points(cfg if cfg is not None else RunConfig(verbose=False))

    out = []
    for i, p in enumerate(points):
        fp = directory / f"point_{i:02d}_{p.name}.pkl"
        if fp.exists():
            with open(fp, "rb") as f:
                out.append(pickle.load(f))
    return out


# =============================================================================
# 9. Plots: the swept variable is the legend; the method is the line style
# =============================================================================

_METHOD_STYLES = {
    "threshold": dict(ls=":", lw=1.5, marker="o", ms=3.0),
    "count": dict(ls="--", lw=1.5, marker="^", ms=3.5),
    "fixed_count_mmpp": dict(ls="-.", lw=1.6, marker="d", ms=3.2),
    "mmpp": dict(ls="-", lw=2.4, marker=None),
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_sweep(
    results: list[dict],
    spec: SweepSpec,
    save_path: str | None = None,
):
    """Fidelity-vs-time plus both speedup curves, one colour per sweep point."""
    plt = _pyplot()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 4, figsize=(24.0, 5.4))

    # ---- panel (a): fidelity vs mean run time ------------------------------
    ax = axes[0]
    for k, res in enumerate(results):
        c = SWEEP_COLORS[k % len(SWEEP_COLORS)]
        ax.plot(
            res["T_threshold"], res["F_threshold"], color=c,
            **_METHOD_STYLES["threshold"],
        )
        fc = res["frontier_adaptive_count"]
        ax.plot(
            np.asarray(res["T_adaptive_count"])[fc],
            np.asarray(res["F_adaptive_count"])[fc],
            color=c,
            **_METHOD_STYLES["count"],
        )
        if res.get("F_fixed_count_mmpp") is not None:
            ff = res["frontier_fixed_count_mmpp"]
            ax.plot(
                np.asarray(res["T_fixed_count_mmpp"])[ff],
                np.asarray(res["F_fixed_count_mmpp"])[ff],
                color=c,
                **_METHOD_STYLES["fixed_count_mmpp"],
            )
        fa = res["frontier_adaptive_mmpp"]
        ax.plot(
            np.asarray(res["T_adaptive_mmpp"])[fa],
            np.asarray(res["F_adaptive_mmpp"])[fa],
            color=c,
            **_METHOD_STYLES["mmpp"],
        )

    ax.set_xscale("log")
    ax.set_xlabel("mean run time per shot (us)")
    ax.set_ylabel("balanced initial-state fidelity")
    ax.set_title("(a) fidelity vs run time", fontsize=11)
    ax.grid(alpha=0.25)

    value_handles = [
        Line2D(
            [], [],
            color=SWEEP_COLORS[k % len(SWEEP_COLORS)],
            lw=2.4,
            label=res["label"],
        )
        for k, res in enumerate(results)
    ]
    method_handles = [
        Line2D(
            [], [], color="0.35", label="fixed-time threshold",
            **_METHOD_STYLES["threshold"],
        ),
        Line2D(
            [], [], color="0.35", label="adaptive count SPRT",
            **_METHOD_STYLES["count"],
        ),
        Line2D(
            [], [], color="0.35", label="fixed-count MMPP",
            **_METHOD_STYLES["fixed_count_mmpp"],
        ),
        Line2D(
            [], [], color="0.35", label="adaptive MMPP SPRT",
            **_METHOD_STYLES["mmpp"],
        ),
    ]
    leg1 = ax.legend(
        handles=value_handles, title=spec.legend_title, fontsize=9,
        title_fontsize=9, loc="lower right",
    )
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, fontsize=8.5, loc="upper left")

    # ---- panel (b): adaptive MMPP speedup ----------------------------------
    ax = axes[1]
    for k, res in enumerate(results):
        c = SWEEP_COLORS[k % len(SWEEP_COLORS)]
        tbl = res["speedup_table"]
        F = np.array([r["target_fidelity"] for r in tbl])
        S = np.array([r["speedup_mmpp"] for r in tbl])
        lo = np.array([r["speedup_mmpp_ci_low"] for r in tbl])
        hi = np.array([r["speedup_mmpp_ci_high"] for r in tbl])
        m = np.isfinite(S)
        ax.plot(F[m], S[m], "-o", ms=3.5, color=c, lw=1.8, label=res["label"])
        g = m & np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(F[g], lo[g], hi[g], color=c, alpha=0.13)

    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axhline(2.0, color="0.5", lw=0.9, ls="--")
    ax.annotate(
        "D'Anjou bound, decay readout",
        xy=(0.985, 2.0), xycoords=("axes fraction", "data"),
        xytext=(0, 4), textcoords="offset points",
        fontsize=8, color="0.4", ha="right", va="bottom",
    )
    ax.set_xlabel("target balanced fidelity")
    ax.set_ylabel("run-time reduction vs fixed-time threshold")
    ax.set_title("(b) adaptive MMPP speedup", fontsize=11)
    ax.legend(title=spec.legend_title, fontsize=9, title_fontsize=9)
    ax.grid(alpha=0.25)

    # ---- panel (c): fixed-count MMPP speedup -------------------------------
    ax = axes[2]
    for k, res in enumerate(results):
        c = SWEEP_COLORS[k % len(SWEEP_COLORS)]
        tbl = res["speedup_table"]
        F = np.array([r["target_fidelity"] for r in tbl])
        S = np.array([r.get("speedup_fixed_count_mmpp", np.nan) for r in tbl])
        lo = np.array(
            [r.get("speedup_fixed_count_mmpp_ci_low", np.nan) for r in tbl]
        )
        hi = np.array(
            [r.get("speedup_fixed_count_mmpp_ci_high", np.nan) for r in tbl]
        )
        m = np.isfinite(S)
        ax.plot(F[m], S[m], "-.d", ms=3.6, color=c, lw=1.7, label=res["label"])
        g = m & np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(F[g], lo[g], hi[g], color=c, alpha=0.13)

    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("target balanced fidelity")
    ax.set_ylabel("run-time reduction vs fixed-time threshold")
    ax.set_title(
        "(c) fixed-count MMPP speedup (statistic alone)", fontsize=11
    )
    ax.legend(title=spec.legend_title, fontsize=9, title_fontsize=9)
    ax.grid(alpha=0.25)

    # ---- panel (d): adaptive count speedup ---------------------------------
    ax = axes[3]
    for k, res in enumerate(results):
        c = SWEEP_COLORS[k % len(SWEEP_COLORS)]
        tbl = res["speedup_table"]
        F = np.array([r["target_fidelity"] for r in tbl])
        S = np.array([r["speedup_count"] for r in tbl])
        m = np.isfinite(S)
        ax.plot(F[m], S[m], "--^", ms=4.0, color=c, lw=1.6, label=res["label"])

    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("target balanced fidelity")
    ax.set_ylabel("run-time reduction vs fixed-time threshold")
    ax.set_title("(d) adaptive count speedup (stopping alone)", fontsize=11)
    ax.legend(title=spec.legend_title, fontsize=9, title_fontsize=9)
    ax.grid(alpha=0.25)

    ylim = max(ax_.get_ylim()[1] for ax_ in axes[1:])
    for ax_ in axes[1:]:
        ax_.set_ylim(0.6, ylim)

    det = results[0]["detector"] if results else DETECTOR_OFF
    fig.suptitle(f"{spec.title}  --  {det.describe()}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_sweep_vs_x(
    results: list[dict],
    spec: SweepSpec,
    save_path: str | None = None,
):
    """
    Companion view: speedup against the swept variable at fixed fidelity
    targets, plus each method's fidelity ceiling against the swept variable.
    """
    plt = _pyplot()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    x = np.array([r["sweep_value"] for r in results], dtype=float)

    ax = axes[0]
    for F_star, mk in zip(spec.fixed_targets, ["o", "s", "^", "D", "v"]):
        sp = []
        for res in results:
            hit = [
                r["speedup_mmpp"]
                for r in res["speedup_table"]
                if abs(r["target_fidelity"] - F_star) < 5e-3
            ]
            sp.append(hit[0] if hit else np.nan)
        sp = np.array(sp, dtype=float)
        m = np.isfinite(sp)
        if m.any():
            ax.plot(
                x[m], sp[m], f"-{mk}", lw=1.8, ms=5, label=f"F* = {F_star:.2f}"
            )

    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale(spec.xscale)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel("adaptive MMPP speedup")
    ax.set_title("speedup at fixed fidelity", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    series = [
        ("F_threshold", "fixed-time threshold", ":o"),
        ("F_adaptive_count", "adaptive count SPRT", "--^"),
        ("F_adaptive_mmpp", "adaptive MMPP SPRT", "-s"),
    ]
    if results and results[0].get("F_fixed_count_mmpp") is not None:
        series.insert(2, ("F_fixed_count_mmpp", "fixed-count MMPP", "-.d"))
    if results and results[0].get("F_fixed_mmpp") is not None:
        series.insert(1, ("F_fixed_mmpp", "fixed-time MMPP", "--*"))

    for key, lab, st in series:
        ceil = [
            float(np.max(r[key])) if r.get(key) is not None else np.nan
            for r in results
        ]
        ax.plot(x, ceil, st, lw=1.8, ms=5, label=lab)

    ax.set_xscale(spec.xscale)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel("best achievable balanced fidelity")
    ax.set_title("fidelity ceiling", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    det = results[0]["detector"] if results else DETECTOR_OFF
    fig.suptitle(f"{spec.title}  --  {det.describe()}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_operating_point(result: dict, save_path: str | None = None):
    """
    Single-point view:
      (a) fidelity vs run time for every method, with the horizontal time
          reduction at a matched fidelity marked;
      (b) speedup vs target fidelity with its bootstrap CI.
    """
    plt = _pyplot()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax = axes[0]
    ax.plot(
        result["T_threshold"], result["F_threshold"], "o-", ms=3.5, lw=1.6,
        color="#444444", label="fixed-time count threshold",
    )
    if result.get("F_fixed_mmpp") is not None:
        ax.plot(
            result["T_fixed_mmpp"], result["F_fixed_mmpp"], "s--", ms=3.5,
            lw=1.4, color="#1f77b4",
            label="fixed-time MMPP (calibrated cutoff)",
        )

    fc = result["frontier_adaptive_count"]
    ax.plot(
        np.asarray(result["T_adaptive_count"])[fc],
        np.asarray(result["F_adaptive_count"])[fc],
        "^-", ms=3.5, lw=1.4, color="#2ca02c", label="adaptive count SPRT",
    )

    if result.get("F_fixed_count_mmpp") is not None:
        ff = result["frontier_fixed_count_mmpp"]
        ax.plot(
            np.asarray(result["T_fixed_count_mmpp"])[ff],
            np.asarray(result["F_fixed_count_mmpp"])[ff],
            "d-.", ms=3.5, lw=1.6, color="#9467bd",
            label="fixed-count MMPP",
        )

    fa = result["frontier_adaptive_mmpp"]
    ax.plot(
        np.asarray(result["T_adaptive_mmpp"])[fa],
        np.asarray(result["F_adaptive_mmpp"])[fa],
        "-", lw=2.4, color="#d62728", label="adaptive MMPP SPRT (this work)",
    )

    # Mark the time reduction at the largest jointly achievable target.
    rows = [r for r in result["speedup_table"] if np.isfinite(r["speedup_mmpp"])]
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
            f"{r['speedup_mmpp']:.2f}x faster",
            color="#d62728", ha="center", fontsize=9.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("mean run time per shot (us)")
    ax.set_ylabel("balanced initial-state fidelity")
    reg = result["regime"]
    ax.set_title(
        f"{result['label']}\n"
        f"photons/bright dwell = {reg['photons_per_bright_dwell']:.2f}, "
        f"contrast = {reg['contrast']:.2f}, "
        f"ratio = {reg['switching_ratio']:.2f}",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    tbl = result["speedup_table"]
    F = np.array([r["target_fidelity"] for r in tbl])
    S = np.array([r["speedup_mmpp"] for r in tbl])
    lo = np.array([r["speedup_mmpp_ci_low"] for r in tbl])
    hi = np.array([r["speedup_mmpp_ci_high"] for r in tbl])
    S_cnt = np.array([r["speedup_count"] for r in tbl])

    ok = np.isfinite(S)
    ax.plot(F[ok], S[ok], "o-", color="#d62728", label="adaptive MMPP")
    good = np.isfinite(lo) & np.isfinite(hi) & ok
    ax.fill_between(
        F[good], lo[good], hi[good], color="#d62728", alpha=0.18,
        label="95% paired bootstrap CI",
    )
    S_fcm = np.array([r.get("speedup_fixed_count_mmpp", np.nan) for r in tbl])
    ok3 = np.isfinite(S_fcm)
    if ok3.any():
        ax.plot(
            F[ok3], S_fcm[ok3], "d-.", color="#9467bd",
            label="fixed-count MMPP (statistic only)",
        )
    ok2 = np.isfinite(S_cnt)
    ax.plot(
        F[ok2], S_cnt[ok2], "^--", color="#2ca02c",
        label="adaptive count (stopping only)",
    )
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axhline(
        2.0, color="#888888", lw=0.9, ls="--",
        label="D'Anjou bound for decay readout",
    )
    ax.set_xlabel("target balanced fidelity")
    ax.set_ylabel("run-time reduction vs fixed-time threshold")
    ax.set_title(
        f"speedup at matched fidelity -- {result['detector'].describe()}",
        fontsize=9.5,
    )
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def print_summary(results: list[dict], spec: SweepSpec) -> None:
    if not results:
        print(f"\nno saved runs for {spec.key}; run the experiment first")
        return

    print("\n" + "=" * 100)
    print(f"{spec.title.upper()}: run-time reduction at matched balanced fidelity")
    if spec.note:
        print(f"  {spec.note}")
    if results:
        print(f"  detector: {results[0]['detector'].describe()}")
        print(f"  physics layer: {results[0].get('physics_layer', '?')}")
    print("=" * 100)
    print(
        f"{'point':>14}{'ph/dwell':>10}{'F*':>7}{'t_thr(us)':>11}"
        f"{'t_count(us)':>12}{'t_fcm(us)':>11}{'T_mmpp(us)':>12}"
        f"{'sp_mmpp':>9}{'95% CI':>15}{'sp_fcm':>8}{'sp_count':>10}"
    )
    print("-" * 116)
    for res in results:
        ph = res["regime"]["photons_per_bright_dwell"]
        first = True
        for r in res["speedup_table"]:
            if not np.isfinite(r["speedup_mmpp"]):
                continue
            ci = (
                f"[{r['speedup_mmpp_ci_low']:.2f},{r['speedup_mmpp_ci_high']:.2f}]"
                if np.isfinite(r["speedup_mmpp_ci_low"])
                else "n/a"
            )
            sc = (
                f"{r['speedup_count']:10.2f}"
                if np.isfinite(r["speedup_count"])
                else f"{'-':>10}"
            )
            sf_v = r.get("speedup_fixed_count_mmpp", np.nan)
            sf = f"{sf_v:8.2f}" if np.isfinite(sf_v) else f"{'-':>8}"
            t_fcm = r.get("t_fixed_count_mmpp_us", np.nan)
            tf = f"{t_fcm:11.2f}" if np.isfinite(t_fcm) else f"{'-':>11}"
            print(
                f"{res['name'] if first else '':>14}"
                f"{f'{ph:.1f}' if first else '':>10}"
                f"{r['target_fidelity']:7.2f}{r['t_threshold_us']:11.2f}"
                f"{r['t_adaptive_count_us']:12.2f}{tf}"
                f"{r['t_adaptive_mmpp_us']:12.2f}"
                f"{r['speedup_mmpp']:9.2f}{ci:>15}{sf}{sc}"
            )
            first = False
        if first:
            print(
                f"{res['name']:>14}{ph:10.1f}"
                "   (no target fidelity reachable by both methods)"
            )
        print("-" * 116)

    print("\nfidelity ceiling by method")
    keys = list(max_fidelity_summary(results[0]).keys())
    print(f"{'point':>14}" + "".join(f"{k:>18}" for k in keys))
    for res in results:
        s = max_fidelity_summary(res)
        print(f"{res['name']:>14}" + "".join(f"{s[k]:18.4f}" for k in keys))

    print("\nend-to-end speedup including per-shot overhead (F* = 0.85)")
    for res in results:
        line = []
        for oh in (0.0, 5.0, 20.0, 100.0):
            rows = speedup_with_overhead(res, oh, [0.85])
            v = rows[0]["speedup_with_overhead"]
            line.append(f"{oh:g} us: {v:.2f}x" if np.isfinite(v) else f"{oh:g} us: n/a")
        print(f"{res['name']:>14}   " + ",  ".join(line))


# =============================================================================
# 10. Polyspectra coupling: correct posterior coordinates and covariance
# =============================================================================


def polyspectra_laplace_covariance(
    least_squares_result,
    clip_condition: float = 1e12,
) -> np.ndarray:
    """
    Single-dataset Laplace covariance from a polyspectra least_squares fit.

    The polyspectra script estimates rate uncertainties from the scatter across
    random seeds, which requires repeating the whole experiment and is
    therefore unavailable in the lab. With error-weighted residuals
    r_i = (y_i - model_i)/sigma_i, the Gauss-Newton covariance at the optimum is

        Cov ~ (J^T J)^-1

    in FIT coordinates, i.e. (log K, logit p_bright, log beta^2). This is
    obtainable from a single dataset.

    Caveat to check, not assume: polyspectral estimates at neighbouring
    frequencies are correlated, so the independent-residual assumption makes
    this an UNDERestimate of the true sampling covariance. Validate it against
    the across-seed scatter you already compute before trusting it online.
    `compare_laplace_to_seed_scatter` does exactly that.
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

    `seed_samples` has shape (n_seeds, n_params) in the SAME fit coordinates as
    `laplace_cov`. A ratio near 1 licenses using the Laplace covariance online;
    a ratio well below 1 means the residual correlations matter and the
    covariance should be inflated by the measured factor.
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
    p = (
        1.0 / (1.0 + np.exp(-logit_p))
        if logit_p >= 0
        else np.exp(logit_p) / (1.0 + np.exp(logit_p))
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

    `make_bayesian_switching_ensemble` uses a product of INDEPENDENT
    log-normals on Gamma_-0 and Gamma_0-. That is inconsistent with how the
    rates are actually measured: the polyspectra fit is parameterized in
    (log K, logit p_bright) precisely because the likelihood is near-diagonal
    there, which means the posterior is strongly CORRELATED in
    (Gamma_-0, Gamma_0-). An independent product therefore puts quadrature
    weight along the directions polyspectra constrains worst and starves the
    directions it constrains best.

    Building the product rule in (log K, logit p_bright) and mapping through to
    rates fixes this, and the ensemble inherits the fit's correlation structure
    automatically via the Cholesky factor of `cov_fit_coords`.

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
            g_m0, g_0m = _fit_coords_to_rates(log_k0 + d[0], logit_p0 + d[1])
            try:
                p = _replace_params(
                    nominal_params,
                    gamma_minus_to_zero_khz=g_m0,
                    gamma_zero_to_minus_khz=g_0m,
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
            p = _replace_params(
                nominal_params,
                gamma_minus_to_zero_khz=g_m0,
                gamma_zero_to_minus_khz=g_0m,
            )
            p.validate()
            out.append(p)
        except ValueError:
            continue

    return out


def run_parameter_robustness(
    result: dict,
    cfg: RunConfig | None = None,
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
    cfg = cfg if cfg is not None else RunConfig()

    params = result["params"]
    horizon_us = result["horizon_us"]
    labels = result["test_labels"]
    detector = result.get("detector", DETECTOR_OFF)

    # Diagonal covariance of the stated width in fit coordinates.
    sd = float(np.log1p(rate_cv))
    C = np.diag([sd**2, sd**2])

    draws = sample_switching_posterior(params, C, n_draws, seed=seed)

    # Regenerate the same test shots from the true parameters.
    n_per_state = int(len(labels) // 2)
    shots, lab = simulate_balanced_dataset(
        n_per_state, horizon_us / 1000.0, params, result["seed"] + 7717, detector
    )

    t_thr = time_for_fidelity_interp(
        result["T_threshold"], result["F_threshold"], target_fidelity
    )

    widths = np.asarray(cfg.boundary_widths, dtype=float)
    offsets = np.asarray(cfg.offsets, dtype=float)
    deadlines = np.asarray(result["deadlines_us"], dtype=float)

    speedups, fidelities = [], []

    for p_used in draws:
        # The filter's own rates also get the detector correction, since a
        # mis-fit posterior draw is a separate error source from the detector.
        p_filter = _filter_params_for(p_used, detector)
        spec = build_no_click_spectral(p_filter)
        packed = pack_records(
            build_records(shots, horizon_us, p_filter, spec), spec
        )

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

        t_ada = time_for_fidelity_interp(T_arr, F_arr, target_fidelity)
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
            f"\nparameter robustness for {result['name']} "
            f"(CV = {rate_cv:.0%}, F* = {target_fidelity:.3f}, "
            f"{len(draws)} draws)"
        )
        print(
            f"  speedup median {out['speedup_median']:.2f}, "
            f"range [{out['speedup_min']:.2f}, {out['speedup_max']:.2f}], "
            f"draws failing to reach F*: {out['n_failed']}"
        )

    return out


# =============================================================================
# 11. Validation suite
# =============================================================================


def validate_all(verbose: bool = True) -> int:
    """
    Check every numerical claim this file relies on against an independent
    reference. Run after any change to the filter internals or the detector
    model. Returns the number of failures.
    """
    from scipy.linalg import expm
    from scipy.optimize import least_squares

    results: list[tuple[str, bool]] = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok)))
        if verbose:
            tag = "PASS" if ok else "FAIL"
            print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))

    # -- 1. closed-form 2x2 no-click propagator vs scipy.linalg.expm ---------
    params = apply_detection_efficiency(
        shields_2015_params(15.0), 1.0, "signal_only"
    )
    spec = build_no_click_spectral(params)
    A = params.no_click_generator

    worst = 0.0
    for dt in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 5.0]:
        err = np.abs(
            expm(A * dt) - np.exp(spec.mu1 * dt) * spec.scaled_propagator(dt)
        )
        worst = max(worst, float(err.max()))
    check("closed-form propagator matches expm", worst < 1e-12, f"max err {worst:.2e}")
    check(
        "no-click generator has real eigenvalues",
        not spec.degenerate and spec.delta > 0,
        f"mu1={spec.mu1:.3f} mu2={spec.mu2:.3f}",
    )

    # -- 2. event-time LLR matches the dense-expm reference -------------------
    rng = np.random.default_rng(3)
    T_MS = 0.05
    shots = [
        simulate_mmpp_shot(T_MS, s % 2, params, rng) for s in range(40)
    ]
    recs = build_records(shots, T_MS * 1000.0, params)

    grid = np.linspace(1e-5, T_MS, 8)
    worst = 0.0
    for sh, r in zip(shots, recs):
        ref = initial_state_llr_at_times(sh[sh < T_MS], grid, params)
        for t, v in zip(grid, ref):
            k = min(
                int(np.searchsorted(r.t_end, t, side="left")), r.t_start.size - 1
            )
            al, be = _interval_llr_coefficients(r.H_start[k], spec)
            mine = _llr_after(
                al, be, float(r.llr_start[k]), t - float(r.t_start[k]), spec
            )
            worst = max(worst, abs(mine - v))
    check("LLR matches dense-expm reference", worst < 1e-10, f"max err {worst:.2e}")

    # -- 3. structural facts the exact first-passage scan depends on ----------
    mono_ok, jump_ok = True, True
    for r in recs:
        if np.any(r.llr_end > r.llr_start + 1e-9):
            mono_ok = False
        if r.n_clicks and np.any(r.llr_post < r.llr_end[: r.n_clicks] - 1e-9):
            jump_ok = False
    check("LLR decreases monotonically during no-click intervals", mono_ok)
    check("LLR jumps upward at every photon arrival", jump_ok)

    # -- 4. vectorized SPRT reproduces the scalar reference exactly -----------
    packed = pack_records(recs, spec)
    bad, worst_dt = 0, 0.0
    for L in [0.3, 1.0, 2.5, 6.0]:
        for off in [-0.6, -0.2, 0.0, 0.4]:
            if abs(off) >= L:
                continue
            for dl in [10.0, 25.0, 50.0]:
                tv, pv = run_sprt(packed, L, off, dl)
                for i, r in enumerate(recs):
                    ts, pd = first_passage(
                        r, L, off, spec, deadline_ms=dl / 1000.0
                    )
                    bad += int(pd != pv[i])
                    worst_dt = max(worst_dt, abs(ts * 1000.0 - tv[i]))
    check(
        "vectorized SPRT == scalar first_passage",
        bad == 0 and worst_dt < 1e-9,
        f"{bad} mismatches, max dt {worst_dt:.2e} us",
    )

    # -- 5. closed-form lower crossing lands exactly on the boundary ----------
    L, off, dl = 1.5, -0.2, 50.0
    D = off - L
    tv, pv = run_sprt(packed, L, off, dl)
    worst = 0.0
    for i, r in enumerate(recs):
        if pv[i] != 1 or tv[i] >= dl - 1e-9:
            continue
        t_ms = tv[i] / 1000.0
        k = min(int((r.t_start < t_ms).sum()) - 1, r.t_start.size - 1)
        al, be = _interval_llr_coefficients(r.H_start[k], spec)
        llr_at_stop = _llr_after(
            al, be, float(r.llr_start[k]), t_ms - float(r.t_start[k]), spec
        )
        worst = max(worst, abs(llr_at_stop - D))
    check("lower crossing time is exact", worst < 1e-8, f"max |LLR-D| {worst:.2e}")

    # -- 6. cutoff optimizer is self-consistent under heavy ties --------------
    rng = np.random.default_rng(0)
    lab = np.array([0] * 300 + [1] * 300)
    stat = np.where(rng.random(600) < 0.5, -1.234, rng.normal(1.0, 0.5, 600))
    cut, F_rep = optimize_scalar_cutoff(stat, lab)
    F_app = balanced_fidelity(lab, np.where(stat >= cut, 0, 1))
    check(
        "cutoff optimizer consistent with heavy ties",
        abs(F_rep - F_app) < 1e-12,
        f"reported {F_rep:.4f}, applied {F_app:.4f}",
    )

    stat2 = rng.normal(0, 1, 600) + np.where(lab == 0, 0.8, 0.0)
    cut2, F2 = optimize_scalar_cutoff(stat2, lab)
    check(
        "cutoff optimizer consistent without ties",
        abs(F2 - balanced_fidelity(lab, np.where(stat2 >= cut2, 0, 1))) < 1e-12,
    )

    # -- 7. Laplace covariance from a least_squares fit -----------------------
    truth = np.array([np.log(9.42), np.log(0.118 / 0.882)])
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    X[:, 1] += 0.8 * X[:, 0]
    sigma = 0.35
    y = X @ truth + rng.normal(0, sigma, 60)
    fit = least_squares(lambda p: (y - X @ p) / sigma, truth + 0.05)

    C = polyspectra_laplace_covariance(fit)
    analytic = np.sqrt(np.diag(np.linalg.inv(X.T @ X / sigma**2)))
    check(
        "Laplace covariance matches analytic (J'J)^-1",
        np.allclose(np.sqrt(np.diag(C)), analytic, rtol=1e-8),
        f"sd {np.sqrt(np.diag(C)).round(4)}",
    )

    samples = np.array(
        [
            np.linalg.lstsq(
                X,
                X @ truth + np.random.default_rng(100 + k).normal(0, sigma, 60),
                rcond=None,
            )[0]
            for k in range(400)
        ]
    )
    chk = compare_laplace_to_seed_scatter(C, samples)
    check(
        "Laplace sd agrees with empirical sampling sd",
        np.all(np.abs(chk["sd_ratio"] - 1.0) < 0.1),
        f"ratio {chk['sd_ratio'].round(3)}",
    )

    # -- 8. ensemble in fit coordinates is correlated in rate space -----------
    tp = apply_detection_efficiency(
        shields_2015_params(BASE_POWER_UW), 1.0, "signal_only"
    )
    ens, w = make_correlated_switching_ensemble(tp, C, 3)
    g0 = np.array([p.gamma_minus_to_zero_khz for p in ens])
    g1 = np.array([p.gamma_zero_to_minus_khz for p in ens])
    corr_new = float(np.corrcoef(g0, g1)[0, 1])

    e_old, _ = make_bayesian_switching_ensemble(tp, 0.3, 0.3, 3)
    h0 = np.array([p.gamma_minus_to_zero_khz for p in e_old])
    h1 = np.array([p.gamma_zero_to_minus_khz for p in e_old])
    corr_old = float(np.corrcoef(h0, h1)[0, 1])

    check(
        "fit-coordinate ensemble is correlated in rate space",
        abs(corr_new) > 0.5 and abs(corr_old) < 1e-9,
        f"correlated {corr_new:+.3f} vs independent {corr_old:+.3f}",
    )
    check("ensemble weights normalized", abs(w.sum() - 1.0) < 1e-12)

    # -- 9. regime summary reproduces the dimensionless parameters ------------
    reg = regime_summary(tp, t_R_us=100.0)
    check(
        "regime summary consistent",
        abs(
            reg["photons_per_bright_dwell"]
            - tp.lambda_minus_khz / tp.gamma_minus_to_zero_khz
        )
        < 1e-12
        and 0.0 < reg["contrast"] < 1.0
        and abs(
            reg["switching_ratio"]
            - (1.0 - reg["p_bright_stationary"]) / reg["p_bright_stationary"]
        )
        < 1e-9,
        f"ph/dwell {reg['photons_per_bright_dwell']:.2f}, "
        f"Gamma_tot*t_R {reg['gamma_tot_t_R']:.3f}",
    )

    # -- 9b. fixed-count MMPP ------------------------------------------------
    clicks = click_time_matrix(packed)

    same_time, post_ok, pre_ok, dl_ok = True, True, True, True
    for n_up in (1, 2, 3, 5):
        for dl in (10.0, 25.0, 50.0):
            t_cnt, _ = run_adaptive_count(clicks, n_up, dl)
            t_fcm, llr = fixed_count_mmpp_statistic(packed, clicks, n_up, dl)

            # The two methods must stop at exactly the same instants; that
            # identity is what makes 2 -> 3 a controlled comparison.
            if not np.allclose(t_cnt, t_fcm, rtol=0, atol=0):
                same_time = False

            reached = packed.n_clicks >= n_up
            reached &= clicks[:, min(n_up - 1, clicks.shape[1] - 1)] <= dl / 1000.0

            # A shot that reached n_up must be scored with the POST-jump LLR of
            # that photon. Scoring the pre-jump limit would discard one photon
            # of evidence, and the two differ by construction.
            for i in np.nonzero(reached)[0][:25]:
                if abs(llr[i] - packed.llr_post[i, n_up - 1]) > 1e-12:
                    post_ok = False
                pre = llr_at_stop_times(packed, t_fcm)[i]
                if abs(pre - llr[i]) < 1e-12:
                    pre_ok = False   # must NOT coincide with the pre-jump value

            # A shot that timed out must be scored at the deadline.
            ref = llr_at_times(packed, np.array([dl]))[:, 0]
            idx = np.nonzero(~reached)[0]
            if idx.size and not np.allclose(llr[idx], ref[idx], atol=1e-12):
                dl_ok = False

    check("fixed-count MMPP stops exactly when adaptive count does", same_time)
    check("fixed-count MMPP scores reached shots post-jump", post_ok)
    check(
        "post-jump score differs from the pre-jump limit (photon not discarded)",
        pre_ok,
    )
    check("fixed-count MMPP scores timed-out shots at the deadline", dl_ok)

    # llr_at_stop_times must agree with llr_at_times on a constant time vector.
    t_const = 30.0
    a_stop = llr_at_stop_times(packed, np.full(packed.n_shots, t_const))
    a_grid = llr_at_times(packed, np.array([t_const]))[:, 0]
    check(
        "llr_at_stop_times == llr_at_times for a constant stop time",
        np.allclose(a_stop, a_grid, atol=1e-12),
        f"max diff {np.abs(a_stop - a_grid).max():.2e}",
    )

    # -- 10. the detector model ----------------------------------------------
    rng = np.random.default_rng(12345)

    ideal_arrivals = np.sort(rng.random(5000))
    same = apply_detector_response(ideal_arrivals, DETECTOR_OFF, rng)
    check(
        "disabled detector is the identity",
        same.size == ideal_arrivals.size
        and np.allclose(same, ideal_arrivals),
    )

    # A homogeneous Poisson stream isolates the detector from the MMPP.
    T_ms = 400.0
    lam_khz = 200.0
    n_arr = int(rng.poisson(lam_khz * T_ms))
    arrivals = np.sort(rng.random(n_arr) * T_ms)
    measured_lambda = arrivals.size / T_ms

    tau_ns = 1000.0
    det_dt = DetectorModel(
        enabled=True,
        dead_time_ns=tau_ns,
        paralyzable=False,
        afterpulse_probability=0.0,
    )
    rec = apply_detector_response(arrivals, det_dt, rng, t_max_ms=T_ms)
    gaps = np.diff(rec)
    expected = measured_lambda / (1.0 + measured_lambda * det_dt.dead_time_ms)
    got = rec.size / T_ms
    check(
        "non-paralyzable dead time enforces the blind window",
        gaps.size > 0 and gaps.min() >= det_dt.dead_time_ms - 1e-15,
        f"min gap {gaps.min() * 1e6:.1f} ns vs tau {tau_ns:g} ns",
    )
    check(
        "non-paralyzable recorded rate = lambda/(1+lambda tau)",
        abs(got / expected - 1.0) < 0.02,
        f"{got:.2f} vs {expected:.2f} kHz",
    )

    det_par = replace(det_dt, paralyzable=True)
    rec_par = apply_detector_response(arrivals, det_par, rng, t_max_ms=T_ms)
    expected_par = measured_lambda * np.exp(
        -measured_lambda * det_par.dead_time_ms
    )
    got_par = rec_par.size / T_ms
    check(
        "paralyzable recorded rate = lambda exp(-lambda tau)",
        abs(got_par / expected_par - 1.0) < 0.02,
        f"{got_par:.2f} vs {expected_par:.2f} kHz",
    )

    p_ap = 0.10
    det_ap = DetectorModel(
        enabled=True,
        dead_time_ns=0.0,
        afterpulse_probability=p_ap,
        afterpulse_time_constant_ns=200.0,
    )
    rec_ap = apply_detector_response(arrivals, det_ap, rng, t_max_ms=T_ms)
    got_ap = rec_ap.size / T_ms
    expected_ap = measured_lambda / (1.0 - p_ap)
    check(
        "afterpulsing inflates the rate by 1/(1-p)",
        abs(got_ap / expected_ap - 1.0) < 0.02,
        f"{got_ap:.2f} vs {expected_ap:.2f} kHz",
    )

    # The closed-form correction used to build the filter must agree with what
    # the simulator actually records, for both imperfections at once -- in both
    # regimes, afterpulses mostly swallowed by the dead window (tau_ap << tau_d)
    # and afterpulses mostly escaping it (tau_ap >> tau_d).
    probe = MMPPParams(
        gamma_minus_to_zero_khz=1.0,
        gamma_zero_to_minus_khz=1.0,
        lambda_minus_khz=measured_lambda,
        lambda_zero_khz=0.0,
    )
    for tau_ap_ns, regime in [(200.0, "tau_ap < tau_d"), (5000.0, "tau_ap > tau_d")]:
        det_both = DetectorModel(
            enabled=True,
            dead_time_ns=tau_ns,
            afterpulse_probability=p_ap,
            afterpulse_time_constant_ns=tau_ap_ns,
        )
        rec_both = apply_detector_response(
            arrivals, det_both, rng, t_max_ms=T_ms
        )
        predicted = effective_emission_rates(probe, det_both).lambda_minus_khz
        got_both = rec_both.size / T_ms
        check(
            f"effective_emission_rates predicts the recorded rate ({regime})",
            abs(got_both / predicted - 1.0) < 0.03,
            f"{got_both:.2f} vs {predicted:.2f} kHz",
        )

    # An enabled-but-trivial detector must not perturb the physics.
    det_trivial = DetectorModel(
        enabled=True, dead_time_ns=0.0, afterpulse_probability=0.0
    )
    r1 = simulate_mmpp_shot(
        1.0, 0, params, np.random.default_rng(77), DETECTOR_OFF
    )
    r2 = simulate_mmpp_shot(
        1.0, 0, params, np.random.default_rng(77), det_trivial
    )
    check(
        "enabled detector with zero imperfections changes nothing",
        r1.size == r2.size and np.allclose(r1, r2),
    )

    # -- 11. the sweep parameter builders ------------------------------------
    base = shields_2015_params(BASE_POWER_UW)
    gamma_tot_0 = base.gamma_minus_to_zero_khz + base.gamma_zero_to_minus_khz

    ratio_ok = True
    for r_target in SWITCHING_RATIOS:
        p = params_with_switching_ratio(base, r_target)
        reg_r = regime_summary(p)
        gt = p.gamma_minus_to_zero_khz + p.gamma_zero_to_minus_khz
        if (
            abs(reg_r["switching_ratio"] / r_target - 1.0) > 1e-9
            or abs(gt / gamma_tot_0 - 1.0) > 1e-12
        ):
            ratio_ok = False
    check(
        "ratio sweep hits the requested ratio at fixed Gamma_tot", ratio_ok
    )

    contrast_ok = True
    for c_target in CONTRASTS:
        p = params_with_contrast(base, c_target)
        reg_c = regime_summary(p)
        if (
            abs(reg_c["contrast"] - c_target) > 1e-12
            or abs(
                reg_c["photons_per_bright_dwell"]
                - regime_summary(base)["photons_per_bright_dwell"]
            )
            > 1e-12
        ):
            contrast_ok = False
    check(
        "contrast sweep hits the requested contrast at fixed sparsity",
        contrast_ok,
    )

    eff_ok = True
    for eta in EFFICIENCIES:
        p = apply_detection_efficiency(base, eta, "signal_only")
        reg_e = regime_summary(p)
        if (
            abs(p.lambda_minus_khz / (eta * base.lambda_minus_khz) - 1.0) > 1e-12
            or abs(reg_e["contrast"] - regime_summary(base)["contrast"]) > 1e-12
            or abs(reg_e["gamma_tot_khz"] / gamma_tot_0 - 1.0) > 1e-12
        ):
            eff_ok = False
    check(
        "efficiency sweep thins emission only, at fixed contrast", eff_ok
    )

    n_fail = sum(1 for _, ok in results if not ok)
    if verbose:
        print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
        print(f"physics layer: {PHYSICS_LAYER}")
    return n_fail


# =============================================================================
# 12. Command line
# =============================================================================


def detector_from_args(args: argparse.Namespace) -> DetectorModel:
    """
    Build the detector from --detector-preset plus any explicit overrides.

    The default is an IDEAL detector, so omitting every detector flag
    reproduces the original scripts exactly. The model is switched on by
    --detector (which selects the "realistic" preset), by naming any preset
    other than "off", or by setting any imperfection explicitly.
    """
    overrides = {}
    if args.dead_time_ns is not None:
        overrides["dead_time_ns"] = float(args.dead_time_ns)
    if args.afterpulse_prob is not None:
        overrides["afterpulse_probability"] = float(args.afterpulse_prob)
    if args.afterpulse_tau_ns is not None:
        overrides["afterpulse_time_constant_ns"] = float(args.afterpulse_tau_ns)
    if args.paralyzable:
        overrides["paralyzable"] = True

    preset = args.detector_preset
    enable = args.detector or bool(overrides) or preset != "off"

    if preset == "off" and enable:
        # --detector on its own means "a realistic detector".
        preset = "realistic"

    det = replace(DETECTOR_PRESETS[preset], **overrides, enabled=enable)

    # Only meaningful alongside an enabled detector, so it never enables one.
    if args.no_filter_correction:
        det = replace(det, correct_filter_rates=False)

    det.validate()
    return det


def config_from_args(args: argparse.Namespace, spec: SweepSpec) -> RunConfig:
    cfg = RunConfig(
        detector=detector_from_args(args),
        include_fixed_mmpp=spec.include_fixed_mmpp,
        verbose=not args.quiet,
    )
    if getattr(args, "quick", False):
        cfg = cfg.quick()
    if getattr(args, "seed", None) is not None:
        cfg = replace(cfg, seed=int(args.seed))
    if getattr(args, "n_cal", None) is not None:
        cfg = replace(cfg, n_cal=int(args.n_cal))
    if getattr(args, "n_test", None) is not None:
        cfg = replace(cfg, n_test=int(args.n_test))
    if getattr(args, "n_boot", None) is not None:
        cfg = replace(cfg, n_boot=int(args.n_boot))
    return cfg


def _resolve_spec(key: str) -> SweepSpec:
    if key not in EXPERIMENTS:
        raise SystemExit(
            f"unknown experiment {key!r}; choose from "
            f"{', '.join(EXPERIMENTS)}"
        )
    return EXPERIMENTS[key]


def _selected_indices(arg: str, n: int) -> list[int]:
    if arg in ("all", None, ""):
        return list(range(n))
    out = []
    for tok in str(arg).split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if not (0 <= i < n):
            raise SystemExit(f"point index {i} out of range 0..{n - 1}")
        out.append(i)
    return out


def cmd_list(args: argparse.Namespace) -> int:
    print(f"{MODULE_VERSION}   physics layer: {PHYSICS_LAYER}\n")
    print("experiments")
    for key, spec in EXPERIMENTS.items():
        pts = spec.build_points(RunConfig(verbose=False))
        print(f"\n  {key:<11} {spec.title}")
        if spec.note:
            print(f"              {spec.note}")
        for i, p in enumerate(pts):
            reg = regime_summary(p.params)
            print(
                f"              [{i}] {p.name:<14} "
                f"ph/dwell {reg['photons_per_bright_dwell']:7.2f}  "
                f"contrast {reg['contrast']:.3f}  "
                f"ratio {reg['switching_ratio']:7.2f}  "
                f"Gamma_tot {reg['gamma_tot_khz']:7.3f} kHz"
            )
    print("\ndetector presets")
    for name, det in DETECTOR_PRESETS.items():
        print(f"  {name:<12} {det.describe()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args.experiment)
    cfg = config_from_args(args, spec)
    points = spec.build_points(cfg)
    directory = run_directory(spec, Path(args.out), cfg.detector)

    idx = _selected_indices(args.point, len(points))

    if not args.quiet:
        print(f"{MODULE_VERSION}   physics layer: {PHYSICS_LAYER}")
        print(f"experiment: {spec.key} -- {spec.title}")
        print(f"detector:   {cfg.detector.describe()}")
        print(f"output:     {directory}")

    for i in idx:
        res = run_operating_point(points[i], cfg)
        fp = save_result(res, i, directory)
        if not args.quiet:
            print(f"  saved {fp}")

    if args.plot:
        return cmd_plot(args)
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args.experiment)
    cfg = config_from_args(args, spec)
    directory = run_directory(spec, Path(args.out), cfg.detector)
    results = load_results(spec, Path(args.out), cfg.detector, cfg)

    if not results:
        print(f"no saved runs in {directory}; run the experiment first")
        return 1

    print(f"loaded {len(results)} points from {directory}")

    f1 = plot_sweep(results, spec, save_path=str(directory / f"{spec.key}_sweep.png"))
    f2 = plot_sweep_vs_x(
        results, spec, save_path=str(directory / f"{spec.key}_vs_x.png")
    )
    del f1, f2

    for res in results:
        f = plot_operating_point(
            res, save_path=str(directory / f"point_{res['name']}.png")
        )
        del f

    print_summary(results, spec)
    print(f"figures written to {directory}")
    return 0


_SPEEDUP_CSV_COLUMNS = [
    "experiment",
    "detector",
    "point_index",
    "point",
    "sweep_value",
    "power_uw",
    "detection_efficiency",
    "photons_per_bright_dwell",
    "contrast",
    "switching_ratio",
    "p_bright_stationary",
    "gamma_tot_khz",
    "horizon_us",
    "target_fidelity",
    "t_threshold_us",
    "t_adaptive_count_us",
    "t_fixed_count_mmpp_us",
    "t_adaptive_mmpp_us",
    "speedup_mmpp",
    "speedup_mmpp_ci_low",
    "speedup_mmpp_ci_high",
    "speedup_count",
    "speedup_count_ci_low",
    "speedup_count_ci_high",
    "speedup_fixed_count_mmpp",
    "speedup_fixed_count_mmpp_ci_low",
    "speedup_fixed_count_mmpp_ci_high",
]

_CEILING_CSV_COLUMNS = [
    "experiment",
    "detector",
    "point_index",
    "point",
    "sweep_value",
    "method",
    "max_balanced_fidelity",
]


def export_csv(
    results: list[dict],
    spec: SweepSpec,
    directory: Path,
) -> list[Path]:
    """
    Write the speedup table and the fidelity ceilings as CSV.

    The pickles hold everything, but they are only readable from this module.
    These two files are the results in a form a plotting script, a spreadsheet
    or a paper table can consume directly.
    """
    written: list[Path] = []

    fp = directory / "speedup_table.csv"
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SPEEDUP_CSV_COLUMNS)
        w.writeheader()
        for res in results:
            reg = res["regime"]
            base = {
                "experiment": spec.key,
                "detector": res["detector"].tag(),
                "point_index": res.get("point_index"),
                "point": res["name"],
                "sweep_value": res["sweep_value"],
                "power_uw": res["power_uw"],
                "detection_efficiency": res["detection_efficiency"],
                "photons_per_bright_dwell": reg["photons_per_bright_dwell"],
                "contrast": reg["contrast"],
                "switching_ratio": reg["switching_ratio"],
                "p_bright_stationary": reg["p_bright_stationary"],
                "gamma_tot_khz": reg["gamma_tot_khz"],
                "horizon_us": res["horizon_us"],
            }
            for row in res["speedup_table"]:
                out = dict(base)
                for k in _SPEEDUP_CSV_COLUMNS:
                    if k in row:
                        out[k] = row[k]
                w.writerow(out)
    written.append(fp)

    fp = directory / "fidelity_ceilings.csv"
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CEILING_CSV_COLUMNS)
        w.writeheader()
        for res in results:
            for method, value in max_fidelity_summary(res).items():
                w.writerow(
                    {
                        "experiment": spec.key,
                        "detector": res["detector"].tag(),
                        "point_index": res.get("point_index"),
                        "point": res["name"],
                        "sweep_value": res["sweep_value"],
                        "method": method,
                        "max_balanced_fidelity": value,
                    }
                )
    written.append(fp)

    return written


def cmd_export(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args.experiment)
    cfg = config_from_args(args, spec)
    directory = run_directory(spec, Path(args.out), cfg.detector)
    results = load_results(spec, Path(args.out), cfg.detector, cfg)

    if not results:
        print(f"no saved runs in {directory}; run the experiment first")
        return 1

    for fp in export_csv(results, spec, directory):
        print(f"wrote {fp}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args.experiment)
    cfg = config_from_args(args, spec)
    results = load_results(spec, Path(args.out), cfg.detector, cfg)
    print_summary(results, spec)
    return 0 if results else 1


def cmd_robustness(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args.experiment)
    cfg = config_from_args(args, spec)
    results = load_results(spec, Path(args.out), cfg.detector, cfg)

    if not results:
        print("no saved runs; run the experiment first")
        return 1

    # Indices address the SWEEP position, exactly as for `run`, so they mean
    # the same thing whether or not every point has been run.
    n_points = len(spec.build_points(cfg))
    wanted = set(_selected_indices(args.point, n_points))
    selected = [r for r in results if r.get("point_index") in wanted]

    if not selected:
        print(
            f"none of the requested points have been run; available: "
            f"{sorted(r.get('point_index') for r in results)}"
        )
        return 1

    for res in selected:
        run_parameter_robustness(
            res,
            cfg=cfg,
            rate_cv=args.rate_cv,
            n_draws=args.n_draws,
            target_fidelity=args.target_fidelity,
        )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    return 1 if validate_all() else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adaptive_charge_state_master.py",
        description=(
            "Adaptive NV charge-state readout: fixed-time count threshold vs "
            "adaptive count SPRT vs adaptive MMPP SPRT, with an optional "
            "dead-time / afterpulsing detector model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="experiments: " + ", ".join(EXPERIMENTS),
    )
    p.add_argument("--version", action="version", version=MODULE_VERSION)

    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp, with_experiment=True):
        if with_experiment:
            sp.add_argument("experiment", help="which experiment to operate on")
        sp.add_argument(
            "--out", default=str(DEFAULT_OUT_ROOT),
            help="output root directory (default: %(default)s)",
        )
        sp.add_argument("--quiet", action="store_true", help="suppress per-point logs")
        # Detector switches. --detector alone enables the "realistic" preset.
        sp.add_argument(
            "--detector", action="store_true",
            help="enable the detector model (dead time + afterpulsing)",
        )
        sp.add_argument(
            "--detector-preset", default="off",
            choices=sorted(DETECTOR_PRESETS),
            help=(
                "detector settings to start from; anything other than 'off' "
                "enables the model (default: %(default)s)"
            ),
        )
        sp.add_argument("--dead-time-ns", type=float, default=None)
        sp.add_argument(
            "--paralyzable", action="store_true",
            help="extending (paralyzable) dead time instead of non-extending",
        )
        sp.add_argument("--afterpulse-prob", type=float, default=None)
        sp.add_argument("--afterpulse-tau-ns", type=float, default=None)
        sp.add_argument(
            "--no-filter-correction", action="store_true",
            help=(
                "build the MMPP filter from the TRUE emission rates rather "
                "than the dead-time/afterpulse-corrected ones, to measure "
                "the cost of pure model mismatch"
            ),
        )

    sp = sub.add_parser("list", help="show experiments, points and presets")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("run", help="simulate and analyze an experiment")
    add_common(sp)
    sp.add_argument(
        "--point", default="all",
        help="point index, comma-separated indices, or 'all' (default: all)",
    )
    sp.add_argument("--quick", action="store_true", help="small fast smoke run")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--n-cal", type=int, default=None)
    sp.add_argument("--n-test", type=int, default=None)
    sp.add_argument("--n-boot", type=int, default=None)
    sp.add_argument("--plot", action="store_true", help="plot when the run finishes")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("plot", help="figures + summary from saved runs")
    add_common(sp)
    sp.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_plot)

    sp = sub.add_parser("summary", help="text summary from saved runs")
    add_common(sp)
    sp.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_summary)

    sp = sub.add_parser("export", help="write saved runs out as CSV")
    add_common(sp)
    sp.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser(
        "robustness", help="does the speedup survive parameter error?"
    )
    add_common(sp)
    sp.add_argument("--point", default="all")
    sp.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    sp.add_argument("--rate-cv", type=float, default=0.30)
    sp.add_argument("--n-draws", type=int, default=12)
    sp.add_argument("--target-fidelity", type=float, default=0.85)
    sp.set_defaults(func=cmd_robustness)

    sp = sub.add_parser("validate", help="run the numerical validation suite")
    sp.set_defaults(func=cmd_validate)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
