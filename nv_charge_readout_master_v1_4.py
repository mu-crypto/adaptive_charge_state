from __future__ import annotations

"""
NV charge-readout benchmark suite
=================================

ONE FILE for the simulation studies we have been discussing.

Implemented studies
-------------------
1. "power_sweep"
   Reproduces the Shields-style power sweep:
   optimized total-count threshold vs exact event-time MMPP.

2. "readout_time_sweep"
   Surveys INITIAL-STATE readout fidelity versus total readout time at fixed
   laser power and detection efficiency. This is the version relevant to the
   current Shields Fig. 2(b) comparison.

3. "state_tracking_duration_sweep"
   Preserves the ORIGINAL binwise state-tracking readout-duration survey:
   fixed designated MMPP reporting bin, fixed/coarser threshold bin, and
   balanced fidelity against the known hidden state at each designated bin.

4. "sparse_efficiency"
   Holds the physical laser power / switching dynamics fixed and decreases
   the detected photon rates with a relative detection efficiency eta.

5. "switching_mismatch_1d"
   Generates data with the TRUE switching rates, but deliberately gives the
   MMPP incorrect switching rates during inference. The photon-emission rates
   remain correct. This tests how accurate the polyspectra-derived switching
   rates need to be for MMPP to remain useful.

6. "switching_mismatch_2d"
   Same idea, but independently varies the two switching rates and plots
   F_MMPP - F_threshold as a 2D robustness map.

7. "emission_mismatch_1d"
   Optional companion study. The switching rates remain correct while the
   photon-emission rates supplied to the MMPP are deliberately wrong.

8. "bayesian_mmpp"
   Compares:
       - total-count threshold
       - ordinary plug-in MMPP
       - Bayesian ensemble MMPP

   The Bayesian MMPP marginalizes the initial-state likelihood over an
   uncertainty distribution for the two switching rates. By default this
   uncertainty is represented by deterministic Gauss-Hermite quadrature over
   independent log-normal posteriors for Gamma_-0 and Gamma_0-.

   This is deliberately a FINITE ENSEMBLE implementation rather than online
   MCMC. The expensive posterior construction can happen offline; online
   inference is a weighted bank of small 2-state MMPP filters.

9. "hsmm_comparison"
   Compares:
       - total-count threshold
       - ordinary 2-state MMPP
       - event-time HSMM

   Two simulation cases are run:
       A. exponential dwell times
          Erlang shape k=1, where the HSMM MUST reduce to the ordinary MMPP;

       B. non-exponential dwell times
          Erlang/gamma dwell times with integer shape k>1, keeping the same
          mean dwell times as the MMPP switching rates.

   The HSMM is implemented exactly through a phase-type (Erlang) expansion.
   This turns each macro charge state into k sequential hidden phases while
   keeping the same photon emission rate in all phases of that charge state.
   Exact event-time filtering is then performed on the expanded CTMC.

10. "hsmm_dwell_shape_sweep"
    Sweeps the common Erlang dwell-shape parameter

        k = 1, 2, 3, ...

    for both NV- and NV0 while holding their mean dwell times fixed. At each
    k, simulated data are generated from the corresponding HSMM and compared
    with:
        - an optimized total-count threshold,
        - a mean-matched ordinary MMPP that still assumes exponential dwell
          times,
        - the correctly specified HSMM.

    The sweep reports both k and the dwell-time coefficient of variation

        CV(T) = 1 / sqrt(k),

    so k=1 is the exponential limit and larger k corresponds to increasingly
    non-exponential / more regular dwell times.

The important separation in the mismatch studies is:

    DATA GENERATION:
        theta_true = (Gamma_true, lambda_true)

    MMPP INFERENCE:
        theta_used = (Gamma_wrong, lambda_true)

or, for the emission study,

        theta_used = (Gamma_true, lambda_wrong)

The threshold classifier is calibrated directly from photon counts and does
not use Gamma explicitly.

For the mismatch studies the default reference readout time is the
THRESHOLD-OPTIMAL readout time. Every mismatched MMPP is evaluated on the
same test trajectories and at exactly the same t_R. This isolates the effect
of parameter error from changes in measurement duration.

State convention
----------------
0 = NV-  (bright)
1 = NV0  (dark)

All internal rates are kHz = ms^-1.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


SCRIPT_VERSION = "nv-charge-readout-master-v1.7-hsmm-shape-sweep"
SHIELDS_BACKGROUND_KHZ = 0.268


# =============================================================================
# Model
# =============================================================================


@dataclass(frozen=True)
class MMPPParams:
    gamma_minus_to_zero_khz: float
    gamma_zero_to_minus_khz: float
    lambda_minus_khz: float
    lambda_zero_khz: float

    def validate(self) -> None:
        values = np.asarray(
            [
                self.gamma_minus_to_zero_khz,
                self.gamma_zero_to_minus_khz,
                self.lambda_minus_khz,
                self.lambda_zero_khz,
            ],
            dtype=float,
        )

        if np.any(~np.isfinite(values)):
            raise ValueError("All rates must be finite.")
        if np.any(values < 0):
            raise ValueError("All rates must be non-negative.")
        if self.lambda_minus_khz <= self.lambda_zero_khz:
            raise ValueError("NV- must be brighter than NV0.")

    @property
    def Q(self) -> np.ndarray:
        """
        Hidden-state generator for COLUMN probability vectors:
            dp/dt = Q p
        """
        self.validate()

        gm0 = self.gamma_minus_to_zero_khz
        g0m = self.gamma_zero_to_minus_khz

        return np.array(
            [
                [-gm0, +g0m],
                [+gm0, -g0m],
            ],
            dtype=float,
        )

    @property
    def Lambda(self) -> np.ndarray:
        self.validate()
        return np.diag(
            [
                self.lambda_minus_khz,
                self.lambda_zero_khz,
            ]
        )

    @property
    def no_click_generator(self) -> np.ndarray:
        return self.Q - self.Lambda


def shields_2015_params(power_uw: float) -> MMPPParams:
    """
    594-nm charge-readout rate fits used throughout the project.
    """

    P = float(power_uw)

    if not np.isfinite(P) or P <= 0:
        raise ValueError("power_uw must be positive and finite.")

    # NV0 -> NV-
    gamma_zero_to_minus_khz = (
        (39.0 / 1000.0)
        * P**2
        / (1.0 + P / 134.0)
    )

    # NV- -> NV0
    gamma_minus_to_zero_khz = (
        (310.0 / 1000.0)
        * P**2
        / (1.0 + P / 53.2)
    )

    # NV0 detected photon rate
    lambda_zero_khz = (
        1.65 * P / (1.0 + P / 134.0)
        + SHIELDS_BACKGROUND_KHZ
    )

    # NV- detected photon rate
    lambda_minus_khz = (
        46.2 * P / (1.0 + P / 53.0)
        + SHIELDS_BACKGROUND_KHZ
    )

    params = MMPPParams(
        gamma_minus_to_zero_khz=gamma_minus_to_zero_khz,
        gamma_zero_to_minus_khz=gamma_zero_to_minus_khz,
        lambda_minus_khz=lambda_minus_khz,
        lambda_zero_khz=lambda_zero_khz,
    )
    params.validate()
    return params


def apply_detection_efficiency(
    base_params: MMPPParams,
    efficiency: float,
    model: str = "thin_all_counts",
    background_khz: float = SHIELDS_BACKGROUND_KHZ,
) -> MMPPParams:
    """
    Reduce DETECTED photon rates without changing physical switching rates.

    model="thin_all_counts":
        lambda_x -> eta * lambda_x

    model="signal_only":
        background stays fixed and only fluorescence above background is
        multiplied by eta.
    """

    eta = float(efficiency)

    if not np.isfinite(eta) or not (0.0 < eta <= 1.0):
        raise ValueError("efficiency must satisfy 0 < eta <= 1.")

    if model == "thin_all_counts":
        lambda_minus = eta * base_params.lambda_minus_khz
        lambda_zero = eta * base_params.lambda_zero_khz

    elif model == "signal_only":
        bg = float(background_khz)

        signal_minus = max(base_params.lambda_minus_khz - bg, 0.0)
        signal_zero = max(base_params.lambda_zero_khz - bg, 0.0)

        lambda_minus = eta * signal_minus + bg
        lambda_zero = eta * signal_zero + bg

    else:
        raise ValueError(
            "model must be 'thin_all_counts' or 'signal_only'."
        )

    params = MMPPParams(
        gamma_minus_to_zero_khz=base_params.gamma_minus_to_zero_khz,
        gamma_zero_to_minus_khz=base_params.gamma_zero_to_minus_khz,
        lambda_minus_khz=lambda_minus,
        lambda_zero_khz=lambda_zero,
    )
    params.validate()
    return params


def scale_model_parameters(
    params: MMPPParams,
    gamma_minus_to_zero_scale: float = 1.0,
    gamma_zero_to_minus_scale: float = 1.0,
    lambda_minus_scale: float = 1.0,
    lambda_zero_scale: float = 1.0,
) -> MMPPParams:
    """
    Construct the parameters USED BY THE FILTER.

    This does NOT modify already-generated data. It is therefore the function
    used for controlled parameter-mismatch tests.
    """

    scales = np.asarray(
        [
            gamma_minus_to_zero_scale,
            gamma_zero_to_minus_scale,
            lambda_minus_scale,
            lambda_zero_scale,
        ],
        dtype=float,
    )

    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("All parameter scale factors must be positive.")

    out = MMPPParams(
        gamma_minus_to_zero_khz=(
            params.gamma_minus_to_zero_khz
            * float(gamma_minus_to_zero_scale)
        ),
        gamma_zero_to_minus_khz=(
            params.gamma_zero_to_minus_khz
            * float(gamma_zero_to_minus_scale)
        ),
        lambda_minus_khz=(
            params.lambda_minus_khz
            * float(lambda_minus_scale)
        ),
        lambda_zero_khz=(
            params.lambda_zero_khz
            * float(lambda_zero_scale)
        ),
    )
    out.validate()
    return out


# =============================================================================
# Exact continuous-time simulation
# =============================================================================


def simulate_mmpp_shot(
    duration_ms: float,
    initial_state: int,
    params: MMPPParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Gillespie simulation of the hidden charge process plus detected photons.

    Returns exact photon arrival times in milliseconds.
    """

    params.validate()

    if duration_ms <= 0:
        raise ValueError("duration_ms must be > 0.")
    if initial_state not in (0, 1):
        raise ValueError("initial_state must be 0 (NV-) or 1 (NV0).")

    t = 0.0
    state = int(initial_state)
    clicks: list[float] = []

    while t < duration_ms:

        if state == 0:
            switch_rate = params.gamma_minus_to_zero_khz
            photon_rate = params.lambda_minus_khz
        else:
            switch_rate = params.gamma_zero_to_minus_khz
            photon_rate = params.lambda_zero_khz

        total_rate = switch_rate + photon_rate

        if total_rate <= 0:
            break

        t += float(rng.exponential(1.0 / total_rate))

        if t >= duration_ms:
            break

        if rng.random() < photon_rate / total_rate:
            clicks.append(t)
        else:
            state = 1 - state

    return np.asarray(clicks, dtype=float)


def simulate_balanced_dataset(
    n_shots_per_state: int,
    duration_ms: float,
    params: MMPPParams,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Equal numbers of initial NV- and initial NV0 shots.
    """

    if int(n_shots_per_state) < 1:
        raise ValueError("n_shots_per_state must be >= 1.")

    rng = np.random.default_rng(seed)

    shots: list[np.ndarray] = []
    labels: list[int] = []

    for initial_state in (0, 1):
        for _ in range(int(n_shots_per_state)):
            shots.append(
                simulate_mmpp_shot(
                    duration_ms=duration_ms,
                    initial_state=initial_state,
                    params=params,
                    rng=rng,
                )
            )
            labels.append(initial_state)

    labels_arr = np.asarray(labels, dtype=int)

    order = rng.permutation(len(shots))
    shots = [shots[i] for i in order]
    labels_arr = labels_arr[order]

    return shots, labels_arr


# =============================================================================
# Exact event-time INITIAL-state likelihood
# =============================================================================


@lru_cache(maxsize=2048)
def _cached_no_click_eigendecomposition(params: MMPPParams):
    A = np.asarray(params.no_click_generator, dtype=float)
    eigvals, eigvecs = np.linalg.eig(A)
    eigvecs_inv = np.linalg.inv(eigvecs)
    return eigvals, eigvecs, eigvecs_inv


def _propagate_no_click_matrix(
    H: np.ndarray,
    dt_ms: float,
    params: MMPPParams,
) -> np.ndarray:
    """
    exp[(Q - Lambda) dt] @ H
    """

    if dt_ms == 0.0:
        return H

    eigvals, eigvecs, eigvecs_inv = (
        _cached_no_click_eigendecomposition(params)
    )

    coeff = eigvecs_inv @ H

    out = eigvecs @ (
        np.exp(eigvals * float(dt_ms))[:, None] * coeff
    )

    out = np.real_if_close(out, tol=1000)
    return np.asarray(np.real(out), dtype=float)


def _renormalize_hypothesis_columns(
    H: np.ndarray,
    log_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize each initial-state hypothesis separately and retain the removed
    normalization in log space.
    """

    s = np.sum(H, axis=0)

    if np.any(s <= 0) or np.any(~np.isfinite(s)):
        raise FloatingPointError("Invalid MMPP likelihood normalization.")

    H = H / s[None, :]
    log_scale = log_scale + np.log(s)

    return H, log_scale


def initial_state_llr_at_times(
    timestamps_ms: Sequence[float],
    readout_times_ms: np.ndarray,
    params: MMPPParams,
) -> np.ndarray:
    """
    Exact event-time log-likelihood ratio:

        LLR(t)
          = log P(record up to t | initial NV-)
          - log P(record up to t | initial NV0)

    Equal priors:
        LLR >= 0 -> predict initial NV-
        LLR <  0 -> predict initial NV0
    """

    params.validate()

    ts = np.sort(np.asarray(timestamps_ms, dtype=float).reshape(-1))
    times = np.asarray(readout_times_ms, dtype=float).reshape(-1)

    if times.size == 0:
        raise ValueError("readout_times_ms cannot be empty.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("readout_times_ms must be strictly increasing.")
    if times[0] <= 0:
        raise ValueError("readout_times_ms must be > 0.")

    max_t = float(times[-1])

    if ts.size and (ts[0] < 0 or ts[-1] > max_t):
        raise ValueError("Photon timestamps lie outside the readout window.")

    # Column 0 = initial NV- hypothesis
    # Column 1 = initial NV0 hypothesis
    H = np.eye(2, dtype=float)
    log_scale = np.zeros(2, dtype=float)

    Lam = params.Lambda

    llr = np.empty(times.size, dtype=float)

    t_now = 0.0
    click_idx = 0

    for time_idx, t_report in enumerate(times):

        while click_idx < ts.size and ts[click_idx] < t_report:
            t_click = float(ts[click_idx])

            H = _propagate_no_click_matrix(
                H,
                t_click - t_now,
                params,
            )
            H, log_scale = _renormalize_hypothesis_columns(
                H,
                log_scale,
            )

            # Photon jump.
            H = Lam @ H
            H, log_scale = _renormalize_hypothesis_columns(
                H,
                log_scale,
            )

            t_now = t_click
            click_idx += 1

        H = _propagate_no_click_matrix(
            H,
            float(t_report) - t_now,
            params,
        )
        H, log_scale = _renormalize_hypothesis_columns(
            H,
            log_scale,
        )

        llr[time_idx] = log_scale[0] - log_scale[1]
        t_now = float(t_report)

    return llr


def dataset_initial_state_llr(
    shots: Sequence[np.ndarray],
    readout_times_ms: np.ndarray,
    params: MMPPParams,
) -> np.ndarray:
    """
    Return shape:
        (n_shots, n_readout_times)
    """

    out = np.empty(
        (len(shots), len(readout_times_ms)),
        dtype=float,
    )

    for i, timestamps in enumerate(shots):
        out[i] = initial_state_llr_at_times(
            timestamps_ms=timestamps,
            readout_times_ms=readout_times_ms,
            params=params,
        )

    return out


def initial_state_loglikelihoods_at_times(
    timestamps_ms: Sequence[float],
    readout_times_ms: np.ndarray,
    params: MMPPParams,
) -> np.ndarray:
    """
    Exact event-time log likelihoods for BOTH initial-state hypotheses.

    Returns
    -------
    logL : ndarray, shape (n_readout_times, 2)

        logL[:, 0]
            log P(record up to t | initial NV-)

        logL[:, 1]
            log P(record up to t | initial NV0)

    This is the quantity needed for a Bayesian mixture over uncertain model
    parameters. A mixture likelihood is

        P(D | H) = sum_k w_k P(D | H, theta_k),

    so the individual hypothesis likelihoods must be marginalized BEFORE
    taking their ratio.
    """

    params.validate()

    ts = np.sort(np.asarray(timestamps_ms, dtype=float).reshape(-1))
    times = np.asarray(readout_times_ms, dtype=float).reshape(-1)

    if times.size == 0:
        raise ValueError("readout_times_ms cannot be empty.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("readout_times_ms must be strictly increasing.")
    if times[0] <= 0:
        raise ValueError("readout_times_ms must be > 0.")

    max_t = float(times[-1])

    if ts.size and (ts[0] < 0 or ts[-1] > max_t):
        raise ValueError("Photon timestamps lie outside the readout window.")

    # Column 0 = initial NV- hypothesis
    # Column 1 = initial NV0 hypothesis
    H = np.eye(2, dtype=float)
    log_scale = np.zeros(2, dtype=float)

    Lam = params.Lambda

    logL = np.empty((times.size, 2), dtype=float)

    t_now = 0.0
    click_idx = 0

    for time_idx, t_report in enumerate(times):

        while click_idx < ts.size and ts[click_idx] < t_report:
            t_click = float(ts[click_idx])

            H = _propagate_no_click_matrix(
                H,
                t_click - t_now,
                params,
            )
            H, log_scale = _renormalize_hypothesis_columns(
                H,
                log_scale,
            )

            H = Lam @ H
            H, log_scale = _renormalize_hypothesis_columns(
                H,
                log_scale,
            )

            t_now = t_click
            click_idx += 1

        H = _propagate_no_click_matrix(
            H,
            float(t_report) - t_now,
            params,
        )
        H, log_scale = _renormalize_hypothesis_columns(
            H,
            log_scale,
        )

        # After the column-wise normalization, each column sums to one.
        # log_scale therefore contains the total log likelihood of the
        # observed record under the corresponding initial-state hypothesis.
        logL[time_idx] = log_scale
        t_now = float(t_report)

    return logL


def dataset_initial_state_loglikelihoods(
    shots: Sequence[np.ndarray],
    readout_times_ms: np.ndarray,
    params: MMPPParams,
) -> np.ndarray:
    """
    Return exact initial-state log likelihoods with shape

        (n_shots, n_readout_times, 2).
    """

    out = np.empty(
        (len(shots), len(readout_times_ms), 2),
        dtype=float,
    )

    for i, timestamps in enumerate(shots):
        out[i] = initial_state_loglikelihoods_at_times(
            timestamps_ms=timestamps,
            readout_times_ms=readout_times_ms,
            params=params,
        )

    return out


def _logsumexp(
    values: np.ndarray,
    axis: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Small dependency-free logsumexp helper.
    """

    values = np.asarray(values, dtype=float)

    vmax = np.max(values, axis=axis, keepdims=True)

    # Protect the all -inf case.
    finite = np.isfinite(vmax)

    shifted = np.where(
        finite,
        values - vmax,
        -np.inf,
    )

    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)

    out = np.where(
        finite,
        vmax + np.log(summed),
        -np.inf,
    )

    if axis is not None:
        out = np.squeeze(out, axis=axis)

    return out


# =============================================================================
# Threshold classifier and fidelity
# =============================================================================


def total_counts_at_times(
    shots: Sequence[np.ndarray],
    readout_times_ms: np.ndarray,
) -> np.ndarray:
    """
    Total photons detected in [0, t_R).
    """

    times = np.asarray(readout_times_ms, dtype=float)

    counts = np.empty(
        (len(shots), len(times)),
        dtype=int,
    )

    for i, timestamps in enumerate(shots):
        counts[i] = np.searchsorted(
            np.asarray(timestamps, dtype=float),
            times,
            side="left",
        )

    return counts


def balanced_initial_state_fidelity(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """
    F_C = 1/2 [P(correct | initial NV-) + P(correct | initial NV0)]
    """

    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape.")

    minus = labels == 0
    zero = labels == 1

    if not np.any(minus) or not np.any(zero):
        raise ValueError("Both initial states must be represented.")

    F_minus = float(np.mean(predictions[minus] == 0))
    F_zero = float(np.mean(predictions[zero] == 1))

    return 0.5 * (F_minus + F_zero)


def optimize_count_threshold(
    counts: np.ndarray,
    labels: np.ndarray,
) -> tuple[int, float]:
    """
    Choose integer threshold on CALIBRATION data.

    count >= n_threshold -> initial NV-
    count <  n_threshold -> initial NV0
    """

    counts = np.asarray(counts, dtype=int)
    labels = np.asarray(labels, dtype=int)

    max_count = int(np.max(counts)) if counts.size else 0

    best_threshold = 0
    best_fidelity = -np.inf

    for n_threshold in range(max_count + 2):

        pred = np.where(
            counts >= n_threshold,
            0,
            1,
        )

        F = balanced_initial_state_fidelity(
            labels,
            pred,
        )

        if F > best_fidelity:
            best_fidelity = F
            best_threshold = n_threshold

    return int(best_threshold), float(best_fidelity)


# =============================================================================
# Reusable dataset preparation / evaluation
# =============================================================================


def prepare_calibration_and_test_data(
    true_params: MMPPParams,
    readout_times_us: Sequence[float],
    n_calibration_shots_per_state: int,
    n_test_shots_per_state: int,
    seed: int,
) -> dict:
    """
    Generate data ONCE from true_params.

    Mismatch studies reuse exactly these photon records while changing only
    the parameters supplied to the likelihood calculation.
    """

    readout_times_us = np.asarray(readout_times_us, dtype=float)

    if np.any(readout_times_us <= 0):
        raise ValueError("All readout times must be > 0.")

    readout_times_ms = readout_times_us / 1000.0
    max_duration_ms = float(np.max(readout_times_ms))

    cal_shots, cal_labels = simulate_balanced_dataset(
        n_shots_per_state=n_calibration_shots_per_state,
        duration_ms=max_duration_ms,
        params=true_params,
        seed=seed,
    )

    test_shots, test_labels = simulate_balanced_dataset(
        n_shots_per_state=n_test_shots_per_state,
        duration_ms=max_duration_ms,
        params=true_params,
        seed=seed + 1_000_003,
    )

    return {
        "true_params": true_params,
        "readout_times_us": readout_times_us,
        "readout_times_ms": readout_times_ms,
        "cal_shots": cal_shots,
        "cal_labels": cal_labels,
        "test_shots": test_shots,
        "test_labels": test_labels,
        "cal_counts": total_counts_at_times(
            cal_shots,
            readout_times_ms,
        ),
        "test_counts": total_counts_at_times(
            test_shots,
            readout_times_ms,
        ),
    }


def evaluate_threshold_vs_time(data: dict) -> dict:
    """
    Threshold calibration/test fidelity over every t_R.
    """

    cal_counts = data["cal_counts"]
    test_counts = data["test_counts"]
    cal_labels = data["cal_labels"]
    test_labels = data["test_labels"]

    n_times = cal_counts.shape[1]

    thresholds = np.empty(n_times, dtype=int)
    cal_F = np.empty(n_times, dtype=float)
    test_F = np.empty(n_times, dtype=float)

    for j in range(n_times):
        n_th, F_cal = optimize_count_threshold(
            cal_counts[:, j],
            cal_labels,
        )

        thresholds[j] = n_th
        cal_F[j] = F_cal

        pred_test = np.where(
            test_counts[:, j] >= n_th,
            0,
            1,
        )

        test_F[j] = balanced_initial_state_fidelity(
            test_labels,
            pred_test,
        )

    opt_idx = int(np.argmax(cal_F))

    return {
        "thresholds": thresholds,
        "cal_fidelity": cal_F,
        "test_fidelity": test_F,
        "opt_idx": opt_idx,
        "opt_time_us": float(data["readout_times_us"][opt_idx]),
        "opt_threshold": int(thresholds[opt_idx]),
        "opt_test_fidelity": float(test_F[opt_idx]),
    }


def evaluate_mmpp_vs_time(
    data: dict,
    inference_params: MMPPParams,
) -> dict:
    """
    Event-time MMPP fidelity versus t_R using the supplied INFERENCE model.

    Note:
        inference_params may intentionally differ from data["true_params"].
    """

    cal_llr = dataset_initial_state_llr(
        shots=data["cal_shots"],
        readout_times_ms=data["readout_times_ms"],
        params=inference_params,
    )

    test_llr = dataset_initial_state_llr(
        shots=data["test_shots"],
        readout_times_ms=data["readout_times_ms"],
        params=inference_params,
    )

    n_times = len(data["readout_times_us"])

    cal_F = np.empty(n_times, dtype=float)
    test_F = np.empty(n_times, dtype=float)

    for j in range(n_times):

        pred_cal = np.where(
            cal_llr[:, j] >= 0.0,
            0,
            1,
        )

        pred_test = np.where(
            test_llr[:, j] >= 0.0,
            0,
            1,
        )

        cal_F[j] = balanced_initial_state_fidelity(
            data["cal_labels"],
            pred_cal,
        )

        test_F[j] = balanced_initial_state_fidelity(
            data["test_labels"],
            pred_test,
        )

    opt_idx = int(np.argmax(cal_F))

    return {
        "cal_llr": cal_llr,
        "test_llr": test_llr,
        "cal_fidelity": cal_F,
        "test_fidelity": test_F,
        "opt_idx": opt_idx,
        "opt_time_us": float(data["readout_times_us"][opt_idx]),
        "opt_test_fidelity": float(test_F[opt_idx]),
    }


def evaluate_mmpp_at_single_time(
    shots: Sequence[np.ndarray],
    labels: np.ndarray,
    time_us: float,
    inference_params: MMPPParams,
) -> float:
    """
    Faster helper for large mismatch grids.

    Evaluates one inference model at one fixed t_R.
    """

    t_value_ms = float(time_us) / 1000.0
    t_ms = np.asarray([t_value_ms], dtype=float)

    # The trajectories may have been simulated out to a longer maximum
    # readout time. For a single-time evaluation, discard photons occurring
    # after the requested t_R. Without this truncation the exact-likelihood
    # routine correctly rejects timestamps outside its observation window.
    truncated_shots = [
        np.asarray(ts, dtype=float)[
            np.asarray(ts, dtype=float) < t_value_ms
        ]
        for ts in shots
    ]

    llr = dataset_initial_state_llr(
        shots=truncated_shots,
        readout_times_ms=t_ms,
        params=inference_params,
    )[:, 0]

    pred = np.where(
        llr >= 0.0,
        0,
        1,
    )

    return balanced_initial_state_fidelity(
        labels,
        pred,
    )



# =============================================================================
# Bayesian ensemble MMPP helpers
# =============================================================================


def _relative_lognormal_quadrature(
    coefficient_of_variation: float,
    nodes_per_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic quadrature for a positive relative scale s.

    The distribution is log-normal with

        E[s] = 1

    and the requested coefficient of variation

        CV = std(s) / E[s].

    This is used as a synthetic posterior for a switching-rate estimate.
    When CV=0, the posterior collapses to s=1 exactly.

    Returns
    -------
    scales, weights
        Positive scale factors and normalized quadrature weights.
    """

    cv = float(coefficient_of_variation)
    n = int(nodes_per_dimension)

    if not np.isfinite(cv) or cv < 0:
        raise ValueError("coefficient_of_variation must be >= 0.")

    if cv == 0.0:
        return np.array([1.0]), np.array([1.0])

    if n < 2:
        raise ValueError(
            "nodes_per_dimension must be >= 2 when CV > 0."
        )

    # If s ~ LogNormal(mu_ln, sigma_ln^2),
    #
    #   CV^2 = exp(sigma_ln^2) - 1
    #
    # and setting mu_ln=-sigma_ln^2/2 gives E[s]=1.
    sigma_ln = float(np.sqrt(np.log1p(cv * cv)))
    mu_ln = -0.5 * sigma_ln * sigma_ln

    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(n)

    # Hermite-Gauss integrates exp(-x^2) f(x). Convert to z~N(0,1).
    z = np.sqrt(2.0) * gh_nodes
    weights = gh_weights / np.sqrt(np.pi)

    scales = np.exp(
        mu_ln + sigma_ln * z
    )

    weights = weights / np.sum(weights)

    return scales.astype(float), weights.astype(float)


def make_bayesian_switching_ensemble(
    nominal_params: MMPPParams,
    gamma_minus_to_zero_cv: float = 0.25,
    gamma_zero_to_minus_cv: float = 0.25,
    nodes_per_dimension: int = 3,
) -> tuple[list[MMPPParams], np.ndarray]:
    """
    Build a finite Bayesian ensemble over uncertain SWITCHING rates.

    The emission rates lambda_- and lambda_0 are held fixed at their nominal
    values. Only Gamma_-0 and Gamma_0- are marginalized.

    The two switching-rate posteriors are modeled as independent log-normal
    distributions centered in MEAN on the nominal rates.

    Example
    -------
    With 3 Gauss-Hermite nodes per switching rate, the Bayesian MMPP contains

        3 x 3 = 9

    ordinary 2-state MMPP filters.

    This is much cheaper than running MCMC online.
    """

    nominal_params.validate()

    minus_scales, minus_weights = _relative_lognormal_quadrature(
        gamma_minus_to_zero_cv,
        nodes_per_dimension,
    )

    zero_scales, zero_weights = _relative_lognormal_quadrature(
        gamma_zero_to_minus_cv,
        nodes_per_dimension,
    )

    ensemble: list[MMPPParams] = []
    weights: list[float] = []

    for i, scale_minus in enumerate(minus_scales):
        for j, scale_zero in enumerate(zero_scales):

            ensemble.append(
                scale_model_parameters(
                    nominal_params,
                    gamma_minus_to_zero_scale=float(scale_minus),
                    gamma_zero_to_minus_scale=float(scale_zero),
                )
            )

            weights.append(
                float(minus_weights[i] * zero_weights[j])
            )

    weights_arr = np.asarray(weights, dtype=float)
    weights_arr /= np.sum(weights_arr)

    return ensemble, weights_arr


def evaluate_bayesian_mmpp_vs_time(
    data: dict,
    ensemble_params: Sequence[MMPPParams],
    ensemble_weights: Sequence[float],
) -> dict:
    """
    Bayesian model-averaged initial-state inference.

    For each initial-state hypothesis H,

        P(D | H)
            = sum_k w_k P(D | H, theta_k),

    where theta_k is one member of the switching-rate posterior ensemble.

    IMPORTANT:
    We average the HYPOTHESIS LIKELIHOODS and only then form the likelihood
    ratio. Averaging individual model LLRs would be mathematically incorrect.
    """

    params_list = list(ensemble_params)
    weights = np.asarray(ensemble_weights, dtype=float).reshape(-1)

    if len(params_list) == 0:
        raise ValueError("Bayesian ensemble cannot be empty.")

    if len(params_list) != len(weights):
        raise ValueError(
            "ensemble_params and ensemble_weights must have the same length."
        )

    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Bayesian ensemble weights must be finite and >= 0.")

    if np.sum(weights) <= 0:
        raise ValueError("Bayesian ensemble weights must sum to > 0.")

    weights = weights / np.sum(weights)
    log_weights = np.log(weights)

    n_cal = len(data["cal_shots"])
    n_test = len(data["test_shots"])
    n_times = len(data["readout_times_ms"])

    # Accumulate the mixture likelihood online in log space. This avoids
    # storing a (n_models x n_shots x n_times x 2) array.
    log_marginal_cal = np.full(
        (n_cal, n_times, 2),
        -np.inf,
        dtype=float,
    )

    log_marginal_test = np.full(
        (n_test, n_times, 2),
        -np.inf,
        dtype=float,
    )

    for model_index, (params, log_w) in enumerate(
        zip(params_list, log_weights)
    ):
        params.validate()

        cal_logL = dataset_initial_state_loglikelihoods(
            shots=data["cal_shots"],
            readout_times_ms=data["readout_times_ms"],
            params=params,
        )

        test_logL = dataset_initial_state_loglikelihoods(
            shots=data["test_shots"],
            readout_times_ms=data["readout_times_ms"],
            params=params,
        )

        log_marginal_cal = np.logaddexp(
            log_marginal_cal,
            cal_logL + log_w,
        )

        log_marginal_test = np.logaddexp(
            log_marginal_test,
            test_logL + log_w,
        )

    cal_llr = (
        log_marginal_cal[:, :, 0]
        - log_marginal_cal[:, :, 1]
    )

    test_llr = (
        log_marginal_test[:, :, 0]
        - log_marginal_test[:, :, 1]
    )

    n_times = len(data["readout_times_us"])

    cal_F = np.empty(n_times, dtype=float)
    test_F = np.empty(n_times, dtype=float)

    for j in range(n_times):

        pred_cal = np.where(
            cal_llr[:, j] >= 0.0,
            0,
            1,
        )

        pred_test = np.where(
            test_llr[:, j] >= 0.0,
            0,
            1,
        )

        cal_F[j] = balanced_initial_state_fidelity(
            data["cal_labels"],
            pred_cal,
        )

        test_F[j] = balanced_initial_state_fidelity(
            data["test_labels"],
            pred_test,
        )

    opt_idx = int(np.argmax(cal_F))

    return {
        "ensemble_size": len(params_list),
        "ensemble_weights": weights,
        "cal_llr": cal_llr,
        "test_llr": test_llr,
        "cal_fidelity": cal_F,
        "test_fidelity": test_F,
        "opt_idx": opt_idx,
        "opt_time_us": float(data["readout_times_us"][opt_idx]),
        "opt_test_fidelity": float(test_F[opt_idx]),
    }


def run_bayesian_mmpp_comparison(
    base_power_uw: float,
    readout_times_us: Sequence[float],
    detection_efficiency: float = 1.0,
    efficiency_model: str = "thin_all_counts",
    nominal_gamma_minus_to_zero_scale: float = 1.0,
    nominal_gamma_zero_to_minus_scale: float = 1.0,
    gamma_minus_to_zero_cv: float = 0.25,
    gamma_zero_to_minus_cv: float = 0.25,
    quadrature_nodes_per_rate: int = 3,
    n_calibration_shots_per_state: int = 300,
    n_test_shots_per_state: int = 1000,
    seed: int = 12345,
    make_plots: bool = True,
) -> dict:
    """
    Compare threshold, plug-in MMPP, and Bayesian MMPP on the SAME data.

    Data generation
    ---------------
    Trajectories are generated from the true Shields-based parameters.

    Plug-in MMPP
    ------------
    The ordinary MMPP is given one nominal estimate:

        Gamma_hat_-0 = scale_-0 * Gamma_true_-0
        Gamma_hat_0- = scale_0- * Gamma_true_0-

    Bayesian MMPP
    -------------
    Uses the SAME nominal rates as the center of its uncertainty model, but
    marginalizes over a log-normal posterior around those nominal switching
    rates.

    Therefore:
        scales = 1, CV = 0
            -> Bayesian MMPP exactly reduces to the ordinary oracle MMPP.

        scales = 1, CV > 0
            -> tests the cost of acknowledging uncertainty when the nominal
               rate estimate is already correct.

        scales != 1, CV > 0
            -> tests whether model averaging is more robust than a biased
               single plug-in estimate.

    This synthetic log-normal posterior is only a simulation device. In an
    experiment, `ensemble_params` and `ensemble_weights` can instead come
    directly from polyspectra/Bayesian calibration posterior samples.
    """

    times_us = np.unique(
        np.sort(np.asarray(readout_times_us, dtype=float))
    )

    if times_us.size == 0 or np.any(times_us <= 0):
        raise ValueError("readout_times_us must contain positive values.")

    base_params = shields_2015_params(base_power_uw)

    true_params = apply_detection_efficiency(
        base_params=base_params,
        efficiency=detection_efficiency,
        model=efficiency_model,
    )

    # This is the single parameter estimate used by the ordinary MMPP.
    nominal_params = scale_model_parameters(
        true_params,
        gamma_minus_to_zero_scale=
            nominal_gamma_minus_to_zero_scale,
        gamma_zero_to_minus_scale=
            nominal_gamma_zero_to_minus_scale,
    )

    data = prepare_calibration_and_test_data(
        true_params=true_params,
        readout_times_us=times_us,
        n_calibration_shots_per_state=n_calibration_shots_per_state,
        n_test_shots_per_state=n_test_shots_per_state,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------

    t0 = time.perf_counter()
    threshold = evaluate_threshold_vs_time(data)
    threshold_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Ordinary plug-in MMPP
    # ------------------------------------------------------------------

    t0 = time.perf_counter()
    plugin = evaluate_mmpp_vs_time(
        data=data,
        inference_params=nominal_params,
    )
    plugin_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Bayesian finite-ensemble MMPP
    # ------------------------------------------------------------------

    ensemble_params, ensemble_weights = (
        make_bayesian_switching_ensemble(
            nominal_params=nominal_params,
            gamma_minus_to_zero_cv=gamma_minus_to_zero_cv,
            gamma_zero_to_minus_cv=gamma_zero_to_minus_cv,
            nodes_per_dimension=quadrature_nodes_per_rate,
        )
    )

    t0 = time.perf_counter()
    bayes = evaluate_bayesian_mmpp_vs_time(
        data=data,
        ensemble_params=ensemble_params,
        ensemble_weights=ensemble_weights,
    )
    bayes_seconds = time.perf_counter() - t0

    threshold_F = np.asarray(
        threshold["test_fidelity"],
        dtype=float,
    )

    plugin_F = np.asarray(
        plugin["test_fidelity"],
        dtype=float,
    )

    bayes_F = np.asarray(
        bayes["test_fidelity"],
        dtype=float,
    )

    # Common reference time: threshold calibration-selected optimum.
    ref_idx = int(threshold["opt_idx"])
    ref_time_us = float(threshold["opt_time_us"])

    n_total_records = (
        len(data["cal_shots"])
        + len(data["test_shots"])
    )

    result = {
        "base_power_uw": float(base_power_uw),
        "detection_efficiency": float(detection_efficiency),
        "true_params": true_params,
        "nominal_params": nominal_params,
        "readout_times_us": times_us,

        "nominal_gamma_minus_to_zero_scale":
            float(nominal_gamma_minus_to_zero_scale),

        "nominal_gamma_zero_to_minus_scale":
            float(nominal_gamma_zero_to_minus_scale),

        "gamma_minus_to_zero_cv":
            float(gamma_minus_to_zero_cv),

        "gamma_zero_to_minus_cv":
            float(gamma_zero_to_minus_cv),

        "quadrature_nodes_per_rate":
            int(quadrature_nodes_per_rate),

        "bayesian_ensemble_size":
            int(bayes["ensemble_size"]),

        "threshold_fidelity": threshold_F,
        "plugin_mmpp_fidelity": plugin_F,
        "bayesian_mmpp_fidelity": bayes_F,

        "threshold_opt_time_us":
            float(threshold["opt_time_us"]),

        "threshold_opt_fidelity":
            float(threshold["opt_test_fidelity"]),

        "plugin_opt_time_us":
            float(plugin["opt_time_us"]),

        "plugin_opt_fidelity":
            float(plugin["opt_test_fidelity"]),

        "bayesian_opt_time_us":
            float(bayes["opt_time_us"]),

        "bayesian_opt_fidelity":
            float(bayes["opt_test_fidelity"]),

        "reference_time_us": ref_time_us,

        "threshold_at_reference_time":
            float(threshold_F[ref_idx]),

        "plugin_at_reference_time":
            float(plugin_F[ref_idx]),

        "bayesian_at_reference_time":
            float(bayes_F[ref_idx]),

        "threshold_runtime_seconds":
            float(threshold_seconds),

        "plugin_runtime_seconds":
            float(plugin_seconds),

        "bayesian_runtime_seconds":
            float(bayes_seconds),

        "threshold_runtime_per_record_ms":
            1000.0 * float(threshold_seconds) / n_total_records,

        "plugin_runtime_per_record_ms":
            1000.0 * float(plugin_seconds) / n_total_records,

        "bayesian_runtime_per_record_ms":
            1000.0 * float(bayes_seconds) / n_total_records,
    }

    print("")
    print("Bayesian MMPP comparison")
    print("------------------------")
    print(f"physical power                  = {base_power_uw:g} uW")
    print(f"detection efficiency            = {detection_efficiency:g}")
    print(
        f"nominal Gamma_-0 / true         = "
        f"{nominal_gamma_minus_to_zero_scale:g}"
    )
    print(
        f"nominal Gamma_0- / true         = "
        f"{nominal_gamma_zero_to_minus_scale:g}"
    )
    print(
        f"switching-rate posterior CVs    = "
        f"({gamma_minus_to_zero_cv:g}, "
        f"{gamma_zero_to_minus_cv:g})"
    )
    print(
        f"quadrature nodes per rate       = "
        f"{quadrature_nodes_per_rate}"
    )
    print(
        f"Bayesian ensemble size          = "
        f"{bayes['ensemble_size']}"
    )
    print("")
    print(
        f"threshold optimum: "
        f"tR={threshold['opt_time_us']:.6g} us, "
        f"F={threshold['opt_test_fidelity']:.6f}"
    )
    print(
        f"plug-in MMPP optimum: "
        f"tR={plugin['opt_time_us']:.6g} us, "
        f"F={plugin['opt_test_fidelity']:.6f}"
    )
    print(
        f"Bayesian MMPP optimum: "
        f"tR={bayes['opt_time_us']:.6g} us, "
        f"F={bayes['opt_test_fidelity']:.6f}"
    )
    print("")
    print(
        f"At common threshold-selected tR={ref_time_us:.6g} us:"
    )
    print(
        f"    threshold      F={threshold_F[ref_idx]:.6f}"
    )
    print(
        f"    plug-in MMPP   F={plugin_F[ref_idx]:.6f}"
    )
    print(
        f"    Bayesian MMPP  F={bayes_F[ref_idx]:.6f}"
    )
    print("")
    print("Approximate Python inference cost for the full time sweep:")
    print(
        f"    threshold      {threshold_seconds:.4f} s "
        f"({result['threshold_runtime_per_record_ms']:.4f} ms/record)"
    )
    print(
        f"    plug-in MMPP   {plugin_seconds:.4f} s "
        f"({result['plugin_runtime_per_record_ms']:.4f} ms/record)"
    )
    print(
        f"    Bayesian MMPP  {bayes_seconds:.4f} s "
        f"({result['bayesian_runtime_per_record_ms']:.4f} ms/record)"
    )

    if make_plots:

        # --------------------------------------------------------------
        # Fidelity versus readout time
        # --------------------------------------------------------------

        fig1, ax1 = plt.subplots(figsize=(8.2, 5.3))

        ax1.plot(
            times_us,
            threshold_F,
            "o-",
            label="total-count threshold",
        )

        ax1.plot(
            times_us,
            plugin_F,
            "s-",
            label="plug-in MMPP",
        )

        ax1.plot(
            times_us,
            bayes_F,
            "^-",
            label="Bayesian ensemble MMPP",
        )

        ax1.set_xscale("log")
        ax1.set_xlabel(r"readout time $t_R$ ($\mu$s)")
        ax1.set_ylabel(r"initial charge readout fidelity $F_C$")
        ax1.set_ylim(0.48, 1.01)
        ax1.grid(alpha=0.2)
        ax1.legend()
        ax1.set_title(
            "Threshold vs plug-in MMPP vs Bayesian MMPP"
        )
        fig1.tight_layout()

        # --------------------------------------------------------------
        # Error rate
        # --------------------------------------------------------------

        fig2, ax2 = plt.subplots(figsize=(8.2, 5.3))

        ax2.plot(
            times_us,
            100.0 * (1.0 - threshold_F),
            "o-",
            label="total-count threshold",
        )

        ax2.plot(
            times_us,
            100.0 * (1.0 - plugin_F),
            "s-",
            label="plug-in MMPP",
        )

        ax2.plot(
            times_us,
            100.0 * (1.0 - bayes_F),
            "^-",
            label="Bayesian ensemble MMPP",
        )

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"readout time $t_R$ ($\mu$s)")
        ax2.set_ylabel("initial charge readout error (%)")
        ax2.grid(alpha=0.2)
        ax2.legend()
        ax2.set_title("Readout error with switching-rate uncertainty")
        fig2.tight_layout()

        # --------------------------------------------------------------
        # Gain over threshold
        # --------------------------------------------------------------

        fig3, ax3 = plt.subplots(figsize=(8.2, 5.3))

        ax3.plot(
            times_us,
            100.0 * (plugin_F - threshold_F),
            "s-",
            label="plug-in MMPP - threshold",
        )

        ax3.plot(
            times_us,
            100.0 * (bayes_F - threshold_F),
            "^-",
            label="Bayesian MMPP - threshold",
        )

        ax3.axhline(0.0, ls="--", lw=1.0)

        ax3.set_xscale("log")
        ax3.set_xlabel(r"readout time $t_R$ ($\mu$s)")
        ax3.set_ylabel("fidelity gain (percentage points)")
        ax3.grid(alpha=0.2)
        ax3.legend()
        ax3.set_title("Inference advantage versus threshold")
        fig3.tight_layout()

        # --------------------------------------------------------------
        # Approximate computational cost
        # --------------------------------------------------------------

        fig4, ax4 = plt.subplots(figsize=(7.0, 4.8))

        method_names = [
            "Threshold",
            "Plug-in MMPP",
            "Bayesian MMPP",
        ]

        runtime_ms = [
            result["threshold_runtime_per_record_ms"],
            result["plugin_runtime_per_record_ms"],
            result["bayesian_runtime_per_record_ms"],
        ]

        ax4.bar(
            method_names,
            runtime_ms,
        )

        ax4.set_yscale("log")
        ax4.set_ylabel(
            "Python runtime per record for full time sweep (ms)"
        )
        ax4.set_title(
            "Computational cost of the three inference schemes"
        )
        ax4.grid(axis="y", alpha=0.2)
        fig4.tight_layout()

        result["figures"] = (fig1, fig2, fig3, fig4)

        plt.show()

    return result



# =============================================================================
# Event-time HSMM via Erlang / phase-type dwell-time model
# =============================================================================


@dataclass(frozen=True)
class ErlangHSMMParams:
    """
    Two-macrostate semi-Markov charge model represented as a phase-type CTMC.

    Macrostate convention
    ---------------------
    NV-  = bright
    NV0  = dark

    Each macrostate is expanded into k sequential exponential phases.

    If a macrostate has:
        mean exit rate Gamma
        Erlang shape k

    each internal phase transition uses rate

        k * Gamma.

    Therefore the total macrostate dwell time is

        T ~ Gamma(shape=k, rate=k*Gamma),

    with

        E[T] = 1 / Gamma
        CV(T) = 1 / sqrt(k).

    k=1 gives an exponential dwell distribution and exactly recovers the
    ordinary two-state MMPP.
    """

    gamma_minus_to_zero_khz: float
    gamma_zero_to_minus_khz: float
    lambda_minus_khz: float
    lambda_zero_khz: float
    shape_minus: int = 1
    shape_zero: int = 1

    def validate(self) -> None:
        rates = np.asarray(
            [
                self.gamma_minus_to_zero_khz,
                self.gamma_zero_to_minus_khz,
                self.lambda_minus_khz,
                self.lambda_zero_khz,
            ],
            dtype=float,
        )

        if np.any(~np.isfinite(rates)):
            raise ValueError("All HSMM rates must be finite.")

        if self.gamma_minus_to_zero_khz <= 0:
            raise ValueError("gamma_minus_to_zero_khz must be > 0.")

        if self.gamma_zero_to_minus_khz <= 0:
            raise ValueError("gamma_zero_to_minus_khz must be > 0.")

        if self.lambda_minus_khz < 0 or self.lambda_zero_khz < 0:
            raise ValueError("Photon rates must be >= 0.")

        if self.lambda_minus_khz <= self.lambda_zero_khz:
            raise ValueError("NV- must be brighter than NV0.")

        if int(self.shape_minus) != self.shape_minus or self.shape_minus < 1:
            raise ValueError("shape_minus must be a positive integer.")

        if int(self.shape_zero) != self.shape_zero or self.shape_zero < 1:
            raise ValueError("shape_zero must be a positive integer.")

    @property
    def n_hidden_states(self) -> int:
        return int(self.shape_minus + self.shape_zero)

    @property
    def minus_initial_index(self) -> int:
        return 0

    @property
    def zero_initial_index(self) -> int:
        return int(self.shape_minus)

    @property
    def Q(self) -> np.ndarray:
        """
        Expanded hidden-state generator for COLUMN probability vectors.

        Minus phases:
            0, ..., shape_minus-1

        Zero phases:
            shape_minus, ..., shape_minus+shape_zero-1
        """

        self.validate()

        km = int(self.shape_minus)
        k0 = int(self.shape_zero)
        n = km + k0

        Q = np.zeros((n, n), dtype=float)

        r_minus = km * float(self.gamma_minus_to_zero_khz)
        r_zero = k0 * float(self.gamma_zero_to_minus_khz)

        # NV- Erlang chain.
        for phase in range(km):
            src = phase

            if phase < km - 1:
                dst = phase + 1
            else:
                dst = km  # enter the first NV0 phase

            Q[dst, src] = r_minus
            Q[src, src] = -r_minus

        # NV0 Erlang chain.
        for phase in range(k0):
            src = km + phase

            if phase < k0 - 1:
                dst = src + 1
            else:
                dst = 0  # return to the first NV- phase

            Q[dst, src] = r_zero
            Q[src, src] = -r_zero

        return Q

    @property
    def Lambda(self) -> np.ndarray:
        self.validate()

        km = int(self.shape_minus)
        k0 = int(self.shape_zero)

        rates = np.concatenate(
            [
                np.full(km, self.lambda_minus_khz, dtype=float),
                np.full(k0, self.lambda_zero_khz, dtype=float),
            ]
        )

        return np.diag(rates)

    @property
    def no_click_generator(self) -> np.ndarray:
        return self.Q - self.Lambda


def erlang_hsmm_from_mmpp(
    params: MMPPParams,
    shape_minus: int = 1,
    shape_zero: int = 1,
) -> ErlangHSMMParams:
    """
    Convert an MMPP parameter set into a mean-matched Erlang HSMM.

    The switching-rate parameters retain their interpretation as inverse mean
    macrostate dwell times.

    Thus increasing k changes the SHAPE of the dwell distribution without
    changing its mean.
    """

    params.validate()

    out = ErlangHSMMParams(
        gamma_minus_to_zero_khz=params.gamma_minus_to_zero_khz,
        gamma_zero_to_minus_khz=params.gamma_zero_to_minus_khz,
        lambda_minus_khz=params.lambda_minus_khz,
        lambda_zero_khz=params.lambda_zero_khz,
        shape_minus=int(shape_minus),
        shape_zero=int(shape_zero),
    )

    out.validate()
    return out


def simulate_hsmm_shot(
    duration_ms: float,
    initial_state: int,
    params: ErlangHSMMParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one exact photon record from the macrostate HSMM.

    The macrostate dwell time is sampled directly from its Erlang/gamma
    distribution. Conditional on a macrostate dwell interval, photons are a
    homogeneous Poisson process with the state's photon rate.

    Initial-state convention
    ------------------------
    The prepared initial charge state starts at age zero, i.e. at the beginning
    of a fresh dwell. This matches the initial-state-discrimination benchmark.

    Returns
    -------
    Sorted exact photon arrival times in milliseconds.
    """

    params.validate()

    if duration_ms <= 0:
        raise ValueError("duration_ms must be > 0.")

    if initial_state not in (0, 1):
        raise ValueError("initial_state must be 0 (NV-) or 1 (NV0).")

    t = 0.0
    macrostate = int(initial_state)
    clicks: list[float] = []

    while t < duration_ms:

        if macrostate == 0:
            shape = int(params.shape_minus)
            gamma = float(params.gamma_minus_to_zero_khz)
            photon_rate = float(params.lambda_minus_khz)
        else:
            shape = int(params.shape_zero)
            gamma = float(params.gamma_zero_to_minus_khz)
            photon_rate = float(params.lambda_zero_khz)

        # Erlang(shape=k, rate=k*Gamma), mean = 1/Gamma.
        dwell_ms = float(
            rng.gamma(
                shape=shape,
                scale=1.0 / (shape * gamma),
            )
        )

        interval_ms = min(
            dwell_ms,
            duration_ms - t,
        )

        if photon_rate > 0 and interval_ms > 0:
            n_photons = int(
                rng.poisson(photon_rate * interval_ms)
            )

            if n_photons > 0:
                local_times = np.sort(
                    rng.uniform(
                        0.0,
                        interval_ms,
                        size=n_photons,
                    )
                )
                clicks.extend(
                    (t + local_times).tolist()
                )

        t += interval_ms

        if t >= duration_ms:
            break

        # The full dwell ended before the readout ended.
        macrostate = 1 - macrostate

    return np.asarray(clicks, dtype=float)


def simulate_balanced_hsmm_dataset(
    n_shots_per_state: int,
    duration_ms: float,
    params: ErlangHSMMParams,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Equal numbers of initial NV- and NV0 trajectories generated by the HSMM.
    """

    if int(n_shots_per_state) < 1:
        raise ValueError("n_shots_per_state must be >= 1.")

    rng = np.random.default_rng(seed)

    shots: list[np.ndarray] = []
    labels: list[int] = []

    for initial_state in (0, 1):
        for _ in range(int(n_shots_per_state)):
            shots.append(
                simulate_hsmm_shot(
                    duration_ms=duration_ms,
                    initial_state=initial_state,
                    params=params,
                    rng=rng,
                )
            )
            labels.append(initial_state)

    labels_arr = np.asarray(labels, dtype=int)

    order = rng.permutation(len(shots))

    shots = [shots[i] for i in order]
    labels_arr = labels_arr[order]

    return shots, labels_arr


def prepare_hsmm_calibration_and_test_data(
    generation_params: ErlangHSMMParams,
    mean_matched_mmpp_params: MMPPParams,
    readout_times_us: Sequence[float],
    n_calibration_shots_per_state: int,
    n_test_shots_per_state: int,
    seed: int,
) -> dict:
    """
    Generate one calibration/test pair from an HSMM.

    The returned dictionary follows the same interface used by the existing
    threshold and MMPP evaluation functions.
    """

    times_us = np.unique(
        np.sort(
            np.asarray(
                readout_times_us,
                dtype=float,
            )
        )
    )

    if times_us.size == 0:
        raise ValueError("readout_times_us cannot be empty.")

    if np.any(times_us <= 0) or np.any(~np.isfinite(times_us)):
        raise ValueError("All readout times must be positive and finite.")

    times_ms = times_us / 1000.0
    max_duration_ms = float(times_ms[-1])

    cal_shots, cal_labels = simulate_balanced_hsmm_dataset(
        n_shots_per_state=n_calibration_shots_per_state,
        duration_ms=max_duration_ms,
        params=generation_params,
        seed=seed,
    )

    test_shots, test_labels = simulate_balanced_hsmm_dataset(
        n_shots_per_state=n_test_shots_per_state,
        duration_ms=max_duration_ms,
        params=generation_params,
        seed=seed + 1_000_003,
    )

    return {
        "true_params": mean_matched_mmpp_params,
        "generation_hsmm_params": generation_params,
        "readout_times_us": times_us,
        "readout_times_ms": times_ms,
        "cal_shots": cal_shots,
        "cal_labels": cal_labels,
        "test_shots": test_shots,
        "test_labels": test_labels,
        "cal_counts": total_counts_at_times(
            cal_shots,
            times_ms,
        ),
        "test_counts": total_counts_at_times(
            test_shots,
            times_ms,
        ),
    }


def _hsmm_propagate_no_click(
    H: np.ndarray,
    dt_ms: float,
    params: ErlangHSMMParams,
) -> np.ndarray:
    """
    Exact no-click propagation for the expanded phase-type model:

        H(t+dt) = exp[(Q-Lambda)dt] H(t).

    scipy.linalg.expm is used because Erlang phase chains can contain repeated
    eigenvalues and need not be numerically safe under a naive eigendecomposition.
    """

    if dt_ms == 0.0:
        return H

    A = np.asarray(
        params.no_click_generator,
        dtype=float,
    )

    out = expm(
        A * float(dt_ms)
    ) @ H

    out = np.real_if_close(out, tol=1000)
    out = np.asarray(np.real(out), dtype=float)

    # expm of a Metzler subgenerator should be nonnegative. Clip only tiny
    # floating-point negative values.
    tiny_negative = (
        (out < 0.0)
        & (out > -1.0e-12)
    )
    out[tiny_negative] = 0.0

    if np.any(out < -1.0e-10):
        raise FloatingPointError(
            "HSMM propagation produced a materially negative probability."
        )

    return out


def hsmm_initial_state_llr_at_times(
    timestamps_ms: Sequence[float],
    readout_times_ms: np.ndarray,
    params: ErlangHSMMParams,
) -> np.ndarray:
    """
    Exact event-time LLR for the initial MACRO charge state under the HSMM.

    The two hypotheses are initialized at the first phase of NV- or NV0.

    LLR(t)
        = log P(record up to t | initial NV-)
        - log P(record up to t | initial NV0).
    """

    params.validate()

    ts = np.sort(
        np.asarray(
            timestamps_ms,
            dtype=float,
        ).reshape(-1)
    )

    times = np.asarray(
        readout_times_ms,
        dtype=float,
    ).reshape(-1)

    if times.size == 0:
        raise ValueError("readout_times_ms cannot be empty.")

    if np.any(np.diff(times) <= 0):
        raise ValueError("readout_times_ms must be strictly increasing.")

    if times[0] <= 0:
        raise ValueError("readout_times_ms must be > 0.")

    max_t = float(times[-1])

    if ts.size and (
        ts[0] < 0
        or ts[-1] > max_t
    ):
        raise ValueError(
            "Photon timestamps lie outside the HSMM observation window."
        )

    n_hidden = params.n_hidden_states

    # Two columns = two initial macrostate hypotheses.
    H = np.zeros(
        (n_hidden, 2),
        dtype=float,
    )

    H[params.minus_initial_index, 0] = 1.0
    H[params.zero_initial_index, 1] = 1.0

    log_scale = np.zeros(2, dtype=float)

    Lam = params.Lambda

    llr = np.empty(
        times.size,
        dtype=float,
    )

    t_now = 0.0
    click_idx = 0

    for time_index, t_report in enumerate(times):

        while click_idx < ts.size and ts[click_idx] < t_report:
            t_click = float(ts[click_idx])

            H = _hsmm_propagate_no_click(
                H,
                t_click - t_now,
                params,
            )

            H, log_scale = _renormalize_hypothesis_columns(
                H,
                log_scale,
            )

            H = Lam @ H

            H, log_scale = _renormalize_hypothesis_columns(
                H,
                log_scale,
            )

            t_now = t_click
            click_idx += 1

        H = _hsmm_propagate_no_click(
            H,
            float(t_report) - t_now,
            params,
        )

        H, log_scale = _renormalize_hypothesis_columns(
            H,
            log_scale,
        )

        llr[time_index] = (
            log_scale[0]
            - log_scale[1]
        )

        t_now = float(t_report)

    return llr


def dataset_hsmm_initial_state_llr(
    shots: Sequence[np.ndarray],
    readout_times_ms: np.ndarray,
    params: ErlangHSMMParams,
) -> np.ndarray:
    """
    Return HSMM initial-state LLRs with shape

        (n_shots, n_readout_times).
    """

    out = np.empty(
        (len(shots), len(readout_times_ms)),
        dtype=float,
    )

    for i, timestamps in enumerate(shots):
        out[i] = hsmm_initial_state_llr_at_times(
            timestamps_ms=timestamps,
            readout_times_ms=readout_times_ms,
            params=params,
        )

    return out


def evaluate_hsmm_vs_time(
    data: dict,
    inference_params: ErlangHSMMParams,
) -> dict:
    """
    Event-time HSMM initial-state fidelity versus readout time.
    """

    t0 = time.perf_counter()

    cal_llr = dataset_hsmm_initial_state_llr(
        shots=data["cal_shots"],
        readout_times_ms=data["readout_times_ms"],
        params=inference_params,
    )

    test_llr = dataset_hsmm_initial_state_llr(
        shots=data["test_shots"],
        readout_times_ms=data["readout_times_ms"],
        params=inference_params,
    )

    elapsed = time.perf_counter() - t0

    n_times = len(
        data["readout_times_us"]
    )

    cal_F = np.empty(
        n_times,
        dtype=float,
    )

    test_F = np.empty(
        n_times,
        dtype=float,
    )

    for j in range(n_times):

        pred_cal = np.where(
            cal_llr[:, j] >= 0.0,
            0,
            1,
        )

        pred_test = np.where(
            test_llr[:, j] >= 0.0,
            0,
            1,
        )

        cal_F[j] = balanced_initial_state_fidelity(
            data["cal_labels"],
            pred_cal,
        )

        test_F[j] = balanced_initial_state_fidelity(
            data["test_labels"],
            pred_test,
        )

    opt_idx = int(
        np.argmax(cal_F)
    )

    return {
        "cal_llr": cal_llr,
        "test_llr": test_llr,
        "cal_fidelity": cal_F,
        "test_fidelity": test_F,
        "opt_idx": opt_idx,
        "opt_time_us": float(
            data["readout_times_us"][opt_idx]
        ),
        "opt_test_fidelity": float(
            test_F[opt_idx]
        ),
        "runtime_seconds": float(elapsed),
    }


def _run_one_hsmm_case(
    case_name: str,
    mean_mmpp_params: MMPPParams,
    generation_shape_minus: int,
    generation_shape_zero: int,
    readout_times_us: Sequence[float],
    n_calibration_shots_per_state: int,
    n_test_shots_per_state: int,
    seed: int,
) -> dict:
    """
    Run threshold, mean-matched MMPP, and correctly specified HSMM on one
    dwell-time-generating model.
    """

    generation_hsmm = erlang_hsmm_from_mmpp(
        mean_mmpp_params,
        shape_minus=generation_shape_minus,
        shape_zero=generation_shape_zero,
    )

    data = prepare_hsmm_calibration_and_test_data(
        generation_params=generation_hsmm,
        mean_matched_mmpp_params=mean_mmpp_params,
        readout_times_us=readout_times_us,
        n_calibration_shots_per_state=
            n_calibration_shots_per_state,
        n_test_shots_per_state=
            n_test_shots_per_state,
        seed=seed,
    )

    t0 = time.perf_counter()
    threshold = evaluate_threshold_vs_time(
        data
    )
    threshold_runtime = (
        time.perf_counter()
        - t0
    )

    t0 = time.perf_counter()
    mmpp = evaluate_mmpp_vs_time(
        data,
        mean_mmpp_params,
    )
    mmpp_runtime = (
        time.perf_counter()
        - t0
    )

    hsmm = evaluate_hsmm_vs_time(
        data,
        generation_hsmm,
    )

    max_llr_difference = float(
        np.max(
            np.abs(
                mmpp["test_llr"]
                - hsmm["test_llr"]
            )
        )
    )

    return {
        "case_name": case_name,
        "generation_hsmm": generation_hsmm,
        "data": data,
        "threshold": threshold,
        "mmpp": mmpp,
        "hsmm": hsmm,
        "threshold_runtime_seconds":
            float(threshold_runtime),
        "mmpp_runtime_seconds":
            float(mmpp_runtime),
        "hsmm_runtime_seconds":
            float(hsmm["runtime_seconds"]),
        "max_test_llr_difference_mmpp_vs_hsmm":
            max_llr_difference,
    }


def run_hsmm_comparison(
    base_power_uw: float,
    readout_times_us: Sequence[float],
    detection_efficiency: float = 1.0,
    efficiency_model: str = "thin_all_counts",
    nonexponential_shape_minus: int = 4,
    nonexponential_shape_zero: int = 4,
    n_calibration_shots_per_state: int = 300,
    n_test_shots_per_state: int = 1000,
    seed: int = 12345,
    make_plots: bool = True,
) -> dict:
    """
    Compare threshold, MMPP, and HSMM in TWO controlled simulation cases.

    Case A: exponential dwell times
    --------------------------------
        shape_minus = shape_zero = 1

    The HSMM is mathematically identical to the ordinary MMPP. This is a
    built-in implementation sanity check.

    Case B: non-exponential dwell times
    ------------------------------------
        shape_minus, shape_zero > 1

    Data are generated from Erlang/gamma macrostate dwell times, but the
    ordinary MMPP is still given only the SAME MEAN switching rates and
    therefore assumes exponential dwell times.

    The HSMM is given the correct Erlang shapes.

    Because

        mean dwell = 1/Gamma

    in both models, this comparison isolates dwell-time SHAPE rather than
    changing the mean switching timescale.
    """

    if int(nonexponential_shape_minus) < 2:
        raise ValueError(
            "nonexponential_shape_minus should be >= 2."
        )

    if int(nonexponential_shape_zero) < 2:
        raise ValueError(
            "nonexponential_shape_zero should be >= 2."
        )

    base_params = shields_2015_params(
        base_power_uw
    )

    mean_mmpp_params = apply_detection_efficiency(
        base_params=base_params,
        efficiency=detection_efficiency,
        model=efficiency_model,
    )

    # Case A: k=1 exactly reproduces a Markov / exponential dwell process.
    exponential = _run_one_hsmm_case(
        case_name="exponential dwell times",
        mean_mmpp_params=mean_mmpp_params,
        generation_shape_minus=1,
        generation_shape_zero=1,
        readout_times_us=readout_times_us,
        n_calibration_shots_per_state=
            n_calibration_shots_per_state,
        n_test_shots_per_state=
            n_test_shots_per_state,
        seed=seed,
    )

    # Case B: non-exponential Erlang dwell process.
    nonexponential = _run_one_hsmm_case(
        case_name="non-exponential Erlang dwell times",
        mean_mmpp_params=mean_mmpp_params,
        generation_shape_minus=int(
            nonexponential_shape_minus
        ),
        generation_shape_zero=int(
            nonexponential_shape_zero
        ),
        readout_times_us=readout_times_us,
        n_calibration_shots_per_state=
            n_calibration_shots_per_state,
        n_test_shots_per_state=
            n_test_shots_per_state,
        seed=seed + 10_000_019,
    )

    result = {
        "base_power_uw":
            float(base_power_uw),

        "detection_efficiency":
            float(detection_efficiency),

        "mean_mmpp_params":
            mean_mmpp_params,

        "nonexponential_shape_minus":
            int(nonexponential_shape_minus),

        "nonexponential_shape_zero":
            int(nonexponential_shape_zero),

        "exponential":
            exponential,

        "nonexponential":
            nonexponential,
    }

    print("")
    print("HSMM comparison")
    print("===============")
    print(
        f"physical power               = "
        f"{base_power_uw:g} uW"
    )
    print(
        f"detection efficiency         = "
        f"{detection_efficiency:g}"
    )
    print("")

    for case in (
        exponential,
        nonexponential,
    ):
        th = case["threshold"]
        mmpp = case["mmpp"]
        hsmm = case["hsmm"]

        print(case["case_name"])
        print("-" * len(case["case_name"]))
        print(
            f"generation shapes: "
            f"k-={case['generation_hsmm'].shape_minus}, "
            f"k0={case['generation_hsmm'].shape_zero}"
        )
        print(
            f"dwell CVs: "
            f"NV-={1.0 / np.sqrt(case['generation_hsmm'].shape_minus):.4f}, "
            f"NV0={1.0 / np.sqrt(case['generation_hsmm'].shape_zero):.4f}"
        )
        print(
            f"threshold optimum: "
            f"tR={th['opt_time_us']:.6g} us, "
            f"F={th['opt_test_fidelity']:.6f}"
        )
        print(
            f"MMPP optimum:      "
            f"tR={mmpp['opt_time_us']:.6g} us, "
            f"F={mmpp['opt_test_fidelity']:.6f}"
        )
        print(
            f"HSMM optimum:      "
            f"tR={hsmm['opt_time_us']:.6g} us, "
            f"F={hsmm['opt_test_fidelity']:.6f}"
        )
        print(
            f"max |MMPP LLR - HSMM LLR| on test data = "
            f"{case['max_test_llr_difference_mmpp_vs_hsmm']:.6e}"
        )
        print(
            f"runtimes: threshold={case['threshold_runtime_seconds']:.4f} s, "
            f"MMPP={case['mmpp_runtime_seconds']:.4f} s, "
            f"HSMM={case['hsmm_runtime_seconds']:.4f} s"
        )
        print("")

    # Strong sanity check for the exponential case.
    #
    # k=1 phase-type HSMM is exactly the same generative / inference model as
    # the two-state MMPP, so their LLRs should agree to numerical precision.
    exponential_llr_error = (
        exponential[
            "max_test_llr_difference_mmpp_vs_hsmm"
        ]
    )

    if exponential_llr_error > 1.0e-8:
        warnings.warn(
            "Exponential HSMM does not numerically match the MMPP as closely "
            "as expected. Inspect the phase-type implementation.",
            RuntimeWarning,
        )

    if make_plots:

        times_exp = exponential[
            "data"
        ]["readout_times_us"]

        times_nonexp = nonexponential[
            "data"
        ]["readout_times_us"]

        # --------------------------------------------------------------
        # Case A: exponential dwell times
        # --------------------------------------------------------------

        fig1, ax1 = plt.subplots(
            figsize=(8.2, 5.3)
        )

        ax1.plot(
            times_exp,
            exponential["threshold"]["test_fidelity"],
            "o-",
            label="threshold",
        )

        ax1.plot(
            times_exp,
            exponential["mmpp"]["test_fidelity"],
            "s-",
            label="MMPP",
        )

        ax1.plot(
            times_exp,
            exponential["hsmm"]["test_fidelity"],
            "^-",
            label="HSMM (k=1)",
        )

        ax1.set_xscale("log")
        ax1.set_xlabel(
            r"readout time $t_R$ ($\mu$s)"
        )
        ax1.set_ylabel(
            r"initial charge readout fidelity $F_C$"
        )
        ax1.set_ylim(0.48, 1.01)
        ax1.grid(alpha=0.2)
        ax1.legend()
        ax1.set_title(
            "Exponential dwell times: HSMM must reduce to MMPP"
        )
        fig1.tight_layout()

        # --------------------------------------------------------------
        # Case B: non-exponential dwell times
        # --------------------------------------------------------------

        fig2, ax2 = plt.subplots(
            figsize=(8.2, 5.3)
        )

        ax2.plot(
            times_nonexp,
            nonexponential["threshold"]["test_fidelity"],
            "o-",
            label="threshold",
        )

        ax2.plot(
            times_nonexp,
            nonexponential["mmpp"]["test_fidelity"],
            "s-",
            label="MMPP (exponential assumption)",
        )

        ax2.plot(
            times_nonexp,
            nonexponential["hsmm"]["test_fidelity"],
            "^-",
            label=(
                "HSMM "
                f"(k-={nonexponential_shape_minus}, "
                f"k0={nonexponential_shape_zero})"
            ),
        )

        ax2.set_xscale("log")
        ax2.set_xlabel(
            r"readout time $t_R$ ($\mu$s)"
        )
        ax2.set_ylabel(
            r"initial charge readout fidelity $F_C$"
        )
        ax2.set_ylim(0.48, 1.01)
        ax2.grid(alpha=0.2)
        ax2.legend()
        ax2.set_title(
            "Non-exponential dwell times: MMPP vs HSMM"
        )
        fig2.tight_layout()

        # --------------------------------------------------------------
        # Direct HSMM advantage over the ordinary MMPP
        # --------------------------------------------------------------

        fig3, ax3 = plt.subplots(
            figsize=(8.2, 5.3)
        )

        exp_gain = 100.0 * (
            exponential["hsmm"]["test_fidelity"]
            - exponential["mmpp"]["test_fidelity"]
        )

        nonexp_gain = 100.0 * (
            nonexponential["hsmm"]["test_fidelity"]
            - nonexponential["mmpp"]["test_fidelity"]
        )

        ax3.plot(
            times_exp,
            exp_gain,
            "o-",
            label="exponential data",
        )

        ax3.plot(
            times_nonexp,
            nonexp_gain,
            "s-",
            label="non-exponential data",
        )

        ax3.axhline(
            0.0,
            ls="--",
            lw=1.0,
        )

        ax3.set_xscale("log")
        ax3.set_xlabel(
            r"readout time $t_R$ ($\mu$s)"
        )
        ax3.set_ylabel(
            "HSMM - MMPP fidelity (percentage points)"
        )
        ax3.grid(alpha=0.2)
        ax3.legend()
        ax3.set_title(
            "When does semi-Markov dwell-time information help?"
        )
        fig3.tight_layout()

        # --------------------------------------------------------------
        # Runtime comparison for the non-exponential case
        # --------------------------------------------------------------

        fig4, ax4 = plt.subplots(
            figsize=(7.0, 4.8)
        )

        runtime_names = [
            "Threshold",
            "MMPP",
            "HSMM",
        ]

        runtime_values = [
            nonexponential[
                "threshold_runtime_seconds"
            ],
            nonexponential[
                "mmpp_runtime_seconds"
            ],
            nonexponential[
                "hsmm_runtime_seconds"
            ],
        ]

        ax4.bar(
            runtime_names,
            runtime_values,
        )

        ax4.set_yscale("log")
        ax4.set_ylabel(
            "runtime for full comparison (s)"
        )
        ax4.set_title(
            "Computational cost on non-exponential data"
        )
        ax4.grid(
            axis="y",
            alpha=0.2,
        )
        fig4.tight_layout()

        result["figures"] = (
            fig1,
            fig2,
            fig3,
            fig4,
        )

        plt.show()

    return result



# =============================================================================
# HSMM dwell-shape sweep
# =============================================================================


def run_hsmm_dwell_shape_sweep(
    shape_values: Sequence[int],
    base_power_uw: float,
    readout_times_us: Sequence[float],
    detection_efficiency: float = 1.0,
    efficiency_model: str = "thin_all_counts",
    n_calibration_shots_per_state: int = 200,
    n_test_shots_per_state: int = 600,
    seed: int = 12345,
    make_plots: bool = True,
) -> dict:
    """
    Sweep a COMMON Erlang dwell-shape k for NV- and NV0.

    For every k:
        1. Generate photon records from an HSMM with
               shape_minus = shape_zero = k.
        2. Keep the mean macrostate dwell times fixed at 1/Gamma.
        3. Re-optimize the total-count threshold on calibration data.
        4. Evaluate a mean-matched ordinary MMPP that assumes exponential
           dwell times.
        5. Evaluate the correctly specified event-time HSMM.

    The dwell distribution is

        T ~ Gamma(shape=k, rate=k*Gamma),

    so

        E[T] = 1/Gamma
        CV(T) = 1/sqrt(k).

    Therefore this sweep changes ONLY the dwell-time shape, not the mean
    switching timescale.

    Notes
    -----
    This study can be computationally expensive because the exact HSMM uses
    a 2k-state phase-type model and a matrix exponential between photon
    events. Start with modest shot counts, then increase them for final
    figures after the qualitative behavior is established.
    """

    shape_values = np.asarray(shape_values, dtype=int).reshape(-1)

    if shape_values.size == 0:
        raise ValueError("shape_values cannot be empty.")

    if np.any(shape_values < 1):
        raise ValueError("Every HSMM shape value must be >= 1.")

    if len(np.unique(shape_values)) != len(shape_values):
        raise ValueError("shape_values should not contain duplicates.")

    # Sort k from exponential -> increasingly non-exponential.
    shape_values = np.sort(shape_values)

    times_us = np.unique(
        np.sort(
            np.asarray(
                readout_times_us,
                dtype=float,
            )
        )
    )

    if times_us.size == 0 or np.any(times_us <= 0):
        raise ValueError(
            "readout_times_us must contain positive values."
        )

    base_params = shields_2015_params(
        base_power_uw
    )

    mean_mmpp_params = apply_detection_efficiency(
        base_params=base_params,
        efficiency=detection_efficiency,
        model=efficiency_model,
    )

    n_shapes = len(shape_values)

    dwell_cv = 1.0 / np.sqrt(
        shape_values.astype(float)
    )

    threshold_opt_F = np.empty(
        n_shapes,
        dtype=float,
    )
    mmpp_opt_F = np.empty(
        n_shapes,
        dtype=float,
    )
    hsmm_opt_F = np.empty(
        n_shapes,
        dtype=float,
    )

    threshold_opt_t = np.empty(
        n_shapes,
        dtype=float,
    )
    mmpp_opt_t = np.empty(
        n_shapes,
        dtype=float,
    )
    hsmm_opt_t = np.empty(
        n_shapes,
        dtype=float,
    )

    threshold_runtime = np.empty(
        n_shapes,
        dtype=float,
    )
    mmpp_runtime = np.empty(
        n_shapes,
        dtype=float,
    )
    hsmm_runtime = np.empty(
        n_shapes,
        dtype=float,
    )

    max_llr_difference = np.empty(
        n_shapes,
        dtype=float,
    )

    # Same-time comparisons at the threshold-selected t_R for each k.
    threshold_ref_F = np.empty(
        n_shapes,
        dtype=float,
    )
    mmpp_ref_F = np.empty(
        n_shapes,
        dtype=float,
    )
    hsmm_ref_F = np.empty(
        n_shapes,
        dtype=float,
    )
    ref_time_us = np.empty(
        n_shapes,
        dtype=float,
    )

    per_shape: list[dict] = []

    print("")
    print("HSMM dwell-shape sweep")
    print("======================")
    print(
        f"physical power               = "
        f"{base_power_uw:g} uW"
    )
    print(
        f"detection efficiency         = "
        f"{detection_efficiency:g}"
    )
    print(
        f"shape values                 = "
        f"{shape_values.tolist()}"
    )
    print("")

    for i, k in enumerate(shape_values):

        case = _run_one_hsmm_case(
            case_name=f"Erlang dwell shape k={int(k)}",
            mean_mmpp_params=mean_mmpp_params,
            generation_shape_minus=int(k),
            generation_shape_zero=int(k),
            readout_times_us=times_us,
            n_calibration_shots_per_state=
                n_calibration_shots_per_state,
            n_test_shots_per_state=
                n_test_shots_per_state,
            seed=seed + 1_000_003 * i,
        )

        per_shape.append(case)

        th = case["threshold"]
        mmpp = case["mmpp"]
        hsmm = case["hsmm"]

        threshold_opt_F[i] = (
            th["opt_test_fidelity"]
        )
        mmpp_opt_F[i] = (
            mmpp["opt_test_fidelity"]
        )
        hsmm_opt_F[i] = (
            hsmm["opt_test_fidelity"]
        )

        threshold_opt_t[i] = (
            th["opt_time_us"]
        )
        mmpp_opt_t[i] = (
            mmpp["opt_time_us"]
        )
        hsmm_opt_t[i] = (
            hsmm["opt_time_us"]
        )

        threshold_runtime[i] = (
            case["threshold_runtime_seconds"]
        )
        mmpp_runtime[i] = (
            case["mmpp_runtime_seconds"]
        )
        hsmm_runtime[i] = (
            case["hsmm_runtime_seconds"]
        )

        max_llr_difference[i] = (
            case[
                "max_test_llr_difference_mmpp_vs_hsmm"
            ]
        )

        # Evaluate all methods at the threshold-selected readout time for
        # this k. This isolates inference performance from method-specific
        # choices of measurement duration.
        idx_ref = int(
            th["opt_idx"]
        )

        ref_time_us[i] = float(
            case["data"]["readout_times_us"][
                idx_ref
            ]
        )

        threshold_ref_F[i] = float(
            th["test_fidelity"][idx_ref]
        )

        mmpp_ref_F[i] = float(
            mmpp["test_fidelity"][idx_ref]
        )

        hsmm_ref_F[i] = float(
            hsmm["test_fidelity"][idx_ref]
        )

        print(
            f"k={int(k):2d} | "
            f"CV={dwell_cv[i]:.4f} | "
            f"threshold Fopt={threshold_opt_F[i]:.5f}, "
            f"tR={threshold_opt_t[i]:8.3f} us | "
            f"MMPP Fopt={mmpp_opt_F[i]:.5f}, "
            f"tR={mmpp_opt_t[i]:8.3f} us | "
            f"HSMM Fopt={hsmm_opt_F[i]:.5f}, "
            f"tR={hsmm_opt_t[i]:8.3f} us | "
            f"HSMM-MMPP={100.0 * (hsmm_opt_F[i] - mmpp_opt_F[i]):+.3f} pp"
        )

    # Exact sanity check at k=1, if included.
    if np.any(shape_values == 1):
        idx_k1 = int(
            np.where(shape_values == 1)[0][0]
        )

        if max_llr_difference[idx_k1] > 1.0e-8:
            warnings.warn(
                "At k=1 the HSMM should reduce exactly to the MMPP, "
                "but the LLR difference exceeded the expected numerical "
                "tolerance.",
                RuntimeWarning,
            )

    result = {
        "shape_values":
            shape_values,

        "dwell_cv":
            dwell_cv,

        "base_power_uw":
            float(base_power_uw),

        "detection_efficiency":
            float(detection_efficiency),

        "mean_mmpp_params":
            mean_mmpp_params,

        "threshold_opt_fidelity":
            threshold_opt_F,

        "mmpp_opt_fidelity":
            mmpp_opt_F,

        "hsmm_opt_fidelity":
            hsmm_opt_F,

        "threshold_opt_time_us":
            threshold_opt_t,

        "mmpp_opt_time_us":
            mmpp_opt_t,

        "hsmm_opt_time_us":
            hsmm_opt_t,

        "reference_time_us":
            ref_time_us,

        "threshold_at_reference_time":
            threshold_ref_F,

        "mmpp_at_reference_time":
            mmpp_ref_F,

        "hsmm_at_reference_time":
            hsmm_ref_F,

        "hsmm_minus_mmpp_opt_gain_pp":
            100.0 * (
                hsmm_opt_F
                - mmpp_opt_F
            ),

        "hsmm_minus_threshold_opt_gain_pp":
            100.0 * (
                hsmm_opt_F
                - threshold_opt_F
            ),

        "hsmm_minus_mmpp_same_time_gain_pp":
            100.0 * (
                hsmm_ref_F
                - mmpp_ref_F
            ),

        "hsmm_minus_threshold_same_time_gain_pp":
            100.0 * (
                hsmm_ref_F
                - threshold_ref_F
            ),

        "threshold_runtime_seconds":
            threshold_runtime,

        "mmpp_runtime_seconds":
            mmpp_runtime,

        "hsmm_runtime_seconds":
            hsmm_runtime,

        "max_test_llr_difference_mmpp_vs_hsmm":
            max_llr_difference,

        "per_shape":
            per_shape,
    }

    if make_plots:

        # --------------------------------------------------------------
        # 1. Optimized fidelity vs dwell-shape k
        # --------------------------------------------------------------

        fig1, ax1 = plt.subplots(
            figsize=(8.2, 5.3)
        )

        ax1.plot(
            shape_values,
            threshold_opt_F,
            "o-",
            label="optimized threshold",
        )

        ax1.plot(
            shape_values,
            mmpp_opt_F,
            "s-",
            label="optimized MMPP",
        )

        ax1.plot(
            shape_values,
            hsmm_opt_F,
            "^-",
            label="optimized HSMM",
        )

        ax1.set_xlabel(
            r"Erlang dwell shape $k$"
        )
        ax1.set_ylabel(
            r"optimized initial-state fidelity $F_C$"
        )
        ax1.set_ylim(
            0.48,
            1.01,
        )
        ax1.grid(
            alpha=0.2,
        )
        ax1.legend()
        ax1.set_title(
            "Readout fidelity versus dwell-time shape"
        )

        fig1.tight_layout()

        # --------------------------------------------------------------
        # 2. Optimized fidelity vs dwell-time CV
        # --------------------------------------------------------------

        # Sort by increasing CV for a conventional left-to-right axis.
        cv_order = np.argsort(
            dwell_cv
        )

        fig2, ax2 = plt.subplots(
            figsize=(8.2, 5.3)
        )

        ax2.plot(
            dwell_cv[cv_order],
            threshold_opt_F[cv_order],
            "o-",
            label="optimized threshold",
        )

        ax2.plot(
            dwell_cv[cv_order],
            mmpp_opt_F[cv_order],
            "s-",
            label="optimized MMPP",
        )

        ax2.plot(
            dwell_cv[cv_order],
            hsmm_opt_F[cv_order],
            "^-",
            label="optimized HSMM",
        )

        ax2.set_xlabel(
            r"dwell-time coefficient of variation "
            r"$1/\sqrt{k}$"
        )

        ax2.set_ylabel(
            r"optimized initial-state fidelity $F_C$"
        )

        ax2.set_ylim(
            0.48,
            1.01,
        )

        ax2.grid(
            alpha=0.2,
        )
        ax2.legend()
        ax2.set_title(
            "Readout fidelity versus dwell-time variability"
        )

        fig2.tight_layout()

        # --------------------------------------------------------------
        # 3. HSMM advantage
        # --------------------------------------------------------------

        fig3, ax3 = plt.subplots(
            figsize=(8.2, 5.3)
        )

        ax3.plot(
            shape_values,
            result[
                "hsmm_minus_mmpp_opt_gain_pp"
            ],
            "o-",
            label="HSMM - MMPP",
        )

        ax3.plot(
            shape_values,
            result[
                "hsmm_minus_threshold_opt_gain_pp"
            ],
            "s-",
            label="HSMM - threshold",
        )

        ax3.axhline(
            0.0,
            ls="--",
            lw=1.0,
        )

        ax3.set_xlabel(
            r"Erlang dwell shape $k$"
        )

        ax3.set_ylabel(
            "optimized fidelity gain "
            "(percentage points)"
        )

        ax3.grid(
            alpha=0.2,
        )
        ax3.legend()

        ax3.set_title(
            "When is an HSMM worth the extra model complexity?"
        )

        fig3.tight_layout()

        # --------------------------------------------------------------
        # 4. Optimal readout time
        # --------------------------------------------------------------

        fig4, ax4 = plt.subplots(
            figsize=(8.2, 5.3)
        )

        ax4.plot(
            shape_values,
            threshold_opt_t,
            "o-",
            label="threshold",
        )

        ax4.plot(
            shape_values,
            mmpp_opt_t,
            "s-",
            label="MMPP",
        )

        ax4.plot(
            shape_values,
            hsmm_opt_t,
            "^-",
            label="HSMM",
        )

        ax4.set_yscale(
            "log"
        )

        ax4.set_xlabel(
            r"Erlang dwell shape $k$"
        )

        ax4.set_ylabel(
            r"calibration-selected optimum $t_R$ ($\mu$s)"
        )

        ax4.grid(
            alpha=0.2,
        )
        ax4.legend()

        ax4.set_title(
            "Optimal readout duration versus dwell shape"
        )

        fig4.tight_layout()

        # --------------------------------------------------------------
        # 5. Runtime
        # --------------------------------------------------------------

        fig5, ax5 = plt.subplots(
            figsize=(8.2, 5.3)
        )

        ax5.plot(
            shape_values,
            threshold_runtime,
            "o-",
            label="threshold",
        )

        ax5.plot(
            shape_values,
            mmpp_runtime,
            "s-",
            label="MMPP",
        )

        ax5.plot(
            shape_values,
            hsmm_runtime,
            "^-",
            label="HSMM",
        )

        ax5.set_yscale(
            "log"
        )

        ax5.set_xlabel(
            r"Erlang dwell shape $k$"
        )

        ax5.set_ylabel(
            "runtime for full calibration/test sweep (s)"
        )

        ax5.grid(
            alpha=0.2,
        )
        ax5.legend()

        ax5.set_title(
            "Computational cost versus HSMM state-space size"
        )

        fig5.tight_layout()

        result["figures"] = (
            fig1,
            fig2,
            fig3,
            fig4,
            fig5,
        )

        plt.show()

    return result


# =============================================================================
# Study 1: Shields-style power sweep
# =============================================================================


def run_power_sweep(
    powers_uw: Sequence[float],
    readout_times_us: Sequence[float],
    n_calibration_shots_per_state: int = 500,
    n_test_shots_per_state: int = 2000,
    seed: int = 12345,
    make_plots: bool = True,
) -> dict:

    powers_uw = np.asarray(powers_uw, dtype=float)

    threshold_F = []
    threshold_t = []
    threshold_n = []

    mmpp_same_F = []
    mmpp_opt_F = []
    mmpp_opt_t = []

    for i, power in enumerate(powers_uw):

        true_params = shields_2015_params(float(power))

        data = prepare_calibration_and_test_data(
            true_params=true_params,
            readout_times_us=readout_times_us,
            n_calibration_shots_per_state=n_calibration_shots_per_state,
            n_test_shots_per_state=n_test_shots_per_state,
            seed=seed + 100_000 * i,
        )

        th = evaluate_threshold_vs_time(data)
        mmpp = evaluate_mmpp_vs_time(data, true_params)

        threshold_F.append(th["opt_test_fidelity"])
        threshold_t.append(th["opt_time_us"])
        threshold_n.append(th["opt_threshold"])

        mmpp_same_F.append(
            mmpp["test_fidelity"][th["opt_idx"]]
        )
        mmpp_opt_F.append(mmpp["opt_test_fidelity"])
        mmpp_opt_t.append(mmpp["opt_time_us"])

        print(
            f"P={power:7.3f} uW | "
            f"threshold: tR={th['opt_time_us']:9.3f} us, "
            f"n>={th['opt_threshold']:2d}, "
            f"F={th['opt_test_fidelity']:.5f} | "
            f"MMPP @ same tR="
            f"{mmpp['test_fidelity'][th['opt_idx']]:.5f} | "
            f"MMPP optimum: tR={mmpp['opt_time_us']:9.3f} us, "
            f"F={mmpp['opt_test_fidelity']:.5f}"
        )

    result = {
        "powers_uw": powers_uw,
        "threshold_opt_fidelity": np.asarray(threshold_F),
        "threshold_opt_time_us": np.asarray(threshold_t),
        "threshold_opt_n": np.asarray(threshold_n),
        "mmpp_at_threshold_time_fidelity": np.asarray(mmpp_same_F),
        "mmpp_opt_fidelity": np.asarray(mmpp_opt_F),
        "mmpp_opt_time_us": np.asarray(mmpp_opt_t),
    }

    if make_plots:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))

        ax.plot(
            powers_uw,
            result["threshold_opt_fidelity"],
            "o-",
            label="optimized threshold",
        )
        ax.plot(
            powers_uw,
            result["mmpp_at_threshold_time_fidelity"],
            "s-",
            label="MMPP at threshold-selected $t_R$",
        )
        ax.plot(
            powers_uw,
            result["mmpp_opt_fidelity"],
            "^-",
            label="optimized MMPP",
        )

        ax.set_xlabel("594-nm readout power (uW)")
        ax.set_ylabel(r"initial charge readout fidelity $F_C$")
        ax.set_ylim(0.48, 1.01)
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_title("Shields-style power sweep")

        fig.tight_layout()
        plt.show()

    return result


# =============================================================================
# Study 2: readout-time survey
# =============================================================================


def run_readout_time_sweep(
    base_power_uw: float,
    readout_times_us: Sequence[float],
    detection_efficiency: float = 1.0,
    efficiency_model: str = "thin_all_counts",
    n_calibration_shots_per_state: int = 500,
    n_test_shots_per_state: int = 2000,
    seed: int = 12345,
    make_plots: bool = True,
) -> dict:
    """
    Survey initial-charge readout performance versus total readout time.

    This is the readout-duration survey restored from the original project,
    adapted to the current INITIAL-STATE discrimination benchmark.

    At every t_R:
        * the total-count threshold is chosen using CALIBRATION data only;
        * the exact event-time MMPP uses the physical parameters and an
          equal-prior LLR cutoff of zero;
        * both methods are evaluated on the SAME independent TEST shots.

    The physical laser power and detection efficiency are fixed throughout
    the sweep. Only the amount of photon record made available to each
    classifier changes with t_R.

    Notes
    -----
    Because a single simulated trajectory is generated out to the largest
    requested t_R and then truncated at each earlier time, all points for a
    given shot are naturally correlated. This is desirable for a clean
    readout-time comparison and greatly reduces Monte Carlo noise between
    neighboring time points.
    """

    times_us = np.asarray(readout_times_us, dtype=float)

    if times_us.size == 0:
        raise ValueError("readout_times_us cannot be empty.")
    if np.any(~np.isfinite(times_us)) or np.any(times_us <= 0):
        raise ValueError("All readout times must be positive and finite.")

    # Sort once so plots and cumulative likelihood evaluation are well defined.
    times_us = np.unique(np.sort(times_us))

    base_params = shields_2015_params(base_power_uw)
    true_params = apply_detection_efficiency(
        base_params=base_params,
        efficiency=detection_efficiency,
        model=efficiency_model,
    )

    data = prepare_calibration_and_test_data(
        true_params=true_params,
        readout_times_us=times_us,
        n_calibration_shots_per_state=n_calibration_shots_per_state,
        n_test_shots_per_state=n_test_shots_per_state,
        seed=seed,
    )

    threshold = evaluate_threshold_vs_time(data)
    mmpp = evaluate_mmpp_vs_time(data, true_params)

    threshold_F = np.asarray(threshold["test_fidelity"], dtype=float)
    mmpp_F = np.asarray(mmpp["test_fidelity"], dtype=float)
    gain_pp = 100.0 * (mmpp_F - threshold_F)

    result = {
        "base_power_uw": float(base_power_uw),
        "detection_efficiency": float(detection_efficiency),
        "efficiency_model": efficiency_model,
        "true_params": true_params,
        "readout_times_us": times_us,
        "threshold_fidelity": threshold_F,
        "mmpp_fidelity": mmpp_F,
        "threshold_error": 1.0 - threshold_F,
        "mmpp_error": 1.0 - mmpp_F,
        "gain_percentage_points": gain_pp,
        "threshold_integer_cutoff": np.asarray(
            threshold["thresholds"], dtype=int
        ),
        "threshold_opt_time_us": float(threshold["opt_time_us"]),
        "threshold_opt_fidelity": float(threshold["opt_test_fidelity"]),
        "mmpp_opt_time_us": float(mmpp["opt_time_us"]),
        "mmpp_opt_fidelity": float(mmpp["opt_test_fidelity"]),
    }

    print("\nReadout-time survey")
    print("-------------------")
    print(f"physical power       = {base_power_uw:g} uW")
    print(f"detection efficiency = {detection_efficiency:g}")
    print("")

    for t, F_th, F_mmpp, gain, n_th in zip(
        times_us,
        threshold_F,
        mmpp_F,
        gain_pp,
        threshold["thresholds"],
    ):
        print(
            f"tR={t:10.4g} us | "
            f"threshold n>={int(n_th):2d}, F={F_th:.5f} | "
            f"MMPP F={F_mmpp:.5f} | "
            f"gain={gain:+.3f} pp"
        )

    print("")
    print(
        f"threshold calibration-selected optimum: "
        f"tR={threshold['opt_time_us']:.6g} us, "
        f"test F={threshold['opt_test_fidelity']:.6f}"
    )
    print(
        f"MMPP calibration-selected optimum:      "
        f"tR={mmpp['opt_time_us']:.6g} us, "
        f"test F={mmpp['opt_test_fidelity']:.6f}"
    )

    if make_plots:

        # Fidelity vs readout time.
        fig1, ax1 = plt.subplots(figsize=(8.0, 5.2))

        ax1.plot(
            times_us,
            threshold_F,
            "o-",
            label="total-count threshold",
        )
        ax1.plot(
            times_us,
            mmpp_F,
            "s-",
            label="exact event-time MMPP",
        )

        ax1.set_xscale("log")
        ax1.set_xlabel(r"readout time $t_R$ ($\mu$s)")
        ax1.set_ylabel(r"initial charge readout fidelity $F_C$")
        ax1.set_ylim(0.48, 1.01)
        ax1.grid(alpha=0.2)
        ax1.legend()
        ax1.set_title(
            f"Readout-time survey at {base_power_uw:g} uW, "
            f"efficiency={detection_efficiency:g}"
        )
        fig1.tight_layout()

        # Error-rate plot is often more sensitive to small improvements.
        fig2, ax2 = plt.subplots(figsize=(8.0, 5.2))

        ax2.plot(
            times_us,
            100.0 * (1.0 - threshold_F),
            "o-",
            label="total-count threshold",
        )
        ax2.plot(
            times_us,
            100.0 * (1.0 - mmpp_F),
            "s-",
            label="exact event-time MMPP",
        )

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"readout time $t_R$ ($\mu$s)")
        ax2.set_ylabel("initial charge readout error (%)")
        ax2.grid(alpha=0.2)
        ax2.legend()
        ax2.set_title("Readout error versus measurement duration")
        fig2.tight_layout()

        # Direct processing advantage.
        fig3, ax3 = plt.subplots(figsize=(8.0, 5.2))

        ax3.plot(
            times_us,
            gain_pp,
            "o-",
        )
        ax3.axhline(0.0, ls="--", lw=1.0)

        ax3.set_xscale("log")
        ax3.set_xlabel(r"readout time $t_R$ ($\mu$s)")
        ax3.set_ylabel("MMPP - threshold fidelity (percentage points)")
        ax3.grid(alpha=0.2)
        ax3.set_title("Event-time information advantage versus readout time")
        fig3.tight_layout()

        result["figures"] = (fig1, fig2, fig3)
        plt.show()

    return result


# =============================================================================
# Study 3: original binwise state-tracking duration survey
# =============================================================================


def _simulate_mmpp_shot_with_state_events(
    duration_ms: float,
    initial_state: int,
    params: MMPPParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[float, int]]]:
    """Simulate exact photons plus the hidden-state switching trajectory."""

    params.validate()

    if duration_ms <= 0:
        raise ValueError("duration_ms must be > 0.")
    if initial_state not in (0, 1):
        raise ValueError("initial_state must be 0 (NV-) or 1 (NV0).")

    t = 0.0
    state = int(initial_state)
    clicks: list[float] = []
    state_events: list[tuple[float, int]] = [(0.0, state)]

    while t < duration_ms:
        if state == 0:
            switch_rate = params.gamma_minus_to_zero_khz
            photon_rate = params.lambda_minus_khz
        else:
            switch_rate = params.gamma_zero_to_minus_khz
            photon_rate = params.lambda_zero_khz

        total_rate = switch_rate + photon_rate
        if total_rate <= 0:
            break

        t_next = t + float(rng.exponential(1.0 / total_rate))
        if t_next >= duration_ms:
            break

        t = t_next

        if rng.random() < photon_rate / total_rate:
            clicks.append(t)
        else:
            state = 1 - state
            state_events.append((t, state))

    return np.asarray(clicks, dtype=float), state_events


def _state_at_time(
    state_events: Sequence[tuple[float, int]],
    time_ms: float,
) -> int:
    state = int(state_events[0][1])
    for t, s in state_events[1:]:
        if t > time_ms:
            break
        state = int(s)
    return state


def _uniform_edges_ms(
    duration_us: float,
    bin_us: float,
) -> np.ndarray:
    ratio = float(duration_us) / float(bin_us)
    n = int(round(ratio))
    if n < 1 or not np.isclose(
        n * float(bin_us),
        float(duration_us),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "For the legacy state-tracking survey, duration_us must be an "
            "integer multiple of the selected bin size."
        )
    return np.arange(n + 1, dtype=float) * float(bin_us) / 1000.0


def _legacy_default_threshold_bin_us(params: MMPPParams) -> float:
    """
    Reproduce the original project's historical threshold-bin convention:
    compute the duration where the continuous Poisson crossing nu=1 and use
    the smallest power of ten at least that large.

    This convention is retained ONLY for exact backward compatibility with
    the original state-tracking duration survey. It is not required by the
    newer initial-state threshold benchmark.
    """

    lb = float(params.lambda_minus_khz)
    ld = float(params.lambda_zero_khz)

    if not (lb > ld > 0):
        raise ValueError("Require lambda_minus > lambda_zero > 0.")

    dt_min_ms = np.log(lb / ld) / (lb - ld)
    dt_min_us = 1000.0 * dt_min_ms

    return float(10.0 ** np.ceil(np.log10(dt_min_us) - 1e-12))


def _poisson_crossing_nu(
    lambda_minus_khz: float,
    lambda_zero_khz: float,
    duration_ms: float,
) -> float:
    """Continuous equal-prior static-Poisson count crossing."""

    lb = float(lambda_minus_khz)
    ld = float(lambda_zero_khz)
    dt = float(duration_ms)

    if not (lb > ld >= 0) or dt <= 0:
        raise ValueError("Invalid rates or duration for Poisson threshold.")
    if ld == 0:
        return 0.0

    return float((lb - ld) / np.log(lb / ld) * dt)


def _propagate_no_click_vector(
    alpha: np.ndarray,
    dt_ms: float,
    params: MMPPParams,
) -> np.ndarray:
    """exp[(Q-Lambda)dt] @ alpha for a single current-state vector."""

    if dt_ms == 0.0:
        return np.asarray(alpha, dtype=float)

    eigvals, eigvecs, eigvecs_inv = (
        _cached_no_click_eigendecomposition(params)
    )
    coeff = eigvecs_inv @ np.asarray(alpha, dtype=float)
    out = eigvecs @ (np.exp(eigvals * float(dt_ms)) * coeff)
    out = np.real_if_close(out, tol=1000)
    return np.asarray(np.real(out), dtype=float)


def _legacy_tracking_predictions(
    timestamps_ms: np.ndarray,
    duration_us: float,
    designated_bin_us: float,
    threshold_bin_us: float,
    params: MMPPParams,
    prior: Sequence[float] = (0.5, 0.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return MMPP current-state predictions, threshold predictions, and the
    designated-bin edges for one trajectory.
    """

    designated_edges = _uniform_edges_ms(duration_us, designated_bin_us)
    threshold_edges = _uniform_edges_ms(duration_us, threshold_bin_us)

    ratio = float(threshold_bin_us) / float(designated_bin_us)
    n_designated_per_threshold = int(round(ratio))
    if n_designated_per_threshold < 1 or not np.isclose(
        ratio,
        n_designated_per_threshold,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "threshold_bin_us must be an integer multiple of "
            "designated_bin_us for the legacy survey."
        )

    ts = np.sort(np.asarray(timestamps_ms, dtype=float))
    Lam = params.Lambda

    alpha = np.asarray(prior, dtype=float).reshape(2)
    if np.any(alpha < 0) or not np.isclose(alpha.sum(), 1.0):
        raise ValueError("prior must be a length-2 probability vector.")

    mmpp_pred = np.empty(len(designated_edges) - 1, dtype=int)
    t_prev = 0.0
    click_idx = 0

    for b, bin_end in enumerate(designated_edges[1:]):
        bin_end = float(bin_end)

        while click_idx < len(ts) and ts[click_idx] < bin_end:
            t_click = float(ts[click_idx])

            alpha = _propagate_no_click_vector(
                alpha,
                t_click - t_prev,
                params,
            )
            scale = float(alpha.sum())
            if scale <= 0 or not np.isfinite(scale):
                raise FloatingPointError("Invalid MMPP no-click posterior.")
            alpha /= scale

            alpha = Lam @ alpha
            scale = float(alpha.sum())
            if scale <= 0 or not np.isfinite(scale):
                raise FloatingPointError("Invalid MMPP click posterior.")
            alpha /= scale

            t_prev = t_click
            click_idx += 1

        alpha = _propagate_no_click_vector(
            alpha,
            bin_end - t_prev,
            params,
        )
        scale = float(alpha.sum())
        if scale <= 0 or not np.isfinite(scale):
            raise FloatingPointError("Invalid MMPP posterior at bin end.")
        alpha /= scale

        mmpp_pred[b] = int(np.argmax(alpha))
        t_prev = bin_end

    threshold_counts = np.histogram(ts, bins=threshold_edges)[0]
    nu = _poisson_crossing_nu(
        params.lambda_minus_khz,
        params.lambda_zero_khz,
        float(threshold_bin_us) / 1000.0,
    )

    # State 0 = NV- if count > nu; state 1 = NV0 otherwise.
    threshold_coarse = np.where(threshold_counts > nu, 0, 1).astype(int)
    threshold_pred = np.repeat(
        threshold_coarse,
        n_designated_per_threshold,
    )

    if len(threshold_pred) != len(mmpp_pred):
        raise RuntimeError("Legacy threshold/designated grid length mismatch.")

    return mmpp_pred, threshold_pred, designated_edges


def run_state_tracking_duration_sweep(
    durations_us: Sequence[float],
    base_power_uw: float,
    designated_bin_us: float = 1.0,
    threshold_bin_us: float | None = None,
    detection_efficiency: float = 1.0,
    efficiency_model: str = "thin_all_counts",
    n_shots_per_state: int = 1000,
    seed: int = 12345,
    make_plots: bool = True,
) -> dict:
    """
    Exact legacy-style readout-duration survey from the original project.

    Unlike `run_readout_time_sweep`, this evaluates CURRENT HIDDEN-STATE
    tracking at every designated time bin, not initial-state discrimination.

    IMPORTANT: this intentionally preserves the ORIGINAL coarse-threshold
    behavior, where one decision made from the full threshold bin is repeated
    over all designated sub-bins inside it. Therefore early sub-bins can use
    photons that arrive later in the same coarse threshold bin. Keep this mode
    for reproducing the historical survey, not as the preferred fair causal
    comparison.

    Fixed across the duration sweep:
        designated MMPP reporting bin
        threshold integration bin
        physical power / rates

    Only total trajectory duration changes.
    """

    durations = np.asarray(durations_us, dtype=float)
    if durations.size == 0 or np.any(durations <= 0):
        raise ValueError("durations_us must contain positive values.")

    base_params = shields_2015_params(base_power_uw)
    params = apply_detection_efficiency(
        base_params=base_params,
        efficiency=detection_efficiency,
        model=efficiency_model,
    )

    if threshold_bin_us is None:
        threshold_bin_us = _legacy_default_threshold_bin_us(params)

    threshold_bin_us = float(threshold_bin_us)
    designated_bin_us = float(designated_bin_us)

    mmpp_F = []
    threshold_F = []

    print("\nOriginal binwise state-tracking duration survey")
    print("-----------------------------------------------")
    print(f"physical power       = {base_power_uw:g} uW")
    print(f"detection efficiency = {detection_efficiency:g}")
    print(f"designated bin       = {designated_bin_us:g} us")
    print(f"threshold bin        = {threshold_bin_us:g} us")
    print("")

    for i, duration_us in enumerate(durations):
        # Validate grids before doing any simulation.
        _uniform_edges_ms(duration_us, designated_bin_us)
        _uniform_edges_ms(duration_us, threshold_bin_us)

        ratio = threshold_bin_us / designated_bin_us
        if not np.isclose(ratio, round(ratio), rtol=1e-10, atol=1e-10):
            raise ValueError(
                "threshold_bin_us must be an integer multiple of "
                "designated_bin_us."
            )

        rng = np.random.default_rng(seed + 1000 * i)

        truth_all = []
        mmpp_all = []
        threshold_all = []

        for initial_state in (0, 1):
            for _ in range(int(n_shots_per_state)):
                clicks, events = _simulate_mmpp_shot_with_state_events(
                    duration_ms=float(duration_us) / 1000.0,
                    initial_state=initial_state,
                    params=params,
                    rng=rng,
                )

                mmpp_pred, threshold_pred, designated_edges = (
                    _legacy_tracking_predictions(
                        timestamps_ms=clicks,
                        duration_us=float(duration_us),
                        designated_bin_us=designated_bin_us,
                        threshold_bin_us=threshold_bin_us,
                        params=params,
                    )
                )

                # Original code used the hidden state just before each
                # designated-bin end when truth_convention="end".
                truth = np.asarray(
                    [
                        _state_at_time(
                            events,
                            np.nextafter(float(t), -np.inf),
                        )
                        for t in designated_edges[1:]
                    ],
                    dtype=int,
                )

                truth_all.append(truth)
                mmpp_all.append(mmpp_pred)
                threshold_all.append(threshold_pred)

        truth_flat = np.concatenate(truth_all)
        mmpp_flat = np.concatenate(mmpp_all)
        threshold_flat = np.concatenate(threshold_all)

        F_mmpp = balanced_initial_state_fidelity(truth_flat, mmpp_flat)
        F_threshold = balanced_initial_state_fidelity(
            truth_flat,
            threshold_flat,
        )

        mmpp_F.append(F_mmpp)
        threshold_F.append(F_threshold)

        print(
            f"duration={duration_us:10.5g} us | "
            f"MMPP={F_mmpp:.5f} | "
            f"threshold={F_threshold:.5f} | "
            f"gain={100.0 * (F_mmpp - F_threshold):+.3f} pp"
        )

    mmpp_F = np.asarray(mmpp_F, dtype=float)
    threshold_F = np.asarray(threshold_F, dtype=float)

    result = {
        "durations_us": durations,
        "base_power_uw": float(base_power_uw),
        "detection_efficiency": float(detection_efficiency),
        "designated_bin_us": designated_bin_us,
        "threshold_bin_us": threshold_bin_us,
        "mmpp_fidelity": mmpp_F,
        "threshold_fidelity": threshold_F,
        "gain_percentage_points": 100.0 * (mmpp_F - threshold_F),
    }

    if make_plots:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))

        ax.plot(
            durations,
            mmpp_F,
            "o-",
            label="event-time MMPP state tracking",
        )
        ax.plot(
            durations,
            threshold_F,
            "s-",
            label="coarse-bin threshold state tracking",
        )

        ax.set_xscale("log")
        ax.set_xlabel(r"total readout duration ($\mu$s)")
        ax.set_ylabel("balanced binwise state-tracking fidelity")
        ax.set_ylim(0.48, 1.01)
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_title("Original readout-duration state-tracking survey")
        fig.tight_layout()

        result["figure"] = fig
        plt.show()

    return result


# =============================================================================
# Study 4: sparse detection-efficiency sweep
# =============================================================================


def run_sparse_efficiency_sweep(
    efficiencies: Sequence[float],
    base_power_uw: float,
    readout_times_us: Sequence[float],
    efficiency_model: str = "thin_all_counts",
    n_calibration_shots_per_state: int = 500,
    n_test_shots_per_state: int = 2000,
    seed: int = 12345,
    make_plots: bool = True,
) -> dict:

    efficiencies = np.sort(
        np.asarray(efficiencies, dtype=float)
    )

    base_params = shields_2015_params(base_power_uw)

    threshold_F = []
    threshold_t = []
    threshold_n = []

    mmpp_same_F = []
    mmpp_opt_F = []
    mmpp_opt_t = []

    for i, eta in enumerate(efficiencies):

        true_params = apply_detection_efficiency(
            base_params=base_params,
            efficiency=float(eta),
            model=efficiency_model,
        )

        data = prepare_calibration_and_test_data(
            true_params=true_params,
            readout_times_us=readout_times_us,
            n_calibration_shots_per_state=n_calibration_shots_per_state,
            n_test_shots_per_state=n_test_shots_per_state,
            seed=seed + 100_000 * i,
        )

        th = evaluate_threshold_vs_time(data)
        mmpp = evaluate_mmpp_vs_time(data, true_params)

        threshold_F.append(th["opt_test_fidelity"])
        threshold_t.append(th["opt_time_us"])
        threshold_n.append(th["opt_threshold"])

        mmpp_same_F.append(
            mmpp["test_fidelity"][th["opt_idx"]]
        )
        mmpp_opt_F.append(mmpp["opt_test_fidelity"])
        mmpp_opt_t.append(mmpp["opt_time_us"])

        print(
            f"eta={eta:8.5f} | "
            f"threshold F={th['opt_test_fidelity']:.5f}, "
            f"tR={th['opt_time_us']:9.3f} us | "
            f"MMPP same-time="
            f"{mmpp['test_fidelity'][th['opt_idx']]:.5f} | "
            f"MMPP opt={mmpp['opt_test_fidelity']:.5f}, "
            f"tR={mmpp['opt_time_us']:9.3f} us"
        )

    threshold_F = np.asarray(threshold_F)
    mmpp_same_F = np.asarray(mmpp_same_F)
    mmpp_opt_F = np.asarray(mmpp_opt_F)

    result = {
        "efficiencies": efficiencies,
        "base_power_uw": float(base_power_uw),
        "threshold_opt_fidelity": threshold_F,
        "threshold_opt_time_us": np.asarray(threshold_t),
        "threshold_opt_n": np.asarray(threshold_n),
        "mmpp_at_threshold_time_fidelity": mmpp_same_F,
        "mmpp_opt_fidelity": mmpp_opt_F,
        "mmpp_opt_time_us": np.asarray(mmpp_opt_t),
        "same_time_gain_pp": 100.0 * (mmpp_same_F - threshold_F),
        "optimized_gain_pp": 100.0 * (mmpp_opt_F - threshold_F),
    }

    if make_plots:

        fig1, ax1 = plt.subplots(figsize=(8.0, 5.2))

        ax1.plot(
            efficiencies,
            threshold_F,
            "o-",
            label="optimized threshold",
        )
        ax1.plot(
            efficiencies,
            mmpp_same_F,
            "s-",
            label="MMPP at threshold-selected $t_R$",
        )
        ax1.plot(
            efficiencies,
            mmpp_opt_F,
            "^-",
            label="optimized MMPP",
        )

        ax1.set_xscale("log")
        ax1.set_xlabel("relative detection efficiency")
        ax1.set_ylabel(r"initial charge readout fidelity $F_C$")
        ax1.set_ylim(0.48, 1.01)
        ax1.grid(alpha=0.2)
        ax1.legend()
        ax1.set_title(
            f"Sparse-photon benchmark at {base_power_uw:g} uW"
        )
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(8.0, 5.2))

        ax2.plot(
            efficiencies,
            result["same_time_gain_pp"],
            "o-",
            label="same $t_R$",
        )
        ax2.plot(
            efficiencies,
            result["optimized_gain_pp"],
            "s-",
            label="each method optimized",
        )
        ax2.axhline(0.0, ls="--", lw=1.0)

        ax2.set_xscale("log")
        ax2.set_xlabel("relative detection efficiency")
        ax2.set_ylabel("MMPP fidelity gain (percentage points)")
        ax2.grid(alpha=0.2)
        ax2.legend()
        ax2.set_title("MMPP advantage as photon counts become sparse")
        fig2.tight_layout()

        plt.show()

    return result


# =============================================================================
# Study 5: 1D switching-rate mismatch
# =============================================================================


def run_switching_mismatch_1d(
    mismatch_factors: Sequence[float],
    base_power_uw: float,
    detection_efficiency: float,
    readout_times_us: Sequence[float],
    efficiency_model: str = "thin_all_counts",
    n_calibration_shots_per_state: int = 500,
    n_test_shots_per_state: int = 2000,
    seed: int = 12345,
    make_plots: bool = True,
) -> dict:
    """
    Generate ONE dataset using true rates, then reuse exactly the same photon
    records while changing the switching rates given to the MMPP.

    Three perturbations are tested:

    A. common scale:
        Gamma_-0_used = a * Gamma_-0_true
        Gamma_0-_used = a * Gamma_0-_true

    B. NV- -> NV0 only:
        Gamma_-0_used = a * Gamma_-0_true
        Gamma_0-_used = Gamma_0-_true

    C. NV0 -> NV- only:
        Gamma_-0_used = Gamma_-0_true
        Gamma_0-_used = a * Gamma_0-_true

    Photon-emission rates supplied to the MMPP remain EXACTLY correct.

    The reference time is the threshold-optimal t_R selected on calibration
    data. This means threshold and every MMPP curve are compared using the
    same measurement duration.
    """

    factors = np.asarray(mismatch_factors, dtype=float)

    if np.any(factors <= 0):
        raise ValueError("All mismatch factors must be positive.")

    base_params = shields_2015_params(base_power_uw)

    true_params = apply_detection_efficiency(
        base_params=base_params,
        efficiency=detection_efficiency,
        model=efficiency_model,
    )

    data = prepare_calibration_and_test_data(
        true_params=true_params,
        readout_times_us=readout_times_us,
        n_calibration_shots_per_state=n_calibration_shots_per_state,
        n_test_shots_per_state=n_test_shots_per_state,
        seed=seed,
    )

    threshold = evaluate_threshold_vs_time(data)
    oracle_mmpp = evaluate_mmpp_vs_time(data, true_params)

    ref_idx = threshold["opt_idx"]
    ref_time_us = threshold["opt_time_us"]

    threshold_ref_F = threshold["test_fidelity"][ref_idx]
    oracle_ref_F = oracle_mmpp["test_fidelity"][ref_idx]

    common_F = np.empty(len(factors), dtype=float)
    minus_to_zero_F = np.empty(len(factors), dtype=float)
    zero_to_minus_F = np.empty(len(factors), dtype=float)

    print("\nSwitching-rate mismatch study")
    print("--------------------------------")
    print(f"physical power           = {base_power_uw:g} uW")
    print(f"detection efficiency     = {detection_efficiency:g}")
    print(f"reference t_R            = {ref_time_us:.6g} us")
    print(f"threshold reference F    = {threshold_ref_F:.6f}")
    print(f"oracle MMPP reference F  = {oracle_ref_F:.6f}")
    print("")

    for i, factor in enumerate(factors):

        common_params = scale_model_parameters(
            true_params,
            gamma_minus_to_zero_scale=float(factor),
            gamma_zero_to_minus_scale=float(factor),
        )

        minus_only_params = scale_model_parameters(
            true_params,
            gamma_minus_to_zero_scale=float(factor),
            gamma_zero_to_minus_scale=1.0,
        )

        zero_only_params = scale_model_parameters(
            true_params,
            gamma_minus_to_zero_scale=1.0,
            gamma_zero_to_minus_scale=float(factor),
        )

        common_F[i] = evaluate_mmpp_at_single_time(
            shots=data["test_shots"],
            labels=data["test_labels"],
            time_us=ref_time_us,
            inference_params=common_params,
        )

        minus_to_zero_F[i] = evaluate_mmpp_at_single_time(
            shots=data["test_shots"],
            labels=data["test_labels"],
            time_us=ref_time_us,
            inference_params=minus_only_params,
        )

        zero_to_minus_F[i] = evaluate_mmpp_at_single_time(
            shots=data["test_shots"],
            labels=data["test_labels"],
            time_us=ref_time_us,
            inference_params=zero_only_params,
        )

        print(
            f"factor={factor:6.3f} | "
            f"both={common_F[i]:.5f} | "
            f"Gamma_-0 only={minus_to_zero_F[i]:.5f} | "
            f"Gamma_0- only={zero_to_minus_F[i]:.5f}"
        )

    result = {
        "mismatch_factors": factors,
        "base_power_uw": float(base_power_uw),
        "detection_efficiency": float(detection_efficiency),
        "true_params": true_params,
        "reference_time_us": float(ref_time_us),
        "threshold_fidelity": float(threshold_ref_F),
        "oracle_mmpp_fidelity": float(oracle_ref_F),
        "common_scale_fidelity": common_F,
        "gamma_minus_to_zero_only_fidelity": minus_to_zero_F,
        "gamma_zero_to_minus_only_fidelity": zero_to_minus_F,
        "common_scale_gain_vs_threshold_pp":
            100.0 * (common_F - threshold_ref_F),
        "gamma_minus_to_zero_gain_vs_threshold_pp":
            100.0 * (minus_to_zero_F - threshold_ref_F),
        "gamma_zero_to_minus_gain_vs_threshold_pp":
            100.0 * (zero_to_minus_F - threshold_ref_F),
    }

    if make_plots:

        fig1, ax1 = plt.subplots(figsize=(8.2, 5.3))

        ax1.plot(
            factors,
            common_F,
            "o-",
            label="both switching rates scaled",
        )

        ax1.plot(
            factors,
            minus_to_zero_F,
            "s-",
            label=r"only $\Gamma_{-\to0}$ scaled",
        )

        ax1.plot(
            factors,
            zero_to_minus_F,
            "^-",
            label=r"only $\Gamma_{0\to-}$ scaled",
        )

        ax1.axhline(
            threshold_ref_F,
            ls="--",
            label="threshold",
        )

        ax1.axhline(
            oracle_ref_F,
            ls=":",
            label="oracle MMPP",
        )

        ax1.axvline(
            1.0,
            ls="--",
            lw=1.0,
        )

        ax1.set_xlabel(
            r"switching-rate scale factor "
            r"$\hat{\Gamma}/\Gamma_{\rm true}$"
        )
        ax1.set_xscale("log")
        ax1.set_ylabel(r"initial charge readout fidelity $F_C$")
        ax1.grid(alpha=0.2)
        ax1.legend()
        ax1.set_title(
            "MMPP robustness to switching-rate estimation error"
        )

        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(8.2, 5.3))

        ax2.plot(
            factors,
            result["common_scale_gain_vs_threshold_pp"],
            "o-",
            label="both switching rates scaled",
        )

        ax2.plot(
            factors,
            result["gamma_minus_to_zero_gain_vs_threshold_pp"],
            "s-",
            label=r"only $\Gamma_{-\to0}$ scaled",
        )

        ax2.plot(
            factors,
            result["gamma_zero_to_minus_gain_vs_threshold_pp"],
            "^-",
            label=r"only $\Gamma_{0\to-}$ scaled",
        )
        ax2.set_xscale("log")

        ax2.axhline(0.0, ls="--", lw=1.0)
        ax2.axvline(1.0, ls="--", lw=1.0)

        ax2.set_xlabel(
            r"switching-rate scale factor "
            r"$\hat{\Gamma}/\Gamma_{\rm true}$"
        )
        ax2.set_ylabel(
            "MMPP - threshold fidelity (percentage points)"
        )
        ax2.grid(alpha=0.2)
        ax2.legend()
        ax2.set_title(
            "How much switching-rate error can MMPP tolerate?"
        )

        fig2.tight_layout()

        plt.show()

    return result


# =============================================================================
# Study 6: independent two-rate switching mismatch heatmap
# =============================================================================


def run_switching_mismatch_2d(
    gamma_minus_to_zero_factors: Sequence[float],
    gamma_zero_to_minus_factors: Sequence[float],
    base_power_uw: float,
    detection_efficiency: float,
    readout_times_us: Sequence[float],
    efficiency_model: str = "thin_all_counts",
    n_calibration_shots_per_state: int = 300,
    n_test_shots_per_state: int = 1000,
    seed: int = 12345,
    make_plots: bool = True,
    heatmap_color_percentile: float = 99.0,
) -> dict:
    """
    Robustness map:

        x = Gamma_-0_used / Gamma_-0_true
        y = Gamma_0-_used / Gamma_0-_true
        color = F_MMPP - F_threshold

    Data are generated ONCE with the true model. Only inference parameters
    change across the grid.

    The threshold-optimal t_R is used for every grid point.
    """

    x_factors = np.sort(
        np.asarray(
            gamma_minus_to_zero_factors,
            dtype=float,
        )
    )

    y_factors = np.sort(
        np.asarray(
            gamma_zero_to_minus_factors,
            dtype=float,
        )
    )

    if np.any(x_factors <= 0) or np.any(y_factors <= 0):
        raise ValueError("All mismatch factors must be positive.")

    base_params = shields_2015_params(base_power_uw)

    true_params = apply_detection_efficiency(
        base_params=base_params,
        efficiency=detection_efficiency,
        model=efficiency_model,
    )

    data = prepare_calibration_and_test_data(
        true_params=true_params,
        readout_times_us=readout_times_us,
        n_calibration_shots_per_state=n_calibration_shots_per_state,
        n_test_shots_per_state=n_test_shots_per_state,
        seed=seed,
    )

    threshold = evaluate_threshold_vs_time(data)
    oracle_mmpp = evaluate_mmpp_vs_time(data, true_params)

    ref_idx = threshold["opt_idx"]
    ref_time_us = threshold["opt_time_us"]

    threshold_F = float(
        threshold["test_fidelity"][ref_idx]
    )

    oracle_F = float(
        oracle_mmpp["test_fidelity"][ref_idx]
    )

    fidelity = np.empty(
        (len(y_factors), len(x_factors)),
        dtype=float,
    )

    print("\n2D switching mismatch map")
    print("-------------------------")
    print(f"reference t_R = {ref_time_us:.6g} us")
    print(f"threshold F   = {threshold_F:.6f}")
    print(f"oracle MMPP F = {oracle_F:.6f}")

    for iy, y_factor in enumerate(y_factors):

        print(
            f"row {iy + 1}/{len(y_factors)}: "
            f"Gamma_0- factor = {y_factor:.4g}"
        )

        for ix, x_factor in enumerate(x_factors):

            inference_params = scale_model_parameters(
                true_params,
                gamma_minus_to_zero_scale=float(x_factor),
                gamma_zero_to_minus_scale=float(y_factor),
            )

            fidelity[iy, ix] = evaluate_mmpp_at_single_time(
                shots=data["test_shots"],
                labels=data["test_labels"],
                time_us=ref_time_us,
                inference_params=inference_params,
            )

    gain_pp = 100.0 * (
        fidelity - threshold_F
    )

    result = {
        "gamma_minus_to_zero_factors": x_factors,
        "gamma_zero_to_minus_factors": y_factors,
        "base_power_uw": float(base_power_uw),
        "detection_efficiency": float(detection_efficiency),
        "reference_time_us": float(ref_time_us),
        "threshold_fidelity": threshold_F,
        "oracle_mmpp_fidelity": oracle_F,
        "mmpp_fidelity": fidelity,
        "gain_vs_threshold_pp": gain_pp,
    }

    if make_plots:

        fig, ax = plt.subplots(figsize=(7.5, 6.0))

        finite_abs = np.abs(gain_pp[np.isfinite(gain_pp)])

        if finite_abs.size == 0:
            raise ValueError("2D gain map contains no finite values.")

        pct = float(heatmap_color_percentile)
        if not (0.0 < pct <= 100.0):
            raise ValueError(
                "heatmap_color_percentile must satisfy 0 < percentile <= 100."
            )

        # Robust color normalization:
        # the underlying data are NOT modified. Only the displayed colormap
        # range is limited so one extreme grid point cannot flatten the rest
        # of the map. Values outside +/-vmax are clipped by the colormap and
        # explicitly marked below.
        vmax = float(np.percentile(finite_abs, pct))

        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.max(finite_abs))

        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0

        clipped_mask = np.abs(gain_pp) > vmax

        # Always report the most extreme grid point numerically.
        extreme_flat = int(np.nanargmax(np.abs(gain_pp)))
        extreme_iy, extreme_ix = np.unravel_index(
            extreme_flat,
            gain_pp.shape,
        )
        extreme_value = float(gain_pp[extreme_iy, extreme_ix])
        extreme_x = float(x_factors[extreme_ix])
        extreme_y = float(y_factors[extreme_iy])

        print("")
        print("2D heatmap display diagnostics")
        print("------------------------------")
        print(
            f"largest |gain| point: "
            f"Gamma_-0 factor={extreme_x:.6g}, "
            f"Gamma_0- factor={extreme_y:.6g}, "
            f"gain={extreme_value:+.6f} pp"
        )
        print(
            f"color scale uses {pct:g}th percentile of |gain|: "
            f"+/-{vmax:.6f} pp"
        )
        print(
            f"grid cells clipped only for display: "
            f"{int(np.count_nonzero(clipped_mask))}/{gain_pp.size}"
        )

        # Use pcolormesh rather than imshow because the mismatch factors are
        # logarithmically spaced. imshow assumes linearly spaced image pixels.
        def log_cell_edges(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=float)

            if np.any(values <= 0):
                raise ValueError(
                    "Log-scale heatmap factors must all be positive."
                )

            if len(values) == 1:
                return np.array(
                    [
                        values[0] / np.sqrt(10.0),
                        values[0] * np.sqrt(10.0),
                    ],
                    dtype=float,
                )

            log_values = np.log(values)
            log_edges = np.empty(len(values) + 1, dtype=float)

            log_edges[1:-1] = 0.5 * (
                log_values[:-1] + log_values[1:]
            )

            log_edges[0] = (
                log_values[0]
                - 0.5 * (log_values[1] - log_values[0])
            )

            log_edges[-1] = (
                log_values[-1]
                + 0.5 * (log_values[-1] - log_values[-2])
            )

            return np.exp(log_edges)

        x_edges = log_cell_edges(x_factors)
        y_edges = log_cell_edges(y_factors)

        im = ax.pcolormesh(
            x_edges,
            y_edges,
            gain_pp,
            shading="flat",
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        # Requested mismatch range: 0.01x to 100x.
        ax.set_xlim(1.0e-2, 1.0e2)
        ax.set_ylim(1.0e-2, 1.0e2)

        major_ticks = [1.0e-2, 1.0e-1, 1.0, 10.0, 100.0]
        major_tick_labels = ["0.01", "0.1", "1", "10", "100"]

        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_tick_labels)
        ax.set_yticks(major_ticks)
        ax.set_yticklabels(major_tick_labels)

        ax.scatter(
            [1.0],
            [1.0],
            marker="x",
            s=80,
            label="true parameters",
        )

        ax.set_xlabel(
            r"$\hat{\Gamma}_{-\to0}/\Gamma_{-\to0}^{true}$"
        )
        ax.set_ylabel(
            r"$\hat{\Gamma}_{0\to-}/\Gamma_{0\to-}^{true}$"
        )
        ax.set_title(
            "MMPP advantage under switching-rate mismatch"
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
            extend="both" if np.any(clipped_mask) else "neither",
        )
        cbar.set_label(
            "MMPP - threshold fidelity (percentage points)"
        )

        # Mark values that lie outside the displayed color range so they are
        # not silently hidden by the robust normalization.
        if np.any(clipped_mask):
            clipped_y, clipped_x = np.where(clipped_mask)
            ax.scatter(
                x_factors[clipped_x],
                y_factors[clipped_y],
                facecolors="none",
                edgecolors="black",
                s=55,
                linewidths=1.2,
                label="outside color scale",
            )

        if np.min(gain_pp) <= 0.0 <= np.max(gain_pp):
            X, Y = np.meshgrid(x_factors, y_factors)
            ax.contour(
                X,
                Y,
                gain_pp,
                levels=[0.0],
                linewidths=1.5,
            )

        ax.legend()
        fig.tight_layout()

        plt.show()

    return result


# =============================================================================
# Study 7: optional emission-rate mismatch
# =============================================================================


def run_emission_mismatch_1d(
    mismatch_factors: Sequence[float],
    base_power_uw: float,
    detection_efficiency: float,
    readout_times_us: Sequence[float],
    efficiency_model: str = "thin_all_counts",
    n_calibration_shots_per_state: int = 500,
    n_test_shots_per_state: int = 2000,
    seed: int = 12345,
    make_plots: bool = True,
) -> dict:
    """
    Companion robustness test.

    Switching rates supplied to MMPP are correct.
    Photon-emission rates supplied to MMPP are deliberately mismatched.

    This is kept separate from switching mismatch so that the source of any
    degradation is interpretable.
    """

    factors = np.asarray(mismatch_factors, dtype=float)

    if np.any(factors <= 0):
        raise ValueError("All mismatch factors must be positive.")

    base_params = shields_2015_params(base_power_uw)

    true_params = apply_detection_efficiency(
        base_params=base_params,
        efficiency=detection_efficiency,
        model=efficiency_model,
    )

    data = prepare_calibration_and_test_data(
        true_params=true_params,
        readout_times_us=readout_times_us,
        n_calibration_shots_per_state=n_calibration_shots_per_state,
        n_test_shots_per_state=n_test_shots_per_state,
        seed=seed,
    )

    threshold = evaluate_threshold_vs_time(data)
    oracle_mmpp = evaluate_mmpp_vs_time(data, true_params)

    ref_idx = threshold["opt_idx"]
    ref_time_us = threshold["opt_time_us"]

    threshold_F = threshold["test_fidelity"][ref_idx]
    oracle_F = oracle_mmpp["test_fidelity"][ref_idx]

    both_F = np.empty(len(factors), dtype=float)
    minus_F = np.empty(len(factors), dtype=float)
    zero_F = np.empty(len(factors), dtype=float)

    for i, factor in enumerate(factors):

        both_params = scale_model_parameters(
            true_params,
            lambda_minus_scale=float(factor),
            lambda_zero_scale=float(factor),
        )

        minus_params = scale_model_parameters(
            true_params,
            lambda_minus_scale=float(factor),
            lambda_zero_scale=1.0,
        )

        zero_params = scale_model_parameters(
            true_params,
            lambda_minus_scale=1.0,
            lambda_zero_scale=float(factor),
        )

        both_F[i] = evaluate_mmpp_at_single_time(
            data["test_shots"],
            data["test_labels"],
            ref_time_us,
            both_params,
        )

        minus_F[i] = evaluate_mmpp_at_single_time(
            data["test_shots"],
            data["test_labels"],
            ref_time_us,
            minus_params,
        )

        zero_F[i] = evaluate_mmpp_at_single_time(
            data["test_shots"],
            data["test_labels"],
            ref_time_us,
            zero_params,
        )

    result = {
        "mismatch_factors": factors,
        "reference_time_us": float(ref_time_us),
        "threshold_fidelity": float(threshold_F),
        "oracle_mmpp_fidelity": float(oracle_F),
        "both_emission_rates_scaled_fidelity": both_F,
        "lambda_minus_only_fidelity": minus_F,
        "lambda_zero_only_fidelity": zero_F,
    }

    if make_plots:

        fig, ax = plt.subplots(figsize=(8.2, 5.3))

        ax.plot(
            factors,
            both_F,
            "o-",
            label="both emission rates scaled",
        )
        ax.plot(
            factors,
            minus_F,
            "s-",
            label=r"only $\lambda_-$ scaled",
        )
        ax.plot(
            factors,
            zero_F,
            "^-",
            label=r"only $\lambda_0$ scaled",
        )

        ax.axhline(
            threshold_F,
            ls="--",
            label="threshold",
        )

        ax.axhline(
            oracle_F,
            ls=":",
            label="oracle MMPP",
        )

        ax.axvline(1.0, ls="--", lw=1.0)

        ax.set_xlabel(
            r"emission-rate scale factor "
            r"$\hat{\lambda}/\lambda_{\rm true}$"
        )
        ax.set_ylabel(r"initial charge readout fidelity $F_C$")
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_title(
            "MMPP robustness to photon-rate calibration error"
        )

        fig.tight_layout()
        plt.show()

    return result


# =============================================================================
# Main configuration
# =============================================================================


if __name__ == "__main__":

    print(f"script version = {SCRIPT_VERSION}")

    # -------------------------------------------------------------------------
    # Select studies here.
    #
    # Add/remove names from this list. You no longer need separate Python files.
    #
    # Available:
    #     "power_sweep"
    #     "readout_time_sweep"
    #     "state_tracking_duration_sweep"
    #     "sparse_efficiency"
    #     "switching_mismatch_1d"
    #     "switching_mismatch_2d"
    #     "emission_mismatch_1d"
    #     "bayesian_mmpp"
    #     "hsmm_comparison"
    #     "hsmm_dwell_shape_sweep"
    # -------------------------------------------------------------------------

    RUN_STUDIES = [
        "hsmm_dwell_shape_sweep",
        # "hsmm_comparison",
        # "bayesian_mmpp",
        # "readout_time_sweep",
        # "state_tracking_duration_sweep",
        # "sparse_efficiency",
        # "switching_mismatch_1d",
        # "switching_mismatch_2d",
        # "emission_mismatch_1d",
        # "power_sweep",
    ]

    # -------------------------------------------------------------------------
    # Shared physical / Monte Carlo settings
    # -------------------------------------------------------------------------

    BASE_POWER_UW = 5.437

    EFFICIENCY_MODEL = "thin_all_counts"

    # Readout-time grid used to calibrate the threshold-optimal t_R and/or
    # MMPP-optimal t_R.
    READOUT_TIMES_US = np.geomspace(
        2.0,
        10_000.0,
        100,
    )

    # Development values.
    # For smooth final figures, increase e.g. to 2000 / 10000.
    N_CALIBRATION_SHOTS_PER_STATE = 500
    N_TEST_SHOTS_PER_STATE = 2000

    BASE_SEED = 12345

    # -------------------------------------------------------------------------
    # Readout-time survey
    #
    # These are the durations used in the original project survey. The survey
    # now compares INITIAL-state discrimination fidelity for MMPP and total
    # counting at each total readout duration.
    # -------------------------------------------------------------------------

    READOUT_TIME_SURVEY_POWER_UW = 10.0
    READOUT_TIME_SURVEY_DETECTION_EFFICIENCY = 1.0

    READOUT_TIME_SURVEY_US = np.array(
        [
            10.0,
            20.0,
            50.0,
            100.0,
            200.0,
            500.0,
            1000.0,
            2000.0,
            5000.0,
        ],
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Original binwise state-tracking duration survey
    #
    # These settings mirror the old duration-sweep concept. At 10 uW and
    # eta=1, the historical automatic threshold-bin rule gives ~10 us.
    # -------------------------------------------------------------------------

    STATE_TRACKING_SURVEY_POWER_UW = 10.0
    STATE_TRACKING_SURVEY_DETECTION_EFFICIENCY = 1.0
    STATE_TRACKING_DESIGNATED_BIN_US = 1.0
    STATE_TRACKING_THRESHOLD_BIN_US = None  # historical automatic rule
    STATE_TRACKING_DURATIONS_US = np.array(
        [
            10.0,
            20.0,
            50.0,
            100.0,
            200.0,
            500.0,
            1000.0,
            2000.0,
            5000.0,
        ],
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Sparse-efficiency study
    # -------------------------------------------------------------------------

    SPARSE_EFFICIENCIES = np.array(
        [
            1.0,
            0.7,
            0.5,
            0.3,
            0.2,
            0.1,
            0.07,
            0.05,
            0.03,
            0.02,
            0.01,
            0.007,
            0.005,
            0.003,
            0.002,
            0.001,
        ],
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Parameter-mismatch studies
    #
    # This is intentionally in the sparse regime by default.
    # Change to 1.0 to test the original Shields photon-rich regime.
    # -------------------------------------------------------------------------

    MISMATCH_DETECTION_EFFICIENCY = 1.0

    # 0.5 means filter assumes 50% of the true rate.
    # 1.0 means perfect/oracle parameters.
    # 1.5 means filter assumes 150% of the true rate.
    MISMATCH_FACTORS_1D = np.linspace(
        0.01,
        100,
        31,
    )

    # Keep this moderate because the 2D study is computationally heavier.
    # Logarithmic 2D mismatch grid from 0.01x to 100x.
    # Because the exponent range is symmetric (-2 to +2), 1.0 is included.
    MISMATCH_FACTORS_2D = np.geomspace(
        0.01,
        100.0,
        31,
    )

    # Robust 2D heatmap display range.
    # 99 means the largest ~1% of |gain| values do not set the color scale.
    # Those cells are still present and are marked with black open circles.
    # Set to 100.0 to recover the full raw min/max color scale.
    MISMATCH_HEATMAP_COLOR_PERCENTILE = 99.0

    # -------------------------------------------------------------------------
    # HSMM dwell-shape sweep
    #
    # k=1 is exponential. Larger k keeps the same mean dwell time but reduces
    # dwell-time variability:
    #
    #     CV(T) = 1 / sqrt(k)
    #
    # This sweep can be slow because the HSMM has 2k hidden phase states.
    # Start with the development shot counts below; increase them for final
    # figures after the trend is established.
    # -------------------------------------------------------------------------

    HSMM_SHAPE_SWEEP_VALUES = np.array(
        [
            1,
            2,
            3,
            4,
            6,
            8,
        ],
        dtype=int,
    )

    HSMM_SHAPE_SWEEP_POWER_UW = 5.437
    HSMM_SHAPE_SWEEP_DETECTION_EFFICIENCY = 1.0

    HSMM_SHAPE_SWEEP_READOUT_TIMES_US = np.geomspace(
        2.0,
        1500.0,
        36,
    )

    HSMM_SHAPE_SWEEP_N_CALIBRATION_SHOTS_PER_STATE = 150
    HSMM_SHAPE_SWEEP_N_TEST_SHOTS_PER_STATE = 500

    # -------------------------------------------------------------------------
    # HSMM comparison
    #
    # Two cases are always run:
    #
    #   1. Exponential dwell times:
    #          k_minus = k_zero = 1
    #      The HSMM should numerically collapse onto the ordinary MMPP.
    #
    #   2. Non-exponential dwell times:
    #          k_minus, k_zero > 1
    #      The mean dwell times are unchanged, but the dwell-time CV becomes
    #          CV = 1 / sqrt(k).
    #
    # k=4 therefore gives CV=0.5 rather than the exponential CV=1.
    # -------------------------------------------------------------------------

    HSMM_POWER_UW = 5.437
    HSMM_DETECTION_EFFICIENCY = 1.0

    HSMM_READOUT_TIMES_US = np.geomspace(
        2.0,
        3000.0,
        50,
    )

    HSMM_NONEXPONENTIAL_SHAPE_MINUS = 4
    HSMM_NONEXPONENTIAL_SHAPE_ZERO = 4

    HSMM_N_CALIBRATION_SHOTS_PER_STATE = 300
    HSMM_N_TEST_SHOTS_PER_STATE = 1000

    # -------------------------------------------------------------------------
    # Bayesian MMPP comparison
    #
    # The ordinary plug-in MMPP receives ONE nominal switching-rate estimate.
    # The Bayesian MMPP receives the SAME nominal center plus uncertainty around
    # it, represented by a finite Gauss-Hermite ensemble.
    #
    # With both nominal scales = 1.0, the plug-in MMPP is the oracle model.
    # In that special case a Bayesian mixture should not systematically beat
    # the plug-in model; it tests the cost of parameter uncertainty.
    #
    # To simulate a biased polyspectra estimate, try for example:
    #
    #     BAYESIAN_NOMINAL_GAMMA_MINUS_TO_ZERO_SCALE = 1.25
    #     BAYESIAN_NOMINAL_GAMMA_ZERO_TO_MINUS_SCALE = 0.80
    #
    # and compare whether marginalization recovers some robustness.
    # -------------------------------------------------------------------------

    BAYESIAN_POWER_UW = 5.437
    BAYESIAN_DETECTION_EFFICIENCY = 0.10

    BAYESIAN_READOUT_TIMES_US = np.geomspace(
        2.0,
        5000.0,
        60,
    )

    BAYESIAN_NOMINAL_GAMMA_MINUS_TO_ZERO_SCALE = 1.0
    BAYESIAN_NOMINAL_GAMMA_ZERO_TO_MINUS_SCALE = 1.0

    # Coefficient of variation of each synthetic switching-rate posterior.
    BAYESIAN_GAMMA_MINUS_TO_ZERO_CV = 0.30
    BAYESIAN_GAMMA_ZERO_TO_MINUS_CV = 0.30

    # 3 nodes/rate -> 9 parallel 2-state MMPP models.
    # 5 nodes/rate -> 25 models, more accurate but slower.
    BAYESIAN_QUADRATURE_NODES_PER_RATE = 3

    # Keep the first run moderate. Increase after confirming behavior.
    BAYESIAN_N_CALIBRATION_SHOTS_PER_STATE = 300
    BAYESIAN_N_TEST_SHOTS_PER_STATE = 1000

    # -------------------------------------------------------------------------
    # Shields power sweep
    # -------------------------------------------------------------------------

    POWER_SWEEP_UW = np.geomspace(
        0.875,
        15.0,
        15,
    )

    # -------------------------------------------------------------------------
    # Dispatch
    # -------------------------------------------------------------------------

    results = {}

    if "hsmm_dwell_shape_sweep" in RUN_STUDIES:
        results["hsmm_dwell_shape_sweep"] = (
            run_hsmm_dwell_shape_sweep(
                shape_values=
                    HSMM_SHAPE_SWEEP_VALUES,
                base_power_uw=
                    HSMM_SHAPE_SWEEP_POWER_UW,
                readout_times_us=
                    HSMM_SHAPE_SWEEP_READOUT_TIMES_US,
                detection_efficiency=
                    HSMM_SHAPE_SWEEP_DETECTION_EFFICIENCY,
                efficiency_model=
                    EFFICIENCY_MODEL,
                n_calibration_shots_per_state=
                    HSMM_SHAPE_SWEEP_N_CALIBRATION_SHOTS_PER_STATE,
                n_test_shots_per_state=
                    HSMM_SHAPE_SWEEP_N_TEST_SHOTS_PER_STATE,
                seed=BASE_SEED + 60_000_000,
                make_plots=True,
            )
        )

    if "hsmm_comparison" in RUN_STUDIES:
        results["hsmm_comparison"] = run_hsmm_comparison(
            base_power_uw=HSMM_POWER_UW,
            readout_times_us=HSMM_READOUT_TIMES_US,
            detection_efficiency=
                HSMM_DETECTION_EFFICIENCY,
            efficiency_model=EFFICIENCY_MODEL,
            nonexponential_shape_minus=
                HSMM_NONEXPONENTIAL_SHAPE_MINUS,
            nonexponential_shape_zero=
                HSMM_NONEXPONENTIAL_SHAPE_ZERO,
            n_calibration_shots_per_state=
                HSMM_N_CALIBRATION_SHOTS_PER_STATE,
            n_test_shots_per_state=
                HSMM_N_TEST_SHOTS_PER_STATE,
            seed=BASE_SEED + 50_000_000,
            make_plots=True,
        )

    if "bayesian_mmpp" in RUN_STUDIES:
        results["bayesian_mmpp"] = run_bayesian_mmpp_comparison(
            base_power_uw=BAYESIAN_POWER_UW,
            readout_times_us=BAYESIAN_READOUT_TIMES_US,
            detection_efficiency=BAYESIAN_DETECTION_EFFICIENCY,
            efficiency_model=EFFICIENCY_MODEL,
            nominal_gamma_minus_to_zero_scale=
                BAYESIAN_NOMINAL_GAMMA_MINUS_TO_ZERO_SCALE,
            nominal_gamma_zero_to_minus_scale=
                BAYESIAN_NOMINAL_GAMMA_ZERO_TO_MINUS_SCALE,
            gamma_minus_to_zero_cv=
                BAYESIAN_GAMMA_MINUS_TO_ZERO_CV,
            gamma_zero_to_minus_cv=
                BAYESIAN_GAMMA_ZERO_TO_MINUS_CV,
            quadrature_nodes_per_rate=
                BAYESIAN_QUADRATURE_NODES_PER_RATE,
            n_calibration_shots_per_state=
                BAYESIAN_N_CALIBRATION_SHOTS_PER_STATE,
            n_test_shots_per_state=
                BAYESIAN_N_TEST_SHOTS_PER_STATE,
            seed=BASE_SEED + 40_000_000,
            make_plots=True,
        )

    if "readout_time_sweep" in RUN_STUDIES:
        results["readout_time_sweep"] = run_readout_time_sweep(
            base_power_uw=READOUT_TIME_SURVEY_POWER_UW,
            readout_times_us=READOUT_TIME_SURVEY_US,
            detection_efficiency=
                READOUT_TIME_SURVEY_DETECTION_EFFICIENCY,
            efficiency_model=EFFICIENCY_MODEL,
            n_calibration_shots_per_state=
                N_CALIBRATION_SHOTS_PER_STATE,
            n_test_shots_per_state=
                N_TEST_SHOTS_PER_STATE,
            seed=BASE_SEED + 5_000_000,
            make_plots=True,
        )

    if "state_tracking_duration_sweep" in RUN_STUDIES:
        results["state_tracking_duration_sweep"] = (
            run_state_tracking_duration_sweep(
                durations_us=STATE_TRACKING_DURATIONS_US,
                base_power_uw=STATE_TRACKING_SURVEY_POWER_UW,
                designated_bin_us=STATE_TRACKING_DESIGNATED_BIN_US,
                threshold_bin_us=STATE_TRACKING_THRESHOLD_BIN_US,
                detection_efficiency=
                    STATE_TRACKING_SURVEY_DETECTION_EFFICIENCY,
                efficiency_model=EFFICIENCY_MODEL,
                n_shots_per_state=N_CALIBRATION_SHOTS_PER_STATE,
                seed=BASE_SEED + 6_000_000,
                make_plots=True,
            )
        )

    if "power_sweep" in RUN_STUDIES:
        results["power_sweep"] = run_power_sweep(
            powers_uw=POWER_SWEEP_UW,
            readout_times_us=READOUT_TIMES_US,
            n_calibration_shots_per_state=
                N_CALIBRATION_SHOTS_PER_STATE,
            n_test_shots_per_state=
                N_TEST_SHOTS_PER_STATE,
            seed=BASE_SEED,
            make_plots=True,
        )

    if "sparse_efficiency" in RUN_STUDIES:
        results["sparse_efficiency"] = run_sparse_efficiency_sweep(
            efficiencies=SPARSE_EFFICIENCIES,
            base_power_uw=BASE_POWER_UW,
            readout_times_us=READOUT_TIMES_US,
            efficiency_model=EFFICIENCY_MODEL,
            n_calibration_shots_per_state=
                N_CALIBRATION_SHOTS_PER_STATE,
            n_test_shots_per_state=
                N_TEST_SHOTS_PER_STATE,
            seed=BASE_SEED,
            make_plots=True,
        )

    if "switching_mismatch_1d" in RUN_STUDIES:
        results["switching_mismatch_1d"] = (
            run_switching_mismatch_1d(
                mismatch_factors=MISMATCH_FACTORS_1D,
                base_power_uw=BASE_POWER_UW,
                detection_efficiency=
                    MISMATCH_DETECTION_EFFICIENCY,
                readout_times_us=READOUT_TIMES_US,
                efficiency_model=EFFICIENCY_MODEL,
                n_calibration_shots_per_state=
                    N_CALIBRATION_SHOTS_PER_STATE,
                n_test_shots_per_state=
                    N_TEST_SHOTS_PER_STATE,
                seed=BASE_SEED + 10_000_000,
                make_plots=True,
            )
        )

    if "switching_mismatch_2d" in RUN_STUDIES:
        results["switching_mismatch_2d"] = (
            run_switching_mismatch_2d(
                gamma_minus_to_zero_factors=
                    MISMATCH_FACTORS_2D,
                gamma_zero_to_minus_factors=
                    MISMATCH_FACTORS_2D,
                base_power_uw=BASE_POWER_UW,
                detection_efficiency=
                    MISMATCH_DETECTION_EFFICIENCY,
                readout_times_us=READOUT_TIMES_US,
                efficiency_model=EFFICIENCY_MODEL,
                n_calibration_shots_per_state=300,
                n_test_shots_per_state=1000,
                seed=BASE_SEED + 20_000_000,
                make_plots=True,
                heatmap_color_percentile=
                    MISMATCH_HEATMAP_COLOR_PERCENTILE,
            )
        )

    if "emission_mismatch_1d" in RUN_STUDIES:
        results["emission_mismatch_1d"] = (
            run_emission_mismatch_1d(
                mismatch_factors=MISMATCH_FACTORS_1D,
                base_power_uw=BASE_POWER_UW,
                detection_efficiency=
                    MISMATCH_DETECTION_EFFICIENCY,
                readout_times_us=READOUT_TIMES_US,
                efficiency_model=EFFICIENCY_MODEL,
                n_calibration_shots_per_state=
                    N_CALIBRATION_SHOTS_PER_STATE,
                n_test_shots_per_state=
                    N_TEST_SHOTS_PER_STATE,
                seed=BASE_SEED + 30_000_000,
                make_plots=True,
            )
        )
