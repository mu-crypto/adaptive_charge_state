#!/usr/bin/env python3
"""
Regenerate every number quoted in results/README.md from the committed CSVs.

The writeup makes claims that span experiments -- the stopping-rule/statistic
factorization, the SNR-vs-sparsity collapse, the D'Anjou comparison -- and none
of those are produced by `summary`, which works one sweep at a time. Rather
than recompute them by hand each time the matrix is re-run, they live here.

    python results/analysis.py [--results results]

Reads only `speedup_table.csv` and `fidelity_ceilings.csv` under
results/<experiment>/<detector>/, so it needs no pickles and no simulation.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

# Speedups are t_threshold / t_method, so all three share the baseline and the
# ratios between them factor cleanly.
METHODS = {
    "count": "adaptive count",
    "fixed_count_mmpp": "fixed-count MMPP",
    "mmpp": "adaptive MMPP",
}
CEILING_METHODS = [
    "threshold",
    "adaptive_count",
    "fixed_count_mmpp",
    "adaptive_mmpp",
]

# Comparisons between operating points are only meaningful at matched headroom
# below the THRESHOLD's own ceiling: the baseline time diverges as the target
# approaches it, so an unmatched comparison mostly measures how close each
# point's grid happens to sit to its own turnover.
BAND = (0.02, 0.08)
HEADROOM_BANDS = [
    (0.00, 0.01),
    (0.01, 0.02),
    (0.02, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 1.00),
]


def _f(row: dict, key: str) -> float:
    v = row.get(key, "")
    if v in ("", None, "nan"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def load(results: Path) -> tuple[list[dict], dict]:
    """Every speedup row, and ceilings keyed by (experiment, detector, point)."""
    rows: list[dict] = []
    for path in sorted(results.glob("*/*/speedup_table.csv")):
        with path.open() as fh:
            rows.extend(csv.DictReader(fh))

    ceilings: dict = defaultdict(dict)
    for path in sorted(results.glob("*/*/fidelity_ceilings.csv")):
        with path.open() as fh:
            for r in csv.DictReader(fh):
                key = (r["experiment"], r["detector"], r["point"])
                ceilings[key][r["method"]] = _f(r, "max_balanced_fidelity")
    return rows, ceilings


def annotate(rows: list[dict], ceilings: dict) -> list[dict]:
    """Attach the threshold ceiling and the headroom of each target below it."""
    out = []
    for r in rows:
        key = (r["experiment"], r["detector"], r["point"])
        thr_max = ceilings.get(key, {}).get("threshold", float("nan"))
        r = dict(r)
        r["_thr_ceiling"] = thr_max
        r["_headroom"] = thr_max - _f(r, "target_fidelity")
        r["_ideal"] = r["detector"] == "ideal"
        r["_noisy"] = _f(r, "noise_sigma") > 0.0
        out.append(r)
    return out


def in_band(r: dict, lo: float, hi: float) -> bool:
    h = r["_headroom"]
    return np.isfinite(h) and lo <= h < hi


def med(vals) -> float:
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    return float(np.median(v)) if v.size else float("nan")


def ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares with an intercept, returning coefficients and R^2."""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return beta, r2


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# -----------------------------------------------------------------------------


def report_headroom_bands(rows: list[dict]) -> None:
    """Median speedup by distance below the threshold's ceiling."""
    section("Median speedup by headroom (ideal detector, no rate noise)")
    pool = [r for r in rows if r["_ideal"] and not r["_noisy"]]
    n_def = sum(1 for r in pool if np.isfinite(_f(r, "speedup_mmpp")))
    print(f"{n_def} rows where the adaptive-MMPP speedup is defined\n")
    print(f"{'headroom':>14} {'n':>5} {'a.MMPP':>9} {'fc.MMPP':>9} {'a.count':>9}")
    for lo, hi in HEADROOM_BANDS:
        sel = [r for r in pool if in_band(r, lo, hi)]
        if not sel:
            continue
        print(
            f"  [{lo:.2f}, {hi:.2f}) {len(sel):5d} "
            f"{med(_f(r, 'speedup_mmpp') for r in sel):9.3f} "
            f"{med(_f(r, 'speedup_fixed_count_mmpp') for r in sel):9.3f} "
            f"{med(_f(r, 'speedup_count') for r in sel):9.3f}"
        )


def report_factorization(rows: list[dict]) -> None:
    """
    The 2x2 read: which axis, stopping rule or decision statistic, buys speed.

    Methods 2 and 3 stop at identical instants, so 2 -> 3 isolates the
    statistic and 3 -> 4 isolates the stopping rule. Taken as medians of
    per-row ratios, which is paired at matched target.
    """
    section("Stopping rule vs decision statistic")
    pool = [r for r in rows if r["_ideal"] and not r["_noisy"]]

    steps = [
        ("threshold -> adaptive count", "stopping: time -> count", "speedup_count", None),
        (
            "adaptive count -> fixed-count MMPP",
            "statistic: count -> LLR",
            "speedup_fixed_count_mmpp",
            "speedup_count",
        ),
        (
            "fixed-count MMPP -> adaptive MMPP",
            "stopping: count -> LLR boundary",
            "speedup_mmpp",
            "speedup_fixed_count_mmpp",
        ),
    ]
    print(f"{'step':<36} {'what changes':<32} {'factor':>8} {'n':>6}")
    for label, what, num, den in steps:
        vals = []
        for r in pool:
            a = _f(r, num)
            b = 1.0 if den is None else _f(r, den)
            if np.isfinite(a) and np.isfinite(b) and b > 0:
                vals.append(a / b)
        print(f"{label:<36} {what:<32} {med(vals):8.3f} {len(vals):6d}")

    # A statistic swap at frozen stopping can only lift the fidelity curve, it
    # cannot shift it left. Quantify how rarely it shifts anything at all.
    gains = [
        _f(r, "speedup_fixed_count_mmpp") / _f(r, "speedup_count")
        for r in pool
        if np.isfinite(_f(r, "speedup_fixed_count_mmpp"))
        and _f(r, "speedup_count") > 0
    ]
    gains = np.asarray(gains)
    print(
        f"\nstatistic-only step above 1.01 in "
        f"{100 * float((gains > 1.01).mean()):.1f}% of {gains.size} rows"
    )

    section("Per-method speedup, and what the bootstrap resolves")
    print(f"{'method':<20} {'median':>8} {'CI>1':>7} {'CI<1':>7} {'max':>7} {'n':>6}")
    for suffix, name in METHODS.items():
        sp, lo, hi = [], [], []
        for r in pool:
            v = _f(r, f"speedup_{suffix}")
            if not np.isfinite(v):
                continue
            sp.append(v)
            lo.append(_f(r, f"speedup_{suffix}_ci_low"))
            hi.append(_f(r, f"speedup_{suffix}_ci_high"))
        sp, lo, hi = map(np.asarray, (sp, lo, hi))
        above = float((lo > 1.0).mean()) if sp.size else float("nan")
        below = float((hi < 1.0).mean()) if sp.size else float("nan")
        print(
            f"{name:<20} {np.median(sp):8.3f} {100 * above:6.0f}% "
            f"{100 * below:6.0f}% {sp.max():7.3f} {sp.size:6d}"
        )


def report_ceilings(rows: list[dict], ceilings: dict) -> None:
    """Where the statistic pays: the achievable fidelity, not the run time."""
    section("Ceiling gains along the same two axes")
    # Deduplicated: the same regime reached through several sweeps must not
    # be averaged in more than once.
    wanted = set(matched_rows(rows, dedup=True))
    keys = sorted(
        {
            (r["experiment"], r["detector"], r["point"])
            for r in rows
            if r["_ideal"]
            and not r["_noisy"]
            and (r["experiment"], r["point"]) in wanted
        }
    )
    stat, stop, thr_vs_fcm, thr_vs_cnt = [], [], 0, 0
    for k in keys:
        c = ceilings.get(k, {})
        if not all(m in c for m in CEILING_METHODS):
            continue
        stat.append(c["fixed_count_mmpp"] - c["adaptive_count"])
        stop.append(c["adaptive_mmpp"] - c["fixed_count_mmpp"])
        thr_vs_fcm += c["fixed_count_mmpp"] > c["threshold"]
        thr_vs_cnt += c["adaptive_count"] > c["threshold"]
    n = len(stat)
    print(f"over {n} ideal-detector operating points\n")
    print(
        f"  adaptive count -> fixed-count MMPP (statistic) "
        f"{np.mean(stat):+.4f}"
    )
    print(
        f"  fixed-count MMPP -> adaptive MMPP (LLR stopping) "
        f"{np.mean(stop):+.4f}"
    )
    print(f"\n  fixed-count MMPP ceiling beats the threshold's at {thr_vs_fcm}/{n}")
    print(f"  adaptive count   ceiling beats the threshold's at {thr_vs_cnt}/{n}")


def _physical_key(r: dict) -> tuple:
    """
    The four dimensionless numbers that define an operating point.

    Several sweeps pass through the same physics: the 5.437 uW reference
    point is the eta = 1 end of the efficiency sweep, the sigma = 0 end of
    three noise sweeps and the `moderate` demo point, and 15 uW is both the
    top of the power sweep and the `high_flux` demo point. Counting those as
    five independent points would put five copies of one regime into a
    pooled regression over ~25, which is not a small distortion.
    """
    return (
        round(_f(r, "photons_per_bright_dwell"), 6),
        round(_f(r, "contrast"), 6),
        round(_f(r, "switching_ratio"), 6),
        round(_f(r, "snr_per_bright_dwell"), 6),
    )


def matched_rows(rows: list[dict], dedup: bool = False) -> dict:
    """
    Per operating point, the matched-headroom summary the README tabulates.

    With `dedup`, operating points that are physically the same regime are
    collapsed to one, keeping the first sweep alphabetically. Use it for
    anything that pools points; leave it off for the per-sweep table, where
    the repetition is the point.
    """
    per = defaultdict(list)
    for r in rows:
        if r["_ideal"] and not r["_noisy"]:
            per[(r["experiment"], r["point"])].append(r)

    if dedup:
        seen: dict = {}
        for key in sorted(per):
            phys = _physical_key(per[key][0])
            if phys not in seen:
                seen[phys] = key
        per = {k: v for k, v in per.items() if k in set(seen.values())}

    out = {}
    for key, rs in per.items():
        band = [r for r in rs if in_band(r, *BAND)]
        sp = [_f(r, "speedup_mmpp") for r in band]
        width = [
            _f(r, "speedup_mmpp_ci_high") - _f(r, "speedup_mmpp_ci_low")
            for r in band
        ]
        peak = max(
            (_f(r, "speedup_mmpp") for r in rs if np.isfinite(_f(r, "speedup_mmpp"))),
            default=float("nan"),
        )
        out[key] = {
            "n": _f(rs[0], "photons_per_bright_dwell"),
            "snr": _f(rs[0], "snr_per_bright_dwell"),
            "contrast": _f(rs[0], "contrast"),
            "ratio": _f(rs[0], "switching_ratio"),
            "speedup": med(sp),
            "ci_width": med(width),
            "peak": peak,
            "n_band": len(band),
        }
    return out


def report_matched(rows: list[dict], ceilings: dict) -> None:
    section(f"Matched headroom [{BAND[0]}, {BAND[1]}), ideal detector")
    table = matched_rows(rows)
    print(
        f"{'experiment':<11} {'point':<14} {'ph/dwell':>9} {'SNR':>7} "
        f"{'thr':>7} {'a.cnt':>7} {'fc.MMPP':>7} {'a.MMPP':>7} "
        f"{'speedup':>8} {'+-CI':>6} {'peak':>6}"
    )
    for (exp, pt), v in sorted(table.items()):
        c = ceilings.get((exp, "ideal", pt), {})
        print(
            f"{exp:<11} {pt:<14} {v['n']:9.2f} {v['snr']:7.2f} "
            f"{c.get('threshold', float('nan')):7.3f} "
            f"{c.get('adaptive_count', float('nan')):7.3f} "
            f"{c.get('fixed_count_mmpp', float('nan')):7.3f} "
            f"{c.get('adaptive_mmpp', float('nan')):7.3f} "
            f"{v['speedup']:8.3f} {v['ci_width']:6.2f} {v['peak']:6.2f}"
        )


def report_collapse(rows: list[dict], ceilings: dict) -> None:
    """
    Do the ceiling and the speedup collapse onto the same variable, or two?

    Pools every ideal-detector operating point from every sweep and regresses
    each figure of merit on log SNR, log sparsity, and both.
    """
    section("Collapse test: SNR against sparsity, pooled over all sweeps")
    table = matched_rows(rows, dedup=True)

    pts = []
    for (exp, pt), v in table.items():
        c = ceilings.get((exp, "ideal", pt), {})
        if "adaptive_mmpp" not in c:
            continue
        if not (np.isfinite(v["snr"]) and v["snr"] > 0 and v["n"] > 0):
            continue
        pts.append((np.log(v["snr"]), np.log(v["n"]), c["adaptive_mmpp"], v["speedup"]))

    log_snr = np.array([p[0] for p in pts])
    log_n = np.array([p[1] for p in pts])
    ceil = np.array([p[2] for p in pts])
    speed = np.array([p[3] for p in pts])
    ok = np.isfinite(speed)

    print(f"{len(pts)} operating points ({int(ok.sum())} with a resolved speedup)\n")
    print(f"{'target':<28} {'SNR only':>10} {'sparsity':>10} {'both':>10}")
    for name, y, mask in (
        ("fidelity ceiling", ceil, np.ones(len(pts), bool)),
        ("speedup at matched headroom", np.log(np.where(ok, speed, 1.0)), ok),
    ):
        _, r2_s = ols([log_snr[mask]], y[mask])
        _, r2_n = ols([log_n[mask]], y[mask])
        _, r2_b = ols([log_snr[mask], log_n[mask]], y[mask])
        print(f"{name:<28} {r2_s:10.3f} {r2_n:10.3f} {r2_b:10.3f}")

    beta, r2 = ols([log_n[ok]], np.log(speed[ok]))
    print(
        f"\nsparsity law: log(speedup) = {beta[0]:.3f} + {beta[1]:.3f} log(n)"
        f"   R^2 = {r2:.2f}, n = {int(ok.sum())}"
    )
    print(f"  a factor of 10 in photons per bright dwell buys {10 ** beta[1]:.2f}x")

    rho_c = np.corrcoef(
        np.argsort(np.argsort(log_snr)), np.argsort(np.argsort(ceil))
    )[0, 1]
    rho_n = np.corrcoef(
        np.argsort(np.argsort(log_n)), np.argsort(np.argsort(ceil))
    )[0, 1]
    print(f"  ceiling rank correlation: {rho_c:+.3f} with SNR, {rho_n:+.3f} with sparsity")


def report_danjou(rows: list[dict]) -> None:
    """D'Anjou gives 2x as the bound for decay readout. Does anything clear it?"""
    section("Rows exceeding the D'Anjou 2x bound (ideal detector)")
    # Deduplicated: the same regime appears in up to five sweeps, and a row
    # that crosses 2x would otherwise be counted once per sweep.
    wanted = set(matched_rows(rows, dedup=True))
    pool = [
        r
        for r in rows
        if r["_ideal"]
        and not r["_noisy"]
        and (r["experiment"], r["point"]) in wanted
    ]
    hits = [r for r in pool if _f(r, "speedup_mmpp") > 2.0]
    n_def = sum(1 for r in pool if np.isfinite(_f(r, "speedup_mmpp")))
    print(f"{len(hits)} of {n_def} rows\n")
    if hits:
        print(
            f"{'experiment':<11} {'point':<14} {'F*':>5} {'speedup':>8} "
            f"{'CI low':>7} {'CI high':>8} {'t_thr':>9} {'t_mmpp':>9}"
        )
        for r in sorted(hits, key=lambda r: -_f(r, "speedup_mmpp")):
            print(
                f"{r['experiment']:<11} {r['point']:<14} "
                f"{_f(r, 'target_fidelity'):5.2f} {_f(r, 'speedup_mmpp'):8.2f} "
                f"{_f(r, 'speedup_mmpp_ci_low'):7.2f} "
                f"{_f(r, 'speedup_mmpp_ci_high'):8.2f} "
                f"{_f(r, 't_threshold_us'):9.1f} "
                f"{_f(r, 't_adaptive_mmpp_us'):9.1f}"
            )
        n_clear = sum(1 for r in hits if _f(r, "speedup_mmpp_ci_low") > 2.0)
        print(f"\nCI lower bound clears 2x in {n_clear} of {len(hits)}")


def report_overhead(rows: list[dict]) -> None:
    """
    Adaptive readout shortens the measurement window only.

    At each power point take the peak-speedup row and dilute it by per-shot
    overhead, which is what an actual duty cycle does to the gain.
    """
    section("Peak speedup diluted by per-shot overhead (power sweep, ideal)")
    per = defaultdict(list)
    for r in rows:
        if r["experiment"] == "power" and r["_ideal"] and not r["_noisy"]:
            per[r["point"]].append(r)

    print(
        f"{'point':<12} {'F*':>5} {'t_thr':>9} {'t_mmpp':>9} "
        f"{'0us':>7} {'5us':>7} {'20us':>7} {'100us':>7}"
    )
    for pt in sorted(per, key=lambda p: _f(per[p][0], "power_uw")):
        best = max(
            (r for r in per[pt] if np.isfinite(_f(r, "speedup_mmpp"))),
            key=lambda r: _f(r, "speedup_mmpp"),
            default=None,
        )
        if best is None:
            continue
        t_thr = _f(best, "t_threshold_us")
        t_ada = _f(best, "t_adaptive_mmpp_us")
        cells = " ".join(
            f"{(t_thr + oh) / (t_ada + oh):7.2f}" for oh in (0.0, 5.0, 20.0, 100.0)
        )
        print(
            f"{pt:<12} {_f(best, 'target_fidelity'):5.2f} "
            f"{t_thr:9.1f} {t_ada:9.1f} {cells}"
        )


def report_detector(rows: list[dict], ceilings: dict) -> None:
    """Do dead time and afterpulsing overturn anything? Peaks and ceilings."""
    section("Detector settings: peak speedup and ceiling shift")
    dets = sorted({r["detector"] for r in rows if not r["_noisy"]})
    ideal = "ideal"
    others = [d for d in dets if d != ideal]

    peaks: dict = defaultdict(dict)
    for r in rows:
        if r["_noisy"]:
            continue
        v = _f(r, "speedup_mmpp")
        if not np.isfinite(v):
            continue
        key = (r["experiment"], r["point"])
        peaks[key][r["detector"]] = max(peaks[key].get(r["detector"], 0.0), v)

    print(f"{'experiment':<11} {'point':<14} " + " ".join(f"{d[:18]:>19}" for d in dets))
    for key in sorted(peaks):
        if not all(d in peaks[key] for d in dets):
            continue
        cells = " ".join(f"{peaks[key][d]:19.2f}" for d in dets)
        print(f"{key[0]:<11} {key[1]:<14} {cells}")

    print()
    for d in others:
        shifts = []
        for exp, det, pt in ceilings:
            if det != ideal:
                continue
            a = ceilings[(exp, ideal, pt)].get("adaptive_mmpp")
            b = ceilings.get((exp, d, pt), {}).get("adaptive_mmpp")
            if a is not None and b is not None:
                shifts.append(b - a)
        if shifts:
            s = np.asarray(shifts)
            print(
                f"adaptive-MMPP ceiling shift under {d}: mean {s.mean():+.4f}, "
                f"range [{s.min():+.4f}, {s.max():+.4f}], "
                f"{int((s < 0).sum())} of {s.size} negative"
            )


def report_noise(rows: list[dict]) -> None:
    """Rate noise: matched-headroom speedup and ceiling against every knob."""
    section("Rate noise")
    noise_exps = ["noise", "noise_setpoint", "noise_tau", "noise_kind"]
    per = defaultdict(list)
    for r in rows:
        if r["experiment"] in noise_exps and r["detector"] == "ideal":
            per[(r["experiment"], r["point"])].append(r)

    print(
        f"{'experiment':<15} {'point':<12} {'sigma':>6} {'tau_ms':>7} "
        f"{'kind':>10} {'speedup':>8} {'+-CI':>6} {'Qexcess':>8}"
    )
    q_all = []
    for key in sorted(per, key=lambda k: (k[0], _f(per[k][0], "point_index"))):
        rs = per[key]
        band = [r for r in rs if in_band(r, *BAND)]
        sp = med(_f(r, "speedup_mmpp") for r in band)
        w = med(
            _f(r, "speedup_mmpp_ci_high") - _f(r, "speedup_mmpp_ci_low")
            for r in band
        )
        q = _f(rs[0], "q_excess_finest_bin")
        if np.isfinite(q):
            q_all.append(q)
        print(
            f"{key[0]:<15} {key[1]:<12} {_f(rs[0], 'noise_sigma'):6.2f} "
            f"{_f(rs[0], 'noise_tau_c_ms'):7.3f} {rs[0]['noise_kind']:>10} "
            f"{sp:8.3f} {w:6.2f} {q:8.3f}"
        )
    if q_all:
        print(
            f"\nMandel Q excess across every noise point: "
            f"[{min(q_all):+.3f}, {max(q_all):+.3f}]"
        )


def report_snr_sweep(rows: list[dict], ceilings: dict) -> None:
    section("SNR sweep at fixed sparsity")
    per = defaultdict(list)
    for r in rows:
        if r["experiment"] == "snr" and r["detector"] == "ideal":
            per[r["point"]].append(r)
    print(
        f"{'point':<10} {'SNR':>6} {'contrast':>9} {'ph/dwell':>9} "
        f"{'thr ceil':>9} {'MMPP ceil':>10} {'speedup':>8} {'+-CI':>6}"
    )
    for pt in sorted(per, key=lambda p: _f(per[p][0], "snr_per_bright_dwell")):
        rs = per[pt]
        band = [r for r in rs if in_band(r, *BAND)]
        c = ceilings.get(("snr", "ideal", pt), {})
        print(
            f"{pt:<10} {_f(rs[0], 'snr_per_bright_dwell'):6.2f} "
            f"{_f(rs[0], 'contrast'):9.3f} "
            f"{_f(rs[0], 'photons_per_bright_dwell'):9.2f} "
            f"{c.get('threshold', float('nan')):9.4f} "
            f"{c.get('adaptive_mmpp', float('nan')):10.4f} "
            f"{med(_f(r, 'speedup_mmpp') for r in band):8.3f} "
            f"{med(_f(r, 'speedup_mmpp_ci_high') - _f(r, 'speedup_mmpp_ci_low') for r in band):6.2f}"
        )


def report_ci_widths(rows: list[dict]) -> None:
    section("Bootstrap interval widths")
    pool = [r for r in rows if r["_ideal"] and not r["_noisy"]]
    for label, sel in (
        ("all rows", pool),
        (f"matched headroom [{BAND[0]}, {BAND[1]})", [r for r in pool if in_band(r, *BAND)]),
        ("near the ceiling [0.00, 0.01)", [r for r in pool if in_band(r, 0.0, 0.01)]),
    ):
        w = [
            _f(r, "speedup_mmpp_ci_high") - _f(r, "speedup_mmpp_ci_low")
            for r in sel
        ]
        print(f"{label:<38} median width {med(w):.3f}  (n = {len(w)})")

    outside = sum(
        1
        for r in rows
        if np.isfinite(_f(r, "speedup_mmpp"))
        and not (
            _f(r, "speedup_mmpp_ci_low")
            <= _f(r, "speedup_mmpp")
            <= _f(r, "speedup_mmpp_ci_high")
        )
    )
    total = sum(1 for r in rows if np.isfinite(_f(r, "speedup_mmpp")))
    print(
        f"\npoint estimate outside its own interval: {outside} of {total} rows "
        f"({100 * outside / max(total, 1):.2f}%)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", help="results directory")
    args = ap.parse_args()

    root = Path(args.results)
    rows, ceilings = load(root)
    if not rows:
        raise SystemExit(f"no speedup_table.csv found under {root}/")
    rows = annotate(rows, ceilings)

    layers = {r.get("physics_layer", "") for r in rows}
    print(f"{len(rows)} rows from {root}/")
    print(
        f"experiments: {', '.join(sorted({r['experiment'] for r in rows}))}\n"
        f"detectors:   {', '.join(sorted({r['detector'] for r in rows}))}"
    )
    if layers - {""}:
        print(f"physics layer: {', '.join(sorted(layers - {''}))}")

    report_matched(rows, ceilings)
    report_headroom_bands(rows)
    report_factorization(rows)
    report_ceilings(rows, ceilings)
    report_collapse(rows, ceilings)
    report_snr_sweep(rows, ceilings)
    report_danjou(rows)
    report_overhead(rows)
    report_detector(rows, ceilings)
    report_noise(rows)
    report_ci_widths(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
