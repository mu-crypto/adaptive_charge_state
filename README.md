# adaptve_charge_state

Adaptive NV charge-state readout: does a sequential (SPRT) readout reach the
same initial-state charge fidelity as an optimized fixed-time count threshold
in less average run time?

Everything lives in one file, `adaptive_charge_state_master.py`, which merges
what used to be four scripts (`adaptive_charge_readout.py`,
`run_speedup_demo.py`, `power_sweep_speedup.py`,
`validate_adaptive_readout.py`).

## Methods compared

Always, on identical shots:

1. **fixed-time count threshold** — the standard method, the baseline
2. **adaptive count SPRT** — adaptive stopping, count statistic
3. **adaptive MMPP SPRT** — adaptive stopping, event-time statistic

Method 2 is the control that splits the gain: 1 → 2 is the value of adaptive
stopping alone, 2 → 3 is the value of the event-time statistic on top of it.
The `demo` experiment additionally runs a fixed-time MMPP baseline with a
calibrated cutoff, plus a paired exact McNemar test.

## Experiments

| key | what it sweeps |
|-----|----------------|
| `demo` | three operating points (high flux, moderate, sparse) |
| `power` | 594 nm laser power over the measured Shields range, 0.875–15 µW |
| `ratio` | Γ₋₀/Γ₀₋, ionization over recombination, at fixed Γ_tot |
| `contrast` | (λ₋ − λ₀)/(λ₋ + λ₀) at fixed bright rate |
| `efficiency` | detection efficiency η |

## Usage

```bash
python adaptive_charge_state_master.py list              # experiments, points, presets
python adaptive_charge_state_master.py validate          # 25 numerical checks
python adaptive_charge_state_master.py run power         # simulate + analyze
python adaptive_charge_state_master.py run ratio --point 0,2 --quick
python adaptive_charge_state_master.py plot contrast     # figures + summary
python adaptive_charge_state_master.py summary efficiency
python adaptive_charge_state_master.py robustness demo --point 1
```

A full five-point sweep takes well under two minutes; `--quick` runs a small
smoke version. Results are pickled under `out_master/<experiment>/<detector>/`
so figures and the speedup analysis can be regenerated without re-simulating.

## Detector model (dead time and afterpulsing)

Off by default, so omitting every detector flag reproduces ideal-detector
behaviour.

```bash
python adaptive_charge_state_master.py run power --detector                 # realistic preset
python adaptive_charge_state_master.py run power --detector-preset stress   # abusive
python adaptive_charge_state_master.py run power --detector-preset paralyzable
python adaptive_charge_state_master.py run power --detector \
    --dead-time-ns 500 --afterpulse-prob 0.03 --afterpulse-tau-ns 1000
python adaptive_charge_state_master.py run power --detector-preset stress \
    --no-filter-correction    # cost of pure model mismatch
```

Dead time is non-paralyzable by default (`--paralyzable` for extending).
Afterpulses are spawned only by recorded clicks, are themselves subject to dead
time, and can cascade — so the two knobs interact: an afterpulse whose delay is
shorter than the dead time is swallowed by its own parent's dead window. Set
`--afterpulse-tau-ns` above `--dead-time-ns` if you want afterpulsing to be
visible.

With the detector on, the photon stream is no longer a Markov-modulated
*Poisson* process, so the MMPP likelihood is approximate. By default the filter
is built from dead-time/afterpulse-corrected emission rates, which is what a
calibration measurement returns; `--no-filter-correction` measures what pure
model mismatch costs instead. Each detector setting writes to its own output
directory, so ideal and non-ideal runs never overwrite each other.

## Dependency

The physics primitives come from `nv_charge_readout_master_v1_4` when it is
importable. When it is not, the file falls back to a self-contained reference
implementation whose constants are pinned to the same 5.437 µW operating point
(Γ_tot = 9.42 kHz, p_bright = 0.118) but are a documented stand-in, not a
re-measurement. `list` and `validate` report which layer is active, and every
saved result records it.

Requires `numpy`, `scipy` and `matplotlib`.
