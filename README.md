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
2. **adaptive count SPRT** — stop at the n-th photon, decide on whether you got there
3. **fixed-count MMPP** — same stopping rule as 2, decide with the calibrated MMPP LLR
4. **adaptive MMPP SPRT** — stop when the LLR crosses a calibrated boundary
5. **learned optimal stopping** — the Bayes-optimal rule, via `optimal`

The design is a 2×2 of stopping rule against decision statistic, plus the
optimum the 2×2 is measured against:

| stopping rule | decision statistic | method |
|---|---|---|
| fixed time | photon count | 1 fixed-time threshold |
| fixed time | MMPP LLR | fixed-time MMPP (optional) |
| fixed count | count (one bit) | 2 adaptive count SPRT |
| fixed count | MMPP LLR | 3 fixed-count MMPP |
| LLR boundary | MMPP LLR | 4 adaptive MMPP SPRT |
| learned, on (LLR, information gap) | MMPP LLR | 5 optimal stopping |

Methods 2 and 3 share their stopping rule *exactly* — identical per-shot stop
times, hence identical mean run time, which `validate` asserts — so 2 → 3
isolates the value of the event-time statistic with stopping held fixed, and
3 → 4 isolates the value of the LLR as a stopping rule with the statistic held
fixed. Without both controls a speedup could be attributed to either axis.

The `demo` experiment additionally runs a fixed-time MMPP baseline with a
calibrated cutoff, plus a paired exact McNemar test.

## Optimal stopping

Methods 1–4 are rules that were written down and then tuned; none is claimed
optimal. Method 5 computes the one that is — the Ludkovski–Sezer Bayes
problem on the augmented chain, solved by Longstaff–Schwartz regression Monte
Carlo over the initial-state LLR and the information gap
`d = logit(u) − logit(v)`.

```bash
python adaptive_charge_state_master.py optimal demo
python adaptive_charge_state_master.py optimal power --point 2 --n-epochs 256
```

It reports two currencies: Bayes risk against `a/c` (microseconds of readout
per avoided error), which is what the policy actually optimises, and time to
reach a target fidelity with the same paired bootstrap as the other methods.

It also measures the **information-exhaustion exit** separately. A photon
translates both hypotheses equally in log-odds, so `d` is unchanged at every
click and shrinks only while waiting; once it closes the LLR is frozen. At
fixed boundary the exit therefore gives bit-identical decisions at
never-longer run times — one comparison per epoch to implement, up to 130 µs
a shot saved. `validate` asserts all three facts.

Answer, at the three demo points: the tuned SPRT is within a few percent of
the optimum over most of the frontier and loses only in the top ~3% of the
fidelity range. See `results/README.md`.

## A third action: abandoning the shot

A finite `cost_discard` adds a third terminal action, so the decision becomes
two thresholds with an inconclusive band between them instead of one
threshold. The band edges follow from the costs — this is the post-selection
experiments already do by hand, priced rather than tuned.

```bash
python adaptive_charge_state_master.py discard demo
python adaptive_charge_state_master.py discard demo --a-over-c 2000
```

`cost_discard = inf` is the default and reproduces the two-action problem
bit-identically. Discard is only ever the cheapest action when
`w < ab/(a+b)`, which at a = b = 1 is 0.5, not 1; above that the band is
empty. `Economics.validate()` rejects degenerate costs on both sides.

Once shots can be thrown away, balanced fidelity is no longer a figure of
merit — abstain on everything and it goes to 1. So `three_action_metrics`
reports the Bayes risk as primary, accuracy only alongside the discard rate
that bought it, and retained shots per millisecond as the throughput number.

## Correlated rate noise in one currency

`noiserisk` compares all six rules in Bayes risk while sweeping the noise
*correlation time*, which the `noise_*` sweeps cannot do — they report the
matched-fidelity speedup, a two-method ratio in which both sides move
together.

```bash
python adaptive_charge_state_master.py noiserisk demo
python adaptive_charge_state_master.py noiserisk demo --sigma 0.2 --noise-kind-only telegraph
```

Every rule is re-tuned on calibration at each noise level, and the learned
policy is reported twice — trained clean and retrained on matched noise — so
damage from the noise is separated from damage from training on the wrong
distribution. Result: the fixed-time threshold is immune, everything that
reads arrival times pays, the damage peaks at Γ_tot·τ_c ≈ 1, and the ranking
never changes.

## Experiments

| key | what it sweeps |
|-----|----------------|
| `demo` | three operating points (high flux, moderate, sparse) |
| `power` | 594 nm laser power over the measured Shields range, 0.875–15 µW |
| `ratio` | Γ₋₀/Γ₀₋, ionization over recombination, at fixed Γ_tot |
| `contrast` | (λ₋ − λ₀)/(λ₋ + λ₀) at fixed bright rate |
| `efficiency` | detection efficiency η, at fixed contrast |
| `snr` | Δλ/√(λ̄ Γ₋₀), at fixed photons per bright dwell |
| `noise` | rate-noise amplitude σ (Gaussian, τ_c = 100 µs) |
| `noise_setpoint` | the same, without mean renormalisation |
| `noise_tau` | rate-noise correlation time τ_c |
| `noise_kind` | Gaussian against telegraph noise at matched variance |

`efficiency` and `snr` both move the readout's information content, but along
different axes: η scales SNR² and sparsity together, while the SNR sweep holds
sparsity fixed and moves the dark rate alone. The `snr` sweep is what settles
which of the two actually drives the speedup — see `results/README.md`.

The four `noise_*` experiments vary the rates *within* a shot, which no other
knob here can express: a rate error that is constant for a shot only rescales
the LLR and is absorbed by the calibrated boundary.

## Usage

```bash
python adaptive_charge_state_master.py list              # experiments, points, presets
python adaptive_charge_state_master.py validate          # 86 numerical checks
python adaptive_charge_state_master.py run power         # simulate + analyze
python adaptive_charge_state_master.py run ratio --point 0,2 --quick
python adaptive_charge_state_master.py plot contrast     # figures + summary
python adaptive_charge_state_master.py summary efficiency
python adaptive_charge_state_master.py robustness demo --point 1
python adaptive_charge_state_master.py optimal demo      # the Bayes-optimal rule
python adaptive_charge_state_master.py discard demo      # add the abandon action
python adaptive_charge_state_master.py noiserisk demo    # all six under rate noise
python results/analysis.py                              # cross-experiment tables
```

A full five-point sweep takes a couple of minutes; `--quick` runs a small
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

## Detection efficiency

`--efficiency-model` chooses how η acts on the count rates, and it is not a
cosmetic choice:

- `thin_all_counts` (default) multiplies both λ by η. Contrast is preserved
  exactly, so the efficiency sweep moves one thing, photons per bright dwell.
- `signal_only` leaves the state-independent background floor (0.268 kHz of
  dark counts and stray fluorescence) in place and thins only the NV
  fluorescence above it. That is what a real collection-efficiency loss does,
  but contrast then degrades along with η — at η = 0.02 it falls 0.925 → 0.833
  — so the sweep no longer isolates either variable.

Non-default runs get their own output directory, so the two cannot overwrite
each other.

## Physics layer

The rate laws, the MMPP parameter object and the fixed-time baseline come from
`nv_charge_readout_master_v1_4`, which is in this repo. If it is ever absent
the file falls back to a self-contained reference implementation pinned to the
same 5.437 µW operating point (Γ_tot = 9.42 kHz, p_bright = 0.118) — a
documented stand-in, not a re-measurement, and its emission rates and power
dependence differ substantially. `list` and `validate` report which layer is
active, every saved result records it, and `validate` passes 86/86 on both.

Requires `numpy`, `scipy` and `matplotlib`.
