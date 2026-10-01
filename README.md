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
5. **learned stopping policy** — a regression approximation of the optimum
6. **the Ludkovski–Sezer optimum** — the Bayes-optimal rule itself, by exact dynamic programming

All six run in every sweep (`run`); `optimal` studies 5 and 6 in more depth
at the `demo` points, in Bayes risk as well as speedup.

`run` also scores the **HMM readout** of Spethmann, Stano & Loss (2025): the
exact posterior of the initial state from a fixed-length record, called at
P = ½. For this chain that is the MMPP filter's initial-state LLR with a
zero cutoff. Under the paper's own correlated-noise model (`paper_noise_*`)
it keeps a 10–15% infidelity lead over the threshold at every T_c, and that
lead closes only as the noise SNR falls; see `results/README.md`.

The design is a 2×2 of stopping rule against decision statistic, plus the
optimum the 2×2 is measured against:

| stopping rule | decision statistic | method |
|---|---|---|
| fixed time | photon count | 1 fixed-time threshold |
| fixed time | MMPP LLR | fixed-time MMPP (optional) |
| fixed count | count (one bit) | 2 adaptive count SPRT |
| fixed count | MMPP LLR | 3 fixed-count MMPP |
| LLR boundary | MMPP LLR | 4 adaptive MMPP SPRT |
| learned, on (LLR, information gap) | MMPP LLR | 5 learned policy |
| optimal, on the full posterior (LLR, u, v) | MMPP LLR | 6 exact DP optimum |

Methods 2 and 3 share their stopping rule *exactly* — identical per-shot stop
times, hence identical mean run time, which `validate` asserts — so 2 → 3
isolates the value of the event-time statistic with stopping held fixed, and
3 → 4 isolates the value of the LLR as a stopping rule with the statistic held
fixed. Without both controls a speedup could be attributed to either axis.

The `demo` experiment additionally runs a fixed-time MMPP baseline with a
calibrated cutoff, plus a paired exact McNemar test.

## Optimal stopping

Methods 1–4 are rules that were written down and then tuned; none is claimed
optimal. Method 6 is the one that is: the Ludkovski–Sezer (2012) Bayes
problem on the augmented chain (initial state, current state), solved by
backward induction on the full posterior — the initial-state LLR and the two
hypothesis columns u, v — plus time-to-go. Method 5 approximates the same
problem by Longstaff–Schwartz regression on (LLR, information gap) at 128
epochs; it is feasible, so its risk bounds the optimum from above, but it is
not the optimum.

```bash
python adaptive_charge_state_master.py optimal demo
python adaptive_charge_state_master.py optimal demo --no-dp   # skip the DP
```

The DP is checked in `validate` against the paper's own published example
(§6.2: R\* = 0.6813, continuation region [0.225, 0.705]) and by simulating its
policy on fresh shots, which must score its own R\*. It costs ~40 s per
(point, a/c) and skips itself where it would need more than 5000 time steps
(the long-horizon end of the `power` sweep).

It reports two currencies: Bayes risk against `a/c` (microseconds of readout
per avoided error), which is what the optimum minimises, and time to reach a
target fidelity with the same paired bootstrap as the other methods. Every
rule's gap to the optimum carries a paired 95% interval.

It also measures the **information-exhaustion exit** separately. A photon
translates both hypotheses equally in log-odds, so `d` is unchanged at every
click and shrinks only while waiting; once it closes the LLR is frozen. At
fixed boundary the exit therefore gives bit-identical decisions at
never-longer run times. The optimum stops earlier still: once
`d <= d* = 4c / (max(a,b)(λ_max + Δλ/4))` no remaining information can pay
for itself, which is proven and pinned in `validate`, and the learned policy
enforces it.

**In the sweeps** each point runs the learned policy at 24 values of a/c and
the DP at 14, on the same test shots and bootstrap resamples as methods 1–4.
Every a/c gives one Bayes-optimal (T, F) point; mixing two such rules shot by
shot reaches any point on the chord between them, so the optimum's
fidelity-vs-time frontier is the **concave hull** of its points (with the
trivial (0, ½)), interpolated linearly in T. The grid is anchored on the
immediate-stop threshold a/c ≈ 2/Δλ (L&S Remark 6.1), below which the
optimum guesses at t = 0. Switching moves the real threshold up by 17–53% (median 17%), so
the run bisects for it and adds five rules just above it, where the whole
low-fidelity end of the frontier is traced. Rules that stop within a few µs
are re-solved on a 4× finer step over a truncated horizon, so the decision
grid does not handicap them against the grid-free SPRT. The sweep DP uses
λ·dt ≤ 0.1 and a 0.7 y/z grid (+0.1% in R\* against the defaults, at a fifth
of the cost) and still skips points needing more than 5000 steps.
`--no-learned` / `--no-optimal` turn them off; `refine-optimum` adds the
near-threshold rules to runs saved before that step existed.

Under the detector and noise presets the DP is optimal for the *filter's*
model, not for the data, so there it is a strong model-based rule rather
than a ceiling.

See `results/README.md` for how far each rule sits from the optimum.

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

It also scores every rule under **both action sets** — two, and three with
`--cost-discard` — on identical shots, and plots them against each other.
The third action buys most for the rules that have no other way to handle an
ambiguous shot (+31% for the fixed-time threshold) and least for those that
already wait for confidence (+13% for the learned policy), which compresses
the field: with post-selection the best-to-worst spread falls from 40% to
11%, and the simplest event-time rules come within 1–2% of the best.

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
| `paper_noise_tau` | the noise model of Spethmann et al. (2025) — additive, Gaussian spectrum, one trace per state — sweeping T_c at SNR 3 |
| `paper_noise_snr` | the same noise sweeping its SNR = Δλ/σ at T_c = 10 µs |

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
python adaptive_charge_state_master.py validate          # 99 numerical checks
python adaptive_charge_state_master.py run power         # simulate + analyze
python adaptive_charge_state_master.py run ratio --point 0,2 --quick
python adaptive_charge_state_master.py run power --no-optimal   # methods 1-5 only, fast
python adaptive_charge_state_master.py refine-optimum demo      # upgrade older saved runs
python adaptive_charge_state_master.py add-hmm noise_tau        # add the HMM readout to saved runs
python adaptive_charge_state_master.py plot contrast     # figures + summary
python adaptive_charge_state_master.py summary efficiency
python adaptive_charge_state_master.py robustness demo --point 1
python adaptive_charge_state_master.py optimal demo      # every rule vs the exact optimum
python adaptive_charge_state_master.py discard demo      # add the abandon action
python adaptive_charge_state_master.py noiserisk demo    # all six under rate noise
python results/analysis.py                              # cross-experiment tables
```

With the exact optimum a five-point sweep at the committed sample size
(`--n-test 7200 --n-boot 400`) takes 30–90 min; `--no-optimal` brings it back
to a few minutes, and `--quick` runs a small smoke version. Run parallel
sweeps with `OPENBLAS_NUM_THREADS=1`: the DP's matrix products are small,
and several processes each spawning a BLAS thread per core ran 10× slower
than single-threaded ones. Results are pickled under `out_master/<experiment>/<detector>/`
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
active, every saved result records it, and `validate` passes 99/99 on both.

Requires `numpy`, `scipy` and `matplotlib`.
