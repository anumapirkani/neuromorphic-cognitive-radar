# Neuromorphic Cognitive Radar Simulator

Closed-loop simulation framework for the paper

> A. Pirkani and B. Jelfs, "Towards Neuromorphic Cognitive Radar: A Spiking
> Perception-Action Loop for Real-Time Performance Optimisation," NICE 2027.

A 77 GHz FMCW radar with a phased transmit array and 16-channel digital beamforming
schedules its beam over sixteen 7.5° sectors (±60°). In every coherent processing
interval (CPI), a controller allocates the chirp budget (dwell, and hence transmit
energy) and the sweep bandwidth of every sector for the next CPI. Three controllers
are compared in closed loop:

| Controller | Perception | Learning |
|---|---|---|
| Uniform | none (fixed dwell and bandwidth) | none |
| Rule-based | motion saliency with a Gaussian clutter-notch (MTI-type) prior, attention field | none |
| Neuromorphic | local winner-take-all (WTA) spike encoder and weight-shared leaky integrate-and-fire (LIF) network | topographic reward-modulated STDP, online and without labels |

The repository reproduces every number, table and figure of the paper.

## Repository layout

```
neuromorphic_cognitive_radar.m   single-file version (all modules as local functions)
src/                             modular MATLAB sources (ncr_*.m) and run_ncr_paper.m
tools/build_single_file.py       merges src/ into neuromorphic_cognitive_radar.m
tools/make_macros.py             summary.json -> results_macros.tex (numbers in the paper)
tools/make_figures.py            figure_data.mat -> publication figures (vector PDF)
results/summary.json             results of the paper (every quoted number)
results/figure_data.mat          arrays behind the paper figures
examples/reward_sensitivity.m    sensitivity of learning to the reward values
```

## Quick start

Single file (MATLAB or GNU Octave):

```matlab
neuromorphic_cognitive_radar              % self-tests, all experiments, figures, tables
neuromorphic_cognitive_radar('quick')     % smoke test, 2 runs per configuration
neuromorphic_cognitive_radar('selftest')  % verification of the radar model only
neuromorphic_cognitive_radar('replot')    % figures and tables from saved results

% one closed-loop episode with parameter overrides
L = neuromorphic_cognitive_radar('episode', 'Policy', 'neuromorphic', 'Seed', 1, ...
        'Config', {'enc.type', 'latency', 'lrn.bits', 8});
```

Modular sources:

```matlab
cd src
run_ncr_paper                                 % same pipeline as the single file
cfg = ncr_config('intf.EIRP_dBm', 29);        % any parameter can be overridden
L   = ncr_episode(cfg, ncr_scene(cfg), 'rule-based', 1);
```

Policies are `'uniform'`, `'rule-based'` and `'neuromorphic'`. All parameters, with
comments, are in `src/ncr_config.m`.

Paper figures and macros from the saved results (Python 3 with NumPy, SciPy and
Matplotlib):

```bash
python3 tools/make_figures.py results/figure_data.mat figures
python3 tools/make_macros.py  results/summary.json  results_macros.tex
```

After editing `src/`, regenerate the single file with
`python3 tools/build_single_file.py src neuromorphic_cognitive_radar.m`.

## Requirements and runtime

MATLAB R2019b or later without toolboxes (the Parallel Computing Toolbox is optional:
`'Parallel', true`), or GNU Octave 7 or later. The full study comprises about 400
episodes of 64 CPIs. Random-number streams differ between MATLAB and Octave, so single
runs differ, while Monte Carlo means agree within their confidence intervals.

## What is modelled

**Radar front-end.** Monostatic radar equation with coherent phased-array EIRP
(`Ntx^2`), noise figure and losses; dechirped FMCW signal per sector with the exact
slow-time phase history of every scattering centre (Doppler and micro-Doppler),
receive steering vectors, transmit beam pattern and 12-bit IQ ADC.

**Scenario.** Extended vehicle (six scattering centres), pedestrian with arm and leg
micro-motion, cyclist with pedalling legs that appears halfway through the sequence;
Swerling I fluctuation; two parked cars, a pole and vegetation with K-distributed,
slowly decorrelating amplitudes.

**Mutual interference.** Unsynchronised FMCW interferer producing bursts of
`Fs^2/|S_v - S_I|` samples after dechirping, one-way propagation, coherent across the
receive array; optional time-domain excision.

**Processing and tracking.** Hann-windowed range and Doppler FFTs, three receive beams
per sector, 2-D CA-CFAR, peak picking with sub-bin interpolation, clustering, EKF
tracking with Hungarian (global nearest neighbour) association and M-of-N confirmation.
Metrics: detection probability, false moving clusters, OSPA distance (cut-off 5 m,
order 2) with localisation and cardinality components.

**Neuromorphic loop.** Perception grid of 25 × 48 range-angle cells (2 m × 2.5°) with
64 Doppler channels; local WTA rate code (alternatives: plain rate, latency and delta
codes); 76,800 input neurons, 4,800 hidden LIF neurons in four weight-shared Doppler
kernels (256 plastic weights) and 16 sector neurons. Pair-based STDP eligibility is
converted into weight changes by a per-cell reward derived from the radar's own
detections and tracks (alternative: global reward-prediction error). Optional
low-precision weights with stochastic rounding.

**Indicative energy.** 23.6 pJ per synaptic operation and 120 pJ per weight update
(Loihi, Davies et al., IEEE Micro 2018); 4.6 pJ per 32-bit floating-point
multiply-accumulate at 45 nm (Horowitz, ISSCC 2014).

## Experiments

| ID | Content | Runs |
|---|---|---|
| E1 | Uniform vs rule-based vs neuromorphic | 20 |
| E2 | Ablation: no learning, global R-STDP, rate / latency / delta codes, 8- and 4-bit weights, interference excision | 6 per variant |
| E3 | Interferer EIRP sweep (off, 20 to 29 dBm), including uniform allocation with excision | 4 per point |
| E4 | Network time steps per CPI (5 to 50) | 4 per point |

All configurations use paired random seeds. Results are reported as means with 95%
confidence intervals (Student t) and paired t-test p-values.

## Outputs

Written to `results/` (or the folder given by `'OutDir'`): `E1_main.mat` to
`E4_latency.mat`; `figures/` with the paper figures (`fig_scene_attention`,
`fig_dynamics`, `fig_kernel`, `fig_sweeps`) and supplementary figures (`supp_*`);
`tables/` with LaTeX tables and `summary.txt`; `summary.json` and `figure_data.mat`.

## Verification

`'selftest'` checks point-target placement in range, Doppler and angle, processed SNR
against the link budget, the CA-CFAR false-alarm rate, the interference burst length,
the Hungarian assignment against brute force, OSPA properties, the resource budgets
and tracker convergence.

## Citation

If you use this code, please cite the paper (see also `CITATION.cff`):

```bibtex
@inproceedings{pirkani2027neuromorphic,
  author    = {A. Pirkani and B. Jelfs},
  title     = {Towards Neuromorphic Cognitive Radar: A Spiking Perception-Action
               Loop for Real-Time Performance Optimisation},
  booktitle = {Neuro-Inspired Computational Elements Conference (NICE)},
  year      = {2027}
}
```

## Licence

MIT licence, see `LICENSE`.

Microwave Integrated Systems Laboratory, University of Birmingham, UK.
