#!/usr/bin/env python3
"""Merge the modular ncr_*.m sources into one MATLAB function file.

Usage: python3 build_single_file.py <src_dir> <out_file>
The output neuromorphic_cognitive_radar.m contains the entry function followed by
every module as a local function, in pipeline order.
"""
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])

ORDER = [
    ('CONFIGURATION', ['ncr_config', 'ncr_set', 'ncr_derive']),
    ('SCENARIO AND GROUND TRUTH', ['ncr_scene', 'ncr_scatterers', 'ncr_ranges', 'ncr_truth']),
    ('RF FRONT-END AND INTERFERENCE', ['ncr_link_budget', 'ncr_frontend', 'ncr_interference', 'ncr_mitigate']),
    ('SIGNAL PROCESSING AND DETECTION', ['ncr_process', 'ncr_cfar', 'ncr_boxsum', 'ncr_localmax3', 'ncr_detect',
                                         'ncr_cluster', 'ncr_coarse_map', 'ncr_sense']),
    ('TRACKING AND METRICS', ['ncr_tracker', 'ncr_hungarian', 'ncr_ospa', 'ncr_evaluate']),
    ('NEUROMORPHIC PERCEPTION AND LEARNING', ['ncr_encode', 'ncr_snn_init', 'ncr_snn_run', 'ncr_rstdp', 'ncr_reward']),
    ('ATTENTION AND RESOURCE ALLOCATION', ['ncr_attention']),
    ('EXPERIMENTS AND STATISTICS', ['ncr_episode', 'ncr_monte_carlo', 'ncr_summarise', 'ncr_stats', 'ncr_experiments']),
    ('OUTPUTS: FIGURES, TABLES, EXPORT', ['ncr_figures', 'ncr_tables', 'ncr_load_results', 'ncr_export_summary',
                                          'ncr_export_figure_data']),
    ('VERIFICATION AND UTILITIES', ['ncr_selftest', 'ncr_hann']),
]

MAIN = r'''function X = neuromorphic_cognitive_radar(mode, varargin)
%NEUROMORPHIC_COGNITIVE_RADAR  Spiking perception-action loop for a cognitive FMCW radar.
%
%   Complete simulation framework for the paper
%     "Towards Neuromorphic Cognitive Radar: A Spiking Perception-Action Loop
%      for Real-Time Performance Optimisation", NICE 2027.
%   A. Pirkani and B. Jelfs, Microwave Integrated Systems Laboratory,
%   University of Birmingham, UK.
%   https://github.com/anumpirkani123/neuromorphic-cognitive-radar
%
%   USAGE
%     neuromorphic_cognitive_radar              % self-tests, all experiments, figures, tables
%     neuromorphic_cognitive_radar('quick')     % smoke test (2 runs per configuration)
%     neuromorphic_cognitive_radar('replot')    % figures and tables from saved results
%     neuromorphic_cognitive_radar('selftest')  % verification of the radar model only
%     X = neuromorphic_cognitive_radar(..., 'OutDir', 'results', ...
%             'Experiments', {'main', 'intf'}, 'Parallel', true, ...
%             'Config', {'enc.type', 'latency', 'lrn.bits', 8});
%
%   Options (name-value, case-insensitive)
%     OutDir       output folder (default 'results', or 'results_quick' in quick mode)
%     Experiments  any of 'main' (E1), 'ablation' (E2), 'intf' (E3), 'latency' (E4)
%     Parallel     true to run Monte Carlo episodes with parfor (Parallel Computing Toolbox)
%     Config       cell array of parameter overrides passed to NCR_CONFIG
%
%   Single episodes can be run from the command line with the local functions
%   exposed through the 'episode' mode:
%     L = neuromorphic_cognitive_radar('episode', 'Policy', 'neuromorphic', 'Seed', 1);
%   Policies: 'uniform', 'rule-based', 'neuromorphic'.
%
%   SYSTEM
%   A 77 GHz FMCW radar with a phased transmit array (16 elements) and 16-channel
%   digital beamforming schedules its beam over sixteen 7.5-degree sectors (+/-60 deg).
%   Every coherent processing interval (CPI) a controller re-allocates the chirp
%   budget (dwell, i.e. transmit energy) and the sweep bandwidth across sectors.
%     Uniform       fixed dwell and bandwidth (non-cognitive baseline)
%     Rule-based    saliency with a resolution-aware Gaussian clutter-notch (MTI-type) prior
%     Neuromorphic  spike encoder + leaky integrate-and-fire network trained online,
%                   without labels, by reward-modulated STDP (three-factor rule)
%
%   WHAT IS MODELLED
%   Radar front-end: monostatic radar equation, coherent phased-array EIRP, noise
%     figure and losses, dechirped FMCW signal per sector with exact slow-time phase
%     history (Doppler and micro-Doppler), receive steering vectors, 12-bit ADC.
%   Scene: extended vehicle (6 scattering centres), pedestrian with limb micro-motion,
%     cyclist appearing mid-sequence; Swerling I fluctuation; parked cars and a pole;
%     vegetation with K-distributed amplitudes and internal motion.
%   Mutual interference: unsynchronised FMCW radar; bursts of Fs^2/|S_v - S_i| samples,
%     one-way Friis power, coherent across the array; optional time-domain excision.
%   Processing: Hann-windowed range FFT, zero-padded Doppler FFT, receive DBF (3 beams
%     per sector), 2-D CA-CFAR, 3-D peak picking with sub-bin interpolation, clustering.
%   Tracking and metrics: EKF (constant velocity, polar measurements), Hungarian GNN
%     association, M-of-N confirmation; P_D, false clusters, OSPA (c = 5 m, p = 2).
%   Neuromorphic loop: local winner-take-all rate code (or rate, latency, delta codes);
%     76,800 inputs, 4,800 LIF neurons in 4 weight-shared Doppler feature maps,
%     16 sector neurons; R-STDP with a spatially resolved reward from the radar's own
%     outputs (confirmed, unconfirmed, stationary detections, idle activity);
%     optional low-precision synapses with stochastic rounding.
%   Energy: indicative per-operation costs, Loihi synaptic operation 23.6 pJ and
%     update 120 pJ (Davies et al. 2018); FP32 MAC 4.6 pJ at 45 nm (Horowitz 2014).
%
%   EXPERIMENTS (paired seeds; mean +/- 95 % CI, Student t; paired t-tests)
%     E1 main      uniform vs rule-based vs neuromorphic, 20 runs
%     E2 ablation  no learning, global R-STDP, rate/latency/delta codes, 8/4-bit, excision
%     E3 intf      interferer EIRP off, 20, 23, 26, 29 dBm, plus uniform + excision
%     E4 latency   network time steps per CPI, 5 to 50
%
%   OUTPUTS (in OutDir)
%     E1_main.mat, E2_ablation.mat, E3_interference.mat, E4_latency.mat
%     figures/  fig_scene_attention, fig_dynamics, fig_kernel, fig_sweeps (paper)
%               supp_* (range-Doppler maps, spike rasters, resource maps, OSPA
%               components, activity and energy, ablation); vector PDF and PNG
%     tables/   LaTeX tables (booktabs) and summary.txt
%     summary.json     every number quoted in the paper (read by make_macros.py)
%     figure_data.mat  arrays behind the paper figures (read by make_figures.py)
%
%   REQUIREMENTS
%   MATLAB R2019b or later, no toolboxes (exportgraphics from R2020a is used when
%   available), or GNU Octave 7 or later. Runtime: about 0.2 s per CPI in MATLAB;
%   the full study is about 400 episodes of 64 CPIs. Random-number streams differ
%   between MATLAB and Octave, so single runs differ while Monte Carlo means agree
%   within their confidence intervals.
%
%   See also the local functions NCR_CONFIG (all parameters), NCR_EPISODE (one
%   closed-loop episode) and NCR_SELFTEST (verification).

if nargin < 1 || isempty(mode), mode = 'full'; end
opt = struct('outdir', '', 'experiments', {{'main', 'ablation', 'intf', 'latency'}}, ...
             'parallel', false, 'config', {{}}, 'policy', 'neuromorphic', 'seed', 1);
if mod(numel(varargin), 2) ~= 0, error('ncr:args', 'Options must be name-value pairs.'); end
for k = 1:2:numel(varargin)
    key = lower(varargin{k});
    if ~isfield(opt, key), error('ncr:args', 'Unknown option "%s".', varargin{k}); end
    opt.(key) = varargin{k + 1};
end
if ischar(opt.experiments), opt.experiments = {opt.experiments}; end

cfg = ncr_config(opt.config{:});
cfg.exp.parallel = logical(opt.parallel);
mode = lower(mode);
switch mode
    case 'quick'
        cfg.exp.mainNmc = 2;  cfg.exp.ablNmc = 2;  cfg.exp.sweepNmc = 2;
        if isempty(opt.outdir), opt.outdir = 'results_quick'; end
    case {'full', 'replot', 'selftest', 'episode'}
    otherwise
        error('ncr:mode', 'Unknown mode "%s" (full, quick, replot, selftest, episode).', mode);
end
if ~isempty(opt.outdir), cfg.out.dir = opt.outdir; end

switch mode
    case 'selftest'
        X = ncr_selftest(cfg);
        return
    case 'episode'
        X = ncr_episode(cfg, ncr_scene(cfg), opt.policy, opt.seed);
        return
    case 'replot'
        X = ncr_load_results(cfg.out.dir);
        if isempty(fieldnames(X))
            error('ncr:replot', 'No saved results (E1_main.mat, ...) found in "%s".', cfg.out.dir);
        end
    otherwise
        T = ncr_selftest(cfg);
        if ~all([T.pass])
            warning('ncr:selftest', 'Some self-tests failed; see the table above.');
        end
        t0 = tic;
        X = ncr_experiments(cfg, opt.experiments);
        fprintf('\nAll experiments finished in %.1f min.\n', toc(t0) / 60);
end

ncr_figures(cfg, X);
ncr_tables(cfg, X);
if isfield(X, 'E1'), ncr_export_figure_data(cfg, X); end
if all(isfield(X, {'E1', 'E2', 'E3', 'E4'})), ncr_export_summary(cfg.out.dir); end
fprintf('\nFigures: %s\nTables:  %s\n', fullfile(cfg.out.dir, 'figures'), fullfile(cfg.out.dir, 'tables'));
if nargout == 0, clear X; end
end
'''


def functions_in(text):
    return re.findall(r'^\s*function\s+(?:\[[^\]]*\]\s*=\s*|\w+\s*=\s*)?(\w+)', text, flags=re.M)


def main():
    parts = [MAIN]
    used = set(functions_in(MAIN))
    listed = [m for _, mods in ORDER for m in mods]
    missing = sorted(p.stem for p in SRC.glob('ncr_*.m') if p.stem not in listed)
    if missing:
        raise SystemExit(f'modules not placed in ORDER: {missing}')
    for title, mods in ORDER:
        bar = '%' * 2 + ' ' + '=' * 76
        parts.append(f'\n{bar}\n%%  {title}\n{bar}\n')
        for m in mods:
            text = (SRC / f'{m}.m').read_text().rstrip() + '\n'
            names = functions_in(text)
            dup = used.intersection(names)
            if dup:
                raise SystemExit(f'duplicate function names in {m}: {sorted(dup)}')
            used.update(names)
            parts.append('\n' + text)
    out = ''.join(parts)
    OUT.write_text(out)
    print(f'{OUT}: {out.count(chr(10))} lines, {len(used)} functions')


main()
