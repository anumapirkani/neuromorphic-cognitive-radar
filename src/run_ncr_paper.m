%RUN_NCR_PAPER  Reproduce every result, figure and table of the paper.
%
%   Towards Neuromorphic Cognitive Radar: A Spiking Perception-Action Loop
%   for Real-Time Performance Optimisation (NICE 2027).
%
%   Steps
%     1  self-tests of the radar model, processing and utilities
%     2  experiments E1-E4 (see NCR_EXPERIMENTS); results saved as .mat
%     3  figures (PNG + PDF) and LaTeX tables in <out.dir>/figures, /tables
%
%   Runtime: about 0.2 s per CPI in MATLAB and 0.6 s in Octave on one core;
%   the full study is about 400 episodes of 64 CPIs. Set QUICK = true for a
%   smoke test, or cfg.exp.parallel = true to use parfor with the Parallel
%   Computing Toolbox. Saved results can be re-plotted with REPLOT_ONLY.
%
%   Requirements: MATLAB R2019b+ (no toolboxes) or GNU Octave 7+.

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);

QUICK       = false;     % true: 2 runs per configuration (minutes instead of hours)
REPLOT_ONLY = false;     % true: skip simulation, load results from cfg.out.dir
EXPERIMENTS = {'main', 'ablation', 'intf', 'latency'};

cfg = ncr_config();
if QUICK
    cfg.exp.mainNmc = 2;  cfg.exp.ablNmc = 2;  cfg.exp.sweepNmc = 2;
    cfg.out.dir = 'results_quick';
end

%% 1 - self-tests
T = ncr_selftest(cfg);
if ~all([T.pass])
    warning('run_ncr_paper:selftest', 'Some self-tests failed; check the table above.');
end

%% 2 - experiments
if REPLOT_ONLY
    X = ncr_load_results(cfg.out.dir);
else
    tic;
    X = ncr_experiments(cfg, EXPERIMENTS);
    fprintf('\nAll experiments finished in %.1f min.\n', toc / 60);
end

%% 3 - figures and tables
ncr_figures(cfg, X);
ncr_tables(cfg, X);
if all(isfield(X, {'E1', 'E2', 'E3', 'E4'}))
    ncr_export_summary(cfg.out.dir);      % numbers for the paper (make_macros.py)
end
fprintf('\nFigures: %s\nTables:  %s\n', fullfile(cfg.out.dir, 'figures'), fullfile(cfg.out.dir, 'tables'));
