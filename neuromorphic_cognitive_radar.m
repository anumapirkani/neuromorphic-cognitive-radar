function X = neuromorphic_cognitive_radar(mode, varargin)
%NEUROMORPHIC_COGNITIVE_RADAR  Spiking perception-action loop for a cognitive FMCW radar.
%
%   Complete simulation framework for the paper
%     "Towards Neuromorphic Cognitive Radar: A Spiking Perception-Action Loop
%      for Real-Time Performance Optimisation", NICE 2027.
%   A. Pirkani and B. Jelfs, Microwave Integrated Systems Laboratory,
%   University of Birmingham, UK.
%   https://github.com/anumapirkani/neuromorphic-cognitive-radar
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

%% ============================================================================
%%  CONFIGURATION
%% ============================================================================

function cfg = ncr_config(varargin)
%NCR_CONFIG  Parameters of the neuromorphic cognitive radar study.
%
%   cfg = NCR_CONFIG() returns the default configuration used in the paper.
%   cfg = NCR_CONFIG('group.field', value, ...) overrides any parameter, e.g.
%         cfg = ncr_config('enc.type', 'latency', 'lrn.bits', 8);
%
%   All quantities are SI unless the field name states otherwise (_dB, _dBm,
%   _dBi, _dBsm, _deg). Derived quantities are recomputed by NCR_DERIVE, so
%   overrides are always self-consistent.
%
%   Groups
%     sim   simulation length, frame rate, random seed
%     rf    RF front-end and link budget (77 GHz MIMO/phased radar)
%     wf    FMCW waveform (sampling, ramp, bandwidth budget)
%     ar    transmit/receive arrays and angular sectors
%     bud   per-CPI chirp (dwell) budget
%     proc  range/Doppler/angle processing
%     det   CFAR, peak picking, interpolation and clustering
%     trk   EKF multi-target tracker and OSPA metric
%     grid  coarse perception grid fed to the spiking network
%     enc   spike encoder
%     snn   leaky integrate-and-fire network
%     lrn   reward-modulated STDP
%     rew   cognitive reward
%     ctl   attention and resource controller
%     intf  FMCW-to-FMCW mutual interference
%     en    energy model (indicative, per-operation figures)
%     exp   experiment sizes (Monte Carlo runs, sweeps)
%     out   output folder and figure settings

%% ---------------------------------------------------------------- simulation
cfg.sim.Nscans   = 64;          % CPIs per episode
cfg.sim.Tf       = 50e-3;       % frame (CPI repetition) interval, s
cfg.sim.seed     = 1;           % base random seed (run r uses seed + r)
cfg.sim.verbose  = true;

%% ---------------------------------------------------------------- constants
cfg.const.c  = 299792458;       % m/s
cfg.const.kB = 1.380649e-23;    % J/K
cfg.const.T0 = 290;             % K

%% ---------------------------------------------------------------- RF / link budget
cfg.rf.fc       = 77e9;         % carrier, Hz
cfg.rf.Pt_dBm   = 4;            % transmit power per Tx channel (EIRP 33.1 dBm with 16 elements)
cfg.rf.Ntx      = 16;           % phased transmit elements (coherent EIRP gain Ntx^2)
cfg.rf.Nrx      = 16;           % receive channels (digital beamforming)
cfg.rf.Gt_dBi   = 5;            % Tx element gain
cfg.rf.Gr_dBi   = 5;            % Rx element gain
cfg.rf.NF_dB    = 14;           % receiver noise figure
cfg.rf.Lsys_dB  = 8;            % radome, feed, implementation and fluctuation losses
cfg.rf.adcBits  = 12;           % ADC resolution (Inf disables quantisation)
cfg.rf.adcFS    = 300;          % ADC full scale in units of the noise standard deviation

%% ---------------------------------------------------------------- waveform
cfg.wf.Fs   = 10e6;             % complex (IQ) sampling rate, Hz
cfg.wf.Nr   = 256;              % fast-time samples per chirp (Rmax = Nr c / 2B)
cfg.wf.Tc   = 40e-6;            % chirp repetition interval, s
cfg.wf.Bmin = 150e6;            % minimum sweep bandwidth per sector, Hz
cfg.wf.Bmax = 600e6;            % maximum sweep bandwidth per sector, Hz
cfg.wf.B0   = 300e6;            % mean-bandwidth (spectrum occupancy) budget, Hz

%% ---------------------------------------------------------------- arrays and sectors
cfg.ar.M        = 16;           % angular sectors (beam positions)
cfg.ar.secEdges = -60:7.5:60;   % sector edges, deg
cfg.ar.Na       = 3;            % receive beams per sector
cfg.ar.txTaper  = 'uniform';    % 'uniform' | 'hann'
cfg.ar.uMargin  = 0.2;          % scatterers within this sine-space margin of a sector are simulated

%% ---------------------------------------------------------------- dwell budget
cfg.bud.Nc   = 512;             % chirps per CPI (all sectors): 32 per sector when uniform
cfg.bud.nMin = 16;              % exploration floor per sector
cfg.bud.nMax = 128;             % per-sector cap (Doppler FFT size)

%% ---------------------------------------------------------------- processing
cfg.proc.Nd    = 128;           % Doppler FFT size (zero-padded)
cfg.proc.Rproc = 50;            % processed range interval, m

%% ---------------------------------------------------------------- detection
cfg.det.cfar    = 'CA';         % 'CA' (2-D cell averaging) | 'OS' (ordered statistic, range)
cfg.det.Pfa     = 1e-6;         % design false-alarm probability
cfg.det.Gr      = 2;            % range guard cells
cfg.det.Tr      = 8;            % range training cells (one side)
cfg.det.osK     = 0.75;         % OS-CFAR order (fraction of training cells)
cfg.det.vStatic = 1.0;          % |v_r| below this is stationary, m/s
cfg.det.clD     = 5.0;          % clustering gates: Cartesian distance, m (> vehicle length)
cfg.det.clV     = 2.5;          %                   radial velocity, m/s (covers micro-Doppler)

%% ---------------------------------------------------------------- tracker
cfg.trk.q       = 1.5;          % white-acceleration process noise, m/s^2
cfg.trk.sigR    = 0.3;          % measurement noise: range, m
cfg.trk.sigTh   = 2.0;          %                    angle, deg
cfg.trk.sigV    = 0.5;          %                    radial velocity, m/s
cfg.trk.sigV0   = 5.0;          % initial (unobserved tangential) velocity std, m/s
cfg.trk.gate    = 11.34;        % chi-square gate, 3 dof, 99 %
cfg.trk.M       = 3;            % M-of-N confirmation
cfg.trk.N       = 4;
cfg.trk.maxMissConf = 4;        % consecutive misses before a confirmed track is deleted
cfg.trk.maxMissTent = 2;        % ... before a tentative track is deleted
cfg.trk.ospaC   = 5;            % OSPA cut-off, m
cfg.trk.ospaP   = 2;            % OSPA order
cfg.trk.coverR  = 3;            % a target is covered if a confirmed track lies within this distance, m

%% ---------------------------------------------------------------- perception grid
cfg.grid.dRc = 2;               % coarse range cell, m
cfg.grid.dAc = 2.5;             % coarse angle cell, deg (3 per sector)

%% ---------------------------------------------------------------- spike encoder
cfg.enc.type    = 'peak-rate';  % 'peak-rate' | 'rate' | 'latency' | 'delta'
cfg.enc.zth_dB  = 10;           % encoding threshold above the noise floor
cfg.enc.rho0    = 30;           % saturation constant of the rate law
cfg.enc.pMax    = 0.4;          % maximum spike probability per time step
cfg.enc.T       = 25;           % time steps per CPI
cfg.enc.dtau    = 0.4e-3;       % time-step duration, s (25 x 0.4 ms = 10 ms of network time per CPI)
cfg.enc.delta_dB = 6;           % temporal-contrast threshold of the delta encoder

%% ---------------------------------------------------------------- spiking network
cfg.snn.K       = 4;            % feature maps (shared Doppler kernels)
cfg.snn.lamV    = 0.8;          % membrane leak factor per step
cfg.snn.Vth     = 1.0;          % hidden threshold
cfg.snn.refrac  = 0;            % refractory period, steps
cfg.snn.w0      = 1.0;          % mean initial weight (starts as a permissive energy detector)
cfg.snn.wSpread = 0.3;          % initial weights uniform in w0*(1 +/- wSpread)
cfg.snn.wMax    = 1.5;          % weight bound
cfg.snn.lamOut  = 0.9;          % output-neuron leak
cfg.snn.VthOut  = 2.0;          % output-neuron threshold

%% ---------------------------------------------------------------- learning
cfg.lrn.rule = 'spatial';       % 'spatial' (local neuromodulation) | 'global' (scalar RPE) | 'none'
cfg.lrn.Ap   = 1.0;             % LTP amplitude
cfg.lrn.Am   = 0.9;             % LTD amplitude
cfg.lrn.tauP = 2;               % pre-synaptic trace constant, steps
cfg.lrn.tauM = 2;               % post-synaptic trace constant, steps
cfg.lrn.eta  = 0.015;           % learning rate
cfg.lrn.lamE = 0;               % eligibility carried across CPIs (0 = within CPI only)
cfg.lrn.rBeta = 0.9;            % running-average constant of the reward baseline ('global')
cfg.lrn.bits = Inf;             % weight precision; finite values use stochastic rounding

%% ---------------------------------------------------------------- reward
cfg.rew.conf   = 1.0;           % moving detection on a confirmed track
cfg.rew.unc    = -0.1;          % moving detection without confirmed track
cfg.rew.static = -0.5;          % stationary detection (attention spent on clutter)
cfg.rew.idle   = -0.05;         % cell without detections (activity cost)

%% ---------------------------------------------------------------- controller
cfg.ctl.beta    = 0.8;          % leaky memory of the attention field
cfg.ctl.alpha   = 2.0;          % attention sharpening exponent
cfg.ctl.eps     = 0.02;         % attention floor
cfg.ctl.sigV    = 0.75;         % width of the analytical motion prior, m/s
cfg.ctl.sSatAlg = 0.2;          % saturation of the rule-based saliency
cfg.ctl.sSatNeu = 0.1;          % saturation of the hidden-layer rate
cfg.ctl.ySat    = 2;            % output spikes per CPI for full sector relevance

%% ---------------------------------------------------------------- interference
cfg.intf.enable   = true;
cfg.intf.theta    = -40;        % direction of the interfering radar, deg
cfg.intf.R        = 45;         % range of the interfering radar, m
cfg.intf.EIRP_dBm = 26;         % interferer EIRP (automotive limit 55 dBm peak)
cfg.intf.B        = 100e6;      % interferer sweep bandwidth, Hz (slow, long chirp)
cfg.intf.Tramp    = 50e-6;      % interferer ramp duration, s
cfg.intf.Tc       = 52e-6;      % interferer chirp period, s (unsynchronised)
cfg.intf.f0       = 100e6;      % interferer start frequency above fc, Hz
cfg.intf.Lrx_dB   = 4;          % receive-side losses for the one-way path
cfg.intf.on       = 20;         % first and last CPI with the interferer active
cfg.intf.off      = 44;
cfg.intf.mitigation = 'none';   % 'none' | 'zeroing' (time-domain excision)
cfg.intf.zeroK    = 3;          % excision threshold (x robust noise std)
cfg.intf.zeroGuard = 4;         % samples zeroed either side of a burst

%% ---------------------------------------------------------------- energy model
% Indicative per-operation energies: Loihi synaptic spike operation and STDP
% update [Davies et al., IEEE Micro 38(1) 2018]; 32-bit floating-point
% multiply (3.7 pJ) + add (0.9 pJ) at 45 nm [Horowitz, ISSCC 2014].
cfg.en.Esop_pJ = 23.6;
cfg.en.Eupd_pJ = 120;
cfg.en.Emac_pJ = 4.6;

%% ---------------------------------------------------------------- experiments
cfg.exp.mainNmc   = 20;                     % E1 main comparison
cfg.exp.ablNmc    = 6;                      % E2 ablation
cfg.exp.sweepNmc  = 4;                      % E3/E4 sweeps
cfg.exp.intfEIRP  = [-Inf 20 23 26 29];     % E3 interferer EIRP values, dBm (-Inf = off)
cfg.exp.Tsweep    = [5 10 25 50];           % E4 time steps per CPI
cfg.exp.late      = 16;                     % first CPI of the post-learning window
cfg.exp.parallel  = false;                  % use parfor (Parallel Computing Toolbox)

%% ---------------------------------------------------------------- output
cfg.out.dir     = 'results';
cfg.out.dpi     = 600;
cfg.out.font    = 'Times New Roman';

%% ---------------------------------------------------------------- overrides
for i = 1:2:numel(varargin)
    cfg = ncr_set(cfg, varargin{i}, varargin{i+1});
end
cfg = ncr_derive(cfg);
end

function cfg = ncr_set(cfg, path, value)
%NCR_SET  Set a (possibly nested) configuration field given as 'group.field'.
%   cfg = NCR_SET(cfg, 'lrn.bits', 8). Call NCR_DERIVE afterwards (NCR_CONFIG
%   does this automatically for its own overrides).
parts = regexp(path, '\.', 'split');
switch numel(parts)
    case 1
        cfg.(parts{1}) = value;
    case 2
        if ~isfield(cfg, parts{1}) || ~isfield(cfg.(parts{1}), parts{2})
            error('ncr_set:unknown', 'Unknown configuration field "%s".', path);
        end
        cfg.(parts{1}).(parts{2}) = value;
    otherwise
        error('ncr_set:depth', 'Only group.field paths are supported ("%s").', path);
end
end

function cfg = ncr_derive(cfg)
%NCR_DERIVE  Quantities derived from the primary configuration.
d = struct();
d.lambda  = cfg.const.c / cfg.rf.fc;
d.Tramp   = cfg.wf.Nr / cfg.wf.Fs;                       % sampled ramp duration
d.Tcpi    = cfg.bud.Nc * cfg.wf.Tc;                      % CPI duration
d.noiseW  = cfg.const.kB * cfg.const.T0 * 10^(cfg.rf.NF_dB/10) * cfg.wf.Fs;   % noise power per sample
d.nUni    = cfg.bud.Nc / cfg.ar.M;                       % uniform dwell per sector

% receive beams (deg) for every sector
d.beams = zeros(cfg.ar.M, cfg.ar.Na);
w = diff(cfg.ar.secEdges(1:2));
for j = 1:cfg.ar.M
    d.beams(j, :) = cfg.ar.secEdges(j) + w / cfg.ar.Na * ((1:cfg.ar.Na) - 0.5);
end
d.secCentre = (cfg.ar.secEdges(1:end-1) + cfg.ar.secEdges(2:end)) / 2;

% Doppler axes: fine (FFT) and coarse (stride 2, centred on 0 m/s)
d.dv     = d.lambda / (2 * cfg.wf.Tc * cfg.proc.Nd);
d.vFine  = (-cfg.proc.Nd/2 : cfg.proc.Nd/2 - 1) * d.dv;
d.vCoarse = d.vFine(1:2:end);
d.vmax   = d.lambda / (4 * cfg.wf.Tc);

% perception grid
d.NRc = round(cfg.proc.Rproc / cfg.grid.dRc);
d.NAc = round((cfg.ar.secEdges(end) - cfg.ar.secEdges(1)) / cfg.grid.dAc);
d.cellsPerSector = d.NAc / cfg.ar.M;
d.NDc = cfg.proc.Nd / 2;
d.Np  = d.NRc * d.NAc;
d.rCoarse  = cfg.grid.dRc * ((1:d.NRc) - 0.5);
d.thCoarse = cfg.ar.secEdges(1) + cfg.grid.dAc * ((1:d.NAc) - 0.5);
d.sectorOfCell = ceil((1:d.NAc) / d.cellsPerSector);

% window coherent gains (used by the link budget and self-tests)
d.cgR = coherent_gain(ncr_hann(cfg.wf.Nr));
d.cgX = coherent_gain(ncr_hann(cfg.rf.Nrx));

% sizes of the neuromorphic network
d.Nin  = d.NDc * d.Np;
d.Nhid = cfg.snn.K * d.Np;
d.Nout = cfg.ar.M;
d.denseMACs = d.NDc * d.Np * cfg.snn.K;                  % one frame-based pass
cfg.d = d;

if mod(d.NAc, cfg.ar.M) ~= 0
    error('ncr_derive:grid', 'Coarse angle cells must divide evenly into sectors.');
end
end

function g = coherent_gain(w)
g = sum(w)^2 / sum(w.^2);
end

%% ============================================================================
%%  SCENARIO AND GROUND TRUTH
%% ============================================================================

function scene = ncr_scene(cfg)
%NCR_SCENE  Urban scenario seen by a stationary 77 GHz radar at the origin.
%
%   Coordinates: x lateral (positive right), y boresight. Angles are measured
%   from boresight, positive towards +x. Moving targets are extended: each is
%   a set of scattering centres defined in the body frame (along = heading,
%   across = left-hand normal). Limbs and pedals oscillate along the heading
%   (micro-Doppler).
%
%   Targets
%     Vehicle     10 dBsm, 6 scattering centres, passes a parked car and the
%                 direction of an interfering radar (sector -45..-30 deg)
%     Pedestrian  -5.4 dBsm, torso + 2 legs + 2 arms (gait 0.9 Hz), weak
%     Cyclist     -1.3 dBsm, rider + frame + 2 legs (cadence 1.3 Hz),
%                 appears at CPI 30 (novelty)
%   Clutter
%     two parked cars (4 points each), a pole, and a vegetation strip with
%     K-distributed amplitudes and internal motion (33..58 deg).

% ---------------------------------------------------------------- moving targets
T = struct('name', {}, 'p0', {}, 'v', {}, 'on', {}, 'off', {}, 'parts', {});

veh.name = 'Vehicle';  veh.p0 = [-28 38];  veh.v = [3.5 -6.5];
veh.on = 1;  veh.off = cfg.sim.Nscans;
veh.parts = parts_table([ ...
%   along  across  rcs_dBsm swingAmp swingHz phase
     2.25   0.9    4        0        0       0
     2.25  -0.9    4        0        0       0
     1.40   0.9    0        0        0       0
     1.40  -0.9    0        0        0       0
    -2.25   0.9    2        0        0       0
    -2.25  -0.9    2        0        0       0]);
T(1) = veh;

ped.name = 'Pedestrian';  ped.p0 = [13 34];  ped.v = [-0.5 -1.4];
ped.on = 1;  ped.off = cfg.sim.Nscans;
ped.parts = parts_table([ ...
     0      0      -8       0       0       0       % torso
     0      0.15   -14      0.35    0.9     0       % legs (antiphase)
     0     -0.15   -14      0.35    0.9     pi
     0      0.25   -16      0.20    0.9     pi      % arms (opposite to legs)
     0     -0.25   -16      0.20    0.9     0]);
T(2) = ped;

cyc.name = 'Cyclist';  cyc.p0 = [-6 12];  cyc.v = [-1.2 4.6];
cyc.on = 30;  cyc.off = cfg.sim.Nscans;
cyc.parts = parts_table([ ...
     0      0      -4       0       0       0       % rider
     0.5    0      -6       0       0       0       % frame and wheels
     0.1    0.15   -13      0.15    1.3     0       % pedalling legs
     0.1   -0.15   -13      0.15    1.3     pi]);
T(3) = cyc;
scene.tgt = T;

% ---------------------------------------------------------------- stationary clutter
cars = [-20 16; -24 27];                       % parked cars, aligned with y
pts = [];
for c = 1:size(cars, 1)
    pts = [pts; cars(c, :) + [ 0.9  2.2]; cars(c, :) + [-0.9  2.2]; ...
                cars(c, :) + [ 0.9 -2.2]; cars(c, :) + [-0.9 -2.2]]; %#ok<AGROW>
end
pts = [pts; 4 42];                             % pole
scene.stat.p   = pts;
scene.stat.rcs = [4 * ones(8, 1); 5];          % dBsm (cars ~10 dBsm each)

% vegetation strip (fixed geometry; amplitudes evolve per CPI)
nVeg = 120;
s = rng;  rng(12345);                          % geometry independent of the run seed
th = (33 + 25 * rand(nVeg, 1)) * pi / 180;
R  = 8 + 40 * rand(nVeg, 1);
rng(s);
scene.veg.p     = [R .* sin(th), R .* cos(th)];
scene.veg.rcs   = -14 * ones(nVeg, 1);         % mean RCS per scatterer, dBsm
scene.veg.nu    = 1.5;                         % K-distribution texture shape
scene.veg.rho   = 0.9;                         % CPI-to-CPI texture correlation
scene.veg.sigv  = 0.1;                         % internal-motion velocity std, m/s

% ---------------------------------------------------------------- interferer
scene.intf = cfg.intf;
scene.intf.pos = cfg.intf.R * [sind(cfg.intf.theta), cosd(cfg.intf.theta)];
end

function P = parts_table(A)
P = struct('along', num2cell(A(:, 1)), 'across', num2cell(A(:, 2)), ...
           'rcs', num2cell(A(:, 3)), 'swAmp', num2cell(A(:, 4)), ...
           'swHz', num2cell(A(:, 5)), 'swPh', num2cell(A(:, 6)));
end

function [sc, state] = ncr_scatterers(cfg, scene, t, state)
%NCR_SCATTERERS  Scattering centres of the scene for CPI t.
%
%   [sc, state] = NCR_SCATTERERS(cfg, scene, t, state) returns a structure of
%   column arrays (one row per scattering centre) describing the scene at the
%   start time t0 = (t-1)*Tf of CPI t:
%       p, v      position and velocity (K x 2)
%       swDir     unit direction of micro-motion (K x 2)
%       swAmp, swHz, swPh   sinusoidal micro-motion (limbs, pedals)
%       rcs       instantaneous RCS (linear, includes fluctuation)
%       phi       random phase (fluctuating scatterers)
%       tgt       owning target index (0 = stationary clutter, -1 = vegetation)
%   state carries the vegetation texture between CPIs (empty on first call).
%
%   Moving-target parts follow Swerling I per scattering centre (independent
%   from CPI to CPI); parked cars and the pole are deterministic; vegetation
%   is compound-Gaussian (K-distributed) with an AR(1) texture and internal
%   motion.
t0 = (t - 1) * cfg.sim.Tf;
P = []; V = []; D = []; A = []; F = []; PH = []; RCS = []; PHI = []; TG = [];

% ---------------------------------------------------------------- moving targets
for k = 1:numel(scene.tgt)
    T = scene.tgt(k);
    if t < T.on || t > T.off, continue; end
    c  = T.p0 + T.v * (t - T.on) * cfg.sim.Tf;          % body centre
    u  = T.v / norm(T.v);                               % heading
    nrm = [-u(2) u(1)];                                 % left-hand normal
    for q = 1:numel(T.parts)
        pq = T.parts(q);
        P   = [P;  c + pq.along * u + pq.across * nrm];  %#ok<AGROW>
        V   = [V;  T.v];                                  %#ok<AGROW>
        D   = [D;  u];                                    %#ok<AGROW>
        A   = [A;  pq.swAmp];  F = [F; pq.swHz];  PH = [PH; pq.swPh]; %#ok<AGROW>
        fl  = -log(rand);                                 % Swerling I power fluctuation
        RCS = [RCS; 10^(pq.rcs / 10) * fl];               %#ok<AGROW>
        PHI = [PHI; 2 * pi * rand];                       %#ok<AGROW>
        TG  = [TG;  k];                                   %#ok<AGROW>
    end
end

% ---------------------------------------------------------------- stationary clutter
ns = size(scene.stat.p, 1);
P = [P; scene.stat.p];  V = [V; zeros(ns, 2)];  D = [D; zeros(ns, 2)];
A = [A; zeros(ns, 1)];  F = [F; zeros(ns, 1)];  PH = [PH; zeros(ns, 1)];
RCS = [RCS; 10.^(scene.stat.rcs / 10)];
PHI = [PHI; zeros(ns, 1)];                             % deterministic (Swerling 0)
TG  = [TG;  zeros(ns, 1)];

% ---------------------------------------------------------------- vegetation
nv = size(scene.veg.p, 1);
nu = scene.veg.nu;
g  = gamrnd_local(nu, 1 / nu, nv);                      % unit-mean gamma texture
if isempty(state) || ~isfield(state, 'vegTex')
    state.vegTex = g;
else
    state.vegTex = scene.veg.rho * state.vegTex + (1 - scene.veg.rho) * g;
end
P = [P; scene.veg.p];
V = [V; scene.veg.sigv * randn(nv, 1) .* scene.veg.p ./ sqrt(sum(scene.veg.p.^2, 2))];
D = [D; zeros(nv, 2)];  A = [A; zeros(nv, 1)];  F = [F; zeros(nv, 1)];  PH = [PH; zeros(nv, 1)];
RCS = [RCS; 10.^(scene.veg.rcs / 10) .* state.vegTex .* (-log(rand(nv, 1)))];   % texture x speckle
PHI = [PHI; 2 * pi * rand(nv, 1)];
TG  = [TG; -ones(nv, 1)];

sc = struct('p', P, 'v', V, 'swDir', D, 'swAmp', A, 'swHz', F, 'swPh', PH, ...
            'rcs', RCS, 'phi', PHI, 'tgt', TG, 't0', t0);
sc.theta = atan2(P(:, 1), P(:, 2)) * 180 / pi;
end

function g = gamrnd_local(a, b, n)
% Gamma(a, b) samples without the Statistics Toolbox (Marsaglia-Tsang, a >= 1;
% boosted for a < 1).
boost = ones(n, 1);
if a < 1
    boost = rand(n, 1).^(1 / a);
    a = a + 1;
end
d = a - 1/3;  c = 1 / sqrt(9 * d);
g = zeros(n, 1);
for i = 1:n
    while true
        x = randn;  v = (1 + c * x)^3;
        if v <= 0, continue; end
        u = rand;
        if log(u) < 0.5 * x^2 + d - d * v + d * log(v)
            g(i) = d * v;
            break
        end
    end
end
g = g .* boost * b;
end

function R = ncr_ranges(sc, idx, tm)
%NCR_RANGES  Ranges of selected scattering centres at the chirp times tm.
%
%   R = NCR_RANGES(sc, idx, tm) returns a numel(idx) x numel(tm) matrix of
%   ranges for the scatterers sc(idx) at absolute times tm (row vector),
%   including bulk motion and sinusoidal micro-motion. The resulting
%   slow-time phase history exp(j 4 pi R / lambda) carries Doppler,
%   micro-Doppler and range-rate exactly.
tm = tm(:)';
dt = tm - sc.t0;                                           % time since CPI start
sw = sc.swAmp(idx) .* sin(2 * pi * sc.swHz(idx) .* tm + sc.swPh(idx));   % K x n
x = sc.p(idx, 1) + sc.v(idx, 1) .* dt + sc.swDir(idx, 1) .* sw;
y = sc.p(idx, 2) + sc.v(idx, 2) .* dt + sc.swDir(idx, 2) .* sw;
R = sqrt(x.^2 + y.^2);
end

function tr = ncr_truth(cfg, scene, t)
%NCR_TRUTH  Ground truth of the moving targets at the centre of CPI t.
%   tr(k) has fields: name, active, pos (1x2), vel (1x2), R, theta (deg),
%   vr (radial velocity), sector, Rparts (ranges of its scattering centres).
tr = struct('name', {}, 'active', {}, 'pos', {}, 'vel', {}, 'R', {}, ...
            'theta', {}, 'vr', {}, 'sector', {}, 'Rparts', {});
for k = 1:numel(scene.tgt)
    T = scene.tgt(k);
    a = t >= T.on && t <= T.off;
    c = T.p0 + T.v * ((t - T.on) * cfg.sim.Tf + cfg.d.Tcpi / 2);
    u = T.v / norm(T.v);  n = [-u(2) u(1)];
    Rp = zeros(numel(T.parts), 1);
    for q = 1:numel(T.parts)
        pq = T.parts(q);
        Rp(q) = norm(c + pq.along * u + pq.across * n);
    end
    R = norm(c);  th = atan2(c(1), c(2)) * 180 / pi;
    tr(k).name = T.name;  tr(k).active = a;  tr(k).pos = c;  tr(k).vel = T.v;
    tr(k).R = R;  tr(k).theta = th;  tr(k).vr = dot(c, T.v) / R;
    tr(k).sector = min(cfg.ar.M, max(1, floor((th - cfg.ar.secEdges(1)) / diff(cfg.ar.secEdges(1:2))) + 1));
    tr(k).Rparts = Rp;
end
end

%% ============================================================================
%%  RF FRONT-END AND INTERFERENCE
%% ============================================================================

function [snr, info] = ncr_link_budget(cfg, R, rcsLin, gTx)
%NCR_LINK_BUDGET  Per-sample, per-channel SNR from the monostatic radar equation.
%
%   snr = NCR_LINK_BUDGET(cfg, R, rcsLin, gTx) returns the linear SNR of one
%   complex ADC sample on one receive channel for scatterers at range R (m)
%   with RCS rcsLin (m^2), illuminated with the normalised transmit-pattern
%   gain gTx (0..1, 1 = beam peak). All inputs broadcast.
%
%       Pr = Pt Ntx^2 Gt Gr lambda^2 sigma gTx / ((4 pi)^3 R^4 L)
%       N  = k T0 F Fs
%
%   [snr, info] also returns the individual terms (dB) for reporting.
if nargin < 4, gTx = 1; end
rf = cfg.rf;  lam = cfg.d.lambda;
Pt = 10^((rf.Pt_dBm - 30) / 10);
Gt = 10^(rf.Gt_dBi / 10);   Gr = 10^(rf.Gr_dBi / 10);
L  = 10^(rf.Lsys_dB / 10);
Pr = Pt * rf.Ntx^2 * Gt * Gr * lam^2 .* rcsLin .* gTx ./ ((4*pi)^3 .* R.^4 * L);
snr = Pr / cfg.d.noiseW;
if nargout > 1
    info.Pt_dBm      = rf.Pt_dBm;
    info.arrayGain_dB = 20 * log10(rf.Ntx);
    info.EIRP_dBm    = rf.Pt_dBm + 20*log10(rf.Ntx) + rf.Gt_dBi;
    info.lambda2_dB  = 10 * log10(lam^2);
    info.spreading_dB = -10 * log10((4*pi)^3);
    info.noise_dBm   = 10 * log10(cfg.d.noiseW) + 30;
    info.Pr_dBm      = 10 * log10(Pr) + 30;
    info.snr_dB      = 10 * log10(snr);
end
end

function [X, info] = ncr_frontend(cfg, sc, intf, j, nj, bj, tm, intfOn)
%NCR_FRONTEND  ADC data cube of one sector sub-frame.
%
%   X = NCR_FRONTEND(cfg, sc, intf, j, nj, bj, tm, intfOn) synthesises the
%   dechirped, sampled FMCW signal (Nr x Nrx x nj, noise variance 1 per
%   sample) received while the phased transmit beam points at sector j.
%
%   Signal model for scatterer s, fast-time sample n, channel k, chirp m:
%       x = a_s exp(j 2 pi f_b,s n / Fs) exp(j 4 pi R_s(t_m) / lambda)
%             exp(j pi k sin(theta_s)) + noise + interference
%   with beat frequency f_b = 2 S R / c (S = bj / Tramp) and the exact
%   slow-time range history R_s(t_m) (bulk motion and micro-motion). Range
%   migration inside a chirp is neglected (< 1 mm). The amplitude a_s
%   follows the link budget, the instantaneous RCS and the transmit array
%   factor towards theta_s. Thermal noise is complex white Gaussian; the
%   ADC quantises I and Q to cfg.rf.adcBits bits.
c = cfg.const.c;  lam = cfg.d.lambda;  Nr = cfg.wf.Nr;  Nrx = cfg.rf.Nrx;
u = sind(sc.theta);

% ---- scatterers that can reach this sector's beams
uLo = sind(cfg.ar.secEdges(j)) - cfg.ar.uMargin;
uHi = sind(cfg.ar.secEdges(j+1)) + cfg.ar.uMargin;
sel = find(u >= uLo & u <= uHi);

% ---- transmit array factor (normalised to 1 at the beam peak)
thC = cfg.d.secCentre(j);
switch lower(cfg.ar.txTaper)
    case 'hann', wt = ncr_hann(cfg.rf.Ntx);
    otherwise,   wt = ones(cfg.rf.Ntx, 1);
end
af = abs(wt.' * exp(1j * pi * (0:cfg.rf.Ntx-1)' * (u(sel)' - sind(thC)))).^2 / sum(wt)^2;

% ---- deterministic part (Khatri-Rao factorisation keeps this cheap)
Rm = ncr_ranges(sc, sel, tm);                  % K x nj range history
Rc = mean(Rm, 2);
a  = sqrt(ncr_link_budget(cfg, Rc, sc.rcs(sel), af(:))) .* exp(1j * sc.phi(sel));
K  = numel(sel);
Ar = exp(1j * 2 * pi * (0:Nr-1)' * (2 * bj * Rc' / (c * Nr)));      % Nr x K
Ad = exp(1j * 4 * pi * Rm.' / lam);                                 % nj x K
Ax = exp(1j * pi * (0:Nrx-1)' * u(sel)');                           % Nrx x K
KR = reshape(reshape(Ax, Nrx, 1, K) .* reshape(Ad, 1, nj, K), Nrx * nj, K);
X  = reshape(double(single(Ar) * single(a .* KR.')), Nr, Nrx, nj);   % single-precision synthesis

% ---- thermal noise
X = X + (randn(Nr, Nrx, nj) + 1j * randn(Nr, Nrx, nj)) / sqrt(2);

% ---- mutual interference
info.intfDuty = 0;
if intfOn
    [XI, ii] = ncr_interference(cfg, intf, nj, bj, tm);
    X = X + XI;
    info.intfDuty = ii.dutyFraction;
end

% ---- ADC quantisation (I and Q)
info.clipFraction = 0;
if isfinite(cfg.rf.adcBits)
    FS = cfg.rf.adcFS;  q = 2 * FS / 2^cfg.rf.adcBits;
    re = min(max(round(real(X) / q) * q, -FS), FS);
    im = min(max(round(imag(X) / q) * q, -FS), FS);
    info.clipFraction = mean(abs(real(X(:))) > FS | abs(imag(X(:))) > FS);
    X = re + 1j * im;
end
info.nScatterers = K;
end

function [XI, info] = ncr_interference(cfg, intf, nj, bj, tm)
%NCR_INTERFERENCE  FMCW-to-FMCW mutual interference at the victim ADC.
%
%   XI = NCR_INTERFERENCE(cfg, intf, nj, bj, tm) returns the interference
%   component (Nr x Nrx x nj, in units of the noise standard deviation) for a
%   victim sub-frame of nj chirps with sweep bandwidth bj starting at the
%   absolute times tm (1 x nj).
%
%   The interferer transmits its own sawtooth chirps (bandwidth intf.B, ramp
%   intf.Tramp, period intf.Tc, start offset intf.f0) that are not
%   synchronised with the victim. After dechirping, its contribution is a
%   chirp at the difference frequency f_v(t) - f_i(t); only the portion that
%   falls inside the complex IF band |f| < Fs/2 reaches the ADC, which
%   produces short, broadband bursts whose length scales with
%   Fs / |S_v - S_i|. The interferer is a point source at intf.theta, so
%   the burst is coherent across the receive array. Its power follows the
%   one-way Friis equation:
%       P_I = EIRP + G_r - 20 log10(4 pi R_I / lambda) - L_rx.
%
%   info returns INR per sample (dB) and the fraction of samples affected.
Fs = cfg.wf.Fs;  Nr = cfg.wf.Nr;  lam = cfg.d.lambda;
fspl = 20 * log10(4 * pi * intf.R / lam);
PI_dBm = intf.EIRP_dBm + cfg.rf.Gr_dBi - fspl - intf.Lrx_dB;
INR = 10^((PI_dBm - 30) / 10) / cfg.d.noiseW;

Sv = bj / cfg.d.Tramp;                     % victim slope
SI = intf.B / intf.Tramp;                  % interferer slope
n  = (0:Nr-1) / Fs;                        % fast time within the victim ramp
t  = tm(:) + n;                            % nj x Nr absolute sample times
tauI = mod(t - intf.t0, intf.Tc);          % time inside the interferer chirp period
active = tauI < intf.Tramp;
fif = Sv * n - (intf.f0 + SI * tauI);      % IF (difference) frequency, Hz
pass = active & abs(fif) < Fs / 2;         % ideal complex anti-aliasing filter
ph = 2 * pi * cumsum(fif, 2) / Fs + 2 * pi * rand(nj, 1);
xi = sqrt(INR) * pass .* exp(1j * ph);     % nj x Nr
steer = exp(1j * pi * sind(intf.theta) * (0:cfg.rf.Nrx - 1));
XI = reshape(xi.', Nr, 1, nj) .* reshape(steer, 1, cfg.rf.Nrx);
if nargout > 1
    info.INR_dB = 10 * log10(INR);
    info.dutyFraction = mean(pass(:));
end
end

function [X, frac] = ncr_mitigate(cfg, X)
%NCR_MITIGATE  Time-domain interference excision (zeroing) per chirp.
%
%   Samples whose channel-averaged magnitude exceeds zeroK times a robust
%   noise estimate are set to zero together with zeroGuard neighbours. The
%   robust estimate uses the median magnitude of a unit-variance complex
%   Gaussian, sqrt(ln 2).
frac = 0;
if ~strcmpi(cfg.intf.mitigation, 'zeroing'), return; end
mag = squeeze(mean(abs(X), 2));                          % Nr x nj
s = median(mag, 1) / sqrt(log(2));
mask = mag > cfg.intf.zeroK * s;
g = cfg.intf.zeroGuard;
if g > 0
    mask = conv2(double(mask), ones(2*g + 1, 1), 'same') > 0;
end
X = X .* reshape(~mask, size(X, 1), 1, size(X, 3));
frac = mean(mask(:));
end

%% ============================================================================
%%  SIGNAL PROCESSING AND DETECTION
%% ============================================================================

function cube = ncr_process(cfg, X, j, nj, bj)
%NCR_PROCESS  Range FFT, Doppler FFT and receive beamforming for sector j.
%
%   cube = NCR_PROCESS(cfg, X, j, nj, bj) returns
%       P      power cube, range x Doppler x beam (Nk x Nd x Na)
%       Z      P normalised by a robust per-beam noise-floor estimate
%       r      range axis (m), bins with 0 < r <= Rproc
%       v      radial-velocity axis (m/s), positive = receding
%       th     beam angles (deg)
%       floor  noise floor per beam (same units as P)
%       dR, dv range bin and physical Doppler resolution
%   Hann windows are applied in all three dimensions.
Nr = cfg.wf.Nr;  Nrx = cfg.rf.Nrx;  Nd = cfg.proc.Nd;  Na = cfg.ar.Na;

Xr = fft(X .* ncr_hann(Nr), Nr, 1);
r  = (0:Nr-1)' * cfg.const.c / (2 * bj);
kk = find(r > 0 & r <= cfg.proc.Rproc);
Xr = Xr(kk, :, :);
Nk = numel(kk);

Xd = fftshift(fft(Xr .* reshape(ncr_hann(nj), 1, 1, nj), Nd, 3), 3);

ax  = (0:Nrx-1)';
th  = cfg.d.beams(j, :);
Wbf = ncr_hann(Nrx) .* exp(-1j * pi * ax * sind(th));               % Nrx x Na
Y   = Wbf.' * reshape(permute(Xd, [2 1 3]), Nrx, Nk * Nd);
P   = abs(permute(reshape(Y, Na, Nk, Nd), [2 3 1])).^2;             % Nk x Nd x Na

fl = median(reshape(P, Nk * Nd, Na), 1) / log(2);                   % exponential median -> mean
cube.P = P;
cube.Z = P ./ reshape(fl, 1, 1, Na);
cube.r = r(kk);
cube.v = cfg.d.vFine;
cube.th = th;
cube.floor = fl;
cube.dR = cfg.const.c / (2 * bj);
cube.dvRes = cfg.d.lambda / (2 * nj * cfg.wf.Tc);
cube.nj = nj;  cube.bj = bj;  cube.sector = j;
end

function [hit, thr] = ncr_cfar(cfg, P, nj)
%NCR_CFAR  Constant-false-alarm-rate detection on a range x Doppler x beam cube.
%
%   hit = NCR_CFAR(cfg, P, nj) applies, independently on every beam:
%     'CA'  two-dimensional cell-averaging CFAR over range and Doppler. The
%           Doppler guard/training extents scale with the zero-padding
%           factor Nd/nj so that they always span the same number of
%           physical resolution cells. Threshold factor for N training cells:
%               alpha = N (Pfa^(-1/N) - 1).
%     'OS'  ordered-statistic CFAR along range (k-th of 2 Tr cells); alpha
%           is found numerically from Pfa = prod_{i=0}^{k-1} (N-i)/(N-i+alpha).
%   Range edges use only the available training cells; Doppler is circular.
d = cfg.det;  Nd = size(P, 2);
switch upper(d.cfar)
    case 'CA'
        Gd = ceil(1.5 * Nd / nj);
        Td = ceil(2 * Nd / nj) + 2;
        so = ncr_boxsum(P, d.Gr + d.Tr, Gd + Td);
        si = ncr_boxsum(P, d.Gr, Gd);
        on = ncr_boxsum(ones(size(P, 1), Nd), d.Gr + d.Tr, Gd + Td);
        in = ncr_boxsum(ones(size(P, 1), Nd), d.Gr, Gd);
        N  = on - in;
        thr = (d.Pfa.^(-1 ./ N) - 1) .* (so - si);
    case 'OS'
        Nk = size(P, 1);
        offs = [-(d.Gr + d.Tr):-(d.Gr + 1), (d.Gr + 1):(d.Gr + d.Tr)];
        S = nan([size(P), numel(offs)]);
        for q = 1:numel(offs)
            o = offs(q);
            src = (1:Nk) + o;  ok = src >= 1 & src <= Nk;
            S(ok, :, :, q) = P(src(ok), :, :);
        end
        S = sort(S, 4);                                   % NaN sorted last
        nValid = sum(~isnan(S), 4);
        kIdx = max(1, round(d.osK * nValid));
        [I1, I2, I3] = ndgrid(1:size(P, 1), 1:size(P, 2), 1:size(P, 3));
        Zk = S(sub2ind(size(S), I1, I2, I3, kIdx));
        alpha = os_alpha(2 * d.Tr, round(d.osK * 2 * d.Tr), d.Pfa);
        thr = alpha * Zk;
    otherwise
        error('ncr_cfar:type', 'Unknown CFAR type "%s".', d.cfar);
end
hit = P > thr;
end

function a = os_alpha(N, k, Pfa)
f = @(a) prod((N - (0:k-1)) ./ (N - (0:k-1) + a)) - Pfa;
lo = 0;  hi = 1e4;
for it = 1:200
    m = (lo + hi) / 2;
    if f(m) > 0, lo = m; else, hi = m; end
end
a = (lo + hi) / 2;
end

function S = ncr_boxsum(A, hr, hd)
%NCR_BOXSUM  Sliding (2hr+1) x (2hd+1) window sums over the first two dims.
%   Works page-wise on 3-D arrays. Zero padding along dim 1 (range) and
%   circular padding along dim 2 (Doppler, an FFT axis). Uses an integral
%   image, so the cost does not depend on the window size.
[a, c, p] = size(A);
hd = min(hd, floor((c - 1) / 2));
Ap = cat(2, A(:, end-hd+1:end, :), A, A(:, 1:hd, :));
Ap = cat(1, zeros(hr, size(Ap, 2), p), Ap, zeros(hr, size(Ap, 2), p));
C  = cumsum(cumsum(Ap, 1), 2);
C  = cat(1, zeros(1, size(C, 2), p), C);
C  = cat(2, zeros(size(C, 1), 1, p), C);
I = 1:a;  J = 1:c;
S = C(I + 2*hr + 1, J + 2*hd + 1, :) - C(I, J + 2*hd + 1, :) ...
  - C(I + 2*hr + 1, J, :) + C(I, J, :);
end

function L = ncr_localmax3(A)
%NCR_LOCALMAX3  True where A is a local maximum of its 3 x 3 x 3 neighbourhood.
%   Dimension 2 (Doppler) is treated as circular; dims 1 and 3 are not.
[a, b, c] = size(A);
Ap = -inf(a + 2, b + 2, c + 2);
Ap(2:a+1, 2:b+1, 2:c+1) = A;
Ap(2:a+1, 1, 2:c+1)   = A(:, b, :);
Ap(2:a+1, b+2, 2:c+1) = A(:, 1, :);
L = true(a, b, c);
for di = -1:1
    for dj = -1:1
        for dk = -1:1
            if di == 0 && dj == 0 && dk == 0, continue; end
            L = L & (A >= Ap((2:a+1) + di, (2:b+1) + dj, (2:c+1) + dk));
        end
    end
end
end

function det = ncr_detect(cfg, cube, hit)
%NCR_DETECT  Peak picking and sub-bin interpolation of CFAR detections.
%
%   det = NCR_DETECT(cfg, cube, hit) returns one row per detection:
%       [R  v  theta  sector  P  SNR_dB  dR  dvRes]
%   Detections are CFAR hits that are also 3-D local maxima. Range, Doppler
%   and angle are refined by parabolic interpolation of the log-power around
%   the peak (Doppler interpolation is on the zero-padded axis).
P = cube.P;
idx = find(hit & ncr_localmax3(P));
det = zeros(numel(idx), 8);
if isempty(idx), return; end
[ir, id, ia] = ind2sub(size(P), idx);
[Nk, Nd, Na] = size(P);
LP = 10 * log10(P + realmin);
dr = zeros(numel(idx), 1);  dd = dr;  da = dr;
for q = 1:numel(idx)
    i = ir(q);  k = id(q);  a = ia(q);
    if i > 1 && i < Nk
        dr(q) = parab(LP(i-1, k, a), LP(i, k, a), LP(i+1, k, a));
    end
    km = mod(k - 2, Nd) + 1;  kp = mod(k, Nd) + 1;
    dd(q) = parab(LP(i, km, a), LP(i, k, a), LP(i, kp, a));
    if a > 1 && a < Na
        da(q) = parab(LP(i, k, a-1), LP(i, k, a), LP(i, k, a+1));
    end
end
dvF = cfg.d.dv;
dth = cube.th(min(2, Na)) - cube.th(1);
R  = cube.r(ir) + dr * cube.dR;
v  = cube.v(id)' + dd * dvF;
th = cube.th(ia)' + da * dth;
snr = 10 * log10(P(idx) ./ cube.floor(ia)');
det = [R, v, th, cube.sector * ones(numel(idx), 1), P(idx), snr, ...
       cube.dR * ones(numel(idx), 1), cube.dvRes * ones(numel(idx), 1)];
end

function d = parab(ym, y0, yp)
den = ym - 2 * y0 + yp;
if den >= 0 || ~isfinite(den)
    d = 0;
else
    d = 0.5 * (ym - yp) / den;
    d = max(-0.5, min(0.5, d));
end
end

function meas = ncr_cluster(cfg, det)
%NCR_CLUSTER  Single-linkage clustering of moving detections into measurements.
%
%   meas = NCR_CLUSTER(cfg, det) groups detections with |v| >= vStatic that
%   lie within clD metres (Cartesian) and clV m/s of each other (single
%   linkage, DBSCAN-like with min points 1) and returns one
%   power-weighted centroid per cluster:
%       [R  theta  v  Psum  nDet  SNRmax  sector]
%   Extended targets (vehicles) and micro-Doppler (limbs) generate several
%   detections; clustering turns them into one measurement for the tracker.
d = cfg.det;
mv = find(abs(det(:, 2)) >= d.vStatic);
n = numel(mv);
meas = zeros(0, 7);
if n == 0, return; end
D = det(mv, :);
xy = D(:, 1) .* [sind(D(:, 3)), cosd(D(:, 3))];
lab = 1:n;                                         % union-find labels
for a = 1:n-1
    for b = a+1:n
        if norm(xy(a, :) - xy(b, :)) <= d.clD && abs(D(a,2) - D(b,2)) <= d.clV
            la = find_root(lab, a);  lb = find_root(lab, b);
            if la ~= lb, lab(max(la, lb)) = min(la, lb); end
        end
    end
end
for a = 1:n, lab(a) = find_root(lab, a); end
u = unique(lab);
meas = zeros(numel(u), 7);
for q = 1:numel(u)
    m = lab == u(q);
    w = D(m, 5) / sum(D(m, 5));
    meas(q, :) = [w' * D(m, 1), w' * D(m, 3), w' * D(m, 2), sum(D(m, 5)), ...
                  nnz(m), max(D(m, 6)), mode(D(m, 4))];
end
end

function r = find_root(lab, a)
r = a;
while lab(r) ~= r, r = lab(r); end
end

function Zc = ncr_coarse_map(cfg, cube, Zc)
%NCR_COARSE_MAP  Max-pool one sector's normalised cube onto the perception grid.
%
%   Zc (NRc x NDc x NAc) is updated in the angle cells of sector cube.sector.
%   Pooling: Doppler stride 2 with a 3-bin window centred on 0 m/s, pairs of
%   receive beams per coarse angle cell, and all range bins inside each
%   coarse range cell. The perception grid is identical for every sector,
%   whatever its bandwidth and dwell.
d = cfg.d;  Z = cube.Z;  Nd = size(Z, 2);
ic = 1:2:Nd;  im = mod(ic - 2, Nd) + 1;  ip = mod(ic, Nd) + 1;
Zp = max(max(Z(:, ic, :), Z(:, im, :)), Z(:, ip, :));
bpc = cfg.ar.Na / d.cellsPerSector;                      % beams per coarse cell
Zq = zeros(size(Zp, 1), size(Zp, 2), d.cellsPerSector);
for a = 1:d.cellsPerSector
    Zq(:, :, a) = max(Zp(:, :, (a - 1) * bpc + (1:bpc)), [], 3);
end
rc = min(d.NRc, max(1, ceil(cube.r / cfg.grid.dRc)));
cols = (cube.sector - 1) * d.cellsPerSector + (1:d.cellsPerSector);
for r = 1:d.NRc
    rows = rc == r;
    if any(rows)
        Zc(r, :, cols) = max(Zq(rows, :, :), [], 1);
    end
end
end

function [obs, state] = ncr_sense(cfg, scene, t, n, b, state, keepCubes)
%NCR_SENSE  One CPI of the radar: all sector sub-frames, processing, detection.
%
%   [obs, state] = NCR_SENSE(cfg, scene, t, n, b, state) transmits n(j) chirps
%   with sweep bandwidth b(j) into each sector j in turn, processes each
%   sub-frame and returns
%       obs.det     detections [R v theta sector P SNR dR dvRes]
%       obs.Zc      coarse normalised map (NRc x NDc x NAc) for perception
%       obs.intfDuty, obs.excised, obs.clip   front-end diagnostics
%       obs.cubes   per-sector cubes (only when keepCubes is true)
%   state carries scene memory (vegetation texture) between CPIs.
if nargin < 7, keepCubes = false; end
[sc, state] = ncr_scatterers(cfg, scene, t, state);
intfOn = scene.intf.enable && isfinite(scene.intf.EIRP_dBm) && ...
         t >= scene.intf.on && t <= scene.intf.off;
d = cfg.d;
obs.Zc = zeros(d.NRc, d.NDc, d.NAc);
obs.det = zeros(0, 8);
obs.intfDuty = 0;  obs.excised = 0;  obs.clip = 0;
if keepCubes, obs.cubes = cell(1, cfg.ar.M); end
tStart = (t - 1) * cfg.sim.Tf;
for j = 1:cfg.ar.M
    tm = tStart + (0:n(j) - 1) * cfg.wf.Tc;
    [X, fi] = ncr_frontend(cfg, sc, scene.intf, j, n(j), b(j), tm, intfOn);
    [X, ex] = ncr_mitigate(cfg, X);
    cube = ncr_process(cfg, X, j, n(j), b(j));
    hit  = ncr_cfar(cfg, cube.P, n(j));
    obs.det = [obs.det; ncr_detect(cfg, cube, hit)];
    obs.Zc  = ncr_coarse_map(cfg, cube, obs.Zc);
    obs.intfDuty = obs.intfDuty + fi.intfDuty * n(j) / cfg.bud.Nc;
    obs.excised  = obs.excised + ex * n(j) / cfg.bud.Nc;
    obs.clip     = max(obs.clip, fi.clipFraction);
    if keepCubes, obs.cubes{j} = cube; end
    tStart = tm(end) + cfg.wf.Tc;
end
end

%% ============================================================================
%%  TRACKING AND METRICS
%% ============================================================================

function [trk, assoc] = ncr_tracker(cfg, trk, meas, t)
%NCR_TRACKER  One update of the EKF / GNN multi-target tracker.
%
%   trk = NCR_TRACKER(cfg) initialises an empty tracker.
%   [trk, assoc] = NCR_TRACKER(cfg, trk, meas, t) predicts all tracks to CPI
%   t, associates the measurements meas = [R theta v ...] (moving clusters
%   only) by global nearest neighbour (Hungarian on the Mahalanobis
%   distance, chi-square gated), updates with an extended Kalman filter,
%   manages track status, and returns for every measurement
%       assoc(i) = id of the associated track (0 = none), and
%       assoc(i, 2) = 1 if that track is confirmed after this update.
%
%   Model: state [x y vx vy]', constant velocity with white acceleration
%   (std q), measurement h(x) = [R, theta, v_r] with
%       R = sqrt(x^2 + y^2), theta = atan2(x, y), v_r = (x vx + y vy) / R.
%   Status: tentative -> confirmed after M hits in the last N CPIs;
%   deleted after maxMissConf (confirmed) or maxMissTent (tentative)
%   consecutive misses.
if nargin == 1
    trk = struct('tracks', struct('id', {}, 'x', {}, 'P', {}, 'hits', {}, ...
                 'misses', {}, 'conf', {}, 'hist', {}, 'born', {}), 'nextId', 1);
    return
end
c = cfg.trk;  T = cfg.sim.Tf;
F = [eye(2), T * eye(2); zeros(2), eye(2)];
G = [T^2 / 2 * eye(2); T * eye(2)];
Q = c.q^2 * (G * G');
Rm = diag([c.sigR^2, (c.sigTh * pi / 180)^2, c.sigV^2]);

% ---------------------------------------------------------------- predict
tr = trk.tracks;
for i = 1:numel(tr)
    tr(i).x = F * tr(i).x;
    tr(i).P = F * tr(i).P * F' + Q;
end

% ---------------------------------------------------------------- associate
nm = size(meas, 1);  nt = numel(tr);
assoc = zeros(nm, 2);
cost = inf(nt, nm);
S_all = cell(nt, 1);  H_all = cell(nt, 1);  zp_all = cell(nt, 1);
for i = 1:nt
    [zp, H] = hfun(tr(i).x);
    S = H * tr(i).P * H' + Rm;
    S_all{i} = S;  H_all{i} = H;  zp_all{i} = zp;
    Si = inv(S);
    for q = 1:nm
        z = [meas(q, 1); meas(q, 2) * pi / 180; meas(q, 3)];
        y = z - zp;  y(2) = wrap(y(2));
        d2 = y' * Si * y;
        if d2 <= c.gate, cost(i, q) = d2; end
    end
end
a = ncr_hungarian(cost);                  % a(i) = measurement given to track i
used = false(nm, 1);

% ---------------------------------------------------------------- update
keep = true(nt, 1);
for i = 1:nt
    if a(i) > 0
        q = a(i);  used(q) = true;
        z = [meas(q, 1); meas(q, 2) * pi / 180; meas(q, 3)];
        y = z - zp_all{i};  y(2) = wrap(y(2));
        K = tr(i).P * H_all{i}' / S_all{i};
        tr(i).x = tr(i).x + K * y;
        tr(i).P = (eye(4) - K * H_all{i}) * tr(i).P;
        tr(i).hits = tr(i).hits + 1;  tr(i).misses = 0;
        tr(i).hist = [tr(i).hist(2:end), 1];
    else
        tr(i).misses = tr(i).misses + 1;
        tr(i).hist = [tr(i).hist(2:end), 0];
    end
    if ~tr(i).conf && sum(tr(i).hist) >= c.M
        tr(i).conf = true;
    end
    if (tr(i).conf && tr(i).misses > c.maxMissConf) || ...
       (~tr(i).conf && tr(i).misses >= c.maxMissTent)
        keep(i) = false;
    end
    if a(i) > 0
        assoc(a(i), :) = [tr(i).id, double(tr(i).conf)];
    end
end
tr = tr(keep);

% ---------------------------------------------------------------- initiate
for q = find(~used)'
    Rr = meas(q, 1);  th = meas(q, 2) * pi / 180;  vr = meas(q, 3);
    J = [sin(th), Rr * cos(th); cos(th), -Rr * sin(th)];        % d[x y]/d[R th]
    Pp = J * diag([c.sigR^2, (c.sigTh * pi / 180)^2]) * J';
    u = [sin(th); cos(th)];
    x0 = [Rr * u; vr * u];
    Pv = c.sigV0^2 * eye(2) + (c.sigV^2 - c.sigV0^2) * (u * u');  % radial component known
    nt = numel(tr) + 1;
    tr(nt).id = trk.nextId;  tr(nt).x = x0;
    tr(nt).P = blkdiag(Pp, Pv);
    tr(nt).hits = 1;  tr(nt).misses = 0;  tr(nt).conf = false;
    tr(nt).hist = [zeros(1, c.N - 1), 1];  tr(nt).born = t;
    assoc(q, :) = [trk.nextId, 0];
    trk.nextId = trk.nextId + 1;
end
trk.tracks = tr;
end

function [z, H] = hfun(x)
px = x(1);  py = x(2);  vx = x(3);  vy = x(4);
R = max(sqrt(px^2 + py^2), 1e-3);
vr = (px * vx + py * vy) / R;
z = [R; atan2(px, py); vr];
H = [px / R,              py / R,              0,      0;
     py / R^2,           -px / R^2,            0,      0;
     (vx - vr * px / R) / R, (vy - vr * py / R) / R, px / R, py / R];
end

function a = wrap(a)
a = mod(a + pi, 2 * pi) - pi;
end

function [assign, total] = ncr_hungarian(C)
%NCR_HUNGARIAN  Optimal assignment (Hungarian / Kuhn-Munkres, O(n^3)).
%
%   [assign, total] = NCR_HUNGARIAN(C) minimises the total cost of assigning
%   rows to columns of the n x m cost matrix C. assign(i) is the column given
%   to row i (0 if unassigned or if the only option has infinite cost).
%   Implementation: shortest augmenting path with row/column potentials;
%   index 1 plays the role of the dummy column 0 of the classical algorithm.
[n, m] = size(C);
assign = zeros(n, 1);  total = 0;
if n == 0 || m == 0, return; end
N = max(n, m);
fin = C(isfinite(C));
if isempty(fin), return; end
big = (max(abs(fin)) + 1) * 10 * N;
A = big * ones(N);
Cf = C;  Cf(~isfinite(Cf)) = big;
A(1:n, 1:m) = Cf;

u = zeros(1, N + 1);  v = zeros(1, N + 1);
p = zeros(1, N + 1);  way = zeros(1, N + 1);
for i = 1:N
    p(1) = i;  j0 = 1;
    minv = inf(1, N + 1);  used = false(1, N + 1);
    while true
        used(j0) = true;
        i0 = p(j0);  delta = inf;  j1 = 0;
        for j = 2:N+1
            if ~used(j)
                cur = A(i0, j - 1) - u(i0 + 1) - v(j);
                if cur < minv(j), minv(j) = cur;  way(j) = j0; end
                if minv(j) < delta, delta = minv(j);  j1 = j; end
            end
        end
        for j = 1:N+1
            if used(j)
                u(p(j) + 1) = u(p(j) + 1) + delta;  v(j) = v(j) - delta;
            else
                minv(j) = minv(j) - delta;
            end
        end
        j0 = j1;
        if p(j0) == 0, break; end
    end
    while true
        j1 = way(j0);  p(j0) = p(j1);  j0 = j1;
        if j0 == 1, break; end
    end
end
for j = 2:N+1
    r = p(j);  c = j - 1;
    if r >= 1 && r <= n && c <= m && isfinite(C(r, c))
        assign(r) = c;
        total = total + C(r, c);
    end
end
end

function [d, loc, card] = ncr_ospa(X, Y, c, p)
%NCR_OSPA  Optimal sub-pattern assignment distance between two point sets.
%
%   d = NCR_OSPA(X, Y, c, p) for X (m x 2) and Y (n x 2) positions, cut-off c
%   and order p (Schuhmacher, Vo and Vo, IEEE TSP 2008):
%       d = ( (1/n) ( min_pi sum_i min(c, |x_i - y_pi(i)|)^p + c^p (n - m) ) )^(1/p),
%   with m <= n (sets swapped if needed). loc and card return the
%   localisation and cardinality components.
m = size(X, 1);  n = size(Y, 1);
if m == 0 && n == 0, d = 0; loc = 0; card = 0; return; end
if m == 0 || n == 0, d = c; loc = 0; card = c; return; end
if m > n, [X, Y] = deal(Y, X); [m, n] = deal(n, m); end
D = zeros(m, n);
for i = 1:m
    D(i, :) = sqrt(sum((Y - X(i, :)).^2, 2))';
end
D = min(c, D).^p;
[~, cost] = ncr_hungarian(D);
d    = ((cost + c^p * (n - m)) / n)^(1 / p);
loc  = (cost / n)^(1 / p);
card = (c^p * (n - m) / n)^(1 / p);
end

function ev = ncr_evaluate(cfg, truth, det, meas, trk, n)
%NCR_EVALUATE  Per-CPI performance metrics against ground truth.
%
%   Detection level (clusters of moving detections):
%     pd(k)      target k detected (a measurement within the association gates)
%     nFA        moving measurements not associated with any target
%     nPts(k)    moving detections on target k (point-cloud density)
%     rErr       range errors of detections w.r.t. the nearest scattering centre
%   Track level (confirmed tracks):
%     cover(k)   a confirmed track within coverR of target k
%     ospa       OSPA distance (cut-off c, order p) between confirmed tracks
%                and active targets, with localisation and cardinality parts
%     posErr     position errors of covering tracks
%   Resources:
%     dwellOnTgt fraction of the chirp budget spent on sectors with moving targets
nT = numel(truth);
ev.pd = nan(1, nT);  ev.nPts = nan(1, nT);  ev.cover = nan(1, nT);
ev.rErr = [];  ev.posErr = [];
used = false(size(meas, 1), 1);
mvDet = det(abs(det(:, 2)) >= cfg.det.vStatic, :);
for k = 1:nT
    tr = truth(k);
    if ~tr.active, continue; end
    j = tr.sector;
    gV = max(1.5, 1.5 * cfg.d.lambda / (2 * n(j) * cfg.wf.Tc));
    if ~isempty(meas)
        m = abs(meas(:, 1) - tr.R) <= 3 & abs(meas(:, 2) - tr.theta) <= 6 & ...
            abs(meas(:, 3) - tr.vr) <= gV + 2.5;
        ev.pd(k) = any(m);  used = used | m;
    else
        ev.pd(k) = 0;
    end
    if ~isempty(mvDet)
        md = abs(mvDet(:, 1) - tr.R) <= 3 & abs(mvDet(:, 3) - tr.theta) <= 6;
        ev.nPts(k) = nnz(md);
        for q = find(md)'
            [~, i] = min(abs(tr.Rparts - mvDet(q, 1)));
            ev.rErr = [ev.rErr; mvDet(q, 1) - tr.Rparts(i)]; %#ok<AGROW>
        end
    else
        ev.nPts(k) = 0;
    end
end
ev.nFA = nnz(~used);

% ---------------------------------------------------------------- tracks
tp = zeros(0, 2);
for i = 1:numel(trk.tracks)
    if trk.tracks(i).conf, tp = [tp; trk.tracks(i).x(1:2)']; end %#ok<AGROW>
end
act = find([truth.active]);
gt = reshape([truth(act).pos], 2, [])';
for k = act
    if isempty(tp)
        ev.cover(k) = 0;
    else
        dk = sqrt(sum((tp - truth(k).pos).^2, 2));
        ev.cover(k) = any(dk <= cfg.trk.coverR);
        if ev.cover(k), ev.posErr = [ev.posErr; min(dk)]; end
    end
end
[ev.ospa, ev.ospaLoc, ev.ospaCard] = ncr_ospa(tp, gt, cfg.trk.ospaC, cfg.trk.ospaP);
ev.nConf = size(tp, 1);
tgtSec = unique([truth(act).sector]);
ev.dwellOnTgt = sum(n(tgtSec)) / cfg.bud.Nc;
end

%% ============================================================================
%%  NEUROMORPHIC PERCEPTION AND LEARNING
%% ============================================================================

function E = ncr_encode(cfg, Zc, ZcPrev)
%NCR_ENCODE  Spike encoding of the coarse radar map.
%
%   E = NCR_ENCODE(cfg, Zc, ZcPrev) returns the encoder drive E in [0, 1)
%   for every input neuron (NRc x NDc x NAc). How E becomes spikes depends
%   on cfg.enc.type (see NCR_SNN_RUN):
%     'peak-rate'  peak selection along Doppler and angle: only local
%                  maxima above zth emit; Poisson rate pMax * E with the
%                  saturating law E = rho / (rho + rho0), rho = Z - Z_th.
%     'rate'       same rate law for every cell above zth (no peak selection).
%     'latency'    peaks as in 'peak-rate', one spike per neuron at step
%                  ceil(T (1 - E)) (time-to-first-spike).
%     'delta'      temporal contrast: peaks whose power rose by more than
%                  delta_dB since the previous CPI (event-camera style).
zth = 10^(cfg.enc.zth_dB / 10);
rho = max(Zc - zth, 0);
Esat = rho ./ (rho + cfg.enc.rho0);
switch lower(cfg.enc.type)
    case 'rate'
        E = Esat;
    case {'peak-rate', 'latency'}
        E = Esat .* peaks(cfg, Zc, zth);
    case 'delta'
        if isempty(ZcPrev), ZcPrev = zeros(size(Zc)); end
        rise = 10 * log10((Zc + 1) ./ (ZcPrev + 1));
        E = Esat .* peaks(cfg, Zc, zth) .* (rise > cfg.enc.delta_dB);
    otherwise
        error('ncr_encode:type', 'Unknown encoder "%s".', cfg.enc.type);
end
end

function pk = peaks(cfg, Zc, zth)
d = cfg.d;
Za = cat(3, -inf(d.NRc, d.NDc, 1), Zc, -inf(d.NRc, d.NDc, 1));
pk = Zc > circshift(Zc, 1, 2) & Zc >= circshift(Zc, -1, 2) & ...   % ties -> one peak
     Zc > Za(:, :, 1:end-2) & Zc >= Za(:, :, 3:end) & Zc > zth;
end

function net = ncr_snn_init(cfg)
%NCR_SNN_INIT  Spiking network: input layer, LIF feature maps, sector outputs.
%
%   Topology (weight-shared, convolution-like):
%     input   NDc x Np Poisson/TTFS neurons (Doppler channel x range-angle cell)
%     hidden  K LIF neurons per cell; neuron (k, p) sees the Doppler spectrum
%             of cell p through the shared kernel W(k, :)  (K x NDc plastic weights)
%     output  one LIF neuron per sector pooling the hidden spikes of its cells
%   The network starts as a permissive energy detector (weights near w0).
s = cfg.snn;
net.W = s.w0 * (1 + s.wSpread * (2 * rand(s.K, cfg.d.NDc) - 1));
net.W = min(net.W, s.wMax);
net.eTrace = zeros(s.K, cfg.d.NDc);        % eligibility carried across CPIs
net.rBar = 0;                              % running reward baseline ('global' rule)
net.nUpdates = 0;
end

function [res, rec] = ncr_snn_run(cfg, net, E)
%NCR_SNN_RUN  Simulate the spiking network for one CPI.
%
%   res = NCR_SNN_RUN(cfg, net, E) drives the network with the encoder output
%   E (NRc x NDc x NAc) for cfg.enc.T time steps and returns
%       res.C      hidden spike counts (K x Np)
%       res.y      output spike counts per sector (1 x M)
%       res.Elig   STDP eligibility of the shared synapses per cell (K x NDc x Np)
%       res.nIn, res.nHid, res.nOut   spike totals
%       res.synOps  synaptic events (input spikes x K fan-out + hidden spikes)
%   rec (optional) holds per-step activity per sector for raster plots.
%
%   Hidden dynamics (discrete-time LIF, reset by subtraction):
%       V[t] = lamV (V[t-1] - Vth S[t-1]) + sum_d W(k,d) s_d[t]
%       S[t] = H(V[t] - Vth)
%   Pair-based STDP eligibility with traces xbar (pre) and ybar (post):
%       e += Ap * S_post[t] * xbar[t]  -  Am * ybar[t-1] * s_pre[t]
s = cfg.snn;  lr = cfg.lrn;  d = cfg.d;
K = s.K;  Np = d.Np;  ND = d.NDc;  T = cfg.enc.T;  M = cfg.ar.M;
Ein = reshape(permute(E, [2 1 3]), ND, Np);            % Doppler x cell
lat = strcmpi(cfg.enc.type, 'latency');
if lat
    tFire = ceil(T * (1 - Ein));  tFire(Ein <= 0) = 0;
else
    Pin = cfg.enc.pMax * Ein;
end
active = any(Ein > 0, 1);                               % cells that can receive input
V = zeros(K, Np);  ref = zeros(K, Np);  Vo = zeros(1, M);
xpre = zeros(ND, Np);  ypost = zeros(K, Np);
dP = exp(-1 / lr.tauP);  dM = exp(-1 / lr.tauM);
C = zeros(K, Np);  y = zeros(1, M);  Elig = zeros(K, ND, Np);
nIn = 0;  nHid = 0;  nOut = 0;
wantRec = nargout > 1;
if wantRec
    rec.in = zeros(T, M);  rec.hid = zeros(T, M);  rec.out = false(T, M);
end
cellSector = d.sectorOfCell(ceil((1:Np) / d.NRc));     % sector of every cell
for tau = 1:T
    if lat
        Sin = double(tFire == tau);
    else
        Sin = double(rand(ND, Np) < Pin);
    end
    xpre = dP * xpre + Sin;
    V = s.lamV * V + net.W * Sin;
    Sh = double(V >= s.Vth & ref <= 0);
    V = V - s.Vth * Sh;
    if s.refrac > 0
        ref = max(ref - 1, 0) + s.refrac * Sh;
    end
    if any(active)
        a = active;
        Elig(:, :, a) = Elig(:, :, a) ...
            + lr.Ap * reshape(Sh(:, a), K, 1, []) .* reshape(xpre(:, a), 1, ND, []) ...
            - lr.Am * reshape(ypost(:, a), K, 1, []) .* reshape(Sin(:, a), 1, ND, []);
    end
    ypost = dM * ypost + Sh;
    C = C + Sh;
    hs = accumarray(cellSector(:), sum(Sh, 1)', [M 1])';
    Vo = s.lamOut * Vo + hs;
    So = Vo >= s.VthOut;
    Vo = Vo - s.VthOut * So;
    y = y + So;
    ni = sum(Sin(:));  nh = sum(Sh(:));
    nIn = nIn + ni;  nHid = nHid + nh;  nOut = nOut + sum(So);
    if wantRec
        rec.in(tau, :)  = accumarray(cellSector(:), sum(Sin, 1)', [M 1])';
        rec.hid(tau, :) = hs;
        rec.out(tau, :) = So;
    end
end
res.C = C;  res.y = y;  res.Elig = Elig;
res.nIn = nIn;  res.nHid = nHid;  res.nOut = nOut;
res.synOps = nIn * K + nHid;
res.neuronUpdates = T * (d.Nhid + d.Nout);
end

function [net, info] = ncr_rstdp(cfg, net, res, rmap)
%NCR_RSTDP  Reward-modulated STDP update of the shared Doppler kernels.
%
%   [net, info] = NCR_RSTDP(cfg, net, res, rmap) applies the three-factor rule
%       e_kd   <- lamE e_kd + sum_p m_p Elig(k, d, p)
%       W_kd   <- clip(W_kd + eta e_kd, 0, wMax)
%   where the neuromodulator m_p depends on cfg.lrn.rule:
%     'spatial'  m_p = r_p, the cognitive reward released locally in cell p
%                (topographic neuromodulation);
%     'global'   m_p = r_g - rbar for every cell, a single scalar reward
%                prediction error (activity-weighted mean reward minus its
%                running average), as in classical R-STDP;
%     'none'     no plasticity (fixed weights).
%   With finite cfg.lrn.bits the weights are kept on a uniform grid of
%   2^bits levels in [0, wMax] using stochastic rounding (as on neuromorphic
%   hardware with low-precision synapses).
lr = cfg.lrn;  s = cfg.snn;
info.rg = NaN;  info.dWnorm = 0;
if strcmpi(lr.rule, 'none'), return; end
K = s.K;  ND = cfg.d.NDc;  Np = cfg.d.Np;
E2 = reshape(res.Elig, K * ND, Np);
switch lower(lr.rule)
    case 'spatial'
        drive = reshape(E2 * rmap(:), K, ND);
    case 'global'
        act = sum(res.C, 1)';
        if sum(act) > 0
            rg = sum(rmap(:) .* act) / sum(act);
        else
            rg = 0;
        end
        delta = rg - net.rBar;
        net.rBar = lr.rBeta * net.rBar + (1 - lr.rBeta) * rg;
        drive = delta * reshape(sum(E2, 2), K, ND);
        info.rg = rg;
    otherwise
        error('ncr_rstdp:rule', 'Unknown learning rule "%s".', lr.rule);
end
net.eTrace = lr.lamE * net.eTrace + drive;
dW = lr.eta * net.eTrace;
W = min(max(net.W + dW, 0), s.wMax);
if isfinite(lr.bits)
    q = s.wMax / (2^lr.bits - 1);
    lo = floor(W / q);
    W = (lo + (rand(size(W)) < (W / q - lo))) * q;      % stochastic rounding
    W = min(W, s.wMax);
end
info.dWnorm = norm(W(:) - net.W(:));
net.W = W;
net.nUpdates = net.nUpdates + nnz(dW);
end

function rmap = ncr_reward(cfg, det, meas, assoc)
%NCR_REWARD  Spatially resolved cognitive reward on the perception grid.
%
%   rmap(r, a) is computed from the radar's own outputs only (no ground truth):
%     + rew.conf    moving measurement associated with a confirmed track
%     + rew.unc     moving measurement without a confirmed track
%     + rew.static  stationary detection (|v| < vStatic): attention spent on clutter
%       rew.idle    cells without detections (activity cost)
%   Contributions in a cell are summed and clipped to [-1, 1].
d = cfg.d;  w = cfg.rew;
acc = zeros(d.NRc, d.NAc);  touched = false(d.NRc, d.NAc);
st = det(abs(det(:, 2)) < cfg.det.vStatic, :);
for i = 1:size(st, 1)
    [r, a] = cell_of(cfg, st(i, 1), st(i, 3));
    acc(r, a) = acc(r, a) + w.static;  touched(r, a) = true;
end
for i = 1:size(meas, 1)
    [r, a] = cell_of(cfg, meas(i, 1), meas(i, 2));
    if assoc(i, 2) > 0, v = w.conf; else, v = w.unc; end
    acc(r, a) = acc(r, a) + v;  touched(r, a) = true;
end
rmap = max(-1, min(1, acc));
rmap(~touched) = w.idle;
end

function [r, a] = cell_of(cfg, R, th)
r = min(cfg.d.NRc, max(1, ceil(R / cfg.grid.dRc)));
a = min(cfg.d.NAc, max(1, floor((th - cfg.ar.secEdges(1)) / cfg.grid.dAc) + 1));
end

%% ============================================================================
%%  ATTENTION AND RESOURCE ALLOCATION
%% ============================================================================

function varargout = ncr_attention(op, cfg, varargin)
%NCR_ATTENTION  Saliency, attention fields and resource allocation.
%
%   S  = NCR_ATTENTION('saliency', cfg, E, n)     rule-based saliency
%   F  = NCR_ATTENTION('field', cfg, F, S, sSat)  attention-field update
%   u  = NCR_ATTENTION('peak', cfg, F)            sector attention (peak of the field per sector)
%   ut = NCR_ATTENTION('normalise', cfg, u)       normalised attention
%   [n, b] = NCR_ATTENTION('allocate', cfg, ut)   dwell and bandwidth maps
%
%   Saliency (rule-based controller): S(p) = max_d m_j(v_d) E(p, d), with the
%   resolution-aware motion prior m_j(v) = 1 - exp(-v^2 / (2 sigma_j^2)),
%   sigma_j = max(sigV, v_res,j / 2), because a stationary return can appear
%   up to half a Doppler resolution cell away from 0 m/s.
%
%   Field: F <- beta F + (1 - beta) G * min(1, S / sSat), where G is a 3 x 3
%   binomial spatial kernel and the saturation makes any salient return count
%   fully (weak but relevant targets are not out-competed by strong ones).
%
%   Allocation: dwell n_j = nMin + (Nc - M nMin) ut_j (capped at nMax, with
%   the excess redistributed, integer-rounded to sum to Nc); bandwidth
%   b_j = min(Bmax, Bmin + kappa ut_j), with kappa set by bisection so that
%   mean(b) = B0 when attainable.
switch lower(op)
    case 'saliency', varargout{1} = saliency(cfg, varargin{:});
    case 'field',    varargout{1} = field(cfg, varargin{:});
    case 'peak',     varargout{1} = sector_peak(cfg, varargin{:});
    case 'normalise', varargout{1} = normalise(cfg, varargin{:});
    case 'allocate', [varargout{1}, varargout{2}] = allocate(cfg, varargin{:});
    otherwise, error('ncr_attention:op', 'Unknown operation "%s".', op);
end
end

function S = saliency(cfg, E, n)
d = cfg.d;
vres = d.lambda ./ (2 * n * cfg.wf.Tc);
sig  = max(cfg.ctl.sigV, 0.5 * vres(d.sectorOfCell));
m = 1 - exp(-d.vCoarse(:).^2 ./ (2 * sig.^2));             % NDc x NAc
S = reshape(max(E .* reshape(m, 1, d.NDc, d.NAc), [], 2), d.NRc, d.NAc);
end

function F = field(cfg, F, S, sSat)
S = min(1, S / sSat);
k = [1 2 1]' * [1 2 1] / 16;
F = cfg.ctl.beta * F + (1 - cfg.ctl.beta) * conv2(S, k, 'same');
end

function u = sector_peak(cfg, F)
u = max(reshape(max(F, [], 1), cfg.d.cellsPerSector, cfg.ar.M), [], 1);
end

function ut = normalise(cfg, u)
u = u / max(max(u), eps);
ut = (u + cfg.ctl.eps).^cfg.ctl.alpha;
ut = ut / sum(ut);
end

function [n, b] = allocate(cfg, ut)
B = cfg.bud;  wf = cfg.wf;  M = cfg.ar.M;
ut = ut(:)' / sum(ut);
n = B.nMin + (B.Nc - M * B.nMin) * ut;
for it = 1:M
    over = n > B.nMax;
    if ~any(over), break; end
    ex = sum(n(over) - B.nMax);  n(over) = B.nMax;
    fr = n < B.nMax;
    w = ut(fr);  if sum(w) == 0, w = ones(size(w)); end
    n(fr) = n(fr) + ex * w / sum(w);
end
nf = floor(n);  r = B.Nc - sum(nf);
[~, ix] = sort(n - nf, 'descend');
nf(ix(1:r)) = nf(ix(1:r)) + 1;
n = min(nf, B.nMax);
lo = 0;  hi = 100 * M * (wf.Bmax - wf.Bmin);
for it = 1:60
    kb = (lo + hi) / 2;
    if mean(min(wf.Bmax, wf.Bmin + kb * ut)) > wf.B0, hi = kb; else, lo = kb; end
end
b = min(wf.Bmax, wf.Bmin + lo * ut);
end

%% ============================================================================
%%  EXPERIMENTS AND STATISTICS
%% ============================================================================

function log = ncr_episode(cfg, scene, policy, seed, keepSnap)
%NCR_EPISODE  Closed-loop simulation of one episode (Nscans CPIs).
%
%   log = NCR_EPISODE(cfg, scene, policy, seed, keepSnap) runs the
%   perception-action loop with
%     'uniform'       fixed dwell Nc/M and bandwidth B0 in every sector
%     'rule-based'    saliency -> attention field -> maps ('algorithmic' is an alias)
%     'neuromorphic'  spike encoder -> LIF network (R-STDP) -> attention -> maps
%   Every CPI: sense (front-end, processing, CFAR, detection), cluster,
%   track, evaluate against ground truth, compute the cognitive reward,
%   perceive, learn (neuromorphic) and allocate dwell/bandwidth for the next
%   CPI. The neuromorphic episode also evaluates the rule-based field in
%   shadow mode on the same data, to measure their agreement rho_F.
%   keepSnap stores attention fields, example cubes and spike rasters.
if nargin < 5, keepSnap = false; end
rng(seed);
Ns = cfg.sim.Nscans;  M = cfg.ar.M;  nT = numel(scene.tgt);  d = cfg.d;
scene.intf.t0 = rand * scene.intf.Tc;                 % unsynchronised interferer

[n, b] = ncr_attention('allocate', cfg, ones(1, M) / M);
Falg = zeros(d.NRc, d.NAc);  Fneu = Falg;  Ytr = zeros(1, M);
net = ncr_snn_init(cfg);
trk = ncr_tracker(cfg);
state = [];  ZcPrev = [];

log.policy = policy;  log.seed = seed;
log.pd = nan(Ns, nT);  log.nPts = nan(Ns, nT);  log.cover = nan(Ns, nT);
log.nFA = zeros(Ns, 1);  log.ospa = zeros(Ns, 1);  log.nConf = zeros(Ns, 1);
log.ospaLoc = zeros(Ns, 1);  log.ospaCard = zeros(Ns, 1);
log.dwellOnTgt = zeros(Ns, 1);  log.n = zeros(Ns, M);  log.b = zeros(Ns, M);
log.Rcog = nan(Ns, 1);  log.rhoF = nan(Ns, 1);
log.rateIn = nan(Ns, 1);  log.rateHid = nan(Ns, 1);  log.synOps = nan(Ns, 1);
log.Esnn_uJ = nan(Ns, 1);  log.kernel = nan(d.NDc, Ns);  log.dW = nan(Ns, 1);
log.intfDuty = zeros(Ns, 1);  log.excised = zeros(Ns, 1);  log.nDet = zeros(Ns, 1);
log.rErr = [];  log.posErr = [];
snapCPI = unique([2 6 16 Ns]);
cubeCPI = unique([1 round(mean([scene.intf.on scene.intf.off])) Ns]);
if keepSnap
    log.snap.cpi = snapCPI;
    log.snap.Falg = cell(1, numel(snapCPI));  log.snap.Fneu = log.snap.Falg;
    log.snap.rec = {};  log.snap.recCPI = [];
    log.snap.cubeCPI = cubeCPI;
    log.snap.cube = cell(numel(cubeCPI), M);
end
Tcpi = cfg.enc.T * cfg.enc.dtau;

for t = 1:Ns
    % ------------------------------------------------------------ sense
    wantCube = keepSnap && any(t == cubeCPI);
    [obs, state] = ncr_sense(cfg, scene, t, n, b, state, wantCube);
    log.n(t, :) = n;  log.b(t, :) = b;
    log.intfDuty(t) = obs.intfDuty;  log.excised(t) = obs.excised;
    log.nDet(t) = size(obs.det, 1);

    % ------------------------------------------------------------ track and evaluate
    meas = ncr_cluster(cfg, obs.det);
    [trk, assoc] = ncr_tracker(cfg, trk, meas, t);
    truth = ncr_truth(cfg, scene, t);
    ev = ncr_evaluate(cfg, truth, obs.det, meas, trk, n);
    log.pd(t, :) = ev.pd;  log.nPts(t, :) = ev.nPts;  log.cover(t, :) = ev.cover;
    log.nFA(t) = ev.nFA;  log.ospa(t) = ev.ospa;  log.nConf(t) = ev.nConf;
    log.ospaLoc(t) = ev.ospaLoc;  log.ospaCard(t) = ev.ospaCard;
    log.dwellOnTgt(t) = ev.dwellOnTgt;
    log.rErr = [log.rErr; ev.rErr];  log.posErr = [log.posErr; ev.posErr];

    % ------------------------------------------------------------ perceive and act
    rmap = ncr_reward(cfg, obs.det, meas, assoc);
    E = ncr_encode(cfg, obs.Zc, ZcPrev);
    ZcPrev = obs.Zc;
    Salg = ncr_attention('saliency', cfg, E, n);
    switch lower(policy)
        case 'uniform'
            [n, b] = ncr_attention('allocate', cfg, ones(1, M) / M);

        case {'rule-based', 'algorithmic'}
            Falg = ncr_attention('field', cfg, Falg, Salg, cfg.ctl.sSatAlg);
            log.Rcog(t) = weighted_reward(rmap, Salg);
            u = ncr_attention('peak', cfg, Falg);
            [n, b] = ncr_attention('allocate', cfg, ncr_attention('normalise', cfg, u));

        case 'neuromorphic'
            if keepSnap && any(t == [1 Ns])
                [res, rec] = ncr_snn_run(cfg, net, E);
                log.snap.rec{end+1} = rec;  log.snap.recCPI(end+1) = t;
            else
                res = ncr_snn_run(cfg, net, E);
            end
            [net, li] = ncr_rstdp(cfg, net, res, rmap);
            Sneu = reshape(sum(res.C, 1), d.NRc, d.NAc) / (cfg.snn.K * cfg.enc.T);
            Fneu = ncr_attention('field', cfg, Fneu, Sneu, cfg.ctl.sSatNeu);
            Falg = ncr_attention('field', cfg, Falg, Salg, cfg.ctl.sSatAlg);   % shadow
            log.rhoF(t) = pearson(Falg(:), Fneu(:));
            log.Rcog(t) = weighted_reward(rmap, Sneu);
            Ytr = cfg.ctl.beta * Ytr + (1 - cfg.ctl.beta) * min(1, res.y / cfg.ctl.ySat);
            [n, b] = ncr_attention('allocate', cfg, ncr_attention('normalise', cfg, Ytr));
            log.rateIn(t)  = res.nIn  / d.Nin  / Tcpi;
            log.rateHid(t) = res.nHid / d.Nhid / Tcpi;
            log.synOps(t)  = res.synOps;
            nUpd = 0;  if ~strcmpi(cfg.lrn.rule, 'none'), nUpd = numel(net.W); end
            log.Esnn_uJ(t) = (res.synOps * cfg.en.Esop_pJ + nUpd * cfg.en.Eupd_pJ) * 1e-6;
            log.kernel(:, t) = mean(net.W, 1)';
            log.dW(t) = li.dWnorm;
        otherwise
            error('ncr_episode:policy', 'Unknown policy "%s".', policy);
    end
    if keepSnap
        k = find(snapCPI == t);
        if ~isempty(k), log.snap.Falg{k} = Falg; log.snap.Fneu{k} = Fneu; end
        if wantCube
            k = find(cubeCPI == t);
            for j = unique([truth(1).sector truth(2).sector])   % vehicle and pedestrian sectors
                cb = obs.cubes{j};
                log.snap.cube{k, j} = struct('PdB', single(10 * log10(cb.P ./ reshape(cb.floor, 1, 1, []))), ...
                    'r', cb.r, 'v', cb.v, 'th', cb.th, 'nj', cb.nj, 'bj', cb.bj);
            end
            log.snap.Zc{k} = single(obs.Zc);
            log.snap.E{k} = single(E);
        end
    end
end
log.Wfinal = net.W;
log.denseMACs = d.denseMACs;
end

function r = weighted_reward(rmap, S)
s = sum(S(:));
if s <= 0, r = 0; else, r = sum(rmap(:) .* S(:)) / s; end
end

function r = pearson(x, y)
x = x - mean(x);  y = y - mean(y);
dn = sqrt(sum(x.^2) * sum(y.^2));
if dn == 0, r = NaN; else, r = sum(x .* y) / dn; end
end

function R = ncr_monte_carlo(cfg, scene, policies, Nmc, label, keepSnap)
%NCR_MONTE_CARLO  Monte Carlo evaluation of one or more policies.
%
%   R = NCR_MONTE_CARLO(cfg, scene, policies, Nmc, label) runs Nmc episodes
%   for every policy in the cell array policies. Run r uses the seed
%   cfg.sim.seed + r for all policies (paired design: identical initial
%   network, interferer timing and scene statistics), which allows paired
%   significance tests. Run 1 keeps snapshots for figures. Uses parfor when
%   cfg.exp.parallel is true and the Parallel Computing Toolbox is present.
%
%   R.logs{r, p}  per-CPI logs (see NCR_EPISODE)
%   R.sum{p}      summary statistics (see NCR_SUMMARISE)
if nargin < 6, keepSnap = true; end
nP = numel(policies);
logs = cell(Nmc, nP);
t0 = tic;
for p = 1:nP
    pol = policies{p};
    usePar = cfg.exp.parallel && ~exist('OCTAVE_VERSION', 'builtin') && ...
             license('test', 'Distrib_Computing_Toolbox');
    if usePar
        L = cell(Nmc, 1);
        parfor r = 1:Nmc
            L{r} = ncr_episode(cfg, scene, pol, cfg.sim.seed + r, keepSnap && r == 1);
        end
        logs(:, p) = L;
    else
        for r = 1:Nmc
            logs{r, p} = ncr_episode(cfg, scene, pol, cfg.sim.seed + r, keepSnap && r == 1);
            if cfg.sim.verbose
                fprintf('  [%s] %-13s run %2d/%d  (%.0f s)\n', label, pol, r, Nmc, toc(t0));
            end
        end
    end
end
R.label = label;  R.policies = policies;  R.Nmc = Nmc;
R.logs = logs;
R.sum = cell(1, nP);
for p = 1:nP
    R.sum{p} = ncr_summarise(cfg, scene, logs(:, p));
end
end

function S = ncr_summarise(cfg, scene, logs)
%NCR_SUMMARISE  Aggregate Monte Carlo logs of one policy (means and 95 % CIs).
%
%   Per-CPI curves (mean and CI half-width over runs):
%     pdCurve      P_D averaged over active moving targets
%     ospaCurve    OSPA distance of confirmed tracks
%     RcogCurve, rhoCurve, rateHidCurve, rateInCurve, dwellCurve, EsnnCurve
%   Scalars (struct with fields m, h: mean and CI half-width over runs):
%     PDall, PDlate, PDtgt(k), PDvehIntf, FA, OSPAlate, OSPAloc, OSPAcard,
%     falseTracks, cover(k), posRMSE, rangeRMSE, ptsVeh, dwellOnTgt, Rcog,
%     rho, synOps, Esnn_uJ, rateIn, rateHid, rateHid1
%   Per-run vectors (for paired tests) are kept in S.run.
Nmc = numel(logs);  Ns = cfg.sim.Nscans;  nT = numel(scene.tgt);
late = cfg.exp.late:Ns;
iI = scene.intf.on:min(scene.intf.off, Ns);     % same window with the interferer on or off
veh = 1;

pdT = nan(Ns, Nmc);  ospa = nan(Ns, Nmc);  Rc = nan(Ns, Nmc);  rho = nan(Ns, Nmc);
rIn = nan(Ns, Nmc);  rHid = nan(Ns, Nmc);  dw = nan(Ns, Nmc);  syn = nan(Ns, Nmc);
Esn = nan(Ns, Nmc);  ker = [];
run = struct('PDall', nan(Nmc, 1), 'PDlate', nan(Nmc, 1), 'PDvehIntf', nan(Nmc, 1), ...
             'FA', nan(Nmc, 1), 'OSPAlate', nan(Nmc, 1), 'OSPAloc', nan(Nmc, 1), ...
             'OSPAcard', nan(Nmc, 1), 'falseTracks', nan(Nmc, 1), 'posRMSE', nan(Nmc, 1), ...
             'rangeRMSE', nan(Nmc, 1), 'ptsVeh', nan(Nmc, 1), 'dwellOnTgt', nan(Nmc, 1), ...
             'Rcog', nan(Nmc, 1), 'rho', nan(Nmc, 1), 'synOps', nan(Nmc, 1), ...
             'Esnn_uJ', nan(Nmc, 1), 'rateIn', nan(Nmc, 1), 'rateHid', nan(Nmc, 1), ...
             'rateHid1', nan(Nmc, 1), 'PDtgt', nan(Nmc, nT), 'cover', nan(Nmc, nT));
for r = 1:Nmc
    L = logs{r};
    pdT(:, r) = mean_omit(L.pd, 2);
    ospa(:, r) = L.ospa;  Rc(:, r) = L.Rcog;  rho(:, r) = L.rhoF;
    rIn(:, r) = L.rateIn;  rHid(:, r) = L.rateHid;  dw(:, r) = L.dwellOnTgt;
    syn(:, r) = L.synOps;  Esn(:, r) = L.Esnn_uJ;
    run.PDall(r)  = mean_omit(L.pd(:), 1);
    run.PDlate(r) = mean_omit(reshape(L.pd(late, :), [], 1), 1);
    run.PDvehIntf(r) = mean_omit(L.pd(iI, veh), 1);
    run.FA(r) = mean(L.nFA);
    run.OSPAlate(r) = mean(L.ospa(late));
    run.OSPAloc(r)  = mean(L.ospaLoc(late));
    run.OSPAcard(r) = mean(L.ospaCard(late));
    nAct = sum(~isnan(L.pd), 2);
    run.falseTracks(r) = mean(max(L.nConf(late) - nAct(late), 0));
    run.posRMSE(r) = sqrt(mean(L.posErr.^2));
    run.rangeRMSE(r) = sqrt(mean(L.rErr.^2));
    run.ptsVeh(r) = mean_omit(L.nPts(late, veh), 1);
    run.dwellOnTgt(r) = mean(L.dwellOnTgt(late));
    run.Rcog(r) = mean_omit(L.Rcog(late), 1);
    run.rho(r) = mean_omit(L.rhoF(late), 1);
    run.synOps(r) = mean_omit(L.synOps(late), 1);
    run.Esnn_uJ(r) = mean_omit(L.Esnn_uJ(late), 1);
    run.rateIn(r) = mean_omit(L.rateIn(late), 1);
    run.rateHid(r) = mean_omit(L.rateHid(late), 1);
    run.rateHid1(r) = mean_omit(L.rateHid(1:3), 1);
    run.PDtgt(r, :) = mean_omit(L.pd, 1);
    run.cover(r, :) = mean_omit(L.cover, 1);
    if all(isfinite(L.kernel(:))), ker = cat(3, ker, L.kernel); end
end
S.run = run;  S.Nmc = Nmc;
[S.pdCurve.m, S.pdCurve.h]   = ncr_stats('ci', pdT, 2);
[S.ospaCurve.m, S.ospaCurve.h] = ncr_stats('ci', ospa, 2);
[S.RcogCurve.m, S.RcogCurve.h] = ncr_stats('ci', Rc, 2);
[S.rhoCurve.m, S.rhoCurve.h] = ncr_stats('ci', rho, 2);
[S.rateHidCurve.m, S.rateHidCurve.h] = ncr_stats('ci', rHid, 2);
[S.rateInCurve.m, S.rateInCurve.h] = ncr_stats('ci', rIn, 2);
[S.dwellCurve.m, S.dwellCurve.h] = ncr_stats('ci', dw, 2);
[S.EsnnCurve.m, S.EsnnCurve.h] = ncr_stats('ci', Esn, 2);
if ~isempty(ker), S.kernelMean = mean(ker, 3); else, S.kernelMean = []; end
f = fieldnames(run);
for i = 1:numel(f)
    [S.(f{i}).m, S.(f{i}).h] = ncr_stats('ci', run.(f{i}), 1);
end
% convergence: first CPI at which the mean attention correlation reaches
% 90 % of its post-learning value
S.tConv = NaN;
if any(isfinite(S.rhoCurve.m))
    target = 0.9 * mean_omit(S.rhoCurve.m(late), 1);
    S.tConv = find(S.rhoCurve.m >= target, 1, 'first');
end
S.denseMACs = cfg.d.denseMACs;
S.Edense_uJ = cfg.d.denseMACs * cfg.en.Emac_pJ * 1e-6;
end

function m = mean_omit(X, dim)
ok = ~isnan(X);  X(~ok) = 0;
m = sum(X, dim) ./ sum(ok, dim);
end

function varargout = ncr_stats(op, varargin)
%NCR_STATS  Small statistics toolkit (no toolboxes required).
%
%   [m, h] = NCR_STATS('ci', X, dim)   mean and 95 % confidence half-width
%                                      (Student t) along dim, ignoring NaN
%   p      = NCR_STATS('pairedt', a, b) two-sided paired t-test p-value
%   tc     = NCR_STATS('tinv', p, df)   Student-t quantile
%   F      = NCR_STATS('tcdf', t, df)   Student-t CDF (via the incomplete beta)
switch lower(op)
    case 'ci',      [varargout{1}, varargout{2}] = ci(varargin{:});
    case 'pairedt', varargout{1} = pairedt(varargin{:});
    case 'tinv',    varargout{1} = tinv_(varargin{:});
    case 'tcdf',    varargout{1} = tcdf_(varargin{:});
    otherwise, error('ncr_stats:op', 'Unknown operation "%s".', op);
end
end

function [m, h] = ci(X, dim)
if nargin < 2, dim = 1; end
ok = ~isnan(X);
n = sum(ok, dim);
X0 = X;  X0(~ok) = 0;
m = sum(X0, dim) ./ max(n, 1);
dev = (X - m);  dev(~ok) = 0;
s = sqrt(sum(dev.^2, dim) ./ max(n - 1, 1));
tc = arrayfun(@(k) tinv_(0.975, max(k - 1, 1)), n);
h = tc .* s ./ sqrt(max(n, 1));
m(n == 0) = NaN;  h(n < 2) = NaN;
end

function p = pairedt(a, b)
d = a(:) - b(:);  d = d(~isnan(d));
n = numel(d);
if n < 2 || std(d) == 0, p = NaN; return; end
t = mean(d) / (std(d) / sqrt(n));
p = 2 * (1 - tcdf_(abs(t), n - 1));
end

function F = tcdf_(t, df)
x = df ./ (df + t.^2);
F = 1 - 0.5 * betainc(x, df / 2, 0.5);
F(t < 0) = 1 - F(t < 0);
end

function t = tinv_(p, df)
lo = 0;  hi = 1e3;
for it = 1:100
    m = (lo + hi) / 2;
    if tcdf_(m, df) < p, lo = m; else, hi = m; end
end
t = (lo + hi) / 2;
end

function X = ncr_experiments(cfg, which)
%NCR_EXPERIMENTS  Experiments reported in the paper.
%
%   X = NCR_EXPERIMENTS(cfg, which) runs the experiments listed in the cell
%   array which (default: all) and saves each to cfg.out.dir/<name>.mat.
%
%   E1 'main'      uniform vs rule-based vs neuromorphic (exp.mainNmc runs)
%   E2 'ablation'  neuromorphic variants (exp.ablNmc runs):
%                    full model; no learning; global (scalar) R-STDP;
%                    rate code without peak selection; latency (TTFS)
%                    code; delta (temporal-contrast) code; 8-bit and 4-bit
%                    synapses; time-domain interference excision
%   E3 'intf'      interferer EIRP sweep for all three policies, plus the
%                    uniform policy with excision (exp.sweepNmc runs)
%   E4 'latency'   time steps per CPI (accuracy vs latency vs energy)
if nargin < 2 || isempty(which), which = {'main', 'ablation', 'intf', 'latency'}; end
if ischar(which), which = {which}; end
if ~exist(cfg.out.dir, 'dir'), mkdir(cfg.out.dir); end
scene = ncr_scene(cfg);
X = struct();
pol3 = {'uniform', 'rule-based', 'neuromorphic'};

if any(strcmpi(which, 'main'))
    say(cfg, 'E1 main comparison');
    E1 = ncr_monte_carlo(cfg, scene, pol3, cfg.exp.mainNmc, 'E1', true);
    E1.cfg = cfg;  E1.scene = scene;
    save_result(cfg, 'E1_main', E1);  X.E1 = E1;
end

if any(strcmpi(which, 'ablation'))
    say(cfg, 'E2 ablation');
    V = { 'Full model',                 {};
          'No learning',                {'lrn.rule', 'none'};
          'Global R-STDP (scalar RPE)', {'lrn.rule', 'global'};
          'Plain rate code',            {'enc.type', 'rate'};
          'Latency code (TTFS)',        {'enc.type', 'latency'};
          'Delta code (contrast)',      {'enc.type', 'delta'};
          '8-bit synapses',             {'lrn.bits', 8};
          '4-bit synapses',             {'lrn.bits', 4};
          'With interference excision', {'intf.mitigation', 'zeroing'} };
    E2.names = V(:, 1)';  E2.res = cell(1, size(V, 1));
    for v = 1:size(V, 1)
        c = variant(cfg, V{v, 2});
        E2.res{v} = ncr_monte_carlo(c, ncr_scene(c), {'neuromorphic'}, cfg.exp.ablNmc, ...
                                    sprintf('E2.%d', v), false);
        E2.res{v}.logs = [];                         % keep summaries only
    end
    save_result(cfg, 'E2_ablation', E2);  X.E2 = E2;
end

if any(strcmpi(which, 'intf'))
    say(cfg, 'E3 interference sweep');
    eirp = cfg.exp.intfEIRP;
    E3.eirp = eirp;  E3.policies = [pol3, {'uniform + excision'}];
    E3.res = cell(numel(eirp), numel(E3.policies));
    for e = 1:numel(eirp)
        c = variant(cfg, {'intf.EIRP_dBm', eirp(e)});
        sc = ncr_scene(c);
        R = ncr_monte_carlo(c, sc, pol3, cfg.exp.sweepNmc, sprintf('E3.%g', eirp(e)), false);
        for p = 1:3, E3.res{e, p} = R.sum{p}; end
        cz = variant(c, {'intf.mitigation', 'zeroing'});
        Rz = ncr_monte_carlo(cz, sc, {'uniform'}, cfg.exp.sweepNmc, sprintf('E3z.%g', eirp(e)), false);
        E3.res{e, 4} = Rz.sum{1};
    end
    save_result(cfg, 'E3_interference', E3);  X.E3 = E3;
end

if any(strcmpi(which, 'latency'))
    say(cfg, 'E4 time-step sweep');
    Tl = cfg.exp.Tsweep;
    E4.T = Tl;  E4.res = cell(1, numel(Tl));
    for k = 1:numel(Tl)
        % keep the encoder's spike budget per CPI constant: pMax * T fixed
        c = variant(cfg, {'enc.T', Tl(k), 'enc.pMax', min(1, cfg.enc.pMax * cfg.enc.T / Tl(k)), ...
                          'enc.dtau', cfg.enc.dtau * cfg.enc.T / Tl(k)});
        R = ncr_monte_carlo(c, ncr_scene(c), {'neuromorphic'}, cfg.exp.sweepNmc, ...
                            sprintf('E4.T%d', Tl(k)), false);
        E4.res{k} = R.sum{1};
    end
    save_result(cfg, 'E4_latency', E4);  X.E4 = E4;
end
end

function c = variant(cfg, nv)
c = cfg;
for i = 1:2:numel(nv)
    c = ncr_set(c, nv{i}, nv{i+1});
end
c = ncr_derive(c);
end

function save_result(cfg, name, S) %#ok<INUSD>
f = fullfile(cfg.out.dir, [name '.mat']);
if exist('OCTAVE_VERSION', 'builtin')
    save('-v7', f, 'S');
else
    save(f, 'S', '-v7.3');
end
end

function say(cfg, msg)
if cfg.sim.verbose, fprintf('\n=== %s (%s) ===\n', msg, datestr(now, 'HH:MM:SS')); end
end

%% ============================================================================
%%  OUTPUTS: FIGURES, TABLES, EXPORT
%% ============================================================================

function ncr_figures(cfg, X)
%NCR_FIGURES  Paper and supplementary figures (vector PDF and PNG).
%
%   NCR_FIGURES(cfg, X) draws every figure for which results are present in
%   X (fields E1..E4 from NCR_EXPERIMENTS or NCR_LOAD_RESULTS) and writes
%   them to cfg.out.dir/figures. Sizes follow the IEEE two-column layout
%   (column 8.86 cm, text width 18.2 cm), with 7-8 pt serif fonts.
%
%   Paper figures
%     fig_scene_attention  scenario and attention fields, rule-based vs SNN
%     fig_dynamics         attention agreement and reward, P_D and OSPA per CPI
%     fig_kernel           learned shared Doppler kernel vs analytical motion prior
%     fig_sweeps           interferer EIRP sweep and time steps per CPI
%   Supplementary figures
%     supp_range_doppler, supp_spike_raster, supp_resource_maps,
%     supp_ospa_components, supp_activity_energy, supp_ablation
%
%   MATLAB R2020a+ writes PDFs with exportgraphics (vector content); older
%   releases and GNU Octave use print. Heat maps are drawn as patches
%   (pcolor), so they remain vector graphics in every PDF.
st = fig_style(cfg);
fd = fullfile(cfg.out.dir, 'figures');
if ~exist(fd, 'dir'), mkdir(fd); end
if isfield(X, 'E1')
    E1 = X.E1;  c = E1.cfg;  sc = E1.scene;
    fig_scene_attention(c, sc, E1, st, fd);
    fig_dynamics(c, sc, E1, st, fd);
    fig_kernel(c, sc, E1, st, fd);
    supp_range_doppler(c, E1, st, fd);
    supp_raster(c, E1, st, fd);
    supp_resources(c, E1, st, fd);
    supp_ospa_components(c, E1, st, fd);
    supp_activity(c, E1, st, fd);
end
if isfield(X, 'E3') || isfield(X, 'E4'), fig_sweeps(X, st, fd); end
if isfield(X, 'E2'), supp_ablation(X.E2, st, fd); end
end

%% ===================================================================== style
function st = fig_style(cfg)
st.octave = exist('OCTAVE_VERSION', 'builtin') > 0;
st.uni = [110 109 104] / 255;      % non-cognitive baseline (neutral)
st.alg = [235 104  52] / 255;      % rule-based controller
st.neu = [ 42 120 214] / 255;      % neuromorphic controller
st.exc = [ 27 175 122] / 255;      % uniform + interference excision
st.ink = [31 31 29] / 255;   st.ink2 = [82 81 78] / 255;
st.grid = [220 219 215] / 255;  st.band = [236 235 231] / 255;
st.veg = [93 138 93] / 255;  st.clut = [138 137 133] / 255;
st.fs = 8;  st.fsS = 7;  st.fsXS = 6.5;
st.font = cfg.out.font;
if st.octave, st.font = 'Liberation Serif'; end
st.dpi = cfg.out.dpi;
st.pcol = {st.uni, st.alg, st.neu};
st.pname = {'Uniform', 'Rule-based', 'Neuromorphic'};
st.pls = {'-.', '--', '-'};  st.plw = [1.0 1.1 1.4];
st.viridis = cmap_ramp([0.2670 0.0049 0.3294; 0.2823 0.0950 0.4173; 0.2788 0.1755 0.4834; ...
    0.2590 0.2515 0.5247; 0.2297 0.3224 0.5457; 0.1994 0.3876 0.5546; 0.1727 0.4488 0.5579; ...
    0.1490 0.5081 0.5573; 0.1276 0.5669 0.5506; 0.1206 0.6258 0.5335; 0.1579 0.6838 0.5017; ...
    0.2461 0.7389 0.4520; 0.3692 0.7889 0.3829; 0.5160 0.8312 0.2943; 0.6785 0.8637 0.1895; ...
    0.8456 0.8873 0.0997; 0.9932 0.9062 0.1439], 256);
st.rdbu = cmap_ramp([0.0196 0.1882 0.3804; 0.1273 0.3958 0.6687; 0.2627 0.5765 0.7647; ...
    0.5665 0.7687 0.8685; 0.8196 0.8980 0.9412; 0.9691 0.9665 0.9649; 0.9922 0.8588 0.7804; ...
    0.9546 0.6418 0.5057; 0.8392 0.3765 0.3020; 0.6923 0.0923 0.1677; 0.4039 0.0000 0.1216], 256);
end

function cm = cmap_ramp(k, N)
x = linspace(0, 1, size(k, 1));  xi = linspace(0, 1, N);
cm = [interp1(x, k(:, 1), xi)', interp1(x, k(:, 2), xi)', interp1(x, k(:, 3), xi)'];
end

function f = fig_new(w, h, st)
% w, h in centimetres
f = figure('Visible', 'off', 'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 w h], ...
           'PaperUnits', 'centimeters', 'PaperPosition', [0 0 w h], 'PaperSize', [w h], ...
           'InvertHardcopy', 'off');
set(f, 'DefaultAxesFontName', st.font, 'DefaultTextFontName', st.font, ...
       'DefaultAxesFontSize', st.fsS, 'DefaultTextFontSize', st.fsS);
end

function ax = fig_axes(pos, st, gridAxis)
ax = axes('Position', pos);  hold(ax, 'on');
set(ax, 'FontName', st.font, 'FontSize', st.fsS, 'TickDir', 'out', 'Box', 'off', ...
        'LineWidth', 0.6, 'XColor', st.ink2, 'YColor', st.ink2, 'TickLength', [0.018 0.018]);
if nargin > 2 && ~isempty(gridAxis)
    set(ax, [upper(gridAxis) 'Grid'], 'on', 'GridColor', st.grid, 'GridAlpha', 1, 'GridLineStyle', '-');
end
end

function fig_labels(ax, xl, yl, st)
if ~isempty(xl), xlabel(ax, xl, 'FontSize', st.fs, 'Color', st.ink); end
if ~isempty(yl), ylabel(ax, yl, 'FontSize', st.fs, 'Color', st.ink); end
end

function fig_panel(ax, str, x, y)
if nargin < 3, x = -0.22; end
if nargin < 4, y = 1.05; end
text(ax, x, y, str, 'Units', 'normalized', 'FontSize', 9, 'FontWeight', 'bold', ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
end

function fig_text(f, x, y, str, varargin)
% text in figure-normalised coordinates (an invisible full-figure axes)
ax0 = findobj(f, 'Tag', 'fig_overlay');
if isempty(ax0)
    ax0 = axes('Parent', f, 'Position', [0 0 1 1], 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
               'Tag', 'fig_overlay');
end
text(ax0, x, y, str, varargin{:});
end

function fig_save(f, fd, name, st)
base = fullfile(fd, name);
if ~st.octave && exist('exportgraphics', 'file') > 0          % MATLAB R2020a and later
    exportgraphics(f, [base '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
    exportgraphics(f, [base '.png'], 'Resolution', st.dpi, 'BackgroundColor', 'white');
else
    print(f, [base '.png'], '-dpng', sprintf('-r%d', st.dpi));
    print(f, [base '.pdf'], '-dpdf', '-painters');
end
close(f);
end

function fig_band(ax, x, m, h, col, alpha)
if nargin < 6, alpha = 0.16; end
x = x(:)';  m = m(:)';  h = h(:)';
ok = isfinite(m) & isfinite(h);
if nnz(ok) < 2, return; end
x = x(ok);  m = m(ok);  h = h(ok);
fill(ax, [x fliplr(x)], [m - h, fliplr(m + h)], col, 'EdgeColor', 'none', 'FaceAlpha', alpha);
end

function y = fig_ma(x)
% centred 3-point moving average that ignores NaN (shrinking window at the ends)
x = x(:)';  n = numel(x);  y = nan(1, n);
for i = 1:n
    w = x(max(1, i - 1):min(n, i + 1));  w = w(isfinite(w));
    if ~isempty(w), y(i) = mean(w); end
end
end

function fig_events(ax, sc, yl, ytxt, st)
% interference window (shaded) and cyclist appearance (dotted)
cy = sc.tgt(3).on;
if sc.intf.enable && isfinite(sc.intf.EIRP_dBm)
    on = sc.intf.on;  off = sc.intf.off;
    patch(ax, [on off off on], [yl(1) yl(1) yl(2) yl(2)], st.band, 'EdgeColor', 'none');
    text(ax, (on + cy) / 2, ytxt, 'interf.', 'FontSize', st.fsXS, 'Color', st.ink2, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
end
plot(ax, [cy cy], yl, ':', 'Color', st.ink2, 'LineWidth', 0.7);
text(ax, cy + 1, ytxt, 'cyclist', 'FontSize', st.fsXS, 'Color', st.ink2, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
end

function fig_legend(ax, h, names, pos, st, framed)
if nargin < 6, framed = false; end
lg = legend(ax, h, names);
set(lg, 'Position', pos, 'TextColor', st.ink);
set(lg, 'FontSize', st.fsS - 1.5 * st.octave);     % Octave renders legend text larger
if framed
    set(lg, 'Box', 'on', 'Color', 'w', 'EdgeColor', 'w');
else
    set(lg, 'Box', 'off', 'Color', 'none');
end
end

function cb = fig_colorbar(ax, pos, ticks, labels, txt, st)
% colour bar at a fixed position (the parent axes keeps its size)
p0 = get(ax, 'Position');
horiz = pos(3) > pos(4);
if horiz, loc = 'northoutside'; else, loc = 'eastoutside'; end
cb = colorbar(ax, 'Location', loc);
set(ax, 'Position', p0);
set(cb, 'Position', pos, 'FontSize', st.fsXS, 'LineWidth', 0.5);
if st.octave
    if horiz, set(cb, 'xtick', ticks, 'xticklabel', labels);
    else,     set(cb, 'ytick', ticks, 'yticklabel', labels); end
else
    set(cb, 'Ticks', ticks, 'TickLabels', labels, 'TickDirection', 'out', 'Color', st.ink2);
end
if isempty(txt), return; end
if horiz       % title above the tick labels, placed in the parent axes' normalised units
    text(ax, (pos(1) + pos(3) / 2 - p0(1)) / p0(3), (pos(2) + pos(4) + 0.065 - p0(2)) / p0(4), txt, ...
         'Units', 'normalized', 'FontSize', st.fsS, 'Color', st.ink, 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom');
else
    ylabel(cb, txt, 'FontSize', st.fsS, 'Color', st.ink);
end
end

function fig_fov(ax, Rm, edges, col, lw, sectors)
a = linspace(edges(1), edges(end), 200);
plot(ax, Rm * sind(a), Rm * cosd(a), '-', 'Color', col, 'LineWidth', lw);
if ~sectors, edges = edges([1 end]); end
for e = edges
    plot(ax, [0 Rm * sind(e)], [0 Rm * cosd(e)], '-', 'Color', col, 'LineWidth', lw);
end
end

function fig_arrowhead(ax, p0, p1, len, col)
d = p1 - p0;  d = d / max(norm(d), eps);  n = [-d(2) d(1)];
tip = p1;  b = p1 - len * d;
patch(ax, [tip(1), b(1) + 0.45 * len * n(1), b(1) - 0.45 * len * n(1)], ...
          [tip(2), b(2) + 0.45 * len * n(2), b(2) - 0.45 * len * n(2)], col, 'EdgeColor', 'none');
end

function [XY, names] = fig_truth_tracks(cfg, sc)
% ground-truth positions of every target per CPI (NaN while inactive)
Ns = cfg.sim.Nscans;  nT = numel(sc.tgt);
XY = nan(nT, Ns, 3);  names = {sc.tgt.name};
for t = 1:Ns
    tr = ncr_truth(cfg, sc, t);
    for k = 1:nT
        if tr(k).active, XY(k, t, :) = [tr(k).pos(1), tr(k).pos(2), tr(k).vr]; end
    end
end
end

%% ===================================================================== paper figures
function fig_scene_attention(cfg, sc, E1, st, fd)
L = E1.logs{1, 3};
if ~isfield(L, 'snap'), return; end
want = [2 16 cfg.sim.Nscans];
sel = find(ismember(L.snap.cpi, want));
if isempty(sel), return; end
cp = L.snap.cpi(sel);  n = numel(sel);
Rm = cfg.proc.Rproc;  edges = cfg.ar.secEdges;
[XY, names] = fig_truth_tracks(cfg, sc);
f = fig_new(18.2, 6.5, st);

% ---------------------------------------------------------------- (a) scenario
ax = fig_axes([0.06 0.15 0.27 0.75], st);
fig_fov(ax, Rm, edges, st.grid, 0.5, true);
for j = 1:cfg.ar.M
    a = cfg.d.secCentre(j);
    text(ax, (Rm + 3) * sind(a), (Rm + 3) * cosd(a), sprintf('%d', j), 'FontSize', st.fsXS, ...
         'Color', st.ink2, 'HorizontalAlignment', 'center');
end
plot(ax, sc.veg.p(:, 1), sc.veg.p(:, 2), '.', 'Color', st.veg, 'MarkerSize', 3);
P = sc.stat.p;  nCar = floor((size(P, 1) - 1) / 4);
for q = 1:nCar
    cc = mean(P(4 * q - 3:4 * q, :), 1);
    rectangle(ax, 'Position', [cc(1) - 1, cc(2) - 2.4, 2, 4.8], 'FaceColor', st.clut, 'EdgeColor', 'none');
end
plot(ax, P(end, 1), P(end, 2), '^', 'Color', st.clut, 'MarkerFaceColor', st.clut, 'MarkerSize', 3.5);
text(ax, P(end, 1) + 2, P(end, 2), 'pole', 'FontSize', st.fsXS, 'Color', st.ink2);
ip = sc.intf.pos;
plot(ax, [0 ip(1)], [0 ip(2)], '--', 'Color', st.alg, 'LineWidth', 0.6);
plot(ax, ip(1), ip(2), 'p', 'Color', st.alg, 'MarkerFaceColor', st.alg, 'MarkerSize', 7);
text(ax, ip(1) - 1.5, ip(2) + 1.8, 'interferer', 'FontSize', st.fsXS, 'Color', st.alg, ...
     'HorizontalAlignment', 'right');
off = [1.5 1.2; 1.5 1.0; 1.6 -1.5];
for k = 1:numel(names)
    x = XY(k, :, 1);  y = XY(k, :, 2);  ok = isfinite(x);  x = x(ok);  y = y(ok);
    if numel(x) < 2, continue; end                       % target not active in this sequence
    plot(ax, x, y, '-', 'Color', st.neu, 'LineWidth', 1.3);
    fig_arrowhead(ax, [x(max(1, end - 3)) y(max(1, end - 3))], [x(end) y(end)], 2.6, st.neu);
    plot(ax, x(1), y(1), 'o', 'MarkerSize', 3.5, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', st.neu, 'LineWidth', 0.9);
    text(ax, x(1) + off(min(k, 3), 1), y(1) + off(min(k, 3), 2), names{k}, 'FontSize', st.fsXS, 'Color', st.ink);
end
text(ax, 28, 3.5, 'vegetation', 'FontSize', st.fsXS, 'Color', st.veg * 0.75, 'HorizontalAlignment', 'center');
text(ax, -27, 9.5, 'parked cars', 'FontSize', st.fsXS, 'Color', st.ink2, 'HorizontalAlignment', 'center');
plot(ax, 0, 0, '^', 'Color', st.ink, 'MarkerFaceColor', st.ink, 'MarkerSize', 5);
set(ax, 'DataAspectRatio', [1 1 1], 'XLim', [-47 47], 'YLim', [-1.5 54], ...
        'XTick', -40:20:40, 'YTick', [0 25 50]);
fig_labels(ax, 'x (m)', 'y (m)', st);
fig_text(f, 0.004, 0.975, '(a)', 'FontSize', 9, 'FontWeight', 'bold', 'VerticalAlignment', 'top');

% ---------------------------------------------------------------- (b) attention fields
re = (0:cfg.d.NRc) * cfg.grid.dRc;
te = edges(1) + (0:cfg.d.NAc) * cfg.grid.dAc;
[RR, TT] = ndgrid(re, te);
Xg = RR .* sind(TT);  Yg = RR .* cosd(TT);
F = cell(2, n);
for k = 1:n
    F{1, k} = L.snap.Falg{sel(k)};  F{2, k} = L.snap.Fneu{sel(k)};
end
vmax = max(cellfun(@(A) max([A(:); eps]), F(:)));
rows = {'Rule-based', 'Neuromorphic'};
w = 0.178;  gap = 0.008;  x0 = 0.385;
for r = 1:2
    for k = 1:n
        ax = fig_axes([x0 + (k - 1) * (w + gap), 0.535 - (r - 1) * 0.43, w, 0.39], st);
        if isempty(F{r, k}), continue; end
        C = zeros(size(Xg));  C(1:end - 1, 1:end - 1) = sqrt(F{r, k} / vmax);   % square-root scale
        pcolor(ax, Xg, Yg, C);  shading(ax, 'flat');
        colormap(ax, st.viridis);  caxis(ax, [0 1]);
        fig_fov(ax, Rm, edges, [0.6 0.6 0.59], 0.4, false);
        for q = 1:numel(names)
            p = squeeze(XY(q, cp(k), 1:2));
            if all(isfinite(p))
                plot(ax, p(1), p(2), 'o', 'MarkerSize', 6.5, 'MarkerEdgeColor', 'w', 'LineWidth', 1.0);
                plot(ax, p(1), p(2), 'o', 'MarkerSize', 6.5, 'MarkerEdgeColor', st.alg, 'LineWidth', 0.5);
            end
        end
        set(ax, 'DataAspectRatio', [1 1 1], 'XLim', [-45 45], 'YLim', [-1.5 52], 'XTick', [-40 0 40], ...
                'YTick', [0 25 50], 'TickLength', [0 0], 'XColor', 'w', 'YColor', 'w', 'Layer', 'top');
        if r == 1, title(ax, sprintf('CPI %d', cp(k)), 'FontSize', st.fs, 'FontWeight', 'normal', 'Color', st.ink); end
        if r == 2
            set(ax, 'XColor', st.ink2);  fig_labels(ax, 'x (m)', '', st);
        else
            set(ax, 'XTickLabel', []);
        end
        if k == 1
            set(ax, 'YColor', st.ink2);
            ylabel(ax, rows{r}, 'FontSize', st.fs, 'Color', st.ink);
        else
            set(ax, 'YTickLabel', []);
        end
    end
end
fig_text(f, 0.33, 0.975, '(b)', 'FontSize', 9, 'FontWeight', 'bold', 'VerticalAlignment', 'top');
tv = [0 0.05 0.1 0.2 0.3];  tv = tv(tv <= vmax);
fig_colorbar(ax, [0.94 0.17 0.012 0.66], sqrt(tv / vmax), arrayfun(@(v) sprintf('%.2g', v), tv, ...
             'UniformOutput', false), 'Attention field', st);
fig_save(f, fd, 'fig_scene_attention', st);
end

function fig_dynamics(cfg, sc, E1, st, fd)
t = 1:cfg.sim.Nscans;  Ns = cfg.sim.Nscans;
f = fig_new(18.2, 4.7, st);
w = 0.2555;  xs = [0.06, 0.3998, 0.7396];

% (a) attention agreement and cognitive reward
ax = fig_axes([xs(1) 0.2 w 0.7], st, 'y');
yl = [-0.45 1.15];
fig_events(ax, sc, yl, 1.13, st);
N = E1.sum{3};
fig_band(ax, t, N.rhoCurve.m, N.rhoCurve.h, st.ink2, 0.15);
hh = plot(ax, t, N.rhoCurve.m, '-', 'Color', st.ink, 'LineWidth', 1.3);
for p = 2:3
    S = E1.sum{p};
    fig_band(ax, t, fig_ma(S.RcogCurve.m), fig_ma(S.RcogCurve.h), st.pcol{p}, 0.18);
    hh(end + 1) = plot(ax, t, fig_ma(S.RcogCurve.m), st.pls{p}, 'Color', st.pcol{p}, 'LineWidth', st.plw(p)); %#ok<AGROW>
end
plot(ax, [1 Ns], [0 0], '-', 'Color', st.ink2, 'LineWidth', 0.5);
set(ax, 'XLim', [1 Ns], 'YLim', yl, 'YTick', -0.4:0.2:1);
fig_labels(ax, 'CPI index', 'Correlation / reward', st);
fig_legend(ax, hh, {'\rho_F (SNN vs rule-based field)', 'R^{cog}, rule-based', 'R^{cog}, neuromorphic'}, ...
           [xs(1) + 0.09, 0.56, 0.16, 0.2], st, true);
fig_panel(ax, '(a)', -0.24);

% (b) detection probability and (c) OSPA
spec = {'pdCurve', [0.5 1.04], 1.035, 'P_D (moving targets)'; 'ospaCurve', [0 5.35], 5.3, 'OSPA distance (m)'};
for q = 1:2
    ax = fig_axes([xs(q + 1) 0.2 w 0.7], st, 'y');
    yl = spec{q, 2};
    fig_events(ax, sc, yl, spec{q, 3}, st);
    hh = gobjects_(3);
    for p = 1:3
        S = E1.sum{p}.(spec{q, 1});
        fig_band(ax, t, fig_ma(S.m), fig_ma(S.h), st.pcol{p}, 0.15);
    end
    for p = 1:3
        S = E1.sum{p}.(spec{q, 1});
        hh(p) = plot(ax, t, fig_ma(S.m), st.pls{p}, 'Color', st.pcol{p}, 'LineWidth', st.plw(p));
    end
    set(ax, 'XLim', [1 Ns], 'YLim', yl);
    fig_labels(ax, 'CPI index', spec{q, 4}, st);
    if q == 1
        fig_legend(ax, hh([3 2 1]), st.pname([3 2 1]), [xs(2) + 0.14, 0.24, 0.11, 0.2], st, true);
    end
    fig_panel(ax, sprintf('(%c)', 'a' + q), -0.22);
end
fig_save(f, fd, 'fig_dynamics', st);
end

function h = gobjects_(n)
% placeholder graphics array (gobjects is not available in GNU Octave)
if exist('OCTAVE_VERSION', 'builtin'), h = zeros(1, n); else, h = gobjects(1, n); end
end

function fig_kernel(cfg, sc, E1, st, fd)
Nmc = size(E1.logs, 1);  Ns = cfg.sim.Nscans;
if ~isfield(E1.logs{1, 3}, 'kernel'), return; end
K = zeros(cfg.d.NDc, Ns, Nmc);
for r = 1:Nmc, K(:, :, r) = E1.logs{r, 3}.kernel / cfg.snn.wMax; end
v = cfg.d.vCoarse(:);  dv = v(2) - v(1);  w0 = cfg.snn.w0 / cfg.snn.wMax;
sel = abs(v) <= 10;  vs = v(sel);
dW = mean(K, 3) - w0;  D = dW(sel, :);
vmin = min(min(D(:)), -0.05);  vmax = max(max(D(:)), 0.05);
Z = D;  Z(D < 0) = D(D < 0) / abs(vmin);  Z(D > 0) = D(D > 0) / vmax;   % two-slope map, white at zero
f = fig_new(8.86, 6.3, st);

ax = fig_axes([0.14 0.17 0.555 0.585], st);
C = zeros(numel(vs) + 1, Ns + 1);  C(1:end - 1, 1:end - 1) = Z;
pcolor(ax, 0.5:1:Ns + 0.5, [vs - dv / 2; vs(end) + dv / 2], C);  shading(ax, 'flat');
colormap(ax, st.rdbu);  caxis(ax, [-1 1]);
[XY, names] = fig_truth_tracks(cfg, sc);
lab = lower(names);
for k = 1:numel(names)
    vr = XY(k, :, 3);  i1 = find(isfinite(vr), 1);
    if isempty(i1), continue; end
    plot(ax, 1:Ns, vr, '--', 'Color', st.ink, 'LineWidth', 0.7);
    if k == 3, ha = 'right'; xt = i1 - 1; else, ha = 'left'; xt = i1 + 1; end
    text(ax, xt, vr(i1) + 0.45, lab{k}, 'FontSize', st.fsXS, 'Color', st.ink, 'HorizontalAlignment', ha, ...
         'VerticalAlignment', 'bottom', 'BackgroundColor', 'w', 'Margin', 0.5);
end
set(ax, 'XLim', [0.5 Ns + 0.5], 'YLim', [-10 10], 'XTick', [1 16 32 48 64], 'Layer', 'top');
fig_labels(ax, 'CPI index', 'Radial velocity (m/s)', st);
tv = [-0.6 -0.3 0 0.15 0.3];  tv = tv(tv >= vmin & tv <= vmax);
tp = tv;  tp(tv < 0) = tv(tv < 0) / abs(vmin);  tp(tv > 0) = tv(tv > 0) / vmax;
fig_colorbar(ax, [0.14 0.865 0.555 0.035], tp, arrayfun(@(x) sprintf('%g', x), tv, 'UniformOutput', false), ...
             'Weight change \DeltaW/w_{max} since initialisation', st);

% profile at the first and last CPI (mean and 95 % CI over runs)
ax2 = fig_axes([0.745 0.17 0.235 0.585], st, 'x');
cols = {1 - 0.45 * (1 - st.neu), st.neu};  idx = [1 Ns];  lw = [1.0 1.4];
hh = gobjects_(3);
for q = 1:2
    [m, h] = ncr_stats('ci', squeeze(K(:, idx(q), :)), 2);
    fill(ax2, [m - h; flipud(m + h)]', [v; flipud(v)]', cols{q}, 'EdgeColor', 'none', 'FaceAlpha', 0.25);
    hh(q) = plot(ax2, m, v, '-', 'Color', cols{q}, 'LineWidth', lw(q));
end
sig = max(cfg.ctl.sigV, cfg.d.lambda / (2 * cfg.d.nUni * cfg.wf.Tc) / 2);
vv = linspace(-10, 10, 400);
hh(3) = plot(ax2, 1 - exp(-vv.^2 / (2 * sig^2)), vv, '--', 'Color', st.alg, 'LineWidth', 1.1);
plot(ax2, [w0 w0], [-10 10], ':', 'Color', st.ink2, 'LineWidth', 0.6);
set(ax2, 'XLim', [0 1.05], 'YLim', [-10 10], 'XTick', [0 0.5 1], 'YTickLabel', []);
fig_labels(ax2, 'W/w_{max}', '', st);
fig_legend(ax2, hh, {'SNN, CPI 1', sprintf('SNN, CPI %d', Ns), 'clutter-notch prior'}, ...
           [0.72 0.78 0.27 0.15], st);
fig_save(f, fd, 'fig_kernel', st);
end

function fig_sweeps(X, st, fd)
f = fig_new(18.2, 4.6, st);
w = 0.2366;  xs = [0.06, 0.3816, 0.7032];
cols = {st.uni, st.alg, st.neu, st.exc};  ls = {'-.', '--', '-', ':'};  mk = {'s', 'd', 'o', '^'};
pn = {'Uniform', 'Rule-based', 'Neuromorphic', 'Uniform + excision'};
if isfield(X, 'E3')
    E3 = X.E3;  e = E3.eirp;  ne = numel(e);
    lab = arrayfun(@(x) sprintf('%g', x), e, 'UniformOutput', false);  lab(~isfinite(e)) = {'off'};
    spec = {'PDvehIntf', [0.45 1.04], 'Vehicle P_D (CPIs 20-44)'; 'FA', [0 4.8], 'False clusters per CPI'};
    for q = 1:2
        ax = fig_axes([xs(q) 0.22 w 0.68], st, 'y');
        hh = gobjects_(4);
        for p = 1:size(E3.res, 2)
            Y = cellfun(@(S) S.(spec{q, 1}).m, E3.res(:, p))';
            H = cellfun(@(S) S.(spec{q, 1}).h, E3.res(:, p))';
            x = (1:ne) + (p - 2.5) * 0.09;
            for i = 1:ne
                if isfinite(H(i)), plot(ax, [x(i) x(i)], Y(i) + [-1 1] * H(i), '-', 'Color', cols{p}, 'LineWidth', 0.7); end
            end
            hh(p) = plot(ax, x, Y, ls{p}, 'Color', cols{p}, 'LineWidth', 1.2, 'Marker', mk{p}, ...
                         'MarkerFaceColor', cols{p}, 'MarkerEdgeColor', 'w', 'MarkerSize', 4);
        end
        set(ax, 'XTick', 1:ne, 'XTickLabel', lab, 'XLim', [0.65 ne + 0.35], 'YLim', spec{q, 2});
        fig_labels(ax, 'Interferer EIRP (dBm)', spec{q, 3}, st);
        if q == 1, fig_legend(ax, hh, pn(1:numel(hh)), [xs(1) + 0.012, 0.25, 0.1, 0.28], st); end
        fig_panel(ax, sprintf('(%c)', 'a' + q - 1), -0.22);
    end
end
if isfield(X, 'E4')
    E4 = X.E4;  T = E4.T;
    pos = [xs(3) 0.22 w 0.68];
    Y = cellfun(@(S) S.PDlate.m, E4.res);  H = cellfun(@(S) S.PDlate.h, E4.res);
    syn = cellfun(@(S) S.synOps.m, E4.res) / 1e3;
    ax = fig_axes(pos, st, 'y');
    for i = 1:numel(T)
        if isfinite(H(i)), plot(ax, [T(i) T(i)], Y(i) + [-1 1] * H(i), '-', 'Color', st.neu, 'LineWidth', 0.7); end
    end
    h1 = plot(ax, T, Y, '-o', 'Color', st.neu, 'MarkerFaceColor', st.neu, 'MarkerEdgeColor', 'w', ...
              'MarkerSize', 4.5, 'LineWidth', 1.4);
    h2 = plot(ax, NaN, NaN, '--s', 'Color', st.ink2, 'MarkerFaceColor', 'w', 'MarkerSize', 4, 'LineWidth', 1.0);
    xl = [T(1) * 0.8, T(end) * 1.25];
    set(ax, 'XScale', 'log', 'XLim', xl, 'XTick', T, 'XTickLabel', arrayfun(@num2str, T, 'UniformOutput', false), ...
            'XMinorTick', 'off', 'YLim', [0.8 1.0]);
    fig_labels(ax, 'Network time steps T per CPI', 'P_D (CPIs 16-64)', st);
    fig_legend(ax, [h1 h2], {'P_D', 'synaptic events'}, [xs(3) + 0.1, 0.25, 0.12, 0.17], st);
    fig_panel(ax, '(c)', -0.22);
    ax2 = axes('Position', pos, 'Color', 'none', 'YAxisLocation', 'right', 'XScale', 'log', 'XLim', xl, ...
               'XTick', [], 'Box', 'off', 'FontName', st.font, 'FontSize', st.fsS, 'TickDir', 'out', ...
               'LineWidth', 0.6, 'YColor', st.ink2, 'XColor', 'none', 'TickLength', [0.018 0.018]);
    hold(ax2, 'on');
    plot(ax2, T, syn, '--s', 'Color', st.ink2, 'MarkerFaceColor', 'w', 'MarkerSize', 4, 'LineWidth', 1.0);
    set(ax2, 'YLim', [0 2]);
    ylabel(ax2, 'Events per CPI (\times10^3)', 'FontSize', st.fs, 'Color', st.ink2);
end
fig_save(f, fd, 'fig_sweeps', st);
end

%% ===================================================================== supplementary figures
function supp_range_doppler(cfg, E1, st, fd)
L = {E1.logs{1, 1}, E1.logs{1, 3}};      % uniform and neuromorphic, run 1
if ~isfield(L{2}, 'snap') || ~isfield(L{1}, 'snap') || ~isfield(L{2}.snap, 'cube'), return; end
cp = L{2}.snap.cubeCPI;
k1 = min(2, numel(cp));  kN = numel(cp);
tr = ncr_truth(cfg, E1.scene, cp(k1));  jV = tr(1).sector;
trN = ncr_truth(cfg, E1.scene, cp(kN));  jP = trN(2).sector;
f = fig_new(18.2, 9.5, st);
cases = {k1, jV, 'Vehicle sector'; kN, jP, 'Pedestrian sector'};
names = {'uniform', 'neuromorphic'};
for row = 1:2
    for col = 1:2
        snap = L{col}.snap;
        if size(snap.cube, 1) < cases{row, 1} || size(snap.cube, 2) < cases{row, 2} || ...
           isempty(snap.cube{cases{row, 1}, cases{row, 2}})
            continue
        end
        cb = snap.cube{cases{row, 1}, cases{row, 2}};
        ax = fig_axes([0.08 + (col - 1) * 0.47, 0.58 - (row - 1) * 0.47, 0.33, 0.34], st);
        imagesc(ax, cb.v, cb.r, max(double(cb.PdB), [], 3));  axis(ax, 'xy');
        colormap(ax, st.viridis);  caxis(ax, [0 40]);
        set(ax, 'XLim', [-12 12], 'YLim', [0 cfg.proc.Rproc], 'Layer', 'top');
        title(ax, sprintf('%s, CPI %d, %s: %d chirps, %.0f MHz', cases{row, 3}, snap.cubeCPI(cases{row, 1}), ...
              names{col}, cb.nj, cb.bj / 1e6), 'FontSize', st.fsS, 'FontWeight', 'normal');
        if row == 2, fig_labels(ax, 'Radial velocity (m/s)', '', st); end
        if col == 1, fig_labels(ax, '', 'Range (m)', st); end
        p = get(ax, 'Position');
        fig_colorbar(ax, [p(1) + p(3) + 0.01, p(2), 0.01, p(4)], 0:10:40, {'0', '10', '20', '30', '40'}, ...
                     'Power above noise (dB)', st);
    end
end
fig_save(f, fd, 'supp_range_doppler', st);
end

function supp_raster(cfg, E1, st, fd)
L = E1.logs{1, 3};
if ~isfield(L, 'snap') || ~isfield(L.snap, 'rec') || numel(L.snap.rec) < 2, return; end
f = fig_new(18.2, 8.0, st);
titles = {'Input events per sector', 'Hidden spikes per sector', 'Output (sector) neurons'};
for row = 1:2
    rec = L.snap.rec{row};
    data = {rec.in, rec.hid, double(rec.out)};
    for col = 1:3
        ax = fig_axes([0.06 + (col - 1) * 0.33, 0.58 - (row - 1) * 0.47, 0.24, 0.33], st);
        if col < 3
            imagesc(ax, 1:cfg.enc.T, 1:cfg.ar.M, data{col}');  axis(ax, 'xy');
            colormap(ax, st.viridis);
            p = get(ax, 'Position');  mx = max([data{col}(:); 1]);
            fig_colorbar(ax, [p(1) + p(3) + 0.008, p(2), 0.008, p(4)], [0 mx], {'0', sprintf('%g', mx)}, '', st);
        else
            [tt, jj] = find(rec.out);
            plot(ax, tt, jj, '|', 'Color', st.neu, 'MarkerSize', 6, 'LineWidth', 1.2);
        end
        set(ax, 'XLim', [0.5 cfg.enc.T + 0.5], 'YLim', [0.5 cfg.ar.M + 0.5], 'YTick', 1:cfg.ar.M, 'Layer', 'top');
        title(ax, sprintf('%s, CPI %d', titles{col}, L.snap.recCPI(row)), 'FontSize', st.fsS, 'FontWeight', 'normal');
        if row == 2, fig_labels(ax, 'Time step', '', st); end
        if col == 1, fig_labels(ax, '', 'Sector', st); end
    end
end
fig_save(f, fd, 'supp_spike_raster', st);
end

function supp_resources(cfg, E1, st, fd)
f = fig_new(18.2, 9.0, st);
t = 1:cfg.sim.Nscans;  scs = cfg.d.secCentre;
pol = [2 3];  nm = {'rule-based', 'neuromorphic'};
for r = 1:2
    L = E1.logs{1, pol(r)};
    ax = fig_axes([0.08, 0.58 - (r - 1) * 0.46, 0.34, 0.33], st);
    imagesc(ax, t, scs, L.n');  axis(ax, 'xy');  colormap(ax, st.viridis);
    caxis(ax, [cfg.bud.nMin cfg.bud.nMax]);
    p = get(ax, 'Position');
    fig_colorbar(ax, [p(1) + p(3) + 0.01, p(2), 0.01, p(4)], [cfg.bud.nMin 64 cfg.bud.nMax], ...
                 {sprintf('%d', cfg.bud.nMin), '64', sprintf('%d', cfg.bud.nMax)}, 'Chirps', st);
    title(ax, sprintf('Dwell (transmit energy), %s', nm{r}), 'FontSize', st.fsS, 'FontWeight', 'normal');
    fig_labels(ax, '', 'Sector centre (deg)', st);
    if r == 2, fig_labels(ax, 'CPI index', '', st); end
    ax = fig_axes([0.57, 0.58 - (r - 1) * 0.46, 0.34, 0.33], st);
    imagesc(ax, t, scs, L.b' / 1e6);  axis(ax, 'xy');  colormap(ax, st.viridis);
    caxis(ax, [cfg.wf.Bmin cfg.wf.Bmax] / 1e6);
    p = get(ax, 'Position');  bt = [cfg.wf.Bmin cfg.wf.Bmax] / 1e6;
    fig_colorbar(ax, [p(1) + p(3) + 0.01, p(2), 0.01, p(4)], bt, arrayfun(@(x) sprintf('%g', x), bt, ...
                 'UniformOutput', false), 'Bandwidth (MHz)', st);
    title(ax, sprintf('Sweep bandwidth, %s', nm{r}), 'FontSize', st.fsS, 'FontWeight', 'normal');
    if r == 2, fig_labels(ax, 'CPI index', '', st); end
end
fig_save(f, fd, 'supp_resource_maps', st);
end

function supp_ospa_components(cfg, E1, st, fd)
f = fig_new(8.86, 6.0, st);
ax = fig_axes([0.15 0.14 0.8 0.74], st, 'y');
Y = zeros(2, 3);  H = Y;
for p = 1:3
    Y(:, p) = [E1.sum{p}.OSPAloc.m; E1.sum{p}.OSPAcard.m];
    H(:, p) = [E1.sum{p}.OSPAloc.h; E1.sum{p}.OSPAcard.h];
end
w = 0.8 / 3;
for g = 1:2
    for s = 1:3
        x0 = g - 0.4 + (s - 1) * w;
        patch(ax, [x0 x0 + w * 0.9 x0 + w * 0.9 x0], [0 0 Y(g, s) Y(g, s)], st.pcol{s}, 'EdgeColor', 'none');
        if isfinite(H(g, s))
            plot(ax, [1 1] * (x0 + w * 0.45), Y(g, s) + [-1 1] * H(g, s), '-', 'Color', st.ink, 'LineWidth', 0.7);
        end
    end
end
top = max(Y(:) + max(H(:), 0));
if ~isfinite(top) || top <= 0, top = 1; end
set(ax, 'XTick', 1:2, 'XTickLabel', {'Localisation', 'Cardinality'}, 'XLim', [0.4 2.6], 'YLim', [0 top * 1.35]);
fig_labels(ax, '', sprintf('OSPA component (m), CPIs %d-%d', cfg.exp.late, cfg.sim.Nscans), st);
hb = gobjects_(3);
for p = 1:3, hb(p) = patch(ax, NaN, NaN, st.pcol{p}, 'EdgeColor', 'none'); end
fig_legend(ax, hb, st.pname, [0.55 0.68 0.35 0.18], st);
fig_save(f, fd, 'supp_ospa_components', st);
end

function supp_activity(cfg, E1, st, fd)
S = E1.sum{3};  t = 1:cfg.sim.Nscans;
f = fig_new(18.2, 6.0, st);
ax = fig_axes([0.07 0.2 0.38 0.72], st, 'y');
fig_band(ax, t, S.rateInCurve.m, S.rateInCurve.h, st.ink2);
fig_band(ax, t, S.rateHidCurve.m, S.rateHidCurve.h, st.neu);
h1 = plot(ax, t, S.rateInCurve.m, '-', 'Color', st.ink2, 'LineWidth', 1.2);
h2 = plot(ax, t, S.rateHidCurve.m, '-', 'Color', st.neu, 'LineWidth', 1.4);
set(ax, 'XLim', [1 cfg.sim.Nscans]);
fig_labels(ax, 'CPI index', 'Mean firing rate per neuron (Hz)', st);
fig_legend(ax, [h1 h2], {'Input layer', 'Hidden layer (LIF)'}, [0.3 0.7 0.14 0.15], st);
ax = fig_axes([0.58 0.2 0.38 0.72], st, 'y');
fig_band(ax, t, S.EsnnCurve.m, S.EsnnCurve.h, st.neu);
h1 = plot(ax, t, S.EsnnCurve.m, '-', 'Color', st.neu, 'LineWidth', 1.4);
h2 = plot(ax, [1 cfg.sim.Nscans], S.Edense_uJ * [1 1], '--', 'Color', st.ink2, 'LineWidth', 1.2);
set(ax, 'YScale', 'log', 'XLim', [1 cfg.sim.Nscans], 'YLim', [3e-3 3]);
fig_labels(ax, 'CPI index', 'Energy per CPI (\muJ, indicative)', st);
fig_legend(ax, [h1 h2], {'SNN (synaptic events + updates)', 'Dense frame-based pass'}, [0.62 0.28 0.25 0.15], st);
fig_save(f, fd, 'supp_activity_energy', st);
end

function supp_ablation(E2, st, fd)
nv = numel(E2.names);
m = @(fld) cellfun(@(R) R.sum{1}.(fld).m, E2.res);
h = @(fld) cellfun(@(R) R.sum{1}.(fld).h, E2.res);
f = fig_new(18.2, 7.5, st);
flds = {'PDlate', 'OSPAlate', 'Esnn_uJ'};
xl = {'P_D (CPIs 16-64)', 'OSPA (m)', 'SNN energy per CPI (\muJ)'};
for q = 1:3
    ax = fig_axes([0.17 + (q - 1) * 0.28, 0.16, 0.24, 0.8], st, 'x');
    Y = m(flds{q});  H = h(flds{q});
    for k = 1:nv
        c = [0.72 0.72 0.70];  if k == 1, c = st.neu; end
        patch(ax, [0 Y(k) Y(k) 0], [k - 0.35 k - 0.35 k + 0.35 k + 0.35], c, 'EdgeColor', 'none');
        if isfinite(H(k)), plot(ax, Y(k) + [-1 1] * H(k), [k k], '-', 'Color', st.ink, 'LineWidth', 0.7); end
    end
    set(ax, 'YDir', 'reverse', 'YTick', 1:nv, 'YLim', [0.4 nv + 0.6]);
    if q == 1, set(ax, 'YTickLabel', E2.names, 'XLim', [0 1]); else, set(ax, 'YTickLabel', {}); end
    fig_labels(ax, xl{q}, '', st);
end
fig_save(f, fd, 'supp_ablation', st);
end

function ncr_tables(cfg, X)
%NCR_TABLES  LaTeX (booktabs) tables and a plain-text summary of all results.
%
%   Writes to cfg.out.dir/tables:
%     tab_link_budget.tex   radar parameters and link budget per target
%     tab_main_results.tex  E1 metrics, mean +/- 95 % CI, paired p-values
%     tab_ablation.tex      E2 neuromorphic variants
%     tab_network.tex       network size, activity and indicative energy
%     summary.txt           all of the above as text
td = fullfile(cfg.out.dir, 'tables');
if ~exist(td, 'dir'), mkdir(td); end
txt = {};

% ---------------------------------------------------------------- link budget
sc = ncr_scene(cfg);
[~, info] = ncr_link_budget(cfg, 10, 1, 1);
nj = cfg.d.nUni;
gD = sum(ncr_hann(nj))^2 / sum(ncr_hann(nj).^2);
Gproc = 10 * log10(cfg.d.cgR * gD * cfg.d.cgX);
rows = {
 'Carrier frequency', sprintf('%.0f GHz', cfg.rf.fc / 1e9);
 'Tx power per channel / Tx elements', sprintf('%g dBm / %d', cfg.rf.Pt_dBm, cfg.rf.Ntx);
 'EIRP (phased beam peak)', sprintf('%.1f dBm', info.EIRP_dBm);
 'Rx channels, element gain', sprintf('%d, %g dBi', cfg.rf.Nrx, cfg.rf.Gr_dBi);
 'Noise figure / system losses', sprintf('%g dB / %g dB', cfg.rf.NF_dB, cfg.rf.Lsys_dB);
 'Sampling rate, samples per chirp', sprintf('%g MHz, %d', cfg.wf.Fs / 1e6, cfg.wf.Nr);
 'Chirp interval, chirps per CPI', sprintf('%g us, %d', cfg.wf.Tc * 1e6, cfg.bud.Nc);
 'Bandwidth per sector (mean budget)', sprintf('%g-%g MHz (%g MHz)', cfg.wf.Bmin/1e6, cfg.wf.Bmax/1e6, cfg.wf.B0/1e6);
 'Noise power per sample', sprintf('%.1f dBm', info.noise_dBm);
 'Per-sample SNR, 1 m^2 at 10 m', sprintf('%.1f dB', info.snr_dB);
 'Processing gain (uniform dwell, Hann)', sprintf('%.1f dB', Gproc)};
tr1 = ncr_truth(cfg, sc, 1);  trN = ncr_truth(cfg, sc, cfg.sim.Nscans);
for k = 1:numel(sc.tgt)
    [~, q] = max([sc.tgt(k).parts.rcs]);
    rcs = 10^(sc.tgt(k).parts(q).rcs / 10);
    if sc.tgt(k).on > 1, R1 = ncr_truth(cfg, sc, sc.tgt(k).on); R1 = R1(k).R; else, R1 = tr1(k).R; end
    s1 = 10 * log10(ncr_link_budget(cfg, R1, rcs)) + Gproc;
    s2 = 10 * log10(ncr_link_budget(cfg, trN(k).R, rcs)) + Gproc;
    rows(end+1, :) = {sprintf('%s, strongest part (%g dBsm): processed SNR', sc.tgt(k).name, sc.tgt(k).parts(q).rcs), ...
                      sprintf('%.1f dB at %.0f m, %.1f dB at %.0f m', s1, R1, s2, trN(k).R)}; %#ok<AGROW>
end
intf0 = cfg.intf;  intf0.t0 = 0;
[~, ii] = ncr_interference(cfg, intf0, 64, cfg.wf.B0, (0:63) * cfg.wf.Tc);
rows(end+1, :) = {'Interferer INR per sample during a burst', sprintf('%.1f dB (EIRP %g dBm at %g m)', ...
                  ii.INR_dB, cfg.intf.EIRP_dBm, cfg.intf.R)};
write_tab(fullfile(td, 'tab_link_budget.tex'), 'Radar parameters and link budget', ...
          'tab:link', {'Parameter', 'Value'}, rows, 'll');
txt = [txt; {'LINK BUDGET'}; fmt_rows(rows)];

% ---------------------------------------------------------------- main results
if isfield(X, 'E1')
    E1 = X.E1;  S = E1.sum;
    defs = {
      'P_D, all CPIs',                      'PDall',       '%.3f';
      'P_D, CPIs 16-64',                    'PDlate',      '%.3f';
      'P_D, vehicle in interference',       'PDvehIntf',   '%.3f';
      'False (moving) clusters per CPI',    'FA',          '%.2f';
      'OSPA, CPIs 16-64 (m)',               'OSPAlate',    '%.2f';
      'OSPA localisation (m)',              'OSPAloc',     '%.2f';
      'OSPA cardinality (m)',               'OSPAcard',    '%.2f';
      'False confirmed tracks per CPI',     'falseTracks', '%.2f';
      'Track position RMSE (m)',            'posRMSE',     '%.2f';
      'Range RMSE of detections (m)',       'rangeRMSE',   '%.3f';
      'Detections per vehicle',             'ptsVeh',      '%.1f';
      'Dwell on target sectors',            'dwellOnTgt',  '%.2f';
      'Cognitive reward R^{cog}',           'Rcog',        '%.2f'};
    rows = cell(size(defs, 1), 6);
    for i = 1:size(defs, 1)
        rows{i, 1} = defs{i, 1};
        for p = 1:3
            v = S{p}.(defs{i, 2});
            if isnan(v.m), rows{i, p+1} = '--';
            else, rows{i, p+1} = sprintf([defs{i, 3} ' $\\pm$ ' defs{i, 3}], v.m, v.h); end
        end
        rows{i, 5} = pstr(ncr_stats('pairedt', S{3}.run.(defs{i, 2}), S{1}.run.(defs{i, 2})));
        rows{i, 6} = pstr(ncr_stats('pairedt', S{3}.run.(defs{i, 2}), S{2}.run.(defs{i, 2})));
    end
    for k = 1:numel(E1.scene.tgt)
        r = {sprintf('P_D, %s', E1.scene.tgt(k).name)};
        for p = 1:3, r{p+1} = sprintf('%.3f $\\pm$ %.3f', S{p}.PDtgt.m(k), S{p}.PDtgt.h(k)); end
        r{5} = pstr(ncr_stats('pairedt', S{3}.run.PDtgt(:, k), S{1}.run.PDtgt(:, k)));
        r{6} = pstr(ncr_stats('pairedt', S{3}.run.PDtgt(:, k), S{2}.run.PDtgt(:, k)));
        rows(end+1, :) = r; %#ok<AGROW>
    end
    rows(end+1, :) = {'Attention correlation $\rho_F$', '--', '--', ...
        sprintf('%.2f $\\pm$ %.2f', S{3}.rho.m, S{3}.rho.h), '', ''};
    rows(end+1, :) = {'CPIs to 90\% of final $\rho_F$', '--', '--', sprintf('%d', S{3}.tConv), '', ''};
    write_tab(fullfile(td, 'tab_main_results.tex'), ...
        sprintf('Closed-loop results, mean $\\pm$ 95\\%% CI over %d runs; paired $t$-test p-values of neuromorphic vs uniform (U) and vs rule-based (R)', E1.Nmc), ...
        'tab:main', {'Metric', 'Uniform', 'Rule-based', 'Neuromorphic', 'p (U)', 'p (R)'}, rows, 'lccccc');
    txt = [txt; {''; 'MAIN RESULTS'}; fmt_rows(rows)];

    % network and energy
    N = S{3};
    rows = {
      'Input neurons (Doppler x range x angle)', sprintf('%d', cfg.d.Nin);
      'Hidden LIF neurons / output neurons', sprintf('%d / %d', cfg.d.Nhid, cfg.d.Nout);
      'Plastic weights (shared)', sprintf('%d', cfg.snn.K * cfg.d.NDc);
      'Time steps per CPI x step', sprintf('%d x %.1f ms', cfg.enc.T, cfg.enc.dtau * 1e3);
      'Input / hidden rate, CPIs 16-64 (Hz)', sprintf('%.2f / %.2f', N.rateIn.m, N.rateHid.m);
      'Hidden rate, CPIs 1-3 (Hz)', sprintf('%.2f', N.rateHid1.m);
      'Synaptic events per CPI', sprintf('%.0f', N.synOps.m);
      'Dense MACs per frame-based pass', sprintf('%d', cfg.d.denseMACs);
      'Energy per CPI: SNN / dense (uJ, indicative)', sprintf('%.3f / %.3f', N.Esnn_uJ.m, N.Edense_uJ);
      'Energy ratio dense / SNN', sprintf('%.1f', N.Edense_uJ / N.Esnn_uJ.m)};
    write_tab(fullfile(td, 'tab_network.tex'), ...
        'Network size, activity and indicative energy (Loihi synaptic-operation and 45 nm FP32 MAC energies)', ...
        'tab:network', {'Quantity', 'Value'}, rows, 'll');
    txt = [txt; {''; 'NETWORK AND ENERGY'}; fmt_rows(rows)];
end

% ---------------------------------------------------------------- ablation
if isfield(X, 'E2')
    E2 = X.E2;
    rows = cell(numel(E2.names), 5);
    for v = 1:numel(E2.names)
        s = E2.res{v}.sum{1};
        rows(v, :) = {E2.names{v}, sprintf('%.3f $\\pm$ %.3f', s.PDlate.m, s.PDlate.h), ...
            sprintf('%.2f $\\pm$ %.2f', s.OSPAlate.m, s.OSPAlate.h), ...
            sprintf('%.2f $\\pm$ %.2f', s.rho.m, s.rho.h), sprintf('%.0f', s.synOps.m)};
    end
    write_tab(fullfile(td, 'tab_ablation.tex'), ...
        sprintf('Ablation of the neuromorphic controller (%d runs each, CPIs 16-64)', E2.res{1}.Nmc), ...
        'tab:ablation', {'Variant', 'P_D', 'OSPA (m)', '$\rho_F$', 'SynOps/CPI'}, rows, 'lcccc');
    txt = [txt; {''; 'ABLATION'}; fmt_rows(rows)];
end

fid = fopen(fullfile(td, 'summary.txt'), 'w');
fprintf(fid, '%s\n', txt{:});
fclose(fid);
fprintf('%s\n', txt{:});
end

function s = pstr(p)
if isnan(p), s = '--';
elseif p < 1e-3, s = '$<$0.001';
else, s = sprintf('%.3f', p); end
end

function write_tab(file, caption, lab, head, rows, align)
fid = fopen(file, 'w');
fprintf(fid, '%% generated by ncr_tables.m (%s)\n', datestr(now));
fprintf(fid, '\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{%s}\n', caption, lab);
fprintf(fid, '\\begin{tabular}{%s}\n\\toprule\n', align);
head = cellfun(@tex_safe, head, 'UniformOutput', false);
fprintf(fid, '%s \\\\\n\\midrule\n', strjoin(head, ' & '));
for i = 1:size(rows, 1)
    r = cellfun(@tex_safe, rows(i, :), 'UniformOutput', false);
    fprintf(fid, '%s \\\\\n', strjoin(r, ' & '));
end
fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
fclose(fid);
end

function s = tex_safe(s)
if any(s == '$'), return; end          % already TeX
s = strrep(s, '%', '\%');
s = strrep(s, 'P_D', '$P_{\mathrm{D}}$');
s = strrep(s, 'R^{cog}', '$R^{\mathrm{cog}}$');
s = strrep(s, 'm^2', 'm$^2$');
end

function out = fmt_rows(rows)
out = cell(size(rows, 1), 1);
for i = 1:size(rows, 1)
    r = regexprep(rows(i, :), '\$\\pm\$', '+/-');
    r = regexprep(r, '[\$\\{}]', '');
    out{i} = sprintf('%-46s %s', r{1}, strjoin(r(2:end), ' | '));
end
end

function X = ncr_load_results(dirName)
%NCR_LOAD_RESULTS  Load saved experiment results (E1..E4) from a folder.
%   X = NCR_LOAD_RESULTS('results') returns a structure with the fields
%   E1, E2, E3, E4 for every result file present, ready for NCR_FIGURES and
%   NCR_TABLES.
if nargin < 1, dirName = 'results'; end
files = {'E1_main', 'E1'; 'E2_ablation', 'E2'; 'E3_interference', 'E3'; 'E4_latency', 'E4'};
X = struct();
for i = 1:size(files, 1)
    f = fullfile(dirName, [files{i, 1} '.mat']);
    if exist(f, 'file')
        L = load(f);
        X.(files{i, 2}) = L.S;
    end
end
end

function J = ncr_export_summary(resultsDir, jsonFile)
%NCR_EXPORT_SUMMARY  Export every number quoted in the paper to JSON.
%
%   J = NCR_EXPORT_SUMMARY('results') loads E1..E4 from the results folder
%   and writes <resultsDir>/summary.json. The paper's results_macros.tex is
%   generated from this file by make_macros.py, so the LaTeX source never
%   contains hand-copied numbers.
if nargin < 1, resultsDir = 'results'; end
if nargin < 2, jsonFile = fullfile(resultsDir, 'summary.json'); end
cfg = ncr_config('out.dir', resultsDir);
X = ncr_load_results(resultsDir);
if isfield(X.E1, 'cfg'), cfg = X.E1.cfg; end     % configuration the results were produced with
S = X.E1.sum;  N = S{3};
J = struct();
J.Nmc = X.E1.Nmc;  J.NmcAbl = X.E2.res{1}.Nmc;  J.NmcSweep = X.E3.res{1, 1}.Nmc;
J.late = cfg.exp.late;  J.Ns = cfg.sim.Nscans;
% learning dynamics
J.rho1 = N.rhoCurve.m(1);  J.Rcog1 = N.RcogCurve.m(1);  J.RcogAlg1 = S{2}.RcogCurve.m(1);
J.tConv = N.tConv;  J.tConvMs = N.tConv * cfg.sim.Tf * 1e3;
J.rhoLate = N.rho.m;  J.rhoLateH = N.rho.h;
J.RcogNeu = N.Rcog.m;  J.RcogNeuH = N.Rcog.h;  J.RcogAlg = S{2}.Rcog.m;  J.RcogAlgH = S{2}.Rcog.h;
tr = ncr_truth(cfg, X.E1.scene, cfg.sim.Nscans);  J.vrTargets = [tr.vr];
z = find(abs(cfg.d.vCoarse) < 1e-9);
J.k0first = N.kernelMean(z, 1) / cfg.snn.wMax;  J.k0last = N.kernelMean(z, end) / cfg.snn.wMax;
J.kInit = cfg.snn.w0 / cfg.snn.wMax;                 % expected initial weight
J.notchDrop = 100 * (1 - J.k0last / J.kInit);
vv = cfg.d.vCoarse;  kl = N.kernelMean(:, end) / cfg.snn.wMax;
[~, iped] = min(abs(vv - tr(2).vr));  [~, icyc] = min(abs(vv - tr(3).vr));
[~, iveh] = min(abs(vv - tr(1).vr));
J.kPed = kl(iped);  J.kCyc = kl(icyc);  J.kVeh = kl(iveh);
others = setdiff(1:cfg.d.NDc, z + (-1:1));
J.kOtherLast = mean(N.kernelMean(others, end)) / cfg.snn.wMax;
J.kMaxLast = max(N.kernelMean(:, end)) / cfg.snn.wMax;
[~, im] = max(N.kernelMean(:, end));  J.vMaxLast = cfg.d.vCoarse(im);

% activity
J.rateHid1 = N.rateHid1.m;  J.rateHidLate = N.rateHid.m;  J.rateIn = N.rateIn.m;
J.hidDrop = 100 * (1 - N.rateHid.m / N.rateHid1.m);
J.synOps = N.synOps.m;  J.synOpsH = N.synOps.h;  J.denseMACs = cfg.d.denseMACs;
J.opsRatio = cfg.d.denseMACs / N.synOps.m;
J.Esnn = N.Esnn_uJ.m;  J.Edense = N.Edense_uJ;  J.Eratio = N.Edense_uJ / N.Esnn_uJ.m;
J.inSparsity = N.rateIn.m * cfg.enc.dtau;     % spikes per input neuron per step
J.eventsPerCPI = N.rateIn.m * cfg.d.Nin * cfg.enc.T * cfg.enc.dtau;
J.Nin = cfg.d.Nin;  J.Nhid = cfg.d.Nhid;  J.Nweights = cfg.snn.K * cfg.d.NDc;
J.M = cfg.ar.M;  J.nUni = cfg.d.nUni;  J.Nc = cfg.bud.Nc;
% main metrics: mean, CI half-width and paired p-values (neuromorphic vs U, vs A)
f = {'PDall','PDlate','PDvehIntf','FA','OSPAlate','OSPAloc','OSPAcard','falseTracks', ...
     'posRMSE','rangeRMSE','ptsVeh','dwellOnTgt'};
for i = 1:numel(f)
    J.(f{i}) = cellfun(@(s) s.(f{i}).m, S);
    J.([f{i} '_h']) = cellfun(@(s) s.(f{i}).h, S);
    J.([f{i} '_pU']) = ncr_stats('pairedt', N.run.(f{i}), S{1}.run.(f{i}));
    J.([f{i} '_pA']) = ncr_stats('pairedt', N.run.(f{i}), S{2}.run.(f{i}));
    J.([f{i} '_pAU']) = ncr_stats('pairedt', S{2}.run.(f{i}), S{1}.run.(f{i}));
end
J.PDtgt = cell2mat(cellfun(@(s) s.PDtgt.m(:)', S(:), 'UniformOutput', false));
J.PDtgt_h = cell2mat(cellfun(@(s) s.PDtgt.h(:)', S(:), 'UniformOutput', false));
for k = 1:3
    J.(sprintf('PDtgt%d_pU', k)) = ncr_stats('pairedt', N.run.PDtgt(:, k), S{1}.run.PDtgt(:, k));
    J.(sprintf('PDtgt%d_pA', k)) = ncr_stats('pairedt', N.run.PDtgt(:, k), S{2}.run.PDtgt(:, k));
end
J.cover = cell2mat(cellfun(@(s) s.cover.m(:)', S(:), 'UniformOutput', false));
% early window (CPIs 1-8) detection, to quantify the learning transient
for p = 1:3
    J.PDearly(p) = mean(S{p}.pdCurve.m(1:8), 'omitnan');
end
% dwell maps of the example run (run 1): mean chirps on the vehicle sector in the interference window
L = X.E1.logs;  iI = cfg.intf.on:cfg.intf.off;
for p = 1:3
    nv = zeros(numel(iI), 1);
    for q = 1:numel(iI)
        t = iI(q);  trq = ncr_truth(cfg, X.E1.scene, t);  nv(q) = L{1, p}.n(t, trq(1).sector);
    end
    J.nVehIntf(p) = mean(nv);
    J.bVehIntf(p) = mean(arrayfun(@(t) L{1, p}.b(t, getfield(ncr_truth(cfg, X.E1.scene, t), {1}, 'sector')), iI)) / 1e6;
end
% ablation
E2 = X.E2;  J.ablNames = E2.names;
g = @(fld) cellfun(@(R) R.sum{1}.(fld).m, E2.res);
gh = @(fld) cellfun(@(R) R.sum{1}.(fld).h, E2.res);
J.ablPD = g('PDlate');  J.ablPDh = gh('PDlate');  J.ablOSPA = g('OSPAlate');  J.ablOSPAh = gh('OSPAlate');
J.ablRho = g('rho');  J.ablSyn = g('synOps');  J.ablE = g('Esnn_uJ');  J.ablFA = g('FA');
J.ablPDint = g('PDvehIntf');  J.ablFalseTr = g('falseTracks');  J.ablRateHid = g('rateHid');
J.ablConv = cellfun(@(R) R.sum{1}.tConv, E2.res);
J.ablPfull = cellfun(@(R) ncr_stats('pairedt', R.sum{1}.run.PDlate, E2.res{1}.sum{1}.run.PDlate), E2.res);
% interference sweep
E3 = X.E3;  e = E3.eirp;  e(~isfinite(e)) = -999;  J.eirp = e;
J.intfPD = cellfun(@(s) s.PDvehIntf.m, E3.res);  J.intfPDh = cellfun(@(s) s.PDvehIntf.h, E3.res);
J.intfOSPA = cellfun(@(s) s.OSPAlate.m, E3.res);  J.intfPDlate = cellfun(@(s) s.PDlate.m, E3.res);
J.intfFA = cellfun(@(s) s.FA.m, E3.res);
% latency sweep
E4 = X.E4;  J.Tsweep = E4.T;
J.latPD = cellfun(@(s) s.PDlate.m, E4.res);  J.latPDh = cellfun(@(s) s.PDlate.h, E4.res);
J.latE = cellfun(@(s) s.Esnn_uJ.m, E4.res);  J.latSyn = cellfun(@(s) s.synOps.m, E4.res);
J.latRho = cellfun(@(s) s.rho.m, E4.res);  J.latOSPA = cellfun(@(s) s.OSPAlate.m, E4.res);
J.latConv = cellfun(@(s) s.tConv, E4.res);
% link budget
sc = ncr_scene(cfg);
[~, info] = ncr_link_budget(cfg, 10, 1, 1);
J.EIRP = info.EIRP_dBm;  J.noise_dBm = info.noise_dBm;  J.snr10 = info.snr_dB;
nj = cfg.d.nUni;  gD = sum(ncr_hann(nj))^2 / sum(ncr_hann(nj).^2);
J.Gproc = 10 * log10(cfg.d.cgR * gD * cfg.d.cgX);
for k = 1:3
    [~, q] = max([sc.tgt(k).parts.rcs]);
    tk = ncr_truth(cfg, sc, sc.tgt(k).on);  RR = [tk(k).R, tr(k).R];
    J.tgtR(k, :) = RR;
    J.tgtSNR(k, :) = 10 * log10(ncr_link_budget(cfg, RR, 10^(sc.tgt(k).parts(q).rcs / 10))) + J.Gproc;
    J.tgtRCS(k) = 10 * log10(sum(10.^([sc.tgt(k).parts.rcs] / 10)));
end
i0 = cfg.intf;  i0.t0 = 0;
[~, ii] = ncr_interference(cfg, i0, 64, cfg.wf.B0, (0:63) * cfg.wf.Tc);
J.INR = ii.INR_dB;  J.intfDuty = ii.dutyFraction;
J.burstLen = cfg.wf.Fs^2 / abs(cfg.wf.B0 / cfg.d.Tramp - cfg.intf.B / cfg.intf.Tramp);
% noise-floor rise of the beams facing the interferer (sector 2, 32 chirps), mean of 8 timings
c0 = cfg;  c0.rf.adcBits = Inf;  rng(99);
Bv = [150e6 300e6 600e6];  J.floorRise = zeros(numel(Bv), 1);  J.floorRiseMax = zeros(numel(Bv), 1);
for q = 1:numel(Bv)
    acc = [];
    for it = 1:8
        i1 = cfg.intf;  i1.t0 = rand * i1.Tc;  tm = (0:31) * cfg.wf.Tc;
        X0 = (randn(cfg.wf.Nr, cfg.rf.Nrx, 32) + 1j * randn(cfg.wf.Nr, cfg.rf.Nrx, 32)) / sqrt(2);
        XI = ncr_interference(c0, i1, 32, Bv(q), tm);
        a0 = ncr_process(c0, X0, 2, 32, Bv(q));  a1 = ncr_process(c0, X0 + XI, 2, 32, Bv(q));
        acc = [acc; 10 * log10(a1.floor ./ a0.floor)];
    end
    J.floorRise(q) = mean(acc(:));  J.floorRiseMax(q) = max(mean(acc, 1));
end
J.floorBv = Bv / 1e6;
fid = fopen(jsonFile, 'w');  fprintf(fid, '%s', jsonencode(J));  fclose(fid);
fprintf('Summary written to %s\n', jsonFile);
end

function ncr_export_figure_data(cfg, X, file)
%NCR_EXPORT_FIGURE_DATA  Flat arrays behind every paper figure, for external typesetting.
%
%   NCR_EXPORT_FIGURE_DATA(cfg, X, file) writes a MAT file (v7) containing
%   plain numeric arrays and cell strings only, so that it can be read by
%   any tool (e.g. scipy.io.loadmat in the paper's make_figures.py).
if nargin < 3, file = fullfile(cfg.out.dir, 'figure_data.mat'); end
D = struct();
E1 = X.E1;  sc = E1.scene;  c = E1.cfg;  Ns = c.sim.Nscans;  nT = numel(sc.tgt);
D.t = 1:Ns;  D.late = c.exp.late;
D.intf_on = sc.intf.on;  D.intf_off = sc.intf.off;  D.cyc_on = sc.tgt(3).on;
D.policies = {'Uniform', 'Rule-based', 'Neuromorphic'};
for p = 1:3
    S = E1.sum{p};
    D.pd_m(p, :) = S.pdCurve.m(:)';     D.pd_h(p, :) = S.pdCurve.h(:)';
    D.ospa_m(p, :) = S.ospaCurve.m(:)'; D.ospa_h(p, :) = S.ospaCurve.h(:)';
    D.rcog_m(p, :) = S.RcogCurve.m(:)'; D.rcog_h(p, :) = S.RcogCurve.h(:)';
    D.dwell_m(p, :) = S.dwellCurve.m(:)';
end
N = E1.sum{3};
D.rho_m = N.rhoCurve.m(:)';  D.rho_h = N.rhoCurve.h(:)';
D.ratehid_m = N.rateHidCurve.m(:)';  D.ratein_m = N.rateInCurve.m(:)';

% learned kernel of every run (Doppler channel x CPI x run), normalised by wMax
Nmc = size(E1.logs, 1);
K = zeros(c.d.NDc, Ns, Nmc);
for r = 1:Nmc, K(:, :, r) = E1.logs{r, 3}.kernel / c.snn.wMax; end
D.kernel = K;  D.v = c.d.vCoarse;  D.w_init = c.snn.w0 / c.snn.wMax;
D.vres_uniform = c.d.lambda / (2 * c.d.nUni * c.wf.Tc);  D.sigv = c.ctl.sigV;

% ground truth of the targets per CPI
D.tgt_names = {sc.tgt.name};
D.tgt_vr = nan(nT, Ns);  D.tgt_x = nan(nT, Ns);  D.tgt_y = nan(nT, Ns);
for t = 1:Ns
    tr = ncr_truth(c, sc, t);
    for k = 1:nT
        if tr(k).active
            D.tgt_vr(k, t) = tr(k).vr;  D.tgt_x(k, t) = tr(k).pos(1);  D.tgt_y(k, t) = tr(k).pos(2);
        end
    end
end

% attention fields of run 1 (range x angle) at the snapshot CPIs
L = E1.logs{1, 3};
D.snap_cpi = L.snap.cpi;
for k = 1:numel(L.snap.cpi)
    D.(sprintf('Falg_%d', L.snap.cpi(k))) = L.snap.Falg{k};
    D.(sprintf('Fneu_%d', L.snap.cpi(k))) = L.snap.Fneu{k};
end
D.r_edges = (0:c.d.NRc) * c.grid.dRc;
D.th_edges = c.ar.secEdges(1) + (0:c.d.NAc) * c.grid.dAc;

% scene geometry
D.sec_edges = c.ar.secEdges;  D.Rmax = c.proc.Rproc;
D.stat_p = sc.stat.p;  D.veg_p = sc.veg.p;  D.intf_pos = sc.intf.pos;
D.intf_theta = sc.intf.theta;

% interference sweep (rows: EIRP, cols: uniform, rule-based, neuromorphic, uniform + excision)
if isfield(X, 'E3')
    E3 = X.E3;  e = E3.eirp;  e(~isfinite(e)) = NaN;  D.eirp = e;
    g = @(f, w) cellfun(@(S) S.(f).(w), E3.res);
    D.sw_pdint_m = g('PDvehIntf', 'm');  D.sw_pdint_h = g('PDvehIntf', 'h');
    D.sw_fa_m = g('FA', 'm');  D.sw_fa_h = g('FA', 'h');
    D.sw_ospa_m = g('OSPAlate', 'm');  D.sw_ospa_h = g('OSPAlate', 'h');
    D.sw_pdlate_m = g('PDlate', 'm');  D.sw_pdlate_h = g('PDlate', 'h');
end
% time-step sweep
if isfield(X, 'E4')
    E4 = X.E4;  D.T = E4.T;
    g = @(f, w) cellfun(@(S) S.(f).(w), E4.res);
    D.lat_pd_m = g('PDlate', 'm');  D.lat_pd_h = g('PDlate', 'h');
    D.lat_syn_m = g('synOps', 'm');  D.lat_syn_h = g('synOps', 'h');
    D.lat_e_m = g('Esnn_uJ', 'm');  D.lat_conv = cellfun(@(S) scalar_or_nan(S.tConv), E4.res);
end
% ablation
if isfield(X, 'E2')
    E2 = X.E2;  D.abl_names = E2.names;
    g = @(f, w) cellfun(@(R) R.sum{1}.(f).(w), E2.res);
    D.abl_pd_m = g('PDlate', 'm');  D.abl_pd_h = g('PDlate', 'h');
    D.abl_ospa_m = g('OSPAlate', 'm');  D.abl_rho_m = g('rho', 'm');
    D.abl_syn_m = g('synOps', 'm');  D.abl_e_m = g('Esnn_uJ', 'm');
end
if exist('OCTAVE_VERSION', 'builtin')
    save('-v7', file, '-struct', 'D');
else
    save(file, '-struct', 'D', '-v7');
end
fprintf('Figure data written to %s\n', file);
end

function y = scalar_or_nan(x)
% first element of x, or NaN when x is empty (e.g. no convergence within the run)
if isempty(x), y = NaN; else, y = double(x(1)); end
end

%% ============================================================================
%%  VERIFICATION AND UTILITIES
%% ============================================================================

function T = ncr_selftest(cfg)
%NCR_SELFTEST  Verification of the radar model, processing and utilities.
%
%   T = NCR_SELFTEST() runs all checks with the default configuration and
%   prints a pass/fail table. Checks:
%     1 point target appears at the correct range, Doppler and angle
%     2 processed SNR matches the link budget (incl. window losses)
%     3 CA-CFAR false-alarm rate in noise matches the design value
%     4 interference burst length matches Fs^2 / |S_v - S_i|
%     5 Hungarian assignment equals brute force on random problems
%     6 OSPA metric properties (identity, cardinality penalty)
%     7 resource allocation respects the dwell and bandwidth budgets
%     8 tracker confirms a constant-velocity target and estimates its velocity
if nargin < 1, cfg = ncr_config(); end
rng(7);
T = struct('name', {}, 'pass', {}, 'detail', {});

% ---------------------------------------------------------------- 1, 2 point target
j = 6;  nj = 64;  bj = 400e6;
R0 = 23.3;  th0 = cfg.d.beams(j, ceil((cfg.ar.Na + 1) / 2));  v0 = -3.1;  rcs = 1;   % beam nearest the Tx boresight
sc = point_scatterer(R0, th0, v0, rcs);
c2 = cfg;  c2.rf.adcBits = Inf;
tm = (0:nj-1) * cfg.wf.Tc;
X = ncr_frontend(c2, sc, struct(), j, nj, bj, tm, false);
cube = ncr_process(c2, X, j, nj, bj);
[~, i] = max(cube.P(:));
[ir, id, ia] = ind2sub(size(cube.P), i);
errR = abs(cube.r(ir) - R0);  errV = abs(cube.v(id) - v0);  errA = abs(cube.th(ia) - th0);
ok1 = errR <= cube.dR && errV <= 2 * cfg.d.dv && errA < 1e-9;
T(end+1) = rec('Point target bin placement', ok1, ...
    sprintf('|dR| = %.2f m (bin %.2f), |dv| = %.2f m/s, |dth| = %.1f deg', errR, cube.dR, errV, errA));

% expected processed SNR: link budget x coherent gains of the three windows
snr1 = ncr_link_budget(cfg, R0, rcs, 1);
gR = cfg.d.cgR;  gD = sum(ncr_hann(nj))^2 / sum(ncr_hann(nj).^2);  gX = cfg.d.cgX;
expect_dB = 10 * log10(snr1 * gR * gD * gX);
noise = (randn(size(X)) + 1j * randn(size(X))) / sqrt(2);
cn = ncr_process(c2, noise, j, nj, bj);
meas_dB = 10 * log10(max(cube.P(:)) / mean(cn.P(:)));
ok2 = abs(meas_dB - expect_dB) < 2.0;        % scalloping and straddle losses < 2 dB
T(end+1) = rec('Processed SNR vs link budget', ok2, ...
    sprintf('expected %.1f dB, measured %.1f dB', expect_dB, meas_dB));

% ---------------------------------------------------------------- 3 CFAR false alarms
nTrial = 6;  nFA = 0;  nCells = 0;
for k = 1:nTrial
    Xn = (randn(cfg.wf.Nr, cfg.rf.Nrx, 32) + 1j * randn(cfg.wf.Nr, cfg.rf.Nrx, 32)) / sqrt(2);
    cn = ncr_process(c2, Xn, 4, 32, 300e6);
    c3 = cfg;  c3.det.Pfa = 1e-3;
    hit = ncr_cfar(c3, cn.P, 32);
    nFA = nFA + nnz(hit);  nCells = nCells + numel(hit);
end
pfa = nFA / nCells;
ok3 = pfa > 1e-3 / 4 && pfa < 1e-3 * 4;      % correlated (zero-padded) cells: factor-4 tolerance
T(end+1) = rec('CA-CFAR false-alarm rate (design 1e-3)', ok3, sprintf('empirical %.2e', pfa));

% ---------------------------------------------------------------- 4 interference bursts
intf = cfg.intf;  intf.t0 = 0;
bj = 300e6;  nj = 200;  tm = (0:nj-1) * cfg.wf.Tc;
[XI, ii] = ncr_interference(cfg, intf, nj, bj, tm);
on = squeeze(abs(XI(:, 1, :)) > 0);                     % Nr x nj
lens = [];
for m = 1:nj
    dl = diff([0; on(:, m); 0]);  s = find(dl == 1);  e = find(dl == -1);
    full = s > 1 & e <= cfg.wf.Nr;                       % bursts fully inside the ramp
    lens = [lens; (e(full) - s(full))]; %#ok<AGROW>
end
Sv = bj / cfg.d.Tramp;  SI = intf.B / intf.Tramp;
expectLen = cfg.wf.Fs / abs(Sv - SI) * cfg.wf.Fs;      % samples
ok4 = ~isempty(lens) && abs(median(lens) - expectLen) <= 2;
T(end+1) = rec('Interference burst length', ok4, ...
    sprintf('median %.1f samples, theory %.1f; INR %.1f dB, duty %.2f', ...
            median(lens), expectLen, ii.INR_dB, ii.dutyFraction));

% ---------------------------------------------------------------- 5 Hungarian
okH = true;
for k = 1:200
    n = randi(5);  m = randi(5);
    C = rand(n, m);  C(rand(n, m) < 0.15) = Inf;
    [a, tot] = ncr_hungarian(C);
    best = brute(C);
    if abs(tot - best) > 1e-9, okH = false; end
    if any(a > 0 & a > m) || numel(unique(a(a > 0))) ~= nnz(a), okH = false; end
end
T(end+1) = rec('Hungarian = brute force (200 random cases)', okH, '');

% ---------------------------------------------------------------- 6 OSPA
X1 = [0 0; 10 0];
o1 = ncr_ospa(X1, X1, 5, 2);
o2 = ncr_ospa(X1, X1(1, :), 5, 2);
o3 = ncr_ospa(zeros(0, 2), X1, 5, 2);
ok6 = o1 < 1e-12 && abs(o2 - 5 / sqrt(2)) < 1e-12 && o3 == 5;
T(end+1) = rec('OSPA identity / cardinality / empty set', ok6, ...
    sprintf('%.3f, %.3f, %.3f', o1, o2, o3));

% ---------------------------------------------------------------- 7 allocation
ok7 = true;
for k = 1:200
    ut = rand(1, cfg.ar.M).^4;
    [n, b] = ncr_attention('allocate', cfg, ut);
    ok7 = ok7 && sum(n) == cfg.bud.Nc && all(n >= cfg.bud.nMin) && all(n <= cfg.bud.nMax) && ...
          all(n == round(n)) && mean(b) <= cfg.wf.B0 + 1 && all(b >= cfg.wf.Bmin - 1) && ...
          all(b <= cfg.wf.Bmax + 1);
end
T(end+1) = rec('Dwell/bandwidth budgets respected', ok7, '');

% ---------------------------------------------------------------- 8 tracker
trk = ncr_tracker(cfg);
p = [5 30];  v = [1 -4];
for t = 1:15
    pt = p + v * (t - 1) * cfg.sim.Tf;
    R = norm(pt) + 0.2 * randn;  th = atan2d(pt(1), pt(2)) + 1 * randn;
    vr = dot(pt, v) / norm(pt) + 0.3 * randn;
    trk = ncr_tracker(cfg, trk, [R th vr 1 1 20 1], t);
end
ok8 = numel(trk.tracks) == 1 && trk.tracks(1).conf && norm(trk.tracks(1).x(3:4)' - v) < 1.5;
T(end+1) = rec('Tracker confirms and estimates a CV target', ok8, ...
    sprintf('velocity estimate [%.2f %.2f] (true [%g %g])', trk.tracks(1).x(3), trk.tracks(1).x(4), v));

% ---------------------------------------------------------------- report
fprintf('\n%-48s %-6s %s\n', 'Self-test', 'Result', 'Detail');
for k = 1:numel(T)
    if T(k).pass, s = 'PASS'; else, s = 'FAIL'; end
    fprintf('%-48s %-6s %s\n', T(k).name, s, T(k).detail);
end
end

function r = rec(name, pass, detail)
r = struct('name', name, 'pass', logical(pass), 'detail', detail);
end

function sc = point_scatterer(R, th, vr, rcs)
p = R * [sind(th) cosd(th)];
sc = struct('p', p, 'v', vr * p / R, 'swDir', [0 0], 'swAmp', 0, 'swHz', 0, ...
            'swPh', 0, 'rcs', rcs, 'phi', 0, 'tgt', 1, 't0', 0, 'theta', th);
end

function best = brute(C)
% exhaustive search; the Hungarian maximises the number of finite
% assignments first, then minimises their total cost
[n, m] = size(C);
best = 0;
if n == 0 || m == 0, return; end
best = -Inf;  bestCnt = -1;
opts = 0:m;
idx = ones(1, n);
total = (m + 1)^n;
for it = 1:total
    a = opts(idx);
    cols = a(a > 0);
    if numel(unique(cols)) == numel(cols)
        cost = 0;  feasible = true;
        for i = 1:n
            if a(i) > 0
                if ~isfinite(C(i, a(i))), feasible = false; break; end
                cost = cost + C(i, a(i));
            end
        end
        if feasible
            cnt = numel(cols);
            if cnt > bestCnt || (cnt == bestCnt && cost < best)
                best = cost;  bestCnt = cnt;
            end
        end
    end
    k = 1;
    while k <= n
        idx(k) = idx(k) + 1;
        if idx(k) <= m + 1, break; end
        idx(k) = 1;  k = k + 1;
    end
end
end

function w = ncr_hann(N)
%NCR_HANN  Symmetric Hann window (column), toolbox-free.
if N <= 1
    w = ones(max(N, 1), 1);
    return
end
w = 0.5 * (1 - cos(2 * pi * (0:N-1)' / (N - 1)));
end
