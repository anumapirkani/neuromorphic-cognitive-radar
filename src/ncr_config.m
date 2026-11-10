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
