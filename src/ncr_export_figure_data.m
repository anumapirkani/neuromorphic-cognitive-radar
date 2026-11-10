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
