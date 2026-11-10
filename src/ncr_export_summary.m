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
