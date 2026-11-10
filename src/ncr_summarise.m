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
