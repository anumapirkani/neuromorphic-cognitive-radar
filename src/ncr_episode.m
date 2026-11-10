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
