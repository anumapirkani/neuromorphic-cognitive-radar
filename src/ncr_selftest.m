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
