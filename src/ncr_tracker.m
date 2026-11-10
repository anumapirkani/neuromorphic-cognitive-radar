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
