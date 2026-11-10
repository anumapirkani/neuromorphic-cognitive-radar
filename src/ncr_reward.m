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
