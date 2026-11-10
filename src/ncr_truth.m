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
