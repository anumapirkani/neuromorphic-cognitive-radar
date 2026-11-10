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
