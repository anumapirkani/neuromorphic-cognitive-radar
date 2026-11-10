function scene = ncr_scene(cfg)
%NCR_SCENE  Urban scenario seen by a stationary 77 GHz radar at the origin.
%
%   Coordinates: x lateral (positive right), y boresight. Angles are measured
%   from boresight, positive towards +x. Moving targets are extended: each is
%   a set of scattering centres defined in the body frame (along = heading,
%   across = left-hand normal). Limbs and pedals oscillate along the heading
%   (micro-Doppler).
%
%   Targets
%     Vehicle     10 dBsm, 6 scattering centres, passes a parked car and the
%                 direction of an interfering radar (sector -45..-30 deg)
%     Pedestrian  -5.4 dBsm, torso + 2 legs + 2 arms (gait 0.9 Hz), weak
%     Cyclist     -1.3 dBsm, rider + frame + 2 legs (cadence 1.3 Hz),
%                 appears at CPI 30 (novelty)
%   Clutter
%     two parked cars (4 points each), a pole, and a vegetation strip with
%     K-distributed amplitudes and internal motion (33..58 deg).

% ---------------------------------------------------------------- moving targets
T = struct('name', {}, 'p0', {}, 'v', {}, 'on', {}, 'off', {}, 'parts', {});

veh.name = 'Vehicle';  veh.p0 = [-28 38];  veh.v = [3.5 -6.5];
veh.on = 1;  veh.off = cfg.sim.Nscans;
veh.parts = parts_table([ ...
%   along  across  rcs_dBsm swingAmp swingHz phase
     2.25   0.9    4        0        0       0
     2.25  -0.9    4        0        0       0
     1.40   0.9    0        0        0       0
     1.40  -0.9    0        0        0       0
    -2.25   0.9    2        0        0       0
    -2.25  -0.9    2        0        0       0]);
T(1) = veh;

ped.name = 'Pedestrian';  ped.p0 = [13 34];  ped.v = [-0.5 -1.4];
ped.on = 1;  ped.off = cfg.sim.Nscans;
ped.parts = parts_table([ ...
     0      0      -8       0       0       0       % torso
     0      0.15   -14      0.35    0.9     0       % legs (antiphase)
     0     -0.15   -14      0.35    0.9     pi
     0      0.25   -16      0.20    0.9     pi      % arms (opposite to legs)
     0     -0.25   -16      0.20    0.9     0]);
T(2) = ped;

cyc.name = 'Cyclist';  cyc.p0 = [-6 12];  cyc.v = [-1.2 4.6];
cyc.on = 30;  cyc.off = cfg.sim.Nscans;
cyc.parts = parts_table([ ...
     0      0      -4       0       0       0       % rider
     0.5    0      -6       0       0       0       % frame and wheels
     0.1    0.15   -13      0.15    1.3     0       % pedalling legs
     0.1   -0.15   -13      0.15    1.3     pi]);
T(3) = cyc;
scene.tgt = T;

% ---------------------------------------------------------------- stationary clutter
cars = [-20 16; -24 27];                       % parked cars, aligned with y
pts = [];
for c = 1:size(cars, 1)
    pts = [pts; cars(c, :) + [ 0.9  2.2]; cars(c, :) + [-0.9  2.2]; ...
                cars(c, :) + [ 0.9 -2.2]; cars(c, :) + [-0.9 -2.2]]; %#ok<AGROW>
end
pts = [pts; 4 42];                             % pole
scene.stat.p   = pts;
scene.stat.rcs = [4 * ones(8, 1); 5];          % dBsm (cars ~10 dBsm each)

% vegetation strip (fixed geometry; amplitudes evolve per CPI)
nVeg = 120;
s = rng;  rng(12345);                          % geometry independent of the run seed
th = (33 + 25 * rand(nVeg, 1)) * pi / 180;
R  = 8 + 40 * rand(nVeg, 1);
rng(s);
scene.veg.p     = [R .* sin(th), R .* cos(th)];
scene.veg.rcs   = -14 * ones(nVeg, 1);         % mean RCS per scatterer, dBsm
scene.veg.nu    = 1.5;                         % K-distribution texture shape
scene.veg.rho   = 0.9;                         % CPI-to-CPI texture correlation
scene.veg.sigv  = 0.1;                         % internal-motion velocity std, m/s

% ---------------------------------------------------------------- interferer
scene.intf = cfg.intf;
scene.intf.pos = cfg.intf.R * [sind(cfg.intf.theta), cosd(cfg.intf.theta)];
end

function P = parts_table(A)
P = struct('along', num2cell(A(:, 1)), 'across', num2cell(A(:, 2)), ...
           'rcs', num2cell(A(:, 3)), 'swAmp', num2cell(A(:, 4)), ...
           'swHz', num2cell(A(:, 5)), 'swPh', num2cell(A(:, 6)));
end
