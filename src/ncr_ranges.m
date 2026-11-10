function R = ncr_ranges(sc, idx, tm)
%NCR_RANGES  Ranges of selected scattering centres at the chirp times tm.
%
%   R = NCR_RANGES(sc, idx, tm) returns a numel(idx) x numel(tm) matrix of
%   ranges for the scatterers sc(idx) at absolute times tm (row vector),
%   including bulk motion and sinusoidal micro-motion. The resulting
%   slow-time phase history exp(j 4 pi R / lambda) carries Doppler,
%   micro-Doppler and range-rate exactly.
tm = tm(:)';
dt = tm - sc.t0;                                           % time since CPI start
sw = sc.swAmp(idx) .* sin(2 * pi * sc.swHz(idx) .* tm + sc.swPh(idx));   % K x n
x = sc.p(idx, 1) + sc.v(idx, 1) .* dt + sc.swDir(idx, 1) .* sw;
y = sc.p(idx, 2) + sc.v(idx, 2) .* dt + sc.swDir(idx, 2) .* sw;
R = sqrt(x.^2 + y.^2);
end
