function Zc = ncr_coarse_map(cfg, cube, Zc)
%NCR_COARSE_MAP  Max-pool one sector's normalised cube onto the perception grid.
%
%   Zc (NRc x NDc x NAc) is updated in the angle cells of sector cube.sector.
%   Pooling: Doppler stride 2 with a 3-bin window centred on 0 m/s, pairs of
%   receive beams per coarse angle cell, and all range bins inside each
%   coarse range cell. The perception grid is identical for every sector,
%   whatever its bandwidth and dwell.
d = cfg.d;  Z = cube.Z;  Nd = size(Z, 2);
ic = 1:2:Nd;  im = mod(ic - 2, Nd) + 1;  ip = mod(ic, Nd) + 1;
Zp = max(max(Z(:, ic, :), Z(:, im, :)), Z(:, ip, :));
bpc = cfg.ar.Na / d.cellsPerSector;                      % beams per coarse cell
Zq = zeros(size(Zp, 1), size(Zp, 2), d.cellsPerSector);
for a = 1:d.cellsPerSector
    Zq(:, :, a) = max(Zp(:, :, (a - 1) * bpc + (1:bpc)), [], 3);
end
rc = min(d.NRc, max(1, ceil(cube.r / cfg.grid.dRc)));
cols = (cube.sector - 1) * d.cellsPerSector + (1:d.cellsPerSector);
for r = 1:d.NRc
    rows = rc == r;
    if any(rows)
        Zc(r, :, cols) = max(Zq(rows, :, :), [], 1);
    end
end
end
