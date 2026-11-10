function cube = ncr_process(cfg, X, j, nj, bj)
%NCR_PROCESS  Range FFT, Doppler FFT and receive beamforming for sector j.
%
%   cube = NCR_PROCESS(cfg, X, j, nj, bj) returns
%       P      power cube, range x Doppler x beam (Nk x Nd x Na)
%       Z      P normalised by a robust per-beam noise-floor estimate
%       r      range axis (m), bins with 0 < r <= Rproc
%       v      radial-velocity axis (m/s), positive = receding
%       th     beam angles (deg)
%       floor  noise floor per beam (same units as P)
%       dR, dv range bin and physical Doppler resolution
%   Hann windows are applied in all three dimensions.
Nr = cfg.wf.Nr;  Nrx = cfg.rf.Nrx;  Nd = cfg.proc.Nd;  Na = cfg.ar.Na;

Xr = fft(X .* ncr_hann(Nr), Nr, 1);
r  = (0:Nr-1)' * cfg.const.c / (2 * bj);
kk = find(r > 0 & r <= cfg.proc.Rproc);
Xr = Xr(kk, :, :);
Nk = numel(kk);

Xd = fftshift(fft(Xr .* reshape(ncr_hann(nj), 1, 1, nj), Nd, 3), 3);

ax  = (0:Nrx-1)';
th  = cfg.d.beams(j, :);
Wbf = ncr_hann(Nrx) .* exp(-1j * pi * ax * sind(th));               % Nrx x Na
Y   = Wbf.' * reshape(permute(Xd, [2 1 3]), Nrx, Nk * Nd);
P   = abs(permute(reshape(Y, Na, Nk, Nd), [2 3 1])).^2;             % Nk x Nd x Na

fl = median(reshape(P, Nk * Nd, Na), 1) / log(2);                   % exponential median -> mean
cube.P = P;
cube.Z = P ./ reshape(fl, 1, 1, Na);
cube.r = r(kk);
cube.v = cfg.d.vFine;
cube.th = th;
cube.floor = fl;
cube.dR = cfg.const.c / (2 * bj);
cube.dvRes = cfg.d.lambda / (2 * nj * cfg.wf.Tc);
cube.nj = nj;  cube.bj = bj;  cube.sector = j;
end
