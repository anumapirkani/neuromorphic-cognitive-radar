function [X, info] = ncr_frontend(cfg, sc, intf, j, nj, bj, tm, intfOn)
%NCR_FRONTEND  ADC data cube of one sector sub-frame.
%
%   X = NCR_FRONTEND(cfg, sc, intf, j, nj, bj, tm, intfOn) synthesises the
%   dechirped, sampled FMCW signal (Nr x Nrx x nj, noise variance 1 per
%   sample) received while the phased transmit beam points at sector j.
%
%   Signal model for scatterer s, fast-time sample n, channel k, chirp m:
%       x = a_s exp(j 2 pi f_b,s n / Fs) exp(j 4 pi R_s(t_m) / lambda)
%             exp(j pi k sin(theta_s)) + noise + interference
%   with beat frequency f_b = 2 S R / c (S = bj / Tramp) and the exact
%   slow-time range history R_s(t_m) (bulk motion and micro-motion). Range
%   migration inside a chirp is neglected (< 1 mm). The amplitude a_s
%   follows the link budget, the instantaneous RCS and the transmit array
%   factor towards theta_s. Thermal noise is complex white Gaussian; the
%   ADC quantises I and Q to cfg.rf.adcBits bits.
c = cfg.const.c;  lam = cfg.d.lambda;  Nr = cfg.wf.Nr;  Nrx = cfg.rf.Nrx;
u = sind(sc.theta);

% ---- scatterers that can reach this sector's beams
uLo = sind(cfg.ar.secEdges(j)) - cfg.ar.uMargin;
uHi = sind(cfg.ar.secEdges(j+1)) + cfg.ar.uMargin;
sel = find(u >= uLo & u <= uHi);

% ---- transmit array factor (normalised to 1 at the beam peak)
thC = cfg.d.secCentre(j);
switch lower(cfg.ar.txTaper)
    case 'hann', wt = ncr_hann(cfg.rf.Ntx);
    otherwise,   wt = ones(cfg.rf.Ntx, 1);
end
af = abs(wt.' * exp(1j * pi * (0:cfg.rf.Ntx-1)' * (u(sel)' - sind(thC)))).^2 / sum(wt)^2;

% ---- deterministic part (Khatri-Rao factorisation keeps this cheap)
Rm = ncr_ranges(sc, sel, tm);                  % K x nj range history
Rc = mean(Rm, 2);
a  = sqrt(ncr_link_budget(cfg, Rc, sc.rcs(sel), af(:))) .* exp(1j * sc.phi(sel));
K  = numel(sel);
Ar = exp(1j * 2 * pi * (0:Nr-1)' * (2 * bj * Rc' / (c * Nr)));      % Nr x K
Ad = exp(1j * 4 * pi * Rm.' / lam);                                 % nj x K
Ax = exp(1j * pi * (0:Nrx-1)' * u(sel)');                           % Nrx x K
KR = reshape(reshape(Ax, Nrx, 1, K) .* reshape(Ad, 1, nj, K), Nrx * nj, K);
X  = reshape(double(single(Ar) * single(a .* KR.')), Nr, Nrx, nj);   % single-precision synthesis

% ---- thermal noise
X = X + (randn(Nr, Nrx, nj) + 1j * randn(Nr, Nrx, nj)) / sqrt(2);

% ---- mutual interference
info.intfDuty = 0;
if intfOn
    [XI, ii] = ncr_interference(cfg, intf, nj, bj, tm);
    X = X + XI;
    info.intfDuty = ii.dutyFraction;
end

% ---- ADC quantisation (I and Q)
info.clipFraction = 0;
if isfinite(cfg.rf.adcBits)
    FS = cfg.rf.adcFS;  q = 2 * FS / 2^cfg.rf.adcBits;
    re = min(max(round(real(X) / q) * q, -FS), FS);
    im = min(max(round(imag(X) / q) * q, -FS), FS);
    info.clipFraction = mean(abs(real(X(:))) > FS | abs(imag(X(:))) > FS);
    X = re + 1j * im;
end
info.nScatterers = K;
end
