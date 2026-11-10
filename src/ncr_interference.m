function [XI, info] = ncr_interference(cfg, intf, nj, bj, tm)
%NCR_INTERFERENCE  FMCW-to-FMCW mutual interference at the victim ADC.
%
%   XI = NCR_INTERFERENCE(cfg, intf, nj, bj, tm) returns the interference
%   component (Nr x Nrx x nj, in units of the noise standard deviation) for a
%   victim sub-frame of nj chirps with sweep bandwidth bj starting at the
%   absolute times tm (1 x nj).
%
%   The interferer transmits its own sawtooth chirps (bandwidth intf.B, ramp
%   intf.Tramp, period intf.Tc, start offset intf.f0) that are not
%   synchronised with the victim. After dechirping, its contribution is a
%   chirp at the difference frequency f_v(t) - f_i(t); only the portion that
%   falls inside the complex IF band |f| < Fs/2 reaches the ADC, which
%   produces short, broadband bursts whose length scales with
%   Fs / |S_v - S_i|. The interferer is a point source at intf.theta, so
%   the burst is coherent across the receive array. Its power follows the
%   one-way Friis equation:
%       P_I = EIRP + G_r - 20 log10(4 pi R_I / lambda) - L_rx.
%
%   info returns INR per sample (dB) and the fraction of samples affected.
Fs = cfg.wf.Fs;  Nr = cfg.wf.Nr;  lam = cfg.d.lambda;
fspl = 20 * log10(4 * pi * intf.R / lam);
PI_dBm = intf.EIRP_dBm + cfg.rf.Gr_dBi - fspl - intf.Lrx_dB;
INR = 10^((PI_dBm - 30) / 10) / cfg.d.noiseW;

Sv = bj / cfg.d.Tramp;                     % victim slope
SI = intf.B / intf.Tramp;                  % interferer slope
n  = (0:Nr-1) / Fs;                        % fast time within the victim ramp
t  = tm(:) + n;                            % nj x Nr absolute sample times
tauI = mod(t - intf.t0, intf.Tc);          % time inside the interferer chirp period
active = tauI < intf.Tramp;
fif = Sv * n - (intf.f0 + SI * tauI);      % IF (difference) frequency, Hz
pass = active & abs(fif) < Fs / 2;         % ideal complex anti-aliasing filter
ph = 2 * pi * cumsum(fif, 2) / Fs + 2 * pi * rand(nj, 1);
xi = sqrt(INR) * pass .* exp(1j * ph);     % nj x Nr
steer = exp(1j * pi * sind(intf.theta) * (0:cfg.rf.Nrx - 1));
XI = reshape(xi.', Nr, 1, nj) .* reshape(steer, 1, cfg.rf.Nrx);
if nargout > 1
    info.INR_dB = 10 * log10(INR);
    info.dutyFraction = mean(pass(:));
end
end
