function [snr, info] = ncr_link_budget(cfg, R, rcsLin, gTx)
%NCR_LINK_BUDGET  Per-sample, per-channel SNR from the monostatic radar equation.
%
%   snr = NCR_LINK_BUDGET(cfg, R, rcsLin, gTx) returns the linear SNR of one
%   complex ADC sample on one receive channel for scatterers at range R (m)
%   with RCS rcsLin (m^2), illuminated with the normalised transmit-pattern
%   gain gTx (0..1, 1 = beam peak). All inputs broadcast.
%
%       Pr = Pt Ntx^2 Gt Gr lambda^2 sigma gTx / ((4 pi)^3 R^4 L)
%       N  = k T0 F Fs
%
%   [snr, info] also returns the individual terms (dB) for reporting.
if nargin < 4, gTx = 1; end
rf = cfg.rf;  lam = cfg.d.lambda;
Pt = 10^((rf.Pt_dBm - 30) / 10);
Gt = 10^(rf.Gt_dBi / 10);   Gr = 10^(rf.Gr_dBi / 10);
L  = 10^(rf.Lsys_dB / 10);
Pr = Pt * rf.Ntx^2 * Gt * Gr * lam^2 .* rcsLin .* gTx ./ ((4*pi)^3 .* R.^4 * L);
snr = Pr / cfg.d.noiseW;
if nargout > 1
    info.Pt_dBm      = rf.Pt_dBm;
    info.arrayGain_dB = 20 * log10(rf.Ntx);
    info.EIRP_dBm    = rf.Pt_dBm + 20*log10(rf.Ntx) + rf.Gt_dBi;
    info.lambda2_dB  = 10 * log10(lam^2);
    info.spreading_dB = -10 * log10((4*pi)^3);
    info.noise_dBm   = 10 * log10(cfg.d.noiseW) + 30;
    info.Pr_dBm      = 10 * log10(Pr) + 30;
    info.snr_dB      = 10 * log10(snr);
end
end
