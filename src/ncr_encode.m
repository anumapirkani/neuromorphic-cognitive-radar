function E = ncr_encode(cfg, Zc, ZcPrev)
%NCR_ENCODE  Spike encoding of the coarse radar map.
%
%   E = NCR_ENCODE(cfg, Zc, ZcPrev) returns the encoder drive E in [0, 1)
%   for every input neuron (NRc x NDc x NAc). How E becomes spikes depends
%   on cfg.enc.type (see NCR_SNN_RUN):
%     'peak-rate'  peak selection along Doppler and angle: only local
%                  maxima above zth emit; Poisson rate pMax * E with the
%                  saturating law E = rho / (rho + rho0), rho = Z - Z_th.
%     'rate'       same rate law for every cell above zth (no peak selection).
%     'latency'    peaks as in 'peak-rate', one spike per neuron at step
%                  ceil(T (1 - E)) (time-to-first-spike).
%     'delta'      temporal contrast: peaks whose power rose by more than
%                  delta_dB since the previous CPI (event-camera style).
zth = 10^(cfg.enc.zth_dB / 10);
rho = max(Zc - zth, 0);
Esat = rho ./ (rho + cfg.enc.rho0);
switch lower(cfg.enc.type)
    case 'rate'
        E = Esat;
    case {'peak-rate', 'latency'}
        E = Esat .* peaks(cfg, Zc, zth);
    case 'delta'
        if isempty(ZcPrev), ZcPrev = zeros(size(Zc)); end
        rise = 10 * log10((Zc + 1) ./ (ZcPrev + 1));
        E = Esat .* peaks(cfg, Zc, zth) .* (rise > cfg.enc.delta_dB);
    otherwise
        error('ncr_encode:type', 'Unknown encoder "%s".', cfg.enc.type);
end
end

function pk = peaks(cfg, Zc, zth)
d = cfg.d;
Za = cat(3, -inf(d.NRc, d.NDc, 1), Zc, -inf(d.NRc, d.NDc, 1));
pk = Zc > circshift(Zc, 1, 2) & Zc >= circshift(Zc, -1, 2) & ...   % ties -> one peak
     Zc > Za(:, :, 1:end-2) & Zc >= Za(:, :, 3:end) & Zc > zth;
end
