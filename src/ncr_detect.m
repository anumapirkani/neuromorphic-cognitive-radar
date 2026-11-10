function det = ncr_detect(cfg, cube, hit)
%NCR_DETECT  Peak picking and sub-bin interpolation of CFAR detections.
%
%   det = NCR_DETECT(cfg, cube, hit) returns one row per detection:
%       [R  v  theta  sector  P  SNR_dB  dR  dvRes]
%   Detections are CFAR hits that are also 3-D local maxima. Range, Doppler
%   and angle are refined by parabolic interpolation of the log-power around
%   the peak (Doppler interpolation is on the zero-padded axis).
P = cube.P;
idx = find(hit & ncr_localmax3(P));
det = zeros(numel(idx), 8);
if isempty(idx), return; end
[ir, id, ia] = ind2sub(size(P), idx);
[Nk, Nd, Na] = size(P);
LP = 10 * log10(P + realmin);
dr = zeros(numel(idx), 1);  dd = dr;  da = dr;
for q = 1:numel(idx)
    i = ir(q);  k = id(q);  a = ia(q);
    if i > 1 && i < Nk
        dr(q) = parab(LP(i-1, k, a), LP(i, k, a), LP(i+1, k, a));
    end
    km = mod(k - 2, Nd) + 1;  kp = mod(k, Nd) + 1;
    dd(q) = parab(LP(i, km, a), LP(i, k, a), LP(i, kp, a));
    if a > 1 && a < Na
        da(q) = parab(LP(i, k, a-1), LP(i, k, a), LP(i, k, a+1));
    end
end
dvF = cfg.d.dv;
dth = cube.th(min(2, Na)) - cube.th(1);
R  = cube.r(ir) + dr * cube.dR;
v  = cube.v(id)' + dd * dvF;
th = cube.th(ia)' + da * dth;
snr = 10 * log10(P(idx) ./ cube.floor(ia)');
det = [R, v, th, cube.sector * ones(numel(idx), 1), P(idx), snr, ...
       cube.dR * ones(numel(idx), 1), cube.dvRes * ones(numel(idx), 1)];
end

function d = parab(ym, y0, yp)
den = ym - 2 * y0 + yp;
if den >= 0 || ~isfinite(den)
    d = 0;
else
    d = 0.5 * (ym - yp) / den;
    d = max(-0.5, min(0.5, d));
end
end
