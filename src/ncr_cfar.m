function [hit, thr] = ncr_cfar(cfg, P, nj)
%NCR_CFAR  Constant-false-alarm-rate detection on a range x Doppler x beam cube.
%
%   hit = NCR_CFAR(cfg, P, nj) applies, independently on every beam:
%     'CA'  two-dimensional cell-averaging CFAR over range and Doppler. The
%           Doppler guard/training extents scale with the zero-padding
%           factor Nd/nj so that they always span the same number of
%           physical resolution cells. Threshold factor for N training cells:
%               alpha = N (Pfa^(-1/N) - 1).
%     'OS'  ordered-statistic CFAR along range (k-th of 2 Tr cells); alpha
%           is found numerically from Pfa = prod_{i=0}^{k-1} (N-i)/(N-i+alpha).
%   Range edges use only the available training cells; Doppler is circular.
d = cfg.det;  Nd = size(P, 2);
switch upper(d.cfar)
    case 'CA'
        Gd = ceil(1.5 * Nd / nj);
        Td = ceil(2 * Nd / nj) + 2;
        so = ncr_boxsum(P, d.Gr + d.Tr, Gd + Td);
        si = ncr_boxsum(P, d.Gr, Gd);
        on = ncr_boxsum(ones(size(P, 1), Nd), d.Gr + d.Tr, Gd + Td);
        in = ncr_boxsum(ones(size(P, 1), Nd), d.Gr, Gd);
        N  = on - in;
        thr = (d.Pfa.^(-1 ./ N) - 1) .* (so - si);
    case 'OS'
        Nk = size(P, 1);
        offs = [-(d.Gr + d.Tr):-(d.Gr + 1), (d.Gr + 1):(d.Gr + d.Tr)];
        S = nan([size(P), numel(offs)]);
        for q = 1:numel(offs)
            o = offs(q);
            src = (1:Nk) + o;  ok = src >= 1 & src <= Nk;
            S(ok, :, :, q) = P(src(ok), :, :);
        end
        S = sort(S, 4);                                   % NaN sorted last
        nValid = sum(~isnan(S), 4);
        kIdx = max(1, round(d.osK * nValid));
        [I1, I2, I3] = ndgrid(1:size(P, 1), 1:size(P, 2), 1:size(P, 3));
        Zk = S(sub2ind(size(S), I1, I2, I3, kIdx));
        alpha = os_alpha(2 * d.Tr, round(d.osK * 2 * d.Tr), d.Pfa);
        thr = alpha * Zk;
    otherwise
        error('ncr_cfar:type', 'Unknown CFAR type "%s".', d.cfar);
end
hit = P > thr;
end

function a = os_alpha(N, k, Pfa)
f = @(a) prod((N - (0:k-1)) ./ (N - (0:k-1) + a)) - Pfa;
lo = 0;  hi = 1e4;
for it = 1:200
    m = (lo + hi) / 2;
    if f(m) > 0, lo = m; else, hi = m; end
end
a = (lo + hi) / 2;
end
