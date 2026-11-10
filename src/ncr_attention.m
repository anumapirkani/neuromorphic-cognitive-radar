function varargout = ncr_attention(op, cfg, varargin)
%NCR_ATTENTION  Saliency, attention fields and resource allocation.
%
%   S  = NCR_ATTENTION('saliency', cfg, E, n)     rule-based saliency
%   F  = NCR_ATTENTION('field', cfg, F, S, sSat)  attention-field update
%   u  = NCR_ATTENTION('peak', cfg, F)            sector attention (peak of the field per sector)
%   ut = NCR_ATTENTION('normalise', cfg, u)       normalised attention
%   [n, b] = NCR_ATTENTION('allocate', cfg, ut)   dwell and bandwidth maps
%
%   Saliency (rule-based controller): S(p) = max_d m_j(v_d) E(p, d), with the
%   resolution-aware motion prior m_j(v) = 1 - exp(-v^2 / (2 sigma_j^2)),
%   sigma_j = max(sigV, v_res,j / 2), because a stationary return can appear
%   up to half a Doppler resolution cell away from 0 m/s.
%
%   Field: F <- beta F + (1 - beta) G * min(1, S / sSat), where G is a 3 x 3
%   binomial spatial kernel and the saturation makes any salient return count
%   fully (weak but relevant targets are not out-competed by strong ones).
%
%   Allocation: dwell n_j = nMin + (Nc - M nMin) ut_j (capped at nMax, with
%   the excess redistributed, integer-rounded to sum to Nc); bandwidth
%   b_j = min(Bmax, Bmin + kappa ut_j), with kappa set by bisection so that
%   mean(b) = B0 when attainable.
switch lower(op)
    case 'saliency', varargout{1} = saliency(cfg, varargin{:});
    case 'field',    varargout{1} = field(cfg, varargin{:});
    case 'peak',     varargout{1} = sector_peak(cfg, varargin{:});
    case 'normalise', varargout{1} = normalise(cfg, varargin{:});
    case 'allocate', [varargout{1}, varargout{2}] = allocate(cfg, varargin{:});
    otherwise, error('ncr_attention:op', 'Unknown operation "%s".', op);
end
end

function S = saliency(cfg, E, n)
d = cfg.d;
vres = d.lambda ./ (2 * n * cfg.wf.Tc);
sig  = max(cfg.ctl.sigV, 0.5 * vres(d.sectorOfCell));
m = 1 - exp(-d.vCoarse(:).^2 ./ (2 * sig.^2));             % NDc x NAc
S = reshape(max(E .* reshape(m, 1, d.NDc, d.NAc), [], 2), d.NRc, d.NAc);
end

function F = field(cfg, F, S, sSat)
S = min(1, S / sSat);
k = [1 2 1]' * [1 2 1] / 16;
F = cfg.ctl.beta * F + (1 - cfg.ctl.beta) * conv2(S, k, 'same');
end

function u = sector_peak(cfg, F)
u = max(reshape(max(F, [], 1), cfg.d.cellsPerSector, cfg.ar.M), [], 1);
end

function ut = normalise(cfg, u)
u = u / max(max(u), eps);
ut = (u + cfg.ctl.eps).^cfg.ctl.alpha;
ut = ut / sum(ut);
end

function [n, b] = allocate(cfg, ut)
B = cfg.bud;  wf = cfg.wf;  M = cfg.ar.M;
ut = ut(:)' / sum(ut);
n = B.nMin + (B.Nc - M * B.nMin) * ut;
for it = 1:M
    over = n > B.nMax;
    if ~any(over), break; end
    ex = sum(n(over) - B.nMax);  n(over) = B.nMax;
    fr = n < B.nMax;
    w = ut(fr);  if sum(w) == 0, w = ones(size(w)); end
    n(fr) = n(fr) + ex * w / sum(w);
end
nf = floor(n);  r = B.Nc - sum(nf);
[~, ix] = sort(n - nf, 'descend');
nf(ix(1:r)) = nf(ix(1:r)) + 1;
n = min(nf, B.nMax);
lo = 0;  hi = 100 * M * (wf.Bmax - wf.Bmin);
for it = 1:60
    kb = (lo + hi) / 2;
    if mean(min(wf.Bmax, wf.Bmin + kb * ut)) > wf.B0, hi = kb; else, lo = kb; end
end
b = min(wf.Bmax, wf.Bmin + lo * ut);
end
