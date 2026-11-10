function varargout = ncr_stats(op, varargin)
%NCR_STATS  Small statistics toolkit (no toolboxes required).
%
%   [m, h] = NCR_STATS('ci', X, dim)   mean and 95 % confidence half-width
%                                      (Student t) along dim, ignoring NaN
%   p      = NCR_STATS('pairedt', a, b) two-sided paired t-test p-value
%   tc     = NCR_STATS('tinv', p, df)   Student-t quantile
%   F      = NCR_STATS('tcdf', t, df)   Student-t CDF (via the incomplete beta)
switch lower(op)
    case 'ci',      [varargout{1}, varargout{2}] = ci(varargin{:});
    case 'pairedt', varargout{1} = pairedt(varargin{:});
    case 'tinv',    varargout{1} = tinv_(varargin{:});
    case 'tcdf',    varargout{1} = tcdf_(varargin{:});
    otherwise, error('ncr_stats:op', 'Unknown operation "%s".', op);
end
end

function [m, h] = ci(X, dim)
if nargin < 2, dim = 1; end
ok = ~isnan(X);
n = sum(ok, dim);
X0 = X;  X0(~ok) = 0;
m = sum(X0, dim) ./ max(n, 1);
dev = (X - m);  dev(~ok) = 0;
s = sqrt(sum(dev.^2, dim) ./ max(n - 1, 1));
tc = arrayfun(@(k) tinv_(0.975, max(k - 1, 1)), n);
h = tc .* s ./ sqrt(max(n, 1));
m(n == 0) = NaN;  h(n < 2) = NaN;
end

function p = pairedt(a, b)
d = a(:) - b(:);  d = d(~isnan(d));
n = numel(d);
if n < 2 || std(d) == 0, p = NaN; return; end
t = mean(d) / (std(d) / sqrt(n));
p = 2 * (1 - tcdf_(abs(t), n - 1));
end

function F = tcdf_(t, df)
x = df ./ (df + t.^2);
F = 1 - 0.5 * betainc(x, df / 2, 0.5);
F(t < 0) = 1 - F(t < 0);
end

function t = tinv_(p, df)
lo = 0;  hi = 1e3;
for it = 1:100
    m = (lo + hi) / 2;
    if tcdf_(m, df) < p, lo = m; else, hi = m; end
end
t = (lo + hi) / 2;
end
