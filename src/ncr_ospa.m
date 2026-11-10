function [d, loc, card] = ncr_ospa(X, Y, c, p)
%NCR_OSPA  Optimal sub-pattern assignment distance between two point sets.
%
%   d = NCR_OSPA(X, Y, c, p) for X (m x 2) and Y (n x 2) positions, cut-off c
%   and order p (Schuhmacher, Vo and Vo, IEEE TSP 2008):
%       d = ( (1/n) ( min_pi sum_i min(c, |x_i - y_pi(i)|)^p + c^p (n - m) ) )^(1/p),
%   with m <= n (sets swapped if needed). loc and card return the
%   localisation and cardinality components.
m = size(X, 1);  n = size(Y, 1);
if m == 0 && n == 0, d = 0; loc = 0; card = 0; return; end
if m == 0 || n == 0, d = c; loc = 0; card = c; return; end
if m > n, [X, Y] = deal(Y, X); [m, n] = deal(n, m); end
D = zeros(m, n);
for i = 1:m
    D(i, :) = sqrt(sum((Y - X(i, :)).^2, 2))';
end
D = min(c, D).^p;
[~, cost] = ncr_hungarian(D);
d    = ((cost + c^p * (n - m)) / n)^(1 / p);
loc  = (cost / n)^(1 / p);
card = (c^p * (n - m) / n)^(1 / p);
end
