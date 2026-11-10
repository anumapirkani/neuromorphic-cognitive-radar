function [assign, total] = ncr_hungarian(C)
%NCR_HUNGARIAN  Optimal assignment (Hungarian / Kuhn-Munkres, O(n^3)).
%
%   [assign, total] = NCR_HUNGARIAN(C) minimises the total cost of assigning
%   rows to columns of the n x m cost matrix C. assign(i) is the column given
%   to row i (0 if unassigned or if the only option has infinite cost).
%   Implementation: shortest augmenting path with row/column potentials;
%   index 1 plays the role of the dummy column 0 of the classical algorithm.
[n, m] = size(C);
assign = zeros(n, 1);  total = 0;
if n == 0 || m == 0, return; end
N = max(n, m);
fin = C(isfinite(C));
if isempty(fin), return; end
big = (max(abs(fin)) + 1) * 10 * N;
A = big * ones(N);
Cf = C;  Cf(~isfinite(Cf)) = big;
A(1:n, 1:m) = Cf;

u = zeros(1, N + 1);  v = zeros(1, N + 1);
p = zeros(1, N + 1);  way = zeros(1, N + 1);
for i = 1:N
    p(1) = i;  j0 = 1;
    minv = inf(1, N + 1);  used = false(1, N + 1);
    while true
        used(j0) = true;
        i0 = p(j0);  delta = inf;  j1 = 0;
        for j = 2:N+1
            if ~used(j)
                cur = A(i0, j - 1) - u(i0 + 1) - v(j);
                if cur < minv(j), minv(j) = cur;  way(j) = j0; end
                if minv(j) < delta, delta = minv(j);  j1 = j; end
            end
        end
        for j = 1:N+1
            if used(j)
                u(p(j) + 1) = u(p(j) + 1) + delta;  v(j) = v(j) - delta;
            else
                minv(j) = minv(j) - delta;
            end
        end
        j0 = j1;
        if p(j0) == 0, break; end
    end
    while true
        j1 = way(j0);  p(j0) = p(j1);  j0 = j1;
        if j0 == 1, break; end
    end
end
for j = 2:N+1
    r = p(j);  c = j - 1;
    if r >= 1 && r <= n && c <= m && isfinite(C(r, c))
        assign(r) = c;
        total = total + C(r, c);
    end
end
end
