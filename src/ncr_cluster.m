function meas = ncr_cluster(cfg, det)
%NCR_CLUSTER  Single-linkage clustering of moving detections into measurements.
%
%   meas = NCR_CLUSTER(cfg, det) groups detections with |v| >= vStatic that
%   lie within clD metres (Cartesian) and clV m/s of each other (single
%   linkage, DBSCAN-like with min points 1) and returns one
%   power-weighted centroid per cluster:
%       [R  theta  v  Psum  nDet  SNRmax  sector]
%   Extended targets (vehicles) and micro-Doppler (limbs) generate several
%   detections; clustering turns them into one measurement for the tracker.
d = cfg.det;
mv = find(abs(det(:, 2)) >= d.vStatic);
n = numel(mv);
meas = zeros(0, 7);
if n == 0, return; end
D = det(mv, :);
xy = D(:, 1) .* [sind(D(:, 3)), cosd(D(:, 3))];
lab = 1:n;                                         % union-find labels
for a = 1:n-1
    for b = a+1:n
        if norm(xy(a, :) - xy(b, :)) <= d.clD && abs(D(a,2) - D(b,2)) <= d.clV
            la = find_root(lab, a);  lb = find_root(lab, b);
            if la ~= lb, lab(max(la, lb)) = min(la, lb); end
        end
    end
end
for a = 1:n, lab(a) = find_root(lab, a); end
u = unique(lab);
meas = zeros(numel(u), 7);
for q = 1:numel(u)
    m = lab == u(q);
    w = D(m, 5) / sum(D(m, 5));
    meas(q, :) = [w' * D(m, 1), w' * D(m, 3), w' * D(m, 2), sum(D(m, 5)), ...
                  nnz(m), max(D(m, 6)), mode(D(m, 4))];
end
end

function r = find_root(lab, a)
r = a;
while lab(r) ~= r, r = lab(r); end
end
