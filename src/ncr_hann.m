function w = ncr_hann(N)
%NCR_HANN  Symmetric Hann window (column), toolbox-free.
if N <= 1
    w = ones(max(N, 1), 1);
    return
end
w = 0.5 * (1 - cos(2 * pi * (0:N-1)' / (N - 1)));
end
