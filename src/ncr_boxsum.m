function S = ncr_boxsum(A, hr, hd)
%NCR_BOXSUM  Sliding (2hr+1) x (2hd+1) window sums over the first two dims.
%   Works page-wise on 3-D arrays. Zero padding along dim 1 (range) and
%   circular padding along dim 2 (Doppler, an FFT axis). Uses an integral
%   image, so the cost does not depend on the window size.
[a, c, p] = size(A);
hd = min(hd, floor((c - 1) / 2));
Ap = cat(2, A(:, end-hd+1:end, :), A, A(:, 1:hd, :));
Ap = cat(1, zeros(hr, size(Ap, 2), p), Ap, zeros(hr, size(Ap, 2), p));
C  = cumsum(cumsum(Ap, 1), 2);
C  = cat(1, zeros(1, size(C, 2), p), C);
C  = cat(2, zeros(size(C, 1), 1, p), C);
I = 1:a;  J = 1:c;
S = C(I + 2*hr + 1, J + 2*hd + 1, :) - C(I, J + 2*hd + 1, :) ...
  - C(I + 2*hr + 1, J, :) + C(I, J, :);
end
