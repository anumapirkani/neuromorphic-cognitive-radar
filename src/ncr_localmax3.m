function L = ncr_localmax3(A)
%NCR_LOCALMAX3  True where A is a local maximum of its 3 x 3 x 3 neighbourhood.
%   Dimension 2 (Doppler) is treated as circular; dims 1 and 3 are not.
[a, b, c] = size(A);
Ap = -inf(a + 2, b + 2, c + 2);
Ap(2:a+1, 2:b+1, 2:c+1) = A;
Ap(2:a+1, 1, 2:c+1)   = A(:, b, :);
Ap(2:a+1, b+2, 2:c+1) = A(:, 1, :);
L = true(a, b, c);
for di = -1:1
    for dj = -1:1
        for dk = -1:1
            if di == 0 && dj == 0 && dk == 0, continue; end
            L = L & (A >= Ap((2:a+1) + di, (2:b+1) + dj, (2:c+1) + dk));
        end
    end
end
end
