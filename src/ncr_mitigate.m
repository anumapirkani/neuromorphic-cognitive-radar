function [X, frac] = ncr_mitigate(cfg, X)
%NCR_MITIGATE  Time-domain interference excision (zeroing) per chirp.
%
%   Samples whose channel-averaged magnitude exceeds zeroK times a robust
%   noise estimate are set to zero together with zeroGuard neighbours. The
%   robust estimate uses the median magnitude of a unit-variance complex
%   Gaussian, sqrt(ln 2).
frac = 0;
if ~strcmpi(cfg.intf.mitigation, 'zeroing'), return; end
mag = squeeze(mean(abs(X), 2));                          % Nr x nj
s = median(mag, 1) / sqrt(log(2));
mask = mag > cfg.intf.zeroK * s;
g = cfg.intf.zeroGuard;
if g > 0
    mask = conv2(double(mask), ones(2*g + 1, 1), 'same') > 0;
end
X = X .* reshape(~mask, size(X, 1), 1, size(X, 3));
frac = mean(mask(:));
end
