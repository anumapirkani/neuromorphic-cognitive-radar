function [net, info] = ncr_rstdp(cfg, net, res, rmap)
%NCR_RSTDP  Reward-modulated STDP update of the shared Doppler kernels.
%
%   [net, info] = NCR_RSTDP(cfg, net, res, rmap) applies the three-factor rule
%       e_kd   <- lamE e_kd + sum_p m_p Elig(k, d, p)
%       W_kd   <- clip(W_kd + eta e_kd, 0, wMax)
%   where the neuromodulator m_p depends on cfg.lrn.rule:
%     'spatial'  m_p = r_p, the cognitive reward released locally in cell p
%                (topographic neuromodulation);
%     'global'   m_p = r_g - rbar for every cell, a single scalar reward
%                prediction error (activity-weighted mean reward minus its
%                running average), as in classical R-STDP;
%     'none'     no plasticity (fixed weights).
%   With finite cfg.lrn.bits the weights are kept on a uniform grid of
%   2^bits levels in [0, wMax] using stochastic rounding (as on neuromorphic
%   hardware with low-precision synapses).
lr = cfg.lrn;  s = cfg.snn;
info.rg = NaN;  info.dWnorm = 0;
if strcmpi(lr.rule, 'none'), return; end
K = s.K;  ND = cfg.d.NDc;  Np = cfg.d.Np;
E2 = reshape(res.Elig, K * ND, Np);
switch lower(lr.rule)
    case 'spatial'
        drive = reshape(E2 * rmap(:), K, ND);
    case 'global'
        act = sum(res.C, 1)';
        if sum(act) > 0
            rg = sum(rmap(:) .* act) / sum(act);
        else
            rg = 0;
        end
        delta = rg - net.rBar;
        net.rBar = lr.rBeta * net.rBar + (1 - lr.rBeta) * rg;
        drive = delta * reshape(sum(E2, 2), K, ND);
        info.rg = rg;
    otherwise
        error('ncr_rstdp:rule', 'Unknown learning rule "%s".', lr.rule);
end
net.eTrace = lr.lamE * net.eTrace + drive;
dW = lr.eta * net.eTrace;
W = min(max(net.W + dW, 0), s.wMax);
if isfinite(lr.bits)
    q = s.wMax / (2^lr.bits - 1);
    lo = floor(W / q);
    W = (lo + (rand(size(W)) < (W / q - lo))) * q;      % stochastic rounding
    W = min(W, s.wMax);
end
info.dWnorm = norm(W(:) - net.W(:));
net.W = W;
net.nUpdates = net.nUpdates + nnz(dW);
end
