function [res, rec] = ncr_snn_run(cfg, net, E)
%NCR_SNN_RUN  Simulate the spiking network for one CPI.
%
%   res = NCR_SNN_RUN(cfg, net, E) drives the network with the encoder output
%   E (NRc x NDc x NAc) for cfg.enc.T time steps and returns
%       res.C      hidden spike counts (K x Np)
%       res.y      output spike counts per sector (1 x M)
%       res.Elig   STDP eligibility of the shared synapses per cell (K x NDc x Np)
%       res.nIn, res.nHid, res.nOut   spike totals
%       res.synOps  synaptic events (input spikes x K fan-out + hidden spikes)
%   rec (optional) holds per-step activity per sector for raster plots.
%
%   Hidden dynamics (discrete-time LIF, reset by subtraction):
%       V[t] = lamV (V[t-1] - Vth S[t-1]) + sum_d W(k,d) s_d[t]
%       S[t] = H(V[t] - Vth)
%   Pair-based STDP eligibility with traces xbar (pre) and ybar (post):
%       e += Ap * S_post[t] * xbar[t]  -  Am * ybar[t-1] * s_pre[t]
s = cfg.snn;  lr = cfg.lrn;  d = cfg.d;
K = s.K;  Np = d.Np;  ND = d.NDc;  T = cfg.enc.T;  M = cfg.ar.M;
Ein = reshape(permute(E, [2 1 3]), ND, Np);            % Doppler x cell
lat = strcmpi(cfg.enc.type, 'latency');
if lat
    tFire = ceil(T * (1 - Ein));  tFire(Ein <= 0) = 0;
else
    Pin = cfg.enc.pMax * Ein;
end
active = any(Ein > 0, 1);                               % cells that can receive input
V = zeros(K, Np);  ref = zeros(K, Np);  Vo = zeros(1, M);
xpre = zeros(ND, Np);  ypost = zeros(K, Np);
dP = exp(-1 / lr.tauP);  dM = exp(-1 / lr.tauM);
C = zeros(K, Np);  y = zeros(1, M);  Elig = zeros(K, ND, Np);
nIn = 0;  nHid = 0;  nOut = 0;
wantRec = nargout > 1;
if wantRec
    rec.in = zeros(T, M);  rec.hid = zeros(T, M);  rec.out = false(T, M);
end
cellSector = d.sectorOfCell(ceil((1:Np) / d.NRc));     % sector of every cell
for tau = 1:T
    if lat
        Sin = double(tFire == tau);
    else
        Sin = double(rand(ND, Np) < Pin);
    end
    xpre = dP * xpre + Sin;
    V = s.lamV * V + net.W * Sin;
    Sh = double(V >= s.Vth & ref <= 0);
    V = V - s.Vth * Sh;
    if s.refrac > 0
        ref = max(ref - 1, 0) + s.refrac * Sh;
    end
    if any(active)
        a = active;
        Elig(:, :, a) = Elig(:, :, a) ...
            + lr.Ap * reshape(Sh(:, a), K, 1, []) .* reshape(xpre(:, a), 1, ND, []) ...
            - lr.Am * reshape(ypost(:, a), K, 1, []) .* reshape(Sin(:, a), 1, ND, []);
    end
    ypost = dM * ypost + Sh;
    C = C + Sh;
    hs = accumarray(cellSector(:), sum(Sh, 1)', [M 1])';
    Vo = s.lamOut * Vo + hs;
    So = Vo >= s.VthOut;
    Vo = Vo - s.VthOut * So;
    y = y + So;
    ni = sum(Sin(:));  nh = sum(Sh(:));
    nIn = nIn + ni;  nHid = nHid + nh;  nOut = nOut + sum(So);
    if wantRec
        rec.in(tau, :)  = accumarray(cellSector(:), sum(Sin, 1)', [M 1])';
        rec.hid(tau, :) = hs;
        rec.out(tau, :) = So;
    end
end
res.C = C;  res.y = y;  res.Elig = Elig;
res.nIn = nIn;  res.nHid = nHid;  res.nOut = nOut;
res.synOps = nIn * K + nHid;
res.neuronUpdates = T * (d.Nhid + d.Nout);
end
