function net = ncr_snn_init(cfg)
%NCR_SNN_INIT  Spiking network: input layer, LIF feature maps, sector outputs.
%
%   Topology (weight-shared, convolution-like):
%     input   NDc x Np Poisson/TTFS neurons (Doppler channel x range-angle cell)
%     hidden  K LIF neurons per cell; neuron (k, p) sees the Doppler spectrum
%             of cell p through the shared kernel W(k, :)  (K x NDc plastic weights)
%     output  one LIF neuron per sector pooling the hidden spikes of its cells
%   The network starts as a permissive energy detector (weights near w0).
s = cfg.snn;
net.W = s.w0 * (1 + s.wSpread * (2 * rand(s.K, cfg.d.NDc) - 1));
net.W = min(net.W, s.wMax);
net.eTrace = zeros(s.K, cfg.d.NDc);        % eligibility carried across CPIs
net.rBar = 0;                              % running reward baseline ('global' rule)
net.nUpdates = 0;
end
