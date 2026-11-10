function cfg = ncr_derive(cfg)
%NCR_DERIVE  Quantities derived from the primary configuration.
d = struct();
d.lambda  = cfg.const.c / cfg.rf.fc;
d.Tramp   = cfg.wf.Nr / cfg.wf.Fs;                       % sampled ramp duration
d.Tcpi    = cfg.bud.Nc * cfg.wf.Tc;                      % CPI duration
d.noiseW  = cfg.const.kB * cfg.const.T0 * 10^(cfg.rf.NF_dB/10) * cfg.wf.Fs;   % noise power per sample
d.nUni    = cfg.bud.Nc / cfg.ar.M;                       % uniform dwell per sector

% receive beams (deg) for every sector
d.beams = zeros(cfg.ar.M, cfg.ar.Na);
w = diff(cfg.ar.secEdges(1:2));
for j = 1:cfg.ar.M
    d.beams(j, :) = cfg.ar.secEdges(j) + w / cfg.ar.Na * ((1:cfg.ar.Na) - 0.5);
end
d.secCentre = (cfg.ar.secEdges(1:end-1) + cfg.ar.secEdges(2:end)) / 2;

% Doppler axes: fine (FFT) and coarse (stride 2, centred on 0 m/s)
d.dv     = d.lambda / (2 * cfg.wf.Tc * cfg.proc.Nd);
d.vFine  = (-cfg.proc.Nd/2 : cfg.proc.Nd/2 - 1) * d.dv;
d.vCoarse = d.vFine(1:2:end);
d.vmax   = d.lambda / (4 * cfg.wf.Tc);

% perception grid
d.NRc = round(cfg.proc.Rproc / cfg.grid.dRc);
d.NAc = round((cfg.ar.secEdges(end) - cfg.ar.secEdges(1)) / cfg.grid.dAc);
d.cellsPerSector = d.NAc / cfg.ar.M;
d.NDc = cfg.proc.Nd / 2;
d.Np  = d.NRc * d.NAc;
d.rCoarse  = cfg.grid.dRc * ((1:d.NRc) - 0.5);
d.thCoarse = cfg.ar.secEdges(1) + cfg.grid.dAc * ((1:d.NAc) - 0.5);
d.sectorOfCell = ceil((1:d.NAc) / d.cellsPerSector);

% window coherent gains (used by the link budget and self-tests)
d.cgR = coherent_gain(ncr_hann(cfg.wf.Nr));
d.cgX = coherent_gain(ncr_hann(cfg.rf.Nrx));

% sizes of the neuromorphic network
d.Nin  = d.NDc * d.Np;
d.Nhid = cfg.snn.K * d.Np;
d.Nout = cfg.ar.M;
d.denseMACs = d.NDc * d.Np * cfg.snn.K;                  % one frame-based pass
cfg.d = d;

if mod(d.NAc, cfg.ar.M) ~= 0
    error('ncr_derive:grid', 'Coarse angle cells must divide evenly into sectors.');
end
end

function g = coherent_gain(w)
g = sum(w)^2 / sum(w.^2);
end
