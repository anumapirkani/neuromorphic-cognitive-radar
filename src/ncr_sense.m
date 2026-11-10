function [obs, state] = ncr_sense(cfg, scene, t, n, b, state, keepCubes)
%NCR_SENSE  One CPI of the radar: all sector sub-frames, processing, detection.
%
%   [obs, state] = NCR_SENSE(cfg, scene, t, n, b, state) transmits n(j) chirps
%   with sweep bandwidth b(j) into each sector j in turn, processes each
%   sub-frame and returns
%       obs.det     detections [R v theta sector P SNR dR dvRes]
%       obs.Zc      coarse normalised map (NRc x NDc x NAc) for perception
%       obs.intfDuty, obs.excised, obs.clip   front-end diagnostics
%       obs.cubes   per-sector cubes (only when keepCubes is true)
%   state carries scene memory (vegetation texture) between CPIs.
if nargin < 7, keepCubes = false; end
[sc, state] = ncr_scatterers(cfg, scene, t, state);
intfOn = scene.intf.enable && isfinite(scene.intf.EIRP_dBm) && ...
         t >= scene.intf.on && t <= scene.intf.off;
d = cfg.d;
obs.Zc = zeros(d.NRc, d.NDc, d.NAc);
obs.det = zeros(0, 8);
obs.intfDuty = 0;  obs.excised = 0;  obs.clip = 0;
if keepCubes, obs.cubes = cell(1, cfg.ar.M); end
tStart = (t - 1) * cfg.sim.Tf;
for j = 1:cfg.ar.M
    tm = tStart + (0:n(j) - 1) * cfg.wf.Tc;
    [X, fi] = ncr_frontend(cfg, sc, scene.intf, j, n(j), b(j), tm, intfOn);
    [X, ex] = ncr_mitigate(cfg, X);
    cube = ncr_process(cfg, X, j, n(j), b(j));
    hit  = ncr_cfar(cfg, cube.P, n(j));
    obs.det = [obs.det; ncr_detect(cfg, cube, hit)];
    obs.Zc  = ncr_coarse_map(cfg, cube, obs.Zc);
    obs.intfDuty = obs.intfDuty + fi.intfDuty * n(j) / cfg.bud.Nc;
    obs.excised  = obs.excised + ex * n(j) / cfg.bud.Nc;
    obs.clip     = max(obs.clip, fi.clipFraction);
    if keepCubes, obs.cubes{j} = cube; end
    tStart = tm(end) + cfg.wf.Tc;
end
end
