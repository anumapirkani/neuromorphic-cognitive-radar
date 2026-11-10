function ev = ncr_evaluate(cfg, truth, det, meas, trk, n)
%NCR_EVALUATE  Per-CPI performance metrics against ground truth.
%
%   Detection level (clusters of moving detections):
%     pd(k)      target k detected (a measurement within the association gates)
%     nFA        moving measurements not associated with any target
%     nPts(k)    moving detections on target k (point-cloud density)
%     rErr       range errors of detections w.r.t. the nearest scattering centre
%   Track level (confirmed tracks):
%     cover(k)   a confirmed track within coverR of target k
%     ospa       OSPA distance (cut-off c, order p) between confirmed tracks
%                and active targets, with localisation and cardinality parts
%     posErr     position errors of covering tracks
%   Resources:
%     dwellOnTgt fraction of the chirp budget spent on sectors with moving targets
nT = numel(truth);
ev.pd = nan(1, nT);  ev.nPts = nan(1, nT);  ev.cover = nan(1, nT);
ev.rErr = [];  ev.posErr = [];
used = false(size(meas, 1), 1);
mvDet = det(abs(det(:, 2)) >= cfg.det.vStatic, :);
for k = 1:nT
    tr = truth(k);
    if ~tr.active, continue; end
    j = tr.sector;
    gV = max(1.5, 1.5 * cfg.d.lambda / (2 * n(j) * cfg.wf.Tc));
    if ~isempty(meas)
        m = abs(meas(:, 1) - tr.R) <= 3 & abs(meas(:, 2) - tr.theta) <= 6 & ...
            abs(meas(:, 3) - tr.vr) <= gV + 2.5;
        ev.pd(k) = any(m);  used = used | m;
    else
        ev.pd(k) = 0;
    end
    if ~isempty(mvDet)
        md = abs(mvDet(:, 1) - tr.R) <= 3 & abs(mvDet(:, 3) - tr.theta) <= 6;
        ev.nPts(k) = nnz(md);
        for q = find(md)'
            [~, i] = min(abs(tr.Rparts - mvDet(q, 1)));
            ev.rErr = [ev.rErr; mvDet(q, 1) - tr.Rparts(i)]; %#ok<AGROW>
        end
    else
        ev.nPts(k) = 0;
    end
end
ev.nFA = nnz(~used);

% ---------------------------------------------------------------- tracks
tp = zeros(0, 2);
for i = 1:numel(trk.tracks)
    if trk.tracks(i).conf, tp = [tp; trk.tracks(i).x(1:2)']; end %#ok<AGROW>
end
act = find([truth.active]);
gt = reshape([truth(act).pos], 2, [])';
for k = act
    if isempty(tp)
        ev.cover(k) = 0;
    else
        dk = sqrt(sum((tp - truth(k).pos).^2, 2));
        ev.cover(k) = any(dk <= cfg.trk.coverR);
        if ev.cover(k), ev.posErr = [ev.posErr; min(dk)]; end
    end
end
[ev.ospa, ev.ospaLoc, ev.ospaCard] = ncr_ospa(tp, gt, cfg.trk.ospaC, cfg.trk.ospaP);
ev.nConf = size(tp, 1);
tgtSec = unique([truth(act).sector]);
ev.dwellOnTgt = sum(n(tgtSec)) / cfg.bud.Nc;
end
