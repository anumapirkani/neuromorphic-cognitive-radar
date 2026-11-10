function ncr_tables(cfg, X)
%NCR_TABLES  LaTeX (booktabs) tables and a plain-text summary of all results.
%
%   Writes to cfg.out.dir/tables:
%     tab_link_budget.tex   radar parameters and link budget per target
%     tab_main_results.tex  E1 metrics, mean +/- 95 % CI, paired p-values
%     tab_ablation.tex      E2 neuromorphic variants
%     tab_network.tex       network size, activity and indicative energy
%     summary.txt           all of the above as text
td = fullfile(cfg.out.dir, 'tables');
if ~exist(td, 'dir'), mkdir(td); end
txt = {};

% ---------------------------------------------------------------- link budget
sc = ncr_scene(cfg);
[~, info] = ncr_link_budget(cfg, 10, 1, 1);
nj = cfg.d.nUni;
gD = sum(ncr_hann(nj))^2 / sum(ncr_hann(nj).^2);
Gproc = 10 * log10(cfg.d.cgR * gD * cfg.d.cgX);
rows = {
 'Carrier frequency', sprintf('%.0f GHz', cfg.rf.fc / 1e9);
 'Tx power per channel / Tx elements', sprintf('%g dBm / %d', cfg.rf.Pt_dBm, cfg.rf.Ntx);
 'EIRP (phased beam peak)', sprintf('%.1f dBm', info.EIRP_dBm);
 'Rx channels, element gain', sprintf('%d, %g dBi', cfg.rf.Nrx, cfg.rf.Gr_dBi);
 'Noise figure / system losses', sprintf('%g dB / %g dB', cfg.rf.NF_dB, cfg.rf.Lsys_dB);
 'Sampling rate, samples per chirp', sprintf('%g MHz, %d', cfg.wf.Fs / 1e6, cfg.wf.Nr);
 'Chirp interval, chirps per CPI', sprintf('%g us, %d', cfg.wf.Tc * 1e6, cfg.bud.Nc);
 'Bandwidth per sector (mean budget)', sprintf('%g-%g MHz (%g MHz)', cfg.wf.Bmin/1e6, cfg.wf.Bmax/1e6, cfg.wf.B0/1e6);
 'Noise power per sample', sprintf('%.1f dBm', info.noise_dBm);
 'Per-sample SNR, 1 m^2 at 10 m', sprintf('%.1f dB', info.snr_dB);
 'Processing gain (uniform dwell, Hann)', sprintf('%.1f dB', Gproc)};
tr1 = ncr_truth(cfg, sc, 1);  trN = ncr_truth(cfg, sc, cfg.sim.Nscans);
for k = 1:numel(sc.tgt)
    [~, q] = max([sc.tgt(k).parts.rcs]);
    rcs = 10^(sc.tgt(k).parts(q).rcs / 10);
    if sc.tgt(k).on > 1, R1 = ncr_truth(cfg, sc, sc.tgt(k).on); R1 = R1(k).R; else, R1 = tr1(k).R; end
    s1 = 10 * log10(ncr_link_budget(cfg, R1, rcs)) + Gproc;
    s2 = 10 * log10(ncr_link_budget(cfg, trN(k).R, rcs)) + Gproc;
    rows(end+1, :) = {sprintf('%s, strongest part (%g dBsm): processed SNR', sc.tgt(k).name, sc.tgt(k).parts(q).rcs), ...
                      sprintf('%.1f dB at %.0f m, %.1f dB at %.0f m', s1, R1, s2, trN(k).R)}; %#ok<AGROW>
end
intf0 = cfg.intf;  intf0.t0 = 0;
[~, ii] = ncr_interference(cfg, intf0, 64, cfg.wf.B0, (0:63) * cfg.wf.Tc);
rows(end+1, :) = {'Interferer INR per sample during a burst', sprintf('%.1f dB (EIRP %g dBm at %g m)', ...
                  ii.INR_dB, cfg.intf.EIRP_dBm, cfg.intf.R)};
write_tab(fullfile(td, 'tab_link_budget.tex'), 'Radar parameters and link budget', ...
          'tab:link', {'Parameter', 'Value'}, rows, 'll');
txt = [txt; {'LINK BUDGET'}; fmt_rows(rows)];

% ---------------------------------------------------------------- main results
if isfield(X, 'E1')
    E1 = X.E1;  S = E1.sum;
    defs = {
      'P_D, all CPIs',                      'PDall',       '%.3f';
      'P_D, CPIs 16-64',                    'PDlate',      '%.3f';
      'P_D, vehicle in interference',       'PDvehIntf',   '%.3f';
      'False (moving) clusters per CPI',    'FA',          '%.2f';
      'OSPA, CPIs 16-64 (m)',               'OSPAlate',    '%.2f';
      'OSPA localisation (m)',              'OSPAloc',     '%.2f';
      'OSPA cardinality (m)',               'OSPAcard',    '%.2f';
      'False confirmed tracks per CPI',     'falseTracks', '%.2f';
      'Track position RMSE (m)',            'posRMSE',     '%.2f';
      'Range RMSE of detections (m)',       'rangeRMSE',   '%.3f';
      'Detections per vehicle',             'ptsVeh',      '%.1f';
      'Dwell on target sectors',            'dwellOnTgt',  '%.2f';
      'Cognitive reward R^{cog}',           'Rcog',        '%.2f'};
    rows = cell(size(defs, 1), 6);
    for i = 1:size(defs, 1)
        rows{i, 1} = defs{i, 1};
        for p = 1:3
            v = S{p}.(defs{i, 2});
            if isnan(v.m), rows{i, p+1} = '--';
            else, rows{i, p+1} = sprintf([defs{i, 3} ' $\\pm$ ' defs{i, 3}], v.m, v.h); end
        end
        rows{i, 5} = pstr(ncr_stats('pairedt', S{3}.run.(defs{i, 2}), S{1}.run.(defs{i, 2})));
        rows{i, 6} = pstr(ncr_stats('pairedt', S{3}.run.(defs{i, 2}), S{2}.run.(defs{i, 2})));
    end
    for k = 1:numel(E1.scene.tgt)
        r = {sprintf('P_D, %s', E1.scene.tgt(k).name)};
        for p = 1:3, r{p+1} = sprintf('%.3f $\\pm$ %.3f', S{p}.PDtgt.m(k), S{p}.PDtgt.h(k)); end
        r{5} = pstr(ncr_stats('pairedt', S{3}.run.PDtgt(:, k), S{1}.run.PDtgt(:, k)));
        r{6} = pstr(ncr_stats('pairedt', S{3}.run.PDtgt(:, k), S{2}.run.PDtgt(:, k)));
        rows(end+1, :) = r; %#ok<AGROW>
    end
    rows(end+1, :) = {'Attention correlation $\rho_F$', '--', '--', ...
        sprintf('%.2f $\\pm$ %.2f', S{3}.rho.m, S{3}.rho.h), '', ''};
    rows(end+1, :) = {'CPIs to 90\% of final $\rho_F$', '--', '--', sprintf('%d', S{3}.tConv), '', ''};
    write_tab(fullfile(td, 'tab_main_results.tex'), ...
        sprintf('Closed-loop results, mean $\\pm$ 95\\%% CI over %d runs; paired $t$-test p-values of neuromorphic vs uniform (U) and vs rule-based (R)', E1.Nmc), ...
        'tab:main', {'Metric', 'Uniform', 'Rule-based', 'Neuromorphic', 'p (U)', 'p (R)'}, rows, 'lccccc');
    txt = [txt; {''; 'MAIN RESULTS'}; fmt_rows(rows)];

    % network and energy
    N = S{3};
    rows = {
      'Input neurons (Doppler x range x angle)', sprintf('%d', cfg.d.Nin);
      'Hidden LIF neurons / output neurons', sprintf('%d / %d', cfg.d.Nhid, cfg.d.Nout);
      'Plastic weights (shared)', sprintf('%d', cfg.snn.K * cfg.d.NDc);
      'Time steps per CPI x step', sprintf('%d x %.1f ms', cfg.enc.T, cfg.enc.dtau * 1e3);
      'Input / hidden rate, CPIs 16-64 (Hz)', sprintf('%.2f / %.2f', N.rateIn.m, N.rateHid.m);
      'Hidden rate, CPIs 1-3 (Hz)', sprintf('%.2f', N.rateHid1.m);
      'Synaptic events per CPI', sprintf('%.0f', N.synOps.m);
      'Dense MACs per frame-based pass', sprintf('%d', cfg.d.denseMACs);
      'Energy per CPI: SNN / dense (uJ, indicative)', sprintf('%.3f / %.3f', N.Esnn_uJ.m, N.Edense_uJ);
      'Energy ratio dense / SNN', sprintf('%.1f', N.Edense_uJ / N.Esnn_uJ.m)};
    write_tab(fullfile(td, 'tab_network.tex'), ...
        'Network size, activity and indicative energy (Loihi synaptic-operation and 45 nm FP32 MAC energies)', ...
        'tab:network', {'Quantity', 'Value'}, rows, 'll');
    txt = [txt; {''; 'NETWORK AND ENERGY'}; fmt_rows(rows)];
end

% ---------------------------------------------------------------- ablation
if isfield(X, 'E2')
    E2 = X.E2;
    rows = cell(numel(E2.names), 5);
    for v = 1:numel(E2.names)
        s = E2.res{v}.sum{1};
        rows(v, :) = {E2.names{v}, sprintf('%.3f $\\pm$ %.3f', s.PDlate.m, s.PDlate.h), ...
            sprintf('%.2f $\\pm$ %.2f', s.OSPAlate.m, s.OSPAlate.h), ...
            sprintf('%.2f $\\pm$ %.2f', s.rho.m, s.rho.h), sprintf('%.0f', s.synOps.m)};
    end
    write_tab(fullfile(td, 'tab_ablation.tex'), ...
        sprintf('Ablation of the neuromorphic controller (%d runs each, CPIs 16-64)', E2.res{1}.Nmc), ...
        'tab:ablation', {'Variant', 'P_D', 'OSPA (m)', '$\rho_F$', 'SynOps/CPI'}, rows, 'lcccc');
    txt = [txt; {''; 'ABLATION'}; fmt_rows(rows)];
end

fid = fopen(fullfile(td, 'summary.txt'), 'w');
fprintf(fid, '%s\n', txt{:});
fclose(fid);
fprintf('%s\n', txt{:});
end

function s = pstr(p)
if isnan(p), s = '--';
elseif p < 1e-3, s = '$<$0.001';
else, s = sprintf('%.3f', p); end
end

function write_tab(file, caption, lab, head, rows, align)
fid = fopen(file, 'w');
fprintf(fid, '%% generated by ncr_tables.m (%s)\n', datestr(now));
fprintf(fid, '\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{%s}\n', caption, lab);
fprintf(fid, '\\begin{tabular}{%s}\n\\toprule\n', align);
head = cellfun(@tex_safe, head, 'UniformOutput', false);
fprintf(fid, '%s \\\\\n\\midrule\n', strjoin(head, ' & '));
for i = 1:size(rows, 1)
    r = cellfun(@tex_safe, rows(i, :), 'UniformOutput', false);
    fprintf(fid, '%s \\\\\n', strjoin(r, ' & '));
end
fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
fclose(fid);
end

function s = tex_safe(s)
if any(s == '$'), return; end          % already TeX
s = strrep(s, '%', '\%');
s = strrep(s, 'P_D', '$P_{\mathrm{D}}$');
s = strrep(s, 'R^{cog}', '$R^{\mathrm{cog}}$');
s = strrep(s, 'm^2', 'm$^2$');
end

function out = fmt_rows(rows)
out = cell(size(rows, 1), 1);
for i = 1:size(rows, 1)
    r = regexprep(rows(i, :), '\$\\pm\$', '+/-');
    r = regexprep(r, '[\$\\{}]', '');
    out{i} = sprintf('%-46s %s', r{1}, strjoin(r(2:end), ' | '));
end
end
