function ncr_figures(cfg, X)
%NCR_FIGURES  Paper and supplementary figures (vector PDF and PNG).
%
%   NCR_FIGURES(cfg, X) draws every figure for which results are present in
%   X (fields E1..E4 from NCR_EXPERIMENTS or NCR_LOAD_RESULTS) and writes
%   them to cfg.out.dir/figures. Sizes follow the IEEE two-column layout
%   (column 8.86 cm, text width 18.2 cm), with 7-8 pt serif fonts.
%
%   Paper figures
%     fig_scene_attention  scenario and attention fields, rule-based vs SNN
%     fig_dynamics         attention agreement and reward, P_D and OSPA per CPI
%     fig_kernel           learned shared Doppler kernel vs analytical motion prior
%     fig_sweeps           interferer EIRP sweep and time steps per CPI
%   Supplementary figures
%     supp_range_doppler, supp_spike_raster, supp_resource_maps,
%     supp_ospa_components, supp_activity_energy, supp_ablation
%
%   MATLAB R2020a+ writes PDFs with exportgraphics (vector content); older
%   releases and GNU Octave use print. Heat maps are drawn as patches
%   (pcolor), so they remain vector graphics in every PDF.
st = fig_style(cfg);
fd = fullfile(cfg.out.dir, 'figures');
if ~exist(fd, 'dir'), mkdir(fd); end
if isfield(X, 'E1')
    E1 = X.E1;  c = E1.cfg;  sc = E1.scene;
    fig_scene_attention(c, sc, E1, st, fd);
    fig_dynamics(c, sc, E1, st, fd);
    fig_kernel(c, sc, E1, st, fd);
    supp_range_doppler(c, E1, st, fd);
    supp_raster(c, E1, st, fd);
    supp_resources(c, E1, st, fd);
    supp_ospa_components(c, E1, st, fd);
    supp_activity(c, E1, st, fd);
end
if isfield(X, 'E3') || isfield(X, 'E4'), fig_sweeps(X, st, fd); end
if isfield(X, 'E2'), supp_ablation(X.E2, st, fd); end
end

%% ===================================================================== style
function st = fig_style(cfg)
st.octave = exist('OCTAVE_VERSION', 'builtin') > 0;
st.uni = [110 109 104] / 255;      % non-cognitive baseline (neutral)
st.alg = [235 104  52] / 255;      % rule-based controller
st.neu = [ 42 120 214] / 255;      % neuromorphic controller
st.exc = [ 27 175 122] / 255;      % uniform + interference excision
st.ink = [31 31 29] / 255;   st.ink2 = [82 81 78] / 255;
st.grid = [220 219 215] / 255;  st.band = [236 235 231] / 255;
st.veg = [93 138 93] / 255;  st.clut = [138 137 133] / 255;
st.fs = 8;  st.fsS = 7;  st.fsXS = 6.5;
st.font = cfg.out.font;
if st.octave, st.font = 'Liberation Serif'; end
st.dpi = cfg.out.dpi;
st.pcol = {st.uni, st.alg, st.neu};
st.pname = {'Uniform', 'Rule-based', 'Neuromorphic'};
st.pls = {'-.', '--', '-'};  st.plw = [1.0 1.1 1.4];
st.viridis = cmap_ramp([0.2670 0.0049 0.3294; 0.2823 0.0950 0.4173; 0.2788 0.1755 0.4834; ...
    0.2590 0.2515 0.5247; 0.2297 0.3224 0.5457; 0.1994 0.3876 0.5546; 0.1727 0.4488 0.5579; ...
    0.1490 0.5081 0.5573; 0.1276 0.5669 0.5506; 0.1206 0.6258 0.5335; 0.1579 0.6838 0.5017; ...
    0.2461 0.7389 0.4520; 0.3692 0.7889 0.3829; 0.5160 0.8312 0.2943; 0.6785 0.8637 0.1895; ...
    0.8456 0.8873 0.0997; 0.9932 0.9062 0.1439], 256);
st.rdbu = cmap_ramp([0.0196 0.1882 0.3804; 0.1273 0.3958 0.6687; 0.2627 0.5765 0.7647; ...
    0.5665 0.7687 0.8685; 0.8196 0.8980 0.9412; 0.9691 0.9665 0.9649; 0.9922 0.8588 0.7804; ...
    0.9546 0.6418 0.5057; 0.8392 0.3765 0.3020; 0.6923 0.0923 0.1677; 0.4039 0.0000 0.1216], 256);
end

function cm = cmap_ramp(k, N)
x = linspace(0, 1, size(k, 1));  xi = linspace(0, 1, N);
cm = [interp1(x, k(:, 1), xi)', interp1(x, k(:, 2), xi)', interp1(x, k(:, 3), xi)'];
end

function f = fig_new(w, h, st)
% w, h in centimetres
f = figure('Visible', 'off', 'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 w h], ...
           'PaperUnits', 'centimeters', 'PaperPosition', [0 0 w h], 'PaperSize', [w h], ...
           'InvertHardcopy', 'off');
set(f, 'DefaultAxesFontName', st.font, 'DefaultTextFontName', st.font, ...
       'DefaultAxesFontSize', st.fsS, 'DefaultTextFontSize', st.fsS);
end

function ax = fig_axes(pos, st, gridAxis)
ax = axes('Position', pos);  hold(ax, 'on');
set(ax, 'FontName', st.font, 'FontSize', st.fsS, 'TickDir', 'out', 'Box', 'off', ...
        'LineWidth', 0.6, 'XColor', st.ink2, 'YColor', st.ink2, 'TickLength', [0.018 0.018]);
if nargin > 2 && ~isempty(gridAxis)
    set(ax, [upper(gridAxis) 'Grid'], 'on', 'GridColor', st.grid, 'GridAlpha', 1, 'GridLineStyle', '-');
end
end

function fig_labels(ax, xl, yl, st)
if ~isempty(xl), xlabel(ax, xl, 'FontSize', st.fs, 'Color', st.ink); end
if ~isempty(yl), ylabel(ax, yl, 'FontSize', st.fs, 'Color', st.ink); end
end

function fig_panel(ax, str, x, y)
if nargin < 3, x = -0.22; end
if nargin < 4, y = 1.05; end
text(ax, x, y, str, 'Units', 'normalized', 'FontSize', 9, 'FontWeight', 'bold', ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
end

function fig_text(f, x, y, str, varargin)
% text in figure-normalised coordinates (an invisible full-figure axes)
ax0 = findobj(f, 'Tag', 'fig_overlay');
if isempty(ax0)
    ax0 = axes('Parent', f, 'Position', [0 0 1 1], 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
               'Tag', 'fig_overlay');
end
text(ax0, x, y, str, varargin{:});
end

function fig_save(f, fd, name, st)
base = fullfile(fd, name);
if ~st.octave && exist('exportgraphics', 'file') > 0          % MATLAB R2020a and later
    exportgraphics(f, [base '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
    exportgraphics(f, [base '.png'], 'Resolution', st.dpi, 'BackgroundColor', 'white');
else
    print(f, [base '.png'], '-dpng', sprintf('-r%d', st.dpi));
    print(f, [base '.pdf'], '-dpdf', '-painters');
end
close(f);
end

function fig_band(ax, x, m, h, col, alpha)
if nargin < 6, alpha = 0.16; end
x = x(:)';  m = m(:)';  h = h(:)';
ok = isfinite(m) & isfinite(h);
if nnz(ok) < 2, return; end
x = x(ok);  m = m(ok);  h = h(ok);
fill(ax, [x fliplr(x)], [m - h, fliplr(m + h)], col, 'EdgeColor', 'none', 'FaceAlpha', alpha);
end

function y = fig_ma(x)
% centred 3-point moving average that ignores NaN (shrinking window at the ends)
x = x(:)';  n = numel(x);  y = nan(1, n);
for i = 1:n
    w = x(max(1, i - 1):min(n, i + 1));  w = w(isfinite(w));
    if ~isempty(w), y(i) = mean(w); end
end
end

function fig_events(ax, sc, yl, ytxt, st)
% interference window (shaded) and cyclist appearance (dotted)
cy = sc.tgt(3).on;
if sc.intf.enable && isfinite(sc.intf.EIRP_dBm)
    on = sc.intf.on;  off = sc.intf.off;
    patch(ax, [on off off on], [yl(1) yl(1) yl(2) yl(2)], st.band, 'EdgeColor', 'none');
    text(ax, (on + cy) / 2, ytxt, 'interf.', 'FontSize', st.fsXS, 'Color', st.ink2, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
end
plot(ax, [cy cy], yl, ':', 'Color', st.ink2, 'LineWidth', 0.7);
text(ax, cy + 1, ytxt, 'cyclist', 'FontSize', st.fsXS, 'Color', st.ink2, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
end

function fig_legend(ax, h, names, pos, st, framed)
if nargin < 6, framed = false; end
lg = legend(ax, h, names);
set(lg, 'Position', pos, 'TextColor', st.ink);
set(lg, 'FontSize', st.fsS - 1.5 * st.octave);     % Octave renders legend text larger
if framed
    set(lg, 'Box', 'on', 'Color', 'w', 'EdgeColor', 'w');
else
    set(lg, 'Box', 'off', 'Color', 'none');
end
end

function cb = fig_colorbar(ax, pos, ticks, labels, txt, st)
% colour bar at a fixed position (the parent axes keeps its size)
p0 = get(ax, 'Position');
horiz = pos(3) > pos(4);
if horiz, loc = 'northoutside'; else, loc = 'eastoutside'; end
cb = colorbar(ax, 'Location', loc);
set(ax, 'Position', p0);
set(cb, 'Position', pos, 'FontSize', st.fsXS, 'LineWidth', 0.5);
if st.octave
    if horiz, set(cb, 'xtick', ticks, 'xticklabel', labels);
    else,     set(cb, 'ytick', ticks, 'yticklabel', labels); end
else
    set(cb, 'Ticks', ticks, 'TickLabels', labels, 'TickDirection', 'out', 'Color', st.ink2);
end
if isempty(txt), return; end
if horiz       % title above the tick labels, placed in the parent axes' normalised units
    text(ax, (pos(1) + pos(3) / 2 - p0(1)) / p0(3), (pos(2) + pos(4) + 0.065 - p0(2)) / p0(4), txt, ...
         'Units', 'normalized', 'FontSize', st.fsS, 'Color', st.ink, 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom');
else
    ylabel(cb, txt, 'FontSize', st.fsS, 'Color', st.ink);
end
end

function fig_fov(ax, Rm, edges, col, lw, sectors)
a = linspace(edges(1), edges(end), 200);
plot(ax, Rm * sind(a), Rm * cosd(a), '-', 'Color', col, 'LineWidth', lw);
if ~sectors, edges = edges([1 end]); end
for e = edges
    plot(ax, [0 Rm * sind(e)], [0 Rm * cosd(e)], '-', 'Color', col, 'LineWidth', lw);
end
end

function fig_arrowhead(ax, p0, p1, len, col)
d = p1 - p0;  d = d / max(norm(d), eps);  n = [-d(2) d(1)];
tip = p1;  b = p1 - len * d;
patch(ax, [tip(1), b(1) + 0.45 * len * n(1), b(1) - 0.45 * len * n(1)], ...
          [tip(2), b(2) + 0.45 * len * n(2), b(2) - 0.45 * len * n(2)], col, 'EdgeColor', 'none');
end

function [XY, names] = fig_truth_tracks(cfg, sc)
% ground-truth positions of every target per CPI (NaN while inactive)
Ns = cfg.sim.Nscans;  nT = numel(sc.tgt);
XY = nan(nT, Ns, 3);  names = {sc.tgt.name};
for t = 1:Ns
    tr = ncr_truth(cfg, sc, t);
    for k = 1:nT
        if tr(k).active, XY(k, t, :) = [tr(k).pos(1), tr(k).pos(2), tr(k).vr]; end
    end
end
end

%% ===================================================================== paper figures
function fig_scene_attention(cfg, sc, E1, st, fd)
L = E1.logs{1, 3};
if ~isfield(L, 'snap'), return; end
want = [2 16 cfg.sim.Nscans];
sel = find(ismember(L.snap.cpi, want));
if isempty(sel), return; end
cp = L.snap.cpi(sel);  n = numel(sel);
Rm = cfg.proc.Rproc;  edges = cfg.ar.secEdges;
[XY, names] = fig_truth_tracks(cfg, sc);
f = fig_new(18.2, 6.5, st);

% ---------------------------------------------------------------- (a) scenario
ax = fig_axes([0.06 0.15 0.27 0.75], st);
fig_fov(ax, Rm, edges, st.grid, 0.5, true);
for j = 1:cfg.ar.M
    a = cfg.d.secCentre(j);
    text(ax, (Rm + 3) * sind(a), (Rm + 3) * cosd(a), sprintf('%d', j), 'FontSize', st.fsXS, ...
         'Color', st.ink2, 'HorizontalAlignment', 'center');
end
plot(ax, sc.veg.p(:, 1), sc.veg.p(:, 2), '.', 'Color', st.veg, 'MarkerSize', 3);
P = sc.stat.p;  nCar = floor((size(P, 1) - 1) / 4);
for q = 1:nCar
    cc = mean(P(4 * q - 3:4 * q, :), 1);
    rectangle(ax, 'Position', [cc(1) - 1, cc(2) - 2.4, 2, 4.8], 'FaceColor', st.clut, 'EdgeColor', 'none');
end
plot(ax, P(end, 1), P(end, 2), '^', 'Color', st.clut, 'MarkerFaceColor', st.clut, 'MarkerSize', 3.5);
text(ax, P(end, 1) + 2, P(end, 2), 'pole', 'FontSize', st.fsXS, 'Color', st.ink2);
ip = sc.intf.pos;
plot(ax, [0 ip(1)], [0 ip(2)], '--', 'Color', st.alg, 'LineWidth', 0.6);
plot(ax, ip(1), ip(2), 'p', 'Color', st.alg, 'MarkerFaceColor', st.alg, 'MarkerSize', 7);
text(ax, ip(1) - 1.5, ip(2) + 1.8, 'interferer', 'FontSize', st.fsXS, 'Color', st.alg, ...
     'HorizontalAlignment', 'right');
off = [1.5 1.2; 1.5 1.0; 1.6 -1.5];
for k = 1:numel(names)
    x = XY(k, :, 1);  y = XY(k, :, 2);  ok = isfinite(x);  x = x(ok);  y = y(ok);
    if numel(x) < 2, continue; end                       % target not active in this sequence
    plot(ax, x, y, '-', 'Color', st.neu, 'LineWidth', 1.3);
    fig_arrowhead(ax, [x(max(1, end - 3)) y(max(1, end - 3))], [x(end) y(end)], 2.6, st.neu);
    plot(ax, x(1), y(1), 'o', 'MarkerSize', 3.5, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', st.neu, 'LineWidth', 0.9);
    text(ax, x(1) + off(min(k, 3), 1), y(1) + off(min(k, 3), 2), names{k}, 'FontSize', st.fsXS, 'Color', st.ink);
end
text(ax, 28, 3.5, 'vegetation', 'FontSize', st.fsXS, 'Color', st.veg * 0.75, 'HorizontalAlignment', 'center');
text(ax, -27, 9.5, 'parked cars', 'FontSize', st.fsXS, 'Color', st.ink2, 'HorizontalAlignment', 'center');
plot(ax, 0, 0, '^', 'Color', st.ink, 'MarkerFaceColor', st.ink, 'MarkerSize', 5);
set(ax, 'DataAspectRatio', [1 1 1], 'XLim', [-47 47], 'YLim', [-1.5 54], ...
        'XTick', -40:20:40, 'YTick', [0 25 50]);
fig_labels(ax, 'x (m)', 'y (m)', st);
fig_text(f, 0.004, 0.975, '(a)', 'FontSize', 9, 'FontWeight', 'bold', 'VerticalAlignment', 'top');

% ---------------------------------------------------------------- (b) attention fields
re = (0:cfg.d.NRc) * cfg.grid.dRc;
te = edges(1) + (0:cfg.d.NAc) * cfg.grid.dAc;
[RR, TT] = ndgrid(re, te);
Xg = RR .* sind(TT);  Yg = RR .* cosd(TT);
F = cell(2, n);
for k = 1:n
    F{1, k} = L.snap.Falg{sel(k)};  F{2, k} = L.snap.Fneu{sel(k)};
end
vmax = max(cellfun(@(A) max([A(:); eps]), F(:)));
rows = {'Rule-based', 'Neuromorphic'};
w = 0.178;  gap = 0.008;  x0 = 0.385;
for r = 1:2
    for k = 1:n
        ax = fig_axes([x0 + (k - 1) * (w + gap), 0.535 - (r - 1) * 0.43, w, 0.39], st);
        if isempty(F{r, k}), continue; end
        C = zeros(size(Xg));  C(1:end - 1, 1:end - 1) = sqrt(F{r, k} / vmax);   % square-root scale
        pcolor(ax, Xg, Yg, C);  shading(ax, 'flat');
        colormap(ax, st.viridis);  caxis(ax, [0 1]);
        fig_fov(ax, Rm, edges, [0.6 0.6 0.59], 0.4, false);
        for q = 1:numel(names)
            p = squeeze(XY(q, cp(k), 1:2));
            if all(isfinite(p))
                plot(ax, p(1), p(2), 'o', 'MarkerSize', 6.5, 'MarkerEdgeColor', 'w', 'LineWidth', 1.0);
                plot(ax, p(1), p(2), 'o', 'MarkerSize', 6.5, 'MarkerEdgeColor', st.alg, 'LineWidth', 0.5);
            end
        end
        set(ax, 'DataAspectRatio', [1 1 1], 'XLim', [-45 45], 'YLim', [-1.5 52], 'XTick', [-40 0 40], ...
                'YTick', [0 25 50], 'TickLength', [0 0], 'XColor', 'w', 'YColor', 'w', 'Layer', 'top');
        if r == 1, title(ax, sprintf('CPI %d', cp(k)), 'FontSize', st.fs, 'FontWeight', 'normal', 'Color', st.ink); end
        if r == 2
            set(ax, 'XColor', st.ink2);  fig_labels(ax, 'x (m)', '', st);
        else
            set(ax, 'XTickLabel', []);
        end
        if k == 1
            set(ax, 'YColor', st.ink2);
            ylabel(ax, rows{r}, 'FontSize', st.fs, 'Color', st.ink);
        else
            set(ax, 'YTickLabel', []);
        end
    end
end
fig_text(f, 0.33, 0.975, '(b)', 'FontSize', 9, 'FontWeight', 'bold', 'VerticalAlignment', 'top');
tv = [0 0.05 0.1 0.2 0.3];  tv = tv(tv <= vmax);
fig_colorbar(ax, [0.94 0.17 0.012 0.66], sqrt(tv / vmax), arrayfun(@(v) sprintf('%.2g', v), tv, ...
             'UniformOutput', false), 'Attention field', st);
fig_save(f, fd, 'fig_scene_attention', st);
end

function fig_dynamics(cfg, sc, E1, st, fd)
t = 1:cfg.sim.Nscans;  Ns = cfg.sim.Nscans;
f = fig_new(18.2, 4.7, st);
w = 0.2555;  xs = [0.06, 0.3998, 0.7396];

% (a) attention agreement and cognitive reward
ax = fig_axes([xs(1) 0.2 w 0.7], st, 'y');
yl = [-0.45 1.15];
fig_events(ax, sc, yl, 1.13, st);
N = E1.sum{3};
fig_band(ax, t, N.rhoCurve.m, N.rhoCurve.h, st.ink2, 0.15);
hh = plot(ax, t, N.rhoCurve.m, '-', 'Color', st.ink, 'LineWidth', 1.3);
for p = 2:3
    S = E1.sum{p};
    fig_band(ax, t, fig_ma(S.RcogCurve.m), fig_ma(S.RcogCurve.h), st.pcol{p}, 0.18);
    hh(end + 1) = plot(ax, t, fig_ma(S.RcogCurve.m), st.pls{p}, 'Color', st.pcol{p}, 'LineWidth', st.plw(p)); %#ok<AGROW>
end
plot(ax, [1 Ns], [0 0], '-', 'Color', st.ink2, 'LineWidth', 0.5);
set(ax, 'XLim', [1 Ns], 'YLim', yl, 'YTick', -0.4:0.2:1);
fig_labels(ax, 'CPI index', 'Correlation / reward', st);
fig_legend(ax, hh, {'\rho_F (SNN vs rule-based field)', 'R^{cog}, rule-based', 'R^{cog}, neuromorphic'}, ...
           [xs(1) + 0.09, 0.56, 0.16, 0.2], st, true);
fig_panel(ax, '(a)', -0.24);

% (b) detection probability and (c) OSPA
spec = {'pdCurve', [0.5 1.04], 1.035, 'P_D (moving targets)'; 'ospaCurve', [0 5.35], 5.3, 'OSPA distance (m)'};
for q = 1:2
    ax = fig_axes([xs(q + 1) 0.2 w 0.7], st, 'y');
    yl = spec{q, 2};
    fig_events(ax, sc, yl, spec{q, 3}, st);
    hh = gobjects_(3);
    for p = 1:3
        S = E1.sum{p}.(spec{q, 1});
        fig_band(ax, t, fig_ma(S.m), fig_ma(S.h), st.pcol{p}, 0.15);
    end
    for p = 1:3
        S = E1.sum{p}.(spec{q, 1});
        hh(p) = plot(ax, t, fig_ma(S.m), st.pls{p}, 'Color', st.pcol{p}, 'LineWidth', st.plw(p));
    end
    set(ax, 'XLim', [1 Ns], 'YLim', yl);
    fig_labels(ax, 'CPI index', spec{q, 4}, st);
    if q == 1
        fig_legend(ax, hh([3 2 1]), st.pname([3 2 1]), [xs(2) + 0.14, 0.24, 0.11, 0.2], st, true);
    end
    fig_panel(ax, sprintf('(%c)', 'a' + q), -0.22);
end
fig_save(f, fd, 'fig_dynamics', st);
end

function h = gobjects_(n)
% placeholder graphics array (gobjects is not available in GNU Octave)
if exist('OCTAVE_VERSION', 'builtin'), h = zeros(1, n); else, h = gobjects(1, n); end
end

function fig_kernel(cfg, sc, E1, st, fd)
Nmc = size(E1.logs, 1);  Ns = cfg.sim.Nscans;
if ~isfield(E1.logs{1, 3}, 'kernel'), return; end
K = zeros(cfg.d.NDc, Ns, Nmc);
for r = 1:Nmc, K(:, :, r) = E1.logs{r, 3}.kernel / cfg.snn.wMax; end
v = cfg.d.vCoarse(:);  dv = v(2) - v(1);  w0 = cfg.snn.w0 / cfg.snn.wMax;
sel = abs(v) <= 10;  vs = v(sel);
dW = mean(K, 3) - w0;  D = dW(sel, :);
vmin = min(min(D(:)), -0.05);  vmax = max(max(D(:)), 0.05);
Z = D;  Z(D < 0) = D(D < 0) / abs(vmin);  Z(D > 0) = D(D > 0) / vmax;   % two-slope map, white at zero
f = fig_new(8.86, 6.3, st);

ax = fig_axes([0.14 0.17 0.555 0.585], st);
C = zeros(numel(vs) + 1, Ns + 1);  C(1:end - 1, 1:end - 1) = Z;
pcolor(ax, 0.5:1:Ns + 0.5, [vs - dv / 2; vs(end) + dv / 2], C);  shading(ax, 'flat');
colormap(ax, st.rdbu);  caxis(ax, [-1 1]);
[XY, names] = fig_truth_tracks(cfg, sc);
lab = lower(names);
for k = 1:numel(names)
    vr = XY(k, :, 3);  i1 = find(isfinite(vr), 1);
    if isempty(i1), continue; end
    plot(ax, 1:Ns, vr, '--', 'Color', st.ink, 'LineWidth', 0.7);
    if k == 3, ha = 'right'; xt = i1 - 1; else, ha = 'left'; xt = i1 + 1; end
    text(ax, xt, vr(i1) + 0.45, lab{k}, 'FontSize', st.fsXS, 'Color', st.ink, 'HorizontalAlignment', ha, ...
         'VerticalAlignment', 'bottom', 'BackgroundColor', 'w', 'Margin', 0.5);
end
set(ax, 'XLim', [0.5 Ns + 0.5], 'YLim', [-10 10], 'XTick', [1 16 32 48 64], 'Layer', 'top');
fig_labels(ax, 'CPI index', 'Radial velocity (m/s)', st);
tv = [-0.6 -0.3 0 0.15 0.3];  tv = tv(tv >= vmin & tv <= vmax);
tp = tv;  tp(tv < 0) = tv(tv < 0) / abs(vmin);  tp(tv > 0) = tv(tv > 0) / vmax;
fig_colorbar(ax, [0.14 0.865 0.555 0.035], tp, arrayfun(@(x) sprintf('%g', x), tv, 'UniformOutput', false), ...
             'Weight change \DeltaW/w_{max} since initialisation', st);

% profile at the first and last CPI (mean and 95 % CI over runs)
ax2 = fig_axes([0.745 0.17 0.235 0.585], st, 'x');
cols = {1 - 0.45 * (1 - st.neu), st.neu};  idx = [1 Ns];  lw = [1.0 1.4];
hh = gobjects_(3);
for q = 1:2
    [m, h] = ncr_stats('ci', squeeze(K(:, idx(q), :)), 2);
    fill(ax2, [m - h; flipud(m + h)]', [v; flipud(v)]', cols{q}, 'EdgeColor', 'none', 'FaceAlpha', 0.25);
    hh(q) = plot(ax2, m, v, '-', 'Color', cols{q}, 'LineWidth', lw(q));
end
sig = max(cfg.ctl.sigV, cfg.d.lambda / (2 * cfg.d.nUni * cfg.wf.Tc) / 2);
vv = linspace(-10, 10, 400);
hh(3) = plot(ax2, 1 - exp(-vv.^2 / (2 * sig^2)), vv, '--', 'Color', st.alg, 'LineWidth', 1.1);
plot(ax2, [w0 w0], [-10 10], ':', 'Color', st.ink2, 'LineWidth', 0.6);
set(ax2, 'XLim', [0 1.05], 'YLim', [-10 10], 'XTick', [0 0.5 1], 'YTickLabel', []);
fig_labels(ax2, 'W/w_{max}', '', st);
fig_legend(ax2, hh, {'SNN, CPI 1', sprintf('SNN, CPI %d', Ns), 'clutter-notch prior'}, ...
           [0.72 0.78 0.27 0.15], st);
fig_save(f, fd, 'fig_kernel', st);
end

function fig_sweeps(X, st, fd)
f = fig_new(18.2, 4.6, st);
w = 0.2366;  xs = [0.06, 0.3816, 0.7032];
cols = {st.uni, st.alg, st.neu, st.exc};  ls = {'-.', '--', '-', ':'};  mk = {'s', 'd', 'o', '^'};
pn = {'Uniform', 'Rule-based', 'Neuromorphic', 'Uniform + excision'};
if isfield(X, 'E3')
    E3 = X.E3;  e = E3.eirp;  ne = numel(e);
    lab = arrayfun(@(x) sprintf('%g', x), e, 'UniformOutput', false);  lab(~isfinite(e)) = {'off'};
    spec = {'PDvehIntf', [0.45 1.04], 'Vehicle P_D (CPIs 20-44)'; 'FA', [0 4.8], 'False clusters per CPI'};
    for q = 1:2
        ax = fig_axes([xs(q) 0.22 w 0.68], st, 'y');
        hh = gobjects_(4);
        for p = 1:size(E3.res, 2)
            Y = cellfun(@(S) S.(spec{q, 1}).m, E3.res(:, p))';
            H = cellfun(@(S) S.(spec{q, 1}).h, E3.res(:, p))';
            x = (1:ne) + (p - 2.5) * 0.09;
            for i = 1:ne
                if isfinite(H(i)), plot(ax, [x(i) x(i)], Y(i) + [-1 1] * H(i), '-', 'Color', cols{p}, 'LineWidth', 0.7); end
            end
            hh(p) = plot(ax, x, Y, ls{p}, 'Color', cols{p}, 'LineWidth', 1.2, 'Marker', mk{p}, ...
                         'MarkerFaceColor', cols{p}, 'MarkerEdgeColor', 'w', 'MarkerSize', 4);
        end
        set(ax, 'XTick', 1:ne, 'XTickLabel', lab, 'XLim', [0.65 ne + 0.35], 'YLim', spec{q, 2});
        fig_labels(ax, 'Interferer EIRP (dBm)', spec{q, 3}, st);
        if q == 1, fig_legend(ax, hh, pn(1:numel(hh)), [xs(1) + 0.012, 0.25, 0.1, 0.28], st); end
        fig_panel(ax, sprintf('(%c)', 'a' + q - 1), -0.22);
    end
end
if isfield(X, 'E4')
    E4 = X.E4;  T = E4.T;
    pos = [xs(3) 0.22 w 0.68];
    Y = cellfun(@(S) S.PDlate.m, E4.res);  H = cellfun(@(S) S.PDlate.h, E4.res);
    syn = cellfun(@(S) S.synOps.m, E4.res) / 1e3;
    ax = fig_axes(pos, st, 'y');
    for i = 1:numel(T)
        if isfinite(H(i)), plot(ax, [T(i) T(i)], Y(i) + [-1 1] * H(i), '-', 'Color', st.neu, 'LineWidth', 0.7); end
    end
    h1 = plot(ax, T, Y, '-o', 'Color', st.neu, 'MarkerFaceColor', st.neu, 'MarkerEdgeColor', 'w', ...
              'MarkerSize', 4.5, 'LineWidth', 1.4);
    h2 = plot(ax, NaN, NaN, '--s', 'Color', st.ink2, 'MarkerFaceColor', 'w', 'MarkerSize', 4, 'LineWidth', 1.0);
    xl = [T(1) * 0.8, T(end) * 1.25];
    set(ax, 'XScale', 'log', 'XLim', xl, 'XTick', T, 'XTickLabel', arrayfun(@num2str, T, 'UniformOutput', false), ...
            'XMinorTick', 'off', 'YLim', [0.8 1.0]);
    fig_labels(ax, 'Network time steps T per CPI', 'P_D (CPIs 16-64)', st);
    fig_legend(ax, [h1 h2], {'P_D', 'synaptic events'}, [xs(3) + 0.1, 0.25, 0.12, 0.17], st);
    fig_panel(ax, '(c)', -0.22);
    ax2 = axes('Position', pos, 'Color', 'none', 'YAxisLocation', 'right', 'XScale', 'log', 'XLim', xl, ...
               'XTick', [], 'Box', 'off', 'FontName', st.font, 'FontSize', st.fsS, 'TickDir', 'out', ...
               'LineWidth', 0.6, 'YColor', st.ink2, 'XColor', 'none', 'TickLength', [0.018 0.018]);
    hold(ax2, 'on');
    plot(ax2, T, syn, '--s', 'Color', st.ink2, 'MarkerFaceColor', 'w', 'MarkerSize', 4, 'LineWidth', 1.0);
    set(ax2, 'YLim', [0 2]);
    ylabel(ax2, 'Events per CPI (\times10^3)', 'FontSize', st.fs, 'Color', st.ink2);
end
fig_save(f, fd, 'fig_sweeps', st);
end

%% ===================================================================== supplementary figures
function supp_range_doppler(cfg, E1, st, fd)
L = {E1.logs{1, 1}, E1.logs{1, 3}};      % uniform and neuromorphic, run 1
if ~isfield(L{2}, 'snap') || ~isfield(L{1}, 'snap') || ~isfield(L{2}.snap, 'cube'), return; end
cp = L{2}.snap.cubeCPI;
k1 = min(2, numel(cp));  kN = numel(cp);
tr = ncr_truth(cfg, E1.scene, cp(k1));  jV = tr(1).sector;
trN = ncr_truth(cfg, E1.scene, cp(kN));  jP = trN(2).sector;
f = fig_new(18.2, 9.5, st);
cases = {k1, jV, 'Vehicle sector'; kN, jP, 'Pedestrian sector'};
names = {'uniform', 'neuromorphic'};
for row = 1:2
    for col = 1:2
        snap = L{col}.snap;
        if size(snap.cube, 1) < cases{row, 1} || size(snap.cube, 2) < cases{row, 2} || ...
           isempty(snap.cube{cases{row, 1}, cases{row, 2}})
            continue
        end
        cb = snap.cube{cases{row, 1}, cases{row, 2}};
        ax = fig_axes([0.08 + (col - 1) * 0.47, 0.58 - (row - 1) * 0.47, 0.33, 0.34], st);
        imagesc(ax, cb.v, cb.r, max(double(cb.PdB), [], 3));  axis(ax, 'xy');
        colormap(ax, st.viridis);  caxis(ax, [0 40]);
        set(ax, 'XLim', [-12 12], 'YLim', [0 cfg.proc.Rproc], 'Layer', 'top');
        title(ax, sprintf('%s, CPI %d, %s: %d chirps, %.0f MHz', cases{row, 3}, snap.cubeCPI(cases{row, 1}), ...
              names{col}, cb.nj, cb.bj / 1e6), 'FontSize', st.fsS, 'FontWeight', 'normal');
        if row == 2, fig_labels(ax, 'Radial velocity (m/s)', '', st); end
        if col == 1, fig_labels(ax, '', 'Range (m)', st); end
        p = get(ax, 'Position');
        fig_colorbar(ax, [p(1) + p(3) + 0.01, p(2), 0.01, p(4)], 0:10:40, {'0', '10', '20', '30', '40'}, ...
                     'Power above noise (dB)', st);
    end
end
fig_save(f, fd, 'supp_range_doppler', st);
end

function supp_raster(cfg, E1, st, fd)
L = E1.logs{1, 3};
if ~isfield(L, 'snap') || ~isfield(L.snap, 'rec') || numel(L.snap.rec) < 2, return; end
f = fig_new(18.2, 8.0, st);
titles = {'Input events per sector', 'Hidden spikes per sector', 'Output (sector) neurons'};
for row = 1:2
    rec = L.snap.rec{row};
    data = {rec.in, rec.hid, double(rec.out)};
    for col = 1:3
        ax = fig_axes([0.06 + (col - 1) * 0.33, 0.58 - (row - 1) * 0.47, 0.24, 0.33], st);
        if col < 3
            imagesc(ax, 1:cfg.enc.T, 1:cfg.ar.M, data{col}');  axis(ax, 'xy');
            colormap(ax, st.viridis);
            p = get(ax, 'Position');  mx = max([data{col}(:); 1]);
            fig_colorbar(ax, [p(1) + p(3) + 0.008, p(2), 0.008, p(4)], [0 mx], {'0', sprintf('%g', mx)}, '', st);
        else
            [tt, jj] = find(rec.out);
            plot(ax, tt, jj, '|', 'Color', st.neu, 'MarkerSize', 6, 'LineWidth', 1.2);
        end
        set(ax, 'XLim', [0.5 cfg.enc.T + 0.5], 'YLim', [0.5 cfg.ar.M + 0.5], 'YTick', 1:cfg.ar.M, 'Layer', 'top');
        title(ax, sprintf('%s, CPI %d', titles{col}, L.snap.recCPI(row)), 'FontSize', st.fsS, 'FontWeight', 'normal');
        if row == 2, fig_labels(ax, 'Time step', '', st); end
        if col == 1, fig_labels(ax, '', 'Sector', st); end
    end
end
fig_save(f, fd, 'supp_spike_raster', st);
end

function supp_resources(cfg, E1, st, fd)
f = fig_new(18.2, 9.0, st);
t = 1:cfg.sim.Nscans;  scs = cfg.d.secCentre;
pol = [2 3];  nm = {'rule-based', 'neuromorphic'};
for r = 1:2
    L = E1.logs{1, pol(r)};
    ax = fig_axes([0.08, 0.58 - (r - 1) * 0.46, 0.34, 0.33], st);
    imagesc(ax, t, scs, L.n');  axis(ax, 'xy');  colormap(ax, st.viridis);
    caxis(ax, [cfg.bud.nMin cfg.bud.nMax]);
    p = get(ax, 'Position');
    fig_colorbar(ax, [p(1) + p(3) + 0.01, p(2), 0.01, p(4)], [cfg.bud.nMin 64 cfg.bud.nMax], ...
                 {sprintf('%d', cfg.bud.nMin), '64', sprintf('%d', cfg.bud.nMax)}, 'Chirps', st);
    title(ax, sprintf('Dwell (transmit energy), %s', nm{r}), 'FontSize', st.fsS, 'FontWeight', 'normal');
    fig_labels(ax, '', 'Sector centre (deg)', st);
    if r == 2, fig_labels(ax, 'CPI index', '', st); end
    ax = fig_axes([0.57, 0.58 - (r - 1) * 0.46, 0.34, 0.33], st);
    imagesc(ax, t, scs, L.b' / 1e6);  axis(ax, 'xy');  colormap(ax, st.viridis);
    caxis(ax, [cfg.wf.Bmin cfg.wf.Bmax] / 1e6);
    p = get(ax, 'Position');  bt = [cfg.wf.Bmin cfg.wf.Bmax] / 1e6;
    fig_colorbar(ax, [p(1) + p(3) + 0.01, p(2), 0.01, p(4)], bt, arrayfun(@(x) sprintf('%g', x), bt, ...
                 'UniformOutput', false), 'Bandwidth (MHz)', st);
    title(ax, sprintf('Sweep bandwidth, %s', nm{r}), 'FontSize', st.fsS, 'FontWeight', 'normal');
    if r == 2, fig_labels(ax, 'CPI index', '', st); end
end
fig_save(f, fd, 'supp_resource_maps', st);
end

function supp_ospa_components(cfg, E1, st, fd)
f = fig_new(8.86, 6.0, st);
ax = fig_axes([0.15 0.14 0.8 0.74], st, 'y');
Y = zeros(2, 3);  H = Y;
for p = 1:3
    Y(:, p) = [E1.sum{p}.OSPAloc.m; E1.sum{p}.OSPAcard.m];
    H(:, p) = [E1.sum{p}.OSPAloc.h; E1.sum{p}.OSPAcard.h];
end
w = 0.8 / 3;
for g = 1:2
    for s = 1:3
        x0 = g - 0.4 + (s - 1) * w;
        patch(ax, [x0 x0 + w * 0.9 x0 + w * 0.9 x0], [0 0 Y(g, s) Y(g, s)], st.pcol{s}, 'EdgeColor', 'none');
        if isfinite(H(g, s))
            plot(ax, [1 1] * (x0 + w * 0.45), Y(g, s) + [-1 1] * H(g, s), '-', 'Color', st.ink, 'LineWidth', 0.7);
        end
    end
end
top = max(Y(:) + max(H(:), 0));
if ~isfinite(top) || top <= 0, top = 1; end
set(ax, 'XTick', 1:2, 'XTickLabel', {'Localisation', 'Cardinality'}, 'XLim', [0.4 2.6], 'YLim', [0 top * 1.35]);
fig_labels(ax, '', sprintf('OSPA component (m), CPIs %d-%d', cfg.exp.late, cfg.sim.Nscans), st);
hb = gobjects_(3);
for p = 1:3, hb(p) = patch(ax, NaN, NaN, st.pcol{p}, 'EdgeColor', 'none'); end
fig_legend(ax, hb, st.pname, [0.55 0.68 0.35 0.18], st);
fig_save(f, fd, 'supp_ospa_components', st);
end

function supp_activity(cfg, E1, st, fd)
S = E1.sum{3};  t = 1:cfg.sim.Nscans;
f = fig_new(18.2, 6.0, st);
ax = fig_axes([0.07 0.2 0.38 0.72], st, 'y');
fig_band(ax, t, S.rateInCurve.m, S.rateInCurve.h, st.ink2);
fig_band(ax, t, S.rateHidCurve.m, S.rateHidCurve.h, st.neu);
h1 = plot(ax, t, S.rateInCurve.m, '-', 'Color', st.ink2, 'LineWidth', 1.2);
h2 = plot(ax, t, S.rateHidCurve.m, '-', 'Color', st.neu, 'LineWidth', 1.4);
set(ax, 'XLim', [1 cfg.sim.Nscans]);
fig_labels(ax, 'CPI index', 'Mean firing rate per neuron (Hz)', st);
fig_legend(ax, [h1 h2], {'Input layer', 'Hidden layer (LIF)'}, [0.3 0.7 0.14 0.15], st);
ax = fig_axes([0.58 0.2 0.38 0.72], st, 'y');
fig_band(ax, t, S.EsnnCurve.m, S.EsnnCurve.h, st.neu);
h1 = plot(ax, t, S.EsnnCurve.m, '-', 'Color', st.neu, 'LineWidth', 1.4);
h2 = plot(ax, [1 cfg.sim.Nscans], S.Edense_uJ * [1 1], '--', 'Color', st.ink2, 'LineWidth', 1.2);
set(ax, 'YScale', 'log', 'XLim', [1 cfg.sim.Nscans], 'YLim', [3e-3 3]);
fig_labels(ax, 'CPI index', 'Energy per CPI (\muJ, indicative)', st);
fig_legend(ax, [h1 h2], {'SNN (synaptic events + updates)', 'Dense frame-based pass'}, [0.62 0.28 0.25 0.15], st);
fig_save(f, fd, 'supp_activity_energy', st);
end

function supp_ablation(E2, st, fd)
nv = numel(E2.names);
m = @(fld) cellfun(@(R) R.sum{1}.(fld).m, E2.res);
h = @(fld) cellfun(@(R) R.sum{1}.(fld).h, E2.res);
f = fig_new(18.2, 7.5, st);
flds = {'PDlate', 'OSPAlate', 'Esnn_uJ'};
xl = {'P_D (CPIs 16-64)', 'OSPA (m)', 'SNN energy per CPI (\muJ)'};
for q = 1:3
    ax = fig_axes([0.17 + (q - 1) * 0.28, 0.16, 0.24, 0.8], st, 'x');
    Y = m(flds{q});  H = h(flds{q});
    for k = 1:nv
        c = [0.72 0.72 0.70];  if k == 1, c = st.neu; end
        patch(ax, [0 Y(k) Y(k) 0], [k - 0.35 k - 0.35 k + 0.35 k + 0.35], c, 'EdgeColor', 'none');
        if isfinite(H(k)), plot(ax, Y(k) + [-1 1] * H(k), [k k], '-', 'Color', st.ink, 'LineWidth', 0.7); end
    end
    set(ax, 'YDir', 'reverse', 'YTick', 1:nv, 'YLim', [0.4 nv + 0.6]);
    if q == 1, set(ax, 'YTickLabel', E2.names, 'XLim', [0 1]); else, set(ax, 'YTickLabel', {}); end
    fig_labels(ax, xl{q}, '', st);
end
fig_save(f, fd, 'supp_ablation', st);
end
