function X = ncr_experiments(cfg, which)
%NCR_EXPERIMENTS  Experiments reported in the paper.
%
%   X = NCR_EXPERIMENTS(cfg, which) runs the experiments listed in the cell
%   array which (default: all) and saves each to cfg.out.dir/<name>.mat.
%
%   E1 'main'      uniform vs rule-based vs neuromorphic (exp.mainNmc runs)
%   E2 'ablation'  neuromorphic variants (exp.ablNmc runs):
%                    full model; no learning; global (scalar) R-STDP;
%                    rate code without peak selection; latency (TTFS)
%                    code; delta (temporal-contrast) code; 8-bit and 4-bit
%                    synapses; time-domain interference excision
%   E3 'intf'      interferer EIRP sweep for all three policies, plus the
%                    uniform policy with excision (exp.sweepNmc runs)
%   E4 'latency'   time steps per CPI (accuracy vs latency vs energy)
if nargin < 2 || isempty(which), which = {'main', 'ablation', 'intf', 'latency'}; end
if ischar(which), which = {which}; end
if ~exist(cfg.out.dir, 'dir'), mkdir(cfg.out.dir); end
scene = ncr_scene(cfg);
X = struct();
pol3 = {'uniform', 'rule-based', 'neuromorphic'};

if any(strcmpi(which, 'main'))
    say(cfg, 'E1 main comparison');
    E1 = ncr_monte_carlo(cfg, scene, pol3, cfg.exp.mainNmc, 'E1', true);
    E1.cfg = cfg;  E1.scene = scene;
    save_result(cfg, 'E1_main', E1);  X.E1 = E1;
end

if any(strcmpi(which, 'ablation'))
    say(cfg, 'E2 ablation');
    V = { 'Full model',                 {};
          'No learning',                {'lrn.rule', 'none'};
          'Global R-STDP (scalar RPE)', {'lrn.rule', 'global'};
          'Plain rate code',            {'enc.type', 'rate'};
          'Latency code (TTFS)',        {'enc.type', 'latency'};
          'Delta code (contrast)',      {'enc.type', 'delta'};
          '8-bit synapses',             {'lrn.bits', 8};
          '4-bit synapses',             {'lrn.bits', 4};
          'With interference excision', {'intf.mitigation', 'zeroing'} };
    E2.names = V(:, 1)';  E2.res = cell(1, size(V, 1));
    for v = 1:size(V, 1)
        c = variant(cfg, V{v, 2});
        E2.res{v} = ncr_monte_carlo(c, ncr_scene(c), {'neuromorphic'}, cfg.exp.ablNmc, ...
                                    sprintf('E2.%d', v), false);
        E2.res{v}.logs = [];                         % keep summaries only
    end
    save_result(cfg, 'E2_ablation', E2);  X.E2 = E2;
end

if any(strcmpi(which, 'intf'))
    say(cfg, 'E3 interference sweep');
    eirp = cfg.exp.intfEIRP;
    E3.eirp = eirp;  E3.policies = [pol3, {'uniform + excision'}];
    E3.res = cell(numel(eirp), numel(E3.policies));
    for e = 1:numel(eirp)
        c = variant(cfg, {'intf.EIRP_dBm', eirp(e)});
        sc = ncr_scene(c);
        R = ncr_monte_carlo(c, sc, pol3, cfg.exp.sweepNmc, sprintf('E3.%g', eirp(e)), false);
        for p = 1:3, E3.res{e, p} = R.sum{p}; end
        cz = variant(c, {'intf.mitigation', 'zeroing'});
        Rz = ncr_monte_carlo(cz, sc, {'uniform'}, cfg.exp.sweepNmc, sprintf('E3z.%g', eirp(e)), false);
        E3.res{e, 4} = Rz.sum{1};
    end
    save_result(cfg, 'E3_interference', E3);  X.E3 = E3;
end

if any(strcmpi(which, 'latency'))
    say(cfg, 'E4 time-step sweep');
    Tl = cfg.exp.Tsweep;
    E4.T = Tl;  E4.res = cell(1, numel(Tl));
    for k = 1:numel(Tl)
        % keep the encoder's spike budget per CPI constant: pMax * T fixed
        c = variant(cfg, {'enc.T', Tl(k), 'enc.pMax', min(1, cfg.enc.pMax * cfg.enc.T / Tl(k)), ...
                          'enc.dtau', cfg.enc.dtau * cfg.enc.T / Tl(k)});
        R = ncr_monte_carlo(c, ncr_scene(c), {'neuromorphic'}, cfg.exp.sweepNmc, ...
                            sprintf('E4.T%d', Tl(k)), false);
        E4.res{k} = R.sum{1};
    end
    save_result(cfg, 'E4_latency', E4);  X.E4 = E4;
end
end

function c = variant(cfg, nv)
c = cfg;
for i = 1:2:numel(nv)
    c = ncr_set(c, nv{i}, nv{i+1});
end
c = ncr_derive(c);
end

function save_result(cfg, name, S) %#ok<INUSD>
f = fullfile(cfg.out.dir, [name '.mat']);
if exist('OCTAVE_VERSION', 'builtin')
    save('-v7', f, 'S');
else
    save(f, 'S', '-v7.3');
end
end

function say(cfg, msg)
if cfg.sim.verbose, fprintf('\n=== %s (%s) ===\n', msg, datestr(now, 'HH:MM:SS')); end
end
