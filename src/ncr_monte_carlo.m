function R = ncr_monte_carlo(cfg, scene, policies, Nmc, label, keepSnap)
%NCR_MONTE_CARLO  Monte Carlo evaluation of one or more policies.
%
%   R = NCR_MONTE_CARLO(cfg, scene, policies, Nmc, label) runs Nmc episodes
%   for every policy in the cell array policies. Run r uses the seed
%   cfg.sim.seed + r for all policies (paired design: identical initial
%   network, interferer timing and scene statistics), which allows paired
%   significance tests. Run 1 keeps snapshots for figures. Uses parfor when
%   cfg.exp.parallel is true and the Parallel Computing Toolbox is present.
%
%   R.logs{r, p}  per-CPI logs (see NCR_EPISODE)
%   R.sum{p}      summary statistics (see NCR_SUMMARISE)
if nargin < 6, keepSnap = true; end
nP = numel(policies);
logs = cell(Nmc, nP);
t0 = tic;
for p = 1:nP
    pol = policies{p};
    usePar = cfg.exp.parallel && ~exist('OCTAVE_VERSION', 'builtin') && ...
             license('test', 'Distrib_Computing_Toolbox');
    if usePar
        L = cell(Nmc, 1);
        parfor r = 1:Nmc
            L{r} = ncr_episode(cfg, scene, pol, cfg.sim.seed + r, keepSnap && r == 1);
        end
        logs(:, p) = L;
    else
        for r = 1:Nmc
            logs{r, p} = ncr_episode(cfg, scene, pol, cfg.sim.seed + r, keepSnap && r == 1);
            if cfg.sim.verbose
                fprintf('  [%s] %-13s run %2d/%d  (%.0f s)\n', label, pol, r, Nmc, toc(t0));
            end
        end
    end
end
R.label = label;  R.policies = policies;  R.Nmc = Nmc;
R.logs = logs;
R.sum = cell(1, nP);
for p = 1:nP
    R.sum{p} = ncr_summarise(cfg, scene, logs(:, p));
end
end
