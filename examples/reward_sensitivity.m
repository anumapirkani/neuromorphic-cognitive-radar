%REWARD_SENSITIVITY  Sensitivity of learning to the reward values (Section III-C).
%
%   Runs the neuromorphic controller with the seeds of the ablation study
%   (cfg.exp.ablNmc runs) for the default reward and four variants, and prints
%   the post-learning P_D, its paired difference to the default, the field
%   correlation rho_F and the hidden firing rate.
%
%   Run from the repository root:  run('examples/reward_sensitivity.m')

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src'));
V = { 'default',               {};
      'stationary -0.25',      {'rew.static', -0.25};
      'stationary -1.0',       {'rew.static', -1.0};
      'unconfirmed 0',         {'rew.unc', 0};
      'empty cells 0',         {'rew.idle', 0} };
cfg0 = ncr_config('sim.verbose', false);
run0 = [];
fprintf('%-18s %8s %8s %6s %6s %9s\n', 'Variant', 'P_D', 'dP_D', 'p', 'rho_F', 'rate (Hz)');
for v = 1:size(V, 1)
    c = ncr_config('sim.verbose', false, V{v, 2}{:});
    R = ncr_monte_carlo(c, ncr_scene(c), {'neuromorphic'}, cfg0.exp.ablNmc, V{v, 1}, false);
    r = R.sum{1}.run;
    if v == 1, run0 = r; end
    d = r.PDlate - run0.PDlate;  n = numel(d);
    if v == 1 || all(d == 0)
        p = 1;
    else
        t = mean(d) / (std(d) / sqrt(n));
        p = betainc((n - 1) / (n - 1 + t^2), (n - 1) / 2, 0.5);   % two-sided paired t-test
    end
    fprintf('%-18s %8.3f %+8.3f %6.2f %6.3f %9.2f\n', V{v, 1}, mean(r.PDlate), mean(d), p, ...
            mean(r.rho), mean(r.rateHid));
end
