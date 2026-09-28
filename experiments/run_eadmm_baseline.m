function exp_result = run_eadmm_baseline(problem, cfg)
%RUN_EADMM_BASELINE Prepare eADMM options and run the SDP baseline once.
%
% Important seed convention:
%   aadmm_3b itself is deterministic for fixed problem/options, so there is
%   no eADMM-solver seed here. Randomness enters only after the SDP solve:
%   (i) vector-clustering rounding uses cfg.seed.eadmm_rounding_base;
%   (ii) the common 2-opt uses cfg.seed.two_opt_base, independently from the
%        rounding seed and shared with Matrix/Random candidate r.

    opts = struct();
    opts.max_iter = cfg.eadmm.max_iter;
    opts.sigma0 = cfg.eadmm.sigma0;
    opts.tol = cfg.eadmm.tol;
    opts.init_mode = cfg.eadmm.init_mode;
    opts.compute_safe_lower_bound = cfg.eadmm.compute_safe_lower_bound;
    opts.met = cfg.eadmm.met;
    opts.rounding_seed_base = cfg.seed.eadmm_rounding_base;
    opts.two_opt_seed_base = cfg.seed.two_opt_base;
    opts.rounding_restarts = cfg.eadmm.rounding_restarts;
    opts.rounding_checkpoints = cfg.eadmm.rounding_checkpoints;
    opts.use_2opt = cfg.eadmm.use_2opt;
    opts.two_opt_tol = cfg.two_opt.cost_tol;
    opts.verbose_2opt = cfg.eadmm.verbose_2opt;
    opts.eval_eta = cfg.matrix.eta;

    wall_tic = tic;
    exp_result = solve_eadmm_batching(problem,opts);
    % Diagnostic only: this wall clock includes MET/LB certification because
    % solve_eadmm_batching computes both branches in one call. It is NOT used
    % in cross-method runtime comparisons.
    exp_result.wall_time_all = toc(wall_tic);
end
