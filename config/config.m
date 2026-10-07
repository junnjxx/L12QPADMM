function cfg = config()
%CONFIG Single user-facing configuration file for the whole experiment.
%
% Canonical execution order:
%
%   eADMM -> Lp -> MMDLP -> Random -> Vector -> Matrix
%
% restricted to cfg.methods.run.
%
% ========================================================================
% EXPERIMENT PROFILES
% ========================================================================
%
% Supported:
%
%   Batch100_4
%       N = 100
%       batch size = 4
%
%   Batch500_10
%       N = 500
%       batch size = 10
%
%   Batch1000_10
%       N = 1000
%       batch size = 10
%
% Change ONLY cfg.exp.profile below when switching benchmark instances.
%

cfg = struct();

%% ========================================================================
% Experiment profile
% ========================================================================
cfg.exp.profile = "Batch100_4";
%% ========================================================================
% Methods to run
% ========================================================================

cfg.methods.run = ["Matrix","eADMM","Lp","Vector","MMDLP"];
%cfg.methods.run = ["eADMM"];
%% ========================================================================
% Matrix clean mode
% ========================================================================
cfg.matrix.matrix_clean = true;
cfg.matrix.matrix_detail = false;
cfg.matrix.matrix_clean_fast = true;

%% ========================================================================
% Profile-specific problem + Matrix parameters
% ========================================================================

switch string(cfg.exp.profile)

    % ================================================================
    % Batch100_4
    % ================================================================

    case "Batch100_4"

        cfg.data.profile = "custom";

        cfg.data.group_sizes =  [20,25,33,22];
        cfg.data.dim_x = 100;
        cfg.batch.size = 4;
        cfg.matrix.c = 10;
        cfg.matrix.beta0 = 10;
        cfg.matrix.tau_X = 1e-5;
        cfg.matrix.eta_stage_initial = -10;
        cfg.matrix.eta = 0.001;
        cfg.matrix.eta_stagnation_iters = 100;
        cfg.matrix.eta_growth_factor = 1.5;
        cfg.matrix.kabeta = 1000000;

    % ================================================================
    % Batch500_10
    % ================================================================

    case "Batch500_10"

        cfg.data.profile = "custom";
        cfg.data.group_sizes = [48,47,46,44,43,56,54,57,50,55];
        cfg.data.dim_x = 100;
        cfg.batch.size = 10;
        cfg.matrix.c = 10;
        cfg.matrix.beta0 = 10;
        cfg.matrix.tau_X = 1e-5;
        cfg.matrix.eta_stage_initial = -10;
        cfg.matrix.eta = 0.001;
        cfg.matrix.eta_stagnation_iters = 100;
        cfg.matrix.eta_growth_factor = 1.5;
        cfg.matrix.kabeta = 1000000;

    % ================================================================
    % Batch1000_10
    % ================================================================

    case "Batch1000_10"

        cfg.data.profile = "custom";
        cfg.data.group_sizes =  [96,94,97,88,86,107,108,114,100,110];
        cfg.data.dim_x = 100;
        cfg.batch.size = 10;
        cfg.matrix.c = 10;
        cfg.matrix.beta0 = 10;
        cfg.matrix.tau_X = 1e-5;
        cfg.matrix.eta_stage_initial = -10;
        cfg.matrix.eta = 0.005;
        cfg.matrix.eta_stagnation_iters = 100;
        cfg.matrix.eta_growth_factor = 1.5;
        cfg.matrix.kabeta = 1000000;

    otherwise

        error( ...
            'config:UnknownExperimentProfile', ...
            'Unknown cfg.exp.profile = %s.', ...
            string(cfg.exp.profile));

end

%% ========================================================================
% Common problem settings
% ========================================================================

cfg.data.mnist_dir = "";

cfg.data.mnist_per_digit = 10;

cfg.data.mnist_normalize = true;

cfg.data.sparsity_density = 4e-3;

cfg.data.linear_relative_noise = 1e-1;

cfg.batch.allow_truncation = false;

%% ========================================================================
% Matrix solver
% ========================================================================

cfg.matrix.max_iter = 2000000;

cfg.matrix.tol = 1e-6;

%% ========================================================================
% eta continuation
% ========================================================================

cfg.matrix.eta_adapt_enabled = true;

% eta_stage_initial / eta / stagnation / growth factor are set
% inside the experiment profile above.

% eta_max is computed at runtime:
%
%   eta_max = 4*||A||_2*(1+1e-6)

cfg.matrix.eta_max = [];

%% ========================================================================
% Nearest-assignment search diagnostics
% ========================================================================

cfg.matrix.search_assignment_enabled = false;
cfg.matrix.search_round = 1000;
cfg.matrix.search_verbose = true;

%% ========================================================================
% Matrix stopping
% ========================================================================

cfg.matrix.stop_check_interval = 10;

%% ========================================================================
% early_exist
% ========================================================================

cfg.matrix.early_exist = true;

cfg.matrix.early_exist_dxy_tol = 1e-4;

cfg.matrix.early_exist_round_tol = 0.5;

cfg.matrix.early_exist_interval = 10;

%% ========================================================================
% Matrix initialization
% ========================================================================

cfg.matrix.init_mode = "uniform";

cfg.matrix.lp_init_variant = "alg2";



%% ========================================================================
% MEX solver
% ========================================================================

% Current MEX implementation does not yet implement the paper-aligned
% X-first update with tau_X.

cfg.matrix.matrix_mex = false;

%% ========================================================================
% MEX profiling
% ========================================================================

cfg.matrix.mex_profile = false;

cfg.matrix.mex_profile_warmup = 100000;

cfg.matrix.mex_profile_iters = 10000;

cfg.matrix.n_outer = 1;

%% ========================================================================
% Matrix beta strategy
% ========================================================================

% beta0 and kabeta are profile-specific above.

cfg.matrix.beta_mode = "adaptive";

cfg.matrix.use_adaptive_beta = true;

cfg.matrix.beta_adapt_rule = "current";

cfg.matrix.eadmm_beta_decay = 100;

cfg.matrix.eadmm_beta_min = 1e-4;

cfg.matrix.eadmm_beta_max = ...
    cfg.matrix.beta0;

cfg.matrix.eadmm_sigma_decay = 100;

cfg.matrix.eadmm_sigma_min = 1e-4;

cfg.matrix.eadmm_sigma_max = 1e5;

%% ========================================================================
% Matrix diagnostics
% ========================================================================

cfg.matrix.print_every = 100;

cfg.matrix.make_plot = false;

cfg.matrix.save_plot = false;

cfg.matrix.show_plot_title = false;

cfg.matrix.save_result = false;

cfg.matrix.partition_support_tol = 1e-12;

cfg.matrix.figure_dir = ...
    'figures_matrix';

cfg.matrix.multistart.num_starts = 1;

%% ========================================================================
% Matrix variant
% ========================================================================

cfg.matrix.variant = "standard";

cfg.matrix.epsikkt = 1e-3;

cfg.matrix.vc_restarts = 10;

cfg.matrix.vc_seed_base = 9001;

%% ========================================================================
% Lp baseline
% ========================================================================

cfg.lp.run_alg2 = true;

cfg.lp.run_bs = true;

cfg.lp.num_starts = 1;

cfg.lp.p = 0.75;

cfg.lp.eps0 = 0.1;

cfg.lp.eps_min = 1e-3;

cfg.lp.sigma_max = 1e6;

cfg.lp.gamma = 0.9;

cfg.lp.tol_outer = 1e-3;

cfg.lp.alpha0 = 1e-3;

cfg.lp.theta = 1e-4;

cfg.lp.delta = 0.5;

cfg.lp.reference_eta = 0.85;

cfg.lp.tau_x0 = 1e-3;

cfg.lp.tau_f0 = 1e-6;

cfg.lp.tau_x_min = 1e-5;

cfg.lp.tau_f_min = 1e-8;

cfg.lp.sigma_minus = -1;

cfg.lp.use_curvature_sigma0 = true;

%% ========================================================================
% Balanced-polytope projection
% ========================================================================

cfg.lp.proj.tol = 1e-5;
cfg.lp.proj.gap_tol = 1e-5;
cfg.lp.proj.mosek_toolbox_path = ...
    "/home/ubuntu/xlj/mosek/11.2/toolbox/r2019bom";

cfg.lp.proj.mosek_license_file = ...
    "/home/ubuntu/xlj/mosek/mosek.lic";

cfg.lp.proj.mosek_verbose = false;

%% ========================================================================
% Lp numerical safeguards
% ========================================================================

cfg.lp.max_outer = 100;

cfg.lp.max_inner = 500;

cfg.lp.max_backtrack = 60;

cfg.lp.alpha_min = 1e-20;

cfg.lp.alpha_max = 1e20;

cfg.lp.tau_index_shift = 1;

cfg.lp.kkt_map_tol = 1e-8;

cfg.lp.perturb_rho = 0.05;

%% ========================================================================
% Exact accelerated internal N2-best
% ========================================================================

cfg.lp.local_max_swaps = inf;

cfg.lp.improve_tol = 1e-12;

cfg.lp.local_recompute_every = 100;

cfg.lp.local_final_verify = true;

cfg.lp.local_verbose_every = 0;

cfg.lp.local_block = 512;

cfg.lp.do_rounding = true;

cfg.lp.do_internal_local_search = true;

cfg.lp.store_inner_history = false;

cfg.lp.verbose = true;

cfg.lp.strict_paper_checks = true;

% Extra common project-wide postprocessing.
cfg.lp.use_common_2opt = true;

%% ========================================================================
% MMD-LP baseline
% ========================================================================

cfg.mmdlp.verbose = false;

cfg.mmdlp.linprog_display = 'none';

cfg.mmdlp.feasibility_tol = 1e-7;

cfg.mmdlp.store_relaxed_history = false;

cfg.mmdlp.use_common_2opt = true;

% Correctness audit already passed.
% Must remain false during formal timing experiments.
cfg.mmdlp.exact_audit = false;

cfg.mmdlp.exact_audit_max_q = 16;

%% ========================================================================
% Vector baseline
% ========================================================================

cfg.vector.eta = 0.001;

cfg.vector.beta = 100;

cfg.vector.max_iter = 2000;

cfg.vector.epsi = 1e-8;

cfg.vector.print_every = 100;

cfg.vector.seed = 17001;

cfg.vector.use_common_2opt = true;

cfg.vector.fallback = "topk";

%% ========================================================================
% Random baseline
% ========================================================================

cfg.random.num_starts = 100;

%% ========================================================================
% eADMM-SDP
% ========================================================================

cfg.eadmm.max_iter = 20000;

cfg.eadmm.tol = 1e-5;

cfg.eadmm.init_mode = "zero";

cfg.eadmm.rounding_restarts = 10;

cfg.eadmm.sigma0 = 1;

cfg.eadmm.compute_safe_lower_bound = true;

cfg.eadmm.met.enabled = true;

cfg.eadmm.met.max_ineq_per_round = 100;

cfg.eadmm.met.violation_tol = 1e-3;

cfg.eadmm.met.max_rounds = 50;

cfg.eadmm.met.sigma0 = 0.1;

cfg.eadmm.rounding_checkpoints = ...
    [1,10,50,100];

cfg.eadmm.use_2opt = true;

cfg.eadmm.verbose_2opt = false;

%% ========================================================================
% Common 2-opt
% ========================================================================

cfg.two_opt.enabled = true;

cfg.two_opt.cost_tol = 1e-9;

cfg.two_opt.verbose = false;

%% ========================================================================
% Reporting
% ========================================================================

cfg.reporting.time_budgets = ...
    [0.25,0.5,1,5,30,100];

cfg.reporting.print_each_matrix_start = true;

cfg.reporting.print_random_every = 20;

cfg.reporting.print_first_random = 5;

%% ========================================================================
% Output
% ========================================================================

cfg.output.save_results = false;

%% ========================================================================
% Seeds
% ========================================================================

cfg.seed.data = 13;

cfg.seed.matrix_init_base = 1001;

cfg.seed.random_partition_base = 5001;

cfg.seed.eadmm_rounding_base = 9001;

cfg.seed.lp_perturb_base = 11001;

cfg.seed.two_opt_base = 13001;

end