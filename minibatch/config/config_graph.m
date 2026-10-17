function cfg = config_graph()
%CONFIG_GRAPH Independent configuration for ADMM-GP graph partition.
%
% No dependency on config.m or minibatch experiment profiles.
%
% Graph objective:
%   L = diag(sum(W,2)) - W
%   J_common(P) = 0.5 * <L,P*P'>
%
% Matrix:
%   A = L, G = 0
%
% Shared evaluation:
%   Phi = bs^2 * L

cfg = struct();

%% ============================================================
% 1. Graph dataset
% ============================================================

cfg.problem.type = "graph";

% Supported: "rand20", "rand50", "rand80"
cfg.graph.family = "rand20";

% Supported: 100,200,...,1000
cfg.graph.n = 100;

% Number of partitions, must divide n
cfg.graph.k = 10;

cfg.graph.instance_index = 1;
cfg.graph.data_subdir = "admm_gp";

%% ============================================================
% 2. Methods
% ============================================================

cfg.methods.run = ...
    ["Matrix"];

% Examples:
% cfg.methods.run = ["Matrix"];
% cfg.methods.run = ["eADMM","Matrix"];
% cfg.methods.run = ["Lp","Matrix"];

%% ============================================================
% 3. Derived graph dimensions
% ============================================================

n = cfg.graph.n;
k = cfg.graph.k;

if ~ismember(n,100:100:1000)
    error('Graph n must be 100,200,...,1000.');
end

if ~isscalar(k) || k<2 || k~=round(k) || mod(n,k)~=0
    error('Graph k must be an integer >=2 dividing n.');
end

bs = n/k;

cfg.data.profile = ...
    sprintf('ADMMGP_%s_N%d_k%d', ...
    char(cfg.graph.family),n,k);

cfg.data.num_samples = n;
cfg.data.num_groups = k;
cfg.data.group_sizes = bs*ones(1,k);

cfg.batch.size = bs;
cfg.batch.num_used_samples = n;
cfg.batch.num_batches = k;
cfg.batch.allow_truncation = false;

%% ============================================================
% 4. Matrix solver
% ============================================================

% Core parameters: graph-specific, independent of minibatch config.
% Initial trial values; not yet tuned for the Laplacian scale.
cfg.matrix.c = 10000;
cfg.matrix.beta0 = 10000;
cfg.matrix.tau_X = 1e-5;
cfg.matrix.max_iter = 20000;
cfg.matrix.tol = 1e-10;
% Implementation / diagnostics
cfg.matrix.matrix_clean = false;
cfg.matrix.matrix_detail = true;
cfg.matrix.matrix_clean_fast = true;
cfg.matrix.matrix_mex = false;
cfg.matrix.mex_profile = false;
cfg.matrix.mex_profile_warmup = 100000;
cfg.matrix.mex_profile_iters = 10000;

% eta continuation
cfg.matrix.eta_adapt_enabled = false;
cfg.matrix.eta_stage_initial = 5;
cfg.matrix.eta =100;
cfg.matrix.eta_growth_factor = 1.5;

% Computed inside solve_matrix_batching.m:
% eta_max = 4*norm(A,2)*(1+1e-6)
cfg.matrix.eta_max = [];

% beta strategy
cfg.matrix.beta_mode = "fixed";
cfg.matrix.use_adaptive_beta = true;
cfg.matrix.beta_adapt_rule = "current";
cfg.matrix.kabeta = 1000000;

cfg.matrix.eadmm_beta_decay = 100;
cfg.matrix.eadmm_beta_min = 1e-4;
cfg.matrix.eadmm_beta_max = cfg.matrix.beta0;

cfg.matrix.eadmm_sigma_decay = 100;
cfg.matrix.eadmm_sigma_min = 1e-4;
cfg.matrix.eadmm_sigma_max = 1e5;

% Stopping
cfg.matrix.stop_check_interval = 10;

cfg.matrix.early_exist = true;
cfg.matrix.early_exist_dxy_tol = 1e-4;
cfg.matrix.early_exist_round_tol = 0.5;
cfg.matrix.early_exist_interval = 10;

% Initialization
cfg.matrix.init_mode = "uniform";
cfg.matrix.lp_init_variant = "alg2";
cfg.matrix.multistart.num_starts = 1;

% Optional assignment diagnostics
cfg.matrix.search_assignment_enabled = false;
cfg.matrix.search_round = 1000;
cfg.matrix.search_verbose = true;

% Other solver settings
cfg.matrix.n_outer = 1;
cfg.matrix.variant = "standard";
cfg.matrix.epsikkt = 1e-3;
cfg.matrix.vc_restarts = 10;
cfg.matrix.vc_seed_base = 9001;

cfg.matrix.partition_support_tol = 1e-12;

% Reporting and outputs
cfg.matrix.print_every = 100;
cfg.matrix.make_plot = false;
cfg.matrix.save_plot = false;
cfg.matrix.show_plot_title = false;
cfg.matrix.save_result = false;
cfg.matrix.figure_dir = "figures_matrix_graph";

%% ============================================================
% 5. Lp baseline
% ============================================================

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

% Projection
cfg.lp.proj.tol = 1e-5;
cfg.lp.proj.gap_tol = 1e-5;

cfg.lp.proj.mosek_toolbox_path = ...
    "/home/ubuntu/xlj/mosek/11.2/toolbox/r2019bom";

cfg.lp.proj.mosek_license_file = ...
    "/home/ubuntu/xlj/mosek/mosek.lic";

cfg.lp.proj.mosek_verbose = false;

% Numerical safeguards
cfg.lp.max_outer = 100;
cfg.lp.max_inner = 500;
cfg.lp.max_backtrack = 60;

cfg.lp.alpha_min = 1e-20;
cfg.lp.alpha_max = 1e20;
cfg.lp.tau_index_shift = 1;
cfg.lp.kkt_map_tol = 1e-8;
cfg.lp.perturb_rho = 0.05;

% Internal N2
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

cfg.lp.use_common_2opt = true;

%% ============================================================
% 6. MMD-LP
% ============================================================

cfg.mmdlp.verbose = false;
cfg.mmdlp.linprog_display = 'none';

cfg.mmdlp.feasibility_tol = 1e-7;
cfg.mmdlp.store_relaxed_history = false;
cfg.mmdlp.use_common_2opt = true;

cfg.mmdlp.exact_audit = false;
cfg.mmdlp.exact_audit_max_q = 16;

%% ============================================================
% 7. Vector baseline
% ============================================================

cfg.vector.eta = 0.001;
cfg.vector.beta = 100;

cfg.vector.max_iter = 2000;
cfg.vector.epsi = 1e-8;

cfg.vector.print_every = 100;
cfg.vector.seed = 17001;

cfg.vector.use_common_2opt = true;
cfg.vector.fallback = "topk";

%% ============================================================
% 8. Random baseline
% ============================================================

cfg.random.num_starts = 100;

%% ============================================================
% 9. eADMM-SDP
% ============================================================

cfg.eadmm.max_iter = 20000;
cfg.eadmm.tol = 1e-5;
cfg.eadmm.init_mode = "zero";
cfg.eadmm.sigma0 = 1;

% Generate R raw solutions, then 2-opt only the best raw solution.
cfg.eadmm.rounding_restarts = 10;
cfg.eadmm.rounding_checkpoints = [1,10,50,100];

cfg.eadmm.use_2opt = true;
cfg.eadmm.verbose_2opt = false;

% Certification
cfg.eadmm.compute_safe_lower_bound = true;

% MET strengthening
cfg.eadmm.met.enabled = true;
cfg.eadmm.met.max_ineq_per_round = 100;
cfg.eadmm.met.violation_tol = 1e-3;
cfg.eadmm.met.max_rounds = 50;
cfg.eadmm.met.sigma0 = 0.1;

%% ============================================================
% 10. Common 2-opt
% ============================================================

cfg.two_opt.enabled = true;
cfg.two_opt.cost_tol = 1e-9;
cfg.two_opt.verbose = false;

%% ============================================================
% 11. Reporting
% ============================================================

cfg.reporting.time_budgets = [0.25,0.5,1,5,30,100];

cfg.reporting.print_each_matrix_start = true;
cfg.reporting.print_random_every = 20;
cfg.reporting.print_first_random = 5;

%% ============================================================
% 12. Output
% ============================================================

cfg.output.save_results = false;

%% ============================================================
% 13. Random seeds
% ============================================================

cfg.seed.data = 13;
cfg.seed.matrix_init_base = 1001;
cfg.seed.random_partition_base = 5001;
cfg.seed.eadmm_rounding_base = 9001;
cfg.seed.lp_perturb_base = 11001;
cfg.seed.two_opt_base = 13001;

%% ============================================================
% 14. Build method-enabled flags
% ============================================================

valid = ["eadmm","lp","mmdlp","random","vector","matrix"];

selected = lower(strtrim(string(cfg.methods.run)));
selected = selected(:)';
selected = selected(selected~="");

if isempty(selected) || any(~ismember(selected,valid))
    error('Invalid or empty cfg.methods.run.');
end

cfg.methods.run = unique(selected,'stable');

cfg.methods.enabled = struct();

for method = valid
    cfg.methods.enabled.(char(method)) = ...
        any(cfg.methods.run==method);
end

end