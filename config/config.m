function cfg = config()
%CONFIG Single user-facing configuration file for the whole experiment.
% Canonical execution order: eADMM -> Lp -> Random -> Vector -> Matrix,
% restricted to cfg.methods.run.
cfg=struct();
%% Methods to run
cfg.methods.run=["Lp","Vector"];
%% Problem instance
cfg.data.profile="custom";
cfg.data.group_sizes=[96,94,97,88,86,107,108,114,100,110]; % 4:[20,25,33,22] 10:[48,47,46,44,43,56,54,57,50,55]
cfg.data.dim_x=100;
cfg.data.mnist_dir="";
cfg.data.mnist_per_digit=100;
cfg.data.mnist_normalize=true;
cfg.data.sparsity_density=4e-3;
cfg.data.linear_relative_noise=1e-1;
cfg.batch.size=10;
cfg.batch.allow_truncation=false;
%% Matrix solver
cfg.matrix.c=10;
cfg.matrix.beta0=10;
cfg.matrix.eta=0.01;
% Paper-aligned proximal coefficient:
%   tau_X/2 * ||X-X_prev||_F^2
% The current paper requires tau_X>0 but does not state one unique numerical value.
% Use this as the experiment parameter and change it explicitly if needed.
cfg.matrix.tau_X=0.00001;
cfg.matrix.max_iter=2000000;
cfg.matrix.tol=1e-6;
%% eta continuation
% -1 -> -0.5 -> 0 -> 0.01 -> 0.011 -> ... -> eta_max
cfg.matrix.eta_adapt_enabled=true;
cfg.matrix.eta_stage_initial=-10;
cfg.matrix.eta_stagnation_iters=1000;
cfg.matrix.eta_growth_factor=1.1;
% eta_max is computed at runtime from the current problem matrix:
%   eta_max = 4*||A||_2*(1+1e-6),
% so that eta_max is strictly above the paper threshold 4||A||_2.
cfg.matrix.eta_max=[];
%% nearest assignment search diagnostics
% Every search_round ADMM iterations, solve exactly
%   min_{P in F1} ||P-X^k||_F,
% where F1 is the balanced binary assignment set.
% The exact balanced assignment is obtained by a Hungarian solve after
% expanding each batch label into batch_size identical slots.
cfg.matrix.search_assignment_enabled=true;
cfg.matrix.search_round=1000;
% true: print each search online, including per-search and cumulative Hungarian time.
% Final matches_final / stabilization summary is printed after the final assignment is known.
cfg.matrix.search_verbose=true;
%% stopping
cfg.matrix.stop_check_interval=10;
%% early_exist
cfg.matrix.early_exist=true;
cfg.matrix.early_exist_dxy_tol=1e-4;
cfg.matrix.early_exist_round_tol=0.5;
cfg.matrix.early_exist_interval=10;
%% initialization
cfg.matrix.init_mode="uniform";
cfg.matrix.lp_init_variant="alg2";
%% clean mode
cfg.matrix.matrix_clean=true;
cfg.matrix.matrix_detail=false;
cfg.matrix.matrix_clean_fast=true;
%% MEX solver
% IMPORTANT: the current MEX kernel is legacy and does not implement the
% paper-aligned X-first update with tau_X. Keep false until C++ is updated.
cfg.matrix.matrix_mex=false;
%% MEX profiling
cfg.matrix.mex_profile=false;
cfg.matrix.mex_profile_warmup=100000;
cfg.matrix.mex_profile_iters=10000;
cfg.matrix.n_outer=1;
%% beta strategy
cfg.matrix.kabeta=1000000;
cfg.matrix.beta_mode="adaptive";
cfg.matrix.use_adaptive_beta=true;
cfg.matrix.beta_adapt_rule="current";
cfg.matrix.eadmm_beta_decay=100;
cfg.matrix.eadmm_beta_min=1e-4;
cfg.matrix.eadmm_beta_max=cfg.matrix.beta0;
cfg.matrix.eadmm_sigma_decay=100;
cfg.matrix.eadmm_sigma_min=1e-4;
cfg.matrix.eadmm_sigma_max=1e5;
%% Matrix diagnostics
cfg.matrix.print_every=100;
cfg.matrix.make_plot=false;
cfg.matrix.save_plot=false;
cfg.matrix.show_plot_title=false;
cfg.matrix.save_result=false;
cfg.matrix.partition_support_tol=1e-12;
cfg.matrix.figure_dir='figures_matrix';
cfg.matrix.multistart.num_starts=1;
%% Matrix variant
% The paper-aligned tau_X implementation currently supports standard only.
cfg.matrix.variant="standard";
cfg.matrix.epsikkt=1e-3;
cfg.matrix.vc_restarts=10;
cfg.matrix.vc_seed_base=9001;
%% Lp baseline
% Jiang-Liu-Wen (2016) balanced-assignment adaptation.
%
% Main paper numerical parameters from Section 6.1:
%   p = 0.75 in subsequent experiments;
%   eps0 = 0.1, eps_min = 1e-3, sigma_max = 1e6, gamma = 0.9;
%   tol_outer = 1e-3;
%   alpha0 = 1e-3, theta = 1e-4, delta = 0.5, eta = 0.85;
%   tau_x0 = 1e-3, tau_f0 = 1e-6;
%   tau_x_min = 1e-5, tau_f_min = 1e-8.
% Lp-Alg2 uses greedy balanced rounding + fast N2-best.
% Lp-bs disables the INTERNAL N2-best refinement.
%% Lp baseline
cfg.lp.run_alg2=true;
cfg.lp.run_bs=true;
cfg.lp.num_starts=1;

cfg.lp.p=0.75;
cfg.lp.eps0=0.1;
cfg.lp.eps_min=1e-3;
cfg.lp.sigma_max=1e6;
cfg.lp.gamma=0.9;
cfg.lp.tol_outer=1e-3;

cfg.lp.alpha0=1e-3;
cfg.lp.theta=1e-4;
cfg.lp.delta=0.5;
cfg.lp.reference_eta=0.85;

cfg.lp.tau_x0=1e-3;
cfg.lp.tau_f0=1e-6;
cfg.lp.tau_x_min=1e-5;
cfg.lp.tau_f_min=1e-8;

cfg.lp.sigma_minus=-1;
cfg.lp.use_curvature_sigma0=true;


%% Balanced-polytope projection
% For the rectangular balanced-assignment adaptation, all Euclidean
% projections are solved accurately by MOSEK.
%
% Projection:
%   min_X 0.5*||X-C||_F^2
%   s.t.  X*1_m = 1_n,
%         X'*1_n = capacities,
%         X >= 0.

cfg.lp.proj.tol = 1e-6;

cfg.lp.proj.mosek_toolbox_path = ...
    "/home/ubuntu/xlj/mosek/11.2/toolbox/r2019bom";

cfg.lp.proj.mosek_license_file = ...
    "/home/ubuntu/xlj/mosek/mosek.lic";

cfg.lp.proj.mosek_verbose = true;

%% Numerical safeguards
cfg.lp.max_outer=100;
cfg.lp.max_inner=500;
cfg.lp.max_backtrack=60;

cfg.lp.alpha_min=1e-20;
cfg.lp.alpha_max=1e20;

cfg.lp.tau_index_shift=1;
cfg.lp.kkt_map_tol=1e-8;
cfg.lp.perturb_rho=0.05;

%% Exact accelerated internal N2-best
cfg.lp.local_max_swaps=inf;
cfg.lp.improve_tol=1e-12;

% Rebuild the entire exact gain cache periodically to eliminate
% floating-point drift; between rebuilds only affected pair blocks
% are refreshed exactly.
cfg.lp.local_recompute_every=100;
cfg.lp.local_final_verify=true;

% 0 avoids terminal I/O becoming a timing bottleneck.
cfg.lp.local_verbose_every=0;

% Compatibility field; the fast routine does not use row blocking.
cfg.lp.local_block=512;

cfg.lp.do_rounding=true;
cfg.lp.do_internal_local_search=true;

cfg.lp.store_inner_history=false;
cfg.lp.verbose=true;

% Strictly diagnose failure of projection / line-search / stopping rules.
cfg.lp.strict_paper_checks=true;

% Extra project-wide postprocessing, separate from the paper's internal N2.
cfg.lp.use_common_2opt=true;


%% Vector baseline
cfg.vector.eta=0.001;
cfg.vector.beta=100;
cfg.vector.max_iter=2000;
cfg.vector.epsi=1e-8;
cfg.vector.print_every=100;
cfg.vector.seed=17001;
cfg.vector.use_common_2opt=true;
cfg.vector.fallback="topk";
%% Random baseline
cfg.random.num_starts=100;
%% eADMM-SDP
cfg.eadmm.max_iter=20000;
cfg.eadmm.tol=1e-5;
cfg.eadmm.init_mode="zero";
cfg.eadmm.rounding_restarts=10;
cfg.eadmm.sigma0=1;
cfg.eadmm.compute_safe_lower_bound=true;
cfg.eadmm.met.enabled=true;
cfg.eadmm.met.max_ineq_per_round=100;
cfg.eadmm.met.violation_tol=1e-3;
cfg.eadmm.met.max_rounds=50;
cfg.eadmm.met.sigma0=0.1;
cfg.eadmm.rounding_checkpoints=[1,10,50,100];
cfg.eadmm.use_2opt=true;
cfg.eadmm.verbose_2opt=false;
%% Common 2-opt
cfg.two_opt.enabled=true;
cfg.two_opt.cost_tol=1e-9;
cfg.two_opt.verbose=false;
%% Reporting
cfg.reporting.time_budgets=[0.25,0.5,1,5,30,100];
cfg.reporting.print_each_matrix_start=true;
cfg.reporting.print_random_every=20;
cfg.reporting.print_first_random=5;
%% Output
cfg.output.save_results=true;
%% Seeds
cfg.seed.data=13;
cfg.seed.matrix_init_base=1001;
cfg.seed.random_partition_base=5001;
cfg.seed.eadmm_rounding_base=9001;
cfg.seed.lp_perturb_base=11001;
cfg.seed.two_opt_base=13001;
end
