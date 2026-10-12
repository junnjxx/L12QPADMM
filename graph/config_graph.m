function cfg=config_graph()
%CONFIG_GRAPH Single configuration for independent Graph Partition project.
% METHODS: eADMM / Lp / MMDLP / Vector / Random / Matrix (order user-selectable).
% Only k-equipartition; NO training minibatch data or artificial bs^2 scaling.
cfg=struct();
% official：官方给定的 | demo: 自己生成一个测试图
cfg.graph.source='official';
cfg.graph.family='rand80'; % rand20 / rand50 / rand80
cfg.graph.n=100;
cfg.graph.num_groups=10; % k (must divide n); nodes per group=n/k
cfg.graph.instance=1;
cfg.graph.demo_density=0.20;
cfg.graph.demo_seed=20261010;
cfg.graph.mat_path='';
cfg.graph.variable='';
cfg.graph.W=[];
cfg.methods.run={'Matrix'};

% Matrix: exact same X-first, Y-affine updates as v3; only A=L/s_L.
cfg.matrix.c=10;
cfg.matrix.beta0=10;
cfg.matrix.beta_mode='adaptive';
cfg.matrix.kabeta=1e6;
cfg.matrix.tau_X=1e-5;
cfg.matrix.eta=0.000005;
cfg.matrix.eta_stage_initial=-10;
cfg.matrix.eta_adapt_enabled=true;
cfg.matrix.eta_stagnation_iters=100;
cfg.matrix.eta_growth_factor=1.5;
cfg.matrix.eta_max=[]; % 4*||L/s_L||_2*(1+1e-6)
cfg.matrix.max_iter=2000000;
cfg.matrix.tol=1e-13;
cfg.matrix.init_mode='uniform'; % uniform | random_binary | lp_best_jcommon | eadmm_sdp
cfg.matrix.lp_init_variant='alg2'; % alg2 | bs
cfg.matrix.multistart.num_starts=1;
cfg.matrix.seed=1001;
cfg.matrix.print_every=1000;
cfg.matrix.stop_check_interval=10;
cfg.matrix.fast_prox=true;
cfg.matrix.early_exist=true; % success does NOT depend on early_exist
cfg.matrix.early_exist_interval=10;
cfg.matrix.early_exist_dxy_tol=1e-4;
cfg.matrix.early_exist_round_tol=1e-10;
cfg.matrix.support_tol=1e-10;
cfg.matrix.allow_extra_rounding=true; % clearly marked heuristic, never native

% LATEST uploaded Lp Algorithm 2 (MOSEK projection + accelerated exact N2).
cfg.lp.enabled=true; cfg.lp.run_alg2=true; cfg.lp.run_bs=true;
cfg.lp.num_starts=1; cfg.lp.p=0.75;
cfg.lp.eps0=0.1; cfg.lp.eps_min=1e-3;
cfg.lp.sigma_max=1e6; cfg.lp.gamma=0.9;
cfg.lp.tol_outer=1e-3;
cfg.lp.alpha0=1e-3;cfg.lp.theta=1e-4;cfg.lp.delta=0.5;
cfg.lp.reference_eta=0.85;
cfg.lp.tau_x0=1e-3;cfg.lp.tau_f0=1e-6;
cfg.lp.tau_x_min=1e-5;cfg.lp.tau_f_min=1e-8;
cfg.lp.sigma_minus=-1;
cfg.lp.use_curvature_sigma0=true;
cfg.lp.max_outer=20;cfg.lp.max_inner=300;
cfg.lp.max_backtrack=60;
cfg.lp.alpha_min=1e-20;cfg.lp.alpha_max=1e20;
cfg.lp.tau_index_shift=1;cfg.lp.kkt_map_tol=1e-8;
cfg.lp.perturb_rho=0.05;
cfg.lp.local_max_swaps=inf;
cfg.lp.local_block=512;cfg.lp.improve_tol=1e-12;
cfg.lp.local_recompute_every=100;
cfg.lp.local_verbose_every=0;
cfg.lp.local_final_verify=true;
cfg.lp.do_rounding=true;
cfg.lp.do_internal_local_search=true;
cfg.lp.use_common_2opt=true;
cfg.lp.store_inner_history=false;
cfg.lp.verbose=false;
cfg.lp.proj.tol=1e-5;
cfg.lp.proj.gap_tol=1e-5;
% Empty paths use an existing MOSEK MATLAB installation and license.
% Set both paths explicitly on your server when needed.
cfg.lp.proj.mosek_toolbox_path='/home/ubuntu/xlj/mosek/11.2/toolbox/r2019bom';
cfg.lp.proj.mosek_license_file='/home/ubuntu/xlj/mosek/mosek.lic';
cfg.lp.proj.mosek_verbose=false;
cfg.lp.strict_paper_checks=true;

% Upstream ADMM-GP full DNN, optional MET strengthening and safe lower bound.
cfg.eadmm.max_iter=2000;
cfg.eadmm.sigma0=1;
cfg.eadmm.tol=1e-5;
cfg.eadmm.init_mode='zero'; % zero | uniform
cfg.eadmm.rounding_restarts=10;
cfg.eadmm.rounding_checkpoints=[1,10];
cfg.eadmm.use_2opt=false; % common 2-opt is performed ONCE after best raw rounding
cfg.eadmm.verbose_2opt=false;
cfg.eadmm.compute_safe_lower_bound=false; % set true for linprog certificate
cfg.eadmm.met.enabled=false; % set true for full MET tightening (expensive)
cfg.eadmm.met.max_ineq_per_round=100;
cfg.eadmm.met.violation_tol=1e-3;
cfg.eadmm.met.max_rounds=50;
cfg.eadmm.met.sigma0=0.1;

% Latest uploaded MMDLP LP relaxation (AAAI-21 Eq. 10), graph-adapted.
% For each sequential group, Z=0.5*L(Q,Q), not the old MMD kernel Z.
% Requires MATLAB Optimization Toolbox (linprog).
cfg.mmdlp.linprog_display='none';
cfg.mmdlp.feasibility_tol=1e-7;
cfg.mmdlp.verbose=true;
cfg.mmdlp.print_group_every=10;
cfg.mmdlp.store_relaxed_history=false;
cfg.mmdlp.exact_audit=false; % diagnostic only; excluded from method time
cfg.mmdlp.exact_audit_max_q=16;

% User's latest Vector method; graph subproblem uses A_g=L(Q,Q).
% One group at a time, each group contains n/k vertices.
cfg.vector.eta=0.001;
cfg.vector.beta=100;
cfg.vector.max_iter=2000;
cfg.vector.epsi=1e-8;
cfg.vector.print_every=100;
cfg.vector.seed=17001;
cfg.vector.fallback='topk'; % topk | error
cfg.vector.verbose=true;
cfg.vector.print_group_every=10;

cfg.random.num_starts=20;
cfg.two_opt.enabled=true;
cfg.two_opt.cost_tol=1e-9;
cfg.two_opt.verbose=false;

cfg.reporting.print_each_matrix_start=true;
cfg.reporting.print_random_every=20;
cfg.reporting.print_first_random=5;
cfg.seed.matrix_init_base=1001;
cfg.seed.random_partition_base=5001;
cfg.seed.eadmm_rounding_base=9001;
cfg.seed.lp_perturb_base=11001;
cfg.seed.two_opt_base=13001;

cfg.output.save_results=true;
cfg.output.save_prefix='graph';
end
