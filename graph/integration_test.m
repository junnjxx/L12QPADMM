function integration_test()
%INTEGRATION_TEST Verify all solver interfaces on one small graph (MATLAB).
% Runs individually, NEVER invoked automatically by run_exp.
root=fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'problem'),fullfile(root,'experiments'), ...
    fullfile(root,'postprocess'));
addpath(genpath(fullfile(root,'methods')));
addpath(genpath(fullfile(root,'third_party','ADMM-GP')));
rng(101,'twister');
n=12;k=3; T=triu(double(rand(n)<0.4),1); W=sparse(T+T');
p=build_graph_problem(W,k);
cfg=config_graph();problem=graph_problem_compatible(p,cfg);
assert(norm(problem.L*ones(n,1),inf)<1e-12);
assert(abs(problem.matrix_objective_scale*sum(problem.matrix_A(:))-sum(problem.L(:)))<1e-8);

cfg.matrix.max_iter=40;cfg.matrix.print_every=1000;cfg.matrix.early_exist=false;
a=solve_matrix_graph(problem,cfg.matrix);
assert(isfinite(a.residual_r)&&isfinite(a.residual_dxy));
if ~isempty(a.P_raw), assert(evaluate_partition(a.P_raw,problem).valid);end

cfg.random.num_starts=3;
cfg.reporting.print_random_every=3;
rconfig=cfg;rconfig.two_opt.enabled=false;
r=run_random_baseline(problem,rconfig);
I=r.raw_I{r.best_raw_id};
post=apply_common_two_opt(I,problem,struct('seed',999,'cost_tol',1e-9, ...
    'verbose',false,'eta',cfg.matrix.eta));
assert(post.metrics_after.valid);
assert(post.metrics_after.common_objective<=post.metrics_before.common_objective+1e-8);

cfg.eadmm.max_iter=80;cfg.eadmm.rounding_restarts=2;
cfg.eadmm.rounding_checkpoints=[1,2];
cfg.eadmm.met.enabled=false;cfg.eadmm.compute_safe_lower_bound=false;
e=run_eadmm_graph_full(problem,cfg);
assert(~isempty(e.best_vc_I));
assert(evaluate_partition(batches_to_assignment(e.best_vc_I,n,k,n/k),problem).valid);

cfg.vector.max_iter=40;cfg.vector.print_every=100;
cfg.vector.verbose=false;
v=run_vector_graph_baseline(problem,cfg);
assert(evaluate_partition(v.P_raw,problem).valid);
assert(isfinite(v.raw_cut));

% MMDLP's latest paper LP implementation requires MATLAB linprog.
% The test uses exact-audit enumeration on small subproblems to verify
%   f_LP <= f_exact_IQP <= f_topb  and sum_g group_cost = full graph cut.
if isempty(which('linprog')) || isempty(which('optimoptions'))
    fprintf('SKIP: MMDLP graph runtime test needs Optimization Toolbox linprog.\n');
else
    cfg.mmdlp.verbose=false;
    cfg.mmdlp.exact_audit=true;
    cfg.mmdlp.exact_audit_max_q=12;
    a=run_mmdlp_graph_baseline(problem,cfg);
    assert(a.num_lp_solves==k-1);
    assert(evaluate_partition(a.P_raw,problem).valid);
    assert(abs(sum(a.group_contributions)-a.raw_cut)<1e-8*max(1,abs(a.raw_cut)));
    assert(all(a.lp_feasibility_residual(1:k-1)<=1e-5));
    fprintf('PASS: MMDLP graph LP, top-b, feasible partition, group-cut identity.\n');
end

% NO fallback to the older dual-BB projector. Lp test only when MOSEK exists.
if ~isempty(cfg.lp.proj.mosek_toolbox_path) && exist(cfg.lp.proj.mosek_toolbox_path,'dir')
    addpath(cfg.lp.proj.mosek_toolbox_path);
end
if isempty(which('mosekopt'))
    fprintf('SKIP: Lp runtime test needs MOSEK; latest projection source is included.\n');
else
    cfg.lp.max_outer=2;cfg.lp.max_inner=5;
    cfg.lp.num_starts=1;cfg.lp.use_common_2opt=false;cfg.lp.verbose=false;
    l=run_lp_baseline(problem,cfg);
    assert(l.alg2.enabled && l.bs.enabled);
    assert(~isempty(l.alg2.valid_ids) && ~isempty(l.bs.valid_ids));
end
fprintf('PASS: Matrix/eADMM/Vector/Random graph integration; MMDLP and Lp checked when toolboxes available.\n');
end
