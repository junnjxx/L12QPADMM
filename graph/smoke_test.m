function smoke_test()
%SMOKE_TEST Explicit graph-only tests; NEVER invoked automatically by default.
root=fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'problem'),fullfile(root,'data'),fullfile(root,'postprocess'));
addpath(genpath(fullfile(root,'methods')));
addpath(genpath(fullfile(root,'third_party','ADMM-GP')));
rng(19,'twister');
n=12;k=3;B=double(triu(rand(n)<0.3,1));W=sparse(B+B');
p=build_graph_problem(W,k);
cfg=config_graph();problem=graph_problem_compatible(p,cfg);
P=random_equipartition(n,k,8);
m=evaluate_partition(P,problem);
assert(m.valid,'Random infeasible');
assert(abs(m.graph_cut-graph_cut(P,W))<1e-10,'Graph cut mismatch');
s=problem.matrix_objective_scale;
assert(norm(problem.matrix_A-problem.L/s,'fro')<1e-12,'Matrix scaling mismatch');
% check old, directly coded 2opt and shared 2opt both preserve feasibility.
o=struct('seed',12,'cost_tol',1e-9,'verbose',false,'eta',cfg.matrix.eta);
[I,st]=assignment_to_batches(P,p.group_size,1e-12);
assert(~isempty(I),st);
res=apply_common_two_opt(I,problem,o);
assert(res.metrics_after.valid,'Postprocessing invalid');
assert(res.metrics_after.common_objective<=m.graph_cut+1e-8,'Postprocessing increased cut');
[p2,j2]=graph_two_opt(P,W,struct('max_passes',2));
assert(graph_cut(p2,W)<=m.graph_cut+1e-8 && j2<=m.graph_cut+1e-8,'Graph swap failed');
% Matrix short-run checks algorithm executes and finite residual; integer recovery not required here.
cfg.matrix.max_iter=20;cfg.matrix.print_every=1000;cfg.matrix.early_exist=false;
x=solve_matrix_graph(problem,cfg.matrix);
assert(isfinite(x.residual_r) && isfinite(x.residual_dxy),'Matrix nonfinite residual');
% eADMM and Lp are NOT executed automatically; these require MATLAB/toolboxes and can be expensive.
fprintf('Graph objective, Matrix, and 2-opt smoke tests passed.\n');
end
