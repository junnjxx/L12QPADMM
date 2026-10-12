function result = solve_matrix_batching(problem,opts)
%SOLVE_MATRIX_BATCHING Matrix method entry point.
% Paper-aligned standard backend:
%   X = box/L1/2 block with tau_X proximal term.
%   Y = affine assignment block.
% The legacy MEX and VC backends are blocked here because they do not yet
% implement the paper-aligned X/Y order and tau_X proximal term.
if ~isfield(opts,'variant'), opts.variant="standard"; end
if ~isfield(opts,'matrix_mex'), opts.matrix_mex=false; end
if ~isfield(opts,'verbose'), opts.verbose=true; end
if ~isfield(opts,'matrix_detail'), opts.matrix_detail=false; end
if ~isfield(opts,'matrix_clean'), opts.matrix_clean=false; end
if ~isfield(opts,'matrix_clean_fast'), opts.matrix_clean_fast=false; end
if ~isfield(opts,'early_exist'), opts.early_exist=false; end
if ~isfield(opts,'early_exist_dxy_tol'), opts.early_exist_dxy_tol=1e-4; end
if ~isfield(opts,'early_exist_round_tol'), opts.early_exist_round_tol=1e-2; end
if ~isfield(opts,'early_exist_interval'), opts.early_exist_interval=10; end
if ~isfield(opts,'stop_check_interval'), opts.stop_check_interval=10; end
if ~isfield(opts,'eta_adapt_enabled'), opts.eta_adapt_enabled=false; end
if ~isfield(opts,'eta_stage_initial'), opts.eta_stage_initial=-1; end
if ~isfield(opts,'eta_stagnation_iters'), opts.eta_stagnation_iters=100; end
if ~isfield(opts,'eta_growth_factor'), opts.eta_growth_factor=1.1; end
if ~isfield(opts,'eta_max'), opts.eta_max=opts.eta; end
if ~isfield(opts,'n_outer'), opts.n_outer=1; end
if ~isfield(opts,'use_adaptive_beta'), opts.use_adaptive_beta=false; end
if ~isfield(opts,'beta_mode'), opts.beta_mode="fixed"; end
if ~isfield(opts,'partition_support_tol'), opts.partition_support_tol=1e-8; end
if ~isfield(opts,'search_assignment_enabled'), opts.search_assignment_enabled=false; end
if ~isfield(opts,'search_round'), opts.search_round=1000; end
if ~isfield(opts,'search_verbose'), opts.search_verbose=false; end
if ~isfield(opts,'tau_X'), error('cfg.matrix.tau_X is required for the paper-aligned Matrix solver.'); end
if ~isscalar(opts.tau_X) || ~isfinite(opts.tau_X) || opts.tau_X<=0, error('cfg.matrix.tau_X must be a positive finite scalar.'); end
variant=lower(string(opts.variant));
if variant~="standard"
    error(['The paper-aligned tau_X implementation currently supports variant="standard" only. ',...
        'The existing VC branch still uses the legacy block order and must be updated separately.']);
end
if logical(opts.matrix_mex)
    error(['matrix_mex=true is disabled for the paper-aligned tau_X implementation. ',...
        'The current MEX kernel still implements the legacy block order and no tau_X term.']);
end
matrix_clean=logical(opts.matrix_clean);
if matrix_clean
    opts.verbose=false;
    opts.matrix_detail=false;
    opts.make_plot=false;
    opts.save_plot=false;
    opts.save_result=false;
end
fprintf('\n============================================================\n');
fprintf('[Matrix variant] STANDARD, PAPER-ALIGNED\n');
fprintf('[Matrix backend] PURE MATLAB\n');
fprintf('[Matrix tau_X] %.6g\n',opts.tau_X);
fprintf('============================================================\n');
A=problem.A;
G=problem.G;
normA2=norm(A,2);
opts.eta_max=4*normA2*(1+1e-6);
beta0=opts.beta0;
c=opts.c;
[N,B]=size(G);
eta_reference=opts.eta;
if opts.eta_adapt_enabled, eta_state=opts.eta_stage_initial; else, eta_state=eta_reference; end
eta_state_initial=eta_state;
eta_effective_initial=max(eta_state_initial,0);
eta_update_history=struct('outer_id',{},'iteration',{},'nnzY',{},'dxy',{},'eta_state_old',{},'eta_state_new',{},'eta_effective_old',{},'eta_effective_new',{});
eta_update_count=0;
if ~isfield(opts,'init_mode'), opts.init_mode="uniform"; end
switch string(opts.init_mode)
    case "uniform"
        X=ones(N,B)/B+(rand(N,B)-0.5)*(1/B)*1e-1;
        Y=ones(N,B)/B+(rand(N,B)-0.5)*(1/B)*1e-1;
    case "zero"
        X=zeros(N,B);
        Y=zeros(N,B);
    case "provided"
        if ~isfield(opts,'X0') || isempty(opts.X0), error('Matrix init_mode="provided" requires opts.X0.'); end
        X=double(opts.X0);
        if ~isequal(size(X),[N,B]) || any(~isfinite(X(:))), error('Provided Matrix X0 must be a finite N-by-B matrix.'); end
        if isfield(opts,'Y0') && ~isempty(opts.Y0), Y=double(opts.Y0); else, Y=X; end
        if ~isequal(size(Y),[N,B]) || any(~isfinite(Y(:))), error('Provided Matrix Y0 must be a finite N-by-B matrix.'); end
    otherwise
        error('Unknown Matrix init_mode: %s',string(opts.init_mode));
end
if opts.verbose
    fprintf('开始运行 paper-aligned Matrix ADMM。\n');
    fprintf('Matrix tau_X = %.6g | c+tau_X = %.6g\n',opts.tau_X,c+opts.tau_X);
    fprintf('Matrix matrix_clean = %d | matrix_clean_fast = %d | matrix_detail = %d\n',matrix_clean,logical(opts.matrix_clean_fast),logical(opts.matrix_detail));
    fprintf('Matrix early_exist = %d | init = %s\n',logical(opts.early_exist),char(string(opts.init_mode)));
    fprintf('Matrix beta mode = %s | adaptive = %d\n',char(string(opts.beta_mode)),logical(opts.use_adaptive_beta));
    fprintf('Matrix eta: state_initial=%.6g | effective_initial=%.6g | first_positive=%.6g | max=%.6g\n',eta_state_initial,eta_effective_initial,eta_reference,opts.eta_max);
end
error_list=[];
r_history=[];
dxy_history=[];
round_y_history=[];
fractional_ratio_history=[];
Lambda_history=cell(0,1);
lambda_fnorm=[];
dx_history=[];
beta_history=[];
ori_obj_history=[];
flag=0;
actual_iter=0;
final_error=NaN;
final_r=NaN;
final_dxy=NaN;
search_trace=struct();
search_trace.total_time=0;
tsolve=tic;
for outer_id=1:opts.n_outer
    [X,Y,error_list,flag,Lambda_history,r_history,dxy_history,round_y_history,fractional_ratio_history,lambda_fnorm,dx_history,beta_history,ori_obj_history,eta_state,eta_updates_outer,eta_update_count_outer,actual_iter,final_error,final_r,final_dxy,search_trace]=...
        splitadmm20240119(A,G,eta_state,beta0,c,outer_id,opts.max_iter,X,Y,opts.tol,opts.use_adaptive_beta,opts);
    eta_update_count=eta_update_count+eta_update_count_outer;
    if ~matrix_clean && ~isempty(eta_updates_outer)
        eta_update_history=[eta_update_history,eta_updates_outer]; %#ok<AGROW>
    end
end
solve_wall_time=toc(tsolve);
search_diagnostic_time=0;
if isfield(search_trace,'total_time') && isfinite(search_trace.total_time)
    search_diagnostic_time=search_trace.total_time;
end
solve_time=max(0,solve_wall_time-search_diagnostic_time);
% The paper's X block is the sparse/binary block, so extraction uses X.
textract=tic;
[I,extraction_status]=assignment_to_batches(X,problem.batch_size,opts.partition_support_tol);
extract_time=toc(textract);
if isempty(I)
    P=[];
    metrics=evaluate_partition([],problem,eta_reference);
    if opts.verbose, warning('Final paper-X block could not be extracted as a valid balanced partition.'); end
else
    P=batches_to_assignment(I,problem.num_samples,problem.num_batches,problem.batch_size);
    metrics=evaluate_partition(P,problem,eta_reference);
end
search_trace=finalize_search_trace(search_trace,P,X,problem.batch_size);
print_search_trace_summary(search_trace);
result=struct();
result.method="Matrix";
result.variant="standard";
result.backend="PURE MATLAB";
result.paper_aligned=true;
result.matrix_mex=false;
result.tau_X=opts.tau_X;
result.normA2=normA2;
result.eta_max_theory=opts.eta_max;
result.search_trace=search_trace;
result.search_diagnostic_time=search_diagnostic_time;
result.solve_wall_time_including_search=solve_wall_time;
result.I=I;
result.P=P;
result.X=X;
result.Y=Y;
result.flag=flag;
result.eta_initial=eta_reference;
result.eta_evaluation_reference=eta_reference;
result.eta_stage_initial=eta_state_initial;
result.eta_effective_initial=eta_effective_initial;
result.eta_stage_final=eta_state;
result.eta_final=max(eta_state,0);
result.eta_update_count=eta_update_count;
if matrix_clean, result.eta_update_history=[]; else, result.eta_update_history=eta_update_history; end
result.final_error=final_error;
result.final_r=final_r;
result.final_dxy=final_dxy;
result.epsikkt=NaN;
result.kkt_residual=NaN;
result.kkt_triggered=false;
result.early_exist_enabled=logical(opts.early_exist);
result.early_exist_dxy_tol=opts.early_exist_dxy_tol;
result.early_exist_round_tol=opts.early_exist_round_tol;
result.early_exist_triggered=(flag==2);
if result.early_exist_triggered, result.early_exist_iteration=actual_iter; else, result.early_exist_iteration=NaN; end
result.iterations=actual_iter;
result.solve_time=solve_time;
result.extract_time=extract_time;
result.total_time=solve_time+extract_time;
result.avg_iter_time=solve_time/max(actual_iter,1);
result.init_mode=string(opts.init_mode);
result.extraction_status=extraction_status;
result.metrics=metrics;
result.matrix_clean=matrix_clean;
result.matrix_clean_fast=logical(opts.matrix_clean_fast);
result.matrix_detail=logical(opts.matrix_detail);
result.S_vc=[];
result.vc_restarts=0;
result.vc_seed_base=NaN;
result.best_vc_restart=NaN;
result.best_vc_jcommon=NaN;
result.vc_rounding_time=0;
result.vc_restart_results=[];
result.error_history=error_list;
result.r_history=r_history;
result.dxy_history=dxy_history;
result.round_y_history=round_y_history;
result.fractional_ratio_history=fractional_ratio_history;
result.lambda_history=Lambda_history;
result.lambda_fnorm=lambda_fnorm;
result.dx_history=dx_history;
result.beta_history=beta_history;
result.ori_obj_history=ori_obj_history;
if opts.matrix_detail && ~isempty(round_y_history)
    result.final_binary_deviation=round_y_history(end);
    result.final_fractional_ratio=fractional_ratio_history(end);
else
    result.final_binary_deviation=NaN;
    result.final_fractional_ratio=NaN;
end
if isfield(opts,'save_result') && opts.save_result && ~matrix_clean
    save('matrix_method_result.mat','result','opts');
end
if isfield(opts,'make_plot') && opts.make_plot && opts.matrix_detail && ~matrix_clean
    plot_matrix_admm_history(r_history,dxy_history,round_y_history,fractional_ratio_history,beta_history,opts);
end
if opts.verbose
    fprintf('\n[Matrix 最终结果]\n');
    fprintf('*Matrix backend = %s*\n',char(result.backend));
    fprintf('*Matrix tau_X = %.6g*\n',result.tau_X);
    fprintf('*Matrix ||A||_2 = %.10e | eta_max = 4||A||_2(1+1e-6) = %.10e*\n',result.normA2,result.eta_max_theory);
    if result.search_trace.enabled
        fprintf('*Nearest-assignment search uses partition-invariant P*P'' comparison.*\n');
        fprintf('*Search total Hungarian time = %.6f s | stable final match = %g*\n',...
            result.search_trace.total_time,result.search_trace.stable_final_match_iteration);
    end
    fprintf('*Matrix 迭代数 = %d*\n',result.iterations);
    fprintf('*Matrix 每次迭代平均时间 = %.6e 秒*\n',result.avg_iter_time);
    fprintf('*Matrix 求解总时间 = %.6f 秒*\n',result.solve_time);
    fprintf('Matrix 最终停止残差 = %.6e\n',result.final_error);
    fprintf('Matrix 最终 r 残差 = %.6e\n',result.final_r);
    fprintf('Matrix 最终 ||Y-X||_F = %.6e\n',result.final_dxy);
    fprintf('*Matrix eta: state %.10g -> %.10g, effective %.10g -> %.10g, first-positive/reference=%.10g, transitions=%d*\n',...
        result.eta_stage_initial,result.eta_stage_final,result.eta_effective_initial,result.eta_final,eta_reference,result.eta_update_count);
    if result.early_exist_triggered
        fprintf('*Matrix early_exist 在第 %d 次迭代触发。*\n',result.early_exist_iteration);
    elseif opts.early_exist
        fprintf('Matrix early_exist 已开启，但本次未触发。\n');
    end
    if metrics.valid
        fprintf('*Matrix J_common = %.10e*\n',metrics.common_objective);
        fprintf('Matrix 线性项 = %.10e\n',metrics.matrix_linear_term);
        fprintf('Matrix L1/2 项（统一参考 eta=%g）= %.10e\n',eta_reference,metrics.matrix_lhalf_term);
        fprintf('Matrix 完整离散目标值（统一参考 eta=%g）= %.10e\n',eta_reference,metrics.matrix_full_objective);
    end
end

end

function trace=finalize_search_trace(trace,P_final,X_final,b)
if ~isfield(trace,'enabled')
    trace=struct('enabled',false,'method',"exact_balanced_hungarian",'round',NaN,...
        'iterations',zeros(0,1),'assignments',{cell(0,1)},'coassignments',{cell(0,1)},...
        'distance_fro',zeros(0,1),'inner_product',zeros(0,1),'changed',false(0,1),...
        'reassigned_from_previous',zeros(0,1),'change_iterations',zeros(0,1),...
        'change_assignments',{cell(0,1)},'change_coassignments',{cell(0,1)},...
        'search_time',zeros(0,1),'cumulative_search_time',zeros(0,1),'total_time',0,...
        'num_searches',0,'num_changes',0,'last_change_iteration',NaN,...
        'stabilization_iteration',NaN);
end
K=numel(trace.iterations);
P_native=round(X_final);
native_valid=all(sum(P_native,2)==1) && all(sum(P_native,1)==b) && all(P_native(:)==0 | P_native(:)==1);
if native_valid, P_final=P_native; end
trace.matches_final=false(K,1);
trace.first_final_match_iteration=NaN;
trace.stable_final_match_iteration=NaN;
trace.last_search_iteration=NaN;
trace.last_search_matches_final=false;
trace.final_assignment=[];
trace.final_coassignment=[];
trace.stable_assignment=[];
trace.stable_coassignment=[];
trace.final_stable_assignment=[];
trace.final_stable_coassignment=[];
if K==0 || isempty(P_final), return; end
P_final=sparse(P_final~=0);
C_final=sparse((P_final*P_final')>0);
trace.final_assignment=P_final;
trace.final_coassignment=C_final;
for k=1:K
    if isfield(trace,'coassignments') && numel(trace.coassignments)>=k && ~isempty(trace.coassignments{k})
        Ck=sparse(trace.coassignments{k}~=0);
    else
        Pk=sparse(trace.assignments{k}~=0);
        Ck=sparse((Pk*Pk')>0);
    end
    trace.matches_final(k)=isequal(Ck,C_final);
end
trace.last_search_iteration=trace.iterations(end);
trace.last_search_matches_final=trace.matches_final(end);
idx=find(trace.matches_final,1,'first');
if ~isempty(idx), trace.first_final_match_iteration=trace.iterations(idx); end
if isfinite(trace.stabilization_iteration)
    stable_idx0=find(trace.iterations==trace.stabilization_iteration,1,'first');
    if ~isempty(stable_idx0)
        trace.stable_assignment=trace.assignments{stable_idx0};
        if isfield(trace,'coassignments') && numel(trace.coassignments)>=stable_idx0
            trace.stable_coassignment=trace.coassignments{stable_idx0};
        else
            Ps=sparse(trace.stable_assignment~=0);
            trace.stable_coassignment=sparse((Ps*Ps')>0);
        end
    end
end
suffix_all=true;
stable_idx=NaN;
for k=K:-1:1
    suffix_all=suffix_all && trace.matches_final(k);
    if suffix_all, stable_idx=k; end
end
if isfinite(stable_idx)
    trace.stable_final_match_iteration=trace.iterations(stable_idx);
    trace.final_stable_assignment=trace.assignments{stable_idx};
    if isfield(trace,'coassignments') && numel(trace.coassignments)>=stable_idx
        trace.final_stable_coassignment=trace.coassignments{stable_idx};
    else
        Pf=sparse(trace.final_stable_assignment~=0);
        trace.final_stable_coassignment=sparse((Pf*Pf')>0);
    end
end
end

function print_search_trace_summary(trace)
if ~isfield(trace,'enabled') || ~trace.enabled, return; end
fprintf('\n============================================================\n');
fprintf('[Nearest assignment search: partition-invariant P*P'' comparison]\n');
fprintf('method = %s | search_round = %d | searches = %d | partition changes = %d\n',...
    char(string(trace.method)),trace.round,trace.num_searches,trace.num_changes);
fprintf('-----------------------------------------------------------------------------------------------\n');
fprintf('%8s %10s %14s %9s %11s %12s %14s %14s\n',...
    'index','iter','dist','changed','reassigned','match_final','search_time','cum_time');
fprintf('-----------------------------------------------------------------------------------------------\n');
for k=1:trace.num_searches
    if isfield(trace,'cumulative_search_time') && numel(trace.cumulative_search_time)>=k
        cum_t=trace.cumulative_search_time(k);
    else
        cum_t=sum(trace.search_time(1:k));
    end
    fprintf('%8d %10d %14.6e %9d %11d %12d %14.6e %14.6e\n',...
        k,trace.iterations(k),trace.distance_fro(k),trace.changed(k),...
        trace.reassigned_from_previous(k),trace.matches_final(k),...
        trace.search_time(k),cum_t);
end
fprintf('-----------------------------------------------------------------------------------------------\n');
fprintf('last_change_iteration        = %g\n',trace.last_change_iteration);
fprintf('stabilization_iteration      = %g\n',trace.stabilization_iteration);
fprintf('first_final_match_iteration  = %g\n',trace.first_final_match_iteration);
fprintf('stable_final_match_iteration = %g\n',trace.stable_final_match_iteration);
fprintf('last_search_matches_final    = %d\n',trace.last_search_matches_final);
fprintf('Hungarian search total_time  = %.6f s\n',trace.total_time);
fprintf('NOTE: changed/match_final/stable_final_match are all based on P*P'', so batch-label permutations are ignored.\n');
fprintf('============================================================\n');
end
