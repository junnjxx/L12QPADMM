function result=run_mmdlp_graph_baseline(problem,cfg)
%RUN_MMDLP_GRAPH_BASELINE Sequential graph-cut LP relaxation baseline.
% At each stage Q, solve LP for Z=0.5*L(Q,Q) (optionally scaled).
% EXACT global cut decomposition: J_cut=sum_g 0.5*p_g'*L*p_g.
%
% IMPORTANT: MMDLP is NOT ADMM. The original method includes top-k rounding.
% solution_mode='native_only' instead demands integral LP m, failing if
% fractional, and should not be called the paper's usual rounded baseline.
if ~isfield(cfg,'mmdlp')
    error('MMDLP:MissingConfig','Missing cfg.mmdlp.');
end
if isempty(which('linprog')) || isempty(which('optimoptions'))
    error('MMDLP:LinprogRequired','MATLAB Optimization Toolbox linprog is required.');
end
o=cfg.mmdlp;
if ~isfield(o,'solution_mode') || isempty(o.solution_mode), o.solution_mode='native_only'; end
if ~isfield(o,'support_tol') || isempty(o.support_tol), o.support_tol=1e-10; end
if ~isfield(o,'scale_input'), o.scale_input=true; end
if ~isfield(o,'clean_fast'), o.clean_fast=true; end
if ~isfield(o,'store_relaxed_history'), o.store_relaxed_history=false; end
if ~isfield(o,'exact_audit'), o.exact_audit=false; end
if ~isfield(o,'exact_audit_max_q'), o.exact_audit_max_q=16; end
if ~isfield(o,'verbose'), o.verbose=true; end
if ~isfield(o,'print_group_every'), o.print_group_every=10; end
if ~isfield(o,'linprog_display'), o.linprog_display='none'; end
if ~isfield(o,'feasibility_tol'), o.feasibility_tol=1e-7; end
mode=lower(char(o.solution_mode));
if ~any(strcmp(mode,{'native_only','paper_topk'}))
    error('MMDLP:UnknownMode','Unknown mmdlp solution_mode: %s',mode);
end
n=problem.n;k=problem.k;
if isfield(problem,'group_size'), b=problem.group_size;
else, b=problem.batch_size;
end
if mod(n,k)~=0 || b*k~=n
    error('MMDLP:InvalidPartition','Require equal groups and n divisible by k.');
end
L=problem.L;
Q=(1:n)';I=zeros(b,k);np=k-1;
steps=repmat(struct('group',0,'remaining',0,'num_variables',0, ...
    'num_inequalities',0,'lp_objective',NaN, ...
    'rounded_iqp_objective',NaN,'lp_feasibility_residual',NaN, ...
    'lp_cardinality_residual',NaN,'graph_contribution',NaN, ...
    'z_build_time',0,'lp_model_build_time',0,'lp_solve_time',0, ...
    'lp_postsolve_time',0,'rounding_time',0,'selection_time',0, ...
    'rounded_objective_eval_time',0,'exact_audit_time',0, ...
    'objective_scale',1,'native_valid',false,'binary_deviation',NaN, ...
    'used_paper_topk',false),k,1);
want_history=logical(o.store_relaxed_history) && ~logical(o.clean_fast);
if want_history, relaxed_history=cell(np,1);else,relaxed_history={};end
optlp=struct('display',o.linprog_display, ...
    'feasibility_tol',o.feasibility_tol, ...
    'solution_mode',mode,'support_tol',o.support_tol);
solve_time=0;audit_time=0;fallback_groups=0;
total_wall_tic=tic;
for g=1:k
    nq=numel(Q);
    steps(g).group=g;
    steps(g).remaining=nq;
    if nq<b
        error('MMDLP:InvalidRemainder','Group %d has only %d vertices.',g,nq);
    end
    if g==k
        % Last group is forced by the equipartition constraints.
        I(:,g)=Q;
        v=ones(nq,1);
        steps(g).native_valid=true;
        steps(g).binary_deviation=0;
        steps(g).graph_contribution=0.5*full(v'*L(Q,Q)*v);
        Q=zeros(0,1);
        break;
    end
    tz=tic;
    L_block=full(L(Q,Q)); % principal block of FULL graph Laplacian
    Z_raw=0.5*L_block;
    if logical(o.scale_input)
        s=max(1,norm(L_block,inf));  % SAME per-stage rule as new Vector
    else
        s=1;
    end
    Z=Z_raw/s;
    steps(g).objective_scale=s;
    steps(g).z_build_time=toc(tz);
    % Native-only mode errors inside select_batch BEFORE assigning vertices.
    try
        [pick,m_relaxed,m_binary,lpinfo]=mmdlp_select_batch(Z,b,optlp);
        if cfg.mmdlp.verbose && ~cfg.mmdlp.clean_fast

    lp_iterations = NaN;

    if isfield(lpinfo,'output') && ...
       isfield(lpinfo.output,'iterations')
        lp_iterations = lpinfo.output.iterations;
    end

    fprintf(['MMDLP LP group=%d | exitflag=%d | ', ...
             'fval=%.6f | iterations=%g | ', ...
             'solve=%.4fs | feas=%.3e\n'], ...
        g, lpinfo.exitflag, lpinfo.fval, ...
        lp_iterations, lpinfo.solve_time, ...
        lpinfo.max_feasibility_residual);

end
    catch ME
        if strcmp(ME.identifier,'MMDLP:NativeBinaryRequired')
            error('MMDLP:NativeGroupFailed', ...
                'Group %d/%d (remaining=%d) has no integral LP solution. %s', ...
                g,k,nq,ME.message);
        end
        rethrow(ME);
    end
    solve_time=solve_time+lpinfo.solve_time;
    steps(g).num_variables=lpinfo.num_variables;
    steps(g).num_inequalities=lpinfo.num_inequalities;
    % Convert any LP/rounded objective back to the ORIGINAL graph scale.
    steps(g).lp_objective=s*lpinfo.relaxation_objective;
    steps(g).rounded_iqp_objective=s*lpinfo.rounded_iqp_objective;
    steps(g).lp_feasibility_residual=lpinfo.max_feasibility_residual;
    steps(g).lp_cardinality_residual=lpinfo.cardinality_residual;
    steps(g).lp_model_build_time=lpinfo.model_build_time;
    steps(g).lp_solve_time=lpinfo.solve_time;
    steps(g).lp_postsolve_time=lpinfo.postsolve_time;
    steps(g).rounding_time=lpinfo.rounding_time;
    steps(g).selection_time=lpinfo.selection_time;
    steps(g).rounded_objective_eval_time=lpinfo.rounded_objective_eval_time;
    steps(g).graph_contribution=steps(g).rounded_iqp_objective;
    steps(g).native_valid=lpinfo.native_valid;
    steps(g).binary_deviation=lpinfo.binary_deviation;
    steps(g).used_paper_topk=lpinfo.used_paper_topk;
    fallback_groups=fallback_groups+double(lpinfo.used_paper_topk);
    if want_history, relaxed_history{g}=m_relaxed; end
    if logical(o.exact_audit) && nq<=o.exact_audit_max_q
        ta=tic;
        ex=mmdlp_bruteforce_iqp(Z,b);
        flo=lpinfo.relaxation_objective;
        fex=ex.best_objective;
        fro=lpinfo.rounded_iqp_objective;
        epscheck=1e-7*max([1,abs(flo),abs(fex),abs(fro)]);
        assert(flo<=fex+epscheck && fex<=fro+epscheck, ...
            'MMDLP: LP/exact/selected objective ordering failed for group %d.',g);
        steps(g).exact_audit_time=toc(ta);
        audit_time=audit_time+steps(g).exact_audit_time;
    end
    if numel(pick)~=b || nnz(m_binary)~=b || ...
            numel(unique(pick))~=b || any(pick<1) || any(pick>nq)
        error('MMDLP:InvalidSelection','Invalid group %d selection.',g);
    end
    I(:,g)=Q(pick);
    keep=true(nq,1);keep(pick)=false;Q=Q(keep);
    if ~logical(o.clean_fast) && logical(o.verbose) && ...
            (g==1 || mod(g,o.print_group_every)==0 || g==np)
        fprintf(['MMDLP group=%d/%d | q=%d | vars=%d | LP=%.8g | ', ...
            'selected=%.8g | native=%d | frac=%.3e | solve=%.4fs\n'], ...
            g,k,nq,lpinfo.num_variables,steps(g).lp_objective, ...
            steps(g).graph_contribution,steps(g).native_valid, ...
            steps(g).binary_deviation,lpinfo.solve_time);
    end
end
if ~isempty(Q),error('MMDLP:UnassignedNodes','Unassigned vertices remain.');end
if ~isequal(sort(I(:)),(1:n)')
    error('MMDLP:InvalidPartition','Groups do not form a vertex permutation.');
end
P=batches_to_assignment(I,n,k,b);
metrics=evaluate_partition(P,problem);
assert(metrics.valid,'MMDLP:InvalidPartition','Final partition not valid.');
contrib=sum([steps.graph_contribution]);
assert(abs(contrib-metrics.common_objective)<= ...
    1e-8*max([1,abs(contrib),abs(metrics.common_objective)]), ...
    'MMDLP:CutMismatch','Group objectives do not sum to global cut.');
elapsed_with_audit=toc(total_wall_tic);
elapsed=max(0,elapsed_with_audit-audit_time);
remaining_time=max(0,elapsed-solve_time);
result=struct();
result.method='MMDLP';
result.formulation='Sequential graph cut; Z=0.5*L(Q,Q); AAAI-21 Eq.(10) LP relaxation';
result.solution_mode=mode;
result.native_valid=(fallback_groups==0 && all([steps.native_valid]));
result.fallback_groups=fallback_groups;
result.P_raw=P;
result.I_raw=I;
result.raw_cut=metrics.common_objective;
result.raw_jcommon=metrics.common_objective;
result.num_lp_solves=np;
result.steps=steps;
result.lp_objective_per_group=[steps.lp_objective]';
result.rounded_iqp_objective_per_group=[steps.rounded_iqp_objective]';
result.group_contributions=[steps.graph_contribution]';
result.lp_num_variables=[steps.num_variables]';
result.lp_num_inequalities=[steps.num_inequalities]';
result.lp_feasibility_residual=[steps.lp_feasibility_residual]';
result.objective_scale_per_group=[steps.objective_scale]';
result.z_build_time=sum([steps.z_build_time]);
result.lp_model_build_time=sum([steps.lp_model_build_time]);
result.lp_solve_time=solve_time;
result.lp_postsolve_time=sum([steps.lp_postsolve_time]);
result.rounding_time=sum([steps.rounding_time]);
result.selection_time=sum([steps.selection_time]);
result.rounded_objective_eval_time=sum([steps.rounded_objective_eval_time]);
result.exact_audit_time=audit_time;
result.raw_wall_time_with_audit=elapsed_with_audit;
result.raw_wall_time=elapsed;
result.solve_time=solve_time;
result.init_extract_time=remaining_time;
result.total_time=elapsed;
result.iterations=np;
result.relaxed_history=relaxed_history;
fprintf(['MMDLP graph | mode=%s | raw_cut=%.10g | native=%d | ', ...
    'LP groups=%d | linprog=%.4fs | other=%.4fs | total=%.4fs\n'], ...
    mode,result.raw_cut,result.native_valid,np,solve_time,remaining_time,elapsed);
end
