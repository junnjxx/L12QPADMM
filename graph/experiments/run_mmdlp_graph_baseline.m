function result = run_mmdlp_graph_baseline(problem,cfg)
%RUN_MMDLP_GRAPH_BASELINE Sequential graph-cut adaptation of AAAI-21 MMD-LP.
%
% The binary graph k-equipartition objective is
%   J_cut(P) = 0.5*trace(P'*L*P) = sum_g 0.5*p_g'*L*p_g.
%
% At group g, let Q contain the remaining vertices; choose b=n/k of them:
%   min_{m in {0,1}^{|Q|}, 1'*m=b} m'*Z*m, Z = 0.5*L(Q,Q).
% The last group is forced. Each group objective is a contribution to the
% ORIGINAL graph cut, not the induced-subgraph cut.
%
% We keep the latest uploaded MMD-LP sign-dependent LP relaxation and its
% deterministic top-b rounding UNMODIFIED:
%   mmdlp_solve_relaxation.m -> mmdlp_select_batch.m.
% The source paper's MMD-specific Z construction (Eqs. 6--8) is NOT used:
% this is a graph-adapted sequential LP heuristic, not a verbatim MMD
% minibatch-sequencing method. The LP objective is a relaxation of the
% individual group's binary quadratic, NOT a certified global cut bound.
%
% Timing: solve_time is only total linprog time; init_extract_time includes
% model building, extraction, diagnostics, and all other pre-2opt overhead.
% Optional exact-audit brute force is excluded from both times.

    if ~isfield(cfg,'mmdlp')
        error('MMDLP:MissingConfig','Missing cfg.mmdlp; use config_graph.m.');
    end
    if isempty(which('linprog')) || isempty(which('optimoptions'))
        error('MMDLP:LinprogRequired', ...
            'Graph MMDLP requires Optimization Toolbox (linprog, optimoptions).');
    end

    n=problem.n;
    k=problem.k;
    b=problem.batch_size;
    if mod(n,k)~=0 || b*k~=n
        error('MMDLP:InvalidPartition','Graph must have n divisible by k.');
    end
    L=problem.L;
    Q=(1:n)';
    I=zeros(b,k);
    np=k-1;

    % Keep per-group diagnostics consistent with the original project.
    steps=repmat(struct( ...
        'group',0,'remaining',0,'num_variables',0,'num_inequalities',0, ...
        'lp_objective',NaN,'rounded_iqp_objective',NaN, ...
        'lp_feasibility_residual',NaN,'lp_cardinality_residual',NaN, ...
        'graph_contribution',NaN,'z_build_time',0, ...
        'lp_model_build_time',0,'lp_solve_time',0, ...
        'lp_postsolve_time',0,'rounding_time',0, ...
        'rounded_objective_eval_time',0,'exact_audit_time',0),k,1);
    relaxed_history=cell(np,1);
    optlp=struct('display',cfg.mmdlp.linprog_display, ...
                 'feasibility_tol',cfg.mmdlp.feasibility_tol);
    solve_time=0;
    audit_time=0;
    total_wall_tic=tic;

    for g=1:k
        nq=numel(Q);
        steps(g).group=g;
        steps(g).remaining=nq;
        if nq<b
            error('MMDLP:InvalidRemainder','Group %d has only %d remaining vertices.',g,nq);
        end
        if g==k
            % Last group is uniquely determined; no LP is needed.
            I(:,g)=Q;
            v=ones(nq,1);
            steps(g).graph_contribution=0.5*full(v'*L(Q,Q)*v);
            Q=zeros(0,1);
            break;
        end

        tz=tic;
        % IMPORTANT: use the PRINCIPAL block of the full graph Laplacian.
        % Recomputing the Laplacian of the induced graph would omit edges
        % incident to already selected vertices and change the objective.
        Z=0.5*full(L(Q,Q));
        steps(g).z_build_time=toc(tz);

        [pick,m_relaxed,m_binary,lpinfo]=mmdlp_select_batch(Z,b,optlp);
        solve_time=solve_time+lpinfo.solve_time;
        steps(g).num_variables=lpinfo.num_variables;
        steps(g).num_inequalities=lpinfo.num_inequalities;
        steps(g).lp_objective=lpinfo.relaxation_objective;
        steps(g).rounded_iqp_objective=lpinfo.rounded_iqp_objective;
        steps(g).lp_feasibility_residual=lpinfo.max_feasibility_residual;
        steps(g).lp_cardinality_residual=lpinfo.cardinality_residual;
        steps(g).lp_model_build_time=lpinfo.model_build_time;
        steps(g).lp_solve_time=lpinfo.solve_time;
        steps(g).lp_postsolve_time=lpinfo.postsolve_time;
        steps(g).rounding_time=lpinfo.rounding_time;
        steps(g).rounded_objective_eval_time=lpinfo.rounded_objective_eval_time;
        steps(g).graph_contribution=lpinfo.rounded_iqp_objective;

        if logical(cfg.mmdlp.store_relaxed_history)
            relaxed_history{g}=m_relaxed;
        end
        if logical(cfg.mmdlp.exact_audit) && nq<=cfg.mmdlp.exact_audit_max_q
            audit_tic=tic;
            exact=mmdlp_bruteforce_iqp(Z,b);
            flo=lpinfo.relaxation_objective;
            fex=exact.best_objective;
            fro=lpinfo.rounded_iqp_objective;
            epscheck=1e-7*max([1,abs(flo),abs(fex),abs(fro)]);
            assert(flo<=fex+epscheck && fex<=fro+epscheck, ...
                'MMDLP: LP/exact/rounded objective ordering failed in group %d.',g);
            steps(g).exact_audit_time=toc(audit_tic);
            audit_time=audit_time+steps(g).exact_audit_time;
        end

        if numel(pick)~=b || nnz(m_binary)~=b || ...
                numel(unique(pick))~=b || any(pick<1) || any(pick>nq)
            error('MMDLP:TopKInvalid','Invalid top-b rounding in group %d.',g);
        end
        I(:,g)=Q(pick);
        keep=true(nq,1);
        keep(pick)=false;
        Q=Q(keep);

        if logical(cfg.mmdlp.verbose) && ...
            (g==1 || mod(g,cfg.mmdlp.print_group_every)==0 || g==np)
            fprintf(['MMDLP graph group=%d/%d | nQ=%d | vars=%d | ineq=%d | ', ...
                'LPobj=%.9g | roundobj=%.9g | model=%.4fs | LP=%.4fs | ', ...
                'feas=%.3e\n'], ...
                g,k,nq,lpinfo.num_variables,lpinfo.num_inequalities, ...
                lpinfo.relaxation_objective,lpinfo.rounded_iqp_objective, ...
                lpinfo.model_build_time,lpinfo.solve_time, ...
                lpinfo.max_feasibility_residual);
        end
    end

    if ~isempty(Q)
        error('MMDLP:UnassignedNodes','Some vertices remain unassigned.');
    end
    if ~isequal(sort(I(:)),(1:n)')
        error('MMDLP:InvalidPartition','Groups do not form a vertex permutation.');
    end
    P=batches_to_assignment(I,n,k,b);
    metrics=evaluate_partition(P,problem);
    assert(metrics.valid,'MMDLP: Final assignment is not balanced binary.');
    contrib=sum([steps.graph_contribution]);
    assert(abs(contrib-metrics.common_objective)<= ...
        1e-8*max([1,abs(contrib),abs(metrics.common_objective)]), ...
        'MMDLP: Group contributions do not sum to the graph cut.');
    elapsed_with_audit=toc(total_wall_tic);
    elapsed=max(0,elapsed_with_audit-audit_time);
    remaining_time=max(0,elapsed-solve_time);

    result=struct();
    result.method='MMDLP';
    result.formulation='Sequential graph cut; Z=0.5*L(Q,Q); AAAI-21 Eq.(10) LP relaxation';
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
    result.z_build_time=sum([steps.z_build_time]);
    result.lp_model_build_time=sum([steps.lp_model_build_time]);
    result.lp_solve_time=solve_time;
    result.lp_postsolve_time=sum([steps.lp_postsolve_time]);
    result.rounding_time=sum([steps.rounding_time]);
    result.rounded_objective_eval_time=sum([steps.rounded_objective_eval_time]);
    result.exact_audit_time=audit_time;
    result.raw_wall_time_with_audit=elapsed_with_audit;
    result.raw_wall_time=elapsed;
    result.solve_time=solve_time;
    result.init_extract_time=remaining_time;
    result.total_time=elapsed;
    result.iterations=np;
    if logical(cfg.mmdlp.store_relaxed_history)
        result.relaxed_history=relaxed_history;
    else
        result.relaxed_history={};
    end
    fprintf(['MMDLP graph | raw_cut=%.10g | LP groups=%d | ', ...
        'linprog=%.4fs | other=%.4fs | total=%.4fs\n'], ...
        result.raw_cut,np,solve_time,remaining_time,elapsed);
end
