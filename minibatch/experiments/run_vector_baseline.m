function result = run_vector_baseline(problem,cfg)
% Sequential Vector algorithm from supplied vector_method.m, using the
% shared Phi and strict balanced extraction for a comparable feasible result.
% Original code is kept in methods/vector/vector_method_original.txt.
    n=problem.num_samples; bs=problem.batch_size; nb=problem.num_batches;
    phi=problem.Phi;
    rng(cfg.vector.seed,'twister');
    Q=(1:n)'; selected=zeros(bs,nb);
    total_iter=0; solve_time=0; extract_time=0;
    exact_batches=0; fallback_batches=0;
    for b=1:nb
        nq=numel(Q); np=n-nq;
        if b==nb
            selected(:,b)=Q;
            break;
        end
        % Original algorithm: P = previously selected, Q = remaining.
        tbuild=tic;
        P=setdiff((1:n)',Q,'stable');

        [Z,~]=build_mmd_batch_Z(  phi,Q,P,bs,n);
        X=bs/nq*ones(nq,1)+(rand(nq,1)-0.5)/nq*0.1;
        Y=bs/nq*ones(nq,1)+(rand(nq,1)-0.5)/nq*0.1;
        % Original splitadmm20240307 has no iteration output and prints every
        % iteration. Adaptation below keeps its update and phase rules intact.
        extract_time=extract_time+toc(tbuild);
        t=tic;
        [X,Y,iter]=vector_splitadmm(Z,bs,cfg.vector,X,Y);
        solve_time=solve_time+toc(t); total_iter=total_iter+iter;
        t=tic;
        if any(~isfinite(Y)), error('Vector nonfinite iterate in batch %d',b); end
        binary=find(Y==1);
        if numel(binary)==bs
            pick=binary; exact_batches=exact_batches+1;
        else
            fallback_batches=fallback_batches+1;
            if string(cfg.vector.fallback)=="error"
                error('Vector batch %d: %d exact binary entries, expected %d.',b,numel(binary),bs);
            end
            % Explicitly recorded heuristic to preserve cardinality.
            [~,order]=sort(Y,'descend'); pick=order(1:bs);
        end
        selected(:,b)=Q(pick);
        Q(pick)=[];
        extract_time=extract_time+toc(t);
        if mod(b,10)==0 || b==1
            fprintf('Vector batch %d/%d | remaining %d | exact %d | fallback %d\n', ...
                b,nb,numel(Q),exact_batches,fallback_batches);
        end
    end
    M=evaluate_partition(batches_to_assignment(selected,n,nb,bs),problem);
    assert(M.valid,'Vector produced an invalid partition');
    result=struct('I_raw',selected,'P_raw',M.assignment, ...
        'raw_jcommon',M.common_objective,'iterations',total_iter, ...
        'solve_time',solve_time,'extract_time',extract_time, ...
        'exact_batches',exact_batches,'fallback_batches',fallback_batches, ...
        'refined_jcommon',NaN,'two_opt_time',0,'total_time',solve_time+extract_time);
    if cfg.two_opt.enabled && cfg.vector.use_common_2opt
        opts=struct('seed',cfg.seed.two_opt_base,'cost_tol',cfg.two_opt.cost_tol, ...
            'verbose',cfg.two_opt.verbose,'eta',problem.eta);
        post=apply_common_two_opt(selected,problem,opts);
        result.I_refined=post.I_after;
        result.P_refined=post.P_after;
        result.refined_jcommon=post.metrics_after.common_objective;
        result.two_opt_time=post.time;
        result.total_time=result.total_time+post.time;
    end
    fprintf('Vector raw J_common=%.10e | exact=%d | fallback=%d\n', ...
        result.raw_jcommon,exact_batches,fallback_batches);
end
