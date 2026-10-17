function result = run_vector_baseline(problem,cfg)
%RUN_VECTOR_BASELINE Sequential MMD-based minibatch selection with
% Matrix-aligned X-first vector ADMM (same kernel as graph/Vector).
%
% IMPORTANT: The minibatch subproblem remains Z=build_mmd_batch_Z(Phi,Q,P,
% bs,n). This is NOT the graph Laplacian L(Q,Q). Only the ADMM kernel and
% adaptive beta/eta rules are changed to the Matrix-aligned graph version.
%
% The sparse/binary variable is X, not Y. All selected batches are checked.
% A nonnative final X either causes an error or an EXPLICIT top-k fallback.
%
% All ADMM and Vector-specific numeric settings come from cfg.vector,
% initialized once by config/config.m (cfg.vector = cfg.matrix).

    n=problem.num_samples;
    bs=problem.batch_size;
    nb=problem.num_batches;
    phi=problem.Phi;
    assert(n==bs*nb,'Vector requires equal-size complete minibatches.');

    % All parameter mapping and inheritance is done in config/config.m.
    opts=cfg.vector;
    fallback=char(string(opts.fallback));
    if ~any(strcmpi(fallback,{'topk','error'}))
        error('Vector: cfg.vector.fallback must be topk or error.');
    end
    print_group_every=opts.print_group_every;
    assert(isscalar(print_group_every) && isfinite(print_group_every) && ...
        print_group_every>=1 && print_group_every==round(print_group_every));
    if ~any(strcmpi(char(string(opts.beta_mode)),{'fixed','adaptive'}))
        error('Vector: cfg.vector.beta_mode must be fixed or adaptive.');
    end

    rng(cfg.vector.seed,'twister');
    Q=(1:n)';
    selected=zeros(bs,nb);
    total_iter=0;
    solve_time=0;
    extract_time=0;  % build Z, initialization, native extraction
    exact_batches=0;
    fallback_batches=0;
    batch_stats=cell(nb-1,1);
    wall_tic=tic;

    for b=1:nb
        nq=numel(Q);
        if nq<bs, error('Vector: fewer remaining nodes than batch size.'); end
        if b==nb
            % Last batch is determined by cardinality; no ADMM solve.
            selected(:,b)=Q;
            Q=zeros(0,1);
            break;
        end

        tbuild=tic;
        P=setdiff((1:n)',Q,'stable');
        [Z,~]=build_mmd_batch_Z(phi,Q,P,bs,n);
        effective_B=nq/bs;
        % Mirror graph Vector's Matrix-aligned uniform jitter rule.
        X0=ones(nq,1)/effective_B+ ...
            (rand(nq,1)-0.5)/(effective_B*100);
        Y0=ones(nq,1)/effective_B+ ...
            (rand(nq,1)-0.5)/(effective_B*100);
        extract_time=extract_time+toc(tbuild);

        ts=tic;
        [X,~,iter,vinfo]=vector_splitadmm(Z,bs,opts,X0,Y0);
        solve_time=solve_time+toc(ts);
        total_iter=total_iter+iter;
        batch_stats{b}=vinfo;

        tsel=tic;
        if any(~isfinite(X(:)))
            error('Vector: nonfinite X in minibatch %d.',b);
        end
        if vinfo.native_valid
            pick=find(round(X)==1);
            exact_batches=exact_batches+1;
        elseif strcmpi(fallback,'error')
            error(['Vector minibatch %d/%d: not native cardinality-%d binary ', ...
                '(nnzX=%d, max fractionality=%.3e, dxy=%.3e, eta=%.6g).'], ...
                b,nb,bs,vinfo.nnzX,vinfo.binary_deviation, ...
                vinfo.residual_dxy,vinfo.eta_final);
        else
            [~,order]=sort(X,'descend');
            pick=order(1:bs);
            fallback_batches=fallback_batches+1;
        end
        if numel(pick)~=bs || numel(unique(pick))~=bs
            error('Vector: invalid selected indices in minibatch %d.',b);
        end
        selected(:,b)=Q(pick);
        Q(pick)=[];
        extract_time=extract_time+toc(tsel);

        if b==1 || mod(b,print_group_every)==0 || b==nb-1
            fprintf(['Vector minibatch %d/%d | remaining=%d | iter=%d | ', ...
                'native=%d | fallback_total=%d | beta=%.6g | eta=%.6g | ', ...
                'dxy=%.3e\n'], ...
                b,nb,numel(Q),iter,vinfo.native_valid,fallback_batches, ...
                vinfo.beta_final,vinfo.eta_final,vinfo.residual_dxy);
        end
    end
    assert(isempty(Q),'Vector: unassigned samples remain.');
    Praw=batches_to_assignment(selected,n,nb,bs);
    M=evaluate_partition(Praw,problem);
    assert(M.valid,'Vector produced an invalid balanced minibatch partition');

    raw_wall_time=toc(wall_tic);
    extract_time=max(0,raw_wall_time-solve_time); % fair complete raw runtime
    result=struct('I_raw',selected,'P_raw',M.assignment, ...
        'raw_jcommon',M.common_objective,'iterations',total_iter, ...
        'solve_time',solve_time,'extract_time',extract_time, ...
        'exact_batches',exact_batches,'fallback_batches',fallback_batches, ...
        'refined_jcommon',NaN,'two_opt_time',0, ...
        'total_time',solve_time+extract_time, ...
        'raw_wall_time',raw_wall_time,'batch_stats',{batch_stats}, ...
        'solver','Matrix-aligned X-first ADMM, adaptive beta/eta; MMD Z');

    if cfg.two_opt.enabled && cfg.vector.use_common_2opt
        opts2=struct('seed',cfg.seed.two_opt_base, ...
            'cost_tol',cfg.two_opt.cost_tol,'verbose',cfg.two_opt.verbose, ...
            'eta',problem.eta);
        post=apply_common_two_opt(selected,problem,opts2);
        result.I_refined=post.I_after;
        result.P_refined=post.P_after;
        result.refined_jcommon=post.metrics_after.common_objective;
        result.two_opt_time=post.time;
        result.total_time=result.total_time+post.time;
    end
    fprintf(['Vector raw J_common=%.10e | exact=%d/%d | ', ...
        'fallback=%d | solve=%.4fs | total=%.4fs\n'], ...
        result.raw_jcommon,exact_batches,nb-1,fallback_batches, ...
        solve_time,result.total_time);
end
