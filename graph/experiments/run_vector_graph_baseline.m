function result = run_vector_graph_baseline(problem,cfg)
%RUN_VECTOR_GRAPH_BASELINE Sequential graph-cut k-equipartition with Matrix-aligned X-first vector ADMM.
%
% Let P=[p_1,...,p_k] be a balanced binary assignment. Exactly:
%     Cut(P) = 0.5*trace(P'*L*P) = sum_g 0.5*p_g'*L*p_g.
% When selecting group g from the remaining vertices Q, the unselected
% group indicator x is zero outside Q and its exact group contribution is
%     0.5*x'*L(Q,Q)*x,  sum(x)=group_size, x binary.
% We pass A_g=L(Q,Q) to the Matrix-aligned vector_splitadmm solver.
% This is a SEQUENTIAL GREEDY decomposition: the current group is optimized
% without lookahead, unlike simultaneous full-assignment Matrix/Lp.
% In particular NO old build_mmd_batch_Z or minibatch Phi is used here.

n=problem.num_samples;
k=problem.num_batches;
group_size=problem.batch_size;
L=problem.L;
Q=(1:n)';
I=zeros(group_size,k);
solve_time=0;
model_build_time=0;
extract_time=0;
iterations=0;
exact_count=0;
fallback_count=0;
steps=repmat(struct('group',0,'remaining',0,'iterations',0,'exact',false, ...
    'fallback',false,'model_build_sec',0,'solve_sec',0,'extract_sec',0,...
    'group_contribution',0),k,1);
wallclock=tic;
rng(cfg.vector.seed,'twister');
for g=1:k
    nq=numel(Q);
    steps(g).group=g;
    steps(g).remaining=nq;
    if nq<group_size, error('Vector: too few unassigned vertices'); end
    if g==k
        I(:,g)=Q;
        xpick=ones(group_size,1);
        Z=L(Q,Q);
        steps(g).group_contribution=0.5*full(xpick'*Z*xpick);
        Q=zeros(0,1);
        break;
    end
    tbuild=tic;
    % Full graph Laplacian principal block: DO NOT construct an induced
    % subgraph Laplacian here, because that changes the global cut objective.
    Z=L(Q,Q);
    % Match the current Matrix uniform initializer: 1/B + jitter/(B*100),
    % using the effective number of remaining groups B=nq/group_size.
    effective_B=nq/group_size;
    x0=ones(nq,1)/effective_B+(rand(nq,1)-0.5)/(effective_B*100);
    y0=ones(nq,1)/effective_B+(rand(nq,1)-0.5)/(effective_B*100);
    steps(g).model_build_sec=toc(tbuild);
    model_build_time=model_build_time+steps(g).model_build_sec;
    ts=tic;
    [x,~,niter,vinfo]=vector_splitadmm(Z,group_size,cfg.vector,x0,y0);
    steps(g).solve_sec=toc(ts);
    solve_time=solve_time+steps(g).solve_sec;
    iterations=iterations+niter;
    steps(g).iterations=niter;
    if any(~isfinite(x)),error('Vector: nonfinite x in group %d',g);end
    tsel=tic;
    % Matrix puts the half-thresholding on X; extract from X, not Y.
    if vinfo.native_valid
        pick=find(round(x)==1);
        exact_count=exact_count+1;
        steps(g).exact=true;
    else
        if strcmpi(char(string(cfg.vector.fallback)),'error')
            error('Vector group %d did not yield an exact binary cardinality-%d vector',g,group_size);
        end
        [~,ord]=sort(x,'descend');
        pick=ord(1:group_size);
        fallback_count=fallback_count+1;
        steps(g).fallback=true;
    end
    xpick=zeros(nq,1);xpick(pick)=1;
    steps(g).group_contribution=0.5*full(xpick'*Z*xpick);
    I(:,g)=Q(pick);
    Q(pick)=[];
    steps(g).extract_sec=toc(tsel);
    extract_time=extract_time+steps(g).extract_sec;
    if cfg.vector.verbose && (g==1||mod(g,cfg.vector.print_group_every)==0 || g==k-1)
        fprintf('Vector graph group=%d/%d | remaining=%d | iters=%d | exact=%d | fallback=%d\n', ...
            g,k,numel(Q),niter,exact_count,fallback_count);
    end
end
assert(isempty(Q),'Vector: some vertices were left unassigned');
P=batches_to_assignment(I,n,k,group_size);
metrics=evaluate_partition(P,problem);
assert(metrics.valid,'Vector: final graph partition invalid');
contrib=sum([steps.group_contribution]);
assert(abs(contrib-metrics.common_objective)<1e-8*max(1,abs(metrics.common_objective)), ...
    'Vector: group contribution sum disagrees with global cut');
elapsed=toc(wallclock);
result=struct('P_raw',P,'I_raw',I,'raw_cut',metrics.common_objective,...
    'iterations',iterations,'solve_time',solve_time,...
    'model_build_time',model_build_time,'extract_time',extract_time,...
    'init_extract_time',max(0,elapsed-solve_time),...
    'total_time',elapsed,'steps',steps,...
    'exact_groups',exact_count,'fallback_groups',fallback_count,...
    'formulation','greedy blocks of the original 0.5*trace(P''*L*P) objective');
fprintf('Vector graph | raw_cut=%.10f | exact_groups=%d/%d | fallbacks=%d | solve=%.4fs | total=%.4fs\n',...
    result.raw_cut,exact_count,k-1,fallback_count,solve_time,elapsed);
end
