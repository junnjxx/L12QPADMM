function results=run_graph_experiment()
%RUN_GRAPH_EXPERIMENT Independent equal-cardinality graph partition experiment.
% Common objective J_cut(P)=0.5*trace(P'*L*P), original graph weight scale.
root=fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'data'),fullfile(root,'problem'),...
    fullfile(root,'experiments'),fullfile(root,'postprocess'));
addpath(genpath(fullfile(root,'methods')));
addpath(genpath(fullfile(root,'third_party','ADMM-GP')));

cfg=config_graph();
[W,metadata]=load_graph_instance(cfg.graph,root);
p=build_graph_problem(W,cfg.graph.num_groups);
problem=graph_problem_compatible(p,cfg);
methods=cellstr(string(cfg.methods.run));
allowed={'Matrix','eADMM','Lp','MMDLP','Vector','Random'};
for t=1:numel(methods)
    assert(any(strcmpi(methods{t},allowed)),'Unsupported method %s',methods{t});
end
if numel(unique(lower(string(methods))))~=numel(methods),error('Duplicate methods');end
if any(strcmpi(methods,'Lp'))
    % Latest Lp projector requires MOSEK. Do not silently substitute dual-BB.
    if ~isempty(cfg.lp.proj.mosek_toolbox_path)
        if ~exist(cfg.lp.proj.mosek_toolbox_path,'dir')
            error('Lp MOSEK toolbox directory not found: %s',cfg.lp.proj.mosek_toolbox_path);
        end
        addpath(cfg.lp.proj.mosek_toolbox_path);
    end
    if isempty(which('mosekopt'))
        error(['Latest Lp-Alg2/Lp-bs require MOSEK (mosekopt). ', ...
            'Install/configure MOSEK, or exclude Lp from cfg.methods.run.']);
    end
end
if any(strcmpi(methods,'MMDLP'))
    % This baseline uses the original paper's linprog relaxation, not MOSEK.
    if isempty(which('linprog')) || isempty(which('optimoptions'))
        error('MMDLP requires MATLAB Optimization Toolbox (linprog).');
    end
end
if strcmpi(cfg.matrix.init_mode,'lp_best_jcommon') && ~any(strcmpi(methods,'Lp'))
    error('Matrix LP warm start requires running Lp earlier in cfg.methods.run');
end
if strcmpi(cfg.matrix.init_mode,'eadmm_sdp') && ~any(strcmpi(methods,'eADMM'))
    error('Matrix eADMM warm start requires eADMM earlier in cfg.methods.run');
end
if ~exist(fullfile(root,'logs'),'dir'),mkdir(fullfile(root,'logs'));end
if cfg.output.save_results && ~exist(fullfile(root,'results'),'dir'),mkdir(fullfile(root,'results'));end
stamp=datestr(now,'yyyymmdd_HHMMSS');
tag=sprintf('graph_%s_N%d_k%d_%s',char(string(cfg.graph.family)),p.n,p.k,stamp);
tag=regexprep(tag,'[^A-Za-z0-9_-]','-');
idx=0;
while exist(fullfile(root,'logs',[tag,'.txt']),'file') || ...
      exist(fullfile(root,'results',[tag,'.mat']),'file')
    idx=idx+1;tag=sprintf('graph_%s_N%d_k%d_%s_%02d',char(string(cfg.graph.family)),p.n,p.k,stamp,idx);
end
logfile=fullfile(root,'logs',[tag,'.txt']);
diary off;diary(logfile);cleaner=onCleanup(@()diary('off')); %#ok<NASGU>
fprintf('\n========== Independent Graph k-Equipartition ==========\n');
fprintf('Graph source=%s family=%s n=%d k=%d nodes_per_group=%d\n',...
    char(string(cfg.graph.source)),char(string(cfg.graph.family)),p.n,p.k,p.group_size);
fprintf('common J=0.5*trace(P''*L*P); total_edges_weight=%.10g\n',full(sum(W(:)))/2);
fprintf('Matrix-only scale sL=max(1,||L||inf)=%.10g | ||L/sL||2=%.10g\n',...
    problem.matrix_objective_scale,graph_spectral_norm(problem.matrix_A));
fprintf('Methods=%s\n',strjoin(methods,' -> '));

results=struct();results.problem=problem;results.settings=cfg;results.metadata=metadata;
results.methods=methods;results.logfile=logfile;
prior=struct();
summary=struct('method',{},'valid',{},'native',{},'raw_cut',{},'post_cut',{},...
    'SolveTimeSec',{},'InitExtractTimeSec',{},'LocalSearchTimeSec',{},'TotalMethodTimeSec',{},...
    'iterations',{},'lower_bound',{},'gap_pct',{});
for t=1:numel(methods)
    method=char(string(methods{t}));
    fprintf('\n------------------ %s ------------------\n',method);
    switch lower(method)
        case 'eadmm'
            raw=run_eadmm_graph_full(problem,cfg);
            prior.eadmm=raw;
            I=raw.best_vc_I;
            P=batches_to_assignment(I,p.n,p.k,p.group_size);
            met=evaluate_partition(P,problem);
            assert(met.valid,'eADMM rounding infeasible');
            cut=met.common_objective;
            solve_t=raw.solve_time;
            extract_t=raw.vc_time;
            native=false;iters=raw.dnn_iterations;
            lb=NaN;
            if raw.safe_lower_bound_numerically_feasible
                lb=raw.safe_jcommon_lower_bound;
            end
            results.eadmm=raw;
            best_p=P;
        case 'lp'
            lpconfig=cfg;lpconfig.lp.use_common_2opt=cfg.lp.use_common_2opt && cfg.two_opt.enabled;
            raw=run_lp_baseline(problem,lpconfig);
            prior.lp=raw;
            results.lp=raw;
            % Separate two variants as two methods in report.
            kinds={'alg2','bs'};
            for q=1:numel(kinds)
                vk=kinds{q};v=raw.(vk);
                if ~v.enabled,continue;end
                [bestscore,bestid]=min(v.raw_jcommon);
                if ~isfinite(bestscore),error('Lp %s has no valid result',vk);end
                P=batches_to_assignment(v.raw_I{bestid},p.n,p.k,p.group_size);
                val=evaluate_partition(P,problem);
                assert(val.valid,'Lp invalid graph partition');
                ss=struct();ss.method=['Lp-',vk];ss.valid=true;ss.native=false;
                ss.raw_cut=val.common_objective;
                ss.post_cut=val.common_objective;
                ss.SolveTimeSec=sum(v.solve_time);
                ss.InitExtractTimeSec=0;
                ss.LocalSearchTimeSec=0;
                ss.TotalMethodTimeSec=NaN;
                ss.iterations=v.inner_iterations(bestid);
                ss.lower_bound=NaN;
                if cfg.two_opt.enabled && cfg.lp.use_common_2opt
                    % LP's own common 2opt already performed in its run_variant.
                    ss.post_cut=v.refined_jcommon(bestid);
                    ss.LocalSearchTimeSec=sum(v.local_time);
                end
                ss.TotalMethodTimeSec=ss.SolveTimeSec+ss.InitExtractTimeSec+ss.LocalSearchTimeSec;
                ss.gap_pct=NaN;
                summary(end+1)=ss; %#ok<AGROW>
                fprintf('%s raw_cut=%.10g post_cut=%.10g solve=%.4f total=%.4f\n',...
                    ss.method,ss.raw_cut,ss.post_cut,ss.SolveTimeSec,ss.TotalMethodTimeSec);
            end
            continue
        case 'mmdlp'
            % Sequential graph-cut subproblems; latest uploaded AAAI-21
            % LP relaxation + top-b rounding, no MMD-specific kernel Z.
            raw=run_mmdlp_graph_baseline(problem,cfg);
            results.mmdlp=raw;
            cut=raw.raw_cut;
            best_p=raw.P_raw;
            solve_t=raw.solve_time; % actual linprog time only
            extract_t=raw.init_extract_time; % all other raw method work
            iters=raw.num_lp_solves;
            native=false; % LP output needs top-b rounding
            lb=NaN; % stage-wise LP bounds are NOT a global cut LB
        case 'vector'
            % A_g=L(Q,Q) for each sequential graph-cut group subproblem.
            raw=run_vector_graph_baseline(problem,cfg);
            results.vector=raw;
            cut=raw.raw_cut;
            best_p=raw.P_raw;
            solve_t=raw.solve_time;
            extract_t=raw.init_extract_time; % build + extraction + small overhead
            iters=raw.iterations;
            native=(raw.fallback_groups==0);
            lb=NaN;
        case 'random'
            rconfig=cfg; rconfig.two_opt.enabled=false; % one shared 2opt after selecting best raw
            raw=run_random_baseline(problem,rconfig);
            results.random=raw;
            [cut,ibest]=min(raw.raw_jcommon);
            I=raw.raw_I{ibest};
            P=batches_to_assignment(I,p.n,p.k,p.group_size);
            best_p=P;
            solve_t=raw.wall_time;extract_t=0;iters=0;native=false;lb=NaN;
        case 'matrix'
            o=cfg.matrix;
            startmode=char(string(o.init_mode));
            if strcmpi(startmode,'lp_best_jcommon')
                if ~isfield(prior,'lp'),error('Lp must precede Matrix warm start');end
                vv=char(string(o.lp_init_variant));
                rr=prior.lp.(vv);
                [~,bestid]=min(rr.raw_jcommon);
                candidate=rr.raw{bestid}.best_jcommon_pre_round_X;
                if isempty(candidate),candidate=rr.raw{bestid}.X_cont;end
                o.init_mode='provided';o.X0=candidate;o.Y0=candidate;
            elseif strcmpi(startmode,'eadmm_sdp')
                if ~isfield(prior,'eadmm'),error('eADMM must precede Matrix warm start');end
                [P0,projinfo]=eadmm_sdp_to_balanced_assignment_init(...
                    prior.eadmm.X_sdp,p.k,p.group_size,cfg.lp.proj);
                if ~projinfo.projection_converged
                    warning('eADMM warm start projection not certified converged; residual=%g',projinfo.projection_res_inf);
                end
                o.init_mode='provided';o.X0=P0;o.Y0=P0;
            end
            R=cfg.matrix.multistart.num_starts;
            if strcmpi(o.init_mode,'provided'),R=1;end
            allruns=cell(R,1);scores=inf(R,1);
            for rr=1:R
                o.seed=cfg.seed.matrix_init_base+rr-1;
                allruns{rr}=solve_matrix_graph(problem,o);
                if ~isempty(allruns{rr}.P_raw)
                    scores(rr)=allruns{rr}.raw_cut;
                end
            end
            [cut,idx]=min(scores);
            raw=allruns{idx};
            results.matrix=struct('runs',{allruns},'best_index',idx,'best',raw);
            if isempty(raw.P_raw),warning('All Matrix starts failed to return a partition');continue;end
            best_p=raw.P_raw;
            solve_t=sum(cellfun(@(z)z.solve_time,allruns));
            extract_t=sum(cellfun(@(z)z.round_time,allruns));
            iters=raw.iterations;
            native=raw.native_valid;
            lb=NaN;
        otherwise
            error('Unknown method %s',method);
    end
    assert(evaluate_partition(best_p,problem).valid,'%s invalid solution',method);
    after=best_p;
    ts2=0;
    if cfg.two_opt.enabled
        tt=tic;
        twoopt_opts=struct('seed',cfg.seed.two_opt_base,'cost_tol',cfg.two_opt.cost_tol,...
            'verbose',cfg.two_opt.verbose,'eta',cfg.matrix.eta);
        [I,stat]=assignment_to_batches(best_p,p.group_size,1e-12);
        assert(~isempty(I),'Feasible partition extraction failed: %s',char(stat));
        post=apply_common_two_opt(I,problem,twoopt_opts);
        after=post.P_after;
        ts2=toc(tt);
    end
    postmet=evaluate_partition(after,problem);
    assert(postmet.valid && postmet.common_objective<=cut+1e-6*max(1,abs(cut)),...
        'Infeasible or nonmonotone 2-opt: %s',method);
    entry=struct('method',method,'valid',true,'native',native,'raw_cut',cut,...
        'post_cut',postmet.common_objective,'SolveTimeSec',solve_t,...
        'InitExtractTimeSec',extract_t,'LocalSearchTimeSec',ts2,...
        'TotalMethodTimeSec',solve_t+extract_t+ts2,'iterations',iters,...
        'lower_bound',lb,'gap_pct',NaN);
    if isfinite(lb) && lb>0
        entry.gap_pct=100*(entry.post_cut-lb)/max(abs(entry.post_cut),eps);
    end
    summary(end+1)=entry; %#ok<AGROW>
    fprintf('%s raw_cut=%.10g post_cut=%.10g | solve=%.4f extract=%.4f post=%.4f total=%.4f | native=%d\n',...
        method,entry.raw_cut,entry.post_cut,entry.SolveTimeSec,...
        entry.InitExtractTimeSec,entry.LocalSearchTimeSec,entry.TotalMethodTimeSec,entry.native);
end
results.summary=summary;
fprintf('\n=========== Graph Cut Comparison ===========\n');
fprintf('%-12s %15s %15s %12s %12s %12s\n','Method','RawCut','Post2opt','Solve(s)','Post(s)','Total(s)');
for i=1:numel(summary)
 a=summary(i);fprintf('%-12s %15.6f %15.6f %12.4f %12.4f %12.4f\n',...
    a.method,a.raw_cut,a.post_cut,a.SolveTimeSec,a.LocalSearchTimeSec,a.TotalMethodTimeSec);
end
if cfg.output.save_results
    resultfile=fullfile(root,'results',[tag,'.mat']);
    save(resultfile,'results','-v7.3');
    fprintf('Saved: %s\n',resultfile);
end
fprintf('Finished. Log: %s\n',logfile);
end
