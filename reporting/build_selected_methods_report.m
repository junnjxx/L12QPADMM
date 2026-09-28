function report = build_selected_methods_report(results,problem,cfg)
%BUILD_SELECTED_METHODS_REPORT Build a report for any selected method subset.
%
% Only fields that actually exist in RESULTS are accessed. All objective values
% are reported on the common J_common scale.

    Method = strings(0,1);
    Stage = strings(0,1);
    JCommon = zeros(0,1);
    Iterations = zeros(0,1);
    AvgTimePerIterSec = zeros(0,1);
    SolveTimeSec = zeros(0,1);
    InitExtractTimeSec = zeros(0,1);
    LocalSearchTimeSec = zeros(0,1);
    TotalMethodTimeSec = zeros(0,1);

    %% eADMM
    if isfield(results,'eadmm')
        E = results.eadmm;
        addrow(sprintf('eADMM+Vc best-of-%d',numel(E.vc_jcommon)), ...
            'raw',E.best_vc_jcommon,E.dnn_iterations,E.dnn_avg_iter_time, ...
            E.solve_time,E.vc_time,0,E.solution_time_raw);

        if cfg.eadmm.use_2opt
            addrow(sprintf('eADMM+Vc+2opt best-of-%d',numel(E.vc2_jcommon)), ...
                'same 2opt',E.best_vc2_jcommon,E.dnn_iterations,E.dnn_avg_iter_time, ...
                E.solve_time,E.vc_time,E.two_opt_time,E.solution_time_with_2opt);
        end
    end

    %% Lp family
    if isfield(results,'lp') && isfield(results.lp,'enabled') && results.lp.enabled
        L = results.lp;
        if isfield(L,'alg2') && L.alg2.enabled
            add_lp_variant(L.alg2,'Lp-Alg2');
        end
        if isfield(L,'bs') && L.bs.enabled
            add_lp_variant(L.bs,'Lp-bs');
        end
    end

    %% Random
    if isfield(results,'random')
        R = results.random;
        rid = R.best_raw_id;
        addrow(sprintf('Random best-of-%d',numel(R.raw_jcommon)), ...
            'raw',R.raw_jcommon(rid),NaN,NaN,0,sum(R.gen_time,'omitnan'),0, ...
            sum(R.gen_time,'omitnan'));

        if cfg.two_opt.enabled
            r2 = R.best_refined_id;
            addrow(sprintf('Random+2opt best-of-%d',numel(R.refined_jcommon)), ...
                'same 2opt',R.refined_jcommon(r2),NaN,NaN,0, ...
                sum(R.gen_time,'omitnan'),sum(R.local_time,'omitnan'),R.wall_time);
        end
    end

    %% Vector (sequential batch-selection ADMM)
    if isfield(results,'vector')
        V = results.vector;
        addrow('Vector','raw',V.raw_jcommon,V.iterations, ...
            V.solve_time/max(V.iterations,1),V.solve_time,V.extract_time,0, ...
            V.solve_time+V.extract_time);
        if cfg.two_opt.enabled && cfg.vector.use_common_2opt
            addrow('Vector+2opt','same 2opt',V.refined_jcommon,V.iterations, ...
                V.solve_time/max(V.iterations,1),V.solve_time,V.extract_time, ...
                V.two_opt_time,V.total_time);
        end
    end

    %% Matrix
    if isfield(results,'matrix')
        M = results.matrix;
        ids = M.valid_ids(:)';
        mb = M.best_raw_id;
        total_solve = sum(M.solve_time(ids),'omitnan');
        total_extract = M.warmstart_overhead_time + sum(M.extract_time(ids),'omitnan');
        total_iter = sum(M.iterations(ids),'omitnan');
        avg_iter = total_solve / max(total_iter,1);

        addrow(sprintf('Matrix best-of-%d',numel(ids)), ...
            'raw',M.raw_jcommon(mb),total_iter,avg_iter,total_solve,total_extract,0, ...
            total_solve+total_extract);

        if cfg.two_opt.enabled
            m2 = M.best_refined_id;
            total_local = sum(M.local_time(ids),'omitnan');
            addrow(sprintf('Matrix+2opt best-of-%d',numel(ids)), ...
                'same 2opt',M.refined_jcommon(m2),total_iter,avg_iter,total_solve, ...
                total_extract,total_local,total_solve+total_extract+total_local);
        end
    end

    if isempty(JCommon)
        error('No result rows were generated. Check cfg.methods.run.');
    end

    best_feasible = min(JCommon,[],'omitnan');
    GapToBestDiscretePct = 100*(JCommon-best_feasible)./max(abs(best_feasible),eps);

    safe_lb = NaN;
    safe_valid = false;
    safe_source = "not available (eADMM not run)";
    if isfield(results,'eadmm')
        E = results.eadmm;
        if isfield(E,'safe_jcommon_lower_bound')
            safe_lb = E.safe_jcommon_lower_bound;
        end
        if isfield(E,'safe_lower_bound_numerically_feasible')
            safe_valid = isfinite(safe_lb) && E.safe_lower_bound_numerically_feasible;
        end
        if isfield(E,'safe_lower_bound_source')
            safe_source = string(E.safe_lower_bound_source);
        end
        if safe_valid
            tol = 1e-8*max([1,abs(best_feasible),abs(safe_lb)]);
            if safe_lb > best_feasible + tol
                warning(['Safe J_common lower bound %.12e exceeds the best feasible ', ...
                    'J_common %.12e. Certified gaps are suppressed.'],safe_lb,best_feasible);
                safe_valid = false;
            end
        end
    end

    if safe_valid
        CertifiedAbsGap = JCommon-safe_lb;
        tol = 1e-8*max([1,abs(best_feasible),abs(safe_lb)]);
        CertifiedAbsGap(CertifiedAbsGap<0 & CertifiedAbsGap>=-tol)=0;
        CertifiedGapPct = 100*CertifiedAbsGap./max(abs(JCommon),eps);
    else
        CertifiedAbsGap = nan(size(JCommon));
        CertifiedGapPct = nan(size(JCommon));
    end

    summary_table = table(Method,Stage,JCommon,CertifiedAbsGap,CertifiedGapPct, ...
        GapToBestDiscretePct,Iterations,AvgTimePerIterSec,SolveTimeSec, ...
        InitExtractTimeSec,LocalSearchTimeSec,TotalMethodTimeSec);

    report = struct();
    report.selected_methods = cfg.methods.run;
    report.summary_table = summary_table;
    report.best_feasible_jcommon = best_feasible;
    report.safe_jcommon_lower_bound = safe_lb;
    report.safe_lower_bound_valid = safe_valid;
    report.safe_lower_bound_source = safe_source;
    if safe_valid
        report.best_certified_abs_gap = best_feasible-safe_lb;
        if report.best_certified_abs_gap < 0 && report.best_certified_abs_gap >= -tol
            report.best_certified_abs_gap = 0;
        end
        report.best_certified_gap_pct = ...
            100*report.best_certified_abs_gap/max(abs(best_feasible),eps);
    else
        report.best_certified_abs_gap = NaN;
        report.best_certified_gap_pct = NaN;
    end
    report.problem_build_time = problem.build_time;

    function add_lp_variant(V,name)
        ids = V.valid_ids(:)';
        bid = V.best_raw_id;
        total_solve_lp = sum(V.solve_time(ids),'omitnan');
        total_iter_lp = sum(V.inner_iterations(ids),'omitnan');
        avg_lp = total_solve_lp/max(total_iter_lp,1);

        addrow(sprintf('%s best-of-%d',name,numel(ids)), ...
            'raw',V.raw_jcommon(bid),total_iter_lp,avg_lp,total_solve_lp,0,0,total_solve_lp);

        if cfg.lp.use_common_2opt
            b2 = V.best_refined_id;
            total_local_lp = sum(V.local_time(ids),'omitnan');
            addrow(sprintf('%s+common2opt best-of-%d',name,numel(ids)), ...
                'same 2opt',V.refined_jcommon(b2),total_iter_lp,avg_lp,total_solve_lp, ...
                0,total_local_lp,total_solve_lp+total_local_lp);
        end
    end

    function addrow(method,stage,j,iter,avgiter,solve_t,extract_t,local_t,total_t)
        Method(end+1,1) = string(method);
        Stage(end+1,1) = string(stage);
        JCommon(end+1,1) = j;
        Iterations(end+1,1) = iter;
        AvgTimePerIterSec(end+1,1) = avgiter;
        SolveTimeSec(end+1,1) = solve_t;
        InitExtractTimeSec(end+1,1) = extract_t;
        LocalSearchTimeSec(end+1,1) = local_t;
        TotalMethodTimeSec(end+1,1) = total_t;
    end
end
