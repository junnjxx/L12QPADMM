function print_selected_methods_report(report,results,problem,cfg)
%PRINT_SELECTED_METHODS_REPORT Print a report for any selected method subset.

    fprintf('\n======================================================================\n');
    fprintf('★ Selected-method comparison report ★\n');
    fprintf('实际运行方法：%s\n',strjoin(report.selected_methods,' -> '));
    fprintf('统一目标：J_common，越小越好。\n');
    fprintf('======================================================================\n');

    fprintf('\n==================== 最终对比表 ====================\n');
    disp(report.summary_table);
    fprintf(['时间口径：SolveTimeSec=核心求解时间；InitExtractTimeSec=rounding/解提取/', ...
        '外部warm-start开销；LocalSearchTimeSec=统一2-opt；TotalMethodTimeSec=端到端。\n']);

    fprintf('\n*本次找到的最好可行 J_common = %.10e*\n',report.best_feasible_jcommon);
    if report.safe_lower_bound_valid
        fprintf('*安全 J_common 下界 = %.10e*\n',report.safe_jcommon_lower_bound);
        fprintf('*最好解 certified relative gap = %.6f%%*\n',report.best_certified_gap_pct);
        fprintf('安全下界来源 = %s。\n',report.safe_lower_bound_source);
    else
        fprintf('安全下界 / certified gap：不可用或未运行 eADMM 认证分支。\n');
    end

    if isfield(results,'eadmm')
        E = results.eadmm;
        fprintf('\n[eADMM]\n');
        fprintf('*best raw Vc J_common = %.10e*\n',E.best_vc_jcommon);
        fprintf('*DNN 迭代数 = %d* | *sec/iter = %.6e* | *DNN solve = %.6f 秒*\n', ...
            E.dnn_iterations,E.dnn_avg_iter_time,E.solve_time);
        fprintf('*raw 解生成总时间 = %.6f 秒*\n',E.solution_time_raw);
        if cfg.eadmm.use_2opt
            fprintf('*best Vc+2opt J_common = %.10e* | *端到端时间 = %.6f 秒*\n', ...
                E.best_vc2_jcommon,E.solution_time_with_2opt);
        end
        if report.safe_lower_bound_valid
            fprintf('*安全 J_common 下界 = %.10e*\n',report.safe_jcommon_lower_bound);
        end
    end

    if isfield(results,'lp') && isfield(results.lp,'enabled') && results.lp.enabled
        L = results.lp;
        fprintf('\n[Lp]\n');
        if isfield(L,'alg2') && L.alg2.enabled
            print_lp_variant(L.alg2,'Lp-Alg2');
        end
        if isfield(L,'bs') && L.bs.enabled
            print_lp_variant(L.bs,'Lp-bs');
        end
    end

    if isfield(results,'random')
        R = results.random;
        rid = R.best_raw_id;
        fprintf('\n[Random]\n');
        fprintf('*best-of-%d raw J_common = %.10e* | *累计生成时间 = %.6f 秒*\n', ...
            numel(R.raw_jcommon),R.raw_jcommon(rid),sum(R.gen_time,'omitnan'));
        if cfg.two_opt.enabled
            r2 = R.best_refined_id;
            fprintf('*best +2opt J_common = %.10e* | *总 wall time = %.6f 秒*\n', ...
                R.refined_jcommon(r2),R.wall_time);
        end
    end

    if isfield(results,'vector')
        V = results.vector;
        fprintf('\n[Vector] raw J_common=%.10e | iterations=%d | solve=%.6f s | extraction=%.6f s\n', ...
            V.raw_jcommon,V.iterations,V.solve_time,V.extract_time);
        fprintf('Vector ADMM exact-binary batches=%d/%d; fallback batches=%d\n', ...
            V.exact_batches,problem.num_batches,V.fallback_batches);
        if cfg.two_opt.enabled && cfg.vector.use_common_2opt
            fprintf('Vector+2opt J_common=%.10e | 2opt=%.6f s | total=%.6f s\n', ...
                V.refined_jcommon,V.two_opt_time,V.total_time);
        end
    end

    if isfield(results,'matrix')
        M = results.matrix;
        bid = M.best_raw_id;
        B = M.raw{bid};
        ids = M.valid_ids(:)';
        fprintf('\n[Matrix]\n');
        fprintf('初始化方式 = %s。\n',string(M.init_mode));
        if M.warmstart.is_external
            fprintf('warm start = %s。\n',string(M.warmstart_label));
            fprintf('warm-start 上游+转换时间 = %.6f 秒。\n',M.warmstart_overhead_time);
        end
        fprintf('*best raw J_common = %.10e*\n',M.raw_jcommon(bid));
        fprintf('*best start 迭代数 = %d* | *sec/iter = %.6e* | *solve = %.6f 秒*\n', ...
            B.iterations,B.avg_iter_time,B.solve_time);
        if isfield(B,'early_exist_triggered')
            fprintf('early_exist triggered = %d',B.early_exist_triggered);
            if B.early_exist_triggered
                fprintf(' | trigger iter = %d',B.early_exist_iteration);
            end
            fprintf('\n');
        end
        fprintf('best-of-%d Matrix 自身累计 solve+extract = %.6f 秒。\n', ...
            numel(ids),sum(M.solve_time(ids),'omitnan')+sum(M.extract_time(ids),'omitnan'));
        if cfg.two_opt.enabled
            m2 = M.best_refined_id;
            fprintf('*best +2opt J_common = %.10e*\n',M.refined_jcommon(m2));
        end
    end

    fprintf('\n共享问题构造时间 = %.6f 秒（不计入各方法运行时间）。\n',problem.build_time);

    function print_lp_variant(V,name)
        ids = V.valid_ids(:)';
        bid = V.best_raw_id;
        total_inner = sum(V.inner_iterations(ids),'omitnan');
        total_solve = sum(V.solve_time(ids),'omitnan');
        fprintf('*%s best raw J_common = %.10e* | *累计 inner=%d* | *累计 solve=%.6f 秒*\n', ...
            name,V.raw_jcommon(bid),total_inner,total_solve);
        if cfg.lp.use_common_2opt
            b2 = V.best_refined_id;
            fprintf('*%s +common2opt J_common = %.10e*\n',name,V.refined_jcommon(b2));
        end
    end
end
