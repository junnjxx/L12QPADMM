function print_selected_methods_report(report,results,problem,cfg)
%PRINT_SELECTED_METHODS_REPORT
% Print report for any selected method subset.
%
% ========================================================================
% FAIR RUNTIME CONVENTION
% ========================================================================
%
%   RawMethodTimeSec
%       complete time required to produce raw feasible partition
%
%   PostprocessTimeSec
%       common project-wide 2-opt only
%
%   TotalMethodTimeSec
%       RawMethodTimeSec + PostprocessTimeSec
%
% ========================================================================
% DISPLAY CONVENTION
% ========================================================================
%
% All real-valued reported quantities are displayed with exactly
% FOUR digits after the decimal point.
%
% Integer quantities such as iteration counts, variable counts and
% batch counts remain integers.
%
% Internal computations remain full double precision.

    fprintf('\n');
    fprintf('======================================================================\n');
    fprintf('★ Selected-method comparison report ★\n');

    fprintf('实际运行方法：%s\n', ...
        strjoin(report.selected_methods,' -> '));

    fprintf('统一目标：J_common，越小越好。\n');

    fprintf('======================================================================\n');

    %% ============================================================
    % Final comparison table
    % ============================================================

    fprintf('\n');
    fprintf('==================== 最终对比表 ====================\n\n');

    T = ...
        report.summary_table;

    fprintf([ ...
        '%-34s %-12s %12s %14s %14s %14s %12s %14s %14s\n'], ...
        'Method', ...
        'Stage', ...
        'JCommon', ...
        'GapToBest(%)', ...
        'Iterations', ...
        'AvgTime/Iter', ...
        'RawTime', ...
        'PostTime', ...
        'TotalTime');

    fprintf('%s\n',repmat('-',1,142));

    for i = 1:height(T)

        method = ...
            char(T.Method(i));

        stage = ...
            char(T.Stage(i));

        j = ...
            T.JCommon(i);

        gap = ...
            T.GapToBestDiscretePct(i);

        iter = ...
            T.Iterations(i);

        avg = ...
            T.AvgTimePerIterSec(i);

        raw = ...
            T.RawMethodTimeSec(i);

        post = ...
            T.PostprocessTimeSec(i);

        total = ...
            T.TotalMethodTimeSec(i);

        fprintf('%-34s %-12s ', ...
            method,stage);

        print_float(j,12);

        print_float(gap,14);

        if isnan(iter)
            fprintf('%14s ','-');
        else
            fprintf('%14d ',round(iter));
        end

        print_float(avg,14);
        print_float(raw,12);
        print_float(post,14);
        print_float(total,14);

        fprintf('\n');
    end

    fprintf('\n');

    fprintf([ ...
        '统一时间口径：\n', ...
        '  RawTime   = 方法从开始到产生 raw feasible partition 的完整时间；\n', ...
        '  PostTime  = 统一 common 2-opt 时间；\n', ...
        '  TotalTime = RawTime + PostTime。\n']);

    fprintf([ ...
        'linprog、ADMM kernel、projection、rounding、internal N2 等', ...
        '内部时间仅用于 profiling，不作为不同方法之间的主 runtime 指标。\n']);

    fprintf([ ...
        'AvgTime/Iter 是各方法自身诊断量，不建议在不同算法之间直接比较。\n']);

    %% ============================================================
    % Global best / lower bound
    % ============================================================

    fprintf('\n');

    fprintf('*本次找到的最好可行 J_common = %.4f*\n', ...
        report.best_feasible_jcommon);

    if report.safe_lower_bound_valid

        fprintf('*安全 J_common 下界 = %.4f*\n', ...
            report.safe_jcommon_lower_bound);

        fprintf('*最好解 certified relative gap = %.4f%%*\n', ...
            report.best_certified_gap_pct);

        fprintf('安全下界来源 = %s。\n', ...
            report.safe_lower_bound_source);

    else

        fprintf([ ...
            '安全下界 / certified gap：', ...
            '不可用或未运行 eADMM 认证分支。\n']);
    end

    %% ============================================================
    % eADMM
    % ============================================================

    if isfield(results,'eadmm')

        E = results.eadmm;

        fprintf('\n');
        fprintf('[eADMM]\n');

        fprintf('*best raw Vc J_common = %.4f*\n', ...
            E.best_vc_jcommon);

        fprintf([ ...
            'DNN iterations = %d | ', ...
            'DNN sec/iter = %.4f 秒 | ', ...
            'DNN numerical solve = %.4f 秒\n'], ...
            E.dnn_iterations, ...
            E.dnn_avg_iter_time, ...
            E.solve_time);

        fprintf('*fair raw method time = %.4f 秒*\n', ...
            E.solution_time_raw);

        fprintf([ ...
            '说明：fair raw time 包含生成 raw Vc partition 所需时间；', ...
            '不包含仅用于 certification 的 MET/LB 诊断时间。\n']);

        if cfg.eadmm.use_2opt

            fprintf('*best Vc+common2opt J_common = %.4f*\n', ...
                E.best_vc2_jcommon);

            fprintf([ ...
                'common2opt = %.4f 秒 | ', ...
                '*fair total time = %.4f 秒*\n'], ...
                E.two_opt_time, ...
                E.solution_time_with_2opt);
        end

        if report.safe_lower_bound_valid

            fprintf('*安全 J_common 下界 = %.4f*\n', ...
                report.safe_jcommon_lower_bound);
        end
    end

    %% ============================================================
    % Lp family
    % ============================================================

    if isfield(results,'lp') && ...
            isfield(results.lp,'enabled') && ...
            results.lp.enabled

        L = results.lp;

        fprintf('\n');
        fprintf('[Lp]\n');

        if isfield(L,'alg2') && ...
                L.alg2.enabled

            print_lp_variant( ...
                L.alg2, ...
                'Lp-Alg2');
        end

        if isfield(L,'bs') && ...
                L.bs.enabled

            print_lp_variant( ...
                L.bs, ...
                'Lp-bs');
        end
    end

    %% ============================================================
    % MMD-LP
    % ============================================================

    if isfield(results,'mmdlp')

        M = results.mmdlp;

        fprintf('\n');
        fprintf('[MMD-LP]\n');

        fprintf([ ...
            '方法：Banerjee & Chakraborty AAAI-21 ', ...
            'Eq. (10) LP relaxation + top-k rounding。\n']);

        fprintf('*raw J_common = %.4f*\n', ...
            M.raw_jcommon);

        fprintf('*sequential LP stages = %d*\n', ...
            M.num_lp_solves);

        fprintf('*fair raw method time = %.4f 秒*\n', ...
            M.raw_wall_time);

        if M.num_lp_solves > 0

            fprintf([ ...
                '平均每个 sequential stage 的完整时间 = ', ...
                '%.4f 秒\n'], ...
                M.raw_wall_time / ...
                M.num_lp_solves);
        end

        fprintf('\n');
        fprintf('MMD-LP 内部 profiling（不作为跨方法主时间指标）：\n');

        if isfield(M,'z_build_time')

            fprintf('  Z build             = %.4f 秒\n', ...
                M.z_build_time);
        end

        if isfield(M,'lp_model_build_time')

            fprintf('  LP model build      = %.4f 秒\n', ...
                M.lp_model_build_time);
        end

        if isfield(M,'solve_time')

            fprintf('  linprog solve       = %.4f 秒\n', ...
                M.solve_time);
        end

        if isfield(M,'lp_postsolve_time')

            fprintf('  LP postsolve        = %.4f 秒\n', ...
                M.lp_postsolve_time);
        end

        if isfield(M,'rounding_time')

            fprintf('  top-k rounding      = %.4f 秒\n', ...
                M.rounding_time);
        end

        if isfield(M,'rounded_objective_eval_time')

            fprintf('  rounded obj eval    = %.4f 秒\n', ...
                M.rounded_objective_eval_time);
        end

        if isfield(M,'non_solve_time')

            fprintf('  total non-linprog   = %.4f 秒\n', ...
                M.non_solve_time);
        end

        if isfield(M,'exact_audit_time') && ...
                M.exact_audit_time > 0

            fprintf([ ...
                '  exact audit          = %.4f 秒 ', ...
                '（correctness diagnostic，不计入 fair raw time）\n'], ...
                M.exact_audit_time);
        end

        if isfield(M,'lp_num_variables') && ...
                ~isempty(M.lp_num_variables)

            valid_vars = ...
                M.lp_num_variables( ...
                    M.lp_num_variables>0);

            if ~isempty(valid_vars)

                fprintf([ ...
                    'LP 规模：首个子问题变量数=%d | ', ...
                    '最大变量数=%d\n'], ...
                    valid_vars(1), ...
                    max(valid_vars));
            end
        end

        if isfield(M,'lp_num_inequalities') && ...
                ~isempty(M.lp_num_inequalities)

            valid_ineq = ...
                M.lp_num_inequalities( ...
                    M.lp_num_inequalities>0);

            if ~isempty(valid_ineq)

                fprintf([ ...
                    'LP 规模：首个子问题 inequality 数=%d | ', ...
                    '最大 inequality 数=%d\n'], ...
                    valid_ineq(1), ...
                    max(valid_ineq));
            end
        end

        if isfield(M,'lp_feasibility_residual') && ...
                ~isempty(M.lp_feasibility_residual)

            v = ...
                M.lp_feasibility_residual( ...
                    isfinite(M.lp_feasibility_residual));

            if ~isempty(v)

                fprintf('*最大 LP feasibility residual = %.4f*\n', ...
                    max(v));
            end
        end

        if isfield(M,'relaxed_cardinality_residual') && ...
                ~isempty(M.relaxed_cardinality_residual)

            v = ...
                M.relaxed_cardinality_residual( ...
                    isfinite(M.relaxed_cardinality_residual));

            if ~isempty(v)

                fprintf('*最大 relaxed cardinality residual = %.4f*\n', ...
                    max(v));
            end
        end

        if isfield(M,'lp_objective_per_batch') && ...
                isfield(M,'rounded_iqp_objective_per_batch')

            lp_obj = ...
                M.lp_objective_per_batch;

            iq_obj = ...
                M.rounded_iqp_objective_per_batch;

            valid_obj = ...
                isfinite(lp_obj) & ...
                isfinite(iq_obj);

            if any(valid_obj)

                first_id = ...
                    find(valid_obj,1,'first');

                fprintf([ ...
                    '首个 LP 子问题：', ...
                    'relaxation objective = %.4f | ', ...
                    'rounded IQP objective = %.4f\n'], ...
                    lp_obj(first_id), ...
                    iq_obj(first_id));
            end
        end

        if cfg.two_opt.enabled && ...
                cfg.mmdlp.use_common_2opt

            fprintf('*MMD-LP + common2opt J_common = %.4f*\n', ...
                M.refined_jcommon);

            fprintf([ ...
                'common2opt = %.4f 秒 | ', ...
                '*fair total time = %.4f 秒*\n'], ...
                M.two_opt_time, ...
                M.total_time);
        end
    end

    %% ============================================================
    % Random
    % ============================================================

    if isfield(results,'random')

        R = results.random;

        rid = ...
            R.best_raw_id;

        raw_time = ...
            sum(R.gen_time,'omitnan');

        fprintf('\n');
        fprintf('[Random]\n');

        fprintf('*best-of-%d raw J_common = %.4f*\n', ...
            numel(R.raw_jcommon), ...
            R.raw_jcommon(rid));

        fprintf('*fair raw method time = %.4f 秒*\n', ...
            raw_time);

        if cfg.two_opt.enabled

            r2 = ...
                R.best_refined_id;

            post_time = ...
                sum(R.local_time,'omitnan');

            fprintf('*best +common2opt J_common = %.4f*\n', ...
                R.refined_jcommon(r2));

            fprintf([ ...
                'common2opt 累计时间 = %.4f 秒 | ', ...
                '*fair total time = %.4f 秒*\n'], ...
                post_time, ...
                R.wall_time);
        end
    end

    %% ============================================================
    % Vector
    % ============================================================

    if isfield(results,'vector')

        V = results.vector;

        raw_time = ...
            V.solve_time + ...
            V.extract_time;

        fprintf('\n');
        fprintf('[Vector]\n');

        fprintf('*raw J_common = %.4f*\n', ...
            V.raw_jcommon);

        fprintf('iterations = %d\n', ...
            V.iterations);

        fprintf('*fair raw method time = %.4f 秒*\n', ...
            raw_time);

        fprintf('\n');
        fprintf('Vector 内部 profiling：\n');

        fprintf('  split-ADMM time     = %.4f 秒\n', ...
            V.solve_time);

        fprintf('  build/extract time  = %.4f 秒\n', ...
            V.extract_time);

        fprintf([ ...
            'exact-binary batches = %d/%d | ', ...
            'fallback batches = %d\n'], ...
            V.exact_batches, ...
            problem.num_batches, ...
            V.fallback_batches);

        if cfg.two_opt.enabled && ...
                cfg.vector.use_common_2opt

            fprintf('*Vector+common2opt J_common = %.4f*\n', ...
                V.refined_jcommon);

            fprintf([ ...
                'common2opt = %.4f 秒 | ', ...
                '*fair total time = %.4f 秒*\n'], ...
                V.two_opt_time, ...
                V.total_time);
        end
    end

    %% ============================================================
    % Matrix
    % ============================================================

    if isfield(results,'matrix')

        M = results.matrix;

        bid = ...
            M.best_raw_id;

        B = ...
            M.raw{bid};

        ids = ...
            M.valid_ids(:)';

        solver_time = ...
            sum(M.solve_time(ids),'omitnan');

        extract_time = ...
            sum(M.extract_time(ids),'omitnan');

        raw_time = ...
            M.warmstart_overhead_time + ...
            solver_time + ...
            extract_time;

        fprintf('\n');
        fprintf('[Matrix]\n');

        fprintf('初始化方式 = %s。\n', ...
            string(M.init_mode));

        if M.warmstart.is_external

            fprintf('warm start = %s。\n', ...
                string(M.warmstart_label));

            fprintf([ ...
                'warm-start upstream + conversion = %.4f 秒。\n'], ...
                M.warmstart_overhead_time);
        end

        fprintf('*best raw J_common = %.4f*\n', ...
            M.raw_jcommon(bid));

        fprintf('*fair raw method time = %.4f 秒*\n', ...
            raw_time);

        fprintf('\n');
        fprintf('Matrix 内部 profiling：\n');

        fprintf('  solver time         = %.4f 秒\n', ...
            solver_time);

        fprintf('  discrete extraction = %.4f 秒\n', ...
            extract_time);

        if M.warmstart.is_external

            fprintf('  warm-start overhead = %.4f 秒\n', ...
                M.warmstart_overhead_time);
        end

        fprintf([ ...
            'best start iterations = %d | ', ...
            'solver sec/iter = %.4f 秒\n'], ...
            B.iterations, ...
            B.avg_iter_time);

        if isfield(B,'early_exist_triggered')

            fprintf('early_exist triggered = %d', ...
                B.early_exist_triggered);

            if B.early_exist_triggered

                fprintf(' | trigger iter = %d', ...
                    B.early_exist_iteration);
            end

            fprintf('\n');
        end

        if cfg.two_opt.enabled

            m2 = ...
                M.best_refined_id;

            total_local = ...
                sum(M.local_time(ids),'omitnan');

            fprintf('*best +common2opt J_common = %.4f*\n', ...
                M.refined_jcommon(m2));

            fprintf([ ...
                'common2opt = %.4f 秒 | ', ...
                '*fair total time = %.4f 秒*\n'], ...
                total_local, ...
                raw_time+total_local);
        end
    end

    %% ============================================================
    % Shared problem construction
    % ============================================================

    fprintf('\n');

    fprintf([ ...
        '共享 problem 构造时间 = %.4f 秒', ...
        '（所有方法共用，不计入任一方法 runtime）。\n'], ...
        problem.build_time);

    %% ============================================================
    % Nested helper: Lp variant report
    % ============================================================

    function print_lp_variant(V,name)

        ids = ...
            V.valid_ids(:)';

        bid = ...
            V.best_raw_id;

        raw_time = ...
            sum(V.solve_time(ids),'omitnan');

        total_inner = ...
            sum(V.inner_iterations(ids),'omitnan');

        fprintf('\n');

        fprintf('*%s best raw J_common = %.4f*\n', ...
            name, ...
            V.raw_jcommon(bid));

        fprintf('*fair raw method time = %.4f 秒*\n', ...
            raw_time);

        fprintf('累计 inner iterations = %d\n', ...
            total_inner);

        if total_inner > 0

            fprintf('平均完整 inner 时间 = %.4f 秒\n', ...
                raw_time/total_inner);
        end

        if isfield(V,'profile_sum') && ...
                ~isempty(fieldnames(V.profile_sum))

            P = V.profile_sum;

            fprintf('%s 内部 profiling：\n',name);

            if isfield(P,'projection_time')

                fprintf('  projection   = %.4f 秒\n', ...
                    P.projection_time);
            end

            if isfield(P,'rounding_time')

                fprintf('  rounding     = %.4f 秒\n', ...
                    P.rounding_time);
            end

            if isfield(P,'internal_local_search_time')

                fprintf('  internal N2  = %.4f 秒\n', ...
                    P.internal_local_search_time);
            end

            if isfield(P,'line_search_time')

                fprintf('  line search  = %.4f 秒\n', ...
                    P.line_search_time);
            end

            if isfield(P,'other_time')

                fprintf('  other        = %.4f 秒\n', ...
                    P.other_time);
            end
        end

        if cfg.lp.use_common_2opt

            b2 = ...
                V.best_refined_id;

            post_time = ...
                sum(V.local_time(ids),'omitnan');

            fprintf('*%s +common2opt J_common = %.4f*\n', ...
                name, ...
                V.refined_jcommon(b2));

            fprintf([ ...
                'common2opt = %.4f 秒 | ', ...
                '*fair total time = %.4f 秒*\n'], ...
                post_time, ...
                raw_time+post_time);
        end
    end

    %% ============================================================
    % Nested helper: fixed-width four-decimal printing
    % ============================================================

    function print_float(x,width)

        if isnan(x)

            fprintf(['%',num2str(width),'s '],'-');

        elseif isinf(x)

            if x > 0
                fprintf(['%',num2str(width),'s '],'Inf');
            else
                fprintf(['%',num2str(width),'s '],'-Inf');
            end

        else

            fmt = ...
                ['%',num2str(width),'.4f '];

            fprintf(fmt,x);
        end
    end
end