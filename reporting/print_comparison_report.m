function print_comparison_report(report, results, problem, cfg)
%PRINT_COMPARISON_REPORT 中文控制台报告。
%
% 约定：
%   1) 带 * 的项目是建议重点查看的核心比较指标；
%   2) 所有跨方法解质量统一使用 J_common；
%   3) Matrix / Lp / eADMM 的“每次迭代时间”分别对应各自核心迭代，
%      Random 不是迭代算法，因此该项记为“不适用”。

    fprintf('\n======================================================================\n');
    fprintf('★ 重点结果摘要：优先看带 * 的指标 ★\n');
    fprintf('重点指标：J_common、迭代数、平均每次迭代时间、总时间。\n');
    fprintf('注意：J_common 越小越好；总时间均按“生成可行解”的口径说明。\n');
    fprintf('======================================================================\n');

    % ======================== Matrix core summary =======================
    M = results.matrix;
    bid = M.best_raw_id;
    bestM = M.raw{bid};
    matrix_best_raw_core_total = M.solve_time(bid) + M.extract_time(bid);
    matrix_best_raw_e2e_total = M.warmstart_overhead_time + matrix_best_raw_core_total;
    matrix_all_raw_core_total = sum(M.solve_time(M.valid_ids),'omitnan') + ...
        sum(M.extract_time(M.valid_ids),'omitnan');
    matrix_all_raw_e2e_total = M.warmstart_overhead_time + matrix_all_raw_core_total;

    fprintf('\n[Matrix：不含统一 2-opt 的 raw 结果]\n');
    fprintf('说明：这里选取多初值中 J_common 最小的 raw 解。\n');
    fprintf('Matrix 初始化方式 = %s。\n',char(M.init_mode));
    if M.warmstart.is_external
        fprintf('Matrix warm start 来源 = %s。\n',char(M.warmstart_label));
        fprintf('Matrix warm start 上游生成时间 = %.6f 秒；转换时间 = %.6f 秒。\n', ...
            M.warmstart_source_time,M.warmstart_conversion_time);
    end
    fprintf('*Matrix J_common = %.10e*\n', M.raw_jcommon(bid));
    fprintf('*Matrix 迭代数 = %d*\n', bestM.iterations);
    fprintf('*Matrix 每次迭代平均时间 = %.6e 秒/iter*\n', bestM.avg_iter_time);
    fprintf('*Matrix 自身 raw 总时间 = %.6f 秒*（Matrix求解+离散解提取）\n', ...
        matrix_best_raw_core_total);
    if M.warmstart.is_external
        fprintf('*Matrix 含 warm-start 生成的端到端 raw 总时间 = %.6f 秒*。\n', ...
            matrix_best_raw_e2e_total);
    end
    fprintf('Matrix 为获得 best-of-%d raw 解的自身累计时间 = %.6f 秒。\n', ...
        numel(M.valid_ids), matrix_all_raw_core_total);
    if M.warmstart.is_external
        fprintf('Matrix best-of-%d 含 warm-start 的端到端累计时间 = %.6f 秒。\n', ...
            numel(M.valid_ids),matrix_all_raw_e2e_total);
    end
    if M.warmstart.is_external
        fprintf('Matrix 最佳 start = %d；初始化来自外部 warm start，无初始化随机种子。\n',bid);
    else
        fprintf('Matrix 最佳 start / 初始化随机种子 = %d / %d。\n', ...
            bid, M.matrix_init_seed(bid));
    end

    % ======================== Lp core summary ===========================
    if isfield(results,'lp') && results.lp.enabled
        L = results.lp;
        if L.alg2.enabled
            print_lp_key_summary(L.alg2,'Lp-Alg2（完整 Algorithm 2，含内部 N2）');
        end
        if L.bs.enabled
            print_lp_key_summary(L.bs,'Lp-bs（关闭内部 N2）');
        end
    end

    % ======================== Random core summary =======================
    R = results.random;
    rid = R.best_raw_id;
    random_all_raw_total = sum(R.gen_time,'omitnan');
    fprintf('\n[Random balanced：不含统一 2-opt 的 raw 结果]\n');
    fprintf('说明：Random 只是生成平衡随机划分，不存在优化迭代。\n');
    fprintf('*Random best-of-%d J_common = %.10e*\n',numel(R.raw_jcommon),R.raw_jcommon(rid));
    fprintf('*Random 迭代数 = 不适用*（非迭代算法）\n');
    fprintf('*Random 每次迭代时间 = 不适用*（非迭代算法）\n');
    fprintf('Random 最佳候选本身的生成时间 = %.6f 秒。\n',R.gen_time(rid));
    fprintf('*Random best-of-%d 的累计生成总时间 = %.6f 秒*\n', ...
        numel(R.raw_jcommon),random_all_raw_total);
    fprintf('Random 最佳候选编号 / 随机种子 = %d / %d。\n',rid,R.partition_seed(rid));

    % ======================== eADMM core summary ========================
    E = results.eadmm;
    fprintf('\n[eADMM：Base DNN + vector-clustering rounding，不含统一 2-opt]\n');
    fprintf('说明：迭代数和每 iter 时间对应 base DNN 的 aadmm_3b 求解；总时间还包含 rounding。\n');
    fprintf('*eADMM J_common = %.10e*（%d 次 rounding 中最好）\n', ...
        E.best_vc_jcommon,numel(E.vc_jcommon));
    fprintf('*eADMM DNN 迭代数 = %d*\n',E.dnn_iterations);
    fprintf('*eADMM 每次 DNN 迭代平均时间 = %.6e 秒/iter*\n',E.dnn_avg_iter_time);
    fprintf('*eADMM raw 解生成总时间 = %.6f 秒*（DNN求解+全部rounding）\n', ...
        E.solution_time_raw);
    fprintf('其中：DNN 求解时间 = %.6f 秒；全部 rounding 时间 = %.6f 秒。\n', ...
        E.solve_time,E.vc_time);
    fprintf('eADMM 最佳 rounding restart = %d。\n',E.best_vc_restart);

    fprintf('\n----------------------------------------------------------------------\n');
    fprintf('上面是最需要看的核心结果。下面是详细诊断和公平性分析。\n');
    fprintf('----------------------------------------------------------------------\n');

    % ======================== Objective convention =====================
    fprintf('\n==================== 目标函数口径 ====================\n');
    fprintf('所有方法的解质量统一比较：\n');
    fprintf('  *J_common(P) = <Phi,P*P''>/(2*bs^2)*\n');
    fprintf('J_common 越小越好。eADMM 内部可以使用图/Laplacian尺度，但最终全部换回 J_common。\n');

    fprintf('\nMatrix 完整离散目标分解（仅用于诊断，不用于跨方法比较）：\n');
    fprintf('  F_matrix(P) = J_common(P) + C_linear + C_Lhalf\n');
    fprintf('  C_linear = %.10e\n',report.C_linear);
    fprintf('  C_Lhalf  = eta*N = %g * %d = %.10e\n', ...
        cfg.matrix.eta,problem.num_samples,report.C_lhalf);
    fprintf('  常数偏移 = %.10e\n',report.C_total);

    % ======================== Main comparison table ====================
    fprintf('\n==================== 最终 raw / 相同 2-opt 对比表 ====================\n');
    fprintf('说明：这张表同时保留 raw 和统一 2-opt 后的结果。\n');
    fprintf('时间列只统计“解生成”时间；MET / lower-bound 认证时间不计入方法运行时间。\n');
    disp(report.summary_table);
    fprintf(['列说明：JCommon=统一目标；CertifiedGap=相对同一个安全下界的 gap；', ...
        'SolveTimeSec=核心求解器时间；InitExtractTimeSec=初始化/rounding/解提取时间；', ...
        '对外部 warm-start 的 Matrix，该列也计入生成 warm start 所需的上游时间；', ...
        'LocalSearchTimeSec=统一2-opt时间；TotalMethodTimeSec=端到端解生成总时间。\n']);

    % ======================== Init vs 2opt ==============================
    fprintf('\n==================== 初始化质量 vs 2-opt 诊断 ====================\n');
    fprintf('说明：用于判断高质量解来自前面的 solver，还是主要来自后续 2-opt。\n');
    disp(report.family_table);
    fprintf('Lp-Alg2 的 raw 已经包含论文内部 N2；Lp-bs 的 raw 不包含内部 N2。\n');
    if isfinite(report.post_best_spread_pct)
        fprintf('统一2-opt后，各方法 BEST J_common 的相对跨度 = %.6f%%。\n',report.post_best_spread_pct);
        fprintf('统一2-opt后，各方法 MEAN J_common 的相对跨度 = %.6f%%。\n',report.post_mean_spread_pct);
    end

    % ======================== Equal time ================================
    fprintf('\n==================== 相同时间预算下的 +2opt 结果 ====================\n');
    fprintf('说明：每一行表示在给定 wall-clock 预算内，各方法已经完成的最好 +2opt 候选。\n');
    fprintf('eADMM 的时间预算只计 DNN + rounding + 2opt，不计 MET / lower-bound 认证。\n');
    if M.warmstart.is_external
        fprintf('Matrix 的时间预算包含 warm-start 上游生成/转换时间，避免把初始化成本隐藏掉。\n');
    end
    disp(report.time_budget_table);
    fprintf('共享问题构造时间 = %.6f 秒（只付一次，不计入各方法运行时间）。\n',problem.build_time);

    % ======================== Settings =================================
    fprintf('\n==================== 初始化与参数设置 ====================\n');
    fprintf('Matrix 初始化方式 = %s。\n', char(string(cfg.matrix.init_mode)));
    if string(cfg.matrix.init_mode)=="lp_best_jcommon"
        fprintf('Matrix LP warm-start 变体 = %s。\n',char(string(cfg.matrix.lp_init_variant)));
    end
    if M.warmstart.is_external
        fprintf('Matrix warm-start 具体来源 = %s。\n',char(M.warmstart_label));
    end
    fprintf('eADMM 初始化方式 = %s。\n', char(string(cfg.eadmm.init_mode)));
    if cfg.methods.enabled.lp
        fprintf('Lp 参数：p=%.3g，sigma_minus=%.3g，eps0=%.3g，eps_min=%.3g。\n', ...
            cfg.lp.p,cfg.lp.sigma_minus,cfg.lp.eps0,cfg.lp.eps_min);
        fprintf('Lp 是否使用曲率构造 sigma0 = %d。\n',cfg.lp.use_curvature_sigma0);
        fprintf('Lp-Alg2 是否运行 = %d；内部 N2 = %d。\n', ...
            cfg.lp.run_alg2,cfg.lp.do_internal_local_search);
        fprintf('Lp-bs 是否运行 = %d；内部 N2 强制关闭。\n',cfg.lp.run_bs);
    end
    fprintf('Matrix beta 模式 = %s；自适应规则 = %s。\n', ...
        char(string(cfg.matrix.beta_mode)),char(string(cfg.matrix.beta_adapt_rule)));

    % ======================== Iter timing ===============================
    fprintf('\n==================== 各方法迭代时间诊断 ====================\n');
    fprintf('说明：重点看 Iterations、AvgTimePerIterSec、MeanSolveTimeSec。\n');
    disp(report.iteration_timing_table);
    fprintf(['Matrix 行对有效 starts 汇总；Lp 的 iteration 指 inner iteration；', ...
        'eADMM 行对应 base DNN iteration。\n']);

    if isfield(report,'lp_component_timing_table') && ~isempty(report.lp_component_timing_table)
        fprintf('\n==================== Lp 各模块耗时 ====================\n');
        fprintf('说明：用来判断 Lp 时间主要花在投影、线搜索、rounding 还是内部 N2。\n');
        disp(report.lp_component_timing_table);
        fprintf(['ProjectionTimeSec=平衡多面体投影；LineSearchTimeSec=回溯线搜索；', ...
            'RoundingTimeSec=greedy balanced rounding；InternalN2TimeSec=Algorithm 2内部N2。\n']);
    end

    % ======================== Certificate ===============================
    fprintf('\n==================== 安全下界与 certified gap ====================\n');
    fprintf('*安全 J_common 下界 = %.10e*\n',report.safe_jcommon_lower_bound);
    fprintf('*本次实验找到的最好可行 J_common = %.10e*\n',report.best_feasible_jcommon);
    if report.safe_lower_bound_valid
        fprintf('*最好解的 certified absolute gap = %.10e*\n',report.best_certified_abs_gap);
        fprintf('*最好解的 certified relative gap = %.6f%%*\n',report.best_certified_gap_pct);
        fprintf('gap 定义：100*(UB_Jcommon-LB_Jcommon)/abs(UB_Jcommon)。\n');
    else
        fprintf('certified gap 不可用：后处理得到的 J_common 下界未通过数值一致性检查。\n');
    end
    fprintf('eADMM MET 认证时间 = %.6f 秒；LB 后处理时间 = %.6f 秒。\n', ...
        report.certification_met_time,report.safe_lower_bound_postprocess_time);
    fprintf('这些认证时间属于 instance-level certificate 开销，不计入方法解生成时间。\n');

    fprintf('\n[eADMM 数值诊断]\n');
    fprintf('DNN relaxed J_common（诊断值，不是安全下界）= %.10e。\n',E.sdp_jcommon_score);
    fprintf('DNN primal equality 相对残差 = %.6e。\n',E.primal_eq_rel);
    fprintf('DNN dual equality 相对残差 = %.6e。\n',E.dual_eq_rel);
    fprintf('DNN X 最小特征值 = %.6e。\n',E.min_eig_X);
    fprintf('DNN 非负约束违反量 = %.6e。\n',E.nonneg_violation);
    fprintf('安全下界来源 = %s。\n',char(E.safe_lower_bound_source));
    if E.met_enabled
        fprintf('MET separation 扫描次数 = %d；强化 SDP 重求解次数 = %d；累计 cuts = %d。\n', ...
            E.met_separation_calls,E.met_rounds,E.met_num_constraints);
        fprintf('MET separation 时间 = %.6f 秒；强化 SDP 求解时间 = %.6f 秒；MET 总时间 = %.6f 秒。\n', ...
            E.met_separation_time,E.met_solve_time,E.met_time);
    end

    % ======================== Per-start details =========================
    fprintf('\n==================== 各 start / restart 的详细结果 ====================\n');
    fprintf('\n[Matrix 各 start：raw -> 相同 2-opt]\n');
    if M.warmstart.is_external
        fprintf('说明：下面 solve 只计 Matrix 自身；共享 warm-start 上游开销 %.6f 秒只在端到端总时间中计一次。\n', ...
            M.warmstart_overhead_time);
    end
    for rr=M.valid_ids(:)'
        if M.warmstart.is_external
            fprintf('Matrix start=%2d | external warm start | *J_common(raw)=%.10e* | *iter=%d* | *sec/iter=%.3e* | *solve=%.4f秒*', ...
                rr,M.raw_jcommon(rr),M.iterations(rr),M.avg_iter_time(rr),M.solve_time(rr));
        else
            fprintf('Matrix start=%2d | seed=%d | *J_common(raw)=%.10e* | *iter=%d* | *sec/iter=%.3e* | *solve=%.4f秒*', ...
                rr,M.matrix_init_seed(rr),M.raw_jcommon(rr),M.iterations(rr),M.avg_iter_time(rr),M.solve_time(rr));
        end
        if cfg.two_opt.enabled
            fprintf(' | J_common(+2opt)=%.10e | 2opt=%.4f秒',M.refined_jcommon(rr),M.local_time(rr));
        end
        fprintf('\n');
    end

    if isfield(results,'lp') && results.lp.enabled
        L=results.lp;
        fprintf('\n[Lp objective / 曲率诊断]\n');
        fprintf('连续目标定义：%s。\n',L.objective_definition);
        fprintf('曲率下界 = %.10e。\n',L.curvature_lower);
        if L.alg2.enabled
            print_lp_variant(L.alg2,'Lp-Alg2');
        end
        if L.bs.enabled
            print_lp_variant(L.bs,'Lp-bs');
        end
    end

    fprintf('\n[Random 前 10 个候选：raw -> 相同 2-opt]\n');
    for rr=1:min(10,numel(R.raw_jcommon))
        fprintf('Random restart=%2d | seed=%d | *J_common(raw)=%.10e* | 生成时间=%.6f秒', ...
            rr,R.partition_seed(rr),R.raw_jcommon(rr),R.gen_time(rr));
        if cfg.two_opt.enabled
            fprintf(' | J_common(+2opt)=%.10e | 2opt=%.4f秒',R.refined_jcommon(rr),R.local_time(rr));
        end
        fprintf('\n');
    end

    if ~isempty(E.checkpoints)
        fprintf('\n[eADMM rounding checkpoint]\n');
        fprintf('说明：restart=rounding次数；Vc=2-opt前；Vc+2opt=统一2-opt后。\n');
        for q=1:numel(E.checkpoints)
            cp=E.checkpoints(q);
            fprintf('restart=%d | *best Vc J_common=%.10e* | best Vc+2opt J_common=%.10e | rounding累计=%.6f秒 | 2opt累计=%.6f秒\n', ...
                cp.restart,cp.vc_best_jcommon,cp.vc2_best_jcommon,cp.vc_time,cp.two_opt_time);
        end
    end

    fprintf('\n==================== 阅读规则 ====================\n');
    fprintf('1) 带 * 的量是建议重点比较的指标。\n');
    fprintf('2) 所有方法解质量统一看 J_common；越小越好。\n');
    fprintf('3) Matrix / Random / eADMM 的 raw 均在统一 2-opt 之前。\n');
    fprintf('4) Lp-Alg2 raw 已包含论文内部 N2；Lp-bs raw 不包含内部 N2。\n');
    fprintf('5) certified gap 对所有方法使用同一个安全 J_common 下界。\n');
    fprintf('6) MET 和 LB post-processing 是认证开销，不计入 eADMM 的解生成时间。\n');

    function print_lp_key_summary(V,name)
        ids=V.valid_ids(:)';
        bidv=V.best_raw_id;
        inner=V.inner_iterations(bidv);
        avg_inner=V.solve_time(bidv)/max(inner,1);
        fprintf('\n[%s]\n',name);
        fprintf('*%s J_common = %.10e*\n',name,V.raw_jcommon(bidv));
        fprintf('%s outer 迭代数 = %d。\n',name,V.outer_iterations(bidv));
        fprintf('*%s inner 迭代数 = %d*\n',name,inner);
        fprintf('*%s 每个 inner 迭代平均时间 = %.6e 秒/iter*\n',name,avg_inner);
        fprintf('*%s 最佳 start 求解总时间 = %.6f 秒*\n',name,V.solve_time(bidv));
        fprintf('%s 为获得 best-of-%d 的累计求解总时间 = %.6f 秒。\n', ...
            name,numel(ids),sum(V.solve_time(ids),'omitnan'));
    end

    function print_lp_variant(V,name)
        fprintf('\n[%s 各 start]\n',name);
        for rr=V.valid_ids(:)'
            avg_inner=V.solve_time(rr)/max(V.inner_iterations(rr),1);
            fprintf('%s start=%2d | seed=%d | *J_common=%.10e* | outer=%d | *inner=%d* | *sec/inner=%.3e* | *solve=%.4f秒*', ...
                name,rr,V.perturb_seed(rr),V.raw_jcommon(rr),V.outer_iterations(rr), ...
                V.inner_iterations(rr),avg_inner,V.solve_time(rr));
            if cfg.lp.use_common_2opt
                fprintf(' | J_common(+common2opt)=%.10e | 2opt=%.4f秒', ...
                    V.refined_jcommon(rr),V.local_time(rr));
            end
            if ~isempty(V.component_profile{rr})
                pp=V.component_profile{rr};
                fprintf(' | 投影=%.3f秒 | rounding=%.3f秒 | 内部N2=%.3f秒 | 线搜索=%.3f秒 | 其他=%.3f秒', ...
                    pp.projection_time,pp.rounding_time,pp.internal_local_search_time, ...
                    pp.line_search_time,pp.other_time);
            end
            fprintf('\n');
        end
    end
end
