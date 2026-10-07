function result = solve_eadmm_batching(problem, opts)
%SOLVE_EADMM_BATCHING Full Wiegele-Zhao (2022) k-equipartition baseline.
%
% Project-wide quality is always measured by
%   J_common(P) = <Phi,P*P'>/(2*bs^2).
%
% Inside this module only, Phi is translated to the equivalent graph model
% required by ADMM-GP. The author's full lower-bound branch is used:
%
%   1) solve the DNN relaxation with aadmm_3b;
%   2) repeatedly separate the most violated MET/transitivity inequalities;
%   3) re-solve the strengthened SDP with mprw_ineq_general;
%   4) obtain a safe lower bound with post_proc_3 (or post_proc_2 if no MET
%      inequality is found), then convert that bound to J_common.
%
% Consistent with Sect. 6.3.3 of Wiegele-Zhao, the upper-bound heuristic
% (vector-clustering rounding + 2-opt) uses the DNN solution, not the
% MET-strengthened SDP solution. Thus MET strengthens the certified lower bound
% without changing the rounding branch.

    N = problem.num_samples;
    B = problem.num_batches;
    bs = problem.batch_size;

    % eADMM-specific translation only.
    W_eadmm = 1 - problem.Phi;
    W_eadmm(1:N+1:end) = 0;
    W_eadmm = (W_eadmm + W_eadmm')/2;

    % For every equipartition SDP-feasible X,
    %   <L,X>/2 = bs^2*J_common(X) + c_affine.
    graph_to_jcommon_constant = 0.5*(sum(W_eadmm,'all') - bs*N);

    checkpoints = unique(round(opts.rounding_checkpoints(:)'));
    checkpoints = checkpoints(checkpoints>=1 & checkpoints<=opts.rounding_restarts);
    if isempty(checkpoints) || checkpoints(end) ~= opts.rounding_restarts
        checkpoints = unique([checkpoints, opts.rounding_restarts]);
    end

    fprintf('开始运行 eADMM-SDP（Wiegele-Zhao 2022）。\n');
    fprintf('eADMM-SDP 模型规模：n=%d，分组数k=%d，batch size=%d。\n',N,B,bs);
    fprintf('eADMM DNN 初始化方式 = %s。\n', char(string(opts.init_mode)));

    %% ======================== Base DNN model ==============================
    [A_sdp,b_sdp,C_sdp] = kequi_form(W_eadmm, int16(B));

    % Internal graph/J_common affine conversion check.
    Pcheck = zeros(N,B);
    for jj = 1:B
        idx = (jj-1)*bs + (1:bs);
        Pcheck(idx,jj) = 1;
    end
    Hcheck = Pcheck*Pcheck';
    graph_check_internal = sum(C_sdp.*Hcheck,'all');
    jcommon_check = sum(problem.Phi.*Hcheck,'all')/(2*bs^2);
    affine_err = abs(graph_check_internal - ...
        (bs^2*jcommon_check + graph_to_jcommon_constant));
    assert(affine_err <= 1e-9*max(1,abs(graph_check_internal)), ...
        'Internal eADMM graph/J_common conversion failed (err %.3e).',affine_err);

    fprintf(['eADMM 参数：最大迭代数=%d，sigma0=%g，停止阈值=%g，', ...
        'rounding次数=%d，是否使用统一2-opt=%d。\n'], opts.max_iter, opts.sigma0, opts.tol, ...
        opts.rounding_restarts, opts.use_2opt);
    fprintf('eADMM 随机种子基数：rounding=%d，统一2-opt=%d。\n', ...
        opts.rounding_seed_base, opts.two_opt_seed_base);
    if opts.met.enabled
        fprintf(['MET 强化已开启：每轮最多cuts=%d，违反阈值=%g，', ...
            '最大轮数=%g，sigma0=%g。\n'], ...
            opts.met.max_ineq_per_round,opts.met.violation_tol, ...
            opts.met.max_rounds,opts.met.sigma0);
    else
        fprintf('MET 强化已关闭。\n');
    end

    %% ======================== Solve DNN relaxation ========================
    fprintf('\n[eADMM 第1阶段/4] 求解基础 DNN 松弛。\n');
    fprintf('下面打印的是作者原始 aadmm_3b 的 DNN 迭代表；这里还没有加入 MET cuts。\n');
    fprintf(['原始迭代表列含义：iter=迭代号；累计时间=作者代码内部计时；dual/primal=图尺度SDP目标；', ...
        'log10残差=对偶/原始相对残差的log10；sigma=当前罚参数。\n']);
    fprintf('Cholesky 预处理只发生一次；正式跨方法时间比较使用外层 MATLAB tic/toc。\n');
    tdnn = tic;
    switch string(opts.init_mode)
        case "zero"
            [X_dnn,y_dnn,Z_dnn,S_dnn,~,~,sigma_dnn_final,~,~,dnn_iterations] = ...
                aadmm_3b(A_sdp,b_sdp,C_sdp,opts.max_iter,opts.sigma0,opts.tol);

        case "uniform"
            % Uniform feasible co-membership point on the original DNN scale:
            % diag(H0)=1 and H0*1=bs*1.  aadmm_3b internally rescales A,b,
            % so convert H0 to the solver's internal primal scaling first.
            rho0 = (bs-1)/(N-1);
            H0 = rho0*ones(N,N) + (1-rho0)*eye(N);
            normA0 = full(min(1e12,max(1,norm(A_sdp,'fro'))));
            normb0 = max(1,norm(b_sdp));
            Y0 = H0 * normA0 / normb0;
            Z0 = zeros(N,N);
            S0 = zeros(N,N);
            [X_dnn,y_dnn,Z_dnn,S_dnn,~,~,sigma_dnn_final,~,~,dnn_iterations] = ...
                aadmm_3b(A_sdp,b_sdp,C_sdp,opts.max_iter,opts.sigma0,opts.tol,Y0,Z0,S0);

        otherwise
            error('Unknown eADMM init_mode: %s', string(opts.init_mode));
    end
    dnn_solve_time = toc(tdnn);
    dnn_avg_iter_time = dnn_solve_time / max(dnn_iterations,1);
    fprintf('[eADMM-DNN完成] *DNN求解总时间 = %.6f 秒*。\n',dnn_solve_time);
    fprintf('[eADMM-DNN时间] *迭代数 = %d*；*每次迭代平均时间 = %.6e 秒/iter*。\n', ...
        dnn_iterations,dnn_avg_iter_time);

    % DNN diagnostics in original scale.
    if size(A_sdp,1) == numel(X_dnn)
        Aop = A_sdp';
    else
        Aop = A_sdp;
    end
    dnn_primal_eq_rel = norm(Aop*X_dnn(:)-b_sdp)/(1+norm(b_sdp));
    Aty_dnn = reshape(Aop'*y_dnn,N,N);
    dnn_dual_res_mat = C_sdp - Aty_dnn - Z_dnn - S_dnn;
    dnn_dual_eq_rel = norm(dnn_dual_res_mat,'fro')/(1+norm(C_sdp,'fro'));
    Xsym = (X_dnn+X_dnn')/2;
    dnn_min_eig_X = min(eig(full(Xsym)));
    dnn_nonneg_violation = norm(min(X_dnn,0),'fro')/(1+norm(X_dnn,'fro'));
    dnn_sdp_jcommon_score = sum(problem.Phi.*X_dnn,'all')/(2*bs^2);

    %% ======================== DNN + MET strengthening =====================
    % The MET branch is for lower-bound tightening. The DNN X above remains the
    % input to the author's rounding heuristic, matching the paper's numerical
    % protocol for upper bounds.
    met_time = 0;
    met_separation_time = 0;
    met_solve_time = 0;
    met_rounds = 0;
    met_separation_calls = 0;
    met_num_constraints = 0;
    met_terminated_no_violation = false;
    met_hit_round_limit = false;
    met_records = struct([]);

    X_met = X_dnn;
    y_met = [];
    ybar_met = [];
    Z_met = Z_dnn;
    S_met = S_dnn;
    s_met = [];
    v_met = [];
    sigma_met_final = NaN;
    B_met = sparse(0,N*N);
    f_met = zeros(0,1);
    T_met = zeros(0,4);
    hash_met = zeros(0,1);

    if opts.met.enabled
        fprintf('\n[eADMM 第2阶段/4] MET 分离 / cutting-plane 下界强化。\n');
        fprintf('MET 只用于加强 SDP 下界；后续 rounding 仍使用基础 DNN 的 X_dnn，因此 MET 时间不计入解生成时间。\n');
        fprintf('MET 每次扫描寻找违反的三角/传递不等式；当前违反阈值 = %.3e。\n',opts.met.violation_tol);
        fprintf('计数说明：只有加入 cuts 后重新求解一次强化 SDP，才计作 1 个 MET round；仅扫描但没发现 cut 不计 round。\n');
        if N > 1000
            error('Upstream MET hash convention supports n <= 1000.');
        end

        L_box = zeros(N,N);
        U_box = Inf(N,N);
        tmet_total = tic;

        while met_rounds < opts.met.max_rounds
            old_num = numel(f_met);
            met_separation_calls = met_separation_calls + 1;
            fprintf('[MET扫描 %d] 正在检查当前 SDP 解是否存在新的违反不等式...\n', ...
                met_separation_calls);
            tsep = tic;
            [f_candidate,T_candidate,hash_candidate,g_new,brk] = ...
                separation_kequi(X_met,f_met,T_met,hash_met, ...
                opts.met.max_ineq_per_round,opts.met.violation_tol);
            sep_elapsed = toc(tsep);
            met_separation_time = met_separation_time + sep_elapsed;

            new_num = numel(f_candidate);
            added = new_num-old_num;
            if added <= 0
                met_terminated_no_violation = true;
                fprintf('[MET扫描 %d 结果] 没有发现新的 cut，因此本轮无需重新求解强化 SDP。\n', ...
                    met_separation_calls);
                break;
            end

            % Accept the newly separated cuts and rebuild the full accumulated
            % inequality operator B(X)>=f.
            f_met = f_candidate;
            T_met = T_candidate;
            hash_met = hash_candidate;
            [B_met,f_met] = formtri_kc(T_met,f_met,N);

            met_rounds = met_rounds + 1;
            fprintf(['[MET第 %d 轮] 新增 %d 个 cuts；累计 cuts=%d；', ...
                '本轮最大违反量=%.3e。\n'], ...
                met_rounds,added,numel(f_met),max(g_new));
            fprintf('下面打印作者原始 mprw_ineq_general 的强化 SDP 迭代表。\n');
            fprintf('列含义与基础 DNN 相同，但此处 primal/dual 与残差均对应当前已加入 MET cuts 的强化 SDP。\n');

            % Upstream inequality eADMM. The public routine initializes its own
            % blocks; we therefore call its standard deterministic interface.
            tms = tic;
            [X_met,y_met,ybar_met,Z_met,S_met,s_met,v_met,sigma_met_final,iter_met] = ...
                mprw_ineq_general(A_sdp,B_met,b_sdp,C_sdp,f_met, ...
                L_box,U_box,opts.max_iter,opts.met.sigma0,opts.tol);
            solve_elapsed = toc(tms);
            met_solve_time = met_solve_time + solve_elapsed;

            BX = B_met*X_met(:);
            met_violation_rel = norm(max(f_met-BX,0))/(1+norm(f_met));
            met_records(met_rounds).round = met_rounds;
            met_records(met_rounds).added = added;
            met_records(met_rounds).total = numel(f_met);
            met_records(met_rounds).max_new_violation = max(g_new);
            met_records(met_rounds).ineq_violation_rel = met_violation_rel;
            met_records(met_rounds).iterations = iter_met;
            met_records(met_rounds).solve_time = solve_elapsed;
            met_records(met_rounds).separation_time = sep_elapsed;
            fprintf('[MET第 %d 轮完成] *强化SDP迭代数=%d*；*求解时间=%.6f秒*；累计cuts相对违反量=%.3e。\n', ...
                met_rounds,iter_met,solve_elapsed,met_violation_rel);

            % separation_kequi sets brk when the newly found violations are
            % already below its requested tolerance. Solve once with those cuts,
            % then stop the loop.
            if brk
                met_terminated_no_violation = true;
                fprintf('[MET停止] 新发现的违反量已低于设定阈值；当前 cuts 已求解一次后停止。\n');
                break;
            end
        end

        met_time = toc(tmet_total);
        met_num_constraints = numel(f_met);
        if met_rounds >= opts.met.max_rounds && ~met_terminated_no_violation
            met_hit_round_limit = true;
            warning('MET separation reached max_rounds=%g with %d accumulated cuts.', ...
                opts.met.max_rounds,met_num_constraints);
        end
        fprintf(['[MET阶段汇总] 扫描次数=%d；强化SDP重求解次数=%d；累计cuts=%d；', ...
            'separation时间=%.6f秒；强化SDP求解时间=%.6f秒；MET分支总时间=%.6f秒。\n'], ...
            met_separation_calls,met_rounds,met_num_constraints, ...
            met_separation_time,met_solve_time,met_time);
        if met_num_constraints == 0
            fprintf(['[MET解释] X_dnn 上没有发现超过阈值的 violated MET cut；', ...
                '因此没有求解强化 SDP，安全下界回退到 DNN/post_proc_2。\n']);
        end
    else
        fprintf('\n[eADMM 第2阶段/4] 配置中已关闭 MET 强化。\n');
    end

    if met_num_constraints > 0
        met_sdp_jcommon_score = sum(problem.Phi.*X_met,'all')/(2*bs^2);
        met_primal_eq_rel = norm(Aop*X_met(:)-b_sdp)/(1+norm(b_sdp));
        met_ineq_violation_rel = norm(max(f_met-B_met*X_met(:),0))/(1+norm(f_met));
        Aty_met = reshape(Aop'*y_met,N,N);
        Bty_met = reshape(B_met'*ybar_met,N,N);
        met_dual_eq_rel = norm(C_sdp-Aty_met-Bty_met-Z_met-S_met,'fro')/(1+norm(C_sdp,'fro'));
    else
        met_sdp_jcommon_score = NaN;
        met_primal_eq_rel = NaN;
        met_ineq_violation_rel = NaN;
        met_dual_eq_rel = NaN;
    end

    %% ======================== Safe lower bound =============================
    safe_lb = struct();
    safe_lb.common = NaN;
    safe_lb.graph_internal = NaN;
    safe_lb.y = [];
    safe_lb.ybar = [];
    safe_lb.time = 0;
    safe_lb.min_eig_Zplus = NaN;
    safe_lb.min_S = NaN;
    safe_lb.min_ybar = NaN;
    safe_lb.dual_identity_rel = NaN;
    safe_lb.numerically_feasible = false;
    safe_lb.source = "none";

    if opts.compute_safe_lower_bound
        fprintf('\n[eADMM 第3阶段/4] 计算安全下界 certificate。\n');
        if met_num_constraints > 0
            fprintf('由于加入了 MET cuts，使用 post_proc_3 修复对偶可行性并计算安全下界。\n');
        else
            fprintf('由于没有累计 MET cut，使用 post_proc_2 修复对偶可行性并计算安全下界。\n');
        end
        fprintf('说明：后面的 numneg/mineigZnew 是安全下界修复过程的数值诊断；linprog 显示“找到最优解”只代表修复LP求解成功，不代表原离散问题全局最优。\n');
        fprintf('后面的 LBnew 仍是 eADMM 内部图尺度，下方会立即转换为统一 J_common 尺度。\n');
        if met_num_constraints > 0
            safe_lb = compute_safe_lower_bound_met( ...
                A_sdp,B_met,b_sdp,f_met,C_sdp,Z_met,S_met, ...
                bs,graph_to_jcommon_constant);
            safe_lb.source = "DNN+MET/post_proc_3";
        else
            safe_lb = compute_safe_lower_bound( ...
                A_sdp,b_sdp,C_sdp,Z_dnn,S_dnn,bs,graph_to_jcommon_constant);
            safe_lb.ybar = [];
            safe_lb.min_ybar = NaN;
            safe_lb.source = "DNN/post_proc_2";
        end
        fprintf(['[Certificate尺度转换] 图尺度安全下界=%.10e；仿射常数=%.10e；', ...
            'batch_size^2=%d；*安全J_common下界=%.10e*。\n'], ...
            safe_lb.graph_internal,graph_to_jcommon_constant,bs^2,safe_lb.common);
        fprintf(['[Certificate数值诊断] min eig(Zplus)=%.3e；最小非负对偶slack=%.3e；', ...
            '对偶恒等式相对残差=%.3e。\n'], ...
            safe_lb.min_eig_Zplus,safe_lb.min_S,safe_lb.dual_identity_rel);
        if isfinite(safe_lb.min_ybar)
            fprintf('[Certificate数值诊断] 最小 MET 对偶乘子 ybar=%.3e（数值容差下应非负）。\n', ...
                safe_lb.min_ybar);
        end
    else
        fprintf('\n[eADMM 第3阶段/4] 安全下界后处理已关闭。\n');
    end

    %% ======================== DNN rounding / upper bounds ==================
    fprintf('\n[eADMM 第4阶段/4] 对基础 DNN 解做 rounding，生成可行离散划分。\n');
    fprintf(['rounding 使用基础 DNN 解 X_dnn（不是 X_met）；', ...
        '每次 restart 生成一个平衡离散划分，并直接计算 J_common。', ...
        'rounding次数=%d。\n'],opts.rounding_restarts);

    if opts.use_2opt
        fprintf(['先从全部 rounding restart 中选择 raw J_common 最好的解，', ...
            '然后仅对该 raw 最优解做一次统一 common 2-opt。\n']);
    else
        fprintf('eADMM rounding 后的统一 J_common 2-opt 已关闭。\n');
    end

    % -------------------------------------------------------------------------
    % Paper protocol:
    % upper-bound heuristic uses the base DNN solution X_dnn.
    % -------------------------------------------------------------------------

    R = opts.rounding_restarts;

    vc_jcommon = nan(R,1);

    % Keep these fields for backward compatibility.
    % Under the new protocol only ONE raw-best solution is postprocessed.
    vc2_jcommon = nan(R,1);

    vc_best_jcommon_history = nan(R,1);
    vc2_best_jcommon_history = nan(R,1);

    cumulative_vc_time = nan(R,1);
    cumulative_2opt_time = zeros(R,1);

    rounding_seed = nan(R,1);
    two_opt_seed = nan(R,1);

    best_vc_jcommon = inf;
    best_vc_restart = NaN;
    best_vc_I = [];
    best_vc_metrics = struct();

    best_vc2_jcommon = NaN;
    best_vc2_restart = NaN;
    best_vc2_I = [];
    best_vc2_metrics = struct();

    vc_time_total = 0;
    two_opt_time_total = 0;

    checkpoint_records = struct([]);
    cp_id = 0;

    %% ========================================================================
    % Step 1: run ALL Vc rounding restarts
    %         NO 2-opt is performed inside this loop
    % ========================================================================

    for rr = 1:R

        rounding_seed_rr = ...
            opts.rounding_seed_base + rr - 1;

        rounding_seed(rr) = rounding_seed_rr;

        % ---------------------------------------------------------------------
        % Vc rounding
        % ---------------------------------------------------------------------

        tvc = tic;

        [~,~,~,part_cell] = ...
            kequi_rounding( ...
                rounding_seed_rr, ...
                X_dnn, ...
                double(B), ...
                C_sdp);

        vc_time_total = ...
            vc_time_total + toc(tvc);

        % ---------------------------------------------------------------------
        % Convert to project partition representation
        % ---------------------------------------------------------------------

        I_rr = ...
            part_cell_to_I( ...
                part_cell, ...
                bs, ...
                B, ...
                N);

        P_rr = ...
            batches_to_assignment( ...
                I_rr, ...
                N, ...
                B, ...
                bs);

        metrics_rr = ...
            evaluate_partition( ...
                P_rr, ...
                problem, ...
                opts.eval_eta);

        vc_jcommon(rr) = ...
            metrics_rr.common_objective;

        % ---------------------------------------------------------------------
        % Select the best RAW rounding solution
        % ---------------------------------------------------------------------

        if vc_jcommon(rr) < best_vc_jcommon

            best_vc_jcommon = ...
                vc_jcommon(rr);

            best_vc_restart = ...
                rr;

            best_vc_I = ...
                I_rr;

            best_vc_metrics = ...
                metrics_rr;
        end

        vc_best_jcommon_history(rr) = ...
            best_vc_jcommon;

        cumulative_vc_time(rr) = ...
            vc_time_total;

        % No 2-opt has happened yet.
        cumulative_2opt_time(rr) = 0;

        % ---------------------------------------------------------------------
        % Raw-rounding checkpoints
        % ---------------------------------------------------------------------

        if any(rr == checkpoints)

            cp_id = cp_id + 1;

            checkpoint_records(cp_id).restart = ...
                rr;

            checkpoint_records(cp_id).vc_best_jcommon = ...
                best_vc_jcommon;

            checkpoint_records(cp_id).vc_time = ...
                vc_time_total;

            % Under the new protocol 2-opt is done only once,
            % after ALL rounding restarts finish.
            checkpoint_records(cp_id).two_opt_time = ...
                0;

            checkpoint_records(cp_id).vc2_best_jcommon = ...
                NaN;
        end
    end


    %% ========================================================================
    % Step 2: apply common 2-opt ONCE to the best RAW rounding solution
    % ========================================================================

    if opts.use_2opt

        if isempty(best_vc_I) || ~isfinite(best_vc_jcommon)
            error('eADMM raw rounding did not produce a valid solution.');
        end

        % Since there is now only ONE common postprocessing call,
        % use the common base seed directly.
        two_opt_seed_rr = ...
            opts.two_opt_seed_base;

        two_opt_seed(best_vc_restart) = ...
            two_opt_seed_rr;

        opt2 = struct();

        opt2.seed = ...
            two_opt_seed_rr;

        opt2.cost_tol = ...
            opts.two_opt_tol;

        opt2.verbose = ...
            opts.verbose_2opt;

        opt2.eta = ...
            opts.eval_eta;

        r2 = ...
            apply_common_two_opt( ...
                best_vc_I, ...
                problem, ...
                opt2);

        % Exactly ONE common 2-opt time.
        two_opt_time_total = ...
            r2.time;

        best_vc2_jcommon = ...
            r2.metrics_after.common_objective;

        % Refined solution originates from the raw-best restart.
        best_vc2_restart = ...
            best_vc_restart;

        best_vc2_I = ...
            r2.I_after;

        best_vc2_metrics = ...
            r2.metrics_after;

        % Backward-compatible arrays:
        % only the chosen raw-best restart has a refined value.
        vc2_jcommon(best_vc_restart) = ...
            best_vc2_jcommon;

        % The refined solution only becomes available after all R
        % rounding restarts have completed.
        vc2_best_jcommon_history(R) = ...
            best_vc2_jcommon;

        cumulative_2opt_time(R) = ...
            two_opt_time_total;

        % Update the final checkpoint (R is guaranteed to be a checkpoint).
        if ~isempty(checkpoint_records)

            cp_idx = ...
                find( ...
                    [checkpoint_records.restart] == R, ...
                    1, ...
                    'last');

            if ~isempty(cp_idx)

                checkpoint_records(cp_idx).two_opt_time = ...
                    two_opt_time_total;

                checkpoint_records(cp_idx).vc2_best_jcommon = ...
                    best_vc2_jcommon;
            end
        end

        fprintf( ...
            ['[eADMM common2opt] raw 最优 restart=%d | ', ...
            'raw J_common=%.10e -> refined J_common=%.10e | ', ...
            '2-opt time=%.6f 秒\n'], ...
            best_vc_restart, ...
            best_vc_jcommon, ...
            best_vc2_jcommon, ...
            two_opt_time_total);

    end


    %% ========================================================================
    % Final chosen solution
    % ========================================================================

    if opts.use_2opt

        chosen_I = ...
            best_vc2_I;

        chosen_metrics = ...
            best_vc2_metrics;

        chosen_label = ...
            "best raw Vc + common2opt";

        chosen_restart = ...
            best_vc_restart;

    else

        chosen_I = ...
            best_vc_I;

        chosen_metrics = ...
            best_vc_metrics;

        chosen_label = ...
            "Vc";

        chosen_restart = ...
            best_vc_restart;
    end
    %% ======================== Result struct ================================
    result = struct();
    result.method = "eADMM-SDP";
    result.I = chosen_I;
    result.P = batches_to_assignment(chosen_I,N,B,bs);
    result.metrics = chosen_metrics;
    result.chosen_label = chosen_label;
    result.chosen_restart = chosen_restart;

    % DNN SDP (also the rounding input).
    result.A_sdp = A_sdp;
    result.b_sdp = b_sdp;
    result.C_sdp = C_sdp;
    result.X_sdp = X_dnn;
    result.y_sdp = y_dnn;
    result.Z_sdp = Z_dnn;
    result.S_sdp = S_dnn;
    result.sigma_final = sigma_dnn_final;
    result.init_mode = string(opts.init_mode);
    result.dnn_iterations = dnn_iterations;
    result.dnn_avg_iter_time = dnn_avg_iter_time;
    result.primal_eq_rel = dnn_primal_eq_rel;
    result.dual_eq_rel = dnn_dual_eq_rel;
    result.min_eig_X = dnn_min_eig_X;
    result.nonneg_violation = dnn_nonneg_violation;
    result.sdp_jcommon_score = dnn_sdp_jcommon_score;

    % MET-strengthened lower-bound branch.
    result.met_enabled = opts.met.enabled;
    result.met_rounds = met_rounds;
    result.met_separation_calls = met_separation_calls;
    result.met_num_constraints = met_num_constraints;
    result.met_time = met_time;
    result.met_separation_time = met_separation_time;
    result.met_solve_time = met_solve_time;
    result.met_terminated_no_violation = met_terminated_no_violation;
    result.met_hit_round_limit = met_hit_round_limit;
    result.met_records = met_records;
    result.B_met = B_met;
    result.f_met = f_met;
    result.T_met = T_met;
    result.X_met = X_met;
    result.y_met = y_met;
    result.ybar_met = ybar_met;
    result.Z_met = Z_met;
    result.S_met = S_met;
    result.s_met = s_met;
    result.v_met = v_met;
    result.sigma_met_final = sigma_met_final;
    result.met_sdp_jcommon_score = met_sdp_jcommon_score;
    result.met_primal_eq_rel = met_primal_eq_rel;
    result.met_ineq_violation_rel = met_ineq_violation_rel;
    result.met_dual_eq_rel = met_dual_eq_rel;

    % Safe lower bound, always exposed on J_common scale.
    result.safe_jcommon_lower_bound = safe_lb.common;
    result.safe_graph_lower_bound_internal = safe_lb.graph_internal;
    result.graph_to_jcommon_constant = graph_to_jcommon_constant;
    result.safe_lower_bound_source = safe_lb.source;
    result.safe_lower_bound_y = safe_lb.y;
    result.safe_lower_bound_ybar = safe_lb.ybar;

    % Certification timing is deliberately separated from solution-generation
    % timing. MET and the final LP repair do NOT contribute to the runtime
    % used in Matrix/Random/eADMM algorithm comparisons.
    result.safe_lower_bound_postprocess_time = safe_lb.time;
    result.certification_met_time = met_time;
    result.certification_additional_time = met_time + safe_lb.time;
    % Backward-compatible alias: this means additional certification overhead,
    % not eADMM solution-generation time.
    result.safe_lower_bound_time = result.certification_additional_time;
    result.safe_lower_bound_min_eig_Z = safe_lb.min_eig_Zplus;
    slack_checks = [safe_lb.min_S,safe_lb.min_ybar];
    slack_checks = slack_checks(isfinite(slack_checks));
    if isempty(slack_checks)
        result.safe_lower_bound_min_slack = NaN;
    else
        result.safe_lower_bound_min_slack = min(slack_checks);
    end
    result.safe_lower_bound_dual_identity_rel = safe_lb.dual_identity_rel;
    result.safe_lower_bound_numerically_feasible = safe_lb.numerically_feasible;

    result.vc_jcommon = vc_jcommon;
    result.vc2_jcommon = vc2_jcommon;
    result.vc_best_jcommon_history = vc_best_jcommon_history;
    result.vc2_best_jcommon_history = vc2_best_jcommon_history;
    result.best_vc_jcommon = best_vc_jcommon;
    result.best_vc_restart = best_vc_restart;
    result.best_vc_I = best_vc_I;
    result.best_vc_metrics = best_vc_metrics;
    result.best_vc2_jcommon = best_vc2_jcommon;
    result.best_vc2_restart = best_vc2_restart;
    result.best_vc2_I = best_vc2_I;
    result.best_vc2_metrics = best_vc2_metrics;
    result.cumulative_vc_time = cumulative_vc_time;
    result.cumulative_2opt_time = cumulative_2opt_time;
    result.rounding_seed = rounding_seed;
    result.two_opt_seed = two_opt_seed;
    result.checkpoints = checkpoint_records;

    % ==================== Timing convention for comparisons =================
    % Main method comparison uses ONLY the solution-generation branch
    %
    %   DNN solve -> Vc rounding [-> common 2-opt].
    %
    % The MET strengthening and safe-LB LP are instance-level certification
    % work and are NEVER charged to eADMM in the method runtime table or the
    % equal-time comparison. They are reported separately below.
    result.solve_time = dnn_solve_time;
    result.vc_time = vc_time_total;
    result.two_opt_time = two_opt_time_total;

    result.solution_time_raw = dnn_solve_time + vc_time_total;
    result.solution_time_with_2opt = result.solution_time_raw + two_opt_time_total;

    % Backward-compatible total_time: method-comparison time only.
    if opts.use_2opt
        result.total_time = result.solution_time_with_2opt;
    else
        result.total_time = result.solution_time_raw;
    end

    % Certification costs. "additional" assumes the DNN relaxation has already
    % been solved for the eADMM solution branch. "from_scratch" is the cost of
    % obtaining the certificate starting from the DNN solve.
    result.certification_from_scratch_time = ...
        dnn_solve_time + result.certification_additional_time;

    % Full cost when both a rounded solution and a certificate are requested in
    % the same experiment. DNN work is shared and counted only once.
    result.total_time_with_certification = ...
        result.total_time + result.certification_additional_time;

    %% ======================== Console summary ==============================
    fprintf('\n==================== eADMM 结果摘要（统一 J_common 尺度） ====================\n');
    fprintf('下面带 * 的项目是与其他方法比较时最重要的指标。\n');
    fprintf('*eADMM DNN 迭代数 = %d*\n', dnn_iterations);
    fprintf('*eADMM 每次 DNN 迭代平均时间 = %.6e 秒/iter*\n', dnn_avg_iter_time);
    fprintf('*eADMM DNN 求解总时间 = %.6f 秒*\n', dnn_solve_time);
    fprintf(['eADMM raw 最优解 + common2opt J_common = %.10e ', ...
    '（raw restart %d）。\n'], ...
    best_vc2_jcommon,best_vc_restart);
    fprintf('*eADMM raw 解生成总时间 = %.6f 秒*（DNN求解 + 全部rounding）\n', ...
        result.solution_time_raw);
    if opts.use_2opt
        fprintf('eADMM 最佳 Vc+2opt J_common = %.10e（restart %d）。\n', ...
            best_vc2_jcommon,best_vc2_restart);
        fprintf('eADMM +2opt 解生成总时间 = %.6f 秒（DNN求解 + rounding + 2opt）。\n', ...
            result.solution_time_with_2opt);
    end

    fprintf('\n[eADMM DNN 数值诊断]\n');
    fprintf('DNN primal equality 相对残差 = %.6e；越小越好。\n', dnn_primal_eq_rel);
    fprintf('DNN dual equality 相对残差 = %.6e；越小越好。\n', dnn_dual_eq_rel);
    fprintf('DNN X 的最小特征值 = %.6e；接近 0 的微小负值通常是浮点误差。\n', dnn_min_eig_X);
    fprintf('DNN 非负约束违反量 = %.6e。\n', dnn_nonneg_violation);
    fprintf('DNN relaxed J_common = %.10e；这是松弛解诊断值，不是安全下界。\n', ...
        dnn_sdp_jcommon_score);

    if opts.met.enabled
        fprintf('\n[eADMM MET 认证分支]\n');
        fprintf('MET separation 扫描次数 = %d。\n',met_separation_calls);
        fprintf('MET 强化 SDP 重求解次数 = %d。\n',met_rounds);
        fprintf('MET 累计 constraints/cuts = %d。\n',met_num_constraints);
        fprintf('MET separation 时间 = %.6f 秒。\n',met_separation_time);
        fprintf('MET 强化 SDP 求解时间 = %.6f 秒。\n',met_solve_time);
        fprintf('MET 分支总时间 = %.6f 秒；这是认证开销，不计入 eADMM 解生成时间。\n',met_time);
        if met_num_constraints > 0
            fprintf('MET 强化 SDP relaxed J_common = %.10e。\n',met_sdp_jcommon_score);
            fprintf('MET 不等式相对违反量 = %.6e。\n',met_ineq_violation_rel);
        else
            fprintf('MET 结果：没有找到超过阈值的 violated cut，因此没有重求解强化 SDP。\n');
        end
    end

    if opts.compute_safe_lower_bound
        fprintf('\n[eADMM 安全下界]\n');
        fprintf('内部图尺度安全下界 = %.10e（仅内部诊断）。\n', safe_lb.graph_internal);
        fprintf('*安全 J_common 下界 = %.10e*\n', safe_lb.common);
        fprintf('安全下界来源 = %s。\n', char(safe_lb.source));
        fprintf('LB 后处理时间 = %.6f 秒。\n', safe_lb.time);
        fprintf('额外认证时间 = %.6f 秒（MET + LB后处理，不含已完成的DNN）。\n', ...
            result.certification_additional_time);
        fprintf('从头获得 certificate 的时间 = %.6f 秒（DNN + MET + LB后处理）。\n', ...
            result.certification_from_scratch_time);
    end

end

function I = part_cell_to_I(part_cell,bs,B,N)
    if numel(part_cell) ~= B
        error('Expected %d groups, got %d.',B,numel(part_cell));
    end
    I = zeros(bs,B);
    for j = 1:B
        idx = part_cell{j};
        if numel(idx) ~= bs
            error('Rounded group %d has %d samples; expected %d.',j,numel(idx),bs);
        end
        I(:,j) = idx(:);
    end
    flat = I(:);
    if numel(unique(flat)) ~= N || ~isequal(sort(flat),(1:N)')
        error('Rounding did not return a valid equipartition.');
    end
end
