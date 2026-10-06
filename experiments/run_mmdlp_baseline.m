function result = run_mmdlp_baseline(problem,cfg)
%RUN_MMDLP_BASELINE
% Sequential MMD-LP mini-batch selection baseline.
%
% Reference:
% Banerjee & Chakraborty,
% "Deterministic Mini-batch Sequencing for Training Deep Neural Networks",
% AAAI 2021.
%
% At each stage:
%
%   1) let P = already selected samples,
%          Q = currently unselected samples;
%
%   2) construct Z using Eq. (6)-(8);
%
%   3) solve the continuous LP relaxation of Eq. (10);
%
%   4) select the k largest relaxed m_i values;
%
%   5) remove those samples from Q and continue.
%
% The final partition is evaluated using the project's solver-independent
% J_common metric, exactly like Vector / Lp / Matrix / Random / eADMM.

    n  = problem.num_samples;
    bs = problem.batch_size;
    nb = problem.num_batches;

    Phi = problem.Phi;

    if n ~= bs*nb
        error('run_mmdlp_baseline:UnbalancedProblem', ...
            'MMD-LP currently requires N = batch_size*num_batches.');
    end

    % ============================================================
    % Sequential state
    % ============================================================

    Q = (1:n)';
    P = zeros(0,1);

    selected = zeros(bs,nb);

    % ============================================================
    % Profiling
    % ============================================================

    num_lp_solves = nb-1;

    lp_solve_time = zeros(num_lp_solves,1);
    z_build_time = zeros(num_lp_solves,1);
    rounding_time = zeros(num_lp_solves,1);

    lp_objective = nan(num_lp_solves,1);
    rounded_iqp_objective = nan(num_lp_solves,1);

    relaxed_cardinality_residual = nan(num_lp_solves,1);
    lp_feasibility_residual = nan(num_lp_solves,1);

    lp_num_variables = zeros(num_lp_solves,1);
    lp_num_inequalities = zeros(num_lp_solves,1);

    remaining_before = zeros(num_lp_solves,1);

    % Optional storage. Useful for reproducing/debugging the paper method.
    m_relaxed_history = cell(num_lp_solves,1);

    % ============================================================
    % LP options
    % ============================================================

    lp_opts = struct();

    lp_opts.display = ...
        cfg.mmdlp.linprog_display;

    lp_opts.feasibility_tol = ...
        cfg.mmdlp.feasibility_tol;

    % ============================================================
    % Sequential batch selection
    % ============================================================

    wall_tic = tic;

    for b = 1:nb

        nQ = numel(Q);

        % --------------------------------------------------------
        % Final batch is forced: every remaining sample goes there.
        % No optimization problem needs to be solved.
        % --------------------------------------------------------

        if b == nb

            if nQ ~= bs
                error('run_mmdlp_baseline:FinalBatchSize', ...
                    'Final Q contains %d samples; expected %d.', ...
                    nQ,bs);
            end

            selected(:,b) = Q;

            P = [P;Q]; %#ok<AGROW>
            Q = zeros(0,1);

            break;
        end

        remaining_before(b) = nQ;

        % --------------------------------------------------------
        % Build Z exactly from Eq. (6)-(8)
        % --------------------------------------------------------

        t0 = tic;

        [Z,~] = build_mmd_batch_Z( ...
            Phi,Q,P,bs,n);

        z_build_time(b) = toc(t0);

        % --------------------------------------------------------
        % LP solve + paper top-k rounding
        % --------------------------------------------------------

        t0 = tic;

        [pick,m_relaxed,m_binary,lpinfo] = ...
            mmdlp_select_batch( ...
                Z,bs,lp_opts);
        % --------------------------------------------------------
% Optional exact IQP audit for small remaining sets
% --------------------------------------------------------

if cfg.mmdlp.exact_audit && ...
        nQ <= cfg.mmdlp.exact_audit_max_q

    exact = mmdlp_bruteforce_iqp(Z,bs);

    f_lp = lpinfo.relaxation_objective;
    f_exact = exact.best_objective;
    f_round = lpinfo.rounded_iqp_objective;

    scale = max([ ...
        1, ...
        abs(f_lp), ...
        abs(f_exact), ...
        abs(f_round)]);

    audit_tol = 1e-8*scale;

    if f_lp > f_exact + audit_tol
        error(['MMD-LP exact audit failed at batch %d: ', ...
            'LP %.12e > exact IQP %.12e.'], ...
            b,f_lp,f_exact);
    end

    if f_exact > f_round + audit_tol
        error(['MMD-LP exact audit failed at batch %d: ', ...
            'exact IQP %.12e > rounded %.12e.'], ...
            b,f_exact,f_round);
    end

    fprintf([ ...
        '  [exact audit] C(%d,%d)=%d | ', ...
        'LP=% .6e <= exact=% .6e <= round=% .6e | ', ...
        'LP gap=%.3e | round gap=%.3e\n'], ...
        nQ,bs,exact.num_combinations, ...
        f_lp,f_exact,f_round, ...
        f_exact-f_lp, ...
        f_round-f_exact);
end
        total_select_call = toc(t0);

        lp_solve_time(b) = lpinfo.solve_time;

        % Everything in this call excluding linprog itself is counted as
        % rounding / LP-output processing.
        rounding_time(b) = max( ...
            0,total_select_call-lpinfo.solve_time);

        % --------------------------------------------------------
        % Diagnostics
        % --------------------------------------------------------

        lp_objective(b) = ...
            lpinfo.relaxation_objective;

        rounded_iqp_objective(b) = ...
            lpinfo.rounded_iqp_objective;

        relaxed_cardinality_residual(b) = ...
            lpinfo.cardinality_residual;

        lp_feasibility_residual(b) = ...
            lpinfo.max_feasibility_residual;

        lp_num_variables(b) = ...
            lpinfo.num_variables;

        lp_num_inequalities(b) = ...
            lpinfo.num_inequalities;

        if cfg.mmdlp.store_relaxed_history
            m_relaxed_history{b} = m_relaxed;
        end

        if sum(m_binary) ~= bs
            error('run_mmdlp_baseline:RoundingFailure', ...
                'Batch %d did not contain exactly %d selected items.', ...
                b,bs);
        end

        % --------------------------------------------------------
        % Map local Q indices back to global sample indices
        % --------------------------------------------------------

        batch_indices = Q(pick);

        if numel(unique(batch_indices)) ~= bs
            error('run_mmdlp_baseline:DuplicateSelection', ...
                'Duplicate samples detected in batch %d.',b);
        end

        selected(:,b) = batch_indices;

        % --------------------------------------------------------
        % Update P and Q
        %
        % Keep exact sequential ordering.
        % --------------------------------------------------------

        keep = true(nQ,1);
        keep(pick) = false;

        P = [P;batch_indices]; %#ok<AGROW>
        Q = Q(keep);

        % --------------------------------------------------------
        % Online report
        % --------------------------------------------------------

        if cfg.mmdlp.verbose

            fprintf([ ...
                'MMD-LP batch %3d/%3d | nQ=%4d | ', ...
                'vars=%9d | ineq=%8d | ', ...
                'LP=%.6f s | LPobj=% .6e | ', ...
                'rounded IQP=% .6e | feas=%.3e\n'], ...
                b,nb,nQ, ...
                lpinfo.num_variables, ...
                lpinfo.num_inequalities, ...
                lpinfo.solve_time, ...
                lpinfo.relaxation_objective, ...
                lpinfo.rounded_iqp_objective, ...
                lpinfo.max_feasibility_residual);

        end

    end

    raw_wall_time = toc(wall_tic);

    % ============================================================
    % Sanity checks on the generated partition
    % ============================================================

    flat_ids = selected(:);

    if numel(flat_ids) ~= n || ...
            numel(unique(flat_ids)) ~= n || ...
            ~isequal(sort(flat_ids),(1:n)')

        error('run_mmdlp_baseline:InvalidPartitionIndices', ...
            'MMD-LP did not generate a permutation of all N samples.');

    end

    P_assignment = ...
        batches_to_assignment( ...
            selected,n,nb,bs);

    metrics = ...
        evaluate_partition( ...
            P_assignment,problem);

    if ~metrics.valid
        error('run_mmdlp_baseline:InvalidAssignment', ...
            'MMD-LP produced an invalid balanced assignment.');
    end

    % ============================================================
    % Raw result
    % ============================================================

    result = struct();

    result.method = 'MMD-LP (AAAI 2021 Eq. 10 relaxation)';

    result.I_raw = selected;
    result.P_raw = P_assignment;

    result.raw_jcommon = ...
        metrics.common_objective;

    result.num_lp_solves = num_lp_solves;

    result.lp_solve_time_per_batch = ...
        lp_solve_time;

    result.z_build_time_per_batch = ...
        z_build_time;

    result.rounding_time_per_batch = ...
        rounding_time;

    result.lp_objective_per_batch = ...
        lp_objective;

    result.rounded_iqp_objective_per_batch = ...
        rounded_iqp_objective;

    result.remaining_before = ...
        remaining_before;

    result.lp_num_variables = ...
        lp_num_variables;

    result.lp_num_inequalities = ...
        lp_num_inequalities;

    result.relaxed_cardinality_residual = ...
        relaxed_cardinality_residual;

    result.lp_feasibility_residual = ...
        lp_feasibility_residual;

    if cfg.mmdlp.store_relaxed_history
        result.m_relaxed_history = ...
            m_relaxed_history;
    else
        result.m_relaxed_history = {};
    end

    result.solve_time = ...
        sum(lp_solve_time);

    result.build_time = ...
        sum(z_build_time);

    result.rounding_time = ...
        sum(rounding_time);

    result.raw_wall_time = ...
        raw_wall_time;

    % ============================================================
    % Common project-wide 2-opt
    % ============================================================

    result.refined_jcommon = NaN;
    result.two_opt_time = 0;
    result.total_time = raw_wall_time;

    if cfg.two_opt.enabled && ...
            cfg.mmdlp.use_common_2opt

        opt2 = struct();

        opt2.seed = ...
            cfg.seed.two_opt_base;

        opt2.cost_tol = ...
            cfg.two_opt.cost_tol;

        opt2.verbose = ...
            cfg.two_opt.verbose;

        opt2.eta = ...
            problem.eta;

        post = ...
            apply_common_two_opt( ...
                selected,problem,opt2);

        result.I_refined = ...
            post.I_after;

        result.P_refined = ...
            post.P_after;

        result.refined_jcommon = ...
            post.metrics_after.common_objective;

        result.two_opt_time = ...
            post.time;

        result.total_time = ...
            raw_wall_time + post.time;

    end

    % ============================================================
    % Final method report
    % ============================================================

    fprintf('\n');

    fprintf('MMD-LP 完成：LP solve 次数=%d\n', ...
        result.num_lp_solves);

    fprintf('*MMD-LP raw J_common = %.10e*\n', ...
        result.raw_jcommon);

    fprintf(['MMD-LP 时间：LP solve=%.6f 秒 | ', ...
        'Z build=%.6f 秒 | rounding/processing=%.6f 秒 | ', ...
        'raw wall=%.6f 秒\n'], ...
        result.solve_time, ...
        result.build_time, ...
        result.rounding_time, ...
        result.raw_wall_time);

    if cfg.two_opt.enabled && ...
            cfg.mmdlp.use_common_2opt

        fprintf('*MMD-LP + common2opt J_common = %.10e*\n', ...
            result.refined_jcommon);

        fprintf('MMD-LP common2opt=%.6f 秒 | 端到端=%.6f 秒\n', ...
            result.two_opt_time, ...
            result.total_time);

    end
end