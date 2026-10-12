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
%   3) construct and solve the continuous LP relaxation of Eq. (10);
%
%   4) select the k largest relaxed m_i values;
%
%   5) remove those samples from Q and continue.
%
% Timing convention:
%
%   z_build_time
%       construction of Z
%
%   lp_model_build_time
%       construction of f,A,b,Aeq,beq,lb,ub
%
%   lp_solve_time
%       ONLY linprog
%
%   lp_postsolve_time
%       extraction / LP feasibility diagnostics
%
%   rounding_time
%       ONLY top-k rounding
%
%   rounded_objective_eval_time
%       diagnostic m'Zm evaluation
%
%   exact_audit_time
%       brute-force diagnostic; EXCLUDED from official runtime
%
%   raw_wall_time
%       end-to-end raw method runtime, with exact-audit time removed
%

    n  = problem.num_samples;
    bs = problem.batch_size;
    nb = problem.num_batches;

    Phi = problem.Phi;

    if n ~= bs*nb
        error('run_mmdlp_baseline:UnbalancedProblem', ...
            'MMD-LP currently requires N = batch_size*num_batches.');
    end

    %% ============================================================
    % Sequential state
    % ============================================================

    Q = (1:n)';
    P = zeros(0,1);

    selected = zeros(bs,nb);

    %% ============================================================
    % Profiling arrays
    % ============================================================

    num_lp_solves = nb-1;

    z_build_time = ...
        zeros(num_lp_solves,1);

    lp_model_build_time = ...
        zeros(num_lp_solves,1);

    lp_solve_time = ...
        zeros(num_lp_solves,1);

    lp_postsolve_time = ...
        zeros(num_lp_solves,1);

    rounding_time = ...
        zeros(num_lp_solves,1);

    rounded_objective_eval_time = ...
        zeros(num_lp_solves,1);

    exact_audit_time = ...
        zeros(num_lp_solves,1);

    lp_objective = ...
        nan(num_lp_solves,1);

    rounded_iqp_objective = ...
        nan(num_lp_solves,1);

    relaxed_cardinality_residual = ...
        nan(num_lp_solves,1);

    lp_feasibility_residual = ...
        nan(num_lp_solves,1);

    lp_num_variables = ...
        zeros(num_lp_solves,1);

    lp_num_inequalities = ...
        zeros(num_lp_solves,1);

    remaining_before = ...
        zeros(num_lp_solves,1);

    % Optional relaxed-solution storage.
    m_relaxed_history = ...
        cell(num_lp_solves,1);

    %% ============================================================
    % LP options
    % ============================================================

    lp_opts = struct();

    lp_opts.display = ...
        cfg.mmdlp.linprog_display;

    lp_opts.feasibility_tol = ...
        cfg.mmdlp.feasibility_tol;

    %% ============================================================
    % Sequential batch selection
    % ============================================================

    wall_tic = tic;

    for b = 1:nb

        nQ = numel(Q);

        % --------------------------------------------------------
        % Final batch is forced.
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

        %% --------------------------------------------------------
        % Build Z
        % ---------------------------------------------------------

        tZ = tic;

        [Z,~] = ...
            build_mmd_batch_Z( ...
                Phi,Q,P,bs,n);

        z_build_time(b) = ...
            toc(tZ);

        %% --------------------------------------------------------
        % LP + paper top-k rounding
        % ---------------------------------------------------------

        [pick,m_relaxed,m_binary,lpinfo] = ...
            mmdlp_select_batch( ...
                Z,bs,lp_opts);
        % ============================================================
% Debug: inspect relaxed m and top-k selection
% ============================================================

        if cfg.mmdlp.verbose

            fprintf('\n');
            fprintf('============================================================\n');
            fprintf('[MMD-LP batch %d/%d] relaxed m inspection\n',b,nb);
            fprintf('nQ = %d | k = %d\n',nQ,bs);
            fprintf('sum(m) = %.12f | min(m) = %.12f | max(m) = %.12f\n', ...
                sum(m_relaxed),min(m_relaxed),max(m_relaxed));

            frac_mask = ...
                m_relaxed > 1e-10 & ...
                m_relaxed < 1-1e-10;

            fprintf('fractional entries = %d / %d\n', ...
                nnz(frac_mask),nQ);

            fprintf('\nOriginal Q order:\n');
            fprintf('%8s %10s %18s %10s\n', ...
                'local_i','global_i','m_LP','selected');
            fprintf('------------------------------------------------------\n');

            for ii = 1:nQ
                fprintf('%8d %10d %18.12f %10d\n', ...
                    ii,Q(ii),m_relaxed(ii),m_binary(ii));
            end

            % --------------------------------------------------------
            % Also print m in descending order.
            % This is the most useful view for diagnosing top-k rounding.
            % --------------------------------------------------------

            [m_sorted,ord] = sort(m_relaxed,'descend');

            fprintf('\nSorted by m_LP (descending):\n');
            fprintf('%8s %10s %18s %10s\n', ...
                'rank','global_i','m_LP','selected');
            fprintf('------------------------------------------------------\n');

            for rr = 1:nQ
                ii = ord(rr);

                fprintf('%8d %10d %18.12f %10d\n', ...
                    rr,Q(ii),m_sorted(rr),m_binary(ii));
            end

            fprintf('\nSelected local indices:\n');
            disp(pick(:)');

            fprintf('Selected global indices:\n');
            disp(Q(pick)');

            fprintf('============================================================\n\n');

        end
        lp_model_build_time(b) = ...
            lpinfo.model_build_time;

        lp_solve_time(b) = ...
            lpinfo.solve_time;

        lp_postsolve_time(b) = ...
            lpinfo.postsolve_time;

        rounding_time(b) = ...
            lpinfo.rounding_time;

        rounded_objective_eval_time(b) = ...
            lpinfo.rounded_objective_eval_time;

        %% --------------------------------------------------------
        % Optional exact IQP audit
        %
        % IMPORTANT:
        % This is diagnostic only and is excluded from official runtime.
        % ---------------------------------------------------------

        if cfg.mmdlp.exact_audit && ...
                nQ <= cfg.mmdlp.exact_audit_max_q

            audit_tic = tic;

            exact = ...
                mmdlp_bruteforce_iqp( ...
                    Z,bs);

            f_lp = ...
                lpinfo.relaxation_objective;

            f_exact = ...
                exact.best_objective;

            f_round = ...
                lpinfo.rounded_iqp_objective;

            scale = max([ ...
                1, ...
                abs(f_lp), ...
                abs(f_exact), ...
                abs(f_round)]);

            audit_tol = ...
                1e-8*scale;

            if f_lp > f_exact + audit_tol

                error( ...
                    ['MMD-LP exact audit failed at batch %d: ', ...
                     'LP %.12e > exact IQP %.12e.'], ...
                    b,f_lp,f_exact);

            end

            if f_exact > f_round + audit_tol

                error( ...
                    ['MMD-LP exact audit failed at batch %d: ', ...
                     'exact IQP %.12e > rounded %.12e.'], ...
                    b,f_exact,f_round);

            end

            exact_audit_time(b) = ...
                toc(audit_tic);

            fprintf([ ...
                '  [exact audit] C(%d,%d)=%d | ', ...
                'LP=% .6e <= exact=% .6e <= round=% .6e | ', ...
                'LP gap=%.3e | round gap=%.3e | ', ...
                'audit=%.6f s\n'], ...
                nQ,bs,exact.num_combinations, ...
                f_lp,f_exact,f_round, ...
                f_exact-f_lp, ...
                f_round-f_exact, ...
                exact_audit_time(b));

        end

        %% --------------------------------------------------------
        % Diagnostics
        % ---------------------------------------------------------

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

            m_relaxed_history{b} = ...
                m_relaxed;

        end

        if sum(m_binary) ~= bs

            error('run_mmdlp_baseline:RoundingFailure', ...
                'Batch %d did not contain exactly %d selected items.', ...
                b,bs);

        end

        %% --------------------------------------------------------
        % Map local Q indices back to global sample indices
        % ---------------------------------------------------------

        batch_indices = ...
            Q(pick);

        if numel(unique(batch_indices)) ~= bs

            error('run_mmdlp_baseline:DuplicateSelection', ...
                'Duplicate samples detected in batch %d.', ...
                b);

        end

        selected(:,b) = ...
            batch_indices;

        %% --------------------------------------------------------
        % Update P and Q
        % ---------------------------------------------------------

        keep = ...
            true(nQ,1);

        keep(pick) = ...
            false;

        P = ...
            [P;batch_indices]; %#ok<AGROW>

        Q = ...
            Q(keep);

        %% --------------------------------------------------------
        % Online report
        % ---------------------------------------------------------

        if cfg.mmdlp.verbose

            fprintf([ ...
                'MMD-LP batch %3d/%3d | nQ=%4d | ', ...
                'vars=%9d | ineq=%8d | ', ...
                'model=%.6f s | LP=%.6f s | ', ...
                'topk=%.6f s | ', ...
                'LPobj=% .6e | ', ...
                'rounded IQP=% .6e | ', ...
                'feas=%.3e\n'], ...
                b,nb,nQ, ...
                lpinfo.num_variables, ...
                lpinfo.num_inequalities, ...
                lpinfo.model_build_time, ...
                lpinfo.solve_time, ...
                lpinfo.rounding_time, ...
                lpinfo.relaxation_objective, ...
                lpinfo.rounded_iqp_objective, ...
                lpinfo.max_feasibility_residual);

        end

    end

    % Total wall clock including optional audit.
    raw_wall_with_audit = ...
        toc(wall_tic);

    total_exact_audit_time = ...
        sum(exact_audit_time);

    % Official method runtime excludes brute-force diagnostic.
    raw_wall_time = ...
        max(0, ...
            raw_wall_with_audit - ...
            total_exact_audit_time);

    %% ============================================================
    % Sanity checks
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

    %% ============================================================
    % Result
    % ============================================================

    result = struct();

    result.method = ...
        'MMD-LP (AAAI 2021 Eq. 10 relaxation)';

    result.I_raw = ...
        selected;

    result.P_raw = ...
        P_assignment;

    result.raw_jcommon = ...
        metrics.common_objective;

    result.num_lp_solves = ...
        num_lp_solves;

    % ------------------------------------------------------------
    % Per-batch timing
    % ------------------------------------------------------------

    result.z_build_time_per_batch = ...
        z_build_time;

    result.lp_model_build_time_per_batch = ...
        lp_model_build_time;

    result.lp_solve_time_per_batch = ...
        lp_solve_time;

    result.lp_postsolve_time_per_batch = ...
        lp_postsolve_time;

    result.rounding_time_per_batch = ...
        rounding_time;

    result.rounded_objective_eval_time_per_batch = ...
        rounded_objective_eval_time;

    result.exact_audit_time_per_batch = ...
        exact_audit_time;

    % ------------------------------------------------------------
    % Aggregate timing
    % ------------------------------------------------------------

    result.z_build_time = ...
        sum(z_build_time);

    result.lp_model_build_time = ...
        sum(lp_model_build_time);

    result.solve_time = ...
        sum(lp_solve_time);

    result.lp_postsolve_time = ...
        sum(lp_postsolve_time);

    result.rounding_time = ...
        sum(rounding_time);

    result.rounded_objective_eval_time = ...
        sum(rounded_objective_eval_time);

    result.exact_audit_time = ...
        total_exact_audit_time;

    result.raw_wall_time_with_audit = ...
        raw_wall_with_audit;

    % Official raw runtime:
    % exact audit excluded.
    result.raw_wall_time = ...
        raw_wall_time;

    % Everything except linprog.
    result.non_solve_time = ...
        max(0, ...
            raw_wall_time-result.solve_time);

    % ------------------------------------------------------------
    % Optimization diagnostics
    % ------------------------------------------------------------

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

    %% ============================================================
    % Common project-wide 2-opt
    % ============================================================

    result.refined_jcommon = NaN;
    result.two_opt_time = 0;

    result.total_time = ...
        raw_wall_time;

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
            raw_wall_time + ...
            post.time;

    end

    %% ============================================================
    % Final method report
    % ============================================================

    fprintf('\n');

    fprintf( ...
        'MMD-LP 完成：LP solve 次数=%d\n', ...
        result.num_lp_solves);

    fprintf( ...
        '*MMD-LP raw J_common = %.10e*\n', ...
        result.raw_jcommon);

    fprintf([ ...
        'MMD-LP 时间分解：\n', ...
        '  Z build          = %.6f 秒\n', ...
        '  LP model build   = %.6f 秒\n', ...
        '  linprog solve    = %.6f 秒\n', ...
        '  LP postsolve     = %.6f 秒\n', ...
        '  top-k rounding   = %.6f 秒\n', ...
        '  rounded obj eval = %.6f 秒\n', ...
        '  other/non-solve  = %.6f 秒\n', ...
        '  raw wall         = %.6f 秒\n'], ...
        result.z_build_time, ...
        result.lp_model_build_time, ...
        result.solve_time, ...
        result.lp_postsolve_time, ...
        result.rounding_time, ...
        result.rounded_objective_eval_time, ...
        result.non_solve_time, ...
        result.raw_wall_time);

    if result.exact_audit_time > 0

        fprintf([ ...
            '  exact audit       = %.6f 秒 ', ...
            '（诊断时间，不计入 raw wall）\n'], ...
            result.exact_audit_time);

    end

    if cfg.two_opt.enabled && ...
            cfg.mmdlp.use_common_2opt

        fprintf( ...
            '*MMD-LP + common2opt J_common = %.10e*\n', ...
            result.refined_jcommon);

        fprintf([ ...
            'MMD-LP common2opt=%.6f 秒 | ', ...
            '端到端=%.6f 秒\n'], ...
            result.two_opt_time, ...
            result.total_time);

    end
end