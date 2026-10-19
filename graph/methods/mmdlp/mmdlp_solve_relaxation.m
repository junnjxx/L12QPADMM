function [m_relaxed,info] = mmdlp_solve_relaxation(Z,k,opts)
%MMDLP_SOLVE_RELAXATION
% Solve the LP relaxation in Eq. (10) of
%
% Banerjee & Chakraborty (AAAI 2021),
% "Deterministic Mini-batch Sequencing for Training Deep Neural Networks".
%
% Original binary quadratic problem:
%
%       min_m  m' Z m
%
%       s.t.   m_i in {0,1}
%              sum_i m_i = k
%
% Introduce W_ij = m_i*m_j.
%
% If Z_ij < 0:
%
%       -m_i - m_j + 2 W_ij <= 0
%
% If Z_ij >= 0:
%
%        m_i + m_j - 2 W_ij <= 1
%
% Continuous relaxation:
%
%       0 <= m_i <= 1,
%       0 <= W_ij <= 1.
%
% IMPORTANT:
% This reproduces the sign-dependent relaxation in the paper.
% It is NOT replaced by the full McCormick envelope.
%
% Timing convention:
%
%   info.model_build_time
%       construction of f,A,b,Aeq,beq,lb,ub and linprog options
%
%   info.solve_time
%       ONLY the actual linprog call
%
%   info.postsolve_time
%       extraction and feasibility diagnostics after linprog
%

    if nargin < 3 || isempty(opts)
        opts = struct();
    end

    if ~isfield(opts,'display')
        opts.display = 'none';
    end

    if ~isfield(opts,'feasibility_tol')
        opts.feasibility_tol = 1e-7;
    end

    q = size(Z,1);

    if size(Z,2) ~= q
        error('mmdlp_solve_relaxation:NonSquareZ', ...
            'Z must be square.');
    end

    if k < 1 || k > q || k ~= round(k)
        error('mmdlp_solve_relaxation:InvalidK', ...
            'k must be an integer in [1,q].');
    end

    if any(~isfinite(Z(:)))
        error('mmdlp_solve_relaxation:NonfiniteZ', ...
            'Z contains NaN/Inf.');
    end

    %% ============================================================
    % LP model construction
    % ============================================================

    model_tic = tic;

    % ------------------------------------------------------------
    % Variables
    %
    % x = [m ; vec(W)]
    % ------------------------------------------------------------

    n_m = q;
    n_w = q*q;
    n_var = n_m+n_w;

    % ------------------------------------------------------------
    % Objective
    %
    % min sum_ij Z_ij W_ij
    % ------------------------------------------------------------

    f = [zeros(q,1);Z(:)];

    % ------------------------------------------------------------
    % Sign-dependent inequalities
    %
    % Exactly one inequality for every ordered pair (i,j).
    % ------------------------------------------------------------

    n_ineq = q*q;

    I = zeros(3*n_ineq,1);
    J = zeros(3*n_ineq,1);
    V = zeros(3*n_ineq,1);

    b = zeros(n_ineq,1);

    ptr = 0;
    row = 0;

    for jj = 1:q

        for ii = 1:q

            row = row+1;

            w_local = sub2ind([q,q],ii,jj);
            w_col = q+w_local;

            if Z(ii,jj) < 0

                % -m_i -m_j +2 W_ij <= 0

                ptr = ptr+1;
                I(ptr)=row;
                J(ptr)=ii;
                V(ptr)=-1;

                ptr = ptr+1;
                I(ptr)=row;
                J(ptr)=jj;
                V(ptr)=-1;

                ptr = ptr+1;
                I(ptr)=row;
                J(ptr)=w_col;
                V(ptr)=2;

                b(row)=0;

            else

                % m_i +m_j -2 W_ij <= 1

                ptr = ptr+1;
                I(ptr)=row;
                J(ptr)=ii;
                V(ptr)=1;

                ptr = ptr+1;
                I(ptr)=row;
                J(ptr)=jj;
                V(ptr)=1;

                ptr = ptr+1;
                I(ptr)=row;
                J(ptr)=w_col;
                V(ptr)=-2;

                b(row)=1;

            end

        end

    end

    % sparse() automatically sums duplicated entries.
    % Hence when ii==jj the coefficient of m_i becomes +/-2,
    % exactly as required.

    A = sparse( ...
        I(1:ptr), ...
        J(1:ptr), ...
        V(1:ptr), ...
        n_ineq, ...
        n_var);

    % ------------------------------------------------------------
    % Cardinality constraint
    %
    % sum_i m_i = k
    % ------------------------------------------------------------

    Aeq = sparse( ...
        ones(q,1), ...
        (1:q)', ...
        ones(q,1), ...
        1, ...
        n_var);

    beq = k;

    % ------------------------------------------------------------
    % Continuous relaxation bounds
    % ------------------------------------------------------------

    lb = zeros(n_var,1);
    ub = ones(n_var,1);

    % ------------------------------------------------------------
% LP solver options
% ------------------------------------------------------------

% Display level
    if isfield(opts, 'linprog_display') && ...
            ~isempty(opts.linprog_display)
        display_level = opts.linprog_display;
    elseif isfield(opts, 'display') && ...
            ~isempty(opts.display)
        display_level = opts.display;
    else
        display_level = 'none';
    end

   lp_opts = optimoptions( ...
    'linprog', ...
    'Display', opts.display);

    if isfield(opts, 'algorithm') && ~isempty(opts.algorithm)
        lp_opts.Algorithm = opts.algorithm;
    end

    fprintf('[MMDLP LP] algorithm=%s\n', ...
        char(string(lp_opts.Algorithm)));


    % Maximum iteration count
    if isfield(opts, 'max_iterations') && ...
            ~isempty(opts.max_iterations)
        lp_opts.MaxIterations = opts.max_iterations;
    end

    % Maximum wall-clock time, in seconds
    if isfield(opts, 'max_time') && ...
            ~isempty(opts.max_time)
        if isprop(lp_opts, 'MaxTime')
            lp_opts.MaxTime = opts.max_time;
        else
            warning(['This MATLAB version does not expose MaxTime ', ...
                    'for the current linprog options.']);
        end
    end

    model_build_time = toc(model_tic);

    
    %% ============================================================
    % Actual LP solve with detailed failure diagnostics
    % ============================================================
    fprintf(['\n[MMDLP LP START] %s | q=%d | k=%d | ', ...
        'vars=%d | ineq=%d | model_build=%.2fs\n'], ...
        datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
        q, k, n_var, n_ineq, model_build_time);
drawnow;
    solve_tic = tic;

    try
        [x, fval, exitflag, output, lambda] = ...
            linprog( ...
                f, ...
                A, b, ...
                Aeq, beq, ...
                lb, ub, ...
                lp_opts);

    catch ME
        % MATLAB/linprog threw an exception, e.g., a solver or
        % allocation error that MATLAB was able to catch.
        solve_time = toc(solve_tic);

        fprintf(2, '\n[MMDLP LP EXCEPTION]\n');
        fprintf(2, 'q=%d, k=%d, variables=%d, inequalities=%d\n', ...
            q, k, n_var, n_ineq);
        fprintf(2, 'Elapsed time = %.4f seconds\n', solve_time);
        fprintf(2, 'Exception identifier: %s\n', ME.identifier);
        fprintf(2, 'Full MATLAB exception report:\n%s\n', ...
            getReport(ME, 'extended', 'hyperlinks', 'off'));

        % Preserve the original exception and stack.
        rethrow(ME);
    end

    solve_time = toc(solve_tic);

    %% ------------------------------------------------------------
    % Extract solver diagnostics
    % ------------------------------------------------------------

    output_message = '(output.message unavailable)';
    output_iterations = NaN;
    output_algorithm = '(unknown)';

    if isstruct(output)
        if isfield(output, 'message') && ~isempty(output.message)
            output_message = char(string(output.message));
        end

        if isfield(output, 'iterations') && ...
                ~isempty(output.iterations)
            output_iterations = output.iterations;
        end

        if isfield(output, 'algorithm') && ...
                ~isempty(output.algorithm)
            output_algorithm = char(string(output.algorithm));
        end
    end

    exitflag_print = NaN;
    if isnumeric(exitflag) && isscalar(exitflag) && ...
            isfinite(exitflag)
        exitflag_print = exitflag;
    end

    %% ------------------------------------------------------------
    % Fail explicitly, preserving the full solver diagnostics
    % ------------------------------------------------------------

    if isempty(x) || ~isfinite(exitflag_print) || exitflag_print <= 0

        fprintf(2, '\n[MMDLP LP FAILURE]\n');
        fprintf(2, ...
            'q=%d, k=%d, variables=%d, inequalities=%d\n', ...
            q, k, n_var, n_ineq);
        fprintf(2, 'exitflag = %g\n', exitflag_print);
        fprintf(2, 'fval = %.12g\n', fval);
        fprintf(2, 'algorithm = %s\n', output_algorithm);
        fprintf(2, 'iterations = %g\n', output_iterations);
        fprintf(2, 'solve_time = %.4f seconds\n', solve_time);
        fprintf(2, 'linprog output.message:\n%s\n', output_message);

        fprintf(2, 'Full linprog output structure:\n');
        disp(output);

        error('mmdlp_solve_relaxation:LinprogFailure', ...
            ['linprog failed or returned no usable solution.\n', ...
            'exitflag=%g, q=%d, k=%d, variables=%d, ', ...
            'inequalities=%d, solve_time=%.4f seconds.\n', ...
            'algorithm=%s, iterations=%g.\n', ...
            'linprog message:\n%s'], ...
            exitflag_print, q, k, n_var, n_ineq, solve_time, ...
            output_algorithm, output_iterations, output_message);
    end

    if any(~isfinite(x))
        fprintf(2, '\n[MMDLP LP FAILURE: NONFINITE SOLUTION]\n');
        fprintf(2, 'exitflag = %g\n', exitflag_print);
        fprintf(2, 'q=%d, k=%d, variables=%d, inequalities=%d\n', ...
            q, k, n_var, n_ineq);
        fprintf(2, 'solve_time = %.4f seconds\n', solve_time);
        fprintf(2, 'linprog output.message:\n%s\n', output_message);
        disp(output);

        error('mmdlp_solve_relaxation:NonfiniteSolution', ...
            ['linprog returned NaN/Inf in x.\n', ...
            'exitflag=%g, q=%d, k=%d, solve_time=%.4f seconds.\n', ...
            'linprog message:\n%s'], ...
            exitflag_print, q, k, solve_time, output_message);
    end


    %% ============================================================
    % Post-solve extraction and diagnostics
    % ============================================================

    post_tic = tic;

    m_relaxed = x(1:q);

    W_relaxed = ...
        reshape(x(q+1:end),q,q);

    % ------------------------------------------------------------
    % Numerical feasibility checks
    % ------------------------------------------------------------

    bound_violation = max([ ...
        max(-x), ...
        max(x-1), ...
        0]);

    cardinality_residual = ...
        abs(sum(m_relaxed)-k);

    inequality_violation = ...
        max([A*x-b;0]);

    max_feas_residual = max([ ...
        bound_violation, ...
        cardinality_residual, ...
        inequality_violation, ...
        0]);

    % Remove negative signed zero in printed diagnostics.
    if abs(max_feas_residual) == 0
        max_feas_residual = 0;
    end

    if abs(bound_violation) == 0
        bound_violation = 0;
    end

    if abs(cardinality_residual) == 0
        cardinality_residual = 0;
    end

    if abs(inequality_violation) == 0
        inequality_violation = 0;
    end

    if max_feas_residual > opts.feasibility_tol

        warning('mmdlp_solve_relaxation:LPResidual', ...
            ['LP solution feasibility residual %.3e exceeds ', ...
             'diagnostic tolerance %.3e.'], ...
            max_feas_residual, ...
            opts.feasibility_tol);

    end

    postsolve_time = toc(post_tic);

    %% ============================================================
    % Return diagnostics
    % ============================================================

    info = struct();

    info.fval = fval;
    info.exitflag = exitflag;
    info.output = output;
    info.lambda = lambda;

    % Timing
    info.model_build_time = model_build_time;
    info.solve_time = solve_time;
    info.postsolve_time = postsolve_time;

    % Dimensions
    info.num_m_variables = n_m;
    info.num_w_variables = n_w;
    info.num_variables = n_var;

    info.num_inequalities = n_ineq;
    info.num_equalities = 1;

    % Feasibility
    info.cardinality_residual = ...
        cardinality_residual;

    info.bound_violation = ...
        bound_violation;

    info.inequality_violation = ...
        inequality_violation;

    info.max_feasibility_residual = ...
        max_feas_residual;

    % Diagnostic only.
    % The paper rounds m, not W.
    info.W_relaxed = W_relaxed;
end