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
% Introduce W_ij = m_i*m_j. The paper uses:
%
%   if Z_ij < 0:
%
%       -m_i - m_j + 2 W_ij <= 0
%
%   if Z_ij >= 0:
%
%        m_i + m_j - 2 W_ij <= 1
%
% and then relaxes
%
%       0 <= m_i <= 1,
%       0 <= W_ij <= 1.
%
% IMPORTANT
% This intentionally reproduces the sign-dependent relaxation in the paper.
% It is NOT replaced by the full McCormick envelope.
%
% Variable ordering:
%
%       x = [m ; vec(W)]
%
% where W uses MATLAB column-major ordering.
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

    % ============================================================
    % Variables
    %
    % x = [m ; vec(W)]
    % ============================================================

    n_m = q;
    n_w = q*q;
    n_var = n_m + n_w;

    % ============================================================
    % Objective
    %
    % sum_ij Z_ij W_ij
    % ============================================================

    f = [zeros(q,1); Z(:)];

    % ============================================================
    % Inequality constraints
    %
    % Exactly one inequality per ordered pair (i,j).
    %
    % Each row initially contributes three entries:
    %
    %     coefficient of m_i
    %     coefficient of m_j
    %     coefficient of W_ij
    %
    % For i=j the duplicate m entries are automatically summed by
    % sparse(), which gives the correct coefficient +/-2.
    % ============================================================

    n_ineq = q*q;

    I = zeros(3*n_ineq,1);
    J = zeros(3*n_ineq,1);
    V = zeros(3*n_ineq,1);

    b = zeros(n_ineq,1);

    ptr = 0;
    row = 0;

    for jj = 1:q

        for ii = 1:q

            row = row + 1;

            w_local = sub2ind([q,q],ii,jj);
            w_col = q + w_local;

            if Z(ii,jj) < 0

                % ------------------------------------------------
                % -m_i - m_j + 2 W_ij <= 0
                % ------------------------------------------------

                ptr = ptr + 1;
                I(ptr) = row;
                J(ptr) = ii;
                V(ptr) = -1;

                ptr = ptr + 1;
                I(ptr) = row;
                J(ptr) = jj;
                V(ptr) = -1;

                ptr = ptr + 1;
                I(ptr) = row;
                J(ptr) = w_col;
                V(ptr) = 2;

                b(row) = 0;

            else

                % ------------------------------------------------
                % m_i + m_j - 2 W_ij <= 1
                % ------------------------------------------------

                ptr = ptr + 1;
                I(ptr) = row;
                J(ptr) = ii;
                V(ptr) = 1;

                ptr = ptr + 1;
                I(ptr) = row;
                J(ptr) = jj;
                V(ptr) = 1;

                ptr = ptr + 1;
                I(ptr) = row;
                J(ptr) = w_col;
                V(ptr) = -2;

                b(row) = 1;

            end

        end

    end

    A = sparse( ...
        I(1:ptr), ...
        J(1:ptr), ...
        V(1:ptr), ...
        n_ineq, ...
        n_var);

    % ============================================================
    % Equality
    %
    % sum_i m_i = k
    % ============================================================

    Aeq = sparse( ...
        ones(q,1), ...
        (1:q)', ...
        ones(q,1), ...
        1, ...
        n_var);

    beq = k;

    % ============================================================
    % Continuous relaxation
    %
    % 0 <= m,W <= 1
    % ============================================================

    lb = zeros(n_var,1);
    ub = ones(n_var,1);

    % Do not force a specific linprog algorithm here. This makes the code
    % compatible across MATLAB releases while leaving the mathematical LP
    % unchanged.
    lp_opts = optimoptions( ...
        'linprog', ...
        'Display',opts.display);

    % ============================================================
    % Solve
    % ============================================================

    solve_tic = tic;

    [x,fval,exitflag,output,lambda] = ...
        linprog( ...
            f, ...
            A,b, ...
            Aeq,beq, ...
            lb,ub, ...
            lp_opts);

    solve_time = toc(solve_tic);

    if isempty(x) || exitflag <= 0
        error('mmdlp_solve_relaxation:LinprogFailure', ...
            ['linprog failed. exitflag=%d. ', ...
             'See returned solver output for details.'], ...
            exitflag);
    end

    if any(~isfinite(x))
        error('mmdlp_solve_relaxation:NonfiniteSolution', ...
            'linprog returned NaN/Inf.');
    end

    m_relaxed = x(1:q);
    W_relaxed = reshape(x(q+1:end),q,q);

    % ============================================================
    % Numerical checks
    % ============================================================

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
        inequality_violation]);

    if max_feas_residual > opts.feasibility_tol
        warning('mmdlp_solve_relaxation:LPResidual', ...
            ['LP solution feasibility residual %.3e exceeds ', ...
             'diagnostic tolerance %.3e.'], ...
            max_feas_residual,opts.feasibility_tol);
    end

    % ============================================================
    % Return diagnostics
    % ============================================================

    info = struct();

    info.fval = fval;
    info.exitflag = exitflag;
    info.output = output;
    info.lambda = lambda;

    info.solve_time = solve_time;

    info.num_m_variables = n_m;
    info.num_w_variables = n_w;
    info.num_variables = n_var;
    info.num_inequalities = n_ineq;
    info.num_equalities = 1;

    info.cardinality_residual = cardinality_residual;
    info.bound_violation = bound_violation;
    info.inequality_violation = inequality_violation;
    info.max_feasibility_residual = max_feas_residual;

    % Retain W only for diagnostics. The paper rounds m, not W.
    info.W_relaxed = W_relaxed;
end