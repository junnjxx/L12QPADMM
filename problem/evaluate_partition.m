function metrics = evaluate_partition(P, problem, eta)
%EVALUATE_PARTITION Evaluate a final balanced binary assignment on J_common.
%
% This is the ONLY solver-independent partition-quality evaluator.
%
% Common objective:
%   J_common(P) = <Phi, P*P'>/(2*bs^2).
%
% Matrix-specific discrete objective (diagnostic only):
%   F_matrix(P) = 0.5*<P,A*P> + <G,P> + eta*sum_ij sqrt(P_ij).
%
% For a fixed instance and eta, the last two terms are constants on every
% balanced binary assignment. Hence F_matrix and J_common rank feasible
% partitions identically. Cross-method tables and gaps use J_common only.

    if nargin < 3 || isempty(eta)
        eta = problem.eta;
    end

    N = problem.num_samples;
    B = problem.num_batches;
    bs = problem.batch_size;

    metrics = struct();
    metrics.valid = false;
    metrics.common_objective = NaN;
    metrics.matrix_quadratic_term = NaN;
    metrics.matrix_linear_term = NaN;
    metrics.lhalf_unweighted = NaN;
    metrics.matrix_lhalf_term = NaN;
    metrics.matrix_full_objective = NaN;
    metrics.matrix_unregularized_objective = NaN;
    metrics.assignment = P;
    metrics.comembership = [];

    if isempty(P) || ~isequal(size(P), [N, B])
        return;
    end

    tol = 1e-12;
    if any(abs(P(:)) > tol & abs(P(:)-1) > tol)
        return;
    end
    if norm(sum(P,2)-1, inf) > tol
        return;
    end
    if norm(sum(P,1)-bs, inf) > tol
        return;
    end

    H = P * P';
    common = sum(problem.Phi .* H, 'all') / (2 * bs^2);

    % Matrix-specific decomposition; not used to compare methods.
    quadratic = 0.5 * sum(P .* (problem.A * P), 'all');
    linear = sum(problem.G .* P, 'all');
    lhalf_unweighted = sum(sqrt(max(P,0)), 'all');
    lhalf = eta * lhalf_unweighted;
    full_obj = quadratic + linear + lhalf;

    if abs(quadratic-common) > 1e-10 * max(1,abs(common))
        error('Matrix quadratic term and J_common are inconsistent.');
    end
    if abs(lhalf_unweighted-N) > 1e-12
        error('A balanced binary assignment should contain exactly %d ones.', N);
    end

    metrics.valid = true;
    metrics.common_objective = common;
    metrics.matrix_quadratic_term = quadratic;
    metrics.matrix_linear_term = linear;
    metrics.lhalf_unweighted = lhalf_unweighted;
    metrics.matrix_lhalf_term = lhalf;
    metrics.matrix_unregularized_objective = quadratic + linear;
    metrics.matrix_full_objective = full_obj;
    metrics.assignment = P;
    metrics.comembership = H;
end
