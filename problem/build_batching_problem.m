function problem = build_batching_problem(data, cfg)
%BUILD_BATCHING_PROBLEM Build the shared batching problem once.
%
% All solver-independent quality evaluation is expressed on ONE scale:
%
%   J_common(P) = <Phi, P*P'> / (2*bs^2),
%
% for a balanced binary assignment P in {0,1}^{N-by-B} satisfying
%
%   P*1_B = 1_N,      P'*1_N = bs*1_B.
%
% Matrix uses
%
%   A = Phi/bs^2,
%   G = -2*Phi*1_N*1_B'/(bs*N).
%
% IMPORTANT:
% The graph dissimilarity W=1-Phi and its Laplacian L are NOT constructed
% here. They are an eADMM-specific internal representation and are built only
% inside methods/eadmm/solve_eadmm_batching.m.

    t0 = tic;

    batch_size = cfg.batch.size;
    num_samples = data.num_samples;
    num_used = cfg.batch.num_used_samples;
    num_batches = cfg.batch.num_batches;

    if num_used < num_samples
        warning('Using the first %d of %d samples because allow_truncation=true.', ...
            num_used, num_samples);
    end

    Z = data.sample_stationary_points(1:num_used, :);

    pairwise_norms = pdist(Z, 'euclidean');
    distance_matrix = squareform(pairwise_norms);

    bandwidth = sum(distance_matrix, 'all') / (num_used^2 - num_used);
    if ~isfinite(bandwidth) || bandwidth <= eps
        bandwidth = 1;
        warning('Degenerate pairwise distances: bandwidth reset to 1.');
    end

    Phi = exp(-distance_matrix ./ bandwidth);
    Phi = (Phi + Phi') / 2;

    A = Phi / (batch_size^2);
    G = -2 * (Phi * ones(num_used, num_batches)) / ...
        (batch_size * num_used);

    problem = struct();
    problem.sample_representations = Z;
    problem.distance_matrix = distance_matrix;
    problem.bandwidth = bandwidth;
    problem.Phi = Phi;
    problem.A = A;
    problem.G = G;

    problem.num_samples = num_used;
    problem.num_batches = num_batches;
    problem.batch_size = batch_size;
    problem.eta = cfg.matrix.eta;
    problem.build_time = toc(t0);

    % On the balanced binary feasible set, the Matrix linear and L_{1/2}
    % terms are partition-independent constants. They are retained only for
    % Matrix-specific diagnostics; cross-method comparison always uses J_common.
    problem.matrix_linear_constant = sum(G(:, 1));
    problem.matrix_lhalf_constant = cfg.matrix.eta * num_used;
    problem.matrix_constant_offset = ...
        problem.matrix_linear_constant + problem.matrix_lhalf_constant;

    % Basic consistency checks.
    assert(isequal(size(Phi), [num_used, num_used]));
    assert(norm(Phi - Phi', 'fro') <= 1e-12 * max(1, norm(Phi, 'fro')));
    assert(isequal(size(G), [num_used, num_batches]));
end
