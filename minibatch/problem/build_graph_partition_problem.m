function problem = build_graph_partition_problem(data,cfg)
%BUILD_GRAPH_PARTITION_PROBLEM
% Build the shared ADMM-GP k-equipartition problem.
%
% Let W be the official weighted adjacency matrix and
%
%   L = D-W,
%   D = diag(W*1).
%
% The graph objective is
%
%   J_graph(P)
%       = 0.5*<L,P*P'>.
%
% We expose exactly the same objective through the project's J_common:
%
%   Phi = bs^2*L,
%
%   J_common(P)
%       = <Phi,P*P'>/(2*bs^2)
%       = 0.5*<L,P*P'>.
%
% Matrix receives
%
%   A = L,
%   G = 0.
%
% Hence on a binary balanced P,
%
%   F_matrix(P)
%       = 0.5*tr(P'*L*P)
%         + eta*sum_ij sqrt(P_ij)
%
%       = J_graph(P) + eta*N.
%
% Therefore Matrix and the graph problem have exactly the same
% discrete minimizers.

    t0 = tic;

    W = ...
        double(data.W);

    n = ...
        size(W,1);

    k = ...
        double(cfg.graph.k);

    if size(W,2) ~= n

        error('W must be square.');
    end

    if n ~= cfg.graph.n

        error( ...
            'Loaded n=%d but cfg.graph.n=%d.', ...
            n,cfg.graph.n);
    end

    if mod(n,k) ~= 0

        error( ...
            'n=%d must be divisible by k=%d.', ...
            n,k);
    end

    bs = ...
        n/k;

    %% ============================================================
    % Laplacian
    % ============================================================

    degree = ...
        sum(W,2);

    L = ...
        diag(degree)-W;

    L = ...
        0.5*(L+L');

    %% ============================================================
    % Shared project matrices
    % ============================================================

    % Project-wide common objective:
    %
    % J_common
    % = <Phi,H>/(2*bs^2)
    % = 0.5*<L,H>.
    Phi = (bs^2)*L;

    % Matrix model.
    A = ...
        L;

    G = ...
        zeros(n,k);

    %% ============================================================
    % Problem struct
    % ============================================================

    problem = struct();

    problem.type = ...
        "graph";

    problem.family = ...
        string(data.family);

    problem.W = ...
        W;

    problem.L = ...
        L;

    problem.Phi = ...
        Phi;

    problem.A = ...
        A;

    problem.G = ...
        G;

    problem.num_samples = ...
        n;

    problem.num_batches = ...
        k;

    problem.batch_size = ...
        bs;

    problem.eta = ...
        cfg.matrix.eta;

    problem.graph_density = ...
        data.empirical_density;

    problem.source_file = ...
        data.source_file;

    problem.source_entry = ...
        data.source_entry;

    problem.matrix_linear_constant = ...
        0;

    problem.matrix_lhalf_constant = ...
        cfg.matrix.eta*n;

    problem.matrix_constant_offset = ...
        problem.matrix_lhalf_constant;

    %% ============================================================
    % Structural checks
    % ============================================================

    scaleL = ...
        max(1,norm(L,'fro'));

    if norm(L-L','fro') > 1e-12*scaleL

        error('L is unexpectedly asymmetric.');
    end

    row_sum_err = ...
        norm(L*ones(n,1),inf);

    if row_sum_err > 1e-10*max(1,norm(L,inf))

        error( ...
            'Laplacian row-sum residual is too large: %.3e.', ...
            row_sum_err);
    end

    %% ============================================================
    % Objective identity test
    %
    % This is deliberately executed every time a graph problem is built.
    % If any future refactor changes a scale/factor, fail immediately.
    % ============================================================

    Pcheck = ...
        zeros(n,k);

    for jj=1:k

        idx = ...
            (jj-1)*bs + (1:bs);

        Pcheck(idx,jj) = 1;
    end

    Hcheck = ...
        Pcheck*Pcheck';

    J_graph = ...
        0.5*sum(L.*Hcheck,'all');

    J_common = ...
        sum(Phi.*Hcheck,'all') ...
        /(2*bs^2);

    J_matrix = ...
        0.5*sum( ...
            Pcheck.*(A*Pcheck), ...
            'all');

    tol = ...
        1e-10*max( ...
            [1,abs(J_graph),abs(J_common),abs(J_matrix)]);

    if abs(J_graph-J_common) > tol

        error( ...
            'Graph/J_common objective mismatch: %.3e.', ...
            abs(J_graph-J_common));
    end

    if abs(J_graph-J_matrix) > tol

        error( ...
            'Graph/Matrix quadratic objective mismatch: %.3e.', ...
            abs(J_graph-J_matrix));
    end

    %% ============================================================
    % Direct ADMM-GP objective test
    % ============================================================

    if exist('kequi_form','file') ~= 0

        [~,~,Ccheck] = ...
            kequi_form( ...
                W, ...
                int16(k));

        J_eadmm = ...
            sum(Ccheck.*Hcheck,'all');

        tol2 = ...
            1e-9*max( ...
                [1,abs(J_graph),abs(J_eadmm)]);

        if abs(J_graph-J_eadmm) > tol2

            error( ...
                ['ADMM-GP C matrix has unexpected objective scale.\n', ...
                 'J_graph=%.12e, J_eadmm=%.12e.'], ...
                J_graph,J_eadmm);
        end
    end

    problem.build_time = ...
        toc(t0);

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('Graph problem built\n');
    fprintf('n=%d | k=%d | bs=%d\n',n,k,bs);
    fprintf('J_common(P) = 0.5*<L,P*P''>\n');
    fprintf('Matrix: A=L, G=0\n');
    fprintf('Phi=bs^2*L\n');
    fprintf('||L||_1=%.10e | ||L||_2=%.10e\n', ...
        norm(L,1),norm(L,2));
    fprintf('objective identity test: PASSED\n');
    fprintf('build time=%.6f sec\n',problem.build_time);
    fprintf('============================================================\n');
end