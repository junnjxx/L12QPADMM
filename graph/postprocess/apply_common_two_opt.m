function result = apply_common_two_opt(I_in, problem, opts)
%APPLY_COMMON_TWO_OPT Apply the same J_common 2-opt to any initial partition.
%
% Matrix, Random and eADMM all call this solver-independent postprocessor.
% The local search is written directly in terms of problem.Phi and J_common;
% no graph Laplacian is used outside the eADMM solver.

    if ~isfield(opts,'seed'), opts.seed = 1; end
    if ~isfield(opts,'cost_tol'), opts.cost_tol = 1e-9; end
    if ~isfield(opts,'verbose'), opts.verbose = false; end
    if ~isfield(opts,'eta'), opts.eta = problem.eta; end

    P_before = batches_to_assignment(I_in, problem.num_samples, ...
        problem.num_batches, problem.batch_size);
    metrics_before = evaluate_partition(P_before, problem, opts.eta);
    if ~metrics_before.valid
        error('Input is not a valid balanced partition.');
    end

    part_cell = cell(1, problem.num_batches);
    part = cell(1, problem.num_batches);
    for j = 1:problem.num_batches
        part_cell{j} = I_in(:,j)';
        part{j} = num2str(part_cell{j}');
    end
    H_before = metrics_before.comembership;
    J_before = metrics_before.common_objective;

    local_opts.cost_tol = opts.cost_tol;
    local_opts.verbose = opts.verbose;

    rng(opts.seed,'twister');
    t0 = tic;
    [part_cell_out, part_out, H_out, J_after_internal] = ...
        common_two_opt_local(problem.Phi,problem.num_batches,problem.batch_size, ...
            part_cell,part,H_before,J_before,local_opts);
    elapsed = toc(t0);

    I_out = zeros(problem.batch_size, problem.num_batches);
    for j = 1:problem.num_batches
        idx = part_cell_out{j};
        if numel(idx) ~= problem.batch_size
            error('2-opt group %d has %d samples; expected %d.', ...
                j, numel(idx), problem.batch_size);
        end
        I_out(:,j) = idx(:);
    end

    P_after = batches_to_assignment(I_out, problem.num_samples, ...
        problem.num_batches, problem.batch_size);
    metrics_after = evaluate_partition(P_after, problem, opts.eta);

    consistency_err = abs(metrics_after.common_objective-J_after_internal);
    if consistency_err > opts.cost_tol*max([1,abs(metrics_after.common_objective),abs(J_after_internal)])
        error('Common 2-opt J_common bookkeeping mismatch: %.3e.',consistency_err);
    end

    result = struct();
    result.I_before = I_in;
    result.I_after = I_out;
    result.P_before = P_before;
    result.P_after = P_after;
    result.part_before = part;
    result.part_after = part_out;
    result.part_cell_before = part_cell;
    result.part_cell_after = part_cell_out;
    result.H_before = H_before;
    result.H_after = H_out;
    result.metrics_before = metrics_before;
    result.metrics_after = metrics_after;
    result.time = elapsed;
    result.seed = opts.seed;
    result.absolute_improvement = ...
        metrics_before.common_objective - metrics_after.common_objective;
    result.relative_improvement_pct = 100*result.absolute_improvement / ...
        max(abs(metrics_before.common_objective),eps);
end
