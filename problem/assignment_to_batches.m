function [I, status] = assignment_to_batches(P, batch_size, support_tol)
%ASSIGNMENT_TO_BATCHES Convert an assignment-like matrix P to batch indices.
%
% This helper is mainly used to read the final Matrix ADMM variable Y.
% First it tries exact/nonzero support using support_tol; if that does not
% produce a valid balanced partition, it tries the natural threshold P>0.5.
%
% The conversion itself does NOT change the ADMM iterations.

    if nargin < 3 || isempty(support_tol)
        support_tol = 1e-12;
    end

    [num_samples, num_batches] = size(P);
    expected_samples = batch_size * num_batches;
    if num_samples ~= expected_samples
        error('P has %d rows but batch_size*num_batches=%d.', ...
            num_samples, expected_samples);
    end

    [I, ok] = try_rule(@(v) abs(v) > support_tol);
    status = "nonzero_support";

    if ~ok
        [I, ok] = try_rule(@(v) v > 0.5);
        status = "threshold_0.5";
    end

    if ~ok
        I = [];
        status = "failed";
    end

    function [I_local, valid] = try_rule(selector)
        I_local = zeros(batch_size, num_batches);
        valid = true;
        for j = 1:num_batches
            idx_j = find(selector(P(:, j)));
            if numel(idx_j) ~= batch_size
                valid = false;
                I_local = [];
                return;
            end
            I_local(:, j) = idx_j(:);
        end

        flat = I_local(:);
        valid = numel(unique(flat)) == num_samples && ...
            isequal(sort(flat), (1:num_samples)');
        if ~valid
            I_local = [];
        end
    end
end
