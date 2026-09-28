function P = batches_to_assignment(I, num_samples, num_batches, batch_size)
%BATCHES_TO_ASSIGNMENT Convert bs-by-B index representation I to assignment P.
%
% I(:,j) contains the sample indices in batch j.
% P(i,j)=1 means sample i belongs to batch j.

    if ~isequal(size(I), [batch_size, num_batches])
        error('I must be %d-by-%d.', batch_size, num_batches);
    end

    idx = I(:);
    if any(idx < 1) || any(idx > num_samples) || ...
            numel(unique(idx)) ~= num_samples || ...
            ~isequal(sort(idx), (1:num_samples)')
        error('I is not a valid partition of 1:%d.', num_samples);
    end

    P = zeros(num_samples, num_batches);
    for j = 1:num_batches
        P(I(:, j), j) = 1;
    end
end
