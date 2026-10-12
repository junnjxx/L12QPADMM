function I = random_balanced_partition(num_samples, batch_size, seed)
%RANDOM_BALANCED_PARTITION Generate one uniformly permuted balanced partition.
%
% A random permutation of 1:N is reshaped into bs-by-B.  This is deliberately
% kept as a trivial baseline; there is no optimization before optional 2-opt.

    if mod(num_samples, batch_size) ~= 0
        error('num_samples=%d must be divisible by batch_size=%d.', ...
            num_samples, batch_size);
    end

    rng(seed,'twister');
    p = randperm(num_samples);
    I = reshape(p, batch_size, num_samples/batch_size);
end
