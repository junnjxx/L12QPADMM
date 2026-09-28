function data = generate_synthetic_problem(cfg)
%GENERATE_SYNTHETIC_PROBLEM 生成 finite-sum 二次优化测试实例。
%
% 原始优化问题为
%
%   min_x  F(x) = sum_i f_i(x),
%   f_i(x) = 0.5*x'*Q_i*x + g_i'*x.
%
% 这里：
%   - x 是原始 finite-sum 问题的优化变量；
%   - Q_i, g_i 是随机生成后固定下来的 problem data；
%   - batching 方法并不优化 Q_i 或 g_i。
%
% 数据按 latent group 生成。第 k 个 group 先选中心 xhat_k 与共同 Q_k，
% 再令 G_k = -Q_k*xhat_k。若没有噪声，则单样本驻点满足
%
%   -Q_k \ G_k = xhat_k.
%
% 对 group 内每个样本，在 G_k 上施加小的逐坐标相对扰动：
%
%   g_i = G_k .* (1 + noise * epsilon_i),  epsilon_i ~ N(0,I).
%
% 因此相应的 sample stationary point
%
%   z_i = -Q_k \ g_i
%
% 通常位于 xhat_k 附近。后续 batching problem 使用 z_i 之间的距离
% 构造 similarity kernel Phi。

    % Fail fast if an old entry point accidentally routes MNIST to the
    % synthetic generator. A profile label is not a data-source switch.
    if strcmp(string(cfg.data.profile), "mnist_small1000")
        error(['Data routing error: mnist_small1000 was sent to ', ...
            'generate_synthetic_problem. Use load_mnist_small1000 instead.']);
    end

    rng(cfg.seed.data, 'twister');

    dim_x = cfg.data.dim_x;
    group_sizes = cfg.data.group_sizes;
    num_groups = numel(group_sizes);
    num_samples = sum(group_sizes);
    density = cfg.data.sparsity_density;
    rel_noise = cfg.data.linear_relative_noise;

    Q = cell(1, num_samples);
    g = zeros(dim_x, num_samples);
    sample_stationary_points = zeros(num_samples, dim_x);
    group_label = zeros(num_samples, 1);
    group_center = zeros(dim_x, num_groups);
    group_Q = cell(1, num_groups);
    group_G = zeros(dim_x, num_groups);

    Qa = zeros(dim_x);
    ind = 1;

    for k = 1:num_groups
        % latent center of the kth sample type
        xhat_k = rand(dim_x, 1);

        % Preserve the original synthetic-data construction exactly.
        rc = rand(dim_x, 1);
        Qk = sprandsym(dim_x, density, rc);

        % Construct the center linear term so xhat_k solves Qk*x + Gk = 0.
        Gk = -Qk * xhat_k;

        group_center(:, k) = xhat_k;
        group_Q{k} = Qk;
        group_G(:, k) = Gk;

        for j = 1:group_sizes(k)
            Q{ind} = Qk;
            Qa = Qa + Qk;

            % Multiplicative / relative perturbation around Gk.
            g(:, ind) = Gk .* (1 + rel_noise * randn(dim_x, 1));

            % z_i is the stationary point of the individual quadratic.
            % If Qk is positive definite it is also that sample's unique minimizer.
            sample_stationary_points(ind, :) = (-Qk \ g(:, ind))';
            group_label(ind) = k;
            ind = ind + 1;
        end
    end

    global_g = sum(g, 2);
    global_stationary_point = -Qa \ global_g;
    global_objective = 0.5 * global_stationary_point' * Qa * global_stationary_point ...
        + global_g' * global_stationary_point;

    data = struct();
    data.Q = Q;
    data.g = g;
    data.Qa = Qa;
    data.global_g = global_g;
    data.global_stationary_point = global_stationary_point;
    data.global_objective = global_objective;
    data.sample_stationary_points = sample_stationary_points;
    data.group_label = group_label;
    data.group_center = group_center;
    data.group_Q = group_Q;
    data.group_G = group_G;
    data.num_samples = num_samples;
    data.dim_x = dim_x;
    data.num_groups = num_groups;
end
