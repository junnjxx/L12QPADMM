function exp_result = run_random_baseline(problem, cfg)
%RUN_RANDOM_BASELINE Random balanced initialization + optional common 2-opt.
%
% Randomness convention:
%   - Random candidate r uses method-specific partition seed
%         cfg.seed.random_partition_base + r - 1.
%   - The subsequent common 2-opt uses
%         cfg.seed.two_opt_base + r - 1,
%     exactly the same 2-opt seed sequence used for Matrix/eADMM candidate r.

    R = cfg.random.num_starts;
    raw_I = cell(R,1);
    refined = cell(R,1);
    raw_jcommon = nan(R,1);
    full_obj = nan(R,1);
    refined_jcommon = nan(R,1);
    refined_full_obj = nan(R,1);
    gen_time = nan(R,1);
    local_time = nan(R,1);
    partition_seed = nan(R,1);
    two_opt_seed = nan(R,1);

    wall_tic = tic;
    for rr = 1:R
        partition_seed_rr = cfg.seed.random_partition_base + rr - 1;
        partition_seed(rr) = partition_seed_rr;
        tg = tic;
        I = random_balanced_partition(problem.num_samples,problem.batch_size,partition_seed_rr);
        gen_time(rr) = toc(tg);
        raw_I{rr} = I;

        P = batches_to_assignment(I,problem.num_samples,problem.num_batches,problem.batch_size);
        met = evaluate_partition(P,problem,cfg.matrix.eta);
        raw_jcommon(rr) = met.common_objective;
        full_obj(rr) = met.matrix_full_objective;

        if cfg.two_opt.enabled
            two_opt_seed_rr = cfg.seed.two_opt_base + rr - 1;
            two_opt_seed(rr) = two_opt_seed_rr;
            opt2.seed = two_opt_seed_rr;
            opt2.cost_tol = cfg.two_opt.cost_tol;
            opt2.verbose = cfg.two_opt.verbose;
            opt2.eta = cfg.matrix.eta;
            r2 = apply_common_two_opt(I,problem,opt2);
            refined{rr} = r2;
            refined_jcommon(rr) = r2.metrics_after.common_objective;
            refined_full_obj(rr) = r2.metrics_after.matrix_full_objective;
            local_time(rr) = r2.time;
        end

        do_print = rr <= cfg.reporting.print_first_random || ...
            mod(rr,cfg.reporting.print_random_every)==0 || rr==R;
        if do_print
            total_rr = gen_time(rr);
            fprintf(['Random 第 %3d/%3d 个随机解 | *J_common(raw)=%.10e* | ', ...
                '迭代数=不适用（Random 不是迭代算法） | 生成时间=%.6f 秒'], ...
                rr,R,raw_jcommon(rr),gen_time(rr));
            if cfg.two_opt.enabled
                total_rr = total_rr + local_time(rr);
                fprintf(' | J_common(+2opt)=%.10e | 2-opt时间=%.4f 秒', ...
                    refined_jcommon(rr),local_time(rr));
            end
            fprintf(' | *本次总时间=%.6f 秒*\n',total_rr);
        end
    end
    wall_time = toc(wall_tic);

    [~,best_raw_id] = min(raw_jcommon);
    if cfg.two_opt.enabled
        [~,best_refined_id] = min(refined_jcommon);
    else
        best_refined_id = NaN;
    end

    exp_result = struct();
    exp_result.raw_I = raw_I;
    exp_result.refined = refined;
    exp_result.raw_jcommon = raw_jcommon;
    exp_result.full_obj = full_obj;
    exp_result.refined_jcommon = refined_jcommon;
    exp_result.refined_full_obj = refined_full_obj;
    exp_result.gen_time = gen_time;
    exp_result.local_time = local_time;
    exp_result.partition_seed = partition_seed;
    exp_result.two_opt_seed = two_opt_seed;
    exp_result.best_raw_id = best_raw_id;
    exp_result.best_refined_id = best_refined_id;
    exp_result.wall_time = wall_time;
end
