function exp_result = run_lp_baseline(problem,cfg)
%RUN_LP_BASELINE Run two Jiang-Liu-Wen Lp baselines on balanced assignments.
%
% Variants:
%   1) Lp-Alg2 : adapted full Algorithm 2, including greedy rounding +
%                internal best-improvement balanced N2 local search.
%   2) Lp-bs   : basic Lp continuation/rounding variant with the INTERNAL N2
%                local search disabled. This isolates how much of the final
%                quality comes from the Lp continuation itself. Greedy balanced
%                rounding is still used to obtain a feasible discrete candidate.
%
% Both variants may then be passed through the SAME project-wide common 2-opt.
% This is deliberately separate from Algorithm 2's internal N2 search.

    capacities = problem.batch_size * ones(problem.num_batches,1);

    % f(X) = J_common(X) on the continuous transportation polytope.
    model = make_quadratic_model(problem.A,0.5);

    curvature_lower = NaN;
    if cfg.lp.use_curvature_sigma0
        curvature_lower = lp_quadratic_curvature_lower(problem.A);
        model.curvature_lower = curvature_lower;
    else
        model.curvature_lower = [];
    end

    exp_result = struct();
    exp_result.enabled = true;
    exp_result.curvature_lower = curvature_lower;
    exp_result.objective_definition = ...
        '0.5*tr(X''*problem.A*X) = <Phi,XX''>/(2*bs^2)';

    if cfg.lp.run_alg2
        fprintf('\n---- Lp-Alg2：完整 Algorithm 2（包含内部 N2 局部搜索） ----\n');
        exp_result.alg2 = run_variant('Lp-Alg2',true);
    else
        exp_result.alg2 = empty_variant();
    end

    if cfg.lp.run_bs
        fprintf('\n---- Lp-bs：关闭内部 N2，仅保留 Lp continuation + rounding ----\n');
        exp_result.bs = run_variant('Lp-bs',false);
    else
        exp_result.bs = empty_variant();
    end

    % Backward-compatible aliases: old downstream code expecting results.lp.*
    % receives the full Algorithm-2 variant when it is enabled.
    if exp_result.alg2.enabled
        fn = fieldnames(exp_result.alg2);
        for q=1:numel(fn)
            exp_result.(fn{q}) = exp_result.alg2.(fn{q});
        end
    elseif exp_result.bs.enabled
        fn = fieldnames(exp_result.bs);
        for q=1:numel(fn)
            exp_result.(fn{q}) = exp_result.bs.(fn{q});
        end
    end

    function outvar = run_variant(label,do_internal_local_search)
        R = cfg.lp.num_starts;
        raw = cell(R,1);
        refined = cell(R,1);
        raw_I = cell(R,1);
        raw_jcommon = nan(R,1);
        refined_jcommon = nan(R,1);
        solve_time = nan(R,1);
        local_time = zeros(R,1);
        inner_iterations = nan(R,1);
        outer_iterations = nan(R,1);
        perturb_seed = nan(R,1);
        two_opt_seed = nan(R,1);
        valid = false(R,1);
        component_profile = cell(R,1);

        wall_tic = tic;
        for rr = 1:R
            opts = lp_options_from_config(cfg.lp);
            opts.perturb_seed = cfg.seed.lp_perturb_base + rr - 1;
            perturb_seed(rr) = opts.perturb_seed;
            opts.do_local_search = do_internal_local_search;

            t0 = tic;
            algout = lp_balanced_algorithm2(model,capacities,opts);
            solve_time(rr) = toc(t0);
            raw{rr} = algout;
            component_profile{rr} = algout.profile;

            P = algout.X_best;
            metrics = evaluate_partition(P,problem,cfg.matrix.eta);
            valid(rr) = metrics.valid;
            if ~metrics.valid
                warning('%s start %d did not return a valid balanced binary assignment.',label,rr);
                continue;
            end

            obj_err = abs(algout.f_best-metrics.common_objective);
            obj_tol = 1e-9*max([1,abs(algout.f_best),abs(metrics.common_objective)]);
            if obj_err > obj_tol
                error('%s objective/J_common mismatch on start %d: %.3e.',label,rr,obj_err);
            end

            raw_jcommon(rr) = metrics.common_objective;
            [I,status] = assignment_to_batches(P,problem.batch_size,1e-12);
            if isempty(I)
                error('%s start %d returned binary P but assignment_to_batches failed (%s).', ...
                    label,rr,string(status));
            end
            raw_I{rr} = I;

            if isempty(algout.history)
                outer_iterations(rr) = 0;
                inner_iterations(rr) = 0;
            else
                outer_iterations(rr) = numel(algout.history);
                inner_iterations(rr) = sum([algout.history.inner_iter]);
            end

            if cfg.lp.use_common_2opt
                two_opt_seed_rr = cfg.seed.two_opt_base + rr - 1;
                two_opt_seed(rr) = two_opt_seed_rr;
                opt2.seed = two_opt_seed_rr;
                opt2.cost_tol = cfg.two_opt.cost_tol;
                opt2.verbose = cfg.two_opt.verbose;
                opt2.eta = cfg.matrix.eta;
                r2 = apply_common_two_opt(I,problem,opt2);
                refined{rr} = r2;
                refined_jcommon(rr) = r2.metrics_after.common_objective;
                local_time(rr) = r2.time;
            end

            p = algout.profile;
            avg_inner = solve_time(rr) / max(inner_iterations(rr),1);
            total_rr = solve_time(rr);
            fprintf(['%s 第 %2d/%2d 个起点 | 扰动随机种子=%d | 可行=%d | ', ...
                '*J_common=%.10e* | outer迭代=%d | *inner迭代数=%d* | ', ...
                '*每个inner迭代平均时间=%.3e 秒* | *求解总时间=%.4f 秒* | ', ...
                '投影=%.3f秒 | rounding=%.3f秒 | 内部N2=%.3f秒 | 线搜索=%.3f秒 | 其他=%.3f秒'], ...
                label,rr,R,perturb_seed(rr),valid(rr),raw_jcommon(rr), ...
                outer_iterations(rr),inner_iterations(rr),avg_inner,solve_time(rr), ...
                p.projection_time,p.rounding_time,p.internal_local_search_time, ...
                p.line_search_time,p.other_time);
            if cfg.lp.use_common_2opt
                total_rr = total_rr + local_time(rr);
                fprintf(' | J_common(+common2opt)=%.10e | common2opt时间=%.4f秒', ...
                    refined_jcommon(rr),local_time(rr));
            end
            fprintf(' | *本次总时间=%.4f 秒*\n',total_rr);
        end
        wall_time = toc(wall_tic);

        valid_ids = find(valid & isfinite(raw_jcommon));
        if isempty(valid_ids)
            error('All %s starts failed to produce a valid balanced partition.',label);
        end

        [~,q] = min(raw_jcommon(valid_ids));
        best_raw_id = valid_ids(q);
        if cfg.lp.use_common_2opt
            [~,q2] = min(refined_jcommon(valid_ids));
            best_refined_id = valid_ids(q2);
        else
            best_refined_id = NaN;
        end

        outvar = struct();
        outvar.enabled = true;
        outvar.label = label;
        outvar.has_internal_local_search = do_internal_local_search;
        outvar.raw = raw;
        outvar.refined = refined;
        outvar.raw_I = raw_I;
        outvar.valid = valid;
        outvar.valid_ids = valid_ids;
        outvar.raw_jcommon = raw_jcommon;
        outvar.refined_jcommon = refined_jcommon;
        outvar.solve_time = solve_time;
        outvar.local_time = local_time;
        outvar.outer_iterations = outer_iterations;
        outvar.inner_iterations = inner_iterations;
        outvar.perturb_seed = perturb_seed;
        outvar.two_opt_seed = two_opt_seed;
        outvar.best_raw_id = best_raw_id;
        outvar.best_refined_id = best_refined_id;
        outvar.wall_time = wall_time;
        outvar.component_profile = component_profile;
        outvar.profile_sum = sum_profiles(component_profile(valid_ids));
    end
end

function opts = lp_options_from_config(c)
% Map project cfg.lp fields to the standalone Algorithm-2 driver.
    opts = struct();
    opts.p = c.p;
    opts.eps0 = c.eps0;
    opts.eps_min = c.eps_min;
    opts.sigma_max = c.sigma_max;
    opts.gamma = c.gamma;
    opts.tol_outer = c.tol_outer;
    opts.alpha0 = c.alpha0;
    opts.theta = c.theta;
    opts.delta = c.delta;
    opts.eta = c.reference_eta;
    opts.tau_x0 = c.tau_x0;
    opts.tau_f0 = c.tau_f0;
    opts.tau_x_min = c.tau_x_min;
    opts.tau_f_min = c.tau_f_min;
    opts.sigma_minus = c.sigma_minus;
    opts.sigma0 = [];
    opts.max_outer = c.max_outer;
    opts.max_inner = c.max_inner;
    opts.max_backtrack = c.max_backtrack;
    opts.alpha_min = c.alpha_min;
    opts.alpha_max = c.alpha_max;
    opts.tau_index_shift = c.tau_index_shift;
    opts.kkt_map_tol = c.kkt_map_tol;
    opts.perturb_rho = c.perturb_rho;
    opts.perturb_seed = 1; % overwritten per run
    opts.do_rounding = c.do_rounding;
    opts.do_local_search = c.do_internal_local_search;
    opts.local_max_swaps = c.local_max_swaps;
    opts.local_block = c.local_block; % compatibility only; fast N2 does not use row blocks
    opts.improve_tol = c.improve_tol;
    opts.local_recompute_every = c.local_recompute_every;
    opts.local_verbose_every = c.local_verbose_every;
    opts.local_final_verify = c.local_final_verify;
    opts.projector = [];
    opts.rounder = [];
    opts.local_search = [];
    opts.proj = c.proj;
    opts.verbose = c.verbose;
    opts.store_inner_history = c.store_inner_history;
end

function nu = lp_quadratic_curvature_lower(A)
% Lower curvature of f(X)=0.5*tr(X'*A*X) for symmetric A.
    As = 0.5*(A+A');
    n = size(As,1);
    if n <= 600
        ev = eig(full(As));
        nu = min(real(ev));
    else
        try
            nu = real(eigs(As,1,'smallestreal'));
        catch
            d = full(diag(As));
            r = sum(abs(As),2)-abs(d);
            nu = min(d-r);
            warning('eigs failed; using conservative Gershgorin curvature lower bound %.6e.',nu);
        end
    end
end

function s = sum_profiles(profiles)
% Aggregate per-start profiling structs by summing wall-clock components/counts.
    s = struct();
    if isempty(profiles), return; end
    first = profiles{1};
    fn = fieldnames(first);
    for q=1:numel(fn)
        name=fn{q};
        vals=zeros(numel(profiles),1);
        ok=true;
        for r=1:numel(profiles)
            if ~isfield(profiles{r},name) || ~isscalar(profiles{r}.(name)) || ~isnumeric(profiles{r}.(name))
                ok=false; break;
            end
            vals(r)=profiles{r}.(name);
        end
        if ok, s.(name)=sum(vals); end
    end
end

function v = empty_variant()
    v = struct('enabled',false,'label','','valid_ids',[],'raw_jcommon',[], ...
        'refined_jcommon',[],'solve_time',[],'local_time',[], ...
        'inner_iterations',[],'outer_iterations',[],'component_profile',{{}}, ...
        'profile_sum',struct());
end
