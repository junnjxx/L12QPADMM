function init = prepare_matrix_initialization(problem,cfg,prior_results)
%PREPARE_MATRIX_INITIALIZATION Build the initialization requested for Matrix.
%
% User-facing cfg.matrix.init_mode options:
%   "uniform"        : keep the original noisy near-uniform Matrix init;
%   "zero"           : keep the original zero init;
%   "lp_best_jcommon": use the continuous LP iterate immediately BEFORE
%                       rounding/local search that generated the GLOBAL BEST
%                       LP J_common candidate over all outer/inner iterations;
%   "eadmm_sdp"      : use the eADMM base-DNN SDP matrix, converted from the
%                       N-by-N co-membership representation to a continuous
%                       N-by-k balanced assignment warm start.
%
% The returned source_generation_time is the upstream work needed to make the
% warm start available.  It is kept separate from the Matrix core solve time so
% reports can show both Matrix-only and end-to-end composite timing.

    if nargin < 3
        prior_results = struct();
    end

    mode = string(cfg.matrix.init_mode);
    init = struct();
    init.requested_mode = mode;
    init.is_external = false;
    init.X0 = [];
    init.Y0 = [];
    init.label = mode;
    init.source_generation_time = 0;
    init.conversion_time = 0;
    init.overhead_time = 0;
    init.details = struct();

    switch mode
        case "uniform"
            return;
        case "zero"
            return;
         case "random"
            return;
        case "lp_best_jcommon"
            if ~isfield(prior_results,'lp') || ~isfield(prior_results.lp,'enabled') || ...
                    ~prior_results.lp.enabled
                error('Matrix init_mode="lp_best_jcommon" requires the Lp baseline to run first.');
            end

            variant = string(cfg.matrix.lp_init_variant);
            switch variant
                case "alg2"
                    V = prior_results.lp.alg2;
                    variant_label = "Lp-Alg2";
                case "bs"
                    V = prior_results.lp.bs;
                    variant_label = "Lp-bs";
                otherwise
                    error('cfg.matrix.lp_init_variant must be "alg2" or "bs".');
            end
            if ~isfield(V,'enabled') || ~V.enabled
                error('Requested LP warm-start variant %s is disabled.',variant_label);
            end

            R = numel(V.raw);
            score = inf(R,1);
            discovery_time = nan(R,1);
            has_warm = false(R,1);
            for rr = 1:R
                a = V.raw{rr};
                if isempty(a) || ~isfield(a,'best_jcommon_pre_round_X') || ...
                        isempty(a.best_jcommon_pre_round_X)
                    continue;
                end
                Xcand = a.best_jcommon_pre_round_X;
                if ~isequal(size(Xcand),[problem.num_samples,problem.num_batches]) || ...
                        any(~isfinite(Xcand(:)))
                    continue;
                end
                has_warm(rr) = true;
                if isfield(a,'best_jcommon_post_round_score') && ...
                        isfinite(a.best_jcommon_post_round_score)
                    score(rr) = a.best_jcommon_post_round_score;
                end
                if isfield(a,'best_jcommon_discovery_elapsed_time')
                    discovery_time(rr) = a.best_jcommon_discovery_elapsed_time;
                end
            end

            ids = find(has_warm & isfinite(score));
            if isempty(ids)
                error(['No valid LP global-best-J_common pre-rounding warm start was captured. ', ...
                    'Keep cfg.lp.do_rounding=true.']);
            end
            [~,qq] = min(score(ids));
            chosen = ids(qq);
            a = V.raw{chosen};

            init.is_external = true;
            init.X0 = double(a.best_jcommon_pre_round_X);
            init.Y0 = init.X0;
            init.label = sprintf(['%s 全部 outer/inner 中全局最优 J_common 对应的 ', ...
                'pre-rounding 连续解（start %d, outer %d, inner %d, J_common=%.12g）'], ...
                char(variant_label),chosen,a.best_jcommon_outer_index, ...
                a.best_jcommon_inner_index,a.best_jcommon_post_round_score);

            % To certify which candidate is globally best, all LP starts used
            % for selection must finish. Charge the full LP solve time.
            if isfield(V,'solve_time') && numel(V.solve_time) >= max(ids)
                init.source_generation_time = sum(V.solve_time(ids),'omitnan');
            else
                init.source_generation_time = NaN;
            end
            init.overhead_time = init.source_generation_time;
            init.details.variant = variant;
            init.details.chosen_start = chosen;
            init.details.chosen_outer = a.best_jcommon_outer_index;
            init.details.chosen_inner = a.best_jcommon_inner_index;
            init.details.selection_score_after_rounding = a.best_jcommon_post_round_score;
            init.details.discovery_elapsed_time = discovery_time(chosen);
            init.details.all_start_best_scores = score;
            init.details.all_start_discovery_times = discovery_time;


        case "eadmm_sdp"
            if ~isfield(prior_results,'eadmm') || ~isfield(prior_results.eadmm,'X_sdp') || ...
                    isempty(prior_results.eadmm.X_sdp)
                error('Matrix init_mode="eadmm_sdp" requires eADMM to run first.');
            end

            E = prior_results.eadmm;
            tconv = tic;
            [P0,cinfo] = eadmm_sdp_to_balanced_assignment_init( ...
                E.X_sdp,problem.num_batches,problem.batch_size,cfg.lp.proj);
            conversion_elapsed = toc(tconv);

            init.is_external = true;
            init.X0 = P0;
            init.Y0 = P0;
            init.label = sprintf(['eADMM base-DNN SDP 解 -> top-k embedding/anchor similarity ', ...
                '-> balanced continuous projection']);
            init.source_generation_time = E.solve_time; % DNN only; no rounding/MET/LB needed
            init.conversion_time = conversion_elapsed;
            init.overhead_time = init.source_generation_time + init.conversion_time;
            init.details = cinfo;

        otherwise
            error(['Unknown cfg.matrix.init_mode: %s. Valid choices are uniform, zero, ', ...
                'lp_best_jcommon, eadmm_sdp.'],mode);
    end

    if init.is_external
        capacities = problem.batch_size*ones(problem.num_batches,1);
        init.details.row_res_inf = norm(sum(init.X0,2)-1,inf);
        init.details.col_res_inf = norm(sum(init.X0,1)'-capacities,inf);
        init.details.min_entry = min(init.X0(:));
        init.details.max_entry = max(init.X0(:));
    end
end
