function cfg = finalize_config(cfg, project_root)
%FINALIZE_CONFIG Derive quantities and validate the single config.m settings.
%
% This file contains no user-chosen numerical parameters. Edit config/config.m
% for experiments.

    %% ======================== Method selection ==============================
    if ~isfield(cfg,'methods') || ~isfield(cfg.methods,'run')
        error('config.m must define cfg.methods.run.');
    end

    canonical_methods =  ["eadmm","lp","mmdlp","random","vector","matrix"];
    selected = lower(strtrim(string(cfg.methods.run)));
    selected = selected(:)';
    selected = selected(selected ~= "");

    if isempty(selected)
        error('cfg.methods.run must contain at least one method.');
    end

    unknown = selected(~ismember(selected,canonical_methods));
    if ~isempty(unknown)
       error(['Unknown method(s) in cfg.methods.run: %s. ', ...
       'Valid names are: eadmm, lp, mmdlp, random, vector, matrix.'], ...
    strjoin(cellstr(unique(unknown,'stable')),', '));
    end

    % Remove duplicates and force the requested canonical execution order.
    cfg.methods.run = unique(selected,'stable');
    cfg.methods.enabled = struct();
    cfg.methods.enabled.eadmm = any(cfg.methods.run == "eadmm");
    cfg.methods.enabled.lp = any(cfg.methods.run == "lp");
    cfg.methods.enabled.mmdlp = any(cfg.methods.run == "mmdlp");
    cfg.methods.enabled.random = any(cfg.methods.run == "random");
    cfg.methods.enabled.matrix = any(cfg.methods.run == "matrix");
    cfg.methods.enabled.vector = any(cfg.methods.run == "vector");

    %% ======================== Problem/profile ===============================
    switch string(cfg.data.profile)
        case "smoke20"
            cfg.data.group_sizes = [4, 6, 5, 5];
        case "medium60"
            cfg.data.group_sizes = [15, 15, 15, 15];
        case "large220"
            cfg.data.group_sizes = [55, 55, 55, 55];
        case "mnist_small1000"
            cfg.data.group_sizes = repmat(cfg.data.mnist_per_digit,1,10);
            if cfg.data.mnist_per_digit ~= 100
                error('mnist_small1000 requires exactly 100 samples per digit.');
            end
            cfg.data.dim_x = 784;
        case "custom"
            if isempty(cfg.data.group_sizes)
                error('profile="custom" requires cfg.data.group_sizes.');
            end
        otherwise
            error('Unknown cfg.data.profile: %s', string(cfg.data.profile));
    end

    cfg.data.group_sizes = double(cfg.data.group_sizes(:)');
    cfg.data.num_groups = numel(cfg.data.group_sizes);
    cfg.data.num_samples = sum(cfg.data.group_sizes);

    if cfg.data.dim_x < 1 || cfg.data.dim_x ~= round(cfg.data.dim_x)
        error('cfg.data.dim_x must be a positive integer.');
    end
    if any(cfg.data.group_sizes < 1) || any(cfg.data.group_sizes ~= round(cfg.data.group_sizes))
        error('Every entry of cfg.data.group_sizes must be a positive integer.');
    end
    if cfg.batch.size < 1 || cfg.batch.size ~= round(cfg.batch.size)
        error('cfg.batch.size must be a positive integer.');
    end

    remainder = mod(cfg.data.num_samples, cfg.batch.size);
    if remainder ~= 0 && ~cfg.batch.allow_truncation
        error(['num_samples=%d is not divisible by batch_size=%d. ', ...
            'Either change the sizes or explicitly set cfg.batch.allow_truncation=true.'], ...
            cfg.data.num_samples, cfg.batch.size);
    end

    cfg.batch.num_used_samples = floor(cfg.data.num_samples / cfg.batch.size) * cfg.batch.size;
    cfg.batch.num_batches = cfg.batch.num_used_samples / cfg.batch.size;
    if cfg.batch.num_batches < 2
        error('At least two batches are required.');
    end

    %% ======================== Matrix settings ===============================
    matrix_init_mode = string(cfg.matrix.init_mode);
    valid_matrix_init_modes = ["uniform","zero","lp_best_jcommon","eadmm_sdp","random"];
    if ~any(matrix_init_mode == valid_matrix_init_modes)
        error(['cfg.matrix.init_mode must be one of: "uniform", "zero", ', ...
            '"lp_best_jcommon", "eadmm_sdp".']);
    end

    if cfg.methods.enabled.matrix
        if ~islogical(cfg.matrix.matrix_detail) || ~isscalar(cfg.matrix.matrix_detail)
            error('cfg.matrix.matrix_detail must be a logical scalar.');
        end
        if ~islogical(cfg.matrix.early_exist) || ~isscalar(cfg.matrix.early_exist)
            error('cfg.matrix.early_exist must be a logical scalar.');
        end
        if ~isscalar(cfg.matrix.early_exist_dxy_tol) || ...
                ~isfinite(cfg.matrix.early_exist_dxy_tol) || cfg.matrix.early_exist_dxy_tol < 0
            error('cfg.matrix.early_exist_dxy_tol must be a finite nonnegative scalar.');
        end
        if ~isscalar(cfg.matrix.early_exist_round_tol) || ...
                ~isfinite(cfg.matrix.early_exist_round_tol) || cfg.matrix.early_exist_round_tol < 0
            error('cfg.matrix.early_exist_round_tol must be a finite nonnegative scalar.');
        end
        if cfg.matrix.multistart.num_starts < 1 || ...
                cfg.matrix.multistart.num_starts ~= round(cfg.matrix.multistart.num_starts)
            error('cfg.matrix.multistart.num_starts must be a positive integer.');
        end
    end

    if matrix_init_mode == "lp_best_jcommon"
        if ~isfield(cfg.matrix,'lp_init_variant')
            error('cfg.matrix.lp_init_variant is required for lp_best_jcommon initialization.');
        end
        lp_init_variant = string(cfg.matrix.lp_init_variant);
        if ~any(lp_init_variant == ["alg2","bs"])
            error('cfg.matrix.lp_init_variant must be "alg2" or "bs".');
        end
        if cfg.methods.enabled.matrix && ~cfg.methods.enabled.lp
            error(['Matrix init_mode="lp_best_jcommon" requires "lp" in cfg.methods.run ', ...
                'because Lp must run before Matrix.']);
        end
        if lp_init_variant=="alg2" && ~cfg.lp.run_alg2
            error('lp_best_jcommon with lp_init_variant="alg2" requires cfg.lp.run_alg2=true.');
        end
        if lp_init_variant=="bs" && ~cfg.lp.run_bs
            error('lp_best_jcommon with lp_init_variant="bs" requires cfg.lp.run_bs=true.');
        end
    elseif matrix_init_mode == "eadmm_sdp"
        if cfg.methods.enabled.matrix && ~cfg.methods.enabled.eadmm
            error(['Matrix init_mode="eadmm_sdp" requires "eadmm" in cfg.methods.run ', ...
                'because eADMM must run before Matrix.']);
        end
    end

    beta_mode = string(cfg.matrix.beta_mode);
    valid_beta_modes = ["fixed","adaptive","fix_then_adaptive","adaptive_then_fix","eadmm_coupled"];
    if cfg.methods.enabled.matrix && ~any(beta_mode == valid_beta_modes)
        error('Unknown cfg.matrix.beta_mode: %s.', beta_mode);
    end

    beta_adapt_rule = string(cfg.matrix.beta_adapt_rule);
    if cfg.methods.enabled.matrix && ~any(beta_adapt_rule == ["current","eadmm_style"])
        error('cfg.matrix.beta_adapt_rule must be "current" or "eadmm_style".');
    end

    if cfg.methods.enabled.matrix
        if ~isscalar(cfg.matrix.eadmm_beta_decay) || ~isfinite(cfg.matrix.eadmm_beta_decay) || ...
                cfg.matrix.eadmm_beta_decay <= 0
            error('cfg.matrix.eadmm_beta_decay must be a positive finite scalar.');
        end
        if ~isscalar(cfg.matrix.eadmm_beta_min) || ~isfinite(cfg.matrix.eadmm_beta_min) || ...
                cfg.matrix.eadmm_beta_min <= 0
            error('cfg.matrix.eadmm_beta_min must be a positive finite scalar.');
        end
        if ~isscalar(cfg.matrix.eadmm_beta_max) || ~isfinite(cfg.matrix.eadmm_beta_max) || ...
                cfg.matrix.eadmm_beta_max < cfg.matrix.eadmm_beta_min
            error('cfg.matrix.eadmm_beta_max must be finite and >= cfg.matrix.eadmm_beta_min.');
        end
        if ~isscalar(cfg.matrix.eadmm_sigma_decay) || ~isfinite(cfg.matrix.eadmm_sigma_decay) || ...
                cfg.matrix.eadmm_sigma_decay <= 0
            error('cfg.matrix.eadmm_sigma_decay must be a positive finite scalar.');
        end
        if ~isscalar(cfg.matrix.eadmm_sigma_min) || ~isfinite(cfg.matrix.eadmm_sigma_min) || ...
                cfg.matrix.eadmm_sigma_min <= 0
            error('cfg.matrix.eadmm_sigma_min must be a positive finite scalar.');
        end
        if ~isscalar(cfg.matrix.eadmm_sigma_max) || ~isfinite(cfg.matrix.eadmm_sigma_max) || ...
                cfg.matrix.eadmm_sigma_max < cfg.matrix.eadmm_sigma_min
            error('cfg.matrix.eadmm_sigma_max must be finite and >= cfg.matrix.eadmm_sigma_min.');
        end
        if beta_mode == "eadmm_coupled"
            scale0 = max([1, abs(cfg.matrix.c), abs(cfg.matrix.beta0)]);
            if abs(cfg.matrix.c-cfg.matrix.beta0) > 1e-12*scale0
                error(['For cfg.matrix.beta_mode="eadmm_coupled", set ', ...
                    'cfg.matrix.c = cfg.matrix.beta0 so c_0 = beta_0 = sigma_0.']);
            end
        end

        if any(matrix_init_mode == ["zero","lp_best_jcommon","eadmm_sdp"]) && ...
                any(beta_mode == ["adaptive","fix_then_adaptive","adaptive_then_fix"]) && ...
                beta_adapt_rule == "current"
            warning(['This Matrix initialization uses X0=Y0, hence ||X0-Y0||_F=0, while ', ...
                'the current adaptive-beta formula uses the initial primal residual as its ', ...
                'scale. Prefer cfg.matrix.beta_adapt_rule="eadmm_style" or beta_mode="fixed".']);
        end
    end

    %% ======================== Random settings ===============================
    if cfg.methods.enabled.random
        if cfg.random.num_starts < 1 || cfg.random.num_starts ~= round(cfg.random.num_starts)
            error('cfg.random.num_starts must be a positive integer.');
        end
    end

    %% ======================== Lp settings ===================================
    if cfg.methods.enabled.lp
        if ~cfg.lp.run_alg2 && ~cfg.lp.run_bs
            error('Lp is selected but both cfg.lp.run_alg2 and cfg.lp.run_bs are false.');
        end
        if cfg.lp.num_starts < 1 || cfg.lp.num_starts ~= round(cfg.lp.num_starts)
            error('cfg.lp.num_starts must be a positive integer.');
        end
        if ~(cfg.lp.p > 0 && cfg.lp.p < 1)
            error('cfg.lp.p must be in (0,1).');
        end
        if ~isscalar(cfg.lp.sigma_minus) || ~isfinite(cfg.lp.sigma_minus) || cfg.lp.sigma_minus >= 0
            error('cfg.lp.sigma_minus must be a finite negative scalar.');
        end
        if ~islogical(cfg.lp.use_curvature_sigma0) || ~isscalar(cfg.lp.use_curvature_sigma0)
            error('cfg.lp.use_curvature_sigma0 must be a logical scalar.');
        end
    end

    %% ======================== MMD-LP settings ===============================

    if cfg.methods.enabled.mmdlp

        if exist('linprog','file') == 0
            error(['MMD-LP requires MATLAB linprog ', ...
                '(Optimization Toolbox).']);
        end

        if ~islogical(cfg.mmdlp.verbose) || ...
                ~isscalar(cfg.mmdlp.verbose)
            error('cfg.mmdlp.verbose must be a logical scalar.');
        end

        if ~islogical(cfg.mmdlp.store_relaxed_history) || ...
                ~isscalar(cfg.mmdlp.store_relaxed_history)
            error(['cfg.mmdlp.store_relaxed_history ', ...
                'must be a logical scalar.']);
        end

        if ~islogical(cfg.mmdlp.use_common_2opt) || ...
                ~isscalar(cfg.mmdlp.use_common_2opt)
            error(['cfg.mmdlp.use_common_2opt ', ...
                'must be a logical scalar.']);
        end

        if ~isscalar(cfg.mmdlp.feasibility_tol) || ...
                ~isfinite(cfg.mmdlp.feasibility_tol) || ...
                cfg.mmdlp.feasibility_tol <= 0
            error(['cfg.mmdlp.feasibility_tol ', ...
                'must be a positive finite scalar.']);
        end

        valid_lp_displays = ["none","iter","final"];

        if ~any(string(cfg.mmdlp.linprog_display) == ...
                valid_lp_displays)
            error(['cfg.mmdlp.linprog_display must be ', ...
                '''none'', ''iter'', or ''final''.']);
        end

    end

    %% ======================== eADMM settings ================================
    if cfg.methods.enabled.eadmm
        eadmm_init_mode = string(cfg.eadmm.init_mode);
        if ~any(eadmm_init_mode == ["zero","uniform"])
            error('cfg.eadmm.init_mode must be "zero" or "uniform".');
        end
        if cfg.eadmm.rounding_restarts < 1 || ...
                cfg.eadmm.rounding_restarts ~= round(cfg.eadmm.rounding_restarts)
            error('cfg.eadmm.rounding_restarts must be a positive integer.');
        end
        if ~islogical(cfg.eadmm.compute_safe_lower_bound) || ...
                ~isscalar(cfg.eadmm.compute_safe_lower_bound)
            error('cfg.eadmm.compute_safe_lower_bound must be a logical scalar.');
        end
        if cfg.eadmm.compute_safe_lower_bound && exist('linprog','file') == 0
            error(['cfg.eadmm.compute_safe_lower_bound=true requires MATLAB ', ...
                'linprog (Optimization Toolbox), matching the upstream ADMM-GP ', ...
                'post_proc_2/post_proc_3 implementation.']);
        end

        if ~islogical(cfg.eadmm.met.enabled) || ~isscalar(cfg.eadmm.met.enabled)
            error('cfg.eadmm.met.enabled must be a logical scalar.');
        end
        if cfg.eadmm.met.max_ineq_per_round < 1 || ...
                cfg.eadmm.met.max_ineq_per_round ~= round(cfg.eadmm.met.max_ineq_per_round)
            error('cfg.eadmm.met.max_ineq_per_round must be a positive integer.');
        end
        if cfg.eadmm.met.max_ineq_per_round > 20000
            error(['cfg.eadmm.met.max_ineq_per_round cannot exceed 20000 because ', ...
                'the upstream tri_sep_kc separator caps MAX_INEQ at 20000.']);
        end
        if cfg.eadmm.met.max_rounds < 1 || cfg.eadmm.met.max_rounds ~= round(cfg.eadmm.met.max_rounds)
            error('cfg.eadmm.met.max_rounds must be a positive integer.');
        end
        if ~isscalar(cfg.eadmm.met.violation_tol) || ~isfinite(cfg.eadmm.met.violation_tol) || ...
                cfg.eadmm.met.violation_tol <= 0
            error('cfg.eadmm.met.violation_tol must be a positive finite scalar.');
        end
        if ~isscalar(cfg.eadmm.met.sigma0) || ~isfinite(cfg.eadmm.met.sigma0) || cfg.eadmm.met.sigma0 <= 0
            error('cfg.eadmm.met.sigma0 must be a positive finite scalar.');
        end
        if cfg.eadmm.met.enabled && abs(cfg.eadmm.met.violation_tol-1e-3) > 10*eps
            warning(['The upstream tri_sep_kc separator has a hard-coded screening threshold ', ...
                'of 1e-3. The default cfg.eadmm.met.violation_tol=1e-3 keeps the two conventions aligned.']);
        end
    end

    %% ======================== Seeds =========================================
    seed_fields = {'data','matrix_init_base','random_partition_base', ...
        'eadmm_rounding_base','lp_perturb_base','two_opt_base'};
    for ii = 1:numel(seed_fields)
        name = seed_fields{ii};
        value = cfg.seed.(name);
        if ~isscalar(value) || ~isfinite(value) || value < 0 || value ~= round(value)
            error('cfg.seed.%s must be a nonnegative integer scalar.', name);
        end
    end

    %% ======================== Paths =========================================
    cfg.paths.project_root = project_root;
    cfg.paths.third_party_admm_gp = fullfile(project_root,'third_party','ADMM-GP');
    if cfg.methods.enabled.eadmm && ~exist(cfg.paths.third_party_admm_gp,'dir')
        error('Third-party ADMM-GP directory not found: %s',cfg.paths.third_party_admm_gp);
    end
end
