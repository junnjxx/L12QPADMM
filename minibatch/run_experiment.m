%% RUN_EXPERIMENT
% Top-level experiment entry point.
%
% User-facing settings live in ONE file only:
%
%   config/config.m
%
% utils/finalize_config.m only derives quantities and validates settings.
% This file intentionally contains no numerical experiment parameters.


clear;
close all;
clc;


project_root = fileparts(mfilename('fullpath'));


% Add only directories used by the current experiment.
% legacy_downstream is intentionally excluded to avoid
% old/new function-name conflicts.

addpath(fullfile(project_root,'config'));
addpath(fullfile(project_root,'data'));
addpath(fullfile(project_root,'problem'));
addpath(genpath(fullfile(project_root,'methods')));
addpath(fullfile(project_root,'postprocess'));
addpath(fullfile(project_root,'experiments'));
addpath(fullfile(project_root,'reporting'));
addpath(fullfile(project_root,'utils'));
addpath(genpath(fullfile(project_root,'third_party','ADMM-GP')));



%% 1) Load the single configuration file

cfg = config();



%% 2) Resolve derived quantities and validate configuration

cfg = finalize_config(cfg,project_root);



%% 3) 为日志和 MATLAB 数值结果生成同名、唯一的文件路径

% cfg.output.save_results 在 config/config.m 中设为 true 时，
% 自动将 MAT 保存到 <project_root>/results/。
%
% 日志与 MAT 共用同一个时间戳及重复编号，方便一一对应。

log_dir = fullfile(project_root,'logs');

if ~exist(log_dir,'dir')
    mkdir(log_dir);
end


timestamp = datestr(now,'yyyymmdd_HHMMSS');


profile_token = char(string(cfg.data.profile));
profile_token = regexprep(profile_token,'[^A-Za-z0-9_-]','-');


matrix_init_token = char(string(cfg.matrix.init_mode));
matrix_init_token = regexprep(matrix_init_token,'[^A-Za-z0-9_-]','-');


early_exist_token = double(cfg.matrix.early_exist);


method_token = char(strjoin(cfg.methods.run,'-'));
method_token = regexprep(method_token,'[^A-Za-z0-9_-]','-');


log_base_name = sprintf( ...
    'batch_%s_N%d_bs%d_Methods_%s_MatrixInit_%s_EarlyExist%d_%s', ...
    profile_token, ...
    cfg.data.num_samples, ...
    cfg.batch.size, ...
    method_token, ...
    matrix_init_token, ...
    early_exist_token, ...
    timestamp ...
    );


if string(cfg.matrix.init_mode) == "lp_best_jcommon"

    lp_variant_token = char(string(cfg.matrix.lp_init_variant));

    lp_variant_token = regexprep( ...
        lp_variant_token,'[^A-Za-z0-9_-]','-');

    log_base_name = sprintf( ...
        '%s_LPinit_%s',log_base_name,lp_variant_token);

end


% 只有开启 MAT 保存时才创建结果目录；
% 关闭时只保存控制台日志。

save_mat_results = logical(cfg.output.save_results);


if save_mat_results

    result_dir = fullfile(project_root,'results');

    if ~exist(result_dir,'dir')
        mkdir(result_dir);
    end

end


% 同一次实验的 .txt 与 .mat 共用一个 basename；
% 同一秒重复运行也不覆盖。

run_base_name = log_base_name;
duplicate_id = 1;


while true

    log_file = fullfile(log_dir,[run_base_name,'.txt']);

    if save_mat_results

        result_file = fullfile( ...
            result_dir,[run_base_name,'.mat']);

        name_taken = exist(log_file,'file') || ...
                     exist(result_file,'file');

    else

        name_taken = exist(log_file,'file');

    end


    if ~name_taken
        break;
    end


    run_base_name = sprintf( ...
        '%s_%02d',log_base_name,duplicate_id);

    duplicate_id = duplicate_id + 1;

end


% 覆盖 config 中原有的固定 MAT 文件名，
% 避免多次运行覆盖旧结果。

if save_mat_results
    cfg.output.result_file = result_file;
end


diary off;
diary(log_file);

diary_cleanup = onCleanup(@() diary('off'));

cfg.output.log_file = log_file;


fprintf('\n');
fprintf('============================================================\n');
fprintf('★ 本次 minibatch 实验控制台日志已开启 ★\n');

fprintf('日志文件：\n');
fprintf('  %s\n',log_file);


if save_mat_results

    fprintf('本次 MAT 数值结果将在实验结束后保存到：\n');
    fprintf('  %s\n',cfg.output.result_file);

else

    fprintf(['本次 MAT 数值结果保存 = 关闭 ', ...
        '（cfg.output.save_results=false）\n']);

end


fprintf('------------------------------------------------------------\n');

fprintf('本次启用方法（按 cfg.methods.run 顺序执行） = %s\n', ...
    strjoin(cfg.methods.run,' -> '));


fprintf('profile=%s | N=%d | batch size=%d | batch 数=%d\n', ...
    string(cfg.data.profile), ...
    cfg.data.num_samples, ...
    cfg.batch.size, ...
    cfg.batch.num_batches);


if cfg.methods.enabled.matrix

    fprintf(['Matrix init=%s | early_exist=%d | ', ...
        'early_exist_dxy_tol=%.3e | ', ...
        'round_tol=%.3e | c=%g | beta0=%g | ', ...
        'eta=%g | beta_mode=%s\n'], ...
        string(cfg.matrix.init_mode), ...
        double(cfg.matrix.early_exist), ...
        cfg.matrix.early_exist_dxy_tol, ...
        cfg.matrix.early_exist_round_tol, ...
        cfg.matrix.c, ...
        cfg.matrix.beta0, ...
        cfg.matrix.eta, ...
        string(cfg.matrix.beta_mode));


    if string(cfg.matrix.init_mode) == "lp_best_jcommon"

        fprintf('Matrix LP warm-start variant=%s\n', ...
            string(cfg.matrix.lp_init_variant));

    end

end


fprintf('============================================================\n');



%% 4) Generate data and build one shared minibatch-selection problem

% 新增：MNIST 数据集加载

if string(cfg.data.profile) == "mnist_small1000"

    data = load_mnist_small1000(cfg,project_root);


    % Guard against accidentally comparing an MNIST-labelled synthetic
    % problem with a genuine MNIST instance. Validate the actual returned
    % data, not just cfg.data.profile or a log filename.

    assert(isfield(data,'mnist_train_indices') && ...
        numel(data.mnist_train_indices) == 1000 && ...
        numel(unique(data.mnist_train_indices)) == 1000, ...
        'MNIST data source check failed: missing/invalid training IDX indices.');


    assert(isequal(size(data.sample_stationary_points),[1000,784]), ...
        'MNIST data source check failed: expected 1000-by-784 image matrix.');


    mnist_x = data.sample_stationary_points;


    assert(all(isfinite(mnist_x(:))), ...
        'MNIST data source check failed: image matrix contains NaN/Inf.');


    if cfg.data.mnist_normalize

        assert(all(mnist_x(:) >= 0 & mnist_x(:) <= 1), ...
            'MNIST data source check failed: normalized pixels must lie in [0,1].');

    else

        assert(all(mnist_x(:) >= 0 & mnist_x(:) <= 255), ...
            'MNIST data source check failed: raw pixels must lie in [0,255].');

    end


    fprintf(['[Data source verified] MNIST training IDX | seed=%d | ', ...
        'normalize=%d | N=%d | dim=%d | pixel_min=%.6g | pixel_max=%.6g\n'], ...
        cfg.seed.data,logical(cfg.data.mnist_normalize), ...
        data.num_samples,data.dim_x,min(mnist_x(:)),max(mnist_x(:)));


    clear mnist_x;


else

    data = generate_synthetic_problem(cfg);

    fprintf('[Data source verified] synthetic generator | profile=%s\n', ...
        char(string(cfg.data.profile)));

end


% 所有方法使用同一个 problem，确保评价口径一致。

problem = build_batching_problem(data,cfg);



%% 5) Print experiment information

fprintf('\n============================================================\n');

fprintf('minibatch 统一对比实验\n');


fprintf('启用方法：%s\n', ...
    strjoin(cfg.methods.run,', '));


fprintf(['实验配置：profile=%s | 样本数 N=%d | ', ...
    '特征维数=%d | batch size=%d | batch 数=%d\n'], ...
    string(cfg.data.profile), ...
    problem.num_samples, ...
    data.dim_x, ...
    problem.batch_size, ...
    problem.num_batches);


if cfg.methods.enabled.matrix

    fprintf('Matrix 初始化方式 = %s\n', ...
        string(cfg.matrix.init_mode));


    fprintf(['Matrix early_exist = %d | ', ...
        'dxy门槛 = %.3e | round门槛 = %.3e\n'], ...
        double(cfg.matrix.early_exist), ...
        cfg.matrix.early_exist_dxy_tol, ...
        cfg.matrix.early_exist_round_tol);

end


fprintf(['共享问题构造时间 = %.6f 秒', ...
    '（只计算一次，不计入各方法运行时间）\n'], ...
    problem.build_time);


fprintf('============================================================\n\n');



%% 6) Run selected methods EXACTLY in cfg.methods.run order
%
% 执行顺序完全由 cfg.methods.run 决定。
%
% Example:
%
%   cfg.methods.run = ["Matrix", "Lp", "Vector"];
%
% actual execution order:
%
%   Matrix -> Lp -> Vector
%
% Example:
%
%   cfg.methods.run = ["Vector", "Matrix", "Lp"];
%
% actual execution order:
%
%   Vector -> Matrix -> Lp


results = struct();


run_methods = string(cfg.methods.run);


for method_idx = 1:numel(run_methods)

    method_name = strtrim(run_methods(method_idx));


    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('开始运行方法 %d/%d：%s\n', ...
        method_idx,numel(run_methods),method_name);
    fprintf('============================================================\n');


    switch lower(method_name)


        % ========================================================
        % eADMM
        % ========================================================

        case "eadmm"

            fprintf(['==================== ', ...
                'eADMM：SDP 求解 + rounding ', ...
                '====================\n']);


            results.eadmm = run_eadmm_baseline( ...
                problem,cfg);



        % ========================================================
        % Lp
        % ========================================================

        case "lp"

            fprintf(['==================== ', ...
                'Lp 正则化方法（Algorithm 2 / Lp-bs） ', ...
                '====================\n']);


            results.lp = run_lp_baseline( ...
                problem,cfg);



        % ========================================================
        % Random
        % ========================================================

        case "random"

            fprintf(['==================== ', ...
                'Random 平衡随机基线 ', ...
                '====================\n']);


            results.random = run_random_baseline( ...
                problem,cfg);



        % ========================================================
        % Vector
        % ========================================================

        case "vector"

            fprintf(['==================== ', ...
                'Vector 实验 ', ...
                '====================\n']);


            results.vector = run_vector_baseline( ...
                problem,cfg);

        % ========================================================
        % MMDLP 
        % ========================================================
        case "mmdlp"
            fprintf(['\n==================== ', ...
        'MMD-LP：AAAI-21 Eq. (10) LP relaxation ', ...
        '====================\n']);

            results.mmdlp = ...
        run_mmdlp_baseline(problem,cfg);
        % ========================================================
        % Matrix
        % ========================================================
  
        case "matrix"

            fprintf(['==================== ', ...
                'Matrix 实验 ', ...
                '====================\n']);


            % ----------------------------------------------------
            % Dependency protection:
            %
            % 如果 Matrix 明确要求使用 Lp 结果作为 warm start，
            % 那么 Lp 必须已经在 cfg.methods.run 中先执行。
            %
            % 对普通 Matrix 初始化没有任何影响。
            % ----------------------------------------------------

            if string(cfg.matrix.init_mode) == "lp_best_jcommon" && ...
                    ~isfield(results,'lp')

                error( ...
                    'run_experiment:MissingLPWarmStart', ...
                    ['Matrix init_mode="lp_best_jcommon", ', ...
                     'but Lp has not been run yet.\n', ...
                     'Current cfg.methods.run order: %s\n', ...
                     'Please place "Lp" before "Matrix", ', ...
                     'or use a Matrix initialization mode ', ...
                     'that does not depend on Lp.'], ...
                    strjoin(run_methods,' -> ') ...
                    );

            end


            results.matrix = run_matrix_multistart( ...
                problem,cfg,results);



        % ========================================================
        % Unknown method
        % ========================================================

        otherwise

            error( ...
                'run_experiment:UnknownMethod', ...
                ['Unknown method "%s" in cfg.methods.run.\n', ...
                 'Supported methods are: ', ...
                 'eADMM, Lp, Random, Vector, Matrix.'], ...
                char(method_name) ...
                );

    end


    fprintf('\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('方法 %s 运行完成。\n',method_name);
    fprintf('------------------------------------------------------------\n');

end



%% 7) Build and print final report

% Preserve the original rich four-method report when
% all four original methods run and Vector is disabled.
%
% When Vector is enabled, use the vector-aware selected report.
%
% For a selected subset, use the subset-safe report that
% only touches results that actually exist.


all_four = ...
    cfg.methods.enabled.eadmm && ...
    cfg.methods.enabled.lp && ...
    cfg.methods.enabled.random && ...
    cfg.methods.enabled.matrix;


if all_four && ~cfg.methods.enabled.vector&&  ~cfg.methods.enabled.mmdlp

    % 原有四种方法全部运行，且未启用 Vector：
    % 保留原来的四方法完整报告。

    report = build_comparison_report( ...
        results,problem,cfg);


    print_comparison_report( ...
        report,results,problem,cfg);


else

    % 启用 Vector，或者只选择部分方法：
    % 使用支持所选方法的统一报告。

    report = build_selected_methods_report( ...
        results,problem,cfg);


    print_selected_methods_report( ...
        report,results,problem,cfg);

end



%% 8) 保存 MATLAB 数值结果
%
% 目录和唯一文件名在第 3 节已设置。

if save_mat_results

    save(cfg.output.result_file, ...
        'cfg', ...
        'data', ...
        'problem', ...
        'results', ...
        'report', ...
        '-v7.3');


    fprintf('\n★ MATLAB 数值结果已保存到：\n');
    fprintf('  %s\n',cfg.output.result_file);

end



%% 9) Print log location and close diary explicitly

fprintf('\n============================================================\n');

fprintf('★ 本次 minibatch 实验已完成 ★\n');


fprintf('实际运行方法：%s\n', ...
    strjoin(cfg.methods.run,' -> '));


fprintf('控制台完整日志已保存到：\n');
fprintf('  %s\n',log_file);


fprintf('============================================================\n');


diary off;

clear diary_cleanup;