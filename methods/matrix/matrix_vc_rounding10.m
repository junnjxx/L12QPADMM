function vc = matrix_vc_rounding10(Y, problem, opts)
%MATRIX_VC_ROUNDING10
%
% Apply the SAME eADMM vector-clustering rounding heuristic to
%
%       S = Y*Y'
%
% using multiple independent restarts.
%
% The actual clustering routine is the existing eADMM function
%
%       kequi_rounding
%
% rather than a reimplementation.
%
% IMPORTANT PROJECT CONVENTION:
%
%   I has size
%
%       batch_size x num_batches
%
%   NOT
%
%       num_batches x batch_size.
%
% Example:
%
%   N = 100
%   batch_size = 4
%   num_batches = 25
%
% then
%
%       size(I) = 4 x 25.
%
%
% Output:
%
%   vc.S
%   vc.best_I
%   vc.best_P
%   vc.best_metrics
%   vc.best_jcommon
%   vc.best_restart
%   vc.rounding_time
%   vc.restart_results


%% ======================== Defaults ======================================

    if ~isfield(opts,'vc_restarts')
        opts.vc_restarts = 10;
    end

    if ~isfield(opts,'vc_seed_base')
        opts.vc_seed_base = 9001;
    end


    R = ...
        opts.vc_restarts;

    seed_base = ...
        opts.vc_seed_base;


    if R ~= 10

        warning( ...
            ['Matrix-VC is intended for the same Vc(10) comparison ', ...
             'as eADMM; vc_restarts=%d was supplied.'], ...
            R);
    end


%% ======================== Dimensions ====================================

    [N,B] = ...
        size(Y);


    bs = ...
        problem.batch_size;


    if N ~= problem.num_samples

        error( ...
            ['Y row count (%d) does not match ', ...
             'problem.num_samples (%d).'], ...
            N, ...
            problem.num_samples);
    end


    if B ~= problem.num_batches

        error( ...
            ['Y column count (%d) does not match ', ...
             'problem.num_batches (%d).'], ...
            B, ...
            problem.num_batches);
    end


    if N ~= B*bs

        error( ...
            ['Balanced Vc requires ', ...
             'N = num_batches * batch_size. ', ...
             'Got N=%d, B=%d, bs=%d.'], ...
            N, ...
            B, ...
            bs);
    end


%% ======================== Check eADMM Vc ================================

    if exist('kequi_rounding','file') ~= 2

        error( ...
            ['Cannot find eADMM kequi_rounding.m on the MATLAB path. ', ...
             'Add the ADMM-GP heuristics directory to the MATLAB path.']);
    end


%% ======================== Matrix-ADMM Gram matrix =======================

    S = ...
        Y*Y';


    % Numerical symmetrization only.
    S = ...
        0.5*(S+S');


%% ======================== Objective matrix for Vc =======================

% kequi_rounding itself uses C only to evaluate the discrete co-membership
% matrix produced in that restart.
%
% When Phi is available, use the exact J_common coefficient
%
%       J_common = <Phi, PP'> / (2*bs^2).
%
% Thus:
%
%       C_vc = Phi/(2*bs^2).

    if ...
            isfield(problem,'Phi') ...
            && ...
            isequal(size(problem.Phi),[N,N])

        C_vc = ...
            problem.Phi ...
            / ...
            (2*bs^2);

    else

        % Partition generation in kequi_rounding does not depend on C.
        % Final comparison below always uses evaluate_partition.

        C_vc = ...
            zeros(N,N);
    end


%% ======================== Storage =======================================

    restart_template = struct( ...
        'restart',NaN, ...
        'seed',NaN, ...
        'I',[], ...
        'P',[], ...
        'JCommon',Inf, ...
        'metrics',[], ...
        'rounding_time',NaN);


    restart_results = ...
        repmat( ...
            restart_template, ...
            R, ...
            1);


    best_jcommon = ...
        Inf;


    best_restart = ...
        NaN;


    best_I = [];
    best_P = [];
    best_metrics = [];


%% ======================== Vc restarts ===================================

    t_all = ...
        tic;


    for restart = 1:R

        seed = ...
            seed_base ...
            + ...
            restart ...
            - ...
            1;


        t_one = ...
            tic;


        [ ...
            ~, ...
            ~, ...
            ~, ...
            part_cell ...
        ] = ...
            kequi_rounding( ...
                seed, ...
                S, ...
                B, ...
                C_vc);


        round_time = ...
            toc(t_one);


        %% ---------------------------------------------------------------
        % Convert eADMM cell partition to project batch matrix I.
        %
        % Project convention:
        %
        %       size(I) = batch_size x num_batches
        %
        % Example:
        %
        %       bs = 4, B = 25
        %
        %       I is 4 x 25.
        % ---------------------------------------------------------------

        I_candidate = ...
            zeros(bs,B);


        for b = 1:B

            idx = ...
                part_cell{b};


            idx = ...
                idx(:);


            if numel(idx) ~= bs

                error( ...
                    ['Vc restart %d produced cluster %d ', ...
                     'with size %d instead of %d.'], ...
                    restart, ...
                    b, ...
                    numel(idx), ...
                    bs);
            end


            I_candidate(:,b) = ...
                idx;
        end


        %% ---------------------------------------------------------------
        % Feasibility check
        %
        % Every sample 1,...,N must appear exactly once.
        % ---------------------------------------------------------------

        all_idx = ...
            sort( ...
                I_candidate(:));


        if ~isequal( ...
                all_idx, ...
                (1:N)')

            error( ...
                ['Vc restart %d produced an invalid partition: ', ...
                 'indices are not exactly 1:N.'], ...
                restart);
        end


        %% ---------------------------------------------------------------
        % Convert to assignment representation used throughout project
        % ---------------------------------------------------------------

        P_candidate = ...
            batches_to_assignment( ...
                I_candidate, ...
                problem.num_samples, ...
                problem.num_batches, ...
                problem.batch_size);


        %% ---------------------------------------------------------------
        % Evaluate using the common cross-method objective
        % ---------------------------------------------------------------

        metrics_candidate = ...
            evaluate_partition( ...
                P_candidate, ...
                problem, ...
                opts.eta);


        if ~metrics_candidate.valid

            error( ...
                'Vc restart %d produced an invalid evaluated partition.', ...
                restart);
        end


        J = ...
            metrics_candidate.common_objective;


        %% ---------------------------------------------------------------
        % Store this restart
        % ---------------------------------------------------------------

        restart_results(restart).restart = ...
            restart;


        restart_results(restart).seed = ...
            seed;


        restart_results(restart).I = ...
            I_candidate;


        restart_results(restart).P = ...
            P_candidate;


        restart_results(restart).JCommon = ...
            J;


        restart_results(restart).metrics = ...
            metrics_candidate;


        restart_results(restart).rounding_time = ...
            round_time;


        %% ---------------------------------------------------------------
        % Update best restart
        % ---------------------------------------------------------------

        if J < best_jcommon

            best_jcommon = ...
                J;


            best_restart = ...
                restart;


            best_I = ...
                I_candidate;


            best_P = ...
                P_candidate;


            best_metrics = ...
                metrics_candidate;
        end
    end


    total_rounding_time = ...
        toc(t_all);


%% ======================== Output ========================================

    vc = ...
        struct();


    vc.method = ...
        "Matrix-KKT+Vc";


    vc.S = ...
        S;


    vc.num_restarts = ...
        R;


    vc.seed_base = ...
        seed_base;


    vc.best_restart = ...
        best_restart;


    vc.best_I = ...
        best_I;


    vc.best_P = ...
        best_P;


    vc.best_metrics = ...
        best_metrics;


    vc.best_jcommon = ...
        best_jcommon;


    vc.rounding_time = ...
        total_rounding_time;


    vc.restart_results = ...
        restart_results;


%% ======================== Sanity print ==================================

    fprintf( ...
        ['[Matrix-KKT+Vc rounding] ', ...
         'best restart=%d/%d | ', ...
         'J_common=%.10e | ', ...
         'size(I)=%d-by-%d | ', ...
         'time=%.6f s\n'], ...
        best_restart, ...
        R, ...
        best_jcommon, ...
        size(best_I,1), ...
        size(best_I,2), ...
        total_rounding_time);

end