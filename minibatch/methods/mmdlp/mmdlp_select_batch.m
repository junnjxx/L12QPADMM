function [pick,m_relaxed,m_binary,info] = ...
    mmdlp_select_batch(Z,k,opts)
%MMDLP_SELECT_BATCH
%
% Solve the AAAI-21 LP relaxation and recover a size-k binary
% mini-batch by selecting the k largest entries of the relaxed m.
%
% Timing:
%
%   info.rounding_time
%       ONLY deterministic top-k extraction
%
%   info.rounded_objective_eval_time
%       diagnostic evaluation m'Zm after rounding
%

    %% Solve LP relaxation

    [m_relaxed,info] = ...
        mmdlp_solve_relaxation( ...
            Z,k,opts);

    q = numel(m_relaxed);

    %% ============================================================
    % Paper top-k rounding
    % ============================================================

    rounding_tic = tic;

    % Deterministic tie-breaking:
    %
    % 1. larger relaxed m_i first
    % 2. smaller local index first

    sort_matrix = ...
        [-m_relaxed,(1:q)'];

    [~,order] = ...
        sortrows(sort_matrix,[1 2]);

    pick = ...
        order(1:k);

    m_binary = ...
        zeros(q,1);

    m_binary(pick) = 1;

    info.rounding_cardinality = ...
        sum(m_binary);

    if info.rounding_cardinality ~= k

        error('mmdlp_select_batch:RoundingCardinality', ...
            'Top-k rounding did not select exactly k samples.');

    end

    info.rounding_time = ...
        toc(rounding_tic);

    %% ============================================================
    % Objective diagnostics after rounding
    % ============================================================

    obj_tic = tic;

    info.relaxation_objective = ...
        info.fval;

    info.rounded_iqp_objective = ...
        m_binary' * Z * m_binary;

    info.rounded_objective_eval_time = ...
        toc(obj_tic);
end