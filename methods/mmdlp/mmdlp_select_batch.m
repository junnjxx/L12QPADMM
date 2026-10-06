function [pick,m_relaxed,m_binary,info] = ...
    mmdlp_select_batch(Z,k,opts)
%MMDLP_SELECT_BATCH
% Solve the paper LP relaxation and recover a size-k binary batch by
% selecting the k largest entries of the relaxed m vector.
%
% This is the greedy reconstruction described immediately after Eq. (10)
% in Banerjee & Chakraborty (AAAI 2021).
   
    
    [m_relaxed,info] = ...
        mmdlp_solve_relaxation(Z,k,opts);

    q = numel(m_relaxed);

    % Deterministic top-k.
    %
    % Primary key  : larger m first
    % Secondary key: smaller original index first
    %
    % The secondary key only resolves exact numerical ties.
    sort_matrix = [-m_relaxed,(1:q)'];

    [~,order] = sortrows(sort_matrix,[1 2]);

    pick = order(1:k);

    m_binary = zeros(q,1);
    m_binary(pick) = 1;

    % Diagnostics:
    %
    % relaxation objective != rounded IQP objective in general.
    info.relaxation_objective = info.fval;

    info.rounded_iqp_objective = ...
        m_binary' * Z * m_binary;

    info.rounding_cardinality = ...
        sum(m_binary);

    if info.rounding_cardinality ~= k
        error('mmdlp_select_batch:RoundingCardinality', ...
            'Top-k rounding did not select exactly k samples.');
    end
end