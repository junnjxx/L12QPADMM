function [Z,parts] = build_mmd_batch_Z(Phi,Q,P,k,N)
%BUILD_MMD_BATCH_Z
% Build the quadratic matrix Z in Eq. (6)-(8) of
%
% Banerjee & Chakraborty (AAAI 2021),
% "Deterministic Mini-batch Sequencing for Training Deep Neural Networks".
%
% INPUT
%   Phi : N-by-N kernel matrix
%   Q   : indices of currently unselected samples
%   P   : indices of already selected samples
%   k   : mini-batch size
%   N   : total number of samples
%
% OUTPUT
%   Z     : nQ-by-nQ quadratic coefficient matrix
%   parts : diagnostic components Phi1, phi2, phi3
%
% The paper defines
%
%   Phi1(i,j) = phi(x_i,x_j),                  i,j in Q
%
%   phi2(i) = (nP+k)/N * sum_{j in Q} phi(x_i,x_j)
%
%   phi3(i) = (nQ-k)/N * sum_{j in P} phi(x_i,x_j)
%
% and
%
%   Z_ij = Phi1_ij,                            i ~= j
%   Z_ii = Phi1_ii - phi2_i + phi3_i.
%

    Q = Q(:);
    P = P(:);

    nQ = numel(Q);
    nP = numel(P);

    if nQ < k
        error('build_mmd_batch_Z:TooFewSamples', ...
            'nQ=%d is smaller than batch size k=%d.',nQ,k);
    end

    if nP + nQ ~= N
        error('build_mmd_batch_Z:PartitionMismatch', ...
            'nP+nQ=%d but N=%d.',nP+nQ,N);
    end

    if ~isequal(size(Phi),[N,N])
        error('build_mmd_batch_Z:KernelSizeMismatch', ...
            'Phi must be N-by-N.');
    end

    % ------------------------------------------------------------
    % Eq. (6): Phi_1
    % ------------------------------------------------------------

    Phi1 = Phi(Q,Q);

    % ------------------------------------------------------------
    % Eq. (6): phi_2
    %
    % Important: this is a ROW sum over j.
    % ------------------------------------------------------------

    phi2 = ((nP+k)/N) * sum(Phi1,2);

    % ------------------------------------------------------------
    % Eq. (6): phi_3
    % ------------------------------------------------------------

    if nP > 0

        phi3 = ((nQ-k)/N) * sum(Phi(Q,P),2);

    else

        phi3 = zeros(nQ,1);

    end

    % ------------------------------------------------------------
    % Eq. (8): Z
    % ------------------------------------------------------------

    Z = Phi1;

    diag_idx = 1:nQ+1:nQ*nQ;

    Z(diag_idx) = ...
        diag(Phi1) - phi2 + phi3;

    % Phi is symmetric in the current project, so Z should also be
    % symmetric. Do not silently alter Z; diagnose unexpected drift.
    sym_err = norm(Z-Z','fro');

    if sym_err > 1e-10*max(1,norm(Z,'fro'))
        error('build_mmd_batch_Z:UnexpectedAsymmetry', ...
            'Constructed Z is unexpectedly asymmetric: %.3e.',sym_err);
    end

    parts = struct();

    parts.Phi1 = Phi1;
    parts.phi2 = phi2;
    parts.phi3 = phi3;

    parts.nP = nP;
    parts.nQ = nQ;
    parts.k = k;
end