function [P0, info] = eadmm_sdp_to_balanced_assignment_init(Xsdp, num_batches, batch_size, proj_opts)
%EADMM_SDP_TO_BALANCED_ASSIGNMENT_INIT Convert eADMM's N-by-N SDP matrix
% into a continuous N-by-k balanced assignment warm start for Matrix.
%
% eADMM relaxes the co-membership matrix PP', so its SDP variable is N-by-N,
% whereas Matrix works with the assignment matrix P in R^{N-by-k}.  The two
% objects therefore cannot be copied directly.  This routine uses the SDP
% matrix itself, without first applying the author's discrete Vc rounding:
%
%   1) symmetrize Xsdp and form its top-k PSD embedding;
%   2) choose k representative anchor rows by deterministic farthest-point
%      sampling in that embedding;
%   3) use the corresponding SDP similarities Xsdp(:,anchors) as k soft
%      assignment scores;
%   4) Euclidean-project those scores onto the balanced transportation
%      polytope P*1=1, P'*1=batch_size*1, P>=0.
%
% For an ideal integral SDP solution Xsdp=P*P', one anchor per cluster makes
% Xsdp(:,anchors) exactly equal to P up to column permutation.  Thus this is a
% natural continuous bridge from the SDP representation to Matrix's variable.

    Xsdp = full(0.5*(Xsdp + Xsdp'));
    N = size(Xsdp,1);
    if size(Xsdp,2) ~= N
        error('Xsdp must be square.');
    end
    if N ~= num_batches*batch_size
        error('Need N = num_batches * batch_size.');
    end

    % Top-k PSD embedding of the co-membership relaxation.
    [V,Dmat] = eig(Xsdp);
    D = diag(Dmat);
    [lambda,ord] = sort(real(D),'descend');
    ord = ord(1:num_batches);
    lambda = max(lambda(1:num_batches),0);
    Z = real(V(:,ord)) * diag(sqrt(lambda));

    % Deterministic farthest-point anchors.  In an ideal PP' matrix, rows from
    % different clusters are mutually separated and this picks one per cluster.
    anchors = zeros(num_batches,1);
    row_norm2 = sum(Z.^2,2);
    [~,anchors(1)] = max(row_norm2);
    min_dist2 = inf(N,1);
    chosen = false(N,1);
    chosen(anchors(1)) = true;

    for jj = 2:num_batches
        zprev = Z(anchors(jj-1),:);
        d2 = sum((Z-zprev).^2,2);
        min_dist2 = min(min_dist2,d2);
        min_dist2(chosen) = -inf;
        [best_dist,next_anchor] = max(min_dist2);
        if ~isfinite(best_dist)
            next_anchor = find(~chosen,1,'first');
        end
        anchors(jj) = next_anchor;
        chosen(next_anchor) = true;
    end

    % Similarity-to-anchor scores are exactly assignment columns in the ideal
    % integral case.  Clip only tiny/negative numerical SDP violations.
    score = max(Xsdp(:,anchors),0);

    capacities = batch_size*ones(num_batches,1);
    tproj = tic;
    [P0,~,pinfo] = project_balanced_dualbb(score,capacities,proj_opts,[]);
    projection_time = toc(tproj);

    info = struct();
    info.anchors = anchors;
    info.top_eigenvalues = lambda;
    info.score_min = min(score(:));
    info.score_max = max(score(:));
    info.projection_time = projection_time;
    info.projection_res_inf = pinfo.res_inf;
    info.projection_iterations = pinfo.iter;
    info.projection_converged = pinfo.converged;
    info.row_res_inf = norm(sum(P0,2)-1,inf);
    info.col_res_inf = norm(sum(P0,1)'-capacities,inf);
    info.min_entry = min(P0(:));
    info.max_entry = max(P0(:));
end
