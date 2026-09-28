function lb = compute_safe_lower_bound( ...
    A_sdp,b_sdp,C_sdp,Z_sdp,S_sdp,batch_size,graph_to_jcommon_constant)
%COMPUTE_SAFE_LOWER_BOUND Return a certified lower bound on the J_common scale.
%
% The upstream Wiegele-Zhao post_proc_2.m works internally with their graph
% SDP objective
%
%   min <C_sdp,X> = <L,X>/2.
%
% This project does not expose that graph scale.  The upstream bound is
% immediately converted using the equipartition identity
%
%   <L,X>/2 = batch_size^2*J_common(X) + c_affine,
%
% hence
%
%   LB_Jcommon = (LB_graph-c_affine)/batch_size^2.
%
% The returned struct exposes J_common as the project-wide bound.  For logging
% clarity it also stores the author's temporary graph-scale bound in
% lb.graph_internal; this internal value is never used for cross-method gaps.

    if exist('linprog','file') == 0
        error(['Safe lower-bound post-processing requires MATLAB linprog ', ...
            '(Optimization Toolbox), as in the upstream ADMM-GP code.']);
    end

    t0 = tic;

    % Upstream README call convention:
    %   [ynew,LB_graph] = post_proc_2(Z, A', C, b, S)
    [y_safe,LB_graph_internal] = ...
        post_proc_2(Z_sdp,A_sdp',C_sdp,b_sdp,S_sdp);

    elapsed = toc(t0);

    if isempty(y_safe) || ~isfinite(LB_graph_internal) || any(~isfinite(y_safe))
        error('Wiegele-Zhao post_proc_2 failed to return a finite lower bound.');
    end

    % Reconstruct the repaired dual slack for validation/reporting.
    Z_plus = project_W(Z_sdp);

    if size(A_sdp,1) == numel(Z_sdp)
        At_op = A_sdp;      % n^2-by-m representation of A^*
    else
        At_op = A_sdp';
    end

    Aty_safe = reshape(At_op*y_safe,size(C_sdp));
    S_safe = C_sdp - Z_plus - Aty_safe;

    min_eig_Zplus = min(eig(full((Z_plus+Z_plus')/2)));
    min_S_safe = min(S_safe,[],'all');
    dual_identity_rel = norm(C_sdp-Aty_safe-Z_plus-S_safe,'fro') / ...
        (1+norm(C_sdp,'fro'));

    LB_common = ...
        (LB_graph_internal-graph_to_jcommon_constant)/(batch_size^2);

    lb = struct();
    lb.common = LB_common;
    lb.graph_internal = LB_graph_internal;
    lb.y = y_safe;
    lb.time = elapsed;
    lb.min_eig_Zplus = min_eig_Zplus;
    lb.min_S = min_S_safe;
    lb.dual_identity_rel = dual_identity_rel;

    % Numerical feasibility diagnostic. post_proc_2 is the author's safe-bound
    % procedure; this flag only detects unexpectedly large floating-point/LP
    % violations in the returned certificate.
    scale = max([1,norm(C_sdp,'fro'),abs(LB_graph_internal)]);
    check_tol = 1e-7 * scale;
    lb.numerically_feasible = (min_eig_Zplus >= -check_tol) && ...
        (min_S_safe >= -check_tol) && (dual_identity_rel <= 1e-7);

    if ~lb.numerically_feasible
        warning(['Safe-LB certificate has larger-than-expected numerical ', ...
            'violations: minEig(Zplus)=%.3e, min(S)=%.3e, dualRel=%.3e.'], ...
            min_eig_Zplus,min_S_safe,dual_identity_rel);
    end
end
