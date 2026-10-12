function lb = compute_safe_lower_bound_met( ...
    A_sdp,B_met,b_sdp,f_met,C_sdp,Z_sdp,S_sdp,batch_size,graph_to_jcommon_constant)
%COMPUTE_SAFE_LOWER_BOUND_MET Certified J_common LB for the MET-strengthened SDP.
%
% Upstream post_proc_3.m returns a safe lower bound for
%
%   min <C_sdp,X>
%   s.t. A(X)=b, B(X)>=f, X>=0, X psd.
%
% The graph-scale bound is converted immediately to the project's common scale
% using
%   <L,X>/2 = batch_size^2*J_common(X) + c_affine.

    if exist('linprog','file') == 0
        error(['Safe MET lower-bound post-processing requires MATLAB linprog ', ...
            '(Optimization Toolbox), as in the upstream ADMM-GP code.']);
    end

    t0 = tic;
    [dual_safe,LB_graph_internal] = ...
        post_proc_3(Z_sdp,A_sdp',B_met',C_sdp,b_sdp,f_met,S_sdp);
    elapsed = toc(t0);

    m = numel(b_sdp);
    p = numel(f_met);
    if isempty(dual_safe) || numel(dual_safe) ~= m+p || ...
            ~isfinite(LB_graph_internal) || any(~isfinite(dual_safe))
        error('Wiegele-Zhao post_proc_3 failed to return a finite MET lower bound.');
    end

    y_safe = dual_safe(1:m);
    ybar_safe = dual_safe(m+1:end);

    Z_plus = project_W(Z_sdp);

    if size(A_sdp,1) == numel(Z_sdp)
        At_op = A_sdp;
    else
        At_op = A_sdp';
    end
    if size(B_met,2) == numel(Z_sdp)
        Bt_op = B_met';
    else
        Bt_op = B_met;
    end

    Aty_safe = reshape(At_op*y_safe,size(C_sdp));
    Bty_safe = reshape(Bt_op*ybar_safe,size(C_sdp));
    S_safe = C_sdp - Z_plus - Aty_safe - Bty_safe;

    min_eig_Zplus = min(eig(full((Z_plus+Z_plus')/2)));
    min_S_safe = min(S_safe,[],'all');
    min_ybar_safe = min(ybar_safe);
    dual_identity_rel = norm(C_sdp-Aty_safe-Bty_safe-Z_plus-S_safe,'fro') / ...
        (1+norm(C_sdp,'fro'));

    LB_common = ...
        (LB_graph_internal-graph_to_jcommon_constant)/(batch_size^2);

    lb = struct();
    lb.common = LB_common;
    lb.graph_internal = LB_graph_internal;
    lb.y = y_safe;
    lb.ybar = ybar_safe;
    lb.time = elapsed;
    lb.min_eig_Zplus = min_eig_Zplus;
    lb.min_S = min_S_safe;
    lb.min_ybar = min_ybar_safe;
    lb.dual_identity_rel = dual_identity_rel;

    scale = max([1,norm(C_sdp,'fro'),abs(LB_graph_internal)]);
    check_tol = 1e-7 * scale;
    lb.numerically_feasible = (min_eig_Zplus >= -check_tol) && ...
        (min_S_safe >= -check_tol) && (min_ybar_safe >= -check_tol) && ...
        (dual_identity_rel <= 1e-7);

    if ~lb.numerically_feasible
        warning(['MET safe-LB certificate has larger-than-expected numerical ', ...
            'violations: minEig(Zplus)=%.3e, min(S)=%.3e, min(ybar)=%.3e, dualRel=%.3e.'], ...
            min_eig_Zplus,min_S_safe,min_ybar_safe,dual_identity_rel);
    end
end
