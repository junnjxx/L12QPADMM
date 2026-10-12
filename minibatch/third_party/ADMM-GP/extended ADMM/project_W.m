function Wp = project_W(W)
%PROJECT_W Orthogonal projection of a symmetric matrix onto the PSD cone.
%
% Compatibility helper for the public ADMM-GP code.  The upstream
% post_proc_2.m calls project_W, but project_W.m is not present in the public
% repository tree used by this project.  This helper implements the standard
% spectral projection used by the eADMM code itself:
%
%   W = Q*diag(lambda)*Q',
%   Pi_{S_+}(W) = Q*diag(max(lambda,0))*Q'.

    W = (W + W') / 2;
    [Q,D] = eig(full(W));
    lambda = diag(D);
    idx = lambda > 0;

    if any(idx)
        Qp = Q(:,idx);
        Wp = Qp * diag(lambda(idx)) * Qp';
    else
        Wp = zeros(size(W));
    end

    Wp = (Wp + Wp') / 2;
end
