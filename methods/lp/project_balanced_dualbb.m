function [X,state,info] = project_balanced_dualbb(C,capacities,opts,state)
%PROJECT_BALANCED_DUALBB Euclidean projection onto the balanced transportation polytope
%
%   min_X 0.5*||X-C||_F^2
%   s.t.  X*1_m = 1_n,
%         X'*1_n = capacities,
%         X >= 0.
%
% This is the direct adaptation of Eq. (4.13).  Its dual is
%   min_{y,z} 0.5||P_+(C+y*1'+1*z')||_F^2 - 1'y - capacities'*z.
%
% The dual gradient is exactly the primal row/column residual.
%
% The paper states "gradient method using BB step sizes" but does not report
% the dualBB stopping tolerance, initial step, maximum iterations, or BB
% safeguards in Section 6.1. Those are therefore implementation choices.

[n,m] = size(C);
capacities = capacities(:);
if numel(capacities) ~= m
    error('capacities must have length size(C,2).');
end
if abs(sum(capacities)-n) > 1e-10*max(1,n)
    error('sum(capacities) must equal n.');
end

if nargin < 4 || isempty(state) || ~isfield(state,'y') || ~opts.warm_start
    y = zeros(n,1);
    z = zeros(m,1);
else
    y = state.y;
    z = state.z;
    if numel(y)~=n || numel(z)~=m
        y = zeros(n,1);
        z = zeros(m,1);
    end
end

if isempty(opts.alpha0)
    alpha = 1/max(n+m,1);
else
    alpha = opts.alpha0;
end
alpha = min(max(alpha,opts.alpha_min),opts.alpha_max);

w_prev = [];
g_prev = [];
converged = false;

for k = 1:opts.max_iter
    A = bsxfun(@plus,C,y);
    A = bsxfun(@plus,A,z');
    X = max(A,0);

    grow = sum(X,2)-1;
    gcol = sum(X,1)'-capacities;
    g = [grow; gcol];
    res_inf = norm(g,inf);

    if res_inf <= opts.tol
        converged = true;
        break;
    end

    w = [y;z];
    if ~isempty(w_prev)
        s  = w-w_prev;
        dg = g-g_prev;
        sty = s'*dg;
        if sty > 0
            if mod(k,2)==0
                % Large BB step (BB1)
                alpha = (s'*s)/sty;
            else
                % Short BB step (BB2)
                alpha = sty/(dg'*dg);
            end
        end
        if ~isfinite(alpha) || alpha<=0
            if isempty(opts.alpha0)
                alpha = 1/max(n+m,1);
            else
                alpha = opts.alpha0;
            end
        end
        alpha = min(max(alpha,opts.alpha_min),opts.alpha_max);
    end

    w_prev = w;
    g_prev = g;

    w = w-alpha*g;
    y = w(1:n);
    z = w(n+1:end);
end

% Recompute primal from final dual variables.
A = bsxfun(@plus,C,y);
A = bsxfun(@plus,A,z');
X = max(A,0);
grow = sum(X,2)-1;
gcol = sum(X,1)'-capacities;

state.y = y;
state.z = z;
info.iter = k;
info.converged = converged || norm([grow;gcol],inf)<=opts.tol;
info.res_inf = norm([grow;gcol],inf);
info.row_res_inf = norm(grow,inf);
info.col_res_inf = norm(gcol,inf);
info.alpha = alpha;

if opts.verbose && ~info.converged
    fprintf('[dualBB警告] 达到最大迭代数 max_iter=%d，当前无穷范数残差=%.3e。\n', ...
        opts.max_iter,info.res_inf);
end
end
