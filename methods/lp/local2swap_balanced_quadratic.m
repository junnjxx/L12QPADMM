function [X,fval,info] = local2swap_balanced_quadratic(X,model,opts)
%LOCAL2SWAP_BALANCED_QUADRATIC
% Exact best-improvement capacity-preserving 2-swap local search.
%
% This implementation is mathematically equivalent to the original
% exhaustive best-improvement N2 search, but avoids rescanning the entire
% N2 neighborhood after every accepted swap.
%
% For
%   f(X) = qscale * tr(X' * Q * X) + <C,X>,
%
% one swap exchanges
%   a in cluster r
%   b in cluster s,
%
% and preserves all cluster capacities exactly.
%
% Main acceleration:
%   1) maintain the best swap for every cluster pair (r,s);
%   2) after swapping clusters r and s, only recompute cluster-pair
%      blocks involving r or s;
%   3) build QX using cluster memberships instead of dense Q*X;
%   4) periodically rebuild QX and all swap gains to remove floating
%      point drift.
%
% Required:
%   model.kind = 'quadratic'
%   model.Q
%   model.qscale
%   model.fun
%
% Optional:
%   model.C
%
% Options:
%   opts.local_max_swaps       maximum accepted swaps; Inf = run to local
%                              optimality.
%   opts.improve_tol           strict-improvement tolerance.
%   opts.local_recompute_every full numerical refresh every this many
%                              accepted swaps. Default = 100.
%                              Set Inf to disable periodic refresh.
%   opts.local_verbose_every   print every this many accepted swaps.
%                              Default = 0 (silent).
%   opts.local_final_verify    before declaring local optimality, perform
%                              one exact full rebuild. Default = true.
%
% Output info:
%   info.nswap
%   info.nsweep              backward-compatible count with the old code:
%                            accepted swaps plus the final no-improvement
%                            check when local optimality is certified.
%   info.nfullbuild          number of full N2 cache rebuilds.
%   info.npair_refresh       number of cluster-pair gain blocks recomputed.
%   info.local_optimal
%   info.elapsed
%   info.final_best_delta

%% 0. Checks / defaults
if ~isfield(model,'kind') || ~strcmpi(model.kind,'quadratic')
    error('This local search requires model.kind = ''quadratic''.');
end

if ~isfield(opts,'local_max_swaps') || isempty(opts.local_max_swaps)
    opts.local_max_swaps = Inf;
end
if ~isfield(opts,'improve_tol') || isempty(opts.improve_tol)
    opts.improve_tol = 1e-12;
end
if ~isfield(opts,'local_recompute_every') || isempty(opts.local_recompute_every)
    opts.local_recompute_every = 100;
end
if ~isfield(opts,'local_verbose_every') || isempty(opts.local_verbose_every)
    opts.local_verbose_every = 0;
end
if ~isfield(opts,'local_final_verify') || isempty(opts.local_final_verify)
    opts.local_final_verify = true;
end

%% 1. Problem data
Q = 0.5*(model.Q + model.Q');
q = model.qscale;
[n,m] = size(X);

if isfield(model,'C') && ~isempty(model.C)
    C = model.C;
else
    C = zeros(n,m);
end
if ~isequal(size(C),[n,m])
    error('model.C must have the same size as X.');
end

qdiag = full(diag(Q));

%% 2. Current discrete assignment / cluster memberships
[~,lab] = max(X,[],2);
members = cell(m,1);
for r = 1:m
    members{r} = find(lab==r);
end

%% 3. Build QX efficiently from memberships
% Because X is a binary assignment matrix,
%   QX(:,r) = sum_{j in C_r} Q(:,j).
QX = build_QX_from_members(Q,members,n,m);

%% 4. Initial objective / counters
fval = model.fun(X);
nswap = 0;
nfullbuild = 0;
npair_refresh = 0;
local_optimal = false;
t_total = tic;

%% 5. Cache one best swap for every cluster pair
bestDelta = inf(m,m);
bestA = zeros(m,m);
bestB = zeros(m,m);

[bestDelta,bestA,bestB,nrefresh] = rebuild_all_pair_best( ...
    bestDelta,bestA,bestB,members,QX,Q,C,q,qdiag,m);
npair_refresh = npair_refresh + nrefresh;
nfullbuild = nfullbuild + 1;

%% 6. Exact best-improvement local search
while nswap < opts.local_max_swaps
    [best_delta,lin] = min(bestDelta(:));
    if isempty(best_delta)
        best_delta = Inf;
    end

    % No cached improving swap: optionally rebuild everything once before
    % certifying local optimality, to guard against floating-point drift.
    if best_delta >= -opts.improve_tol
        if opts.local_final_verify
            QX = build_QX_from_members(Q,members,n,m);
            fval = model.fun(X);

            [bestDelta,bestA,bestB,nrefresh] = rebuild_all_pair_best( ...
                bestDelta,bestA,bestB,members,QX,Q,C,q,qdiag,m);
            npair_refresh = npair_refresh + nrefresh;
            nfullbuild = nfullbuild + 1;

            [best_delta,lin] = min(bestDelta(:));
            if best_delta >= -opts.improve_tol
                local_optimal = true;
                break;
            end
        else
            local_optimal = true;
            break;
        end
    end

    % Current globally best swap.
    [r,s] = ind2sub([m,m],lin);
    besta = bestA(r,s);
    bestb = bestB(r,s);

    if besta==0 || bestb==0
        % Defensive safeguard; should not occur for a finite improving gain.
        local_optimal = true;
        break;
    end

    % Apply a:r->s and b:s->r.
    X(besta,r) = 0;
    X(besta,s) = 1;
    X(bestb,s) = 0;
    X(bestb,r) = 1;
    lab(besta) = s;
    lab(bestb) = r;

    % Update memberships without changing capacities.
    pos_a = (members{r}==besta);
    pos_b = (members{s}==bestb);
    if ~any(pos_a) || ~any(pos_b)
        error('Internal membership inconsistency in local 2-swap.');
    end
    members{r}(pos_a) = bestb;
    members{s}(pos_b) = besta;

    % Exact incremental QX update:
    % DeltaX = (e_a-e_b)(e_s-e_r)'.
    qu = Q(:,besta) - Q(:,bestb);
    QX(:,s) = QX(:,s) + qu;
    QX(:,r) = QX(:,r) - qu;

    fval = fval + best_delta;
    nswap = nswap + 1;

    if opts.local_verbose_every > 0 && ...
            mod(nswap,opts.local_verbose_every)==0
        fprintf(['[N2-fast] swaps=%d | f=%.12e | last_delta=%.3e | ', ...
                 'fullbuild=%d | pairrefresh=%d | elapsed=%.3fs\n'], ...
                 nswap,fval,best_delta,nfullbuild,npair_refresh,toc(t_total));
    end

    % Periodic exact refresh to remove accumulated numerical drift.
    do_full_refresh = false;
    if isfinite(opts.local_recompute_every) && opts.local_recompute_every>0
        if mod(nswap,opts.local_recompute_every)==0
            do_full_refresh = true;
        end
    end

    if do_full_refresh
        QX = build_QX_from_members(Q,members,n,m);
        fval = model.fun(X);

        [bestDelta,bestA,bestB,nrefresh] = rebuild_all_pair_best( ...
            bestDelta,bestA,bestB,members,QX,Q,C,q,qdiag,m);
        npair_refresh = npair_refresh + nrefresh;
        nfullbuild = nfullbuild + 1;
        continue;
    end

    % KEY ACCELERATION:
    % After swapping between r and s, QX changes only in columns r and s,
    % and memberships change only for r and s. Therefore a pair block
    % (u,v) with u,v not in {r,s} has exactly the same swap gains as before.

    % Refresh every pair involving r; this includes (r,s).
    for t = 1:m
        if t==r
            continue;
        end
        u = min(r,t);
        v = max(r,t);
        [d,a,b] = pair_best_swap(u,v,members,QX,Q,C,q,qdiag);
        bestDelta(u,v) = d;
        bestA(u,v) = a;
        bestB(u,v) = b;
        npair_refresh = npair_refresh + 1;
    end

    % Refresh every pair involving s except (r,s), already refreshed above.
    for t = 1:m
        if t==s || t==r
            continue;
        end
        u = min(s,t);
        v = max(s,t);
        [d,a,b] = pair_best_swap(u,v,members,QX,Q,C,q,qdiag);
        bestDelta(u,v) = d;
        bestA(u,v) = a;
        bestB(u,v) = b;
        npair_refresh = npair_refresh + 1;
    end
end

%% 7. Final exact objective
fval = model.fun(X);

%% 8. Information
info.nswap = nswap;
% Preserve the old routine's sweep-count convention for existing profiling:
% one global best-selection cycle per accepted swap, plus the final
% no-improvement cycle when local optimality is actually established.
info.nsweep = nswap + double(local_optimal);
info.nfullbuild = nfullbuild;
info.npair_refresh = npair_refresh;
info.local_optimal = local_optimal;
info.elapsed = toc(t_total);

[final_best_delta,~] = min(bestDelta(:));
if isempty(final_best_delta)
    final_best_delta = Inf;
end
info.final_best_delta = final_best_delta;
end

% ========================================================================
% Helper 1: build QX from cluster memberships
% ========================================================================
function QX = build_QX_from_members(Q,members,n,m)
QX = zeros(n,m);
for r = 1:m
    idx = members{r};
    if isempty(idx)
        QX(:,r) = 0;
    else
        QX(:,r) = full(sum(Q(:,idx),2));
    end
end
end

% ========================================================================
% Helper 2: best exact swap between one fixed pair of clusters (r,s)
% ========================================================================
function [best_delta,besta,bestb] = pair_best_swap( ...
    r,s,members,QX,Q,C,q,qdiag)

A = members{r};
B = members{s};

if isempty(A) || isempty(B)
    best_delta = Inf;
    besta = 0;
    bestb = 0;
    return;
end

termA = 2*q*(QX(A,s)-QX(A,r)) + ...
        (C(A,s)-C(A,r)) + 2*q*qdiag(A);
termB = 2*q*(QX(B,r)-QX(B,s)) + ...
        (C(B,r)-C(B,s)) + 2*q*qdiag(B);

% Since Q is symmetrized,
% -2q(Q_ab+Q_ba) = -4q*Q_ab.
D = termA + termB' - 4*q*full(Q(A,B));

[best_delta,lin] = min(D(:));
[ia,ib] = ind2sub(size(D),lin);
besta = A(ia);
bestb = B(ib);
end

% ========================================================================
% Helper 3: full rebuild of all cluster-pair best gains
% ========================================================================
function [bestDelta,bestA,bestB,nrefresh] = rebuild_all_pair_best( ...
    bestDelta,bestA,bestB,members,QX,Q,C,q,qdiag,m)

bestDelta(:) = Inf;
bestA(:) = 0;
bestB(:) = 0;
nrefresh = 0;

for r = 1:m-1
    for s = r+1:m
        [d,a,b] = pair_best_swap(r,s,members,QX,Q,C,q,qdiag);
        bestDelta(r,s) = d;
        bestA(r,s) = a;
        bestB(r,s) = b;
        nrefresh = nrefresh + 1;
    end
end
end
