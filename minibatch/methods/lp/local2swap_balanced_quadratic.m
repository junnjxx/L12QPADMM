function [X,fval,info] = local2swap_balanced_quadratic(X,model,opts)
%LOCAL2SWAP_BALANCED_QUADRATIC
% Exact accelerated best-improvement capacity-preserving N2 local search.
%
% Mathematical rule:
%   At every local-search iteration, choose the globally best improving
%   feasible 2-swap in the CURRENT N2(X), exactly as in N2-best search.
%
% Acceleration:
%   - cache the best exact swap for every cluster pair (r,s);
%   - after swapping clusters r and s, only pair blocks involving r or s
%     can change, so only those blocks are recomputed;
%   - update QX exactly after each accepted swap;
%   - periodically rebuild everything, and perform a final full rebuild
%     before certifying N2-local optimality.
%
% This is NOT a heuristic neighborhood restriction: every current
% cluster-pair block has an exact best swap, and the globally smallest
% cached gain is selected.
%
% Objective:
%   f(X) = qscale * tr(X' * Q * X) + <C,X>.
%
% Options:
%   opts.local_max_swaps       maximum accepted swaps; Inf = run to local
%                              optimality.
%   opts.improve_tol           strict-improvement tolerance.
%   opts.local_recompute_every full exact refresh every this many accepted
%                              swaps; default 100, Inf disables periodic
%                              refresh.
%   opts.local_verbose_every   print every this many accepted swaps;
%                              default 0.
%   opts.local_final_verify    full exact rebuild before declaring local
%                              optimality; default true.

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

%% 2. Verify binary one-hot assignment
tol_bin = 1e-10;

if any(abs(sum(X,2)-1) > tol_bin)
    error('Input X must have exactly one assigned cluster per row.');
end

if any(abs(X(:)-round(X(:))) > tol_bin)
    error('Input X must be binary.');
end

X = round(X);

[~,lab] = max(X,[],2);

members = cell(m,1);
for r = 1:m
    members{r} = find(lab==r);
end

%% 3. Exact QX for the current assignment
QX = build_QX_from_members(Q,members,n,m);

%% 4. Initial objective / counters
fval = model.fun(X);

nswap = 0;
nfullbuild = 0;
npair_refresh = 0;
local_optimal = false;
t_total = tic;

%% 5. Exact best swap for every cluster pair
bestDelta = inf(m,m);
bestA = zeros(m,m);
bestB = zeros(m,m);

[bestDelta,bestA,bestB,nrefresh] = rebuild_all_pair_best( ...
    bestDelta,bestA,bestB,members,QX,Q,C,q,qdiag,m);

npair_refresh = npair_refresh + nrefresh;
nfullbuild = nfullbuild + 1;

%% 6. Exact accelerated N2-best iterations
while nswap < opts.local_max_swaps

    [best_delta,lin] = min(bestDelta(:));

    if isempty(best_delta)
        best_delta = Inf;
    end

    % If the cache says no improving swap remains, perform one exact full
    % rebuild before certifying local optimality.
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

    % Globally best swap over all current cluster-pair blocks.
    [r,s] = ind2sub([m,m],lin);
    a = bestA(r,s);
    b = bestB(r,s);

    if a==0 || b==0
        error('Invalid cached N2 swap for a finite improving gain.');
    end

    % Apply a:r->s and b:s->r.
    X(a,r) = 0;
    X(a,s) = 1;
    X(b,s) = 0;
    X(b,r) = 1;

    lab(a) = s;
    lab(b) = r;

    % Update memberships without changing capacities.
    pos_a = find(members{r}==a,1);
    pos_b = find(members{s}==b,1);

    if isempty(pos_a) || isempty(pos_b)
        error('Internal membership inconsistency in fast N2 search.');
    end

    members{r}(pos_a) = b;
    members{s}(pos_b) = a;

    % Exact incremental update:
    % DeltaX = (e_a-e_b)(e_s-e_r)'.
    qu = Q(:,a) - Q(:,b);
    QX(:,s) = QX(:,s) + qu;
    QX(:,r) = QX(:,r) - qu;

    fval = fval + best_delta;
    nswap = nswap + 1;

    if opts.local_verbose_every > 0 && ...
            mod(nswap,opts.local_verbose_every)==0

        fprintf(['[N2-best fast] swaps=%d | f=%.12e | ', ...
                 'last_delta=%.3e | fullbuild=%d | ', ...
                 'pairrefresh=%d | elapsed=%.3fs\n'], ...
                 nswap,fval,best_delta,nfullbuild, ...
                 npair_refresh,toc(t_total));
    end

    % Periodic exact rebuild to eliminate accumulated floating-point drift.
    do_full_refresh = false;

    if isfinite(opts.local_recompute_every) && ...
            opts.local_recompute_every > 0

        do_full_refresh = ...
            mod(nswap,opts.local_recompute_every)==0;
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

    % KEY EXACT UPDATE:
    % A pair block (u,v) with u,v not in {r,s} is unchanged because:
    %   - memberships of u and v are unchanged;
    %   - QX(:,u) and QX(:,v) are unchanged.
    %
    % Therefore only blocks involving r or s need recomputation.

    % All pairs involving r, including (r,s).
    for t = 1:m
        if t==r
            continue;
        end

        u = min(r,t);
        v = max(r,t);

        [d,aa,bb] = pair_best_swap( ...
            u,v,members,QX,Q,C,q,qdiag);

        bestDelta(u,v) = d;
        bestA(u,v) = aa;
        bestB(u,v) = bb;

        npair_refresh = npair_refresh + 1;
    end

    % All pairs involving s, except (r,s), already refreshed above.
    for t = 1:m
        if t==s || t==r
            continue;
        end

        u = min(s,t);
        v = max(s,t);

        [d,aa,bb] = pair_best_swap( ...
            u,v,members,QX,Q,C,q,qdiag);

        bestDelta(u,v) = d;
        bestA(u,v) = aa;
        bestB(u,v) = bb;

        npair_refresh = npair_refresh + 1;
    end
end

%% 7. Final exact objective
fval = model.fun(X);

%% 8. Information
info.nswap = nswap;
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
% Exact QX from current cluster memberships
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
% Exact best swap between one fixed pair of clusters
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

% Q is symmetrized:
% -2*q*(Q_ab+Q_ba) = -4*q*Q_ab.
D = termA + termB' - 4*q*full(Q(A,B));

[best_delta,lin] = min(D(:));
[ia,ib] = ind2sub(size(D),lin);

besta = A(ia);
bestb = B(ib);

end

% ========================================================================
% Exact full rebuild of all cluster-pair best gains
% ========================================================================
function [bestDelta,bestA,bestB,nrefresh] = rebuild_all_pair_best( ...
    bestDelta,bestA,bestB,members,QX,Q,C,q,qdiag,m)

bestDelta(:) = Inf;
bestA(:) = 0;
bestB(:) = 0;

nrefresh = 0;

for r = 1:m-1
    for s = r+1:m

        [d,a,b] = pair_best_swap( ...
            r,s,members,QX,Q,C,q,qdiag);

        bestDelta(r,s) = d;
        bestA(r,s) = a;
        bestB(r,s) = b;

        nrefresh = nrefresh + 1;
    end
end

end
