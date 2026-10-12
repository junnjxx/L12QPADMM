function out = mmdlp_bruteforce_iqp(Z,k)
%MMDLP_BRUTEFORCE_IQP
% Exactly solve
%
%       min m' Z m
%
%       s.t. m in {0,1}^q,
%            sum(m)=k
%
% by exhaustive enumeration.
%
% Intended ONLY for small q as a correctness diagnostic.

    q = size(Z,1);

    if size(Z,2) ~= q
        error('Z must be square.');
    end

    if k < 1 || k > q || k ~= round(k)
        error('k must be an integer in [1,q].');
    end

    C = nchoosek(1:q,k);

    ncomb = size(C,1);

    best_obj = inf;
    best_m = [];
    best_pick = [];

    t0 = tic;

    for r = 1:ncomb

        pick = C(r,:);

        m = zeros(q,1);
        m(pick) = 1;

        obj = m' * Z * m;

        if obj < best_obj
            best_obj = obj;
            best_m = m;
            best_pick = pick;
        end
    end

    elapsed = toc(t0);

    out = struct();

    out.q = q;
    out.k = k;
    out.num_combinations = ncomb;

    out.best_objective = best_obj;
    out.best_m = best_m;
    out.best_pick = best_pick(:);

    out.solve_time = elapsed;
end