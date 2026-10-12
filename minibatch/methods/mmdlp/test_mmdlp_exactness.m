function test_mmdlp_exactness()

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('MMD-LP exactness audit\n');
    fprintf('============================================================\n');

    rng(20261006,'twister');

    q = 12;
    k = 4;
    ntrial = 20;
    tol = 1e-8;

    fprintf('q=%d | k=%d | trials=%d\n',q,k,ntrial);

    for trial = 1:ntrial

        R = randn(q);
        Z = 0.5*(R+R');
        Z = Z - median(Z(:));

        opts = struct();
        opts.display = 'none';
        opts.feasibility_tol = 1e-8;

        [m_relaxed,lpinfo] = ...
            mmdlp_solve_relaxation(Z,k,opts);

        f_lp = lpinfo.fval;

        sort_matrix = [-m_relaxed,(1:q)'];
        [~,order] = sortrows(sort_matrix,[1 2]);

        pick = order(1:k);

        m_round = zeros(q,1);
        m_round(pick) = 1;

        f_round = m_round' * Z * m_round;

        C = nchoosek(1:q,k);

        f_exact = inf;
        m_exact = [];

        for r = 1:size(C,1)

            m = zeros(q,1);
            m(C(r,:)) = 1;

            f = m' * Z * m;

            if f < f_exact
                f_exact = f;
                m_exact = m;
            end
        end

        scale = max([1,abs(f_lp),abs(f_exact),abs(f_round)]);
        this_tol = tol*scale;

        if f_lp > f_exact + this_tol
            error('MMDLP:LowerBoundViolation', ...
                ['Trial %d failed: LP objective %.12e > ', ...
                 'exact IQP optimum %.12e.'], ...
                trial,f_lp,f_exact);
        end

        if f_exact > f_round + this_tol
            error('MMDLP:RoundingViolation', ...
                ['Trial %d failed: exact IQP optimum %.12e > ', ...
                 'rounded objective %.12e.'], ...
                trial,f_exact,f_round);
        end

        W_exact = m_exact*m_exact';
        f_lifted = sum(Z.*W_exact,'all');

        if abs(f_lifted-f_exact) > this_tol
            error('MMDLP:LiftedObjectiveMismatch', ...
                ['Trial %d: lifted objective %.12e differs ', ...
                 'from m''Zm %.12e.'], ...
                trial,f_lifted,f_exact);
        end

        fprintf([ ...
            'trial %2d/%2d | ', ...
            'LP=% .6e | exact=% .6e | rounded=% .6e | ', ...
            'gap1=%.3e | gap2=%.3e\n'], ...
            trial,ntrial, ...
            f_lp,f_exact,f_round, ...
            f_exact-f_lp, ...
            f_round-f_exact);
    end

    fprintf('\nPASS: all MMD-LP exactness checks passed.\n');
    fprintf('============================================================\n');

end