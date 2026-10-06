function [X,state,info] = project_balanced_mosek(C,capacities,opts,state)
%PROJECT_BALANCED_MOSEK
%
% Euclidean projection onto the rectangular balanced-assignment polytope:
%
%       min_X  0.5 * ||X - C||_F^2
%
%       s.t.
%               X * 1_m   = 1_n,
%               X' * 1_n  = capacities,
%               X >= 0.
%
% Equivalently, with x = X(:),
%
%       min_x  0.5*x'*x - C(:)'*x
%
% subject to the balanced-assignment constraints.
%
%
% Numerical implementation:
%
% 1. Remove one redundant equality constraint.
%
%    Since
%
%       sum_i (row sums) = n
%
%    and
%
%       sum_j capacities_j = n,
%
%    one column-sum constraint is implied by all row-sum constraints
%    and the other m-1 column-sum constraints.
%
%
% 2. Center C along equality-normal directions.
%
%    Adding/subtracting a matrix of the form
%
%       u*1' + 1*v'
%
%    changes <C,X> only by a constant over the feasible set.
%    Therefore the Euclidean projection is unchanged.
%
%
% 3. Do NOT clip tiny negative MOSEK entries with
%
%       X = max(X,0).
%
%    Such clipping can accumulate over many entries and noticeably
%    damage row/column feasibility.
%
%
% 4. MOSEK may terminate with
%
%       MSK_RES_TRM_STALL
%
%    and return solsta = UNKNOWN even though the numerical iterate is
%    already sufficiently accurate for our projection subproblem.
%
%    Therefore:
%
%       OPTIMAL:
%           accept if primal feasibility is accurate enough.
%
%       UNKNOWN + MSK_RES_TRM_STALL:
%           accept only if BOTH
%
%               primal feasibility <= projection tolerance
%
%           and
%
%               relative primal-dual gap <= gap tolerance.
%
%
% INPUT
% -----
% C
%     n-by-m matrix.
%
% capacities
%     m-by-1 vector, positive entries, sum(capacities)=n.
%
% opts.tol
%     Required projection feasibility tolerance.
%
% opts.mosek_toolbox_path
%     Path to MOSEK MATLAB Optimization Toolbox.
%
% opts.mosek_license_file
%     MOSEK license file.
%
% opts.mosek_verbose
%     true / false.
%
% opts.gap_tol
%     Optional tolerance for accepting UNKNOWN+STALL solutions.
%     Default = 1e-6.
%
%
% OUTPUT
% ------
% X
%     Projected matrix.
%
% state
%     [].
%
% info
%     Solver diagnostics.

    %#ok<INUSD>


    %% ============================================================
    % 0. Basic checks
    % =============================================================

    [n,m] = size(C);

    capacities = capacities(:);


    if numel(capacities) ~= m
        error( ...
            'Lp:MosekProjectionCapacityDimension', ...
            'capacities must have length size(C,2).');
    end


    if any(capacities <= 0)
        error( ...
            'Lp:MosekProjectionCapacityPositive', ...
            'All capacities must be positive.');
    end


    if abs(sum(capacities)-n) > 1e-10*max(1,n)
        error( ...
            'Lp:MosekProjectionCapacitySum', ...
            'sum(capacities) must equal n.');
    end


    if ~isfield(opts,'tol') || isempty(opts.tol)
        error( ...
            'Lp:MosekProjectionMissingTol', ...
            'opts.tol must be supplied.');
    end


    target_tol = double(opts.tol);


    if ~(isfinite(target_tol) && target_tol > 0)
        error( ...
            'Lp:MosekProjectionInvalidTol', ...
            'opts.tol must be positive and finite.');
    end


    % Gap tolerance used only when MOSEK returns UNKNOWN+STALL.
    if isfield(opts,'gap_tol') && ~isempty(opts.gap_tol)

        gap_tol = double(opts.gap_tol);

    else

        gap_tol = 1e-6;

    end


    if ~(isfinite(gap_tol) && gap_tol > 0)
        error( ...
            'Lp:MosekProjectionInvalidGapTol', ...
            'opts.gap_tol must be positive and finite.');
    end


    %% ============================================================
    % 1. Configure MOSEK
    % =============================================================

    persistent mosek_initialized

    if isempty(mosek_initialized) || ~mosek_initialized


        % ----------------------------------------------------------
        % MOSEK MATLAB toolbox
        % ----------------------------------------------------------

        if isfield(opts,'mosek_toolbox_path') && ...
                ~isempty(opts.mosek_toolbox_path)

            mosek_path = ...
                char(string(opts.mosek_toolbox_path));

            if ~exist(mosek_path,'dir')

                error( ...
                    'Lp:MosekProjectionToolboxPath', ...
                    'MOSEK toolbox path does not exist: %s', ...
                    mosek_path);

            end

            addpath(mosek_path);

        end


        % ----------------------------------------------------------
        % MOSEK license
        % ----------------------------------------------------------

        if isfield(opts,'mosek_license_file') && ...
                ~isempty(opts.mosek_license_file)

            licfile = ...
                char(string(opts.mosek_license_file));

            if ~exist(licfile,'file')

                error( ...
                    'Lp:MosekProjectionLicense', ...
                    'MOSEK license file does not exist: %s', ...
                    licfile);

            end

            setenv('MOSEKLM_LICENSE_FILE',licfile);

        end


        % ----------------------------------------------------------
        % Check mosekopt
        % ----------------------------------------------------------

        if isempty(which('mosekopt'))

            error( ...
                'Lp:MosekProjectionMissingMosek', ...
                ['mosekopt not found. Check ', ...
                 'cfg.lp.proj.mosek_toolbox_path.']);

        end


        mosek_initialized = true;

    end


    %% ============================================================
    % 2. Equality-normal centering
    %
    % Cwork differs from C by row/column potentials only.
    %
    % Therefore
    %
    %       argmin_{X feasible} ||X-Cwork||_F^2
    %
    % equals
    %
    %       argmin_{X feasible} ||X-C||_F^2.
    % =============================================================

    row_mean = mean(C,2);

    col_mean = mean(C,1);

    grand_mean = mean(C(:));


    Cwork = ...
        C ...
        - row_mean ...
        - col_mean ...
        + grand_mean;


    %% ============================================================
    % 3. Build independent equality constraints
    %
    % MATLAB uses column-major vectorization:
    %
    %       x = X(:).
    %
    % Keep:
    %
    %       all n row constraints,
    %       first m-1 column constraints.
    %
    % The last column constraint follows automatically.
    % =============================================================

    nv = n*m;


    % --------------------------------------------------------------
    % Row sums:
    %
    %       X * 1_m = 1_n
    %
    % For x = X(:):
    %
    %       [I I ... I] x = 1_n
    % --------------------------------------------------------------

    Arow = ...
        kron( ...
            sparse(ones(1,m)), ...
            speye(n) ...
        );


    % --------------------------------------------------------------
    % Column sums
    % --------------------------------------------------------------

    if m > 1

        Acol_full = ...
            kron( ...
                speye(m), ...
                sparse(ones(1,n)) ...
            );


        % Drop final redundant column constraint.
        Acol = ...
            Acol_full(1:m-1,:);


        Aeq = ...
            [ ...
                Arow; ...
                Acol ...
            ];


        beq = ...
            [ ...
                ones(n,1); ...
                capacities(1:m-1) ...
            ];

    else

        Aeq = Arow;

        beq = ones(n,1);

    end


    %% ============================================================
    % 4. Build MOSEK QP
    %
    %       min_x 0.5*x'*x - Cwork(:)'*x
    %
    %       s.t.
    %               Aeq*x = beq
    %               x >= 0.
    % =============================================================

    prob = struct();


    % Linear objective term.
    prob.c = -Cwork(:);


    % Equality constraints.
    prob.a   = Aeq;
    prob.blc = beq;
    prob.buc = beq;


    % Nonnegative variables.
    prob.blx = zeros(nv,1);
    prob.bux = inf(nv,1);


    % Quadratic term:
    %
    %       Q = I
    %
    idx = (1:nv)';

    prob.qosubi = idx;
    prob.qosubj = idx;
    prob.qoval  = ones(nv,1);


    %% ============================================================
    % 5. MOSEK tolerances
    %
    % Even when the external projection acceptance tolerance is e.g.
    %
    %       1e-6,
    %
    % ask MOSEK internally for substantially higher accuracy.
    %
    % MOSEK 11 may reformulate the QO into a conic problem, therefore
    % set both QO and CO tolerances.
    % =============================================================

    mosek_tol = ...
        min(1e-10,0.1*target_tol);


    param = struct();


    % --------------------------------------------------------------
    % QO tolerances
    % --------------------------------------------------------------

    param.MSK_DPAR_INTPNT_QO_TOL_PFEAS = ...
        mosek_tol;

    param.MSK_DPAR_INTPNT_QO_TOL_DFEAS = ...
        mosek_tol;

    param.MSK_DPAR_INTPNT_QO_TOL_REL_GAP = ...
        mosek_tol;


    % --------------------------------------------------------------
    % Conic tolerances
    % --------------------------------------------------------------

    param.MSK_DPAR_INTPNT_CO_TOL_PFEAS = ...
        mosek_tol;

    param.MSK_DPAR_INTPNT_CO_TOL_DFEAS = ...
        mosek_tol;

    param.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = ...
        mosek_tol;


    %% ============================================================
    % 6. Solve
    % =============================================================

    if isfield(opts,'mosek_verbose') && ...
            opts.mosek_verbose

        cmd = 'minimize';

    else

        cmd = 'minimize echo(0)';

    end


    solve_tic = tic;


    [rcode,res] = ...
        mosekopt(cmd,prob,param);


    solve_time = ...
        toc(solve_tic);


    %% ============================================================
    % 7. Parse MOSEK termination information
    % =============================================================

    rcodestr = '';

    if isfield(res,'rcodestr') && ...
            ~isempty(res.rcodestr)

        rcodestr = ...
            char(string(res.rcodestr));

    end


    rmsg = '';

    if isfield(res,'rmsg') && ...
            ~isempty(res.rmsg)

        rmsg = ...
            char(string(res.rmsg));

    end


    % --------------------------------------------------------------
    % Genuine MOSEK errors are fatal.
    %
    % Termination codes such as MSK_RES_TRM_STALL are NOT treated as
    % fatal here.  Numerical quality is checked below.
    % --------------------------------------------------------------

    if ~isempty(rcodestr) && ...
            contains(rcodestr,'MSK_RES_ERR')

        error( ...
            'Lp:MosekProjectionFailed', ...
            'MOSEK error: %s (%d), %s', ...
            rcodestr, ...
            rcode, ...
            rmsg);

    end


    %% ============================================================
    % 8. Check returned interior-point solution
    % =============================================================

    if ~isfield(res,'sol') || ...
       ~isfield(res.sol,'itr') || ...
       ~isfield(res.sol.itr,'xx')

        error( ...
            'Lp:MosekProjectionNoSolution', ...
            ['MOSEK did not return an interior-point solution. ', ...
             'rcode=%d, rcodestr=%s, message=%s'], ...
            rcode, ...
            rcodestr, ...
            rmsg);

    end


    solsta = '';

    if isfield(res.sol.itr,'solsta')

        solsta = ...
            char(string(res.sol.itr.solsta));

    end


    %% ============================================================
    % 9. Classify MOSEK solution status
    %
    % Acceptable candidates:
    %
    %   A) OPTIMAL
    %
    %   B) UNKNOWN + MSK_RES_TRM_STALL
    %
    % Case B is only a candidate.  It must still pass independent
    % primal-feasibility and primal-dual-gap checks below.
    % =============================================================

    is_optimal = ...
        strcmpi(solsta,'OPTIMAL');


    is_stall_unknown = ...
        strcmpi(solsta,'UNKNOWN') && ...
        strcmpi(rcodestr,'MSK_RES_TRM_STALL');


    if ~(is_optimal || is_stall_unknown)

        error( ...
            'Lp:MosekProjectionStatus', ...
            ['MOSEK projection returned an unacceptable status. ', ...
             'solsta=%s, rcode=%d, rcodestr=%s'], ...
            solsta, ...
            rcode, ...
            rcodestr);

    end


    %% ============================================================
    % 10. Recover X
    %
    % IMPORTANT:
    %
    % Do NOT apply
    %
    %       X = max(X,0).
    %
    % Tiny negative entries are kept and explicitly measured below.
    % =============================================================

    x = ...
        res.sol.itr.xx;


    if numel(x) ~= nv

        error( ...
            'Lp:MosekProjectionDimension', ...
            'Unexpected MOSEK solution dimension.');

    end


    X = ...
        reshape(x,n,m);


    %% ============================================================
    % 11. Independent primal-feasibility check
    %
    % Check ORIGINAL balanced constraints, including the column
    % constraint omitted from Aeq.
    % =============================================================

    grow = ...
        sum(X,2) - 1;


    gcol = ...
        sum(X,1)' - capacities;


    row_res_inf = ...
        norm(grow,inf);


    col_res_inf = ...
        norm(gcol,inf);


    neg_res_inf = ...
        max(0,-min(X(:)));


    res_inf = ...
        max( ...
            [ ...
                row_res_inf, ...
                col_res_inf, ...
                neg_res_inf ...
            ] ...
        );


    %% ============================================================
    % 12. Extract primal / dual objective values
    %
    % Used to assess UNKNOWN+STALL solutions.
    % =============================================================

    pobj = NaN;

    dobj = NaN;

    rel_gap = Inf;


    if isfield(res.sol.itr,'pobjval') && ...
            ~isempty(res.sol.itr.pobjval)

        pobj = ...
            double(res.sol.itr.pobjval);

    end


    if isfield(res.sol.itr,'dobjval') && ...
            ~isempty(res.sol.itr.dobjval)

        dobj = ...
            double(res.sol.itr.dobjval);

    end


    if isfinite(pobj) && ...
            isfinite(dobj)

        rel_gap = ...
            abs(pobj-dobj) / ...
            max( ...
                [ ...
                    1, ...
                    abs(pobj), ...
                    abs(dobj) ...
                ] ...
            );

    end


    %% ============================================================
    % 13. Numerical acceptance
    %
    % OPTIMAL:
    %
    %       require primal feasibility <= target_tol.
    %
    % UNKNOWN + STALL:
    %
    %       require
    %
    %           primal feasibility <= target_tol
    %
    %       AND
    %
    %           relative primal-dual gap <= gap_tol.
    % =============================================================

    feas_ok = ...
        isfinite(res_inf) && ...
        res_inf <= target_tol;


    gap_ok = ...
        isfinite(rel_gap) && ...
        rel_gap <= gap_tol;


    if is_optimal

        converged = ...
            feas_ok;

    else

        converged = ...
            is_stall_unknown && ...
            feas_ok && ...
            gap_ok;

    end


    %% ============================================================
    % 14. Reject numerically inadequate candidate
    % =============================================================

    if ~converged

        error( ...
            'Lp:MosekProjectionInaccurate', ...
            [ ...
             'MOSEK projection unacceptable: ', ...
             'solsta=%s, rcodestr=%s, ', ...
             'residual=%.3e (tol %.3e), ', ...
             'relative gap=%.3e (gap tol %.3e), ', ...
             'row=%.3e, col=%.3e, neg=%.3e.' ...
            ], ...
            solsta, ...
            rcodestr, ...
            res_inf, ...
            target_tol, ...
            rel_gap, ...
            gap_tol, ...
            row_res_inf, ...
            col_res_inf, ...
            neg_res_inf);

    end


    %% ============================================================
    % 15. Obtain MOSEK iteration count if available
    % =============================================================

    mosek_iter = NaN;


    if isfield(res,'info') && ...
            isfield(res.info,'MSK_IINF_INTPNT_ITER')

        mosek_iter = ...
            double( ...
                res.info.MSK_IINF_INTPNT_ITER ...
            );

    end


    %% ============================================================
    % 16. Diagnostics
    % =============================================================

    info = struct();


    info.solver = ...
        'mosek';


    info.iter = ...
        mosek_iter;


    info.converged = ...
        converged;


    info.res_inf = ...
        res_inf;


    info.row_res_inf = ...
        row_res_inf;


    info.col_res_inf = ...
        col_res_inf;


    info.neg_res_inf = ...
        neg_res_inf;


    % Compatibility with previous projection interface.
    info.alpha = NaN;


    info.solsta = ...
        solsta;


    info.solve_time = ...
        solve_time;


    info.max_abs_input = ...
        max(abs(C(:)));


    info.max_abs_centered_input = ...
        max(abs(Cwork(:)));


    info.target_tol = ...
        target_tol;


    info.gap_tol = ...
        gap_tol;


    info.pobj = ...
        pobj;


    info.dobj = ...
        dobj;


    info.rel_gap = ...
        rel_gap;


    info.is_optimal = ...
        is_optimal;


    info.is_stall_unknown = ...
        is_stall_unknown;


    info.rcode = ...
        rcode;


    info.rcodestr = ...
        rcodestr;


    info.rmsg = ...
        rmsg;


    %% ============================================================
    % 17. No warm-start state
    % =============================================================

    state = [];


    %% ============================================================
    % 18. Diagnostics printing
    % =============================================================

    if isfield(opts,'mosek_verbose') && ...
            opts.mosek_verbose


        % ----------------------------------------------------------
        % Special termination message
        % ----------------------------------------------------------

        if rcode ~= 0 || ...
                ~is_optimal

            fprintf( ...
                [ ...
                 '[MOSEK termination] ', ...
                 '%s (%d) | ', ...
                 'solsta=%s | ', ...
                 'residual=%.3e | ', ...
                 'relgap=%.3e | ', ...
                 'accepted=%d\n' ...
                ], ...
                info.rcodestr, ...
                info.rcode, ...
                info.solsta, ...
                info.res_inf, ...
                info.rel_gap, ...
                info.converged);

        end


        % ----------------------------------------------------------
        % Projection summary
        % ----------------------------------------------------------

        fprintf( ...
            [ ...
             '[MOSEK projection] ', ...
             'n=%d | m=%d | ', ...
             'max|C|=%.3e | ', ...
             'max|Ccenter|=%.3e | ', ...
             'iter=%g | ', ...
             'rowres=%.3e | ', ...
             'colres=%.3e | ', ...
             'negres=%.3e | ', ...
             'residual=%.3e | ', ...
             'relgap=%.3e | ', ...
             'time=%.3f s\n' ...
            ], ...
            n, ...
            m, ...
            info.max_abs_input, ...
            info.max_abs_centered_input, ...
            info.iter, ...
            info.row_res_inf, ...
            info.col_res_inf, ...
            info.neg_res_inf, ...
            info.res_inf, ...
            info.rel_gap, ...
            info.solve_time);

    end

end