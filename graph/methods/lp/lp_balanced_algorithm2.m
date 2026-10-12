function result = lp_balanced_algorithm2(model,capacities,opts)
%LP_BALANCED_ALGORITHM2 Balanced-assignment adaptation of Algorithm 2 in
% Jiang, Liu, and Wen (2016), "Lp-norm Regularization Algorithms for
% Optimization Over Permutation Matrices", SIAM J. Optim. 26(4), 2284-2313.
%
% Feasible set:
%   X >= 0,
%   X*1_m = 1_n,
%   X'*1_n = capacities,
% where capacities are positive integers summing to n.
%
% This implementation keeps projection / rounding / local-search modular so
% CP and negProx variants can reuse the same driver.
%
% PROFILING
% result.profile reports mutually interpretable wall-clock components:
%   total_time                    whole Algorithm-2 call
%   projection_time               all balanced-polytope projections
%   regularized_eval_time         F/G evaluations outside line search
%   line_search_time              backtracking loop (trial F evaluations)
%   rounding_time                 greedy balanced rounding only
%   internal_local_search_time    exact fast N2-best local search only
%   objective_only_time       standalone model.fun calls outside N2
%   other_time                    residual unclassified MATLAB overhead
%
% Counts are also returned: projection_calls, projection_inner_iterations,
% line_search_calls/evals, rounding_calls, local_search_calls/swaps/sweeps.
if nargin < 3 || isempty(opts)
    error('opts must be supplied by the project configuration.');
end
wall_tic = tic;
profile = init_profile();
n = model.n;
capacities = round(capacities(:));
m = numel(capacities);
if sum(capacities)~=n
    error('sum(capacities) must equal model.n.');
end
if any(capacities<=0)
    error('All capacities must be positive integers.');
end
if ~(opts.p>0 && opts.p<1)
    error('opts.p must lie in (0,1).');
end
if ~isfield(opts,'strict_paper_checks') || isempty(opts.strict_paper_checks)
    opts.strict_paper_checks = true;
end
% Default feasible interior starting point. For equal capacities b=n/m this
% is X0=(1/m)*ones(n,m), the direct analogue of (1/n)*ones(n,n).
Xk = ones(n,1)*(capacities'/n);
% sigma0 from Eq. (4.10) when curvature information is available.
if ~isempty(opts.sigma0)
    sigma = opts.sigma0;
elseif isfield(model,'curvature_lower') && ~isempty(model.curvature_lower)
    sigma = min(model.curvature_lower/(opts.p*(1-opts.p))*opts.eps0^(2-opts.p), ...
                opts.sigma_minus);
else
    sigma = opts.sigma_minus;
    if opts.verbose
        fprintf('[Lp说明] 未提供 model.curvature_lower，因此令 sigma0=sigma_minus=%.6g。\n',sigma);
    end
end
if sigma > 0
    error('Algorithm 2 practical continuation expects sigma0 <= 0.');
end
eps_reg = opts.eps0;
% Eq. (4.10): sigma_+ = -2^{-l} sigma0, l=ceil(log2(-sigma0)).
if sigma < 0
    ell = ceil(log2(-sigma));
    sigma_plus = -2^(-ell)*sigma;
else
% UNSPECIFIED edge case: the paper allows sigma0<=0 but the displayed
% sigma_+ formula is undefined at sigma0=0.
    sigma_plus = abs(opts.sigma_minus);
    if sigma_plus==0, sigma_plus=1; end
    if opts.verbose
        fprintf('[Lp说明] sigma0=0 会使论文中的 sigma_+ 公式无法定义，因此采用 sigma_plus=%.6g。\n',sigma_plus);
    end
end
% Projection callback.
proj_state = [];
if isempty(opts.projector)
    projector = @default_projector;
else
    projector = opts.projector;
end
% Rounding callback.
if isempty(opts.rounder)
    rounder = @(X) round_balanced_greedy(X,capacities);
else
    rounder = opts.rounder;
end
% Best discrete incumbent.
f_best = inf;
X_best = [];
% Warm-start diagnostic for the later Matrix run.
% Across ALL outer stages and ALL inner iterations, remember the CONTINUOUS
% iterate immediately BEFORE rounding whose rounded/refined discrete candidate
% attains the smallest J_common seen by this LP run. Thus Matrix receives the
% continuous LP point corresponding to the GLOBAL BEST LP J_common candidate,
% not merely the best point from outer stage k=0 and not a rounded 0-1 point.
best_jcommon_pre_round_X = [];
best_jcommon_post_round_score = inf;
best_jcommon_outer_index = NaN;
best_jcommon_inner_index = NaN;
best_jcommon_discovery_elapsed_time = NaN;
history = repmat(struct('k',[],'sigma',[],'eps',[],'inner_iter',[], ...
    'frac_measure',[],'F_cont',[],'f_cont',[],'stage_best',[], ...
    'global_best',[],'proj_res_inf',[],'time_projection',[], ...
    'time_line_search',[],'time_rounding',[],'time_local_search',[]),0,1);
rng(opts.perturb_seed);
for k = 0:opts.max_outer-1
    frac_measure = sum(Xk(:).^opts.p)/n - 1;
    if frac_measure <= opts.tol_outer
        if opts.verbose
            fprintf('[Lp停止] 在 outer stage %d 前停止：fractionality=%.3e <= 阈值 %.3e。\n', ...
                k,frac_measure,opts.tol_outer);
        end
        break;
    end
    stage_profile_before = profile;
% Choose X_k^(0) according to (4.1)-(4.2).
    if k==0
        X = Xk;
    else
% Test whether Xk is approximately stationary for the NEW subproblem.
        tt = tic;
        [~,Gtmp] = lp_regularized_value_grad(model,Xk,sigma,opts.p,eps_reg);
        profile.regularized_eval_time = profile.regularized_eval_time + toc(tt);
        profile.regularized_eval_calls = profile.regularized_eval_calls + 1;
        [Ztest,proj_state,pinfo] = timed_project(Xk-opts.alpha0*Gtmp,proj_state);
        mapnorm = norm(Ztest-Xk,'fro')/sqrt(n);
        if mapnorm <= opts.kkt_map_tol
% UNSPECIFIED in the paper: "a perturbation in D_n" is not defined.
            Xrand = random_balanced_assignment(n,capacities);
            X = (1-opts.perturb_rho)*Xk + opts.perturb_rho*Xrand;
            profile.perturbation_calls = profile.perturbation_calls + 1;
        else
            X = Xk;
        end
    end
    tau_x = max(opts.tau_x0/(k+opts.tau_index_shift)^3,opts.tau_x_min);
    tau_f = max(opts.tau_f0/(k+opts.tau_index_shift)^3,opts.tau_f_min);
    tt = tic;
    [Fcur,Gcur] = lp_regularized_value_grad(model,X,sigma,opts.p,eps_reg);
    profile.regularized_eval_time = profile.regularized_eval_time + toc(tt);
    profile.regularized_eval_calls = profile.regularized_eval_calls + 1;
    Cref = Fcur;
    Qref = 1;
    alpha = opts.alpha0;
    tolx = inf;
    tolf = inf;
    stage_best = inf;
    stage_Xbest = [];
    last_proj_info.res_inf = inf;

    % Exact one-entry cache for the internal N2 refinement.
    % Consecutive continuous iterates often round to exactly the same
    % balanced binary assignment. Re-running the deterministic N2-best
    % local search from the same binary point would reproduce exactly the
    % same result, so we reuse it.
    last_round_X = [];
    last_refined_X = [];
    last_refined_f = [];
    last_refined_info = struct();
    if opts.store_inner_history
        inner_hist = repmat(struct('i',[],'F',[],'f',[],'tolx',[],'tolf',[], ...
            'alpha',[],'proj_res_inf',[],'stage_best',[]),0,1);
    end
    i = 0;
    while (tolx > tau_x || tolf > tau_f) && i < opts.max_inner
% Step 6: projected-gradient direction.
        [Z,proj_state,last_proj_info] = timed_project(X-alpha*Gcur,proj_state);
        D = Z-X;
        gd = sum(Gcur(:).*D(:));
        %%% 临时加的
        Dnorm = norm(D,'fro')/sqrt(n);

        if mod(i,100)==0
            fprintf(['[LP-inner] k=%d i=%d | ', ...
                    'alpha=%.12e | Dnorm=%.6e | gd=%.6e | ', ...
                    'tolx=%.6e | tolf=%.6e | projres=%.3e\n'], ...
                    k,i,alpha,Dnorm,gd,tolx,tolf,last_proj_info.res_inf);
        end
% Step 7: nonmonotone line search, Eq. (4.8).
%
% Algorithm 2 requires the smallest j satisfying (4.8).
% max_backtrack is only a numerical safeguard.  A point which does not
% satisfy (4.8) must NOT be silently accepted.
        profile.line_search_calls = profile.line_search_calls + 1;
        ls_tic = tic;
        j = 0;
        ls_accepted = false;
        while j <= opts.max_backtrack
            t = opts.delta^j;
            Xtrial = X + t*D;
            Ftrial = lp_regularized_value_grad( ...
                model,Xtrial,sigma,opts.p,eps_reg);
            profile.line_search_evals = ...
                profile.line_search_evals + 1;
            if Ftrial <= Cref + opts.theta*t*gd
                ls_accepted = true;
                break;
            end
            j = j + 1;
        end
        profile.line_search_time = ...
            profile.line_search_time + toc(ls_tic);
        profile.backtrack_steps = ...
            profile.backtrack_steps + min(j,opts.max_backtrack);
        if ~ls_accepted
            error('Lp:LineSearchFailed', ...
                ['Eq. (4.8) was not satisfied after reaching ', ...
                'max_backtrack=%d. The trial point is NOT accepted. ', ...
                'Check the projection accuracy or BB step size.'], ...
                opts.max_backtrack);
        end
        %%临时
        if mod(i,100)==0
        fprintf(['[LP-inner-LS] k=%d i=%d | ', ...
                'backtrack_j=%d | step=%.3e | ', ...
                'Ftrial=%.12e | Cref=%.12e\n'], ...
                k,i,j,opts.delta^j,Ftrial,Cref);
        end
%%
        Xnew = Xtrial;
        tt = tic;
        [Fnew,Gnew] = lp_regularized_value_grad(model,Xnew,sigma,opts.p,eps_reg);
        profile.regularized_eval_time = profile.regularized_eval_time + toc(tt);
        profile.regularized_eval_calls = profile.regularized_eval_calls + 1;
        tolx = norm(Xnew-X,'fro')/sqrt(n);
        tolf = abs(Fnew-Fcur)/(1+abs(Fcur));

% Step 9: greedy rounding + optional local 2-neighborhood refinement.
        if opts.do_rounding
            [Xhat,rt] = timed_round(Xnew);
            profile.rounding_time = profile.rounding_time + rt;
            profile.rounding_calls = profile.rounding_calls + 1;
            if opts.do_local_search
                % Reuse only for the default deterministic quadratic N2
                % routine. Custom local-search callbacks are not assumed
                % deterministic and therefore are never cached.
                can_reuse_n2 = isempty(opts.local_search) && ...
                    isfield(model,'kind') && strcmpi(model.kind,'quadratic');

                if can_reuse_n2 && ~isempty(last_round_X) && ...
                        isequal(Xhat,last_round_X)

                    Xhat = last_refined_X;
                    fhat = last_refined_f;
                    linfo = last_refined_info;
                    profile.local_cache_hits = profile.local_cache_hits + 1;

                else
                    Xround = Xhat;
                    [Xhat,fhat,lt,linfo] = timed_local_search(Xhat);

                    profile.internal_local_search_time = ...
                        profile.internal_local_search_time + lt;
                    profile.local_search_calls = profile.local_search_calls + 1;

                    if isfield(linfo,'nswap')
                        profile.local_swaps = profile.local_swaps + linfo.nswap;
                    end
                    if isfield(linfo,'nsweep')
                        profile.local_sweeps = profile.local_sweeps + linfo.nsweep;
                    end

                    if can_reuse_n2
                        last_round_X = Xround;
                        last_refined_X = Xhat;
                        last_refined_f = fhat;
                        last_refined_info = linfo;
                    end
                end
            else
                tt = tic;
                fhat = model.fun(Xhat);
                profile.objective_only_time = profile.objective_only_time + toc(tt);
                profile.objective_only_calls = profile.objective_only_calls + 1;
            end
            if fhat < stage_best
                stage_best = fhat;
                stage_Xbest = Xhat;
            end
% Matrix LP warm start: GLOBAL best J_common over the whole LP run.
% Save the continuous point Xnew BEFORE rounding/local N2, but use
% the resulting feasible discrete candidate's J_common (fhat) to
% decide which continuous point is best.
            if fhat < best_jcommon_post_round_score
                best_jcommon_pre_round_X = Xnew;
                best_jcommon_post_round_score = fhat;
                best_jcommon_outer_index = k;
                best_jcommon_inner_index = i + 1;
                best_jcommon_discovery_elapsed_time = toc(wall_tic);
            end
        end
% BB step for the NEXT projected-gradient iteration.
% The paper says alternating large/short BB steps; exact initial parity
% and safeguards are not stated.
       s = Xnew-X;
       ybb = Gnew-Gcur;

        sty = sum(s(:).*ybb(:));
        yy  = sum(ybb(:).*ybb(:));
        ss  = sum(s(:).*s(:));

        % Alternating BB step.
        % For the nonconvex regularized subproblem, use |s'y|,
        % consistent with the BB treatment used in Wen-Yin [54].
        if yy > 0 && ss > 0 && abs(sty) > eps

            if mod(i+1,2)==1
                % BB1 / large step
                alpha_new = ss / abs(sty);
            else
                % BB2 / short step
                alpha_new = abs(sty) / yy;
            end

            if isfinite(alpha_new) && alpha_new > 0
                alpha = min(max(alpha_new,opts.alpha_min),opts.alpha_max);
            else
                alpha = opts.alpha0;
            end

        else
            alpha = opts.alpha0;
        end
        if mod(i,100)==0
            fprintf(['[BB] k=%d i=%d | sty=%.6e | ss=%.6e | ', ...
                    'yy=%.6e | alpha_new=%.6e\n'], ...
                    k,i,sty,ss,yy,alpha);
        end
% Paper's stated reference update uses F(X_k^(i)), i.e. the OLD point.
        Qnext = opts.eta*Qref + 1;
        Cref  = (opts.eta*Qref*Cref + Fcur)/Qnext;
        Qref  = Qnext;
        X = Xnew;
        Fcur = Fnew;
        Gcur = Gnew;
        i = i+1;
        if opts.store_inner_history
            tt = tic;
            f_hist = model.fun(X);
            profile.objective_only_time = profile.objective_only_time + toc(tt);
            profile.objective_only_calls = profile.objective_only_calls + 1;
            ih.i = i; ih.F = Fcur; ih.f = f_hist;
            ih.tolx = tolx; ih.tolf = tolf; ih.alpha = alpha;
            ih.proj_res_inf = last_proj_info.res_inf; ih.stage_best = stage_best;
            inner_hist(end+1,1) = ih; %#ok<AGROW>
        end
    end
    % Algorithm 2 approximately solves the current regularized subproblem
    % until the stated inner stopping rules are satisfied. max_inner is only
    % a numerical safeguard; do not silently continue the outer continuation
    % from an unconverged inner solve.
    inner_not_converged = (tolx > tau_x || tolf > tau_f);
    if inner_not_converged && i >= opts.max_inner
        msg = sprintf( ...
            ['Lp inner solve hit max_inner=%d before satisfying the ', ...
             'stopping criteria: tolx=%.3e (target %.3e), ', ...
             'tolf=%.3e (target %.3e).'], ...
             opts.max_inner,tolx,tau_x,tolf,tau_f);
        if opts.strict_paper_checks
            error('Lp:InnerNotConverged','%s',msg);
        else
            warning('Lp:InnerNotConverged','%s',msg);
        end
    end
% If no inner iteration produced a rounded solution, round final point once.
    if isempty(stage_Xbest) && opts.do_rounding
        [Xhat,rt] = timed_round(X);
        profile.rounding_time = profile.rounding_time + rt;
        profile.rounding_calls = profile.rounding_calls + 1;
        if opts.do_local_search
            [Xhat,stage_best,lt,linfo] = timed_local_search(Xhat);
            profile.internal_local_search_time = profile.internal_local_search_time + lt;
            profile.local_search_calls = profile.local_search_calls + 1;
            if isfield(linfo,'nswap'), profile.local_swaps = profile.local_swaps + linfo.nswap; end
            if isfield(linfo,'nsweep'), profile.local_sweeps = profile.local_sweeps + linfo.nsweep; end
        else
            tt = tic;
            stage_best = model.fun(Xhat);
            profile.objective_only_time = profile.objective_only_time + toc(tt);
            profile.objective_only_calls = profile.objective_only_calls + 1;
        end
        stage_Xbest = Xhat;
        if stage_best < best_jcommon_post_round_score
            best_jcommon_pre_round_X = X;
            best_jcommon_post_round_score = stage_best;
            best_jcommon_outer_index = k;
            best_jcommon_inner_index = i;
            best_jcommon_discovery_elapsed_time = toc(wall_tic);
        end
    end
% Eq. (4.9): update epsilon based on whether this stage improves incumbent.
    improved = stage_best < f_best - opts.improve_tol;
    if improved
        f_best = stage_best;
        X_best = stage_Xbest;
        eps_next = eps_reg;
    else
        eps_next = max(opts.gamma*eps_reg,opts.eps_min);
    end
    Xk = X;
    frac_measure = sum(Xk(:).^opts.p)/n - 1;
    tt = tic;
    f_cont = model.fun(Xk);
    profile.objective_only_time = profile.objective_only_time + toc(tt);
    profile.objective_only_calls = profile.objective_only_calls + 1;
    h.k = k;
    h.sigma = sigma;
    h.eps = eps_reg;
    h.inner_iter = i;
    h.frac_measure = frac_measure;
    h.F_cont = Fcur;
    h.f_cont = f_cont;
    h.stage_best = stage_best;
    h.global_best = f_best;
    h.proj_res_inf = last_proj_info.res_inf;
    h.time_projection = profile.projection_time-stage_profile_before.projection_time;
    h.time_line_search = profile.line_search_time-stage_profile_before.line_search_time;
    h.time_rounding = profile.rounding_time-stage_profile_before.rounding_time;
    h.time_local_search = profile.internal_local_search_time-stage_profile_before.internal_local_search_time;
    if opts.store_inner_history, h.inner = inner_hist; end
    history(end+1,1)=h; %#ok<AGROW>
    if opts.verbose
        fprintf(['[Lp outer stage] k=%d | sigma=% .3e | eps=%.3e | ', ...
                 '*本stage inner迭代数=%d* | fractionality=%.3e | ', ...
                 '*本stage最好J_common=%.12g* | *当前全局最好J_common=%.12g* | ', ...
                 '投影残差=%.2e\n'], ...
            k,sigma,eps_reg,i,frac_measure,stage_best,f_best,last_proj_info.res_inf);
    end
    if frac_measure <= opts.tol_outer
        break;
    end
% Eq. (4.10): continuation update for sigma.
    if sigma <= opts.sigma_minus
        sigma_tilde = 0.5*sigma;
    elseif sigma < 0
        sigma_tilde = 0;
    elseif sigma == 0
        sigma_tilde = sigma_plus;
    else
        sigma_tilde = 2*sigma;
    end
    sigma = min(sigma_tilde,opts.sigma_max);
    eps_reg = eps_next;
end
% max_outer is only a numerical safeguard.  If the continuation has not
% reached the stated fractionality stopping criterion, report it explicitly.
final_frac_check = sum(Xk(:).^opts.p)/n - 1;
if final_frac_check > opts.tol_outer && numel(history) >= opts.max_outer
    msg = sprintf( ...
        ['Lp continuation hit max_outer=%d with fractionality=%.3e ', ...
         '> tol_outer=%.3e.'], ...
         opts.max_outer,final_frac_check,opts.tol_outer);
    if opts.strict_paper_checks
        error('Lp:OuterNotConverged','%s',msg);
    else
        warning('Lp:OuterNotConverged','%s',msg);
    end
end
% Guarantee a discrete output even if stopping happened before any rounding.
if isempty(X_best)
    [X_best,rt] = timed_round(Xk);
    profile.rounding_time = profile.rounding_time + rt;
    profile.rounding_calls = profile.rounding_calls + 1;
    if opts.do_local_search
        [X_best,f_best,lt,linfo] = timed_local_search(X_best);
        profile.internal_local_search_time = profile.internal_local_search_time + lt;
        profile.local_search_calls = profile.local_search_calls + 1;
        if isfield(linfo,'nswap'), profile.local_swaps = profile.local_swaps + linfo.nswap; end
        if isfield(linfo,'nsweep'), profile.local_sweeps = profile.local_sweeps + linfo.nsweep; end
    else
        tt=tic; f_best=model.fun(X_best);
        profile.objective_only_time = profile.objective_only_time + toc(tt);
        profile.objective_only_calls = profile.objective_only_calls + 1;
    end
end
profile.total_time = toc(wall_tic);
classified = profile.projection_time + profile.regularized_eval_time + ...
    profile.line_search_time + profile.rounding_time + ...
    profile.internal_local_search_time + profile.objective_only_time;
profile.other_time = max(profile.total_time-classified,0);
profile.classified_fraction = classified/max(profile.total_time,eps);
result.X_best = X_best;
result.f_best = f_best;
result.X_cont = Xk;
result.history = history;
result.settings = opts;
result.profile = profile;
result.final_fractionality = sum(Xk(:).^opts.p)/n - 1;
result.final_sigma = sigma;
result.final_epsilon = eps_reg;
% Expose the requested Matrix warm start. This is NOT a rounded point.
% It is the continuous iterate immediately before the rounding/local-search
% call that produced the GLOBAL BEST discrete J_common candidate over all
% outer stages and all inner iterations of this LP run.
result.best_jcommon_pre_round_X = best_jcommon_pre_round_X;
result.best_jcommon_post_round_score = best_jcommon_post_round_score;
result.best_jcommon_outer_index = best_jcommon_outer_index;
result.best_jcommon_inner_index = best_jcommon_inner_index;
result.best_jcommon_discovery_elapsed_time = best_jcommon_discovery_elapsed_time;
if opts.do_rounding && isfinite(best_jcommon_post_round_score) && isfinite(f_best)
    warm_tol = 1e-10*max([1,abs(best_jcommon_post_round_score),abs(f_best)]);
    if abs(best_jcommon_post_round_score-f_best) > warm_tol
        warning(['LP warm-start score and final f_best differ by %.3e. ', ...
            'Matrix will use the continuous point attached to the smaller recorded J_common.'], ...
            abs(best_jcommon_post_round_score-f_best));
    end
end
function [Z,state_out,pinfo] = default_projector(C,state_in)
%DEFAULT_PROJECTOR
% Rectangular balanced-assignment adaptation:
% solve every Euclidean projection accurately with MOSEK.

    [Z,state_out,pinfo] = ...
        project_balanced_mosek( ...
            C,capacities,opts.proj,state_in);

end


function [Z,state_out,pinfo] = timed_project(C,state_in)
%TIMED_PROJECT
% Timed wrapper around the balanced projection solver.

    pt = tic;

    [Z,state_out,pinfo] = ...
        projector(C,state_in);

    profile.projection_time = ...
        profile.projection_time + toc(pt);

    profile.projection_calls = ...
        profile.projection_calls + 1;


    %% Count inner solver iterations when available.

    if isstruct(pinfo) && ...
            isfield(pinfo,'iter') && ...
            isfinite(pinfo.iter)

        profile.projection_inner_iterations = ...
            profile.projection_inner_iterations + ...
            pinfo.iter;
    end


    %% ------------------------------------------------------------
    % Projection acceptance
    %
    % IMPORTANT:
    %
    % The projection routine itself decides whether its returned solution
    % satisfies the solver-specific numerical acceptance criterion.
    %
    % In particular, project_balanced_mosek uses
    %
    %   target_tol = opts.proj.tol
    %
    % with a small floating-point acceptance margin.
    %
    % Therefore do NOT independently impose
    %
    %   pinfo.res_inf <= opts.proj.tol
    %
    % again here, otherwise a valid result such as
    %
    %   target   = 1.000e-8
    %   residual = 1.049e-8
    %
    % would be rejected after the projection solver already accepted it.
    % -------------------------------------------------------------

    proj_ok = ...
        isstruct(pinfo) && ...
        isfield(pinfo,'converged') && ...
        logical(pinfo.converged) && ...
        isfield(pinfo,'res_inf') && ...
        isfinite(pinfo.res_inf);


    if ~proj_ok

        if isstruct(pinfo) && ...
                isfield(pinfo,'res_inf') && ...
                isfinite(pinfo.res_inf)

            pres = pinfo.res_inf;

        else

            pres = NaN;
        end


        msg = sprintf( ...
            ['Balanced projection did not converge according to ', ...
             'the projection solver acceptance rule; ', ...
             'residual=%.3e.'], ...
            pres);


        if opts.strict_paper_checks

            error( ...
                'Lp:ProjectionNotConverged', ...
                '%s',msg);

        else

            warning( ...
                'Lp:ProjectionNotConverged', ...
                '%s',msg);
        end
    end

end
function [Xhat,elapsed] = timed_round(Xin)
        rtic=tic;
        Xhat=rounder(Xin);
        elapsed=toc(rtic);
end
function [Xhat,fhat,elapsed,linfo] = timed_local_search(Xhat)
        ltic=tic;
        linfo=struct();
        if ~isempty(opts.local_search)
            try
                [Xhat,fhat,linfo]=opts.local_search(Xhat);
            catch ME
                if contains(ME.message,'Too many output arguments') || strcmp(ME.identifier,'MATLAB:maxlhs')
                    [Xhat,fhat]=opts.local_search(Xhat);
                else
                    rethrow(ME);
                end
            end
        elseif isfield(model,'kind') && strcmpi(model.kind,'quadratic')
            [Xhat,fhat,linfo]=local2swap_balanced_quadratic(Xhat,model,opts);
        else
            fhat=model.fun(Xhat);
        end
        elapsed=toc(ltic);
end
end
function p = init_profile()
p = struct();
p.total_time = 0;
p.projection_time = 0;
p.regularized_eval_time = 0;
p.line_search_time = 0;
p.rounding_time = 0;
p.internal_local_search_time = 0;
p.objective_only_time = 0;
p.other_time = 0;
p.classified_fraction = 0;
p.projection_calls = 0;
p.projection_inner_iterations = 0;
p.regularized_eval_calls = 0;
p.line_search_calls = 0;
p.line_search_evals = 0;
p.backtrack_steps = 0;
p.rounding_calls = 0;
p.local_search_calls = 0;
p.local_swaps = 0;
p.local_sweeps = 0;
p.local_cache_hits = 0;
p.objective_only_calls = 0;
p.perturbation_calls = 0;
end
