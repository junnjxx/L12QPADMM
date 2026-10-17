function [X,Y,iter,info]=vector_splitadmm(A,b,opt,X1,Y1)
%VECTOR_SPLITADMM Matrix-aligned X-first ADMM for minibatch vector selection.
%   [X,Y,iter,info]=vector_splitadmm(A,b,opt,X1,Y1)
%
%   A: ORIGINAL symmetric minibatch subproblem Z(Q,P), formed by
%      build_mmd_batch_Z; internally scaled by max(1,norm(A,inf)).
%   b: number of vertices to select from the current remaining set.
%   X1,Y1: supplied initial n-by-1 column vectors.
%
%   EXACTLY MATCHED TO CURRENT MATRIX SOLVER (except the affine projection):
%      X update: box-constrained l1/2 half-thresholding, fast_prox option
%      Y update: Euclidean projection onto {Y : ones(n,1)'*Y=b}
%      beta: Matrix's adaptive/fixed dual stepsize formula
%      eta: Matrix's initial stage, stagnation and growth; minibatch cap
%      residual, early_exist, clean_fast and STOPPING-DISABLED behavior
%
%   IMPORTANT: the native binary candidate is X (NOT Y), since the
%   Matrix-aligned half-thresholding is now on X.
%   The outer graph wrapper must extract selected indices from X.
%
%   No intermediate history is accumulated or saved.

    n=size(A,1);
    assert(size(A,2)==n,'Vector: input A must be square.');
    assert(isscalar(b) && b==fix(b) && b>=1 && b<=n, ...
        'Vector: group size b must be an integer in [1,n].');
    assert(isequal(size(X1),[n,1]) && isequal(size(Y1),[n,1]), ...
        'Vector: X1 and Y1 must both be n-by-1 column vectors.');
    assert(all(isfinite(X1)) && all(isfinite(Y1)), ...
        'Vector: initialization contains nonfinite values.');

    if ~isfield(opt,'clean_fast') || isempty(opt.clean_fast)
        opt.clean_fast=false;
    end
    clean_fast=logical(opt.clean_fast);

    objective_scale=max(1,norm(A,inf));
    A=A/objective_scale;
    G=zeros(n,1); % Minibatch Z absorbs the stage-dependent linear term
    N=n;
    X=X1;
    Y=Y1;

    c=opt.c;
    tau=opt.tau_X;
    cp=c+tau;
    if ~isscalar(c)||c<=0||~isscalar(tau)||tau<=0
        error('Vector: require c and tau_X positive.');
    end
    if isempty(opt.eta_max)
        eta_max=4*norm(A,2)*(1+1e-6); % Match minibatch Matrix cap rule, for scaled Z
    else
        eta_max=opt.eta_max;
    end
    if ~clean_fast
        fprintf('Vector eta_max=%.6g\n',eta_max);
    end
    if opt.eta_adapt_enabled
        eta_state=opt.eta_stage_initial;
    else
        eta_state=opt.eta;
    end

    Lambda=zeros(N,1);
    AY=A*Y;
    p0=max(norm(Y-X,'fro'),eps);
    monitor=opt.eta_adapt_enabled && nnz(X)>b && eta_state<eta_max;
    best_nnz=nnz(X);
    stale=0;
    flag=0;
    r=NaN;
    dxy=NaN;
    beta=opt.beta0;
    t0=tic;

    for iter=1:opt.max_iter
        Xold=X;
        Yold=Y;
        Lold=Lambda;
        AYold=AY;

        % X-first half-thresholding: IDENTICAL to Matrix, column count=1.
        R=(c*Yold+Lold-0.5*AYold+tau*Xold)/cp;
        e=max(eta_state,0);
        if e==0
            X=min(max(R,0),1);
        else
            X=zeros(N,1);
            lambda=2*e/cp;
            if opt.fast_prox
                threshold=(3/2^(5/3))*lambda^(2/3);
                active=R>=threshold;
                v=R(active);
                arg=(3*sqrt(3)*lambda/8)./(v.*sqrt(v));
                arg=min(max(arg,-1),1);
                x=(2/3).*v.*(1+cos(2*pi/3-(2/3)*acos(arg)));
            else
                active=R>(3/4)*lambda^(2/3);
                v=R(active);
                arg=(lambda/8).*(v/3).^(-1.5);
                arg=min(max(arg,-1),1);
                x=(2/3).*v.*(1+cos(2*pi/3-(2/3)*acos(arg)));
                lose=(x-v).^2+lambda*sqrt(max(x,0))-v.^2>0;
                x(lose)=0;
            end
            X(active)=min(x,1);
        end

        % ONLY STRUCTURAL DIFFERENCE FROM MATRIX:
        % Matrix projects onto row sums=1 and column sums=N/B;
        % Vector projects onto the one affine constraint sum(Y)=b.
        AX=A*X;
        H=X-(Lold+0.5*AX+G)/c;
        Y=H+(b-sum(H))/N;

        dxy=norm(Y-X,'fro');
        if strcmpi(opt.beta_mode,'fixed')
            beta=opt.beta0;
        elseif strcmpi(opt.beta_mode,'adaptive')
            beta=opt.beta0*min(1,opt.kabeta*p0/ ...
                (max(iter,1)*log(iter+1)^2*max(dxy,eps)));
        else
            error('Vector: unknown beta_mode.');
        end
        Lambda=Lold+beta*(Y-X);
        AY=A*Y;

        needcheck=mod(iter,opt.stop_check_interval)==0 || iter==opt.max_iter;
        early=opt.early_exist && mod(iter,opt.early_exist_interval)==0;
        if needcheck || early
            dL=Lambda-Lold;
            xstat=dL+0.5*(AYold-AY)+c*(Y-Yold)+tau*(X-Xold);
            r=max(norm(xstat,'fro'),norm(dL,'fro'));
        end
        if ~clean_fast && mod(iter,opt.print_every)==0
            frac=max(abs(X(:)-round(X(:))));
            fprintf(['Vector iter=%d | beta=%.6g | eta_state=%.6g | ', ...
                     'eta_eff=%.6g | nnzX=%d | r=%.3e | dxy=%.3e | fractionality=%.3e\n'], ...
                iter,beta,eta_state,e,nnz(X),r,dxy,frac);
        end

        if early && dxy<=opt.early_exist_dxy_tol
            Ptest=round(X);
            if max(abs(X(:)-Ptest(:)))<opt.early_exist_round_tol && ...
                    is_valid_cardinality(Ptest,N,b)
                X=Ptest;
                flag=2;
                break;
            end
        end

        % EXACTLY as CURRENT Matrix: residual-based stopping is DISABLED.
        % To re-enable in BOTH solvers, uncomment the analogous block.
        % if needcheck && max(r,dxy)<opt.tol
        %     flag=1;break;
        % end

        % IDENTICAL stagnation logic, with Matrix's target N replaced by b.
        if opt.eta_adapt_enabled && eta_state<eta_max
            nz=nnz(X);
            if nz<=b && ~any(X(:)>opt.support_tol & X(:)<1-opt.support_tol)
                monitor=false;best_nnz=nz;stale=0;
            elseif ~monitor
                monitor=true;best_nnz=nz;stale=0;
            elseif nz<best_nnz
                best_nnz=nz;stale=0;
            else
                stale=stale+1;
            end
            if monitor && stale>=opt.eta_stagnation_iters
                if eta_state<0
                    if eta_state<=opt.eta_stage_initial
                        eta_state=0.5*eta_state;
                    else
                        eta_state=0;
                    end
                elseif eta_state==0
                    eta_state=opt.eta;
                else
                    eta_state=min(opt.eta_growth_factor*eta_state,eta_max);
                end
                stale=0;best_nnz=nz;
            end
        end
    end

    solve_time=toc(t0);
    Pnative=round(X);
    native_valid=max(abs(X(:)-Pnative(:)))<=opt.support_tol && ...
        is_valid_cardinality(Pnative,N,b);
    info=struct('objective_scale',objective_scale,'eta_max',eta_max, ...
        'eta_final',eta_state,'beta_final',beta,'iterations',iter, ...
        'flag',flag,'residual_r',r,'residual_dxy',dxy, ...
        'nnzX',nnz(X),'binary_deviation',max(abs(X(:)-round(X(:)))), ...
        'native_valid',native_valid,'solve_time',solve_time);
end

function ok=is_valid_cardinality(P,n,b)
    ok=isequal(size(P),[n,1]) && all(P(:)==0 | P(:)==1) && ...
        sum(P)==b;
end
