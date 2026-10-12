function out=solve_matrix_graph(problem,opt)
%SOLVE_MATRIX_GRAPH Pure-MATLAB paper-aligned X/Y split ADMM for graph cut.
% Equations and half-thresholding mirror splitadmm20240119.m in L12QPADMM.
% Graph-specific A=L/max(1,||L||inf), G=0; report raw graph cut.
matrix_objective_scale=max(1,norm(problem.L,inf));
A=problem.L/matrix_objective_scale;
G=zeros(problem.n,problem.k);[N,B]=size(G);
rng(opt.seed,'twister');
if strcmpi(opt.init_mode,'random')
    X=rand(N,B);Y=rand(N,B);
elseif strcmpi(opt.init_mode,'provided')
    X=opt.X0;Y=opt.Y0;
    assert(isequal(size(X),[N,B]) && isequal(size(Y),[N,B]),'Provided Matrix init size mismatch');
elseif strcmpi(opt.init_mode,'uniform')
    X=ones(N,B)/B+(rand(N,B)-0.5)/(B*100);
    Y=ones(N,B)/B+(rand(N,B)-0.5)/(B*100);
else
    error('Unknown Matrix init_mode');
end
c=opt.c;tau=opt.tau_X;cp=c+tau;
if ~isscalar(c)||c<=0||tau<=0,error('Require c,tau_X positive');end
if isempty(opt.eta_max),eta_max=4*graph_spectral_norm(A)*(1.5);
else,eta_max=opt.eta_max;end
fprintf('eta_max=%.6g',eta_max);
if opt.eta_adapt_enabled,eta_state=opt.eta_stage_initial;
else,eta_state=opt.eta;end
Lambda=zeros(N,B);AY=A*Y;
p0=max(norm(Y-X,'fro'),eps);
monitor=opt.eta_adapt_enabled && nnz(X)>N && eta_state<eta_max;
best_nnz=nnz(X);stale=0;
flag=0;r=NaN;dxy=NaN;beta=opt.beta0;
t0=tic;
for iter=1:opt.max_iter
    Xold=X;Yold=Y;Lold=Lambda;AYold=AY;
    R=(c*Yold+Lold-0.5*AYold+tau*Xold)/cp;
    e=max(eta_state,0);
    if e==0
        X=min(max(R,0),1);
    else
        X=zeros(N,B);
        lambda=2*e/cp;
        % Original half-thresholding, with optional fast branch.
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
    AX=A*X;
    H=X-(Lold+0.5*AX+G)/c;
    es=sum(H,1);
    correction=(1+sum(es)/N-sum(H,2))/B;
    Y=H-es/N+correction;
    dxy=norm(Y-X,'fro');
    if strcmpi(opt.beta_mode,'fixed')
        beta=opt.beta0;
    elseif strcmpi(opt.beta_mode,'adaptive')
        beta=opt.beta0*min(1,opt.kabeta*p0/ ...
            (max(iter,1)*log(iter+1)^2*max(dxy,eps)));
    else
        error('Unknown beta mode');
    end
    Lambda=Lold+beta*(Y-X);
    AY=A*Y;
    needcheck=mod(iter,opt.stop_check_interval)==0||iter==opt.max_iter;
    early=opt.early_exist && mod(iter,opt.early_exist_interval)==0;
    if needcheck || early
        dL=Lambda-Lold;
        xstat=dL+0.5*(AYold-AY)+c*(Y-Yold)+tau*(X-Xold);
        r=max(norm(xstat,'fro'),norm(dL,'fro'));
    end
    if mod(iter,opt.print_every)==0
        fprintf('Matrix iter=%d | beta=%.6g | eta_state=%.6g | eta_eff=%.6g | nnzX=%d | r=%.3e | dxy=%.3e\n', ...
            iter,beta,eta_state,e,nnz(X),r,dxy);
    end
    if early && dxy<=opt.early_exist_dxy_tol
        Ptest=round(X);
        if max(abs(X(:)-Ptest(:)))<opt.early_exist_round_tol && ...
           is_valid_partition(Ptest,problem)
            X=Ptest;flag=2;break;
        end
    end
    % Original stopping rule, including its known continuation-stage behavior.
    %if needcheck && max(r,dxy)<opt.tol
    %    flag=1;break;
    %end
    frac=max(abs(X(:)-round(X(:))));
    fprintf('fractionality=%.3e\n',frac);
    if opt.eta_adapt_enabled && eta_state<eta_max
        nz=nnz(X);
        if nz<=N && ~any(X(:)>opt.support_tol & X(:)<1-opt.support_tol)
    monitor=false;best_nnz=nz;stale=0;
        elseif ~monitor,monitor=true;best_nnz=nz;stale=0;
        elseif nz<best_nnz,best_nnz=nz;stale=0;
        else,stale=stale+1;end
        if monitor && stale>=opt.eta_stagnation_iters
            if eta_state<0
                if eta_state<=opt.eta_stage_initial,eta_state=0.5*eta_state;
                else,eta_state=0;end
            elseif eta_state==0,eta_state=opt.eta;
            else,eta_state=min(opt.eta_growth_factor*eta_state,eta_max);end
            stale=0;best_nnz=nz;
        end
    end
end
solve_time=toc(t0);
P_native=round(X);
if max(abs(X(:)-P_native(:)))>opt.support_tol || ...
        ~is_valid_partition(P_native,problem)
    P_native=[];
end
if isempty(P_native)
    native_cut=NaN;
    fprintf('Matrix: no native balanced-binary output (flag=%d, nnz=%d, residual=%.3e).\n',flag,nnz(X),dxy);
else
    native_cut=graph_cut(P_native,problem.W);
end
P_candidate=P_native;rounded=false;round_time=0;
if isempty(P_candidate)&&opt.allow_extra_rounding
    tt=tic;P_candidate=round_balanced_graph(X);
    round_time=toc(tt);rounded=true;
end
if isempty(P_candidate),candidate_cut=NaN;
else,candidate_cut=graph_cut(P_candidate,problem.W);end
out=struct('P_raw',P_candidate,'raw_cut',candidate_cut, ...
    'matrix_objective_scale',matrix_objective_scale, ...
    'binary_deviation_before_rounding',max(abs(X(:)-round(X(:)))), ...
    'P_native',P_native,'native_cut',native_cut,'native_valid',~isempty(P_native), ...
    'heuristic_rounding',rounded,'round_time',round_time, ...
    'rawtime',solve_time+round_time,'solve_time',solve_time,...
    'iterations',iter,'flag',flag,'eta_final',eta_state,...
    'beta_final',beta,'residual_r',r,'residual_dxy',dxy,...
    'nnzX',nnz(X),'X_final',X,'Y_final',Y);
end

function ok=is_valid_partition(P,p)
ok=isequal(size(P),[p.n,p.k]) && all(P(:)==0|P(:)==1) && ...
    all(sum(P,2)==1) && all(sum(P,1)==p.group_size);
end
