function [X,Y,error_list,flag,Lambda_history,r_collect,dxy_collect,round_y_collect,y_non_ratio,lambda_fnorm,dx_history,beta_history,ori_obj_part_list,eta_state_final,eta_update_history,eta_update_count,actual_iter,final_error,final_r,final_dxy] = splitadmm20240119(A,G,eta_state,beta0,c,outer_id,maxiter,X,Y,epsi,use_adaptive_beta,opts)
%SPLITADMM20240119 Paper-aligned pure MATLAB Matrix ADMM backend.
% Paper notation is used internally:
%   X: box/L1/2 block, 0<=X<=1.
%   Y: affine assignment block, 1_n'Y=b1_m', Y1_m=1_n.
% X-subproblem:
%   min_X L_c(X,Y^{k-1},Lambda^{k-1}) + tau_X/2*||X-X^{k-1}||_F^2.
% Hence
%   R^k=(cY^{k-1}+Lambda^{k-1}-0.5*A*Y^{k-1}+tau_X*X^{k-1})/(c+tau_X).
% The Y-subproblem is the Euclidean projection of
%   B^k=X^k-(Lambda^{k-1}+0.5*A*X^k+G)/c
% onto the balanced assignment affine set.
if ~isfield(opts,'matrix_clean'), opts.matrix_clean=false; end
if ~isfield(opts,'matrix_clean_fast'), opts.matrix_clean_fast=false; end
if ~isfield(opts,'matrix_detail'), opts.matrix_detail=false; end
if ~isfield(opts,'verbose'), opts.verbose=true; end
if ~isfield(opts,'print_every'), opts.print_every=100; end
if ~isfield(opts,'stop_check_interval'), opts.stop_check_interval=10; end
if ~isfield(opts,'early_exist'), opts.early_exist=false; end
if ~isfield(opts,'early_exist_dxy_tol'), opts.early_exist_dxy_tol=1e-4; end
if ~isfield(opts,'early_exist_round_tol'), opts.early_exist_round_tol=1e-2; end
if ~isfield(opts,'early_exist_interval'), opts.early_exist_interval=opts.stop_check_interval; end
if ~isfield(opts,'eta_adapt_enabled'), opts.eta_adapt_enabled=false; end
if ~isfield(opts,'eta'), opts.eta=max(eta_state,0); end
if ~isfield(opts,'eta_stage_initial'), opts.eta_stage_initial=-1; end
if ~isfield(opts,'eta_stagnation_iters'), opts.eta_stagnation_iters=100; end
if ~isfield(opts,'eta_growth_factor'), opts.eta_growth_factor=1.1; end
if ~isfield(opts,'eta_max'), opts.eta_max=opts.eta; end
if ~isfield(opts,'beta_mode'), opts.beta_mode="fixed"; end
if ~isfield(opts,'kabeta'), opts.kabeta=1; end
if ~isfield(opts,'beta_switch_iter'), opts.beta_switch_iter=1000; end
if ~isfield(opts,'tau_X'), error('Paper-aligned Matrix ADMM requires opts.tau_X.'); end
tau_X=double(opts.tau_X);
if ~isscalar(tau_X) || ~isfinite(tau_X) || tau_X<=0, error('opts.tau_X must be a positive finite scalar.'); end
matrix_clean=logical(opts.matrix_clean);
matrix_clean_fast=logical(opts.matrix_clean_fast);
matrix_detail=logical(opts.matrix_detail);
beta_mode=string(opts.beta_mode);
[N,B]=size(G);
if ~isequal(size(A),[N,N]), error('A must be N-by-N where G is N-by-B.'); end
if ~isequal(size(X),[N,B]) || ~isequal(size(Y),[N,B]), error('Initial X and Y must both be N-by-B.'); end
inv_c=1/c;
inv_N=1/N;
inv_B=1/B; 
cprime=c+tau_X;
target_batch_size=N/B;
target_batch_size_round=round(target_batch_size);
integer_capacity=abs(target_batch_size-target_batch_size_round)<=10*eps(max(1,target_batch_size));
Lambda=zeros(N,B);
AY=A*Y;
p0=max(norm(Y-X,'fro'),eps);
error_list=[];
r_collect=[];
dxy_collect=[];
round_y_collect=[];
y_non_ratio=[];
Lambda_history=cell(0,1);
lambda_fnorm=[];
dx_history=[];
beta_history=[];
ori_obj_part_list=[];
flag=0;
actual_iter=0;
final_error=NaN;
final_r=NaN;
final_dxy=NaN;
eta_update_history=struct('outer_id',{},'iteration',{},'nnzY',{},'dxy',{},'eta_state_old',{},'eta_state_new',{},'eta_effective_old',{},'eta_effective_new',{});
eta_update_count=0;
current_nnz=nnz(X);
eta_monitoring=opts.eta_adapt_enabled && eta_state<opts.eta_max && current_nnz>N;
eta_best_nnz=current_nnz;
eta_no_improvement=0;
[eta_effective,prox_threshold,prox_acos_coeff]=make_eta_cache(eta_state,cprime);
if opts.verbose
    fprintf('[splitadmm20240119] paper-aligned blocks | tau_X=%.6g | cprime=%.6g\n',tau_X,cprime);
end
for iter=1:maxiter
    actual_iter=iter;
    X_old=X;
    Y_old=Y;
    Lambda_old=Lambda;
    AY_old=AY;
    % X block: paper Eq. (4.1)-(4.5), including tau_X proximal term.
    R=(c*Y_old+Lambda_old-0.5*AY_old+tau_X*X_old)/cprime;
    if eta_effective==0
        X=min(max(R,0),1);
    else
        lambda_half=2*eta_effective/cprime;
        if matrix_clean_fast
            X=zeros(N,B);
            active=R>=prox_threshold;
            if any(active(:))
                v=R(active);
                arg=prox_acos_coeff./(v.*sqrt(v));
                arg=min(max(arg,-1),1);
                x=(2/3).*v.*(1+cos(2*pi/3-(2/3)*acos(arg)));
                x=min(x,1);
                X(active)=x;
            end
        else
            X=zeros(N,B);
            candidate_threshold=(3/4)*lambda_half^(2/3);
            active=R>candidate_threshold;
            if any(active(:))
                v=R(active);
                arg=(lambda_half/8).*(v/3).^(-1.5);
                arg=min(max(arg,-1),1);
                x=(2/3).*v.*(1+cos(2*pi/3-(2/3)*acos(arg)));
                lose_to_zero=(x-v).^2+lambda_half*sqrt(max(x,0))-v.^2>0;
                x(lose_to_zero)=0;
                x=min(x,1);
                X(active)=x;
            end
        end
    end
    current_nnz=nnz(X);
    % Y block: paper affine projection.
    AX=A*X;
    Bmat=X-inv_c*(Lambda_old+0.5*AX+G);
    eY=sum(Bmat,1);
    row_correction=(1+sum(eY)*inv_N-sum(Bmat,2))*inv_B;
    Y=Bmat-eY*inv_N+row_correction;
    % Primal residual for current iterate.
    dxy_now=norm(Y-X,'fro');
    % beta is chosen from the current primal residual, as in the paper.
    beta=choose_beta(beta_mode,beta0,opts.kabeta,p0,dxy_now,iter,opts.beta_switch_iter,X,N,use_adaptive_beta);
    Lambda=Lambda_old+beta*(Y-X);
    AY_new=A*Y;
    stop_due=mod(iter,opts.stop_check_interval)==0 || iter==maxiter;
    early_due=opts.early_exist && mod(iter,opts.early_exist_interval)==0;
    early_hit=false;
    dxy=NaN;
    if stop_due || early_due, dxy=dxy_now; end
    if opts.early_exist && early_due && dxy<=opts.early_exist_dxy_tol
        X_round=round(X);
        round_diff=max(abs(X(:)-X_round(:)));
        if round_diff<opts.early_exist_round_tol && integer_capacity
            row_ok=all(sum(X_round,2)==1);
            col_ok=all(sum(X_round,1)==target_batch_size_round);
            if row_ok && col_ok
                X=X_round;
                current_nnz=N;
                dxy=norm(Y-X,'fro');
                early_hit=true;
                AX=A*X;
            end
        end
    end
    need_residual=stop_due || early_hit;
    r=NaN;
    stopping_error=NaN;
    if need_residual
        if ~isfinite(dxy), dxy=norm(Y-X,'fro'); end
        dual_delta=Lambda-Lambda_old;
        % Paper Lemma 5.5 X-block stationarity error:
        % dLambda + 0.5*A*(Y^{k-1}-Y^k)+c*(Y^k-Y^{k-1})+tau_X*(X^k-X^{k-1}).
        x_stationarity=dual_delta+0.5*(AY_old-AY_new)+c*(Y-Y_old)+tau_X*(X-X_old);
        r=max(norm(x_stationarity,'fro'),norm(dual_delta,'fro'));
        stopping_error=max(r,dxy);
        final_error=stopping_error;
        final_r=r;
        final_dxy=dxy;
    end
    if ~matrix_clean
        if need_residual
            error_list(end+1,1)=stopping_error; %#ok<AGROW>
            r_collect(end+1,1)=r; %#ok<AGROW>
            dxy_collect(end+1,1)=dxy; %#ok<AGROW>
        end
        beta_history(end+1,1)=beta; %#ok<AGROW>
        lambda_fnorm(end+1,1)=norm(Lambda,'fro'); %#ok<AGROW>
        dx_history(end+1,1)=norm(X-X_old,'fro'); %#ok<AGROW>
        if matrix_detail
            round_dev=norm(X-round(X),'fro');
            fractional=mean(X(:)>1e-8 & X(:)<1-1e-8);
            round_y_collect(end+1,1)=round_dev; %#ok<AGROW>
            y_non_ratio(end+1,1)=fractional; %#ok<AGROW>
            Lambda_history{end+1,1}=Lambda; %#ok<AGROW>
            cross_linear=0.5*sum(sum(X.*AY_new))+sum(sum(G.*Y));
            lhalf_part=eta_effective*sum(sqrt(max(X(:),0)));
            ori_obj_part_list(end+1,1)=cross_linear+lhalf_part; %#ok<AGROW>
        end
    end
    if opts.verbose && ~matrix_clean && mod(iter,opts.print_every)==0
        if isfinite(r) && isfinite(dxy)
            fprintf('Matrix iter=%d | beta=%.6g | eta_state=%.6g | eta_eff=%.6g | nnzX=%d | r=%.3e | dxy=%.3e\n',iter,beta,eta_state,eta_effective,current_nnz,r,dxy);
        else
            fprintf('Matrix iter=%d | beta=%.6g | eta_state=%.6g | eta_eff=%.6g | nnzX=%d\n',iter,beta,eta_state,eta_effective,current_nnz);
        end
    end
    if early_hit
        flag=2;
        AY=AY_new;
        break;
    end
    if stop_due && stopping_error<epsi
        flag=1;
        AY=AY_new;
        break;
    end
    if opts.eta_adapt_enabled && eta_state<opts.eta_max
        if current_nnz<=N
            eta_monitoring=false;
            eta_best_nnz=current_nnz;
            eta_no_improvement=0;
        elseif ~eta_monitoring
            eta_monitoring=true;
            eta_best_nnz=current_nnz;
            eta_no_improvement=0;
        elseif current_nnz<eta_best_nnz
            eta_best_nnz=current_nnz;
            eta_no_improvement=0;
        else
            eta_no_improvement=eta_no_improvement+1;
        end
        if eta_monitoring && eta_no_improvement>=opts.eta_stagnation_iters
            eta_old=eta_state;
            eta_state=next_eta_state(eta_state,opts.eta_stage_initial,opts.eta,opts.eta_growth_factor,opts.eta_max);
            eta_update_count=eta_update_count+1;
            if ~matrix_clean
                if isfinite(dxy), dxy_record=dxy; else, dxy_record=dxy_now; end
                eta_update_history(end+1)=struct('outer_id',outer_id,'iteration',iter,'nnzY',current_nnz,'dxy',dxy_record,'eta_state_old',eta_old,'eta_state_new',eta_state,'eta_effective_old',max(eta_old,0),'eta_effective_new',max(eta_state,0)); %#ok<AGROW>
            end
            eta_best_nnz=current_nnz;
            eta_no_improvement=0;
            [eta_effective,prox_threshold,prox_acos_coeff]=make_eta_cache(eta_state,cprime);
        end
    end
    AY=AY_new;
end
if actual_iter==0
    final_dxy=norm(Y-X,'fro');
    final_r=NaN;
    final_error=final_dxy;
elseif ~isfinite(final_dxy)
    final_dxy=norm(Y-X,'fro');
    final_error=final_dxy;
end
eta_state_final=eta_state;
end

function eta_new=next_eta_state(eta_old,eta_stage_initial,eta_positive_start,eta_growth,eta_max)
if eta_old<0
    if eta_old<=eta_stage_initial, eta_new=0.5*eta_old; else, eta_new=0; end
elseif eta_old==0
    eta_new=eta_positive_start;
else
    eta_new=min(eta_growth*eta_old,eta_max);
end
end

function [eta_effective,threshold,acos_coeff]=make_eta_cache(eta_state,cprime)
eta_effective=max(eta_state,0);
if eta_effective==0
    threshold=NaN;
    acos_coeff=NaN;
    return;
end
lambda_half=2*eta_effective/cprime;
threshold=(3/2^(5/3))*lambda_half^(2/3);
acos_coeff=3*sqrt(3)*lambda_half/8;
end

function beta=choose_beta(beta_mode,beta0,kabeta,p0,pk,iter,beta_switch_iter,X,N,use_adaptive_beta)
if ~use_adaptive_beta || beta_mode=="fixed"
    beta=beta0;
    return;
end
adaptive_beta=beta0*min(1,kabeta*p0/(max(iter,1)*log(iter+1)^2*max(pk,eps)));
switch beta_mode
    case "adaptive"
        beta=adaptive_beta;
    case "fix_then_adaptive"
        if iter<=beta_switch_iter, beta=beta0; else, beta=adaptive_beta; end
    case "adaptive_then_fix"
        if iter<=beta_switch_iter, beta=adaptive_beta; else, beta=beta0; end
    case "adaptive_until_binary"
        if nnz(X)<=N, beta=beta0; else, beta=adaptive_beta; end
    otherwise
        error('Unknown Matrix beta_mode: %s',char(beta_mode));
end
end
