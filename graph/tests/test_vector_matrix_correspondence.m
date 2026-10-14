function test_vector_matrix_correspondence()
%TEST_VECTOR_MATRIX_CORRESPONDENCE Compare 1 ADMM step against Matrix algebra.
% Requires only the Vector solver (no external toolbox, since eta_max set).
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'methods','vector'));

Aorig=[8 -2 -1 0; -2 7 -1 -1; -1 -1 6 -2; 0 -1 -2 5];
b=2;
X0=[0.35;0.45;0.50;0.60];
Y0=[0.55;0.50;0.40;0.45];
opt=struct();
opt.c=10;opt.tau_X=1e-5;
opt.beta0=10;opt.beta_mode='adaptive';opt.kabeta=1e6;
opt.eta=0.08;opt.eta_adapt_enabled=false;
opt.eta_stage_initial=-10;opt.eta_max=3;
opt.eta_stagnation_iters=100;opt.eta_growth_factor=1.5;
opt.max_iter=1;opt.tol=1e-13;
opt.stop_check_interval=1;
opt.early_exist=false;opt.early_exist_interval=10;
opt.early_exist_dxy_tol=1e-4;opt.early_exist_round_tol=1e-10;
opt.support_tol=1e-10;opt.fast_prox=true;
opt.print_every=1000;opt.clean_fast=true;

[X,Y,iter,info]=vector_splitadmm(Aorig,b,opt,X0,Y0);
assert(iter==1);
scale=max(1,norm(Aorig,inf));A=Aorig/scale;
cp=opt.c+opt.tau_X;
R=(opt.c*Y0-0.5*A*Y0+opt.tau_X*X0)/cp;
lam=2*opt.eta/cp;
cutoff=(3/2^(5/3))*lam^(2/3);
Xref=zeros(numel(R),1);
active=R>=cutoff;
v=R(active);
arg=(3*sqrt(3)*lam/8)./(v.*sqrt(v));
arg=min(max(arg,-1),1);
Xref(active)=min((2/3).*v.*(1+cos(2*pi/3-(2/3)*acos(arg))),1);
H=Xref-(0.5*A*Xref)/opt.c;
Yref=H+(b-sum(H))/numel(H);

assert(norm(X-Xref,inf)<1e-12,'X update differs from Matrix.');
assert(norm(Y-Yref,inf)<1e-12,'Y projection differs.');
assert(abs(sum(Y)-b)<1e-12,'Vector affine projection invalid.');
assert(abs(info.objective_scale-scale)<1e-12);

p0=max(norm(Y0-X0),eps);
dxy=norm(Y-X);
betaref=opt.beta0*min(1,opt.kabeta*p0/(log(2)^2*max(dxy,eps)));
assert(abs(info.beta_final-betaref)<1e-10);

fprintf('PASS: one-step X-first prox, affine projection and adaptive beta.\n');
end
