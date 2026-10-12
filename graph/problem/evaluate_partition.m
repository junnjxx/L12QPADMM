function metrics = evaluate_partition(P,problem,eta)
%EVALUATE_PARTITION Original graph cut for exactly balanced binary partitions.
if nargin<3 || isempty(eta), eta=problem.eta; end
n=problem.num_samples; k=problem.num_batches; m=problem.batch_size;
metrics=struct('valid',false,'common_objective',NaN,'graph_cut',NaN, ...
    'matrix_quadratic_term',NaN,'matrix_linear_term',NaN,...
    'lhalf_unweighted',NaN,'matrix_lhalf_term',NaN,...
    'matrix_full_objective',NaN,'matrix_unregularized_objective',NaN,...
    'assignment',P,'comembership',[]);
if isempty(P) || ~isequal(size(P),[n,k]), return; end
if any(~isfinite(P(:))) || any(abs(P(:)-round(P(:)))>1e-12), return; end
if any(P(:)<0) || any(P(:)>1) || any(sum(P,2)~=1) || any(sum(P,1)~=m), return; end
H=P*P';
obj=0.5*full(sum(sum(problem.L.*H)));
obj2=0.5*full(sum(sum(P.*(problem.L*P))));
assert(abs(obj-obj2)<1e-8*max(1,abs(obj)),'Graph cut formulations disagree');
metrics.valid=true;
metrics.common_objective=obj;
metrics.graph_cut=obj;
% Matrix method internally uses L/s; for comparisons use ORIGINAL L in this diagnostic.
metrics.matrix_quadratic_term=obj;
metrics.matrix_linear_term=0;
metrics.lhalf_unweighted=n;
metrics.matrix_lhalf_term=eta*n;
metrics.matrix_unregularized_objective=obj;
metrics.matrix_full_objective=obj+eta*n;
metrics.assignment=P;
metrics.comembership=H;
end
