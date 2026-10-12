function [P,J,swaps] = graph_two_opt(P,W,opts)
%GRAPH_TWO_OPT Exact balanced pairwise swap local search on ORIGINAL cut.
% Recomputes scores after accepted moves to avoid stale incremental gains.
if nargin<3,opts=struct();end
if ~isfield(opts,'max_passes'),opts.max_passes=10;end
if ~isfield(opts,'improve_tol'),opts.improve_tol=1e-10;end
if ~isfield(opts,'verbose'),opts.verbose=false;end
[J,valid]=graph_cut(P,W);if ~valid,error('Invalid initial equipartition');end
[n,k]=size(P);[~,labels]=max(P,[],2);
% For swapping i in cluster a and j in cluster b: within-weight difference
% delta_cut = s_i(a)-s_i(b)+s_j(b)-s_j(a)+2*W(i,j).
swaps=0;
for pass=1:opts.max_passes
    improved=false;
    S=W*P;
    for i=1:n
        for j=i+1:n
            % Important: i may have changed clusters in an earlier swap
            % during this same inner loop. Never cache labels(i) outside.
            a=labels(i);
            b=labels(j);
            if a==b,continue;end
            delta=S(i,a)-S(i,b)+S(j,b)-S(j,a)+2*W(i,j);
            if delta < -opts.improve_tol
                P(i,a)=0;P(i,b)=1;P(j,b)=0;P(j,a)=1;
                labels(i)=b;labels(j)=a;
                S=W*P;swaps=swaps+1;improved=true;
            end
        end
    end
    if ~improved,break;end
    if opts.verbose,fprintf('2-opt pass=%d | swaps=%d\n',pass,swaps);end
end
[J,valid]=graph_cut(P,W);
assert(valid,'2-opt lost balance');
end
