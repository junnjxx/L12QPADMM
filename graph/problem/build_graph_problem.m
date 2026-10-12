function p = build_graph_problem(W,k)
%BUILD_GRAPH_PROBLEM Exactly the original k-equipartition cut model.
W=double(W);[n,m]=size(W);
if n~=m || ~isscalar(k) || k~=round(k) || k<2 || mod(n,k)~=0
    error('W must be square; integer 2<=k<=n must divide n.');
end
if any(~isfinite(W(:))) || any(W(:)<0), error('Weights must be finite and nonnegative.'); end
if norm(W-W','fro')>1e-10*max(1,norm(W,'fro'))
    error('Adjacency must be symmetric.');
end
W=(W+W')/2;W(1:n+1:end)=0;
L=spdiags(sum(W,2),0,n,n)-W;
p=struct('n',n,'k',k,'group_size',n/k,'W',W,'L',L,...
    'A',L,'G',zeros(n,k));
end
