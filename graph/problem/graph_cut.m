function [J,valid] = graph_cut(P,W)
%GRAPH_CUT Cut weight counting each unordered edge once.
[n,k]=size(P);
valid=all(P(:)==0 | P(:)==1) && all(sum(P,2)==1) && ...
    all(sum(P,1)==n/k) && mod(n,k)==0;
if ~valid, J=NaN; return; end
D=spdiags(sum(W,2),0,n,n);
L=D-W;
J=full(0.5*sum(sum(P.*(L*P))));
end
