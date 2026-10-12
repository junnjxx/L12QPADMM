function P=round_balanced_graph(X)
%ROUND_BALANCED_GRAPH Heuristic only; never called a native Matrix solution.
[n,k]=size(X);if mod(n,k)~=0,error('k must divide n');end
capacity=n/k; count=zeros(1,k);label=zeros(n,1);
[~,ord]=sort(X(:),'descend');
for z=1:numel(ord)
    [i,j]=ind2sub([n,k],ord(z));
    if label(i)==0 && count(j)<capacity
        label(i)=j;count(j)=count(j)+1;
        if all(label>0),break;end
    end
end
if any(label==0),error('Balanced greedy rounding failed');end
P=labels_to_assignment(label,k);
end
