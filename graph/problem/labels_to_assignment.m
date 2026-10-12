function P=labels_to_assignment(label,k)
n=numel(label);label=label(:);
if any(label<1|label>k|label~=round(label)),error('Bad partition labels');end
P=sparse((1:n)',label,1,n,k);
P=full(P);
end
