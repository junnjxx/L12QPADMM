function P=random_equipartition(n,k,seed)
if mod(n,k)~=0, error('k must divide n'); end
rng(seed,'twister');order=randperm(n);
label=zeros(n,1);label(order)=repelem((1:k)',n/k);
P=labels_to_assignment(label,k);
end
