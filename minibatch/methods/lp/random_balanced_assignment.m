function X = random_balanced_assignment(n,capacities)
%RANDOM_BALANCED_ASSIGNMENT Generate a random feasible binary assignment.
capacities = round(capacities(:));
m = numel(capacities);
if sum(capacities)~=n
    error('sum(capacities) must equal n.');
end
labels = zeros(n,1);
pos = 1;
for j=1:m
    labels(pos:pos+capacities(j)-1)=j;
    pos = pos+capacities(j);
end
labels = labels(randperm(n));
X = sparse((1:n)',labels,1,n,m);
X = full(X);
end
