function Xbin = round_balanced_greedy(X,capacities)
%ROUND_BALANCED_GREEDY Balanced adaptation of the LagSA-style greedy rounding
% described in Section 6.1 of Jiang-Liu-Wen.
%
% Original permutation version:
%   1) r_i = max_j X_ij;
%   2) process rows in ascending r_i (most difficult first);
%   3) assign each row to its best still-unused column.
%
% Adaptation: each cluster j has capacities(j) available slots rather than one.

[n,m] = size(X);
capacities = round(capacities(:));
if numel(capacities)~=m || sum(capacities)~=n || any(capacities<0)
    error('capacities must be nonnegative integers summing to n.');
end

rowmax = max(X,[],2);
[~,ord] = sort(rowmax,'ascend');
remain = capacities;
Xbin = zeros(n,m);

for t = 1:n
    i = ord(t);
    avail = find(remain>0);
    if isempty(avail)
        error('No remaining capacity; inconsistent capacities.');
    end
    [~,loc] = max(X(i,avail));
    j = avail(loc);
    Xbin(i,j)=1;
    remain(j)=remain(j)-1;
end
end
