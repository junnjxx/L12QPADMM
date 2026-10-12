function [T,b,hash,gamma] = tri_sep_kc(y,n,maxtri,h_tmp)
%TRI_SEP_KC Pure-MATLAB fallback for the upstream tri_sep_kc MEX routine.
%
% Same interface and triangle types as tri_sep_kc.c in ADMM-GP:
%   type 0: x_ik + x_jk <= x_ij + 1
%   type 1: x_ij + x_jk <= x_ik + 1
%   type 2: x_ik + x_ij <= x_jk + 1
%
% The original C source uses violation threshold 1e-3 and supports n<=1000
% because of its hash encoding. This fallback preserves both conventions.
% If a compiled tri_sep_kc MEX file is present, MATLAB will normally dispatch
% to the MEX implementation instead of this fallback.

    if n > 1000
        error('tri_sep_kc hash convention supports n <= 1000.');
    end
    if numel(y) ~= n*n
        error('y and n inconsistent.');
    end
    if nargin < 4 || isempty(h_tmp)
        h_tmp = zeros(0,1);
    end

    X = reshape(y,n,n);
    threshold = 1e-3;
    test_only = (maxtri < 1);
    if test_only
        maxtri = 1;
    end
    maxtri = min(maxtri,20000);

    h_old = sort(uint64(h_tmp(:)));

    best_gamma = zeros(0,1);
    best_T = zeros(0,4);
    best_hash = zeros(0,1,'uint64');

    for i = 1:n-2
        for j = i+1:n-1
            yij = X(i,j);
            for k = j+1:n
                yik = X(i,k);
                yjk = X(j,k);

                violations = [ ...
                    yik + yjk - yij - 1; ... % type 0
                    yij + yjk - yik - 1; ... % type 1
                    yik + yij - yjk - 1];    % type 2

                for typ = 0:2
                    viol = violations(typ+1);
                    if viol <= threshold
                        continue;
                    end

                    h = ((uint64(typ)*1000 + uint64(i))*1000 + uint64(j))*1000 + uint64(k);
                    if hash_is_present(h_old,h)
                        continue;
                    end

                    if numel(best_gamma) < maxtri
                        best_gamma(end+1,1) = viol; %#ok<AGROW>
                        best_T(end+1,:) = [i,j,k,typ]; %#ok<AGROW>
                        best_hash(end+1,1) = h; %#ok<AGROW>
                    else
                        [gmin,imin] = min(best_gamma);
                        if viol > gmin
                            best_gamma(imin) = viol;
                            best_T(imin,:) = [i,j,k,typ];
                            best_hash(imin) = h;
                        end
                    end

                    if test_only
                        break;
                    end
                end
                if test_only && ~isempty(best_gamma)
                    break;
                end
            end
            if test_only && ~isempty(best_gamma)
                break;
            end
        end
        if test_only && ~isempty(best_gamma)
            break;
        end
    end

    if isempty(best_gamma)
        T = zeros(0,1);
        b = zeros(0,1);
        hash = zeros(0,1);
        gamma = zeros(0,1);
        return;
    end

    [gamma,ord] = sort(best_gamma,'descend');
    Tmat = best_T(ord,:);
    hash = double(best_hash(ord));
    b = zeros(numel(gamma),1);
    T = Tmat(:);
end

function tf = hash_is_present(sorted_hash,h)
    lo = 1;
    hi = numel(sorted_hash);
    tf = false;
    while lo <= hi
        mid = floor((lo+hi)/2);
        if sorted_hash(mid) == h
            tf = true;
            return;
        elseif sorted_hash(mid) < h
            lo = mid + 1;
        else
            hi = mid - 1;
        end
    end
end
