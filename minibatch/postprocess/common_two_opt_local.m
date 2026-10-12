function [part_cell_final, part_final, H_final, J_common] = ...
    common_two_opt_local(Phi, k, batch_size, part_cell, part, H, J_common, opts)
%COMMON_TWO_OPT_LOCAL Balanced 2-opt written directly on the J_common scale.
%
% This is the common local-search routine used after Matrix, Random, and
% eADMM rounding.  It does NOT use a graph Laplacian.
%
% Objective:
%   J_common(H) = <Phi,H>/(2*batch_size^2),
% where H is the 0/1 co-membership matrix of a balanced k-partition.
%
% For a selected pair of clusters, encode their membership by
% x in {-1,+1}^{2b}, sum(x)=0.  On that pair,
%
%   H_pair = (11' + xx')/2,
%
% so the variable part of J_common is proportional to x'*Phi_pair*x.
% A 2-opt move flips one +1 and one -1, i.e. swaps one vertex between the
% two clusters, and therefore preserves exact balance.
%
% opts.cost_tol controls floating-point consistency on the J_common scale.
% The original author's 2-opt move threshold 1e-4 is retained for the pairwise
% quadratic gain; because the graph pair objective and x'*Phi_pair*x differ
% only by a constant on balanced swaps, this preserves the original move rule.

    if nargin < 8 || isempty(opts)
        opts = struct();
    end
    if ~isfield(opts,'cost_tol') || isempty(opts.cost_tol)
        opts.cost_tol = 1e-9;
    end
    if ~isfield(opts,'verbose') || isempty(opts.verbose)
        opts.verbose = false;
    end

    n = size(Phi,1);
    if size(Phi,2) ~= n
        error('Phi must be square.');
    end
    if abs(n/k - batch_size) > 1e-12
        error('common_two_opt_local requires n/k = batch_size.');
    end

    if opts.verbose
        fprintf('J_common before local adjustment: %.12e\n', J_common);
    end

    part_cell_final = part_cell;
    part_final = part;
    H_final = H;

    pair_perms = nchoosek(1:k,2);
    count = 0;
    if k == 2
        max_sweeps = 1;
    else
        max_sweeps = 3;
    end

    if isempty(J_common)
        J_common = sum(Phi.*H,'all')/(2*batch_size^2);
    end

    for sweep = 1:max_sweeps
        pairsets = randperm(size(pair_perms,1));
        for pair_id = pairsets
            pair_temp = pair_perms(pair_id,:);
            s_idx = pair_temp(1);
            t_idx = pair_temp(2);

            part_cell_new = part_cell_final;
            part_new = part_final;
            H_new = H_final;

            part_s = part_cell_new{s_idx};
            part_t = part_cell_new{t_idx};
            union_idx = [part_s, part_t];

            subH = H_new(union_idx,union_idx);
            subPhi = Phi(union_idx,union_idx);
            subx = 2*subH(1,:) - 1;

            if abs(sum(subx)) > 1e-12
                error('Infeasible bisection encountered in common 2-opt.');
            end

            [subx_new,new_xqx,old_xqx] = ...
                bisection_2opt_common(subPhi,subx,batch_size,J_common,opts.cost_tol);

            % The change in J_common contributed by this pair is
            %   (new_xqx-old_xqx)/(4*batch_size^2).
            delta_J = (new_xqx-old_xqx)/(4*batch_size^2);
            accept_tol = opts.cost_tol * max(1,abs(J_common));

            if delta_J < -accept_tol
                subH_new = (subx_new'*subx_new + 1)/2;
                H_new(union_idx,union_idx) = subH_new;

                part_s_new = union( ...
                    part_s(subx_new(1:batch_size)==subx(1:batch_size)), ...
                    part_t(subx_new((batch_size+1):end)~=subx((batch_size+1):end)));
                part_t_new = union( ...
                    part_s(subx_new(1:batch_size)~=subx(1:batch_size)), ...
                    part_t(subx_new((batch_size+1):end)==subx((batch_size+1):end)));

                part_cell_new{s_idx} = part_s_new;
                part_cell_new{t_idx} = part_t_new;
                part_new{s_idx} = num2str(part_s_new');
                part_new{t_idx} = num2str(part_t_new');

                J_common = J_common + delta_J;
                part_cell_final = part_cell_new;
                part_final = part_new;
                H_final = H_new;

                if sweep > 1
                    count = 0;
                end
            elseif sweep > 1
                count = count + 1;
                if count > size(pair_perms,1)/2
                    J_common = recompute_jcommon(Phi,H_final,batch_size,J_common,opts);
                    return;
                end
            end
        end
    end

    J_common = recompute_jcommon(Phi,H_final,batch_size,J_common,opts);
end

function [xnew,cost,oldcost] = ...
    bisection_2opt_common(Q,x,batch_size,J_current,cost_tol)
% Minimize x'*Q*x over pairwise balanced swaps.

    m = length(x)/2;
    if m ~= batch_size
        error('Unexpected bisection size in common 2-opt.');
    end

    s_idx = find(x==1);
    t_idx = find(x==-1);

    Qx = Q*x';
    d = diag(Q);
    oldcost = x*Qx;
    cost = oldcost;

    delta = x'.*Qx-d;
    [improve_best,i_best,j_best] = best_swap_gain(Q,x,delta,s_idx,t_idx);

    % x'Qx decreases by 4*improve_best, hence J_common decreases by
    % improve_best/batch_size^2.
    gain_tol = max(1e-4, ...
        cost_tol * max(1,abs(J_current)) * batch_size^2);

    while improve_best > gain_tol
        % Incremental update of Qx after flipping x_i and x_j.
        old_xi = x(i_best);
        old_xj = x(j_best);
        Qx = Qx - 2*old_xi*Q(:,i_best) - 2*old_xj*Q(:,j_best);

        x(i_best) = -x(i_best);
        x(j_best) = -x(j_best);
        cost = cost - 4*improve_best;

        delta = x'.*Qx-d;
        s_idx = find(x==1);
        t_idx = find(x==-1);
        [improve_best,i_best,j_best] = best_swap_gain(Q,x,delta,s_idx,t_idx);
    end

    xnew = x;
    true_cost = xnew*Q*xnew';
    if abs(cost-true_cost) > 1e-9*max([1,abs(cost),abs(true_cost)])
        cost = true_cost;
    else
        cost = true_cost;
    end
end

function [gain_best,i_best,j_best] = best_swap_gain(Q,x,delta,s_idx,t_idx)
    gain_best = 0;
    i_best = [];
    j_best = [];
    for s = 1:numel(s_idx)
        i = s_idx(s);
        for t = 1:numel(t_idx)
            j = t_idx(t);
            gain = delta(i)+delta(j)-2*Q(i,j)*x(i)*x(j);
            if gain > gain_best
                gain_best = gain;
                i_best = i;
                j_best = j;
            end
        end
    end
end

function J = recompute_jcommon(Phi,H,batch_size,J_book,opts)
    off_binary = abs(H) > opts.cost_tol & abs(H-1) > opts.cost_tol;
    if any(off_binary(:))
        error('Common 2-opt returned a nonbinary co-membership matrix.');
    end

    J_true = sum(Phi.*H,'all')/(2*batch_size^2);
    if opts.verbose && abs(J_book-J_true) > opts.cost_tol*max([1,abs(J_book),abs(J_true)])
        fprintf('J_common bookkeeping corrected: %.12e -> %.12e\n',J_book,J_true);
    end
    J = J_true;
end
