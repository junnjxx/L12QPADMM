function test_n2_fast_exactness()
%TEST_N2_FAST_EXACTNESS
% Randomized small-scale verification that local2swap_balanced_quadratic
% returns the same local optimum as a truly brute-force best-improvement
% N2 implementation that evaluates model.fun for every candidate swap.

rng(20260929);

ntrial = 30;
n = 20;
m = 4;
b = n/m;

opts.local_max_swaps = Inf;
opts.improve_tol = 1e-12;
opts.local_recompute_every = 5;
opts.local_verbose_every = 0;
opts.local_final_verify = true;

for trial = 1:ntrial

    A = randn(n);
    Q = 0.5*(A+A');
    C = 0.1*randn(n,m);
    q = 0.5 + rand();

    lab = repelem((1:m)',b);
    lab = lab(randperm(n));

    X0 = zeros(n,m);
    X0(sub2ind([n,m],(1:n)',lab)) = 1;

    model.kind = 'quadratic';
    model.Q = Q;
    model.qscale = q;
    model.C = C;
    model.fun = @(X) q*trace(X'*Q*X) + sum(C(:).*X(:));

    [Xfast,ffast,info] = ...
        local2swap_balanced_quadratic(X0,model,opts);

    [Xbrute,fbrute] = brute_n2best(X0,model,opts.improve_tol);

    ftol = 1e-9*max([1,abs(ffast),abs(fbrute)]);

    assert(abs(ffast-fbrute) <= ftol, ...
        'Trial %d: objective mismatch fast=%.16e brute=%.16e.', ...
        trial,ffast,fbrute);

    assert(isequal(Xfast,Xbrute), ...
        'Trial %d: final assignments differ.',trial);

    assert(info.local_optimal, ...
        'Trial %d: fast N2 did not certify local optimality.',trial);
end

fprintf('PASS: fast N2 matched brute-force N2best on %d random trials.\n', ...
    ntrial);

end

function [X,fval] = brute_n2best(X,model,tol)

[n,m] = size(X);

while true

    [~,lab] = max(X,[],2);

    best_f = model.fun(X);
    best_X = X;

    for r = 1:m-1
        A = find(lab==r);

        for s = r+1:m
            B = find(lab==s);

            for ia = 1:numel(A)
                a = A(ia);

                for ib = 1:numel(B)
                    b = B(ib);

                    Y = X;
                    Y(a,r) = 0;
                    Y(a,s) = 1;
                    Y(b,s) = 0;
                    Y(b,r) = 1;

                    fy = model.fun(Y);

                    if fy < best_f - tol
                        best_f = fy;
                        best_X = Y;
                    end
                end
            end
        end
    end

    fcur = model.fun(X);

    if best_f >= fcur - tol
        break;
    end

    X = best_X;
end

fval = model.fun(X);

end
