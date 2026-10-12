function model = make_quadratic_model(Q, qscale, C)
%MAKE_QUADRATIC_MODEL Construct f(X)=qscale*tr(X'*Q*X)+<C,X>.
%
% model = make_quadratic_model(Q)
% model = make_quadratic_model(Q,qscale)
% model = make_quadratic_model(Q,qscale,C)
%
% Q need not be explicitly symmetrized; the gradient uses (Q+Q').
% For a graph Laplacian objective tr(X'*L*X), use qscale=1 and Q=L.
% For 0.5*tr(X'*L*X), use qscale=0.5.

if nargin < 2 || isempty(qscale)
    qscale = 1.0;
end
n = size(Q,1);
if size(Q,2) ~= n
    error('Q must be square.');
end
if nargin < 3 || isempty(C)
    C = zeros(n,0); % expanded lazily once m is known
end

model.kind   = 'quadratic';
model.n      = n;
model.Q      = Q;
model.qscale = qscale;
model.C      = C;
model.fun    = @fun;
model.grad   = @grad;

% For f(X)=qscale*tr(X'QX), Hessian in each column is
% qscale*(Q+Q').  We do NOT automatically compute lambda_min for large Q.
% Set model.curvature_lower manually if desired.
model.curvature_lower = [];

    function val = fun(X)
        if isempty(C) || size(C,2)==0
            lin = 0;
        else
            if ~isequal(size(C),size(X))
                error('Linear term C must have the same size as X.');
            end
            lin = sum(C(:).*X(:));
        end
        QX = Q*X;
        val = qscale*sum(sum(X.*QX)) + lin;
    end

    function G = grad(X)
        G = qscale*((Q+Q')*X);
        if ~(isempty(C) || size(C,2)==0)
            if ~isequal(size(C),size(X))
                error('Linear term C must have the same size as X.');
            end
            G = G + C;
        end
    end
end
