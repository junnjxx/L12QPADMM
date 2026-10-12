function [F,G,fval] = lp_regularized_value_grad(model,X,sigma,p,eps_reg)
%LP_REGULARIZED_VALUE_GRAD Evaluate
%   F(X)=f(X)+sigma*sum_ij (X_ij+eps_reg)^p
% and its gradient.

if any(X(:) + eps_reg <= 0)
    error('X + eps_reg must be strictly positive for p in (0,1).');
end
fval = model.fun(X);
reg  = sum((X(:)+eps_reg).^p);
F    = fval + sigma*reg;
if nargout >= 2
    G = model.grad(X) + sigma*p*(X+eps_reg).^(p-1);
end
end
