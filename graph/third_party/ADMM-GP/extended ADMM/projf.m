function y = projf(x, lower_bound, upper_bound)
%PROJF Projection onto [lower_bound, upper_bound].
%
% The released aadmm_3b.m calls projf when updating sigma, but the public
% ADMM-GP repository does not contain a projf.m file.  The paper describes
% this step as a bounded adaptive stepsize update, so the required operation
% is the scalar projection onto [t_min,t_max].

y = min(max(x, lower_bound), upper_bound);
end
