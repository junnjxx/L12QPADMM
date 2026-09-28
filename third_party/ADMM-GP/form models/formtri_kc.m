function [B,f] = formtri_kc(T,f,n)
%FORMTRI_KC Build the MET/transitivity operator B from triangle descriptors.
%
% The upstream ADMM-GP README calls formtri_kc(T,f,n), but that helper is not
% present in the public repository tree. This compatibility implementation
% follows the triangle types used by the upstream tri_sep_kc.c exactly:
%
%   type 0:  x_ij - x_ik - x_jk >= -1
%   type 1: -x_ij + x_ik - x_jk >= -1
%   type 2: -x_ij - x_ik + x_jk >= -1
%
% The returned matrix B has one row per inequality and satisfies
%       B * X(:) >= f
% for symmetric X. Coefficients are split equally over (i,j) and (j,i) so
% that B^*(y_bar) is symmetric, matching the SDP operator convention used by
% kequi_form.m and mprw_ineq_general.m.

    if isempty(T)
        B = sparse(0,n*n);
        f = zeros(0,1);
        return;
    end

    if size(T,2) ~= 4
        error('T must have four columns [i,j,k,type].');
    end

    p = size(T,1);
    if isempty(f)
        f = -ones(p,1);
    elseif numel(f) ~= p
        error('Length of f (%d) must equal number of MET constraints (%d).',numel(f),p);
    else
        f = f(:);
    end

    rows = zeros(6*p,1);
    cols = zeros(6*p,1);
    vals = zeros(6*p,1);
    pos = 0;

    for r = 1:p
        i = T(r,1); j = T(r,2); k = T(r,3); typ = T(r,4);
        switch typ
            case 0
                coeff = [ 1, -1, -1]; % ij, ik, jk
            case 1
                coeff = [-1,  1, -1];
            case 2
                coeff = [-1, -1,  1];
            otherwise
                error('Unknown MET triangle type %g.',typ);
        end

        pairs = [i j; i k; j k];
        for q = 1:3
            a = pairs(q,1); b = pairs(q,2); c = coeff(q)/2;
            pos = pos + 1;
            rows(pos)=r; cols(pos)=a+(b-1)*n; vals(pos)=c;
            pos = pos + 1;
            rows(pos)=r; cols(pos)=b+(a-1)*n; vals(pos)=c;
        end
    end

    B = sparse(rows,cols,vals,p,n*n);
end
