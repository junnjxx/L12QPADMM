function s = graph_spectral_norm(A)
%GRAPH_SPECTRAL_NORM Spectral 2-norm for dense or sparse symmetric matrices.
% Sparse norm(A,2) is unsupported by some MATLAB versions.
% eigs(A,1,'lm') estimates the eigenvalue of largest absolute magnitude.
% For a symmetric matrix, its magnitude equals ||A||_2.

    if ~ismatrix(A) || size(A,1) ~= size(A,2)
        error('graph_spectral_norm:NotSquare', ...
              'The input must be a square matrix.');
    end
    if ~isreal(A)
        error('graph_spectral_norm:ComplexInput', ...
              'Graph Laplacians are expected to be real.');
    end
    if nnz(A) == 0
        s = 0;
        return;
    end
    if ~issparse(A) || size(A,1) <= 12
        s = norm(full(A),2);
        return;
    end

    try
        s = abs(eigs(A,1,'lm'));
        if ~isfinite(s)
            error('graph_spectral_norm:Nonfinite', ...
                  'eigs returned a nonfinite value.');
        end
    catch ME
        if size(A,1) <= 2000
            warning('graph_spectral_norm:EigsFallback', ...
                'eigs failed (%s). Falling back to a dense spectral norm.', ...
                ME.message);
            s = norm(full(A),2);
        else
            rethrow(ME);
        end
    end
end
