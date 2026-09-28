function compile_matrix_mex()
%COMPILE_MATRIX_MEX Compile matrix_admm_mex.cpp using MATLAB's mwBLAS.
%
% Output:
%
%   methods/matrix/matrix_admm_mex.mexa64
%
% The MEX binary is deliberately placed next to the MATLAB Matrix solvers so
% the existing project path automatically finds it.

    this_file = mfilename('fullpath');
    this_dir = fileparts(this_file);

    matrix_dir = fileparts(this_dir);

    cpp_file = fullfile( ...
        this_dir, ...
        'matrix_admm_mex.cpp');

    if ~isfile(cpp_file)
        error( ...
            'Cannot find C++ source: %s', ...
            cpp_file);
    end

    fprintf('\n============================================================\n');
    fprintf('Compiling Matrix ADMM MEX\n');
    fprintf('MATLAB release : %s\n', version('-release'));
    fprintf('Source         : %s\n', cpp_file);
    fprintf('Output folder  : %s\n', matrix_dir);
    fprintf('MEX extension  : %s\n', mexext);
    fprintf('============================================================\n\n');

    % -R2017b:
    % Classic C MEX array API, convenient for BLAS mxGetPr usage.
    %
    % -O:
    % Enable MATLAB's optimized MEX build.
    %
    % -O3 / NDEBUG:
    % Additional compiler optimization for the tight numerical loop.
    %
    % -lmwblas:
    % Link MATLAB's BLAS implementation.
    mex( ...
        '-v', ...
        '-R2017b', ...
        '-O', ...
        'CXXFLAGS=$CXXFLAGS -O3 -DNDEBUG -std=c++11', ...
        cpp_file, ...
        '-lmwblas', ...
        '-outdir', ...
        matrix_dir);

    mex_file = fullfile( ...
        matrix_dir, ...
        ['matrix_admm_mex.' mexext]);

    if ~isfile(mex_file)
        error( ...
            'Compilation finished but MEX output was not found: %s', ...
            mex_file);
    end

    fprintf('\n============================================================\n');
    fprintf('MEX compilation succeeded.\n');
    fprintf('Output: %s\n', mex_file);
    fprintf('============================================================\n\n');
end