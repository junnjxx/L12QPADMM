function data = load_admm_gp_kequipartition(cfg,project_root)
%LOAD_ADMM_GP_KEQUIPARTITION
% Load official random k-equipartition graph from ADMM-GP.
%
% Supported families:
%   rand20
%   rand50
%   rand80
%
% Official files:
%   Keq_rand_p0.20.mat
%   Keq_rand_p0.50.mat
%   Keq_rand_p0.80.mat
%
% Expected graph sizes:
%   n = 100,200,...,1000.
%
% Output:
%   data.W              weighted adjacency matrix
%   data.num_samples    n
%   data.family
%   data.requested_density
%   data.empirical_density
%   data.source_file
%   data.source_entry

    family = lower(string(cfg.graph.family));
    n = double(cfg.graph.n);

    switch family

        case "rand20"

            filename = ...
                'Keq_rand_p0.20.mat';

            requested_density = ...
                0.20;

        case "rand50"

            filename = ...
                'Keq_rand_p0.50.mat';

            requested_density = ...
                0.50;

        case "rand80"

            filename = ...
                'Keq_rand_p0.80.mat';

            requested_density = ...
                0.80;

        otherwise

            error( ...
                'load_admm_gp_kequipartition:UnknownFamily', ...
                'Unknown graph family "%s".',family);
    end

    data_dir = ...
        fullfile( ...
            project_root, ...
            'data', ...
            char(cfg.graph.data_subdir));

    file_path = ...
        fullfile(data_dir,filename);

    if ~exist(file_path,'file')

        error( ...
            'load_admm_gp_kequipartition:MissingFile', ...
            ['Official ADMM-GP file not found:\n  %s\n\n', ...
             'Put the three Keq_rand_p0.xx.mat files under ', ...
             '<project_root>/data/admm_gp/.'], ...
            file_path);
    end

    S = load(file_path);

    varname = ...
        sprintf('A%d',n);

    if ~isfield(S,varname)

        fprintf('Variables found in %s:\n',filename);

        disp(fieldnames(S));

        error( ...
            'load_admm_gp_kequipartition:MissingGraphSize', ...
            'Variable %s was not found.',varname);
    end

    raw = ...
        S.(varname);

    [candidates,sources] = ...
        collect_square_graphs( ...
            raw,n,['root.',varname]);

    if isempty(candidates)

        error( ...
            'load_admm_gp_kequipartition:NoAdjacencyFound', ...
            'Could not find an %d-by-%d adjacency matrix inside %s.', ...
            n,n,varname);
    end

    instance_id = ...
        double(cfg.graph.instance_index);

    if instance_id < 1 || ...
            instance_id ~= round(instance_id) || ...
            instance_id > numel(candidates)

        error( ...
            'Requested instance_index=%d, but %s contains %d candidate(s).', ...
            instance_id,varname,numel(candidates));
    end

    W = ...
        full(double(candidates{instance_id}));

    source_entry = ...
        string(sources{instance_id});

    if ~isequal(size(W),[n,n])

        error('Loaded adjacency matrix has wrong size.');
    end

    if any(~isfinite(W(:)))

        error('Adjacency matrix contains NaN/Inf.');
    end

    scale = ...
        max(1,norm(W,'fro'));

    sym_err = ...
        norm(W-W','fro');

    if sym_err > 1e-12*scale

        error( ...
            'Official adjacency matrix is unexpectedly asymmetric: %.3e.', ...
            sym_err);
    end

    W = ...
        0.5*(W+W');

    diag_err = ...
        max(abs(diag(W)));

    if diag_err > 1e-12*scale

        error( ...
            'Official adjacency matrix has nonzero diagonal: %.3e.', ...
            diag_err);
    end

    W(1:n+1:end) = 0;

    if min(W,[],'all') < -1e-12*scale

        error('Adjacency matrix contains negative edge weights.');
    end

    upper_weights = ...
        nonzeros(triu(W,1));

    num_edges = ...
        numel(upper_weights);

    max_edges = ...
        n*(n-1)/2;

    empirical_density = ...
        num_edges/max_edges;

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('Loaded official ADMM-GP k-equipartition graph\n');
    fprintf('family=%s | n=%d | instance=%d\n', ...
        family,n,instance_id);
    fprintf('source=%s\n',source_entry);
    fprintf('requested density=%.0f%% | empirical density=%.4f\n', ...
        100*requested_density,empirical_density);

    if isempty(upper_weights)

        fprintf('graph has no nonzero edges\n');

    else

        fprintf(['edges=%d | edge weight min=%.6g | ', ...
            'max=%.6g | mean=%.6g\n'], ...
            num_edges, ...
            min(upper_weights), ...
            max(upper_weights), ...
            mean(upper_weights));
    end

    fprintf('============================================================\n');

    data = struct();

    data.W = W;

    data.num_samples = n;

    data.family = family;

    data.requested_density = ...
        requested_density;

    data.empirical_density = ...
        empirical_density;

    data.num_edges = ...
        num_edges;

    data.source_file = ...
        string(file_path);

    data.source_entry = ...
        source_entry;

    data.instance_index = ...
        instance_id;
end


function [mats,names] = collect_square_graphs(x,n,prefix)
% Recursively find n-by-n numeric matrices.

    mats = {};
    names = {};

    if isnumeric(x)

        if ismatrix(x) && ...
                isequal(size(x),[n,n])

            mats = {x};
            names = {prefix};

            return;
        end

        if ndims(x)==3 && ...
                size(x,1)==n && ...
                size(x,2)==n

            r = size(x,3);

            mats = cell(r,1);
            names = cell(r,1);

            for ii=1:r

                mats{ii} = ...
                    x(:,:,ii);

                names{ii} = ...
                    sprintf('%s(:,:,%d)',prefix,ii);
            end

            return;
        end

        return;
    end

    if iscell(x)

        for ii=1:numel(x)

            [mi,ni] = ...
                collect_square_graphs( ...
                    x{ii}, ...
                    n, ...
                    sprintf('%s{%d}',prefix,ii));

            mats = [mats;mi(:)]; %#ok<AGROW>
            names = [names;ni(:)]; %#ok<AGROW>
        end

        return;
    end

    if isstruct(x)

        f = fieldnames(x);

        for ee=1:numel(x)

            for jj=1:numel(f)

                name = f{jj};

                [mi,ni] = ...
                    collect_square_graphs( ...
                        x(ee).(name), ...
                        n, ...
                        sprintf('%s(%d).%s', ...
                            prefix,ee,name));

                mats = [mats;mi(:)]; %#ok<AGROW>
                names = [names;ni(:)]; %#ok<AGROW>
            end
        end
    end
end