function [W,meta] = load_graph_instance(cfg,root)
%LOAD_GRAPH_INSTANCE Graph-only. Supports official MAT, arbitrary MAT, and demo.
source=lower(char(cfg.source));meta=struct('source',source,'family',cfg.family,'path','');
if strcmp(source,'demo')
    n=cfg.n;rng(cfg.demo_seed,'twister');
    H=triu(rand(n)<cfg.demo_density,1);
    W=sparse(double(H+H'));
    fprintf('NOTE: Synthetic DEMO instance (not ADMM-GP official benchmark).\n');
elseif strcmp(source,'matrix')
    W=cfg.W;
elseif strcmp(source,'official') || strcmp(source,'mat')
    if strcmp(source,'official')
        switch lower(char(cfg.family))
            case 'rand20',filename='Keq_rand_p0.20.mat';
            case 'rand50',filename='Keq_rand_p0.50.mat';
            case 'rand80',filename='Keq_rand_p0.80.mat';
            otherwise,error('Family must be rand20/rand50/rand80');
        end
        if isempty(cfg.mat_path)
            f=fullfile(root,'data','admm_gp',filename);
        else
            f=char(cfg.mat_path);
        end
    else
        f=char(cfg.mat_path);
    end
    if exist(f,'file')~=2
        error(['Graph MAT not found: %s\n' ...
            'Put the official file in data/admm_gp/ or set cfg.graph.mat_path.'],f);
    end
    meta.path=f;
    S=load(f);
    if ~isempty(cfg.variable)
        key=char(cfg.variable);
        if ~isfield(S,key),error('Variable %s not found in %s',key,f);end
        val=S.(key);
    else
        key=sprintf('A%d',cfg.n);
        if isfield(S,key),val=S.(key);
        else
            keys=fieldnames(S);val=S;key='(automatic)';
            if isempty(keys),error('MAT file is empty');end
        end
    end
    matrices=extract_squares(val,cfg.n,3);
    if numel(matrices)<cfg.instance
        error('Found %d square adjacency candidates; requested instance %d.',numel(matrices),cfg.instance);
    end
    W=matrices{cfg.instance};
    fprintf('Loaded graph: %s | field %s | instance %d\n',f,key,cfg.instance);
else
    error('Unknown cfg.graph.source: %s',source);
end
W=double(W);
if size(W,1)~=cfg.n || size(W,2)~=cfg.n
    error('Loaded graph size %dx%d but cfg.graph.n=%d.',size(W,1),size(W,2),cfg.n);
end
if any(~isfinite(W(:))) || any(W(:)<0)
    error('Graph adjacency must have finite nonnegative weights.');
end
if norm(W-W','fro')>1e-8*max(1,norm(W,'fro'))
    error('Graph adjacency must be symmetric; no automatic symmetrization of directed graph.');
end
W=(W+W')/2;
W(1:size(W,1)+1:end)=0;
end

function out=extract_squares(val,n,depth)
% Recursively handle ADMM-GP A100 struct/cell variants.
out={};if depth<0,return;end
if isnumeric(val) || islogical(val)
    if ismatrix(val) && isequal(size(val),[n,n])
        out={val};
    elseif ndims(val)==3 && size(val,1)==n && size(val,2)==n
        for ii=1:size(val,3),out{end+1}=val(:,:,ii);end %#ok<AGROW>
    end
elseif iscell(val)
    for i=1:numel(val)
        sub=extract_squares(val{i},n,depth-1);
        out=[out,sub]; %#ok<AGROW>
    end
elseif isstruct(val)
    % Favor adjacency variables (W/A/adj) over metadata.
    for i=1:numel(val)
        keys=fieldnames(val(i));
        priority=contains(lower(string(keys)),{'adj','weight','graph'});
        priority=priority | strcmpi(keys,'A') | strcmpi(keys,'W');
        keys=[keys(priority);keys(~priority)];
        for j=1:numel(keys)
            sub=extract_squares(val(i).(keys{j}),n,depth-1);
            out=[out,sub]; %#ok<AGROW>
        end
    end
end
end
