function data = load_mnist_small1000(cfg,project_root)
% Exactly 100 distinct training images for each digit; deterministic sampling.
    root = char(string(cfg.data.mnist_dir));
    if isempty(root), root = fullfile(project_root,'data','mnist_raw'); end
    if ~exist(root,'dir'), mkdir(root); end
    names = {'train-images-idx3-ubyte','train-labels-idx1-ubyte'};
    base = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
    for j=1:2
        raw = fullfile(root,names{j});
        if ~exist(raw,'file')
            gz = [raw,'.gz'];
            if ~exist(gz,'file')
                fprintf('Downloading MNIST official training IDX: %s\n',names{j});
                try
                    websave(gz,[base,names{j},'.gz']);
                catch ME
                    error('MNIST download failed: %s. Place %s (uncompressed IDX) in %s.', ...
                        ME.message,names{j},root);
                end
            end
            gunzip(gz,root);
        end
    end
    fid=fopen(fullfile(root,names{1}),'rb','ieee-be');
    if fid<0, error('Cannot read MNIST images'); end
    c=onCleanup(@()fclose(fid));
    magic=fread(fid,1,'uint32'); n=fread(fid,1,'uint32');
    nr=fread(fid,1,'uint32'); nc=fread(fid,1,'uint32');
    assert(magic==2051 && nr==28 && nc==28,'Unexpected MNIST image header');
    raw=fread(fid,[nr*nc,n],'uint8=>uint8');
    assert(numel(raw)==nr*nc*n,'Incomplete MNIST images file');
    clear c;
    fid=fopen(fullfile(root,names{2}),'rb','ieee-be');
    if fid<0, error('Cannot read MNIST labels'); end
    c=onCleanup(@()fclose(fid));
    magic=fread(fid,1,'uint32'); nlabels=fread(fid,1,'uint32');
    labels=fread(fid,nlabels,'uint8=>uint8');
    assert(magic==2049 && nlabels==n && numel(labels)==n,'Unexpected MNIST label file');
    clear c;
    rng(cfg.seed.data,'twister');
    idx=zeros(1000,1); digits=zeros(1000,1);
    for d=0:9
        candidates=find(labels==d);
        assert(numel(candidates)>=100,'Not enough images for digit %d',d);
        selected=candidates(randperm(numel(candidates),100));
        take=d*100+(1:100); idx(take)=selected; digits(take)=d;
    end
    X=double(raw(:,idx))';
    if cfg.data.mnist_normalize, X=X/255; end
    data=struct();
    data.sample_stationary_points=X;
    data.group_label=digits;
    data.mnist_train_indices=idx;
    data.mnist_labels=digits;
    data.num_samples=1000; data.dim_x=784; data.num_groups=10;
    fprintf('MNIST small1000: 100 per digit, 1000x784, seed=%d, normalize=%d\n', ...
        cfg.seed.data,cfg.data.mnist_normalize);
end
