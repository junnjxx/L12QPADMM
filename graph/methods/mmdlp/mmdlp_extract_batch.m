function [pick,m_binary,info] = mmdlp_extract_batch(m,k,opts)
%MMDLP_EXTRACT_BATCH Native-only selection or original paper top-k rounding.
% IMPORTANT: This function never solves an LP. It only interprets its m.
% 'native_only' refuses fractional LP solutions rather than repairing them.
% 'paper_topk' preserves the original relaxed LP + top-k baseline.
if nargin<3 || isempty(opts), opts=struct(); end
if ~isfield(opts,'solution_mode') || isempty(opts.solution_mode)
    opts.solution_mode='native_only';
end
if ~isfield(opts,'support_tol') || isempty(opts.support_tol)
    opts.support_tol=1e-10;
end
mode=lower(char(opts.solution_mode));
m=m(:);
q=numel(m);
if isempty(m) || any(~isfinite(m)) || k<1 || k>q || k~=round(k)
    error('MMDLP:InvalidBatchInput','Nonfinite m or invalid cardinality k.');
end
if ~isscalar(opts.support_tol) || ~isfinite(opts.support_tol) || ...
        opts.support_tol<=0 || opts.support_tol>=0.5
    error('MMDLP:InvalidSupportTol','support_tol must be in (0,0.5).');
end
info=struct();
info.solution_mode=mode;
info.binary_deviation=max(abs(m-round(m)));
info.native_valid=(info.binary_deviation<=opts.support_tol && ...
    sum(round(m))==k);
info.used_paper_topk=false;
switch mode
    case 'native_only'
        if ~info.native_valid
            error('MMDLP:NativeBinaryRequired', ...
                ['LP m is not a native binary vector with cardinality %d: ', ...
                 'nnz(round(m))=%d, fractional_deviation=%.3e, ', ...
                 'support_tol=%.3e. No top-k fallback is allowed.'], ...
                k,nnz(round(m)),info.binary_deviation,opts.support_tol);
        end
        % round() removes only solver-level numerical noise after validation.
        % This is NOT top-k rounding or a repair of a genuinely fractional m.
        m_binary=round(m);
        pick=find(m_binary==1);
    case 'paper_topk'
        % Original AAAI-21 selection: descending relaxed m, tie by index.
        [~,order]=sortrows([-m,(1:q)'],[1 2]);
        pick=order(1:k);
        m_binary=zeros(q,1);
        m_binary(pick)=1;
        info.used_paper_topk=~info.native_valid;
    otherwise
        error('MMDLP:UnknownMode', ...
            'solution_mode must be native_only or paper_topk; got %s.',mode);
end
if numel(pick)~=k || nnz(m_binary)~=k || any(m_binary~=0 & m_binary~=1)
    error('MMDLP:InvalidSelection','Batch extraction failed cardinality/binary check.');
end
end
