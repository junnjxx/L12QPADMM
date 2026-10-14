function [pick,m_relaxed,m_binary,info] = mmdlp_select_batch(Z,k,opts)
%MMDLP_SELECT_BATCH Original AAAI-21 LP with selectable extraction.
% solution_mode='native_only': require LP m itself to be binary; fail otherwise.
% solution_mode='paper_topk': retain the paper's top-k rounding baseline.
% No ADMM parameters (eta/beta) belong to this LP method.
if nargin<3 || isempty(opts), opts=struct(); end
if ~isfield(opts,'solution_mode') || isempty(opts.solution_mode)
    opts.solution_mode='native_only';
end
if ~isfield(opts,'support_tol') || isempty(opts.support_tol)
    opts.support_tol=1e-10;
end
[m_relaxed,info]=mmdlp_solve_relaxation(Z,k,opts);
extract_tic=tic;
[pick,m_binary,selection]=mmdlp_extract_batch(m_relaxed,k,opts);
info.selection_time=toc(extract_tic);
info.solution_mode=selection.solution_mode;
info.binary_deviation=selection.binary_deviation;
info.native_valid=selection.native_valid;
info.used_paper_topk=selection.used_paper_topk;
info.rounding_cardinality=nnz(m_binary);
% Backward compatible: 0 in native-only mode, actual top-k extraction cost
% in paper mode. Count selection_time separately for either mode.
if strcmp(selection.solution_mode,'paper_topk')
    info.rounding_time=info.selection_time;
else
    info.rounding_time=0;
end
obj_tic=tic;
info.relaxation_objective=info.fval;
info.rounded_iqp_objective=m_binary'*Z*m_binary;
info.rounded_objective_eval_time=toc(obj_tic);
end
