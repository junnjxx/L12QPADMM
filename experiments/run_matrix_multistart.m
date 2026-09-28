function exp_result = run_matrix_multistart(problem,cfg,prior_results)
%RUN_MATRIX_MULTISTART Run Matrix after Random/Lp/eADMM baselines.
if nargin<3, prior_results=struct(); end
if ~isfield(cfg.matrix,'matrix_clean') || isempty(cfg.matrix.matrix_clean), cfg.matrix.matrix_clean=false; end
if ~isfield(cfg.matrix,'tau_X') || isempty(cfg.matrix.tau_X), error('cfg.matrix.tau_X is required for paper-aligned Matrix ADMM.'); end
matrix_clean=logical(cfg.matrix.matrix_clean);
matrix_init=prepare_matrix_initialization(problem,cfg,prior_results);
if ~matrix_clean
    fprintf('Matrix 初始化选择 = %s。\n',char(string(cfg.matrix.init_mode)));
    fprintf('Matrix paper-aligned tau_X = %.6g。\n',cfg.matrix.tau_X);
    if matrix_init.is_external
        fprintf('Matrix 外部 warm start：%s。\n',char(string(matrix_init.label)));
        fprintf('warm start 平衡残差：row=%.3e，col=%.3e；取值范围=[%.3e, %.3e]。\n',...
            matrix_init.details.row_res_inf,matrix_init.details.col_res_inf,matrix_init.details.min_entry,matrix_init.details.max_entry);
        fprintf('warm start 上游生成时间 = %.6f 秒；转换时间 = %.6f 秒。\n',...
            matrix_init.source_generation_time,matrix_init.conversion_time);
        if isfield(matrix_init.details,'selection_score_after_rounding') && isfinite(matrix_init.details.selection_score_after_rounding)
            fprintf('*LP warm-start 选择依据：全局最好 J_common = %.12g*。\n',matrix_init.details.selection_score_after_rounding);
            if isfield(matrix_init.details,'chosen_outer') && isfield(matrix_init.details,'chosen_inner')
                fprintf('对应位置：outer=%d，inner=%d；使用的是该候选 rounding 前的连续解。\n',...
                    matrix_init.details.chosen_outer,matrix_init.details.chosen_inner);
            end
        end
    end
end
R=cfg.matrix.multistart.num_starts;
if matrix_init.is_external && R>1
    if ~matrix_clean
        fprintf('外部 warm start 是固定的连续矩阵，重复 multistart 不会产生不同初值；本次 Matrix 自动只运行 1 个 start。\n');
    end
    R=1;
end
raw=cell(R,1);
refined=cell(R,1);
raw_jcommon=nan(R,1);
full_obj=nan(R,1);
refined_jcommon=nan(R,1);
refined_full_obj=nan(R,1);
valid=false(R,1);
solve_time=nan(R,1);
extract_time=nan(R,1);
local_time=nan(R,1);
iterations=nan(R,1);
avg_iter_time=nan(R,1);
matrix_init_seed=nan(R,1);
two_opt_seed=nan(R,1);
wall_tic=tic;
for rr=1:R
    if matrix_init.is_external
        matrix_seed_rr=NaN;
        matrix_init_seed(rr)=NaN;
    else
        matrix_seed_rr=cfg.seed.matrix_init_base+rr-1;
        matrix_init_seed(rr)=matrix_seed_rr;
        rng(matrix_seed_rr,'twister');
    end
    opts=cfg.matrix;
    if matrix_init.is_external
        opts.init_mode="provided";
        opts.X0=matrix_init.X0;
        opts.Y0=matrix_init.Y0;
        opts.init_label=matrix_init.label;
    end
    if matrix_clean
        opts.verbose=false;
        opts.matrix_detail=false;
        opts.make_plot=false;
        opts.save_plot=false;
        opts.save_result=false;
    else
        opts.verbose=(rr==1);
    end
    result=solve_matrix_batching(problem,opts);
    raw{rr}=result;
    solve_time(rr)=result.solve_time;
    extract_time(rr)=result.extract_time;
    iterations(rr)=result.iterations;
    avg_iter_time(rr)=result.avg_iter_time;
    valid(rr)=result.metrics.valid;
    if valid(rr)
        raw_jcommon(rr)=result.metrics.common_objective;
        full_obj(rr)=result.metrics.matrix_full_objective;
        if cfg.two_opt.enabled
            two_opt_seed_rr=cfg.seed.two_opt_base+rr-1;
            two_opt_seed(rr)=two_opt_seed_rr;
            opt2.seed=two_opt_seed_rr;
            opt2.cost_tol=cfg.two_opt.cost_tol;
            opt2.verbose=cfg.two_opt.verbose;
            opt2.eta=cfg.matrix.eta;
            r2=apply_common_two_opt(result.I,problem,opt2);
            refined{rr}=r2;
            refined_jcommon(rr)=r2.metrics_after.common_objective;
            refined_full_obj(rr)=r2.metrics_after.matrix_full_objective;
            local_time(rr)=r2.time;
        end
    end
    if cfg.reporting.print_each_matrix_start && ~matrix_clean
        total_rr=solve_time(rr)+extract_time(rr);
        if cfg.two_opt.enabled && isfinite(local_time(rr)), total_rr=total_rr+local_time(rr); end
        if matrix_init.is_external
            fprintf('Matrix 第 %2d/%2d 个初值 | 外部warm start | tau_X=%.6g | 可行=%d | *J_common(raw)=%.10e* | *迭代数=%d* | *每次迭代时间=%.3e 秒* | *求解时间=%.4f 秒* | 离散解提取时间=%.4f 秒',...
                rr,R,cfg.matrix.tau_X,valid(rr),raw_jcommon(rr),iterations(rr),avg_iter_time(rr),solve_time(rr),extract_time(rr));
        else
            fprintf('Matrix 第 %2d/%2d 个初值 | 初始化随机种子=%d | tau_X=%.6g | 可行=%d | *J_common(raw)=%.10e* | *迭代数=%d* | *每次迭代时间=%.3e 秒* | *求解时间=%.4f 秒* | 离散解提取时间=%.4f 秒',...
                rr,R,matrix_seed_rr,cfg.matrix.tau_X,valid(rr),raw_jcommon(rr),iterations(rr),avg_iter_time(rr),solve_time(rr),extract_time(rr));
        end
        if cfg.two_opt.enabled && isfinite(local_time(rr))
            fprintf(' | 2-opt时间=%.4f 秒 | *Matrix自身总时间=%.4f 秒*',local_time(rr),total_rr);
        else
            fprintf(' | *Matrix自身总时间=%.4f 秒*',total_rr);
        end
        if matrix_init.is_external
            fprintf(' | *含warm-start端到端时间=%.4f 秒*',matrix_init.overhead_time+total_rr);
        end
        fprintf('\n');
    end
end
wall_time=toc(wall_tic);
valid_ids=find(valid & isfinite(raw_jcommon));
if isempty(valid_ids), error('All Matrix starts failed to produce a valid balanced partition.'); end
[~,p]=min(raw_jcommon(valid_ids));
best_raw_id=valid_ids(p);
if cfg.two_opt.enabled
    finite_refined=valid_ids(isfinite(refined_jcommon(valid_ids)));
    if isempty(finite_refined)
        best_refined_id=NaN;
    else
        [~,p2]=min(refined_jcommon(finite_refined));
        best_refined_id=finite_refined(p2);
    end
else
    best_refined_id=NaN;
end
exp_result=struct();
exp_result.raw=raw;
exp_result.refined=refined;
exp_result.valid=valid;
exp_result.valid_ids=valid_ids;
exp_result.raw_jcommon=raw_jcommon;
exp_result.full_obj=full_obj;
exp_result.refined_jcommon=refined_jcommon;
exp_result.refined_full_obj=refined_full_obj;
exp_result.solve_time=solve_time;
exp_result.extract_time=extract_time;
exp_result.local_time=local_time;
exp_result.iterations=iterations;
exp_result.avg_iter_time=avg_iter_time;
exp_result.init_mode=string(cfg.matrix.init_mode);
exp_result.tau_X=cfg.matrix.tau_X;
exp_result.paper_aligned=true;
exp_result.warmstart=matrix_init;
exp_result.warmstart_source_time=matrix_init.source_generation_time;
exp_result.warmstart_conversion_time=matrix_init.conversion_time;
exp_result.warmstart_overhead_time=matrix_init.overhead_time;
exp_result.warmstart_label=string(matrix_init.label);
exp_result.matrix_init_seed=matrix_init_seed;
exp_result.two_opt_seed=two_opt_seed;
exp_result.best_raw_id=best_raw_id;
exp_result.best_refined_id=best_refined_id;
exp_result.wall_time=wall_time;
end
