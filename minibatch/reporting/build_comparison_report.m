function report = build_comparison_report(results, problem, cfg)
%BUILD_COMPARISON_REPORT Build all cross-method tables on the J_common scale.
%
% IMPORTANT:
% Every method row, every lower bound, every empirical gap, and every certified
% gap in this report uses
%
%   J_common(P) = <Phi,P*P'>/(2*bs^2).
%
% eADMM's graph/Laplacian objective is an internal solver representation only
% and never appears in the comparison tables.

    M = results.matrix;
    L = results.lp;
    LA = L.alg2;
    LB = L.bs;
    R = results.random;
    E = results.eadmm;
    vid = M.valid_ids;
    matrix_warm_overhead = M.warmstart_overhead_time;

    % Matrix-only diagnostic: on any final balanced binary partition, the full
    % Matrix objective differs from J_common by one fixed constant.
    C_linear = problem.matrix_linear_constant;
    C_lhalf = problem.matrix_lhalf_constant;
    C_total = problem.matrix_constant_offset;

    errM = max(abs(M.full_obj(vid) - (M.raw_jcommon(vid)+C_total)));
    errR = max(abs(R.full_obj - (R.raw_jcommon+C_total)));
    if errM > 1e-9 || errR > 1e-9
        error('Matrix full-objective decomposition check failed: Matrix %.3e, Random %.3e.', ...
            errM,errR);
    end

    Method = {};
    Stage = {};
    JCommon = [];
    SolveTimeSec = [];
    InitExtractTimeSec = [];
    LocalSearchTimeSec = [];
    TotalMethodTimeSec = [];

    msingle = vid(1);
    mbest = M.best_raw_id;
    if M.warmstart.is_external
        matrix_single_name = 'Matrix single (external warm start)';
    else
        matrix_single_name = sprintf('Matrix single (seed %d)',M.matrix_init_seed(msingle));
    end
    addrow(matrix_single_name,'raw', ...
        M.raw_jcommon(msingle),M.solve_time(msingle), ...
        matrix_warm_overhead+M.extract_time(msingle),0, ...
        matrix_warm_overhead+M.solve_time(msingle)+M.extract_time(msingle));
    if cfg.two_opt.enabled
        addrow('Matrix single +2opt','same 2opt',M.refined_jcommon(msingle), ...
            M.solve_time(msingle),matrix_warm_overhead+M.extract_time(msingle), ...
            M.local_time(msingle), ...
            matrix_warm_overhead+M.solve_time(msingle)+M.extract_time(msingle)+M.local_time(msingle));
    end
    addrow(sprintf('Matrix best-of-%d',numel(vid)),'raw',M.raw_jcommon(mbest), ...
        sum(M.solve_time(vid)),matrix_warm_overhead+sum(M.extract_time(vid)),0, ...
        matrix_warm_overhead+sum(M.solve_time(vid)+M.extract_time(vid)));
    if cfg.two_opt.enabled
        mid2=M.best_refined_id;
        addrow(sprintf('Matrix best-of-%d +2opt',numel(vid)),'same 2opt', ...
            M.refined_jcommon(mid2),sum(M.solve_time(vid)), ...
            matrix_warm_overhead+sum(M.extract_time(vid)), ...
            sum(M.local_time(vid)), ...
            matrix_warm_overhead+sum(M.solve_time(vid)+M.extract_time(vid)+M.local_time(vid)));
    end

    if LA.enabled
        add_lp_variant_rows(LA,'Lp-Alg2','Algorithm 2');
    end
    if LB.enabled
        add_lp_variant_rows(LB,'Lp-bs','basic/no internal N2');
    end

    addrow('Random balanced single','raw',R.raw_jcommon(1),0,R.gen_time(1),0,R.gen_time(1));
    if cfg.two_opt.enabled
        addrow('Random balanced single +2opt','same 2opt',R.refined_jcommon(1),0, ...
            R.gen_time(1),R.local_time(1),R.gen_time(1)+R.local_time(1));
    end
    rb=R.best_raw_id;
    addrow(sprintf('Random balanced best-of-%d',numel(R.raw_jcommon)),'raw',R.raw_jcommon(rb),0, ...
        sum(R.gen_time),0,sum(R.gen_time));
    if cfg.two_opt.enabled
        r2=R.best_refined_id;
        addrow(sprintf('Random balanced best-of-%d +2opt',numel(R.raw_jcommon)),'same 2opt', ...
            R.refined_jcommon(r2),0,sum(R.gen_time),sum(R.local_time),R.wall_time);
    end

    % eADMM rounded partitions are evaluated by the SAME evaluate_partition()
    % function as Matrix and Random, hence the following values are J_common.
    addrow('eADMM+Vc (1)','raw',E.vc_jcommon(1),E.solve_time, ...
        E.cumulative_vc_time(1),0, ...
        E.solve_time+E.cumulative_vc_time(1));
    if cfg.eadmm.use_2opt
        addrow('eADMM+Vc+2opt (1)','same 2opt',E.vc2_jcommon(1),E.solve_time, ...
            E.cumulative_vc_time(1),E.cumulative_2opt_time(1), ...
            E.solve_time+E.cumulative_vc_time(1)+E.cumulative_2opt_time(1));
    end
    addrow(sprintf('eADMM+Vc (%d)',cfg.eadmm.rounding_restarts),'raw',E.best_vc_jcommon, ...
        E.solve_time,E.vc_time,0,E.solution_time_raw);
    if cfg.eadmm.use_2opt
        addrow(sprintf('eADMM+Vc+2opt (%d)',cfg.eadmm.rounding_restarts),'same 2opt', ...
            E.best_vc2_jcommon,E.solve_time,E.vc_time,E.two_opt_time,E.solution_time_with_2opt);
    end

    best_discrete = min(JCommon);
    GapToBestDiscretePct = 100*(JCommon-best_discrete)/max(abs(best_discrete),eps);

    % Certified optimality gaps: safe lower bound is already on J_common scale.
    safe_lb = E.safe_jcommon_lower_bound;
    lb_valid = isfinite(safe_lb) && E.safe_lower_bound_numerically_feasible;
    if isfinite(safe_lb)
        lb_consistency_tol = 1e-8*max([1,abs(best_discrete),abs(safe_lb)]);
    else
        lb_consistency_tol = NaN;
    end
    if lb_valid && safe_lb > best_discrete + lb_consistency_tol
        warning(['Safe J_common lower bound %.12e exceeds the best feasible ', ...
            'J_common %.12e by more than tolerance. Certified gaps are suppressed.'], ...
            safe_lb,best_discrete);
        lb_valid = false;
    end

    if lb_valid
        CertifiedAbsGap = JCommon-safe_lb;
        CertifiedAbsGap(CertifiedAbsGap<0 & CertifiedAbsGap>=-lb_consistency_tol)=0;
        CertifiedGapPct = 100*CertifiedAbsGap./max(abs(JCommon),eps);
    else
        CertifiedAbsGap = nan(size(JCommon));
        CertifiedGapPct = nan(size(JCommon));
    end

    summary_table = table(Method,Stage,JCommon,CertifiedAbsGap,CertifiedGapPct, ...
        GapToBestDiscretePct,SolveTimeSec,InitExtractTimeSec, ...
        LocalSearchTimeSec,TotalMethodTimeSec);

    % Distribution summary, entirely on J_common. Lp-Alg2 already includes
    % its published internal N2 refinement; Lp-bs does not.
    Family = {'Matrix'};
    NumStarts = numel(vid);
    RawMeanJCommon = mean(M.raw_jcommon(vid));
    RawStdJCommon = std(M.raw_jcommon(vid));
    RawBestJCommon = min(M.raw_jcommon(vid));
    After2optMeanJCommon = mean(M.refined_jcommon(vid));
    After2optStdJCommon = std(M.refined_jcommon(vid));
    After2optBestJCommon = min(M.refined_jcommon(vid));

    if LA.enabled
        [Family,NumStarts,RawMeanJCommon,RawStdJCommon,RawBestJCommon, ...
            After2optMeanJCommon,After2optStdJCommon,After2optBestJCommon] = ...
            append_family(Family,NumStarts,RawMeanJCommon,RawStdJCommon,RawBestJCommon, ...
            After2optMeanJCommon,After2optStdJCommon,After2optBestJCommon, ...
            'Lp Algorithm 2',LA,cfg.lp.use_common_2opt);
    end
    if LB.enabled
        [Family,NumStarts,RawMeanJCommon,RawStdJCommon,RawBestJCommon, ...
            After2optMeanJCommon,After2optStdJCommon,After2optBestJCommon] = ...
            append_family(Family,NumStarts,RawMeanJCommon,RawStdJCommon,RawBestJCommon, ...
            After2optMeanJCommon,After2optStdJCommon,After2optBestJCommon, ...
            'Lp-bs',LB,cfg.lp.use_common_2opt);
    end

    Family{end+1,1}='Random balanced';
    NumStarts(end+1,1)=numel(R.raw_jcommon);
    RawMeanJCommon(end+1,1)=mean(R.raw_jcommon);
    RawStdJCommon(end+1,1)=std(R.raw_jcommon);
    RawBestJCommon(end+1,1)=min(R.raw_jcommon);
    if cfg.two_opt.enabled
        After2optMeanJCommon(end+1,1)=mean(R.refined_jcommon);
        After2optStdJCommon(end+1,1)=std(R.refined_jcommon);
        After2optBestJCommon(end+1,1)=min(R.refined_jcommon);
    else
        After2optMeanJCommon(end+1,1)=NaN;
        After2optStdJCommon(end+1,1)=NaN;
        After2optBestJCommon(end+1,1)=NaN;
    end

    Family{end+1,1}='eADMM Vc';
    NumStarts(end+1,1)=numel(E.vc_jcommon);
    RawMeanJCommon(end+1,1)=mean(E.vc_jcommon);
    RawStdJCommon(end+1,1)=std(E.vc_jcommon);
    RawBestJCommon(end+1,1)=min(E.vc_jcommon);
    if cfg.eadmm.use_2opt
        After2optMeanJCommon(end+1,1)=mean(E.vc2_jcommon);
        After2optStdJCommon(end+1,1)=std(E.vc2_jcommon);
        After2optBestJCommon(end+1,1)=min(E.vc2_jcommon);
    else
        After2optMeanJCommon(end+1,1)=NaN;
        After2optStdJCommon(end+1,1)=NaN;
        After2optBestJCommon(end+1,1)=NaN;
    end

    if lb_valid
        RawBestCertifiedGapPct = 100*(RawBestJCommon-safe_lb)./max(abs(RawBestJCommon),eps);
        After2optBestCertifiedGapPct = ...
            100*(After2optBestJCommon-safe_lb)./max(abs(After2optBestJCommon),eps);
    else
        RawBestCertifiedGapPct=nan(size(RawBestJCommon));
        After2optBestCertifiedGapPct=nan(size(After2optBestJCommon));
    end
    family_table=table(Family,NumStarts,RawMeanJCommon,RawStdJCommon,RawBestJCommon, ...
        RawBestCertifiedGapPct,After2optMeanJCommon,After2optStdJCommon, ...
        After2optBestJCommon,After2optBestCertifiedGapPct);

    % Equal-time diagnostic for complete +2opt candidates.
    BudgetSec=cfg.reporting.time_budgets(:);
    MatrixBestJCommon=nan(size(BudgetSec));
    LpAlg2BestJCommon=nan(size(BudgetSec));
    LpBsBestJCommon=nan(size(BudgetSec));
    eADMMBestJCommon=nan(size(BudgetSec));
    RandomBestJCommon=nan(size(BudgetSec));
    if cfg.two_opt.enabled && cfg.eadmm.use_2opt
        matrix_candidate_times=matrix_warm_overhead + ...
            cumsum(M.solve_time(vid)+M.extract_time(vid)+M.local_time(vid));
        matrix_candidate_best=cummin(M.refined_jcommon(vid));
        random_candidate_times=cumsum(R.gen_time+R.local_time);
        random_candidate_best=cummin(R.refined_jcommon);
        eadmm_candidate_times=E.solve_time+E.cumulative_vc_time+E.cumulative_2opt_time;
        eadmm_candidate_best=E.vc2_best_jcommon_history;
        if LA.enabled && cfg.lp.use_common_2opt
            laids=LA.valid_ids;
            la_times=cumsum(LA.solve_time(laids)+LA.local_time(laids));
            la_best=cummin(LA.refined_jcommon(laids));
        end
        if LB.enabled && cfg.lp.use_common_2opt
            lbids=LB.valid_ids;
            lb_times=cumsum(LB.solve_time(lbids)+LB.local_time(lbids));
            lb_best=cummin(LB.refined_jcommon(lbids));
        end
        for q=1:numel(BudgetSec)
            MatrixBestJCommon(q)=best_at_time_budget(matrix_candidate_times,matrix_candidate_best,BudgetSec(q));
            eADMMBestJCommon(q)=best_at_time_budget(eadmm_candidate_times,eadmm_candidate_best,BudgetSec(q));
            RandomBestJCommon(q)=best_at_time_budget(random_candidate_times,random_candidate_best,BudgetSec(q));
            if LA.enabled && cfg.lp.use_common_2opt
                LpAlg2BestJCommon(q)=best_at_time_budget(la_times,la_best,BudgetSec(q));
            end
            if LB.enabled && cfg.lp.use_common_2opt
                LpBsBestJCommon(q)=best_at_time_budget(lb_times,lb_best,BudgetSec(q));
            end
        end
    end
    time_budget_table=table(BudgetSec,MatrixBestJCommon,LpAlg2BestJCommon,LpBsBestJCommon, ...
        eADMMBestJCommon,RandomBestJCommon);

    % Solver-level iteration timing / N-time normalization.
    TimingMethod = {'Matrix (mean valid starts)'};
    ProblemN = problem.num_samples;
    Iterations = sum(M.iterations(vid));
    SolveTimeSecTiming = sum(M.solve_time(vid));
    NumSolves = numel(vid);
    AvgTimePerIterSec = sum(M.solve_time(vid))/max(sum(M.iterations(vid)),1);

    if LA.enabled
        ids=LA.valid_ids;
        TimingMethod{end+1,1}='Lp Algorithm 2';
        ProblemN(end+1,1)=problem.num_samples;
        Iterations(end+1,1)=sum(LA.inner_iterations(ids));
        SolveTimeSecTiming(end+1,1)=sum(LA.solve_time(ids));
        NumSolves(end+1,1)=numel(ids);
        AvgTimePerIterSec(end+1,1)=sum(LA.solve_time(ids))/max(sum(LA.inner_iterations(ids)),1);
    end
    if LB.enabled
        ids=LB.valid_ids;
        TimingMethod{end+1,1}='Lp-bs';
        ProblemN(end+1,1)=problem.num_samples;
        Iterations(end+1,1)=sum(LB.inner_iterations(ids));
        SolveTimeSecTiming(end+1,1)=sum(LB.solve_time(ids));
        NumSolves(end+1,1)=numel(ids);
        AvgTimePerIterSec(end+1,1)=sum(LB.solve_time(ids))/max(sum(LB.inner_iterations(ids)),1);
    end
    TimingMethod{end+1,1}='eADMM DNN';
    ProblemN(end+1,1)=problem.num_samples;
    Iterations(end+1,1)=E.dnn_iterations;
    SolveTimeSecTiming(end+1,1)=E.solve_time;
    NumSolves(end+1,1)=1;
    AvgTimePerIterSec(end+1,1)=E.dnn_avg_iter_time;

    MeanSolveTimeSec = SolveTimeSecTiming ./ NumSolves;
    TimePerNSec = MeanSolveTimeSec ./ ProblemN;
    NPerSec = ProblemN ./ max(MeanSolveTimeSec,eps);
    iteration_timing_table = table(TimingMethod,ProblemN,NumSolves,Iterations, ...
        MeanSolveTimeSec,AvgTimePerIterSec,TimePerNSec,NPerSec);

    % Detailed Lp component profiling. Times are aggregated over valid starts.
    LpVariant={}; TotalTimeSec=[]; ProjectionTimeSec=[]; RegEvalTimeSec=[];
    LineSearchTimeSec=[]; RoundingTimeSec=[]; InternalN2TimeSec=[];
    ObjectiveOnlyTimeSec=[]; OtherTimeSec=[]; ProjectionCalls=[]; DualBBIterations=[];
    RoundingCalls=[]; N2Calls=[]; N2Swaps=[]; BacktrackSteps=[];
    if LA.enabled
        append_lp_profile('Lp Algorithm 2',LA.profile_sum);
    end
    if LB.enabled
        append_lp_profile('Lp-bs',LB.profile_sum);
    end
    lp_component_timing_table=table(LpVariant,TotalTimeSec,ProjectionTimeSec,RegEvalTimeSec, ...
        LineSearchTimeSec,RoundingTimeSec,InternalN2TimeSec,ObjectiveOnlyTimeSec,OtherTimeSec, ...
        ProjectionCalls,DualBBIterations,RoundingCalls,N2Calls,N2Swaps,BacktrackSteps);

    report=struct();
    report.summary_table=summary_table;
    report.family_table=family_table;
    report.time_budget_table=time_budget_table;
    report.iteration_timing_table=iteration_timing_table;
    report.lp_component_timing_table=lp_component_timing_table;

    % Matrix-only decomposition diagnostics.
    report.C_linear=C_linear;
    report.C_lhalf=C_lhalf;
    report.C_total=C_total;
    report.matrix_warmstart_overhead_time=M.warmstart_overhead_time;
    report.matrix_warmstart_source_time=M.warmstart_source_time;
    report.matrix_warmstart_conversion_time=M.warmstart_conversion_time;
    report.matrix_warmstart_label=M.warmstart_label;

    % Cross-method quantities: J_common scale only.
    report.best_feasible_jcommon=best_discrete;
    report.sdp_jcommon_score=E.sdp_jcommon_score;
    report.safe_jcommon_lower_bound=safe_lb;
    % Timing: method comparison excludes MET/LB certification.
    report.eadmm_solution_time_raw=E.solution_time_raw;
    report.eadmm_solution_time_with_2opt=E.solution_time_with_2opt;
    report.certification_met_time=E.certification_met_time;
    report.safe_lower_bound_postprocess_time=E.safe_lower_bound_postprocess_time;
    report.certification_additional_time=E.certification_additional_time;
    report.certification_from_scratch_time=E.certification_from_scratch_time;
    report.total_time_with_certification=E.total_time_with_certification;
    % Backward-compatible alias.
    report.safe_lower_bound_time=E.safe_lower_bound_time;
    report.safe_lower_bound_valid=lb_valid;
    report.safe_lower_bound_numerically_feasible=E.safe_lower_bound_numerically_feasible;
    report.safe_lower_bound_min_eig_Z=E.safe_lower_bound_min_eig_Z;
    report.safe_lower_bound_min_slack=E.safe_lower_bound_min_slack;
    report.safe_lower_bound_dual_identity_rel=E.safe_lower_bound_dual_identity_rel;

    if lb_valid
        report.best_certified_abs_gap=best_discrete-safe_lb;
        if report.best_certified_abs_gap<0 && ...
                report.best_certified_abs_gap>=-lb_consistency_tol
            report.best_certified_abs_gap=0;
        end
        report.best_certified_gap_pct= ...
            100*report.best_certified_abs_gap/max(abs(best_discrete),eps);
    else
        report.best_certified_abs_gap=NaN;
        report.best_certified_gap_pct=NaN;
    end

    if all(isfinite(After2optBestJCommon))
        report.post_best_spread_pct=100*(max(After2optBestJCommon)-min(After2optBestJCommon)) / ...
            max(abs(min(After2optBestJCommon)),eps);
        report.post_mean_spread_pct=100*(max(After2optMeanJCommon)-min(After2optMeanJCommon)) / ...
            max(abs(min(After2optMeanJCommon)),eps);
    else
        report.post_best_spread_pct=NaN;
        report.post_mean_spread_pct=NaN;
    end

    function add_lp_variant_rows(V,prefix,stage_name)
        ids=V.valid_ids;
        single=ids(1);
        best=V.best_raw_id;
        addrow(sprintf('%s single (seed %d)',prefix,V.perturb_seed(single)), ...
            stage_name,V.raw_jcommon(single),V.solve_time(single),0,0,V.solve_time(single));
        if cfg.lp.use_common_2opt
            addrow(sprintf('%s single +common2opt',prefix),'same 2opt', ...
                V.refined_jcommon(single),V.solve_time(single),0,V.local_time(single), ...
                V.solve_time(single)+V.local_time(single));
        end
        addrow(sprintf('%s best-of-%d',prefix,numel(ids)),stage_name, ...
            V.raw_jcommon(best),sum(V.solve_time(ids)),0,0,sum(V.solve_time(ids)));
        if cfg.lp.use_common_2opt
            id2=V.best_refined_id;
            addrow(sprintf('%s best-of-%d +common2opt',prefix,numel(ids)),'same 2opt', ...
                V.refined_jcommon(id2),sum(V.solve_time(ids)),0,sum(V.local_time(ids)), ...
                sum(V.solve_time(ids)+V.local_time(ids)));
        end
    end

    function [Fam,NS,RM,RS,RB,AM,AS,AB] = append_family(Fam,NS,RM,RS,RB,AM,AS,AB,name,V,use2)
        ids=V.valid_ids;
        Fam{end+1,1}=name;
        NS(end+1,1)=numel(ids);
        RM(end+1,1)=mean(V.raw_jcommon(ids));
        RS(end+1,1)=std(V.raw_jcommon(ids));
        RB(end+1,1)=min(V.raw_jcommon(ids));
        if use2
            AM(end+1,1)=mean(V.refined_jcommon(ids));
            AS(end+1,1)=std(V.refined_jcommon(ids));
            AB(end+1,1)=min(V.refined_jcommon(ids));
        else
            AM(end+1,1)=NaN; AS(end+1,1)=NaN; AB(end+1,1)=NaN;
        end
    end

    function append_lp_profile(name,p)
        LpVariant{end+1,1}=name;
        TotalTimeSec(end+1,1)=getp(p,'total_time');
        ProjectionTimeSec(end+1,1)=getp(p,'projection_time');
        RegEvalTimeSec(end+1,1)=getp(p,'regularized_eval_time');
        LineSearchTimeSec(end+1,1)=getp(p,'line_search_time');
        RoundingTimeSec(end+1,1)=getp(p,'rounding_time');
        InternalN2TimeSec(end+1,1)=getp(p,'internal_local_search_time');
        ObjectiveOnlyTimeSec(end+1,1)=getp(p,'objective_only_time');
        OtherTimeSec(end+1,1)=getp(p,'other_time');
        ProjectionCalls(end+1,1)=getp(p,'projection_calls');
        DualBBIterations(end+1,1)=getp(p,'projection_inner_iterations');
        RoundingCalls(end+1,1)=getp(p,'rounding_calls');
        N2Calls(end+1,1)=getp(p,'local_search_calls');
        N2Swaps(end+1,1)=getp(p,'local_swaps');
        BacktrackSteps(end+1,1)=getp(p,'backtrack_steps');
    end

    function x=getp(p,name)
        if isfield(p,name), x=p.(name); else, x=NaN; end
    end

    function addrow(name,stage,jcommon,solve,init,local,total)
        Method{end+1,1}=name;
        Stage{end+1,1}=stage;
        JCommon(end+1,1)=jcommon;
        SolveTimeSec(end+1,1)=solve;
        InitExtractTimeSec(end+1,1)=init;
        LocalSearchTimeSec(end+1,1)=local;
        TotalMethodTimeSec(end+1,1)=total;
    end
end
