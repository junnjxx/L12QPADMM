function result=run_eadmm_graph_full(problem,cfg)
%RUN_EADMM_GRAPH_FULL Original ADMM-GP SDP+rounding+optional MET/LB.
opt=cfg.eadmm;
opt.rounding_seed_base=cfg.seed.eadmm_rounding_base;
opt.two_opt_seed_base=cfg.seed.two_opt_base;
opt.two_opt_tol=cfg.two_opt.cost_tol;
opt.eval_eta=cfg.matrix.eta;
result=solve_eadmm_batching(problem,opt);
end
