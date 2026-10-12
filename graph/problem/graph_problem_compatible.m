function problem=graph_problem_compatible(p,cfg)
%GRAPH_PROBLEM_COMPATIBLE Original graph objective, with old algorithm field aliases.
% These are graph group/assignment cardinalities, NOT minibatch data.
problem=struct();
problem.problem_type="graph_partition";
problem.W=p.W;
problem.L=p.L;
problem.A=p.L;
problem.G=zeros(p.n,p.k);
problem.Phi=p.L; % legacy local-search matrix; there is NO bs^2 factor
problem.num_samples=p.n;
problem.num_batches=p.k;
problem.batch_size=p.group_size;
problem.group_size=p.group_size;
problem.n=p.n;
problem.k=p.k;
problem.eta=cfg.matrix.eta;
problem.graph_adjacency=p.W;
problem.graph_laplacian=p.L;
problem.matrix_objective_scale=max(1,norm(p.L,inf));
problem.matrix_A=p.L/problem.matrix_objective_scale;
problem.matrix_G=zeros(p.n,p.k);
problem.common_objective_description='0.5*trace(P''*L*P)';
end
