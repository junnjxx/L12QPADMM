function test_graph_core()
%TEST_GRAPH_CORE Basic algebra and balance checks on a known simple graph.
% Two disjoint 3-cliques; their intended equipartition has cut 0.
n=6;W=zeros(n);
for S={1:3,4:6}
    a=S{1};W(a,a)=ones(3)-eye(3);
end
p=build_graph_problem(W,2);
assert(norm(p.L-p.L','fro')<1e-12);
assert(abs(graph_spectral_norm(sparse(p.L)) - norm(full(p.L),2)) < 1e-9);
assert(norm(p.L*ones(n,1),inf)<1e-12);
P=labels_to_assignment([1;1;1;2;2;2],2);
[J,valid]=graph_cut(P,W);assert(valid && abs(J)<1e-12);
P2=labels_to_assignment([1;2;1;2;1;2],2);
[J2,valid]=graph_cut(P2,W);assert(valid && J2==4);
[tP,tJ]=graph_two_opt(P2,W,struct('max_passes',10,'improve_tol',1e-12));
assert(tJ<=J2+1e-12);
assert(graph_cut(tP,W)==tJ);
% Regression test for stale 'a=labels(i)' in nested swap loops.
% Old version accepted i=1,j=3 and later i=1,j=4 while still using
% its original group a=1, causing column sums [3,1] instead of [2,2].
Wbug = [0 0 1 1; 0 0 6 4; 1 6 0 3; 1 4 3 0];
Pbug = labels_to_assignment([1;1;2;2],2);
[Jbug_before,okbug] = graph_cut(Pbug,Wbug);
assert(okbug);
[Pbug_after,Jbug_after,nswapbug] = graph_two_opt( ...
    Pbug,Wbug,struct('max_passes',4,'improve_tol',1e-12));
[Jbug_check,okbug] = graph_cut(Pbug_after,Wbug);
assert(okbug,'2-opt changed a balanced equipartition into an infeasible one');
assert(nswapbug>=1,'Regression graph did not exercise a swap');
assert(isequal(sum(Pbug_after,1),[2,2]));
assert(all(sum(Pbug_after,2)==1));
assert(abs(Jbug_check-Jbug_after)<1e-10);
assert(Jbug_after<=Jbug_before+1e-10);
% Stress check: repeated valid random starting partitions.
for trial=1:12
    P0=random_equipartition(4,2,trial+11000);
    [J0,v0]=graph_cut(P0,Wbug);assert(v0);
    [Pnew,Jnew]=graph_two_opt(P0,Wbug, ...
        struct('max_passes',5,'improve_tol',1e-12));
    [Jcheck,validnew]=graph_cut(Pnew,Wbug);
    assert(validnew && abs(Jcheck-Jnew)<1e-10 && Jnew<=J0+1e-10);
end
At = kequi_form_graph(W,2);
assert(isequal(size(At),[n*n,2*n]));
H=P*P';b=[ones(n,1);3*ones(n,1)];
assert(norm(At'*H(:)-b,inf)<1e-12);
Pr=random_equipartition(n,2,902);
assert(all(sum(Pr,1)==3)&&all(sum(Pr,2)==1));
X=ones(n,2)/2;
Pround=round_balanced_graph(X);
assert(all(sum(Pround,1)==3)&&all(sum(Pround,2)==1));
fprintf('All graph core smoke tests passed.\n');
end
