function test_mmdlp_native_modes()
%TEST_MMDLP_NATIVE_MODES Unit test without Optimization Toolbox.
opts=struct('solution_mode','native_only','support_tol',1e-10);
[p,b,info]=mmdlp_extract_batch([0;1;0;1],2,opts);
assert(isequal(p,[2;4]) && isequal(b,[0;1;0;1]) && info.native_valid);
[~,b2,info2]=mmdlp_extract_batch([1e-12;1-1e-12;0;1],2,opts);
assert(isequal(b2,[0;1;0;1]) && info2.native_valid);
try
    mmdlp_extract_batch([0.5;0.5;1;0],2,opts);
    error('TEST:FractionalAccepted','Fractional LP m was accepted.');
catch ME
    assert(strcmp(ME.identifier,'MMDLP:NativeBinaryRequired'),ME.message);
end
try
    mmdlp_extract_batch([1;1;1;0],2,opts);
    error('TEST:WrongCardinalityAccepted','Wrong-cardinality m was accepted.');
catch ME
    assert(strcmp(ME.identifier,'MMDLP:NativeBinaryRequired'),ME.message);
end
opts.solution_mode='paper_topk';
[p,b,info]=mmdlp_extract_batch([0.5;0.5;1;0],2,opts);
assert(isequal(p,[3;1]) && isequal(b,[1;0;1;0]) && info.used_paper_topk);
% Positive scaling must not alter binary group objective after rescaling.
L=[4 -1 -1; -1 3 -1; -1 -1 3];
s=max(1,norm(L,inf));
Z=0.5*L/s; v=[1;1;0];
assert(abs(s*(v'*Z*v) - 0.5*v'*L*v)<1e-12);
fprintf('test_mmdlp_native_modes: PASS\n');
end
