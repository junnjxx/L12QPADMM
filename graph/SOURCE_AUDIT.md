# Source audit — authoritative user upload

**Authoritative supplied source:** `matrix_eadmm_lp_vector_code.zip`

**Reference repository:** https://github.com/junnjxx/L12QPADMM/tree/main

The user-uploaded ZIP, rather than an older snapshot, is the authoritative
reference for LP and eADMM in this standalone package. This avoids the
previous incorrect fallback to archived dual-BB / older N2 implementations.

| Component | File provenance | Graph adaptation |
|---|---|---|
| Lp solver | Latest uploaded `methods/lp/lp_balanced_algorithm2.m` | Unmodified; fed original graph Laplacian objective |
| Lp projection | Latest uploaded `methods/lp/project_balanced_mosek.m` | Unmodified, **MOSEK required** |
| Lp internal N2 | Latest uploaded `methods/lp/local2swap_balanced_quadratic.m` | Unmodified |
| Lp wrapper | Latest uploaded `experiments/run_lp_baseline.m` | Only graph-specific explanatory objective string |
| eADMM main | Latest uploaded `methods/eadmm/solve_eadmm_batching.m` | Direct W input, graph-cut scale, no minibatch affine scale; same DNN/MET/rounding algorithms |
| eADMM LB helpers | Latest uploaded `compute_safe_lower_bound*.m` | Called with conversion factor 1 and offset 0, original graph lower-bound scale |
| Vector core | Latest uploaded `methods/vector/vector_splitadmm.m` | Unmodified; fresh graph-specific sequential wrapper |
| MMDLP LP core | Latest uploaded `methods/mmdlp/mmdlp_solve_relaxation.m` and `mmdlp_select_batch.m` | **Unmodified** LP relaxation and top-b rounding |
| MMDLP audit helper | Latest uploaded `mmdlp_bruteforce_iqp.m` and `test_mmdlp_exactness.m` | Unmodified, optional diagnostics only |
| MMDLP wrapper | New `experiments/run_mmdlp_graph_baseline.m` | Uses `Z_g=0.5*L(Q,Q)` for the graph group objective; no MMD Eq. (6--8) Z |
| Matrix core | Validated standalone v3 `solve_matrix_graph.m` | A=L/max(1,norm(L,inf)), no early_exist by default |
| Global objective | Standalone graph builder/evaluator | 0.5*trace(P'*L*P) on official adjacency |
| 2-opt | Independent graph adapter from previous standalone package | Original graph-cut scale |

Vector uses a **sequential greedy graph-cut decomposition**, which is not the
same procedure as simultaneously optimizing all k groups; the final discrete
objective and constraints are nevertheless the same. Graph MMDLP is also a
**sequential** method. Its LP core is identical to the uploaded source, but
its group model differs from the source paper's MMD Eq. (6--8); this is
**graph-cut adaptation**, not a claim of identical original MMD objective.
No old `build_mmd_batch_Z` is included.

See `SOURCE_HASHES.tsv` for file-level SHA256 mapping from the user ZIP to the
standalone files, including intentional changed adapters.
