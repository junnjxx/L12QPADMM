# Standalone Graph k-Equipartition — GitHub latest + Vector + MMDLP

This is a **separate MATLAB graph-partition project**. It does not invoke the
minibatch experiment or the MNIST workflow. Files use the latest source provided
in `matrix_eadmm_lp_vector_code.zip`, apart from the explicitly identified
graph-model and entrypoint adapters. See `SOURCE_AUDIT.md`.

## Mathematical model / 统一评价尺度

For symmetric nonnegative adjacency W with zero diagonal, define
L=diag(W*1)-W. We solve equal-cardinality k-equipartition:

    min  J_cut(P) = (1/2) trace(P' * L * P)
    s.t. P in {0,1}^{n x k}, P*1_k=1_n, P'*1_n=(n/k)*1_k.

No `bs^2` rescaling of the shared objective. The number `b=n/k` is merely
**group cardinality** (not a training minibatch size).

* **Matrix** uses the validated v3 update, A_matrix=L/max(1,norm(L,inf)), G=0,
  and evaluates candidates using the original L. `early_exist=false` by default.
* **eADMM** uses the latest supplied DNN/MET/safe-lower-bound/10-rounding code;
  passes W directly to `kequi_form`. Optional common 2-opt is applied once by
  the experiment runner (to the raw-best solution), not twice.
* **Lp-Alg2/Lp-bs** use the supplied latest `lp_balanced_algorithm2.m`,
  `project_balanced_mosek.m` and accelerated exact N2. Both optimize the
  original graph-cut quadratic f(P)=0.5*trace(P'*L*P). Lp-bs disables internal N2.
* **Vector** uses the supplied **unmodified** `vector_splitadmm.m` sequentially.
  At stage g, for unassigned vertex indices Q, select `b` vertices with the
  graph-subproblem quadratic `A_g=L(Q,Q)`:

      min 0.5 * x' * L(Q,Q) * x, s.t. x in {0,1}^|Q|, sum(x)=b.

  Exactly, the sum of these group contributions is `J_cut(P)` for any completed
  partition. This is a **sequential greedy baseline**, NOT a simultaneous
  solver; it may need top-b extraction if the vector ADMM is not binary.
  The principal block `L(Q,Q)` includes degrees from the FULL graph.
  Do not replace with the Laplacian of the induced subgraph or with MMD `Z`.
* **MMDLP (graph-adapted)** retains the uploaded latest AAAI-21 Eq. (10)
  sign-dependent LP relaxation and top-b selection **unchanged**. At each
  sequential stage g, for remaining vertices Q, the graph objective is

      min  m' Z_g m,    Z_g = 0.5 * L(Q,Q),
      s.t. m binary, sum(m)=b.

  The LP relaxation is followed by deterministic top-b rounding. The final
  group's vertices are forced. As with Vector, the sum of rounded group
  objectives is the original global graph cut **exactly**. This is an
  application of the MMDLP LP technique to a graph quadratic, **not** the
  original MMD-specific Z (Eqs. 6--8), so do not label its LP objective as
  a certified global graph-cut lower bound. It requires MATLAB
  Optimization Toolbox `linprog` (not MOSEK).
* **Random** generates multiple balanced partitions.
* **Shared 2-opt** evaluates the original graph cut; raw/post quality and
  solve/extraction/2opt/total timings are recorded.

There are **no minibatch data or MMD kernel inputs**. Both Vector and MMDLP
are sequential group-assignment methods applied to original graph cut.
No `build_mmd_batch_Z` is called: that formula would change the per-group
objective and break direct comparability of the discrete subproblems.

## Run

In MATLAB, inside this directory:

```matlab
smoke_test
integration_test     % small graph solver tests; optional Lp/MMDLP if toolboxes found
run_exp              % formal graph experiments; does not auto-run tests
```

Configure only `config_graph.m`, for example:

```matlab
cfg.graph.source = 'official';
cfg.graph.family = 'rand20';  % rand20/rand50/rand80
cfg.graph.n = 100;
cfg.graph.num_groups = 25;
cfg.methods.run = {'eADMM','Lp','MMDLP','Vector','Random','Matrix'};
```

You may select any subset, e.g. `{'Matrix','MMDLP','Vector','eADMM'}`.

### Lp dependency — MOSEK is mandatory

The latest Lp implementation intentionally uses **MOSEK** projection; it does
**not** silently fall back to older dual-BB code. Ensure `mosekopt` is on your
MATLAB path or specify in `config_graph.m`:

```matlab
cfg.lp.proj.mosek_toolbox_path = '/path/to/mosek/toolbox/r2019bom';
cfg.lp.proj.mosek_license_file = '/path/to/mosek.lic';
```

For testing on a computer without MOSEK, set
`cfg.methods.run = {'Matrix','eADMM','Vector','Random'}`.

### MMDLP dependency — Optimization Toolbox

MMDLP calls the latest uploaded `mmdlp_solve_relaxation.m` using MATLAB's
`linprog`. If `linprog` is unavailable, remove `'MMDLP'` from
`cfg.methods.run`; **no alternative LP or automatic fallback** is used.
The first graph subproblem has `n+n^2` LP variables and `n^2` inequalities;
on large graphs this can be expensive. Optional diagnostics:

```matlab
cfg.mmdlp.exact_audit = false;       % keep false for formal timing
cfg.mmdlp.store_relaxed_history = false;
cfg.mmdlp.verbose = true;
```

Raw MMDLP time is divided into `SolveTimeSec` (linprog calls only) and
`InitExtractTimeSec` (model builds, selection, validation, other overhead);
common 2-opt is accounted for separately. Exact brute-force audit time is
excluded from the method time when auditing is enabled.

### eADMM certification

Use `cfg.eadmm.met.enabled=true` to enable MET cuts and
`cfg.eadmm.compute_safe_lower_bound=true` for the dual-repair LP certificate.
The latter also requires Optimization Toolbox (`linprog`). Certified-LB/MET
extra time is reported separately from DNN + rounding solution generation.

## Inputs and outputs

Bundled **official ADMM-GP** MAT files for rand20/rand50/rand80 (A100 ...
A1000). By default, `source='official'`; not a synthetic demo.

Outputs: `logs/graph_*.txt` and `results/graph_*.mat`.
The report records raw cut and the cut after common 2-opt, plus
`SolveTimeSec`, `InitExtractTimeSec`, `LocalSearchTimeSec`, and
`TotalMethodTimeSec` (MATLAB wall-clock times).

## Validation status

- The shipped `smoke_test.m` checks the cut identity and the balanced 2-opt.
- `integration_test.m` checks solver interfaces, tests graph-adapted MMDLP
  with exact-audit on small graphs if `linprog` is available, and runs Lp
  only when MOSEK is available.
- We audited the included MATLAB source and checked benchmark objectives and
  source fingerprints, but **MATLAB execution was not available in the build
  environment**. Do not treat the integration test as having run successfully
  until you run it locally.
