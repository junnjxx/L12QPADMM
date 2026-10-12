# Source provenance

Upstream repository: https://github.com/shudianzhao/ADMM-GP

Paper: Angelika Wiegele and Shudian Zhao, *SDP-based bounds for graph partition via extended ADMM*, Computational Optimization and Applications 82 (2022), 251–291.

Upstream `main` tree inspected for this integration: `83e5aff6b99d6692082cb66c1064fdb767d9d528`.

Copied or faithfully reproduced from upstream:

- `extended ADMM/aadmm_3b.m`
- `extended ADMM/post_proc_2.m`
- `extended ADMM/mprw_ineq_general.m`
- `extended ADMM/post_proc_3.m`
- `form models/kequi_form.m`
- `form models/separation_kequi.m`
- `form models/tri_sep_kc.c`
- `heuristics/kequi_rounding.m`
- `heuristics/kequi_local.m`
- `LICENSE.md`

Locally added compatibility files:

- `extended ADMM/projf.m`
- `extended ADMM/project_W.m`

The public upstream tree contains calls to `projf` and `project_W` but does not contain the corresponding helper files. The local helpers implement exactly the standard operations required by the upstream routines:

- `projf`: scalar projection onto an interval for the adaptive sigma update;
- `project_W`: spectral projection of a symmetric matrix onto the PSD cone for the safe lower-bound post-processing.


Additional local compatibility for the MET branch:

- `form models/formtri_kc.m`: the upstream README explicitly calls this helper to build the accumulated triangle-inequality operator, but the public repository tree does not contain the file. The local implementation follows the three triangle types encoded in upstream `tri_sep_kc.c` and uses the same symmetric SDP-operator convention as `kequi_form.m`.
- `form models/tri_sep_kc.m`: pure-MATLAB fallback with the same interface, hash convention, triangle types, and 1e-3 separation threshold as upstream `tri_sep_kc.c`. The original C source is also included; users may compile it with `mex tri_sep_kc.c` for faster separation on larger instances.

Project-level integration code (outside `third_party`) uses the full k-equipartition lower-bound path from the paper: DNN via `aadmm_3b.m`, iterative MET separation via `separation_kequi.m`, strengthened inequality SDP solves via `mprw_ineq_general.m`, and safe LP post-processing via `post_proc_3.m`. If no violated MET inequality is found, it falls back to the DNN `post_proc_2.m` certificate. Every safe graph-SDP lower bound is immediately converted to the common batching-objective scale.

Current project note:

- The upstream `heuristics/kequi_local.m` is retained unchanged for provenance/reference.
- The active shared postprocessor used by Matrix / Random / eADMM is
  `postprocess/common_two_opt_local.m`, which implements the same balanced
  pair-swap neighborhood directly on the project's `J_common` / `Phi` scale.
  Therefore no graph Laplacian is used by the common postprocessing stage.
