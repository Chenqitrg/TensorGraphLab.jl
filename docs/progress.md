# Evaluation progress

- Core committed cdf6884; Notation committed d4fb5af after 90/90 and 50/50 direct tests. No push or PR.
- New isolated repository codex/review-04-evaluation implements the already approved fourth package. Existing package versions and implementations unchanged.
- Binding tests failed on absent @coloring; all 17 passed after implementation. Bindings store original tensor references and validate connected spaces through TensorOperations.checkcontractible.
- Evaluation tests written before implementation, covering numerical and planar results, views, scalar/rank-zero, live mutations, awkward labels, symmetry, method rejection, world age and revalidation.
- Ruling: execution uses internal occurrence names and ordered graphdata, not printed user labels or canonical equality. This preserves planar ordering and avoids τ syntax collisions.
- Ruling: planar selection is evaluate(...; method=:planar). Public TensorKit macros are invoked by generated functions, with cached code and invokelatest; no numeric results cached.
- The direct evaluation suite passed 57/57. A test fixture was corrected to mutate tensor blocks (TensorMap does not support array broadcasting). Nonplanar rejection initially surfaced as Core.eval's LoadError wrapper; the adapter now exposes the upstream error, verified by the previously failing test.
- Added boundary-only/rank-zero and Fibonacci/BraidingTensor probes; all passed. Explicit τ bindings use internal operand names and retain numerical meaning.
- Full Pkg.test with --compiled-modules=no and -O0 passed 57/57, exit 0. Five independent example processes from /tmp passed, as did all five twice in one Julia session. A copied four-package environment without Manifests resolved dependencies and ran example 1 successfully. Logs: /tmp/graph-eval-pkgtest.log, /tmp/graph-eval-examples.log, /tmp/graph-eval-repeat.log, /tmp/graph-eval-fresh.log.
- Fresh-context final review found no actionable correctness issues. Four independent probes covered single-node self-traces and disconnected scalar products in both tensor and planar modes. New source totals 161 lines; examples use @graph and remain independently runnable. No new package commit or push.

- User approved the package and explicitly requested a local commit. No version bump, push or PR.
