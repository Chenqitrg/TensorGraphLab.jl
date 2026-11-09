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

## Consolidation into one package

- User requested merging the four reviewed packages and readable main exports. Implemented on codex/consolidate-packages, preserving TensorGraphLab UUID/version and leaving the three old repositories untouched.
- Nodes, graph core and notation now live in separate source files inside the single TensorGraphLab module. Removed all three sibling dependencies and their example setup paths. Replaced dynamic @eval exports with explicit grouped exports; the exported-name set is unchanged (33 names including the module itself).
- Migrated all behavioral suites. Direct tests and full Pkg.test passed 315/315; package tests used --compiled-modules=no and -O0. Logs: /tmp/graph-unified-tests.log and /tmp/graph-unified-pkgtest.log.
- All five examples passed independently from an isolated fresh copy containing only this package, without a Manifest or sibling packages. All five also passed twice in one Julia session. Logs: /tmp/graph-unified-isolated.log and /tmp/graph-unified-repeat.log.
- README documents single-package setup, explicit source organization and the module/type ownership change. Restart Julia after loading the old split packages; transfer old graphs via graphdata/graphfromdata rather than mixing old type identities.
- Independent final review found no actionable integration issues: copied sources/assertions preserved, no helper collisions or sibling dependencies, explicit exports/include order correct. No commit, push or PR for this consolidation yet.

- User approved narrowing exports to TensorNode, TensorGraph, @graph, @coloring, coloring, evaluate, showgraph and substitute. Advanced interfaces remain available by qualification or explicit import; tests import them explicitly and README reflects this intentional API-surface change.

- After export reduction, all 315 tests and five examples passed again. User approved committing consolidation and the reduced export list together. No version bump or push.
