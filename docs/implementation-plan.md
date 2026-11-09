# Coloring and evaluation implementation plan

> Execute inline following executing-plans, with one final fresh-context review.

Goal: complete the approved graph → coloring → tensor/planar evaluation workflow in the fourth independent package, with five runnable examples. User authorized proceeding after committing Core and Notation.

Architecture: TensorGraphLab reexports Notation, binds actual AbstractTensorMaps to basic colors and checks their ports and connected spaces. Evaluation flattens hygienically, replays recorded views and sends a generated contraction expression to the existing upstream macro. Cache generated functions, never numerical results or tensors. No contraction or planarity algorithm is duplicated.

Spec: prior accepted interfaces and ../TensorNodeLab.jl/docs/review-plan.md. No new formal-leg abstraction, graph configuration object, ordinary-array support or ExperimentLab dependency. All Julia comments/docstrings/prints English. Existing versions unchanged. This new package stays uncommitted for review.

## Interfaces and decisions

- coloring(G, A=>TA, B=>TB) returns immutable Coloring with a NamedTuple of borrowed tensor references. Basic-node and Symbol keys accepted; exactly one binding per basic color. Reject missing, unknown, duplicate and incompatible bindings at construction. No tensor copies retained.
- @coloring G A=TA B=TB uses literal color labels, with RHS evaluated in caller scope; also allow a begin block.
- validate(G, bindings) rechecks and returns nothing. Structural checks use stored node metadata; link compatibility delegates to TensorOperations.checkcontractible. CartesianSpace cannot encode formal duals, so validate rank/spaces without imposing its erased orientation on the graph.
- evaluate(G, bindings; method=:tensor) supports :tensor/:planar. Fully expand subgraphs preserving occurrence and port order; evaluate node views through TensorKit. Use fresh generated occurrence names, never interpolate user labels as executable code or allow τ's special macro meaning to override its ordinary graph color.
- Cache keyed by ordered expanded graphdata plus method, not canonical graph equality (which ignores factor order). Cache functions under a lock; use invokelatest for functions compiled during a running caller. Validate on evaluation, so reuse with another graph cannot bypass checks. Cache contains no numerical data. clear_evaluation_cache! releases cached entries.
- Preserve scalar versus zero-leg TensorMap output intent. No upstream macro overloading or user AST evaluation.

## Tasks

- [x] Write failing binding tests; implement coloring.jl and reexports. Test missing/extra/duplicate/wrong rank/dual/space/non-TensorMap bindings, lexical @coloring, real-space direction handling, nested colors.
- [x] Write failing evaluation tests; implement execution.jl using upstream macro expansion. Compare with direct @tensor/@planar for complex and symmetric tensors, scalar/rank-zero, repeated subgraphs, views and ordinary τ color. Test changed numerical data, graph order, method validation, world age and nonplanar rejection. Add safe cache and clear function.
- [x] Five @graph examples: matrix chain, nested rank-four transfer, replacement/views, closed graph, planar/symmetry. Each prepares sibling dependencies using @__DIR__, runs from any cwd, and can repeat in one session.
- [x] Full Pkg.test; independent and repeated examples, fresh environment setup; independent final review. Stop for review, no push/PR.

Review focus: input-only/output-only/rank-zero graphs; real formal arrows surviving numerical self-duality; reserved/awkward labels; no stale tensors in cached execution; planar order must not be erased by structural equality.
