# TensorGraphLab

TensorGraphLab lets you **postpone evaluation of a tensor network**. Define its connectivity with `@graph` before the numerical tensors exist. When you need the result, bind tensors to the nodes with `coloring` and call `evaluate` to perform the contraction using TensorKit and TensorOperations.

You can also [define your own lazy contraction type](#define-your-own-lazy-contraction-type). Store a reusable evaluation plan in the type and choose which operation triggers the contraction.

## Installation

Requires Julia 1.10 or later. The package is not yet registered in Julia General; install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Chenqitrg/TensorGraphLab.jl")
```

For a local checkout, start `julia --project=.` in the repository directory and run `using Pkg; Pkg.instantiate()`. No separate local TensorNodeLab, TensorGraphCore, or TensorGraphNotation packages are needed.

## Your first contraction

Define a graph connecting two nodes, then check that its evaluation agrees with matrix multiplication:

```julia
using TensorKit, TensorGraphLab

A, B = TensorNode(:A, (1, 1)), TensorNode(:B, (1, 1))
@graph G[left; right] := A[left; bond] * B[bond; right]

a = randn(ComplexF64, ℂ^2 ← ℂ^3)
b = randn(ComplexF64, ℂ^3 ← ℂ^4)
bindings = coloring(G, A => a, B => b)
result = evaluate(G, bindings)

@assert result ≈ a * b
```

- `TensorNode(:A, (1, 1))` declares one output and one input. If a representative tensor already exists, `TensorNode(:A, a)` takes its ranks and formal dual directions from `a`.
- In `@graph`, outputs appear before the semicolon and inputs after it. The repeated index `bond` connects the nodes; `left` and `right` form the graph boundary.
- `coloring` binds basic nodes to actual `AbstractTensorMap` objects and checks their ports and connected spaces.
- `evaluate` returns the contraction result, using the upstream `@tensor` macro by default.

The package exports these everyday entry points:

```text
TensorNode, TensorGraph, TubeGraph
@graph, @tube, @coloring
coloring, evaluate, showgraph, substitute, isplanar, open_graph
```

Access other interfaces through `TensorGraphLab.` or import them explicitly with `using TensorGraphLab: ...`. Rank and transformation functions such as `numout`, `numin`, and `permute` remain available through TensorKit.

## Define your own lazy contraction type

Suppose you want to pass the product defined by `G` to another routine and let that routine decide when to compute it. Using `G`, `a`, and `b` from above, define a callable Julia type that holds the contraction plan and the bound operands:

```julia
struct LazyContraction{P,B}
    plan::P
    bindings::B
end

(x::LazyContraction)() = x.plan(x.bindings)

plan = TensorGraphLab.plan_evaluation(G)
product = LazyContraction(plan, coloring(G, A => a, B => b))
@assert product() ≈ a * b
```

`plan_evaluation(G)` prepares the contraction using only the graph. Constructing `product` checks and stores references to the operands. Calling `product()` runs the contraction with their current values. A second `LazyContraction` can share `plan` and supply its own bindings.

## Reuse, nesting, and display

Use an existing graph as a node in another graph. Each occurrence of a subgraph has its own internal indices, so repeated names do not create unintended connections:

```julia
@graph network[left; right] := G[left; middle] * G'[middle; right]
@assert evaluate(network, bindings) ≈ result * result'

showgraph(network)          # Expand all subgraphs.
showgraph(network; depth=0) # Keep direct subgraphs.
```

`showgraph` prints a fully expanded `@graph` expression by default; `depth=1` expands one layer. Partially expanded output uses generated names such as `subgraph1`, which require separate bindings before the expression can be executed. Graphs do not retain the caller's variable name: the displayed output name defaults to `G`; set `name=:network` to change it. Ordinary REPL display shows a compact summary.

Nodes support structural `adjoint`, `permute`, and `transpose` views. Entire graphs support adjoint. `substitute(G, A => B, B => A)` swaps basic colors simultaneously, preserving existing views; inserted replacements are not processed again.

## Binding tensors

These three forms are equivalent:

```julia
bindings = coloring(G, A => a, B => b)
bindings = coloring(G, :A => a, :B => b)
bindings = @coloring G A=a B=b
```

In `@coloring`, left-hand names are literal color labels; each right-hand expression is evaluated once in the caller's scope. Assignments can also appear in a `begin ... end` block. Bindings cover basic colors throughout nested graphs. Repeated occurrences of an ordinary node share a tensor, and different colors may refer to the same tensor.

Provide every color exactly once, with no missing or extra bindings. For an ordinary node, the value is an `AbstractTensorMap`. Construction checks ranks, representable dual directions, and connected spaces. To check again explicitly, call `TensorGraphLab.validate(G, bindings)`; it returns `nothing` on success and throws on failure.

The sector type is shared by every bound tensor and can be read with
`sectortype(bindings)` or `sectortype(typeof(bindings))`.
An indexed color selects one tensor from a tuple under the same label.
For example, when a graph contains `TensorGraphLab.IndexedColor(node, 2)`
with `node = TensorNode(:gamma, (2, 2))`, the binding `:gamma => (hV, hW)` gives that
occurrence `hW`. Both slots print as `gamma`. Each occurrence checks its own
port directions and connected spaces. The caller supplies the tensors in slot
order.

Bindings retain references to the original tensors. Mutating their entries affects subsequent evaluations. Reassigning the Julia variable `a` does not update an existing binding; call `coloring` again to bind the new tensor. Access a bound tensor with `bindings[:A]` or `bindings[A]`.

For a `TubeGraph`, the same binding syntax checks the open patch and each pair of spaces joined by annular closure. For example, given a two-output, two-input tensor `a` with matching spaces on the wire labeled `x`:

```julia
A = TensorNode(:A, a)
@tube annular=(x,) T[u; d] := A[x u; d x]
bindings = @coloring T A=a
```

A single `id[o; r]` in tube notation carries a wire from `r` to `o`. Use `id` from TensorKit. The wire has no color binding. `open_graph(T)` retains it in the planar patch. `TensorGraphLab.closed_graph(T)` glues the annular pairs and suppresses identity segments along the resulting bonds.

## Tensor and planar execution

```julia
result = evaluate(G, bindings)                 # Use @tensor.
result = evaluate(G, bindings; method=:planar) # Use @planar.
```

Evaluation expands subgraphs, applies recorded node views, and passes a contraction expression to the upstream macro. TensorKit determines whether the symmetry, permutation, and contraction are supported. Execution does not automatically switch from `tensor` to `planar`. For explicit braiding, bind a `BraidingTensor` to a basic node; a color named `τ` is still treated as an ordinary label.

The two closed-graph forms preserve distinct output intents:

```julia
@graph z = A[i; j] * A'[j; i]
@graph Z[] := A[i; j] * A'[j; i]

closed_bindings = coloring(z, A => a)
value = evaluate(z, closed_bindings)  # A numerical scalar.
tensor = evaluate(Z, closed_bindings) # A zero-leg TensorMap.

@assert value ≈ norm(a)^2
@assert scalar(tensor) ≈ value
```

Bindings are checked on every evaluation. The cache stores generated contraction functions, never tensors or numerical results. `TensorGraphLab.clear_evaluation_cache!()` clears the function lookup cache; it does not guarantee immediate reclamation of Julia's generated methods or machine code.

## Runnable examples

Open any example in VS Code and run the entire file with **Julia: Execute File in REPL**. Each script activates the package environment relative to its own location, so examples work independently of the current directory and can run in any order. The first run downloads and precompiles dependencies.

| Example | Demonstrates |
|---|---|
| [01_matrix_chain.jl](examples/01_matrix_chain.jl) | From graph notation to matrix multiplication |
| [02_nested_transfer.jl](examples/02_nested_transfer.jl) | Nested four-leg transfers and reuse with updated data |
| [03_views_and_replacement.jl](examples/03_views_and_replacement.jl) | Adjoint, transpose, and simultaneous replacement |
| [04_closed_graph.jl](examples/04_closed_graph.jl) | Scalar and zero-leg TensorMap outputs |
| [05_planar_symmetry.jl](examples/05_planar_symmetry.jl) | Planar contraction with Fibonacci symmetry |

## Structural conventions and scope

`TensorNode{M,N}` records M output ports and N input ports. Nodes can also be defined without a numerical tensor:

```julia
Q = TensorNode(:Q, (2, 2))
```

The pair gives `(nout, nin)`. By default, codomain and domain spaces are
nondual: output flags are `false`, and input flags are `true` in the
all-outgoing convention. Counts may be zero but cannot be negative.
For other directions, specify the flags explicitly, for example
`TensorNode(:Q, 2, 2, (false, true, false, true))`.

Direction flags are ordered output-first, then input, in the all-outgoing convention: output flags follow codomain duality, while input flags negate domain duality. Real Cartesian spaces erase numerical dual distinctions, but graph ports retain formal arrows. Extracting a node from an already transformed real tensor cannot recover direction history that its numerical representation has lost.

`@graph` currently supports indexed nodes, views, subgraphs, and their products, without linear combinations or numerical coefficients. Each object's output/input partition must match. Internal indices occur twice and join opposite formal arrows; boundary indices occur once.

Graph equality (`==`) ignores dummy-index names, nesting, and factor order, while preserving colors, port order, connectivity, and output intent. It does not establish categorical identities or braid equivalence. Execution and its cache retain factor order to preserve planar semantics.

| Qualified interface | Purpose |
|---|---|
| `TensorGraphLab.expand(G; depth=nothing)` | Expand subgraphs to a chosen depth |
| `TensorGraphLab.graph_expr(G)` | Obtain a display `Expr` without executing it |
| `TensorGraphLab.base_node(node)` | Retrieve the basic node underlying a view |
| `TensorGraphLab.graphdata(G)` | Export a versioned, data-only structural record |
| `TensorGraphLab.graphfromdata(data)` | Validate and restore a structural record |

`space(G, bindings)` returns the TensorKit map space in the graph's boundary order.
Both this query and binding validation use space metadata only, including for
node views. They do not contract or transform tensor data. Numerical evaluation
applies the recorded views through TensorKit.

Use `opt=true` in `evaluate` to request TensorOperations contraction-order optimization.
For a fixed-size index, e.g. `evaluate(G, c; method=:planar, small_indices=(:x,))`,
the planner counts `x` as O(1) and all other indices as O(χ). Root-level index
names refer to the graph before expansion. When the unrestricted optimum
violates planar order, the planner tries planar joins using TensorKit's cyclic
index rules. TensorKit checks the resulting expression. The default uses
the upstream contraction order. Its direct runtime dependencies are TensorKit and TensorOperations. TensorOperations is pinned to 5.8.2 because its internal parsing interfaces are used.

## Development and tests

Run from the repository directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The [package entry point](src/TensorGraphLab.jl) lists the exports and includes for three areas. The test directories follow the same layout.

| Source | Responsibility | Tests |
|---|---|---|
| [graphs](src/graphs) | Nodes, ordinary graph construction and notation, structural operations, and planarity | [graph tests](test/graphs) |
| [tubes](src/tubes) | Tube construction, annular boundaries, and tube notation | [tube tests](test/tubes) |
| [evaluation](src/evaluation) | Bindings, space checks, and numerical contraction | [evaluation tests](test/evaluation) |

To review `@tube`, start with the [annular boundary example](examples/06_annular_boundary.jl). Follow the [macro](src/tubes/notation.jl) into the [TubeGraph constructor](src/tubes/tube.jl) and the [boundary checks](src/tubes/boundary.jl). Run just the tube tests with:

```sh
julia --project=. test/runtests.jl tubes/
```

If migrating from the earlier split packages, restart Julia and load only TensorGraphLab. Transfer existing graphs through `graphdata` / `graphfromdata` structural records rather than mixing objects owned by the old modules.
