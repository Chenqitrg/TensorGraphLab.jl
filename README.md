# TensorGraphLab

统一的 tensor network 图包：`@graph` 描述结构，`coloring` 绑定实际 TensorMap，`evaluate` 调用 TensorKit / TensorOperations 做收缩。

```julia
using TensorKit, TensorGraphLab

a = randn(ComplexF64, ℂ^2 ← ℂ^3)
b = randn(ComplexF64, ℂ^3 ← ℂ^4)
A, B = TensorNode(:A,a), TensorNode(:B,b)
@graph G[left; right] := A[left; bond] * B[bond; right]
bindings = coloring(G,A=>a,B=>b)
result = evaluate(G,bindings)
@assert result ≈ a*b
```

默认只导出日常入口：`TensorNode`、`TensorGraph`、`@graph`、`@coloring`、`coloring`、`evaluate`、`showgraph` 和 `substitute`。其他接口通过 `TensorGraphLab.` 访问或显式导入；`numout`、`numin`、`permute` 等沿用 TensorKit 的入口。

## 五个可直接运行的例子

只需要本仓库。打开任一例子，用 VS Code 的 **Julia: Execute File in REPL** 运行整个文件。每个文件按自身路径激活本包环境，不需要其他本地包、切换工作目录或按顺序运行。首次运行会下载和预编译 TensorKit / TensorOperations 等依赖。

| 例子 | 内容 |
|---|---|
| [01_matrix_chain.jl](examples/01_matrix_chain.jl) | 从图到实际矩阵乘法 |
| [02_nested_transfer.jl](examples/02_nested_transfer.jl) | 嵌套四腿 transfer；修改数据后复用 |
| [03_views_and_replacement.jl](examples/03_views_and_replacement.jl) | adjoint、transpose、同时替换 |
| [04_closed_graph.jl](examples/04_closed_graph.jl) | scalar 与零腿 TensorMap 的输出区别 |
| [05_planar_symmetry.jl](examples/05_planar_symmetry.jl) | Fibonacci 对称性下的 planar 收缩 |

## 节点与图结构

`TensorNode(:A, tensor)` 提取 color、输入输出腿数与形式对偶方向，不保留数值 tensor、维数或 sector。也可以显式定义：

```julia
A = TensorNode(:A, 2, 2, (false, false, true, true))
```

`TensorNode{M,N}` 保留 M 个输出和 N 个输入；方向按输出腿、输入腿排列，采用全部腿视为输出的约定。`adjoint`、`permute`、`transpose` 返回带父节点的结构视图；`parent` 取直接父节点，`TensorGraphLab.base_node` 取基本节点。实数空间也保留形式箭头。

`@graph` 支持已有节点、节点视图和子图的乘积，不支持线性组合或数值系数。分号两侧必须匹配对象的输入输出腿数。内部指标出现两次并连接相反的形式箭头，外部指标出现一次。

`TensorGraphLab.expand(G; depth=nothing)` 默认完全展开，`depth=0` 保留直接子图；每次子图出现的私有指标独立。`substitute(G,A=>B,B=>A)` 同时替换基本 color，保留外层视图。`G'` 对整图取 adjoint。

图的 `==` 忽略内部指标名字、嵌套分组与 occurrence 书写顺序，保留 color、端口顺序、连线及输出意图；不判断范畴恒等式或 braid 等价。执行仍使用原有的 occurrence 顺序。

`showgraph(G; name=:G, depth=nothing)` 打印展开的 `@graph` 表达式，`TensorGraphLab.graph_expr` 返回对应 `Expr`。默认对象显示为简短摘要。有限展开深度下的 `subgraph1` 等名字需要另行绑定。`TensorGraphLab.graphdata` / `TensorGraphLab.graphfromdata` 保存和恢复带版本号的纯结构记录。

## 着色

```julia
bindings = coloring(G, A=>a, B=>b)
bindings = coloring(G, :A=>a, :B=>b)
bindings = @coloring G A=a B=b
```

`@coloring` 左侧按字面解释为图的基本 color label，右侧在调用者作用域求值一次。支持 `begin` 中逐行写赋值。子图内的基本 color 也一起绑定，同名节点复用同一个 tensor。

一次必须提供全部 color，不能重复或多给。绑定键可以是基本 TensorNode 或 Symbol，值只支持 AbstractTensorMap。检查腿数、可表示的对偶方向和内部连线空间；空间匹配调用 TensorOperations 的 `checkcontractible`。不同 color 可以绑定同一 tensor。

`TensorGraphLab.Coloring` 保存原 tensor 的引用，不复制数据。`bindings[:A]` 或 `bindings[A]` 读取绑定。修改 tensor 的数值会影响下一次 evaluation；给变量 `a` 重新赋值不会自动更换已有绑定，需要重新 `coloring`。`TensorGraphLab.validate(G,bindings)` 可以重新检查，成功返回 `nothing`，失败抛出异常。

实数 CartesianSpace 无法编码形式对偶，因此不拿其已消去的方向否定图上的箭头，仍然检查腿数和空间连接。对其他空间检查基本节点方向。节点视图通过 TensorKit 实际操作检查；permute/transpose 可能为验证生成临时 tensor，但不会保存在 Coloring 中。

## 执行

```julia
result = evaluate(G,bindings)                 # Use @tensor.
result = evaluate(G,bindings; method=:planar) # Use @planar.
```

完全展开嵌套图，保留 occurrence 和端口顺序，重放节点视图，然后生成上游宏接受的收缩表达式。执行代码中使用内部名字，因此普通 color `τ` 不会被误当成上游特殊 braiding 语法。显式 BraidingTensor 可以像其他 TensorMap 一样绑定。

`@graph z = ...` 的结果是数值 scalar；`@graph Z[] := ...` 的结果是零腿 TensorMap。是否允许特定对称性、排列或 planar contraction 由 TensorKit 判断，错误会向上传递。默认不会自动从 tensor 切换到 planar。

每次执行都重新检查绑定。缓存只存生成的函数，键包含展开后的**有序**结构与执行方法，不使用忽略因子顺序的图相等性，也不保存 tensor 或数值结果。更换 tensor 的尺寸和类型无需改图，Julia 对实际参数类型编译；`TensorGraphLab.clear_evaluation_cache!()` 清空函数查找缓存。缓存条目会随不同结构增长，清空缓存不保证立即回收 Julia 已生成的方法或机器码。

本轮复用上游默认收缩顺序，不提供额外 optimizer/backend 配置，也没有修改上游 `@tensor`、`@planar` 宏。现在的 `showgraph` 接口及默认摘要显示保持不变。

## 源码与验证

- [TensorGraphLab.jl](src/TensorGraphLab.jl)：唯一模块入口，按职责显式列出公共 export。
- [nodes.jl](src/nodes.jl)：基本节点、方向和结构视图。
- [graphs.jl](src/graphs.jl)：图结构、检查、展开与 adjoint。
- [identity.jl](src/identity.jl)、[substitution.jl](src/substitution.jl)、[persistence.jl](src/persistence.jl)：比较、替换和结构记录。
- [notation.jl](src/notation.jl)、[display.jl](src/display.jl)：宏解析与表达式显示。
- [coloring.jl](src/coloring.jl)、[execution.jl](src/execution.jl)：绑定检查、上游收缩与缓存。

原四个包的测试已一起迁入 `test/`。运行时只直接依赖 TensorKit 和 TensorOperations；解析复用了 TensorOperations 内部接口，因此目前固定其版本为 5.8.2。

运行任一例子准备依赖后，在本包目录运行：

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

不依赖 ExperimentLab；未增加形式腿对象。包名、UUID 和版本保持不变。旧拆分仓库保留作历史记录，新代码只需 `using TensorGraphLab`。

合并后节点和图类型归属于 TensorGraphLab。旧会话中若已加载拆分包，请重启 Julia 再使用；迁移已有图应使用 `TensorGraphLab.graphdata` / `TensorGraphLab.graphfromdata` 的结构记录，不直接混用旧包对象。
