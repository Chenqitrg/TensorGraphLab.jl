# TensorGraphLab

第四个独立 review 包，也是完整使用入口：`@graph` 描述结构，`coloring` 绑定实际 TensorMap，`evaluate` 调用 TensorKit / TensorOperations 做收缩。

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

## 五个可直接运行的例子

四个包目录放在同一父目录。打开任一例子，用 VS Code 的 **Julia: Execute File in REPL** 运行整个文件。每个文件按自身路径激活环境、接入三个相邻包，不需要切换工作目录或按顺序运行。首次运行会下载和预编译依赖。

| 例子 | 内容 |
|---|---|
| [01_matrix_chain.jl](examples/01_matrix_chain.jl) | 从图到实际矩阵乘法 |
| [02_nested_transfer.jl](examples/02_nested_transfer.jl) | 嵌套四腿 transfer；修改数据后复用 |
| [03_views_and_replacement.jl](examples/03_views_and_replacement.jl) | adjoint、transpose、同时替换 |
| [04_closed_graph.jl](examples/04_closed_graph.jl) | scalar 与零腿 TensorMap 的输出区别 |
| [05_planar_symmetry.jl](examples/05_planar_symmetry.jl) | Fibonacci 对称性下的 planar 收缩 |

## 着色

```julia
bindings = coloring(G, A=>a, B=>b)
bindings = coloring(G, :A=>a, :B=>b)
bindings = @coloring G A=a B=b
```

`@coloring` 左侧按字面解释为图的基本 color label，右侧在调用者作用域求值一次。支持 `begin` 中逐行写赋值。子图内的基本 color 也一起绑定，同名节点复用同一个 tensor。

一次必须提供全部 color，不能重复或多给。绑定键可以是基本 TensorNode 或 Symbol，值只支持 AbstractTensorMap。检查腿数、可表示的对偶方向和内部连线空间；空间匹配调用 TensorOperations 的 `checkcontractible`。不同 color 可以绑定同一 tensor。

`Coloring` 保存原 tensor 的引用，不复制数据。`bindings[:A]` 或 `bindings[A]` 读取绑定。修改 tensor 的数值会影响下一次 evaluation；给变量 `a` 重新赋值不会自动更换已有绑定，需要重新 `coloring`。`validate(G,bindings)` 可以重新检查，成功返回 `nothing`，失败抛出异常。

实数 CartesianSpace 无法编码形式对偶，因此不拿其已消去的方向否定图上的箭头，仍然检查腿数和空间连接。对其他空间检查基本节点方向。节点视图通过 TensorKit 实际操作检查；permute/transpose 可能为验证生成临时 tensor，但不会保存在 Coloring 中。

## 执行

```julia
result = evaluate(G,bindings)                 # Use @tensor.
result = evaluate(G,bindings; method=:planar) # Use @planar.
```

完全展开嵌套图，保留 occurrence 和端口顺序，重放节点视图，然后生成上游宏接受的收缩表达式。执行代码中使用内部名字，因此普通 color `τ` 不会被误当成上游特殊 braiding 语法。显式 BraidingTensor 可以像其他 TensorMap 一样绑定。

`@graph z = ...` 的结果是数值 scalar；`@graph Z[] := ...` 的结果是零腿 TensorMap。是否允许特定对称性、排列或 planar contraction 由 TensorKit 判断，错误会向上传递。默认不会自动从 tensor 切换到 planar。

每次执行都重新检查绑定。缓存只存生成的函数，键包含展开后的**有序**结构与执行方法，不使用忽略因子顺序的图相等性，也不保存 tensor 或数值结果。更换 tensor 的尺寸和类型无需改图，Julia 对实际参数类型编译；`clear_evaluation_cache!()` 清空函数查找缓存。缓存条目会随不同结构增长，清空缓存不保证立即回收 Julia 已生成的方法或机器码。

本轮复用上游默认收缩顺序，不提供额外 optimizer/backend 配置，也没有修改上游 `@tensor`、`@planar` 宏。现在的 `showgraph` 接口及默认摘要显示保持不变。

## 源码与验证

- [coloring.jl](src/coloring.jl)：绑定、检查和 `@coloring`。
- [execution.jl](src/execution.jl)：表达式生成、上游执行和缓存。
- [TensorGraphLab.jl](src/TensorGraphLab.jl)：入口与底层公共接口重新导出。

运行任一例子准备依赖后，在本包目录运行：

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

不依赖 ExperimentLab；未增加形式腿对象。本轮 review 已通过。
