# TileLangIR Codegen

## 转换对应方案

转换分为两大类：优先转换为 MLIR 已有方言；已有方言无法完整表达 TileLang
语义时，转换为 `tilelang` 自定义方言。

### 1. 转换为已有 MLIR 方言

已有 MLIR 方言的选择遵循以下原则：

1. 如果 `linalg` 有与 TileLang 操作语义对应的 Operation，并且各参数 shape
   一致，则直接转换为对应的 `linalg` Operation。
2. 如果参数 shape 不一致但存在明确的逐元素或广播关系，则转换为
   `linalg.generic`，并在计算 Region 中使用 `arith`、`math` 等标量
   Operation 表达具体计算。例如 `vmuls` 的向量乘标量；后续 `vadd` 遇到
   可表达的不同 shape 时也应采用该方式，而不是直接失败。
3. 如果 `linalg` 有对应 Operation，且该操作天然具有不同的输入、输出
   shape，例如 reduction，则转换为 `linalg.reduce` 等对应结构化 Operation。
4. 如果 `linalg` 没有能够完整对应语义的 Operation，则转换为
   `linalg.generic`，并在计算 Region 中使用相应的 `arith`、`math` 等已有
   MLIR Operation 表达计算。

| TIRX / TileLang 语义 | 目标 MLIR 表示 | 说明 |
| --- | --- | --- |
| `IRModule` | `builtin.module` | 整个 Codegen 结果的顶层容器。 |
| `PrimFunc` | `func.func` | 函数参数、返回值和符号信息保持可追踪。 |
| `Buffer`、函数 Buffer 参数 | `memref` 类型 | 保留 shape、dtype 和动态维度。 |
| Buffer 分配 | `memref.alloc` 或 `memref.alloca` | 根据分配生命周期选择；内存 scope 作为内建 StringAttr 附在分配操作上。 |
| 普通标量表达式中的 `BufferLoad` | `memref.load` | 读取单个元素；`tl.tileop.region` 内嵌的 `BufferLoad` 不属于这种情况。 |
| 普通标量赋值中的 `BufferStore` | `memref.store` | 写入单个元素；当前实现需要先具备对应的 For 和 BufferLoad 支持。 |
| `tl.tileop.region` | `memref.reinterpret_cast` | 将内嵌 `BufferLoad` 拆为原始 Buffer 和起始索引，保留 offset、size 和 stride。 |
| 整数、浮点数和布尔常量 | `arith.constant` | 保留原始 dtype。 |
| 加减乘除、比较、选择和类型转换 | `arith` 方言 | 根据整数、浮点数和有无符号语义选择对应 Operation。 |
| `exp` 等数学表达式 | `math` 方言 | 使用已有数学 Operation。 |
| Serial `For` | `scf.for` | 保留 min、extent、step 和循环体，并增加 `tilelang.loop_kind = "serial"`。 |
| Parallel `For` | `scf.for` | 增加 `tilelang.loop_kind = "parallel"`，并原样保留循环 annotations。 |
| Vectorized `For` | `scf.for` | 增加 `tilelang.loop_kind = "vectorized"`，并原样保留循环 annotations。 |
| Unrolled `For` | `scf.for` | 增加 `tilelang.loop_kind = "unrolled"`，并原样保留 unroll factor 等 annotations。 |
| Pipelined `For` | `scf.for` | TIRX 中实际是 Serial `For`；根据 pipeline annotations 识别，增加 `tilelang.loop_kind = "pipelined"`，并原样保留 `num_stages`、`tl_pipeline_order`、`tl_pipeline_stage` ; 根据`tl_pipeline_group`合并指定的 pipeline units，并消除该 annotation。 |
| 条件语句 | `scf.if` | 保留 then/else Region。 |
| `SBlock` | `scf.execute_region` | 在 Operation 上增加 `tilelang.sblock_name` 等普通 Attribute；不转换自动推导的 `reads`/`writes`。 |
| `SimdVF` | `scf.execute_region` | 增加表示 SimdVF scope 的普通 Attribute。 |
| `vmuls` | `linalg` 结构化逐元素操作 | 向量与标量逐元素相乘；标量计算体根据其dtype使用 `arith` 的 `MulFOp`/`MulIOp` |
| `vadd`、`vmul`、`vmax` | `linalg` 结构化逐元素操作 | 标量计算体使用 `arith` Operation；不在 Codegen 中融合这些操作。 |
| `vreduce_sum`、`vreduce_max` | `linalg.reduce` | reduction body 分别使用加法或最大值 Operation。 |
| `vexp` | `linalg.exp` | 对同 shape 的输入、输出执行逐元素指数计算。 |
| `vexpdif` | `linalg.generic` | 计算 Region 使用 `arith.subf` 和 `math.exp`，并通过 indexing map 表达标量广播。 |
| `vcvt` | `linalg` 结构化逐元素操作 | 计算体使用相应的 `arith` cast Operation。 |

`shared.dyn`、`local.fragment` 等 scope 当前不设计成自定义 MLIR 类型或自定义
Attribute 类型，而是使用普通字符串 Attribute 保留在对应 Buffer 分配 Operation
上。后续如果下游对 memory space 类型提出严格要求，再单独扩展类型系统。

### 2. 转换为 `tilelang` 自定义方言

以下语义目前没有能够完整、一一承接的标准 MLIR Operation，因此保留为自定义
Operation，不在 Codegen 阶段提前展开或硬件化：

| TIRX / TileLang 语义 | 自定义 MLIR Operation | 需要保留的信息 |
| --- | --- | --- |
| `T.copy` / `tl.tileop.copy` | `tilelang.copy` | 源和目标 Region，以及 `split_dim`、`transpose` 等 Attribute。 |
| `T.gemm` / `tl.tileop.gemm` | `tilelang.gemm` | A/B/C Region、transpose、M/N/K、clear_accum 和其他原始参数。 |
| Kernel launch / thread binding | `tilelang.launch_thread` | thread tag、extent、绑定变量和嵌套 Region。 |
| Pipelined 循环中的可执行源语句 | `tilelang.pipeline_stage` | 保持一个源调度单元，显式 sblock 的名称存为 label；内部地址展开不改变 stage/order 的计数。 |

自定义 Operation 的目标是无损保存前端语义，而不是在这里决定最终硬件指令。
后续真正面向某个后端时，可以再对 TileLangIR 运行独立的 lowering 或转换流程。

## 方案原理

整体链路如下：

```text
TileLang Python DSL
    -> TVM Script Parser / TileLang language helpers
    -> TIRX IRModule
    -> OpenTile Pass Pipeline
    -> Python DeviceCodegen
    -> TileLangIR Translator
    -> MLIR Python Builder API
    -> MLIR Module 文本
    -> TVM SourceModule 包装
```

### 1. Codegen 入口

`target="tile"` 被标准化为带有 `tile` key 的 Target。OpenTile Codegen 将
Python 函数注册进现有 `DeviceCodegen` 注册框架，因此 `lower()` 最终选择的是
Python Codegen，而不是已删除的 `src/opentile` C++ Codegen。

### 2. 遍历 TIRX 数据结构

Translator 接收 `tvm.IRModule`，直接访问其中的 `PrimFunc`、`Stmt`、`PrimExpr`、
`Buffer` 和 `Call` 等 TIRX Python ObjectRef。

Translator 使用 TVM `PyStmtExprVisitor` 根据实际节点类型分派普通 TIRX 节点，
并根据 `Call.op.name` 分派 `tl.tileop.copy`、`tl.tileop.gemm` 等 TileLang
Operation。转换过程中维护：

- TIRX `Var` 到 MLIR SSA Value 的映射;TIRX `Buffer` 到 MLIR memref Value 的映射；
- TIRX Region 到 MLIR Region/Block 的嵌套关系；
- dtype、shape、scope、annotation 和符号信息的转换结果。

#### `tvm.IRModule` 数据结构与遍历方法

TIRX 节点首先按照接口类型分为 `Stmt`、`PrimExpr` 等大类，再细分为具体
节点类型。例如，`Stmt` 包含 `For`、`SBlock`、`Evaluate` 等节点，
`PrimExpr` 包含 `Add`、`BufferLoad`、`Call` 等节点。`Buffer` 等对象也是
TIRX ObjectRef，但不属于 `Stmt` 或 `PrimExpr`，由引用它们的节点处理函数读取。

Translator 从 `PrimFunc.body` 开始一次 DFS。`visit_stmt()` 和 `visit_expr()`
都是 TVM `PyStmtExprVisitor` 直接提供的遍历函数，分别接收 `Stmt` 和
`PrimExpr`。它们在内部根据节点的具体类型 dispatch 到固定名称的处理函数。
`visit_for_()`、`visit_sblock_()`、`visit_call_()` 等处理函数由 Translator
按需重新实现；覆盖以后，当前节点的处理和是否继续访问子节点都由覆盖函数负责。

```text
按照节点的大类选择 TVM 遍历函数
    ├── 节点是 Stmt     → visit_stmt(child_stmt)
    └── 节点是 PrimExpr → visit_expr(child_expr)
    ....
                                  ↓
TVM的visit_stmt和visit_expr实现，会按具体节点类型 dispatch
    ├── For        → visit_for_()
    ├── SBlock     → visit_sblock_()
    ├── Evaluate   → visit_evaluate_()
    ├── Add        → visit_add_()
    ├── BufferLoad → visit_buffer_load_()
    └── Call       → visit_call_()
    .....
上述函数，覆盖固定名称的 visit_xxx_()，即可替代tvm原生处理方案，来处理改类节点
                                  ↓
继续 DFS：子 Stmt 调用 visit_stmt()，子 PrimExpr 调用 visit_expr(),...
```
注意，上述的TVM 按具体节点类型 dispatch里面，节点dispatch分为两种：
```text
TVM/TIRX 原生具体节点
    → 根据节点类型分派到固定的 visit_xxx_()

TileLang 新注册的 tl.tileop.*
    → 不产生新的 TIRX 节点类型
    → 统一属于 tirx.CallNode
```
因此，对于新注册的节点：
例如 `T.vadd` 注册的是名为 `tl.tileop.vadd` 的 `Op`，而不是新的
`VAddNode`。它属于visit_call_遍历的节点的某个参数

因此，`AddNode → visit_add_()` 表示 TIRX 原生标量加法节点的类型分派，
不表示 `T.vadd` 的处理路径。`T.copy`、`T.gemm` 和其他在src/op内注册的 `tl.tileop.*`,
也采用相同的 `CallNode + Call.op.name` 二次分派方式。

对于原生的节点：
未覆盖某个 `visit_xxx_()` 时，TVM 使用默认实现，不生成 TileLangIR，只按照
节点结构继续访问子 `Stmt` 和子 `PrimExpr`。覆盖以后，默认实现被替换
### 3. 使用 MLIR Python bindings 构造结果

TileLangIR Codegen 在运行时导入 `mlir.ir`，通过 Python Builder API 创建 Context、
Location、Module、Operation、Region、Block 和 SSA Value。LLVM/MLIR 已经编译在
依赖 wheel 提供的共享库中，因此 TileLang 自身的 CMake 编译过程不需要包含或
链接 LLVM Project。

MLIR Python bindings 采用 LLVM 官方 `llvm/eudsl` 发布的 wheel。`pyproject.toml`
以不限定版本的 `mlir-python-bindings` 声明该依赖，因此生成的 TileLang wheel
会包含对应的 `Requires-Dist` 元数据。

### 4. Codegen无优化、逐节点转换

Codegen 只负责表示转换，不改变程序语义和执行顺序：

- 不融合相邻计算；
- 不消除看似冗余的节点；
- 不改变循环顺序或内存布局；
- 不选择具体硬件指令；
- 不运行 MLIR optimization pipeline。

标准方言和自定义方言的选择，只取决于目标 Operation 能否完整表达源 TIRX
节点，而不取决于哪一种表示运行得更快。
