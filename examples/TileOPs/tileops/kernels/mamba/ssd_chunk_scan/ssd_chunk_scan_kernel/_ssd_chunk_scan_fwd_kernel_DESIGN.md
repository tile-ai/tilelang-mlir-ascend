# _ssd_chunk_scan_fwd_kernel 算子设计文档（SSDChunkScanFwdOp · GPU→NPU 迁移 · Developer 模式 MixCV）

> 项目：`ssd_chunk_scan`（TileOPs Mamba-2/SSD 族，NPU 仓 `examples/TileOPs`）；算子：`_ssd_chunk_scan_fwd_kernel`（迁移任务，harness 模式，**revision v1**，上一版 `history_version/design_v0.md`）。
> 源算子：`/home/tilelang/l00970450/TileOPs/tileops/kernels/mamba/ssd_chunk_scan.py`（GPU TileLang，三维 grid + SMEM swizzle + 双向 GEMM）。
> 用户指定编程模式：**developer**。物理核数实查（2026-09-17，本机 Ascend910B2C）：`NPUUtils.get().get_aicore_num()` 返回 **24**（混合 Cube+Vector 算子直接使用）。

## 相对上一版（design_v0）的关键调整（revision v1）

> 触发：Stage 2 检视不通过（3 阻塞 + 8 建议）。**已确认保留**：§0.1–§0.4 源算子三问解读、§1.6.0 算法族选型（分块双路径）、重设计 R1/R2/R5、OPT-2/OPT-4 优化项、分核三要素、R3 否决结论（K 侧因子移动）——本次仅修错处，未推翻任何已通过决策。

| # | 调整 | 为何不会再犯同一错误 |
|---|------|---------------------|
| 1 | **列因子广播形态错向修复（阻塞 1）**：v0 §3.2/§3.3 把「随列 j 变化的因子」（exp_s/dt）写成 `[bs,1]`（沿 N 轴行广播方向）——T.vmul 文档（§2.2.2）只支持 `[M,N]*[M,1]` 行广播，**无 `[M,N]*[1,N]` 列因子形态**；`[bs,1]` 会把列因子错向施加到 l 轴（bl==bs 时形状兼容、编译不报错、**静默算错**，检视数值演示 max\|Δ\|=7.32）。v1 统一约定：**行因子 `[M,1]`（沿 N 广播，文档明示）/ 列因子 `[1,N]`（沿 M 广播，数学正确方向）**；exp_s_all/dt 列因子改 `(1,Q)` 行形态存储、切片 `[0:1, s0:s0+bs]` 为 `[1,bs]`；`[M,N]⊙[1,N]` 列广播 vmul **无文档/生产先例**（GQA L645 为 `[M,N]⊙[M,1]` 行广播 Expert 先例）→ 显式列为 Stage 3 API 探针 P-1，fallback = vbrc 列展开（`vbrc([1,bs] → [bl,bs])`，T.vbrc.md §2.2.2 明示 (1,N)→(M,N) 合法）+ 同 shape vmul。verify_equiv.py 本就使用 `unsqueeze(0)`（[1,Q] 正确方向，L123），v1 重跑确认 EQUIV_PASS 不变。 | 修复后 §1.6.3/§3.2/§3.3/§4.3 四方形态声明一致；每个列因子表达式显式标注方向与探针/fallback 状态，不再静默假设。 |
| 2 | **§1.6.0 FLOPs 高估 2× 修正（阻塞 2）**：intra 因果路径 = `2·(Q(Q+1)/2)·P·C·H` = **3.234G**（v0 误写 6.46G，把 causal 面积 Q(Q+1)/2 又乘了 2 的系数错置）；总量 **6.46G**（v0 误写 9.7G），与 manifest mamba.yaml roofline 公式代入 w2 = 6.44G 交叉确认。派生数值同步：Cube 项 31µs → **20.7µs**。 | 复杂度表每行补算式；总量与 manifest 权威公式交叉核对（两个独立来源一致才落笔）。 |
| 3 | **对角掩码主选倒置修复（阻塞 3）**：v0 依「PL-1.11 记为 stale 线索」把 int16 vcmp+vselect 定主选、算术惩罚降备选——**误读**：PL-1.11 front-matter `status: verified` 且 2026-09-16 a13585dc 重验维持（kb_stale_check 的 stale 标记是工具链版本戳差异，非结论失效）。实测：vcmp int16 全形态标量化占壁钟 45%（aiv_scalar 94–97%），**算术惩罚掩码 4.4–14.2x（几何 8.53x）**。v1 主选改为**算术惩罚掩码**（整除域/无 stale band trace），并做 **band-carrying 分域**：R6 尾块域（Q%bs≠0，stale cb 列可能含未初始化 NaN——加法掩码无法清除 NaN，PL-1.11 NaN 免疫边界实证）**必须保留 vselect 选择语义** + 列 OOB 守卫。 | 引用 pattern-library 条目改为先核对 front-matter `status` 字段（verified/stale/overturned）再引用，版本戳差异 ≠ 结论失效；掩码形态按 trace 分域而非全局二选一。 |
| 4 | 建议级 8 项（详见各节）：① §3.3/§4.3 acc 初始化语义统一（history n-loop 首块 `initC=True` 覆写初始化，后续 `initC=False` 累加）；② R6 history K 维 stale band 分层论证（K 维 size 优先用真实 tn，tn<32 才钳位 32 + stale 带清零——消除 v0「钳位 tk≥32」与「size 排除 stale」的自相矛盾）；③ OPT-4 安全论证修正（exp_s_all 的 s≥l0 段是**死区**——full-lower 切片严格 s<l0、diag 不读 exp_s_all；深衰减下死区可至 fp32 inf 但不进任何链，v0「≤e^32 有限」表述失实）；④ §4 向量侧 buffer 显式 `T.alloc_ub`（T.vmul.md §2.3 条 1「操作数必须 UB」；其文档示例即 Developer 简单 jit 形态）；⑤ `vcast` f16→f32 上行补 `round_mode="rint"`（vcast.md dtype 矩阵：f16→f32 仅 rint）；⑥ §1.6.2 保留计数统一为 4 类（#10–#13）；⑦ 引用行号更正（flash_attn_dev 全文仅 113 行无 L208——v0 引的是 expert 版行号；标量乘先例 = GQA L519/L654，行广播先例 = GQA L645）；⑧ §3.3 s-loop 与 §6.1 统一为 `T.serial`、cb 装载统一为 cb_f32 直拷（消除伪代码中无预算的 cb_tile）。 | 全部为表述/引用层修正，逐条附证据定位。 |

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 references/migration-analysis.md）：先彻底读懂源算子（0.1–0.4），再做算法调研（结论落 §1.6.0，执行位置在 M0 与 M1 之间），最后判定硬件耦合性并给出 NPU 算法设计决策（0.5–0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**。Mamba-2 SSD（State-Space Dual）分块扫描输出前向：对每个 `(b, c, l, h, p)` 计算

$$
\text{out}[b, cQ{+}l, h, p] = Y_{\text{off}}[b,c,l,h,p] + Y_{\text{diag}}[b,c,l,h,p]
$$

其中历史路径（块间 prev_states 贡献）：

$$
Y_{\text{off}}[b,c,l,h,p] = e^{\text{dA}[b,h,c,l]} \cdot \sum_{n=0}^{N-1} C[b, cQ{+}l, g(h), n] \cdot S_{\text{prev}}[b,c,h,p,n]
$$

分块内因果路径（对角贡献）：

$$
Y_{\text{diag}}[b,c,l,h,p] = \sum_{s=0}^{l} \text{cb}[b,c,g(h),l,s] \cdot e^{\text{dA}[b,h,c,l] - \text{dA}[b,h,c,s]} \cdot \text{dt}[b,h,c,s] \cdot x[b, cQ{+}s, h, p]
$$

其中 `g(h) = h // (H/G)`（heads_per_group 分组），`S = C·Q`（seqlen = chunk 数 × chunk 长）。这是 SSD 半可分（semiseparable）矩阵 `M = diag(L) + off` 与向量作用的分块分解：off-diagonal 块秩 ≤ N（由 `C @ prev_states` 显式给出），对角块为因果下三角（由 `cb ⊙ exp(ΔdA) ⊙ dt` 给出）。

**规约语义（累加顺序）**：
- 两条路径共用一个 fp32 累加器 `acc[bl, bp]`，先 history 后 intra（源码程序顺序）。
- history：沿 n 维分块（`n_blk` 外层 `T.Pipelined` 循环），每块一次 `T.gemm` 累加（`initC` 缺省 = 累加语义，`hist_acc += c_tile @ state_tile`）；块内 K 维累加由 Tensor Core mma 完成。**块间顺序 = n_blk 升序**，块内顺序由硬件 mma 决定。
- intra：沿 s 维分块（`s_blk` 升序），每块一次 `T.gemm` 累加进同一 `acc`；块内 K 维累加由 mma 完成。
- NPU 重设计改变并行度会改变块内累加分组（Cube 分形累加顺序 vs mma），但全程 fp32 累加，块间顺序保持（n_blk/s_blk 升序），舍入差异在 fp32 ulp 量级（2^-24），远小于 fp16 操作数量化噪声（2^-11），对 §8.2 容差无影响（§0.6 R1 论证）。

**dtype 语义（中间精度/提升/截断）**：
- 输入：`x/cb/C/dt` 为 `dtype`（fp16 或 bf16，manifest：float16|bfloat16）；`dA_cumsum/prev_states` 为 fp32；输出 `out` 为 fp32。
- `accum_dtype = "float"`（fp32）：`acc/hist_acc/exp_dA_l/exp_l/exp_s/dA_smem/dt_smem` 全部 fp32。
- **量化的关键位置**（决定数值基线）：
  1. `c_tile/state_tile/cb_tile/x_tile`：保持 `dtype` 进 gemm；其中 `state_tile` 装载时 `prev_states`（fp32）被 `T.cast(…, dtype)` 降精度。
  2. `lcb_cast[ll,ss] = T.cast(cb·exp_l·exp_s → dtype)`：因子乘积在 fp32 域计算后**舍入到 dtype** 再进 gemm。
  3. gemm 累加全程 fp32；`out` 直接写 fp32（无最终舍入）。
- NPU 版必须保持同构数值路径（fp16/bf16 操作数 + fp32 累加 + 相近的量化点），否则对 golden（纯 fp32 PyTorch）的偏差水平会偏离源实现的已验证基线。

**边界语义**：
- 越界读保护：`l_abs < Q`、`s_abs < Q`、`n_abs < N`、`p_abs < P` 的 `if_then_else` guard + `T.min(·, dim-1)` safe 索引（OOB 位置填 `cast(0.0, dtype)`）——即 **pad-0 语义**。
- 因果掩码：对角块内逐元素 `valid = (s_abs <= l_abs)`，无效位置置 0；上三角 s-block 由循环边界 `ceildiv(l0 + block_l, block_s)` **完全跳过**。
- **dA 单调非增假定**：源码注释明确 `dA_cumsum is non-increasing`（Mamba-2 中 dA = -cumsum(dt·A) ≤ 0），L-side anchor 因子化的非正性保证依赖它（§0.4 #9）。测试输入 `dA = -cumsum(rand)` 满足。若上游给出非单调 dA，因子化指数可能为正、exp 可达大值（fp32 仍有限），语义不崩坏但数值路径偏离设计假定——列入 §9 风险。
- NaN/inf 传播：源码无显式 NaN guard（无 softmax 归一化分母），逐元素乘加链保持 NaN 传播；exp 参数在合法输入域（dA ≤ 0 → 指数 ≤ 0）无溢出。
- host 侧：`forward` 仅做 `.contiguous()`（视图/拷贝，不改数值）与 shape 校验；无预处理/后处理计算。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（直接采用 NPU 仓已移植的 `examples/TileOPs/tileops/testing/mamba2_reference.py` 中的 `ssd_chunk_scan_fwd_ref` 函数，纯 fp32 materialize 实现，独立于本设计的 NPU 算法）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| x | (B, S, H, P) | fp16/bf16 | 主输入；S = NC·Q，seqlen-fused |
| cb | (B, C, G, Q, Q) | 同 x | 预计算 C@B 耦合矩阵；group-owned（G < H 时被同组 head 共享） |
| dA_cumsum | (B, H, C, Q) | fp32 | 衰减累积和；单调非增 |
| C_mat | (B, S, G, N) | 同 x | readout 矩阵；seqlen-fused、group-owned |
| prev_states | (B, C, H, P, N) | fp32 | 进入每个 chunk 的状态；P 在 N 前（official 约定） |
| dt | (B, H, C, Q) | 同 x | 步长 |
| **out（输出）** | **(B, S, H, P)** | **fp32** | seqlen-fused 输出；**非转置布局**（与 x 同 layout）；shape 由 `register_fake` 确认：`(batch, num_chunks*chunk_len, n_heads, d_head)` fp32 |

维度记号（后文统一）：`B=batch, C=num_chunks, Q=chunk_len, H=n_heads, P=d_head, N=d_state, G=n_groups, S=C·Q`；`HEADS_PER_GROUP = H//G`；shape_rules：`S == NC*Q`、`H % G == 0`（manifest 约束，host `_validate` 强制）。

### 0.3 实现算法解读（怎么算）

**Kernel 结构**：`@tilelang.jit(out_idx=[-1])` 工厂 `_ssd_chunk_scan_fwd_kernel(batch, …, dtype, diagonal_microtile_size=32)` 返回内层工厂 `kernel_func(block_l, block_p, block_n, block_s, threads, num_stages)`，内含 `@T.prim_func main`。三维 grid：

```python
T.Kernel(ceildiv(Q, block_l) * ceildiv(P, block_p),  # 轴 0: blp = l-tile × p-tile
         B * C,                                        # 轴 1: bc = batch×chunk
         H,                                            # 轴 2: head
         threads=threads)                              # 默认 threads=128（4 warps）
```

解码：`bl = blp // P_tiles, bp = blp % P_tiles, bz = bc // C, bc_idx = bc % C, bg = bh // HEADS_PER_GROUP`。默认 config：`block_l=64, block_p=64, block_n=min(64, N), block_s=64, threads=128, num_stages=3`。

**计算步骤分解**（源码每个计算语句均归入某步骤；GPU TileLang 源码 → 语义公式）：

| 步骤 | 计算（源码语句） | 输入 | 输出 | 对应语义公式部分 |
|------|-----------------|------|------|-----------------|
| S1 | `c_tile[ll,nn] = guard(C_mat[bz, cs+l, bg, n])`（`T.Parallel` 逐元素 + safe 索引） | C_mat | c_tile（SMEM, [bl,bn], dtype） | history 的 C 读出 |
| S2 | `state_tile[nn,pp] = guard(cast(prev_states[bz,bc,bh,p,n]))`（**转置装载**，`T.Parallel`） | prev_states | state_tile（SMEM, [bn,bp], dtype） | history 的 S_prev^T（**转置**） |
| S3 | `T.gemm(c_tile, state_tile, hist_acc)`（n-loop 内，累加） | c_tile, state_tile | hist_acc（fragment, [bl,bp], fp32） | Y_off 的 Σ_n |
| S4 | `dA_smem[q]/dt_smem[q] = dA_cumsum/cast(dt)`（Q 长度 SMEM 缓存 + `T.sync_threads`） | dA_cumsum, dt | dA_smem/dt_smem（fp32） | 因子缓存 |
| S5 | `exp_dA_l[ll] = guard(exp(dA_smem[l]))` | dA_smem | exp_dA_l（[bl], fp32） | history 行缩放因子 |
| S6 | `acc += hist_acc · exp_dA_l[ll]`（行广播乘，`T.Parallel` + guard） | hist_acc, exp_dA_l | acc | Y_off = e^{dA_l}·Σ |
| S7 | `exp_l[ll] = exp(dA_l − anchor)`，anchor = dA_smem[l0]（l-tile 级 hoist） | dA_smem | exp_l（[bl], fp32） | intra 因子化（L-side anchor） |
| S8 | s-loop（`T.Pipelined(ceildiv(l0+bl, bs))`，上三角跳过）内：`cb_tile/x_tile` 装载（guard + `T.Parallel`） | cb, x | cb_tile（[bl,bs]）、x_tile（[bs,bp]），dtype | intra 操作数 |
| S9 | full-lower 块（`s0+bs ≤ l0`）：`exp_s[ss] = exp(anchor − dA_s)·dt[s]`；`lcb_cast = cast(cb·exp_l·exp_s)` | cb_tile, exp_l, exp_s | lcb_cast（fragment, [bl,bs], dtype） | intra 因子化乘积 |
| S9' | 对角块：微块路径（M=32：2×2 微块，(1,0) 因子化 row/col_factor，(0,1) 置零，(0,0)/(1,1) 直接 `exp(dA_l−dA_s)·dt` + valid 掩码）或 fallback 全块直接 exp + `if_then_else(valid, …, 0)` | cb_tile, dA_smem, dt_smem | lcb_cast | intra 对角因果项 |
| S10 | `T.gemm(lcb_cast, x_tile, acc)`（累加） | lcb_cast, x_tile | acc | Y_diag 的 Σ_{s≤l} |
| S11 | `out[bz, cs+l, bh, p] = acc[ll,pp]`（guard + `T.Parallel` 写回） | acc | out（GM, fp32） | 输出 |

**数据流与内存访问模式**（源硬件视角，覆盖全部 buffer）：

```
GM[x] ──T.Parallel guard──▶ SMEM[x_tile, bs×bp]  ─┐
GM[cb] ──T.Parallel guard──▶ SMEM[cb_tile, bl×bs] ─┼─▶ lcb_cast(fragment, cast缩放) ─▶ mma ─┐
GM[dA]──(Q标量缓存)──▶ SMEM[dA_smem] ─▶ exp_l/exp_s/exp_dA_l(共享/寄存器) ──────────────┤
GM[dt] ──(Q标量缓存)──▶ SMEM[dt_smem] ─────────────────────────────────────────────────┤
GM[C]  ──T.Parallel guard──▶ SMEM[c_tile, bl×bn] ─▶ mma(hist) ─▶ hist_acc ─▶ 行缩放 ──┤
GM[prev]──T.Parallel guard(转置+cast)──▶ SMEM[state_tile, bn×bp] ────────────────────┘
                                                                  ▼
                    fragment acc[bl×bp]（fp32 累加器，跨 n-loop/s-loop 驻留寄存器）
                                                                  ▼
                              GM[out, fp32]（guard 写回）
```

标注：`T.annotate_layout(make_swizzled_layout)` 施加于 c_tile/state_tile/cb_tile/x_tile（bank conflict 消除）；`T.Pipelined` 提供 n-loop/s-loop 的 SMEM 多级流水（num_stages=3）；`T.sync_threads()` 在 dA/dt 缓存与 exp_l 预计算后同步线程。

**循环与并行结构**：三维 grid（l×p tiles, B·C, H）× threads=128；核内两级 `T.Pipelined`（n-loop: `ceildiv(N, bn)`；s-loop: `ceildiv(l0+bl, bs)`，causal 上界递增）；块内 `T.Parallel`（fragment 逐元素并行，线程映射由 TileLang 布局推断）。

**host 侧逻辑**：`SSDChunkScanFwdKernel.forward` → 6 输入 `.contiguous()` → `_ssd_chunk_scan_fwd_wrapped`（`torch.library.custom_op("top::ssd_chunk_scan_fwd")`）→ 工厂调用。`register_fake` 返回 `x.new_empty((B, C·Q, H, P), fp32)`。GPU 版另有 `autotune_configs`（6–8 组 block 配置搜索，NCU 证据注释）——NPU 仓 K8 适配已移除。**host 侧无计算逻辑**（无 im2col/reshape/cast 预处理），仅有 lru_cache 工厂与 config 传递。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tiling（c/state/cb/x tile 驻留共享内存） | 减少 GM 重复访问 | 分块装载 + 块内复用 | shared memory（48KB/SM 级） | 硬件强相关（NPU 有 L1/UB 等价） |
| 2 | 三维 grid + H 独立维 | 负载均衡（注释 "Grid redesign: separate H dimension"） | (l×p, B·C, H) 三轴并行，block 数充足 | CUDA grid 调度 | 硬件强相关（本项目仅一维 Kernel） |
| 3 | SMEM swizzle（`make_swizzled_layout`） | 消 bank conflict + Tensor Core 布局适配 | 索引重排 | SMEM bank 结构（32 bank × 4B） | 硬件强相关（NPU 无 bank conflict 概念） |
| 4 | coalesced 装载（state_tile 注释 "consecutive threads vary nn"） | 128B 合并访存 | 线程-索引映射沿连续维 | memory coalescing | 硬件强相关（NPU 由 MTE 引擎搬运，无线程映射） |
| 5 | Tensor Core mma（`T.gemm`） | 矩阵算力 | fp16 mma fragment 累加 | Tensor Core | 硬件强相关（NPU 有 Cube 等价） |
| 6 | `T.Pipelined`（num_stages=3）异步流水 | 访存计算重叠 | SMEM 多缓冲 cp.async | 异步拷贝引擎 | 硬件强相关（NPU 有 MTE 等价） |
| 7 | dA/dt 的 Q 标量 SMEM 缓存 | 消 exp/对角路径的 L2 往返 | 一次装载全 chunk 复用 | shared memory | 硬件强相关（存储介质耦合，算法意图可移植） |
| 8 | exp_dA_l / exp_l 预计算 hoist | 消循环内重复 exp | l-tile 级一次计算 | — | **可移植**（纯算法层） |
| 9 | **L-side anchor 因子化**：`exp(dA_l−dA_s) = exp(dA_l−anchor)·exp(anchor−dA_s)`，anchor = dA[l0] | MUFU（exp 单元）计数从 bl·bs 降为 bl+bs；**数值稳定**（两因子指数均 ≤ 0，无溢出；深衰减 chunk 下不产生下溢为 0 的伪影） | full-lower 块因子分离 | — | **可移植**（纯算法层，依赖 dA 单调性这一语义属性） |
| 10 | 上三角 s-block 完全跳过（loop bound `ceildiv(l0+bl, bs)`） | 省 ~50% 对角 GEMM 计算 | causal 结构裁剪 | — | **可移植**（纯算法层） |
| 11 | 微块 2×2 因子化（M=32，仅 block_l=block_s=64 启用） | 对角块 MUFU 削减（~50%）+ (0,1) 微块零填充跳过 | diag 块拆微块，(1,0) 严格下三角因子化 | GPU 标量 MUFU 吞吐瓶颈 | 算法层可移植，**但收益机制硬件相关**（NPU 向量整块发射，微块化反而增加 op 数） |
| 12 | register 累加（fragment acc 跨循环驻留） | 消中间写回 | 寄存器文件容量 | 大寄存器堆 | 硬件强相关（NPU 用 L0C/UB 等价） |
| 13 | threads=128 / autotune 配置集 | 占用率与 ILP 平衡 | warp 调度 | warp 结构 | 硬件强相关（NPU 无 threads 概念） |
| 14 | `lru_cache` 工厂（shape 特化编译缓存） | 消重复编译 | Python 层缓存 | — | 可移植 |

> 识别不出来 ≠ 不存在。以上 14 项覆盖源码全部显式优化注释与结构（swizzle/流水/hoist/因子化/微块/负载均衡）。迁移中 #1–#7、#12–#13 需按 §0.5 处置；#8–#10、#14 为算法层/工程层直接保留；#11 的**意图**（对角块代价削减）须由 NPU 结构承接（见 §0.5 行 11）。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定依据：migration-analysis.md §5.3 映射表 / examples 佐证 / docs 条目；Phase R 调研结论〔§1.6.0〕作为判定输入——调研确认分块双路径即该数学语义的并行最优结构族，源算法未被替换，重设计方向为「结构保留 + NPU 向量化重构」。）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（§0.1 全部） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 以此为基线 |
| 双路径分解（history + intra-causal）+ causal 上三角跳过（#10） | 算法 | 无（纯数学结构） | — | **保留** | 调研（§1.6.0 R3/R4）确认分块双路径为最优结构；块循环边界 `ceildiv(l0+bl, bs)` 语义照搬 |
| L-side anchor 因子化（#9）+ exp hoist（#8） | 算法 | 无 | — | **保留** | 数值稳定结构必须保留；exp 由向量 `T.vexp`（fp32）实现，hoist 粒度升级为 per-(b,c,h) 的 `[Q,1]` 向量预计算（§0.6 R3） |
| dA/dt 的 Q 标量缓存（#7） | 算法/优化边界 | shared memory | 有（UB） | **等价替换** | `T.alloc_shared((Q,1), fp32)` GM→UB 一次拷贝，全任务复用（映射表：SMEM→UB，migration-analysis.md §5.3 行 1） |
| SMEM tiling（#1） | 优化 | shared memory | 有（L1 512KB / UB 192KB） | **等价替换** | gemm 操作数 tile 走 `T.alloc_shared`（Developer 模式自动映射 L1，flash_attn_npuir_dev.py L29–33 先例）；向量侧 buffer 映射 UB（§4.5 预算） |
| Tensor Core mma（#5） | 优化 | Tensor Core | 有（Cube） | **等价替换** | `T.gemm`（docs/Tilelang.language/线性代数操作/T.gemm.md：Developer Op，fp16/bf16 × fp32 dst；映射表行 3） |
| coalesced 装载（#4） | 优化 | 线程-内存合并 | 有（MTE 引擎） | **等价替换** | `T.copy` slice 形态（2D strided tile 搬运由 MTE 完成，无逐线程映射；flash_attn_npuir_dev.py L30/L58/L80、GQA D1 实证 slice-form 优于 load_nd2nz 处理跨步区） |
| `T.Pipelined` 异步流水（#6） | 优化 | cp.async | 有（MTE + 双缓冲） | **等价替换** | `T.Pipelined(num_stages=2)`；num_stages 按 UB/L1 容量与 buffer 数重算（源 3 → 设计 2，§6.3 论证） |
| register 累加（#12） | 优化 | 寄存器堆 | 有（L0C 128KB / UB） | **等价替换** | `acc` 用 `T.alloc_fragment([bl,bp], fp32)` 作 gemm dst（flash_attn_npuir_dev.py L41 `acc_o` 同款，Developer 模式 fragment 自动映射 L0C）；bl·bp=4096 × 4B = 16KB ≤ 128KB ✓ |
| 三维 grid + H 独立维（#2） | 优化 | CUDA 三维 grid | **无**（本项目 `T.Kernel` 仅一维） | **重新设计** | → §0.6 R1：三维乘积展开为一维逻辑任务 + persistent 24 核核内串行（ascend-constraints.md 限制 1；core-split-strategy.md 三要素） |
| SMEM swizzle（#3） | 优化 | SMEM bank 结构 | **无此问题** | **舍弃** | NPU 无 bank conflict；NZ 分形布局由框架/load_nd2nz 处理（映射表行 7：手动 swizzle 无意义）。**舍弃理由**：其目的（bank conflict 消除）在 NPU 不存在，等价收益由 MTE 搬运与 Cube 分形布局原生获得 |
| threads=128 / autotune（#13） | 优化 | warp 调度 | **无** threads 概念 | **舍弃** | `T.Kernel(threads=)` 在 npuir 无效果（pattern-library TRAP-threads-kwarg-noop，stale 条目仅作线索，NPU 仓 wrapper K9 已移除 threads 传递）；autotune 已被 NPU 仓 K8 移除。**舍弃理由**：调度粒度由核数与向量化轴替代表达 |
| 微块 2×2 因子化 M=32（#11） | 优化 | GPU 标量 MUFU 吞吐 | 无同款瓶颈 | **舍弃（意图承接）** | **舍弃理由**：其目的是削减标量 exp 计数（GPU MUFU 稀贵）；NPU 向量 `T.vexp` 整块发射（一次 op 处理 [bl,bs] 全块），微块化将 7 op 拆为 28 op（4 微块 × 7），在发射开销 ~0.5µs/op 主导域（CONST-vector-launch-overhead 口径）为**负优化**。**意图承接**：对角块 exp 代价削减由「惩罚矩阵核级预计算 + vbrc/vmul 方向化因子链」的向量链承担（§0.6 R4，v1 惩罚主选），微块 (0,1) 置零跳过的意图由惩罚掩码（整除域）/vselect（band 域）承接 |
| 对角块逐元素 `if_then_else` valid 掩码 | 实现 | 标量分支 | 有（向量算术惩罚/选择） | **重新设计** | → §0.6 R4：**算术惩罚掩码为主选**（整除域，PL-1.11 verified 实测 4.4–14.2x vs vcmp 标量化）；band-carrying 域（尾块 stale）保留 vselect 选择语义（NaN 免疫） |
| 尾块 guard `if_then_else` + safe 索引装载 | 实现 | 标量分支 | 有（slice 尾块拷贝） | **重新设计** | → §0.6 R6：`T.min` 尾块尺寸 + slice-form `T.copy` + size-form `T.gemm`（GQA E7 形态） |

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（对照 Phase R 调研结论：源算法族保留，重设计聚焦 NPU 结构落地；源算法优化手段的意图承接已在 §0.5 逐项标注）：

**重设计项 R1: 三维 grid → 一维 persistent 分核**

- **源方案**：`T.Kernel(l_tiles·p_tiles, B·C, H, threads=128)` 三维并行，block 数 = 4·16·48 = 3072（w2）等；意图 = 负载均衡 + 充足并行度。
- **NPU 新算法**：一维 `T.Kernel(NUM_KERNELS=24, is_npu=True) as (kernel_id, subid)`（persistent），逻辑任务 = `(b, c, h)`（**任务粒度取 chunk-head 而非 l×p tile**——dA/dt/exp 因子预计算在任务内一次完成，摊薄向量 op 发射数；核内 `p-loop × l-loop` 串行展开 tile）。任务解码 `cid = task_id·24 + kernel_id`（轮转），`num_local_tasks = T.ceildiv(num_logical − kernel_id, 24)`，**ghost 任务钳位** `cid = T.min(·, num_logical − 1)`（T.ceildiv 对负数返回 1 的陷阱，GQA 2026-09-15 工具链实证 + 绕法）。核内循环边界为静态推导（task_id/kernel_id 派生 PrimExpr，循环前计算一次）。
- **语义保持论证**：任务重排只改变 (b,c,h) 的计算位置与顺序，每个任务的计算语义与累加顺序（n_blk/s_blk 升序）不变；ghost 钳位任务重算已有任务的相同输出（相同输入 + 相同指令序 = bit-identical 写回，GQA 同款论证）；fp32 累加分组与源一致（块内由 Cube 分形完成，块间升序 T.serial）——数值上块内 mma 顺序差异属 fp32 ulp 级（2^-24），远小于 fp16 操作数噪声（2^-11），容差内等价。边界：B·C·H=1 退化（单任务）时 persistent 24 核仅 1 核有效 + 23 ghost 重算——bit-identical 安全，性能由 Stage 4 shape 分派优化（不在 Stage 1 特化）。

**重设计项 R2: state_tile 转置装载 → 直装载 + `b_transpose=True`**

- **源方案**：`state_tile[nn,pp] = prev_states[b,c,h,p,n]`（**转置**装载，[bn,bp] SMEM），`T.gemm(c_tile[bl,bn], state_tile[bn,bp], hist_acc)`；注释表明迭代序 (pp,nn) 换取 nn 连续的 coalesced 128B 装载。
- **NPU 新算法**：`state_t[pp,nn] = prev_states[bz,bc_idx,bh,p0+pp,n0+nn]` **直装载**（[bp,bn]，内存布局原生方向：相邻 pp 行 stride=N、行内 nn 连续），`T.gemm(c_tile, state_t, acc, initC=(n_blk==0), b_transpose=True)`——转置由 Cube gemm 的 B 转置参数在分形内完成（docs T.gemm.md §2.1 `b_transpose` 参数；flash_attn_npuir_dev.py L59 `T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)` 先例）；gemm 输出直写统一累加器 acc（§4.3 单 buffer）。
- **语义保持论证**：`hist[ll,pp] = Σ_nn c[ll,nn]·state_t[pp,nn]` 与源 `Σ_nn c[ll,nn]·state[nn,pp]` 逐元素恒等（转置仅改变操作数布局，K 维求和序由 Cube 分形决定，fp32 累加同上容差内等价）；装载 guard 语义不变（§0.6 R6 统一处理）。**收益**：消除转置索引的跨步写（MTE 直拷连续行），prev_states 的 fp32→dtype cast 由 Developer 模式 `T.copy` 跨 dtype 自动插入 VCast（docs T.copy.md §2.3 条 4）。

**重设计项 R3: 因子计算的标量/fragment 形态 → 向量广播链 + per-(b,c,h) 预计算**

- **源方案**：`exp_dA_l/exp_l/exp_s` 为 `[bl]/[bs]` 标量序列，`T.Parallel` 循环逐元素 `T.exp`（GPU 标量 MUFU）；lcb 缩放为 `T.Parallel` 二重循环 `cb·exp_l[ll]·exp_s[ss]`（行/列因子各自逐元素展开）。
- **NPU 新算法**：
  1. 所有 1D 因子统一 2D 形态并**显式区分方向**（v1 修订）：**行因子 `[M,1]`（沿 N 广播，vmul 文档明示 `[M,N]*[M,1]` 形态）**——exp_l/dA_ub 列形；**列因子 `[1,N]`（沿 M 广播，数学正确方向）**——exp_s_all/dt 因子以 `(1,Q)` **行形态**存储（`dA_ub_row/dt_ub_row`），切片 `[0:1, s0:s0+bs]` 即 `[1,bs]` 列因子。`[M,N]⊙[M,1]` 行广播有 GQA L645 Expert 生产先例；**`[M,N]⊙[1,N]` 列广播无文档/生产先例**（T.vmul.md §2.2.2 仅列 `[M,N]*[M,1]` 行广播与 `[1,N]*[1,1]` 向量×标量）→ 列因子 vmul 列为 **Stage 3 API 探针 P-1**，fallback = `vbrc([1,bs] → [bl,bs])` 列展开（T.vbrc.md §2.2.2 明示 (1,N)→(M,N) 合法）+ 同 shape vmul（+1 op/处，UB +16KB/处）。
  2. **per-(b,c,h) 预计算**（任务头一次）：`dA_ub = copy(dA[b,h,c,:])`、`dt_ub = copy(cast(dt))`（(Q,1) fp32）及其 `(1,Q)` 行形态双份（共 ~4KB），UB 驻留全任务复用。
  3. **per-l-tile**：`exp_l = vexp(vsub(dA_ub[l0:l0+bl], anchor_col [bl,1]))`（[bl,1] 行因子，≤ 1）；**exp_s_all 一次性（(1,Q) 行形态）**：`exp_s_all = vmul(vexp(vsub(anchor_row (1,Q), dA_ub_row (1,Q))), dt_ub_row (1,Q))`——全程同 shape [1,N] 运算（文档支持），full-lower s-block 直接切片 `exp_s_all[0:1, s0:s0+bs]` 为 [1,bs] 列因子（替代源码 per-s-block 的 [bs] 重算，w2 每 (b,c,h) 省 ~6 次 [bs] 级计算）。**数值安全（v1 修正论证）**：full-lower 切片位置满足 s0+bs ≤ l0 ⟹ 切片内 s < l0 ⟹ dA 单调非增 ⟹ anchor−dA_s ≤ 0 ⟹ 值 ≤ dt ≤ 0.11，**恒安全**；`s ≥ l0` 段（含深衰减下指数可至 fp32 inf——v0「≤e^32 有限」表述仅对真实分布成立，已修正）是**死区**：full-lower 切片上界严格 s < l0 不读取该段，对角块不使用 exp_s_all（直接差分链）——死区值（含 inf）不进任何 gemm/向量链；仅当切片越界（实现 bug）时暴露，由 L0 精度用例拦截。
  4. lcb 缩放链（**衰减因子保持在 cb 侧**，§1.6.1 机器验证定型）：full-lower `lcb = vcast(vmul(vcast(cb→f32), exp_s 列因子 [1,bs] 探针/fallback), →dtype)`（exp_l 已由 R5 提出）；对角块直接差分链（见 R4）。
- **语义保持论证**：全部为恒等变形——`exp(dA_l−a)·exp(a−dA_s) = exp(dA_l−dA_s)`（指数律，源码同式）；切片复用与逐块计算是**同一算式的求值位置移动**（同式同 dtype 路径，严格等价，verify_equiv.py OPT-4 torch.equal 级验证——脚本列因子即 `unsqueeze(0)` [1,N] 正确方向）。**注**：候选变形「K 侧因子移动（dt/exp_s 吸收进 x 后 cast）」经 verify_equiv.py 机器验证**否决**——fp16 噪声放大 38–135× 超 atol（被 cast 的 x 侧操作数不随衰减缩小，详见 §1.6.1 否决行）；设计保持源码的 cb 侧因子化数值路径。
- **意图承接**：源 #7（dA/dt 缓存）→ UB 驻留双形态 buffer + 全任务复用；源 #8（exp hoist）→ 预计算粒度从 l-tile 升格到 (b,c,h) 任务级（exp_s_all）。

**重设计项 R4: 对角块因果掩码 → 算术惩罚掩码主选 + band-carrying 域 vselect（v1 修订：主备倒置）**

- **源方案**：对角块逐元素 `valid = (s_abs <= l_abs)` 的 `if_then_else` 标量分支（+ 微块变体的分区掩码）。
- **NPU 新算法（v1 主选：算术惩罚掩码，整除域）**：惩罚矩阵 `(i < j ? 大负 : 0)` 是**核级常量**（与数据无关）：核初始化时一次 `T.arange` 物化行列索引差（fp32 精确整数，PL-1.11 同款 `[-1,1]` 负步长 arange 形态）→ `pen = clamp/arange 派生的 0/−PEN 矩阵`（PEN = 1e30，有限哨兵——GQA PL-1.11 用 −1e38 同理：有限大负数规避 −inf−(−inf) NaN 路径）。任务循环内对角块链为**直接差分**纯向量链：`vbrc(dA_l → [bl,bs])`、`vbrc(dA_s → [bl,bs])` → `vsub(diff)`（in-place）→ **`vadd(diff, pen)`（惩罚融进指数，in-place）** → `vexp` → `vmul(vcast(cb→f32), diff)` → `vmul(·, dt 列因子 [1,bs])` → `vcast(→dtype)`。s > l 位指数 + (−PEN) → 大负 → exp 下溢为**精确 +0.0**；s ≤ l 位惩罚恒 0，指数不变——**数学严格等价**（verify_equiv.py OPT-6 惩罚形精确验证：下三角逐位原值、上三角精确 +0）。**主选依据（PL-1.11，pattern-library/attention.md，front-matter `status: verified`，toolchain 2026-09-15 6797758 / 2026-09-16 a13585dc 重验维持）**：int16 `vcmp` **全形态标量化**（aiv_scalar 94–97%、单链 236µs @[32,256]、对角块聚合 ≈ 壁钟 45%），算术惩罚掩码实测 **4.4–14.2x（几何 8.53x）**；v0 曾误将该条目当 stale 线索而倒置主备——stale 标记源于工具链版本戳差异，结论本身已在当前工具链重验维持。
- **band-carrying 域（R6 尾块，Q%bs≠0 或 ts<bs）：保留 vselect 选择语义**——PL-1.11 NaN 免疫边界实证：**加法掩码无法清除 NaN**（NaN+x=NaN，vmax/vmin 钳位只治 ±Inf）；尾块域的 stale cb 列（[ts,bs) 未初始化 UB 位）可能为 NaN，惩罚加法会把它带进 lcb 污染 gemm K 维整列输出。该域对角链改为：`... vmul(·, dt 列因子) → vselect(mask_const, ·, zero_f32) → vcast`（vselect 按元素**值替换**，未选中 NaN 不参与算术）+ **列 OOB 守卫**（j ≥ ts 的列强制 0，GQA E7 J_lim 形态：额外一次 `vcmp`/惩罚列因子或在装载后以 `vbrc(0)` 清零 stale 列）。mask 常量与 zero_f32 仅 band 域 trace 分配（工厂 trace-time 按 `Q % bs` 分派，整除域不分配）。**当前 manifest/test 全部 Q∈{64,128,256} 对 bs=64 整除——主 trace 走惩罚形态**；band 域为契约完备性路径（L0-5 用例覆盖）。
- **NaN/inf 安全（双域）**：直接差分的下三角指数恒 ≤ 0（s ≤ l ⟹ dA_l ≤ dA_s）；上三角正指数 ≤ exp(bs 步衰减)（真实分布 ~e^32；深衰减 stress 可至 fp32 上限——惩罚域中被 −PEN 吸收为 −PEN（大数吃小数）→ exp 下溢 0，vselect 域中被选路丢弃）；两域 cast 输入均有限（惩罚域：指数和 ≤ 0 或 = −PEN；vselect 域：选后值域受控）。**对角块 L-side 因子化（`exp_l_col × exp_as_col`）维持否决**：对角块内 s ≥ l0 ⟹ anchor−dA_s ≥ 0 ⟹ exp_as ≥ 1，块内衰减 > ~88 时 fp32 溢出污染下三角（真实分布 24σ 罕见但非无条件安全），连同护栏记入 §1.6.3 Stage 4 备选。
- **语义保持论证**：掩码语义 `s ≤ l ? value : 0` 双域精确保持（惩罚域 0 为 fp32 下溢精确零；vselect 域为选路精确 0）；verify_equiv.py OPT-6 双形均 EQUIV_PASS（§1.6.1 表）。
- **意图承接**：源 #11 微块 (0,1) 置零与逐元素 valid 分支的意图（无效面积零贡献 + 对角 exp 代价削减）由惩罚掩码一次性向量化 + 直接差分整块 vexp 承接（NPU 向量 op 整块发射，微块拆分反而增加 op 数——§0.5 行 11 舍弃依据）。

**重设计项 R5: history 行缩放与 intra 行因子合并（数学优化，源于 §1.6.1 采纳项 OPT-2）**

- **源方案**：`acc += hist_acc · exp_dA_l[ll]`（history 行缩放，[bl,bp] 逐元素）与 intra 路径分离；exp_dA_l 独立计算。
- **NPU 新算法**：利用恒等式 `exp(dA_l) = exp(dA_l − anchor)·exp(anchor) = exp_l[ll]·exp(anchor)`（指数律），把 history 缩放并入统一行因子：`acc = exp_l ⊙ ( exp(anchor)·hist_gemm + Σ_full-lower gemm_partial ) + Σ_diag gemm_partial_diag`。执行序：`acc = hist_gemm`（gemm 直写，initC 首块覆写）→ `acc = vmul(acc, exp_anchor_scalar)`（标量乘）→ full-lower s-blocks `gemm(lcb_fl, x_tile, acc, initC=False)` 累加（lcb_fl = cast(cb·exp_s)，exp_l 已提出——**cb 侧因子化**）→ `acc = vmul(acc, exp_l_col)`（一次行广播缩放）→ 对角块 `gemm(lcb_diag, x_tile, acc, initC=False)` 累加。对角块恰为 s 序最后一块（s0 ≤ l0 < s0+bs 由循环边界保证），程序序天然满足"diag 在行缩放之后"。
- **语义保持论证**：数学恒等（指数律 + 分配律），fp32 域内重结合的差异为 0.5 ulp fp32 级；**数值安全**：exp(anchor) 单独计算可能下溢（深衰减 chunk，anchor ≤ −88 时 exp → 0）——**该下溢在数学上正确**（此时 exp_l·exp(anchor)·hist 的真值亦 ~ e^{dA_l}，dA_l ≤ anchor ≤ −88 → 真值 < e^{−88} ≈ 6e−39，与 hist（O(1) 值域）乘积 < 1e−37，相对 out 的 O(0.1–10) 值域为可忽略量；且**源码同样有此下溢**——源 exp_dA_l = exp(dA_l) 直接下溢为 0，行为一致）。verify_equiv.py 覆盖深衰减角点（§1.6.1 表）。
- **收益**：每 l-tile 省 1 次 [bl,bp] 行广播缩放 + 1 次 exp_dA_l 独立计算（op 发射数 −2/l-tile）。

**重设计项 R6: 尾块/非整除 guard 装载 → slice 尾块拷贝 + size-form gemm**

- **源方案**：所有 tile 装载用 `T.Parallel` 逐元素 + `if_then_else(guard)` + `T.min` safe 索引（标量分支填 0）。
- **NPU 新算法**：`tail = T.min(bl, Q − l0)` 等尾块尺寸 → `T.copy(GM[l0 : l0+tail, …], tile[0:tail, …])`（slice-form，尾块裁剪拷贝，docs T.copy.md §2.2.2 条 2–4 尾块借用规则）+ `T.gemm(…, size=[tmc, tk, tnc], initC=…)`（size-form 指定有效范围，docs T.gemm.md §2.1）。**分形钳位分层（v1 修订，消除 v0「钳位 tk≥32」与「size 排除 stale」的自相矛盾）**：M/N 维 `tmc = max(16, ceil16(tail_m))`、`tnp = max(16, ceil16(tail_p))`——stale 行/列带 [tail, tmc) 参与 gemm 但**只污染 stale 行/列自身的输出**（写回裁剪 [0:tail] 丢弃，不污染有效行列）；**K 维优先用真实值 `tk = tn`（不钳位）**——只要 tn ≥ 32（分形 K 下限）则 gemm K 维无 stale band（装载 slice 只拷 [0:tn] 真实列，K 维 size=tn 严格排除越界）；**仅当 tn < 32 时**才钳位 `tk = 32` 引入 stale 列带 [tn, 32)——此时 stale 列的 c_tile/state_t 值为 UB/L1 残留（可能含 NaN），会经 gemm K 维污染**有效行列**输出，必须中和：装载后 `vbrc(0)` 清零 stale 列带（一次向量 op）或 trace-time 改用 bn=32 单块（N<32 契约域）。intra 路径 K 维（s 维）同理：`tks = ts`（ts ≥ 32 时无 stale）；ts < 32 时对角块的 stale 列走 R4 band 域（vselect + 列 OOB 守卫）。**当前 manifest/test 全部 shape 整除**（Q∈{64,128,256}%64=0、P∈{64,128}%64=0、N∈{32,64,128}%32=0——最小 N=32 恰好 ≥ 分形 K 下限，K 维 stale band **不触发**），尾块路径为契约完备性设计，正确性由 L0 边界用例覆盖（§8.2 L0-5 含 N=48〔bn=64 → tn=48 ≥ 32 无 K-stale〕与 Q=96〔ts=32 贴限〕定向用例）。
- **语义保持论证**：slice 拷贝的越界语义 = 不拷贝（T.copy 区域语义规则 1：前向补 1 的 extents 借用），与源 guard 置 0 的差异在于 **tile 内 stale 区不被清零**——M/N 维 stale 行/列的贡献随写回裁剪丢弃（逐元素等价）；K 维在 tn ≥ 32 域由 size=tn 严格排除（与源 guard 置 0 逐元素等价）、在 tn < 32 域由 stale 列清零/掩码中和（清零后贡献 = 0 = 源 guard 语义）；**host 侧不做 padding**（ascend-constraints.md §4 禁止 host 改输入内容）。

**重设计项汇总（无其他「重新设计」条目）**：R1（一维 persistent 分核）、R2（gemm b_transpose 直装载）、R3（向量广播因子链 + 任务级预计算）、R4（常量掩码向量链）、R5（行缩放合并）、R6（slice 尾块 + size gemm）。§0.5 其余条目为保留（语义/算法层 4 项）/ 等价替换（7 项）/ 舍弃（3 项，均附理由与意图承接）。

### 0.7 标杆实现

- **源算子路径（迁移对象）**：`/home/tilelang/l00970450/TileOPs/tileops/kernels/mamba/ssd_chunk_scan.py`（工厂 `_ssd_chunk_scan_fwd_kernel` + custom_op wrapper + `SSDChunkScanFwdKernel` 类）。
- **NPU 仓已提取的 kernel 骨架（迁移目标文件）**：`examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernels.py`（pattern B 提取，本设计 §3.3 伪代码的落地位置——**Stage 3 将其重写为 target="npuir" 实现**）。
- **NPU 仓 wrapper/Kernel class（已移植，接口契约）**：`examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan.py`（K5/K7/K8/K9 适配完成：`npub::ssd_chunk_scan_fwd`、无 autotune、无 threads；default_config = {block_l:64, block_p:64, block_n:min(64,N), block_s:64, num_stages:3}）。
- **golden 参考实现（测试基准，源仓移植）**：`examples/TileOPs/tileops/testing/mamba2_reference.py`（`ssd_chunk_scan_fwd_ref` 函数）——纯 fp32 PyTorch materialize（einsum 双路径 + tril 掩码），**独立于本设计 NPU 算法**（无因子化、无 dtype 量化、无向量化结构），满足 golden 独立性要求（§8.1 直接采用）。
- **测试/性能入口**：`examples/TileOPs/tests/ops/test_mamba.py`（atol=1e-3 fp16 / 2e-3 bf16，rtol=1e-5）；`examples/TileOPs/benchmarks/ops/bench_mamba.py`。
---

## 1. 概述

### 1.1 算子名称

`_ssd_chunk_scan_fwd_kernel`（TileOPs 项目 `SSDChunkScanFwdOp` 的 kernel 实现；custom_op 名 `npub::ssd_chunk_scan_fwd`）

### 1.2 功能描述

Mamba-2 SSD 分块扫描输出前向：融合「历史状态贡献（exp(dA)·C@prev_states）」与「分块内因果衰减贡献（cb·exp(ΔdA)·dt 因果卷积 x）」两条 GEMM 路径，单 kernel 产出 `[B, S, H, P]` fp32 输出。迁移后为 NPU Developer 模式 MixCV 算子（Cube 双 GEMM + Vector 因子链，编译器自动 CV 切分）。

### 1.3 数学公式

$$
\text{out}[b, cQ{+}l, h, p] = \underbrace{e^{\text{dA}[b,h,c,l]} \cdot \sum_{n} C[b, cQ{+}l, g(h), n] \cdot S_{\text{prev}}[b,c,h,p,n]}_{Y_{\text{off}}\ \text{(history path)}} + \underbrace{\sum_{s \le l} \text{cb}[b,c,g(h),l,s] \cdot e^{\text{dA}_l - \text{dA}_s} \cdot \text{dt}[b,h,c,s] \cdot x[b, cQ{+}s, h, p]}_{Y_{\text{diag}}\ \text{(intra-chunk causal path)}}
$$

（完整语义与 dtype/边界语义见 §0.1；`g(h) = h // (H/G)`。）

### 1.4 算法描述（迁移决策后的 NPU 侧 Developer 模式算法）

本节描述 §0.5/§0.6 决策与 §1.6.1 机器验证定型后的 NPU 算法（**注**：§1.6.1 的机器验证否决了「K 侧因子移动（衰减因子乘 x 侧）」候选——fp16 噪声放大 38–135×，衰减因子**保持在 cb 侧**进入 cast，与源码数值路径一致；对角块采用直接差分形式而非 L-side 因子化——无条件数值安全）。与源算法的结构差异逐条标注来源：

**任务结构**：一维 persistent `T.Kernel(24)`（§0.6 R1），逻辑任务 = `(b, c, h)`（chunk-head 粒度；源为 (l×p tile, b·c, h) 三维 grid——**差异来源 R1**：l/p tile 合并进核内循环，任务级预计算摊薄）。核内每任务执行：

1. **任务头预计算**（Vector，[Q,1] 向量 op；**差异来源 R3**：源为逐 l-tile/逐 s-block 的标量 exp）：
   `dA_ub ← copy(dA_cumsum[b,h,c,:])`、`dt_ub ← copy(cast(dt[b,h,c,:]))`（[Q,1] fp32 UB 驻留，全任务复用——§0.4 #7 的意图承接）。
2. **p-loop**（bp=64；P=64 时 1 次）× **l-loop**（bl=64；Q=64 时 1 次）：
   a. **l-tile 因子**（Vector，per l-tile）：`anchor = dA_ub[l0]`；`exp_l = vexp(vsub(dA_ub[l0:l0+bl], anchor_col [bl,1]))`（[bl,1] **行因子**，≤ 1）；`exp_anchor = exp(anchor)`（标量，OPT-2）；`exp_s_all = vmul(vexp(vsub(anchor_row (1,Q), dA_ub_row (1,Q))), dt_ub_row (1,Q))`（**(1,Q) 行形态一次计算**，全程同 shape [1,N] 运算；s-block 切片 `[0:1, s0:s0+bs]` 为 **[1,bs] 列因子**复用——OPT-4；切片内 s < l0 恒安全，s ≥ l0 段为死区不读取，见 §0.6 R3）。
   b. **history 路径**（Cube）：n-loop `T.Pipelined(ceildiv(N, bn))`：`c_tile ← C[b, cs+l, g, n]`（slice copy，[bl,bn] dtype）、`state_t ← prev_states[b,c,h,p,n]`（**直装载 [bp,bn]，R2**，fp32→dtype 自动 VCast）→ `T.gemm(c_tile, state_t, acc, initC=(n_blk==0), b_transpose=True)`（fp32 累加；**acc 由首块 initC=True 覆写初始化**，无需预清零）。
   c. `acc = vmul(acc, exp_anchor)`（标量乘，OPT-2 合并式第一步）。
   d. **full-lower s-loop**（`T.serial(ceildiv(l0, bs))`，只含 `s0+bs ≤ l0` 的块——对角块移出循环单独处理）：每块 `cb_f32 ← copy(cb slice)`（GM→UB 直拷 + 自动 VCast）→ `lcb_fl = vcast(vmul(cb_f32, exp_s_all[0:1, s0:s0+bs] 列因子 [1,bs]——**探针 P-1**，fallback vbrc 列展开), →dtype)`（**衰减因子保持 cb 侧**，机器验证结论；exp_l 已提出）→ `T.gemm(lcb_fl, x_tile, acc, initC=False)`（x_tile 原始 dtype slice copy）。
   e. **行缩放**（OPT-2 第二步）：`acc = vmul(acc, exp_l)`（[bl,bp]⊙[bl,1] **行广播**，vmul 文档明示形态，GQA L645 Expert 先例）。
   f. **对角块**（s0 = `floor(l0/bs)·bs`，直接差分形式——无条件安全；**v1 主选：算术惩罚掩码**〔整除域〕，band-carrying 域〔R6 尾块〕保留 vselect——分域依据 PL-1.11 verified 与 §0.6 R4）：`diff` 链（dA 行/列 vbrc 展开 + vsub）→ `vadd(diff, pen_const)`（**核级常量惩罚矩阵**：i<j 位 −PEN（PEN=1e30 有限哨兵）、i≥j 位 0；s>l → exp 下溢精确 +0.0）→ `lcb_diag = vcast(vmul(vmul(vcast(cb→f32, rint), vexp(diff)), dt 列因子 [1,bs] 探针/fallback), →dtype)` → `T.gemm(lcb_diag, x_tile, acc, initC=False)`。band 域变体：`... vmul(·, dt 列因子) → vselect(mask_const, ·, zero_f32) → vcast`（NaN 免疫 + 列 OOB 守卫，仅 Q%bs≠0 trace 分配）。
   g. **写回**：`T.copy(acc, out[b, cs+l0 : cs+l0+tail_l, h, p0 : p0+tail_p])`（fp32 直写）。
3. **核级常量**（persistent 核初始化一次，**R4**）：整除域 `pen_const`（[bl,bs] fp32：`arange` 行列差 → 0/−PEN 矩阵，PEN=1e30，任务循环外一次）；band 域附加 `mask_const`（[bl,bs] bool，vcmp 生成）与 `zero_f32`。

**执行序要点**（OPT-2 合并式的顺序约束）：行缩放 (e) 必须在全部 full-lower 累加 (d) 之后、对角块累加 (f) 之前——对角块是 s 序最后一个块，程序序天然满足；对角块 gemm 不携带 exp_l（其因子为完整 exp(dA_l−dA_s)）。

**与源算法的结构差异清单**：任务粒度（R1）、state 直装载（R2）、因子向量预计算（R3+OPT-4）、常量掩码（R4+OPT-6）、行缩放合并（R5+OPT-2）、尾块 slice/size（R6）。**保持不变**（机器验证或安全分析裁定）：衰减因子乘 cb 侧后 cast（源数值路径）、对角块直接差分（源 fallback 路径）、上三角跳过（源 #10）、L-side anchor 数值稳定结构（源 #9）。

### 1.5 数据流图

```
GM[dA]─copy──▶ UB[dA_ub (Q,1) / dA_ub_row (1,Q) f32] ──vsub/vexp──▶ UB[exp_l (bl,1) 行因子]──────┐
GM[dt]─copy(cast)──▶ UB[dt_ub (Q,1) / dt_ub_row (1,Q) f32] ──vsub/vexp/vmul(同shape [1,Q])──▶ UB[exp_s_all (1,Q)]─┤ 行缩放[·,1]/列因子[1,·]（Vector 域）
                                                                                  │
GM[cb] ─copy(auto VCast)─▶ UB[cb_f32 (bl,bs) f32] ─vmul(exp_s [1,bs] 列因子·探针P-1)→vcast──▶ fragment[lcb (bl,bs) dtype]（full-lower）
GM[cb] ─copy(auto VCast)─▶ UB[cb_f32 (bl,bs) f32] ─diff链(vbrc×2/vsub/vadd(pen)/vexp/vmul×2)→vcast──▶ fragment[lcb]（diag 整除域；band 域改 vselect(mask)）
GM[C] ─slice copy─▶ L1[c_tile (bl,bn) dtype] ───────┐
GM[prev]─slice copy(cast,直装载)─▶ L1[state_t(bp,bn) dtype] ─┤（Cube 域）
GM[x] ─slice copy─▶ L1[x_tile (bs,bp) dtype] ────────┤
                       ▼                              ▼
            L0C[acc (bl,bp) f32] ◀──T.gemm(b_transpose=True)（n-loop 首块 initC=True 覆写初始化）
                       │ vmul(exp_anchor)（标量，Vector）
                       ▼
            L0C/fragment[acc (bl,bp) f32] ◀──T.gemm(lcb, x_tile) initC=False 累加
                       │ vmul(exp_l)（行广播缩放，Vector）→ diag gemm 累加
                       ▼
                    GM[out (B,S,H,P) f32]
```

（Developer 模式下 Cube/Vector 切分由编译器自动完成；fragment 自动映射 L0C/UB，向量侧 buffer 显式 `alloc_ub`、gemm 操作数 tile `alloc_shared` 自动映射 L1/UB——flash_attn_npuir_dev.py 同构先例。**因子方向约定（v1）**：行因子 `[M,1]` 沿 N 广播〔vmul 文档明示〕；列因子 `[1,N]` 沿 M 广播〔数学正确方向，Stage 3 探针 P-1 + vbrc fallback〕。）

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**——对同一数学语义的算法族回答调研四问（§1.6.0；迁移任务在源算子解读〔§0〕后、耦合性判定与重设计〔§0.5/§0.6〕前完成，调研结论已作为 M1 判定输入）；再在保证数学等价的前提下优化公式（§1.6.1，**含机器验证**）；再把保留下来的计算全部交给向量单元（§1.6.2）；最后回答"在哪个轴上向量化"（§1.6.3，阻塞级）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（§0.1 公式）的算法族。源算法（SSD 分块双路径）只是基线候选之一。调研深度：**完整调研**（矩阵 × 多步 × 融合类算子）。信息源：algorithm-candidates.md 参考表（ALG-attention/ALG-gemv/ALG-elementwise 命中行评估）、pattern-library（kb_search "ssd chunk scan mamba" 无命中，attention/cases 条目作同族参考）、本仓 examples/（flash_attention 两相位与 GQA expert 档案、mixcv）、源算子（§0.3/§0.4 为"该算子族存在什么算法"的直接证据）、外部已知算法（Mamba-2 SSD 论文族：Transformers are SSMs；mamba_ssm 官方 Triton `_chunk_scan_fwd`/`_chunk_cumsum` 知识——只取算法思路）、kb_search 预注入条目（CASE-attention-gqa-expert-full 已消费：作为同族 MixCV 迁移结构参考；VP-2026-0079 等流程提案与本算子算法无关，忽略）。**未使用互联网检索**（本地信息源已覆盖：源码 + 官方对齐注释 + 同族 examples + 模型知识一致收敛于分块双路径族；记录为"已评估无需联网补充"）。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：SSD 分块双路径**（源算法 = mamba_ssm 官方 `_chunk_scan_fwd` 对齐） | Y_off（rank-N off-diagonal 显式给态）+ Y_diag（causal 对角块） | — | — | ✅（基线） |
| 1 | 全序列 materialize（非分块） | `out = (L(cb ⊙ exp(ΔdA) ⊙ dt) · x) + off`，L 为 [S,S] 全 causal 矩阵 | 数学恒等（分块 = 对同一矩阵的分块求值） | 无收益：中间 [S,S] 矩阵 S=32k 时 4G 元素/chunk/head | ❌（访存爆炸，结构上不可行） |
| 2 | 逐 token 递推扫描（RNN 式在线） | `o_t = a_t·o_{t−1} + B_t·x_t`（a = exp(dA_t−dA_{t−1})） | 数学恒等（半可分矩阵的连乘形式） | O(S·H·P) FLOPs 低于分块 | ❌（见 R2：串行依赖不可并行；且本算子输入契约是 cb/prev_states 预处理量，递推形偏离输入契约） |
| 3 | semiseparable 分块分解的抽象重述 | off-diagonal 秩 ≤ N 分解 + 对角因果块 | 与基线同构（数学抽象层） | — | ❌（与 #0 重复） |
| 4 | 行缩放合并（history 与 full-lower 统一行因子） | `exp(dA_l) = exp_l·exp(anchor)`（指数律） | 数学恒等 | 每 l-tile 省 1 次行广播缩放 + 1 次 exp 序列；gemm 前少乘一个因子 | ✅（→ §1.6.1 OPT-2） |
| 5 | exp_s 统一预计算（per-l-tile [Q] 向量替代 per-s-block [bs]） | 同式 `exp(anchor−dA_s)·dt_s` 求值位置上移 + 切片复用 | 严格等价（同式同 dtype 路径） | 省 ~6 次 [bs] 级计算/(b,c,h)；op 发射数下降 | ✅（→ §1.6.1 OPT-4） |
| 6 | K 侧因子吸收（cb 侧 → x 侧） | `Σ_s cb·r[l]·c[s]·x = r[l]·Σ_s cb·(c[s]·x)` | 数学恒等；**舍入位置移动需容差评估** | cb_tile 免 fp32 中转直读 gemm；dt 全局一次吸收 | ✅（→ §1.6.1 评估，**机器验证否决**） |
| 7 | 掩码向量化（算术惩罚 / vselect 值替换） | `exp(指数+pen)`（pen 为 0/−PEN 常量矩阵）/ `s≤l ? v : 0` 的向量选择 | 双形均精确等价（惩罚形下溢精确 0；vselect 形选路替换） | 替代逐元素标量分支；惩罚形规避 vcmp 标量化（PL-1.11 verified 4.4–14.2x） | ✅（→ §1.6.1 OPT-6，惩罚主选 + band 域 vselect 分域） |
| 8 | exp2 域换算（exp → exp2·log2e） | `e^x = 2^{x·log2 e}` | 数学恒等；vexp2 vs vexp 吞吐差未实测 | 潜在发射/吞吐收益 | ❌（收益方向不成立：docs 数学操作目录确认 `T.vexp`（fp32 ✓）与 `T.vexp2` 并存，但无实测吞吐差证据；本算子 exp 调用量已被 OPT-4 削至 [Q]+[bl] 级，非瓶颈项——列为 Stage 4 可选微优化，不作设计期采纳） |
| 9 | 对角块 L-side 因子化（exp_l_col × exp_as_col 替代直接差分） | `exp(dA_l−dA_s) = exp(dA_l−anchor)·exp(anchor−dA_s)`（anchor = dA[l0]） | 数学恒等 | 对角块省 vbrc×2+vsub（diff 链） | ❌（**数值边界淘汰**：对角块内 s ≥ l0 ⟹ anchor−dA_s ≥ 0，列因子 exp_as ≥ 1，块内衰减 > ~88 时 fp32 溢出为 inf 并污染下三角（inf·exp_l ≠ 真值）；真实测试分布下为 24σ 罕见事件但**非无条件安全**。直接差分形式（主选）下三角指数恒 ≤ 0、上三角正指数被掩码丢弃——无条件安全且与源 fallback 路径一致。因子化形态连同其护栏（块内衰减 < 88）记入 §1.6.3 Stage 4 备选） |

**R2 在线算法**：**结论：无适用于本算子契约的在线变体**（结构依据如下，非无据断言）：
1. **本算子的输入已是"在线化的中间产物"**：Mamba-2 SSD 将序列按 chunk 切分后，跨块状态传递（`prev_states` 的链式更新 `S_c = A_c·S_{c−1} + B_c·x_c`）由上游算子 **SSDStatePassing/SSDChunkState** 完成（NPU manifest 注释明确这些算子未迁移）；`cb = C@B` 耦合与 `dA_cumsum` 亦为上游预处理。本算子是**块内双 GEMM 融合输出算子**，输入契约（6 张量）不含跨块 running 状态更新需求——在线性已在流水线上游完成，算子内无 running 统计量可提取（输出对每个 (l,p) 是**有限窗口因果和**，窗口 = 所在 chunk，非前缀依赖）。
2. 逐 token 递推（R1 #2）是数学等价的"流式"形式，但 S 长串行依赖链不可并行（每 token 依赖前一 token 的完整 [H,P] 状态），且需要 `A_t/B_t` 原始序列而非本算子的 `cb/prev_states` 预处理输入——**偏离输入契约**，属于算法族替换而非本算子的变体。
3. chunk 级 prefix-scan（log-depth 扫描传递状态）会改变上游算子分工（state passing 与 scan 融合），超出本算子迁移范围（迁移 prompt 明确接口不可变）。
> 收益口径核对：本算子的"扫描遍数"= 每 chunk 单遍（x 每 (b,c,h) 任务内 1× 驻留复用 + 跨 l-tile 切片 2.5× 平均、cb/C 单遍逻辑读、中间缓冲 O(tile)）——分块结构已给出在线级的遍数与缓冲量纲，无进一步单遍化空间。

**R3 复杂度对比**（四口径；以 workload w2 `mamba2-780m-b1-s4k` 标定：B=1, S=4096, H=48, P=64, C=16, Q=256, G=1, N=128，tokens = B·S·H = 196,608；口径注明含/不含）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| **基线：分块双路径**（causal 跳过后） | history `2·tokens·N·P` = 2·196,608·128·64 = **3.22G** + intra `2·(Q(Q+1)/2)·P·C·H` = 2·32,896·64·16·48 = **3.23G** ≈ **6.46G**（含全部 MAC=2；causal 面积取 Q(Q+1)/2 下三角；不含向量因子链 ~0.15G elem-op；与 manifest mamba.yaml roofline 公式代入 w2 = 6.44G 交叉一致，差值为 Q+1 vs Q 的整化） | 逻辑读：x 25.2M + cb 2.1M + C 1.05M + prev 25.2M + dA 0.79M + dt 0.39M + 写 out 50.3M ≈ **105MB**（HBM 口径；cb/C 的 G<H 复读依赖 L2 命中缓解，L2 命中后 HBM ~53MB） | x 任务内驻留 1× + 切片 2.5×；cb/C/prev/dA/dt 1×；out 1× | tile 级（[64,64]×~6 buffer，~100KB UB/L1） | 任务 (b,c,h) = 768，无跨核归约（GEMM 累加核内完成）；跨核零同步 |
| 候选 1：全 materialize | intra ≈ 19.3G（无 causal 跳过）+ 中间 [S,S,H] 写读 ≈ **5.4TB** 访存 | 爆炸（S=4k 时 [4096,4096,48] fp32 中间 ~3.2TB） | 多遍（写+读中间矩阵） | [S,S] 级（GM 都放不下） | 高并行但访存主导不可行 |
| 候选 2：逐 token 递推 | 2·S·H·P·B ≈ 0.1G（最低） | 输入重释（契约偏离）~ O(输入) ≈ 80MB | 单遍 | [H,P] running 状态（O(tile)） | **串行 S=4096 步依赖链**——并行度 1，时延不可行 |
| 基线 + §1.6.1 采纳优化（选定方案） | 同基线 6.46G（MAC 不变）；向量 op 数 ~140/任务（OPT-2/4 削减后） | 同基线（x 驻留消除 tile 级重复装载数；复读结构不变） | 同基线 | 同基线 | 同基线 |

**R4 硬件亲和性评估**（逐候选对照检查清单）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（分块双路径） | 双 GEMM → Cube（fp16/bf16 mma）；exp/mask/scale → Vector ✓ MixCV | [64,64] tile：L0C 16KB ≤ 128KB ✓；L1 双操作数 ~32KB ≤ 512KB ✓；UB 向量缓冲 ~100KB ≤ 192KB（×1.1–1.7 膨胀裕量评估见 §4.5） | bl/bp/bn/bs = 64：分形 M/N≥16、K≥32 ✓（bn=32 时 K 贴下限 ✓）；尾轴 64×2B = 128B ≥ 32B ✓ | 循环边界 Q/N/P 派生 PrimExpr（静态 shape，工厂特化）✓ | n-loop/s-loop `T.Pipelined` 双缓冲 ✓；CV 天然融合（flash_attn_npuir_dev.py 同构先例） | ✅（选定） |
| 全 materialize | GEMM 匹配 | 中间 [S,S] 超 GM 容量 | ✓ | ✓ | — | ❌（容量淘汰，结构不可行） |
| 逐 token 递推 | 逐元素 → Vector | ✓ | ✓ | ✗（循环携带依赖阻断流水） | 串行链不可并行 | ❌（并行度淘汰） |
| 优化后基线（本设计） | 同基线 + 向量链 op 数削减（OPT-2/4） | 同基线 | 同基线 | 同基线 | 同基线 | ✅（主选） |

**调研结论**：**选定算法族 = SSD 分块双路径（基线族，保留源算法结构）**，关键依据：R3 表中基线四口径全面占优（6.46G FLOPs / 105MB 逻辑访存 / tile 级缓冲 / 768 任务无跨核同步），候选 1 因 [S,S] 中间容量爆炸淘汰、候选 2 因 S 长串行依赖链淘汰（均为结构性依据，非容差权衡）；R4 确认基线结构与 NPU Cube+Vector 双引擎高度匹配（MixCV Developer 模式有 flash_attn_npuir_dev.py 同构先例）。与基线的结构差异：**族不变，块内实现重构**——采纳 R1 #4/#5/#7 三项等价变形（§1.6.1 完成四要素论证与机器验证），#6（K 侧因子）经机器验证**否决**、#9（对角因子化）经数值边界分析**否决**；源实现的标量因子计算/逐元素分支重构为 NPU 向量链。调研范围（"无更优算法族替代"的依据）：algorithm-candidates.md 全表（ALG-attention/ALG-gemv/ALG-elementwise/ALG-transpose 等命中行已评估）、kb_search "ssd chunk scan mamba"（无命中）、examples/flash_attention + TileOPs GQA 档案、源算子注释（官方对齐声明）、外部模型知识（mamba_ssm 官方实现族）。**源算法优化手段的意图承接**：§0.4 的 14 项在 §0.5 逐项标注（#9 anchor 因子化与 #10 上三角跳过直接保留进选定方案；#7/#8 由 R3+OPT-4 预计算承接；#11 由 R4+OPT-6 算术惩罚掩码承接；#1–#6/#12 由等价替换承接；#3/#13 舍弃附理由）。

**设计期 roofline（D-2，估算下界行）**（标定 w2；常数引自 pattern-library/constants.md 并标注版本戳；**stale 提示**：kb_stale_check 报 stale_count=83（当前工具链 tilelang 0.1.2+1990aa9fe4 与条目戳不符），下列常数按"设计期估算口径"使用，Stage 4 须重验）：
- 容量项：L0C 16KB/核（[64,64] fp32 acc）≤ 128KB（CONST-capacity-910B2C，toolchain 0.1.2+3a214cde + CANN 8.5.0）——非地板。
- 流量项：HBM 逻辑 105MB ÷ 混合流量地板 ~1.26TB/s（CONST-mte2-degradation 256M 档，toolchain 0.1.2+ed787bb）≈ **83µs**；L2 命中缓解后 HBM ~53MB ≈ 42µs，取 **~42–83µs**（估算区间；L2 命中率未实测——未实测假设：cb/C 复读按 50% L2 命中估）。
- 发射项：向量链 ~140 op/任务（OPT-2/4 削减后口径，§1.6.2 全清单）× 768 任务 ÷ 24 核 × ~0.5µs/op（CONST-vector-launch-overhead 串行发射口径，toolchain 0.1.2+3a214cde）≈ **~2.2ms 串行发射上界口径**；三引擎（Cube/Vector/MTE）流水重叠后的有效值预期低一个量级（GQA 类比：fa4096 同口径估算亦高实测 ~5×，实测 125–131µs）——**设计期判定向量链为第一疑似瓶颈**，消减手段（掩码链融合、bl=128 块扫描、vsub 广播形态验证）列入 §1.6.3 裁决计划与 Stage 4 清单。
- Cube 项：6.46G ÷ 24 核 ÷ ~13 TFLOPS/核（910B2C fp16 Cube 量级；未实测假设：按公开规格 ~313T/24 估）≈ **20.7µs/核**——非地板。
- **估算下界 = max(流量 42–83µs, 发射（重叠后）~数百µs, Cube 20.7µs)**：首版预期发射/向量链主导；Stage 4 目标 = 向量链压缩后逼近流量地板 ~83µs 量级。结论供 §5 tiling 与 Stage 4 调优参照。
#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（分块双路径）经 §0.6 重设计落地的 NPU 形态。逐项四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）；**机器验证（D-1）**：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/verify_equiv.py`（fp64 oracle + 目标 dtype 舍入路径对照，随机张量 + 角点：深衰减 dA→−256 / dt 上界 / 全零输入 / fp16 与 bf16 双 dtype），结果表内嵌于下。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 (OPT-2) | **history/full-lower 行缩放合并** | `acc += hist_gemm ⊙ exp(dA_l)`（history 行缩放）+ 每 full-lower 块 `lcb = cast(cb·exp_l·exp_s)`（exp_l 逐块乘入） | `acc = exp_l ⊙ ( exp(anchor)·hist_gemm + Σ_full-lower cast(cb·exp_s)@x ) + Σ_diag`（指数律 `exp(dA_l) = exp(dA_l−anchor)·exp(anchor)`；full-lower 的 exp_l 提出gemm，行缩放合并为循环后一次） | 数学恒等（指数律 + 分配律）；cast 对象差异 `cast(cb·exp_s)` vs `cast(cb·exp_l·exp_s)`：两者相对舍入均 ~2^-11，绝对误差前者 ~4.9e-4·\|cb·exp_s\| 经行缩放 exp_l 后 = 4.9e-4·\|cb·exp_l·exp_s\|，与后者**同量级**；fp64 严格恒等验证 rel < 3e-14（含 anchor=−87 下溢边界：exp(anchor)→0 时真值 exp(dA_l) ≤ exp(anchor) 亦下溢，源码直接 exp(dA_l) 同样为 0，行为一致） | 每 l-tile：省 1 次 [bl] exp 序列 + 1 次 [bl,bp] 行广播缩放 + 每 full-lower 块 lcb 链少 1 次 vmul（w2 每 (b,c,h) 平均 1.5 块 → 合计 ~4 op/l-tile，~12 op/任务）；机器验证 ratio=1.000 |
| 2 (OPT-4) | **exp_s_all 统一预计算** | 每 full-lower s-block 计算 `exp_s[bs] = exp(anchor−dA_s)·dt[s]`（[bs] 级，per-block） | per-l-tile 一次 `exp_s_all(1,Q) = exp(anchor−dA_row) ⊙ dt_ub_row`（同 shape [1,N] 运算），s-block 切片 `[0:1, s0:s0+bs]` 为 [1,bs] 列因子复用 | **严格等价**（同式同 dtype 路径，仅求值位置上移；torch.equal 级精确一致——verify 脚本列因子即 unsqueeze(0) [1,N] 正确方向）；切片域 s < l0（full-lower 条件 s0+bs ≤ l0）恒满足 anchor−dA_s ≤ 0 ⟹ 值 ≤ dt ≤ 0.11 **恒安全**；s ≥ l0 段（深衰减下可至 fp32 inf）为**死区**——full-lower 切片上界严格 s < l0 不读取、diag 不使用 exp_s_all，死区值不进任何链（v1 修正论证：v0「≤e^32 有限」仅对真实分布成立） | w2 每 (b,c,h)：6 个 full-lower 块的 [bs] 级计算合并为 1 次 [1,Q] 计算——省 ~5 次 [bs] vexp + ~5 次 [bs] vmul ≈ 10 op/任务；op 发射数（发射开销主导域）显著下降 |
| 3 (OPT-6) | **对角块因果掩码向量化** | 逐元素 `if_then_else(s ≤ l, cast(value), 0)` 标量分支（源 fallback 路径） | **主选（整除域）：算术惩罚掩码**——核级常量 `pen`（arange 行列差派生 0/−PEN 矩阵，PEN=1e30 有限哨兵）`vadd` 融进 exp 指数，s>l 位 exp 下溢为**精确 +0.0**、s≤l 位惩罚恒 0；**band-carrying 域（R6 尾块）：vselect 值替换**（常量 bool mask，NaN 免疫——加法掩码无法清除 stale 列 NaN）+ 列 OOB 守卫 | **精确等价（双形）**：惩罚形下三角逐位原值、上三角 fp32 下溢精确 +0.0；vselect 形选路无算术副作用。**主选依据（PL-1.11 verified，2026-09-16 a13585dc 重验维持）**：int16 vcmp 全形态标量化（占壁钟 45%、aiv_scalar 94–97%），算术惩罚掩码实测 **4.4–14.2x（几何 8.53x）**——v0 曾误读该条目为 stale 而倒置主备，v1 修正（引用条目须核对 front-matter status 字段） | 消对角块逐元素标量分支；惩罚矩阵从每任务一次降为每核一次（w2: 768 任务 → 24 核，÷32）；掩码 op 链从 vselect 形（1 op/任务 + vcmp 常量）换为 vadd（1 op/任务 + arange 常量）且规避 vcmp 标量化税 |
| 4（否决） | ~~K 侧因子吸收（衰减因子乘 x 侧 + dt 全局吸收）~~ | `lcb = cast(cb·exp_l·exp_s)`，`gemm(lcb, x)` | ~~`x_scaled = cast(exp_s·dt·x)`，`gemm(cb_dtype, x_scaled)`~~ | ~~数学恒等（乘法交换律）~~；**机器验证否决**：cast 位置移到 x 侧后，被舍入操作数（x·exp_s·dt）不随衰减缩小，绝对 fp16 噪声放大 38–135×（random 案 2.9e-3 > 测试 atol 1e-3；deep_decay 案 2.3e-4 vs 基线 3.6e-6）——**违反容差内等价，禁止采纳**（见下表否决行） | （不采纳）源码将衰减因子乘在 cb 侧再 cast 是**有数值必要性的结构**：衰减因子使被 cast 量缩小 ⟹ cast 绝对噪声随之缩小 |
| 5（否决） | ~~对角块 L-side 因子化（exp_l_col × exp_as_col）~~ | 对角块逐元素 `exp(dA_l−dA_s)` | ~~`exp_l[i]·exp_as[j]`（anchor = dA[l0] 的行列因子积）~~ | 数学恒等；**数值边界否决**：对角块内 s ≥ l0 ⟹ anchor−dA_s ≥ 0 ⟹ 列因子 exp_as ≥ 1，块内衰减 > ~88（fp32 exp 上限）时溢出 inf，且 `inf·exp_l` 污染下三角（真值 ≤ 1 的位置得 inf）。真实测试分布（dA = −cumsum(rand)，bs=64 步衰减均值 32、σ~2.3）下为 24σ 罕见事件，但**非无条件安全**；直接差分形式（主选）下三角指数恒 ≤ 0、上三角正指数被惩罚掩码（−PEN 吸收）或 vselect（选路丢弃）中和——无条件安全 | （不采纳；连同「块内衰减 < 88」护栏记入 §1.6.3 Stage 4 备选——若实测对角块为热点且分布受控可重评） |

**等价性机器验证结果表**（`verify_equiv.py`，2026-09-17 执行，torch CPU，fp64 oracle；Q=128 双 l-tile 覆盖 full-lower + diag 两路径；BL=BS=64；各 case 含 H=2 重复；判定口径：噪声比 ratio = max\|cand−oracle\| / max\|base−oracle\| ≤ 3 且绝对误差 ≤ 1e-2）：

| 验证项 | dtype | case | max\|cand−oracle\| | max\|base−oracle\| | ratio | 结论 |
|--------|-------|------|-------------------|-------------------|-------|------|
| OPT-2 fp64 严格恒等（4 个 anchor 含 −87 下溢边界） | fp64 | 指数律 | rel 2.86e-14 | — | — | **EQUIV_PASS** |
| OPT-4 切片恒等（全部 l-tile × s-block 组合） | fp32 | torch.equal | 精确一致 | — | 1.000 | **EQUIV_PASS** |
| OPT-6 vselect 形掩码精确性（s≤l 原值保持 / s>l 精确 0） | fp32 | 掩码矩阵 | 精确（含远离 0 的测试值） | — | — | **EQUIV_PASS** |
| OPT-6 算术惩罚形精确性（下三角精确原值 / 上三角精确 +0） | fp32 | 掩码矩阵 | 精确（fp32 下溢为 +0.0） | — | — | **EQUIV_PASS** |
| FULL（OPT-2+4+6 组合，§1.4 定型算法） | fp16 | random | 2.157e-05 | 2.157e-05 | 1.000 | **EQUIV_PASS** |
| FULL | fp16 | deep_decay（dA→−256） | 3.590e-06 | 3.590e-06 | 1.000 | **EQUIV_PASS** |
| FULL | fp16 | dt_upper_bound（dt=0.11 恒定） | 2.433e-05 | 2.433e-05 | 1.000 | **EQUIV_PASS** |
| FULL | fp16 | zero_input | 3.625e-05 | 3.625e-05 | 1.000 | **EQUIV_PASS** |
| FULL | bf16 | random | 1.939e-04 | 1.939e-04 | 1.000 | **EQUIV_PASS** |
| FULL | bf16 | deep_decay | 1.413e-04 | 1.413e-04 | 1.000 | **EQUIV_PASS** |
| FULL | bf16 | dt_upper_bound | 1.734e-04 | 1.734e-04 | 1.000 | **EQUIV_PASS** |
| FULL | bf16 | zero_input | 1.299e-04 | 1.299e-04 | 1.000 | **EQUIV_PASS** |
| ~~OPT-3（K 侧因子，否决项回归记录）~~ | fp16 | random | 2.923e-03 | 2.157e-05 | 135.5 | **EQUIV_FAIL（已否决，不采纳）** |
| ~~OPT-3~~ | fp16 | dt_upper_bound | 1.527e-03 | 2.433e-05 | 62.8 | **EQUIV_FAIL（已否决）** |

**优化结论**：采纳 **3 项**（OPT-2 行缩放合并、OPT-4 exp_s_all 预计算、OPT-6 掩码向量化），全部机器验证 EQUIV_PASS（组合噪声比 1.000，fp16/bf16 双 dtype × 4 角点）；否决 2 项（K 侧因子移动——噪声放大超容差；对角 L-side 因子化——数值边界非无条件安全）。**优化后公式（§3.1 拆解的唯一输入）**：

$$
\text{out}[l_0{+}i, p] = \underbrace{e^{\text{dA}_i - a}}_{\text{exp\_l}[i]} \cdot \Big( e^{a} \cdot \underbrace{\textstyle\sum_n C_i[n] \cdot S_p[n]}_{\text{hist\_gemm}} + \underbrace{\textstyle\sum_{s_0:\,s_0+b_s \le l_0} \textstyle\sum_j \text{cast}\big(\text{cb}[i,j] \cdot \text{exp\_s\_all}[s_0{+}j]\big) \cdot x[s_0{+}j, p]}_{\text{full-lower gemm chain}} \Big) + \underbrace{\textstyle\sum_{j \le i} \text{cast}\big(\text{mask} \odot \text{cb}[i,j] \cdot e^{\text{dA}_i - \text{dA}_j} \cdot \text{dt}[j]\big) \cdot x[s_0^{\text{diag}}{+}j, p]}_{\text{diag gemm}}
$$

其中 `a = dA[l0]`（anchor）、`exp_s_all[s] = e^{a − dA[s]}·dt[s]`（per l-tile 一次，(1,Q) 行形态）、`pen(i,j) = (i < j ? −PEN : 0)`（核级常量，PEN=1e30——整除域主选；band 域等价换为 `mask = (j ≤ i)` + vselect 值替换）。无优化空间残余说明：gemm 的 MAC 计算量已由算法族选型（causal 跳过）与源一致；向量链 op 数已通过 OPT-2/4 压缩，进一步压缩（vsub 广播、掩码融合、块尺寸）依赖未实证常数，归入 §1.6.3 实验裁决而非纸面采纳。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点 §1.4 实现方案中全部循环与标量计算点（§0.3 步骤表 S1–S11 的 NPU 形态）；替代方案 API 均有 `docs/` 与 `examples/` 目录佐证。

| # | 计算点 | 原实现形态（源/GPU 式） | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|------------------------|-------------|---------|------------------------------|
| 1 | dA/dt 任务头缓存（S4） | `T.Parallel(Q)` 逐元素写 SMEM | `T.copy`（GM→UB [Q,1]，跨 dtype 自动 VCast，docs T.copy.md §2.3 条 4） | ✅ | — |
| 2 | exp_dA_l / exp_l 计算（S5/S7） | `T.Parallel(bl)` 逐元素 `T.exp` | `T.vexp`（fp32 ✓，docs T.vexp.md）on [bl,1] UB buffer | ✅ | — |
| 3 | exp_s 计算（S9） | per-s-block [bs] 逐元素 | OPT-4：per-l-tile `T.vsub/vexp/vmul` 同 shape [1,Q] 运算，切片 [1,bs] 列因子复用 | ✅ | — |
| 4 | history 行缩放（S6） | `T.Parallel(bl,bp)` 行广播乘 + guard | `T.vmul(acc, exp_anchor 标量)`（tensor·scalar，GQA L519/L654 先例）+ OPT-2 合并 `T.vmul(acc, exp_l [bl,1])` 行广播（vmul 文档明示 `[M,N]*[M,1]`；GQA L645 Expert 先例） | ✅ | — |
| 5 | c_tile/state_t/x_tile/cb_f32 装载（S1/S2/S8） | `T.Parallel` 逐元素 + `if_then_else` guard + safe 索引 | `T.copy` slice 形态（尾块 `T.min` 裁剪；docs T.copy.md §2.2.2 尾块规则；flash_attn_npuir_dev.py L30/58/80 先例；cb 为 GM→UB 跨 dtype 直拷 cb_f32） | ✅ | — |
| 6 | lcb_cast 因子乘积（S9） | `T.Parallel(bl,bs)` 逐元素三乘 + cast | `T.vcast(cb→f32, rint)` + `T.vmul(·, exp_s 列因子 [1,bs])`（**探针 P-1**；fallback vbrc 列展开 + 同 shape vmul）+ `T.vcast(→dtype, rint)` | ✅ | — |
| 7 | 对角块 diff 与掩码（S9'） | 逐元素 `exp(dA_l−dA_s)` + `if_then_else(valid)` 标量分支 | 整除域：`T.arange`（fp32 索引差，docs T.arange.md）核级一次惩罚常量 + 任务内 `vbrc`×2/vsub/**vadd(pen)**/vexp/vmul×2/vcast 向量链（§1.4 步骤 2f，PL-1.11 verified 主选）；band 域：+ `T.vcmp`/`T.vselect`（docs T.vcmp.md §2.3 条 4 片上使用，NaN 免疫） | ✅ | — |
| 8 | gemm 累加（S3/S10） | Tensor Core mma | `T.gemm`（Cube，Developer Op，docs T.gemm.md） | ✅ | — |
| 9 | out 写回（S11） | `T.Parallel` + guard 写 | `T.copy(acc, out[...])` slice 形态（fp32 直写无 cast） | ✅ | — |
| 10 | 任务解码 `cid → (b,c,h)`、`l0/p0/s0` 块索引 | 标量整除/取余 | 无（每任务 O(1) 次整数标量，非逐元素热点） | ❌ | **block 级索引/任务映射计算**：每任务 ~10 次整数运算，执行一次，无逐元素等价 API（T.ceildiv/div 域为标量 PrimExpr）；GQA 同款（L285–291） |
| 11 | persistent 任务循环 `for task_id in T.serial(num_local_tasks)` | 块级串行 | 无 | ❌ | **tile 级顺序依赖/任务映射结构**：任务间无数据依赖但核数适配要求串行分派（core-split-strategy.md 要素③ 极大规模方案）；`T.serial` 是分核标准结构，非逐元素计算 |
| 12 | 尾块尺寸 `T.min(bl, Q−l0)` 等边界标量 | 标量 min | 无 | ❌ | **依赖动态边界的块级元数据**：每 tile O(1) 次，进 slice extents/gemm size（T.copy 尾块借用规则的输入）；非逐元素热点 |
| 13 | host 侧 shape 校验/工厂参数 | Python 标量 | 无 | ❌ | **host 元数据计算**（不在 kernel 内；ascend-constraints.md §4 允许的视图/元数据操作） |

**向量化结论**：逐元素计算点（#1–#9）**全部向量化**（替代 API 均有 docs/Tilelang.language/ 条目与 examples 先例佐证：T.copy/T.vexp/T.vmul/T.vcast/T.vcmp/T.vselect/T.arange/T.gemm）；**保留 4 类**标量/循环（表 #10–#13，v1 统一计数），理由类别：block 索引/任务映射（#10）、分核结构串行（#11）、动态边界元数据（#12，含 gemm 尾块 size 标量）、host 元数据（#13）——均非逐元素热点，与 §6 循环结构一致（§6 无任何逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（x/cb/C/dA/prev/dt/out 的 GM 布局固定，见 §0.2），但**核内布局与 lane 映射是自由变量**。本算子为 Cube/Vector 混合类（§1.6.3 类别清单的 Cube/MixCV 类 + 规约类子问题），候选矩阵按类别枚举。GPU 源码的并行轴选择（thread→(ll,nn) 映射、三维 grid）是**输入而非结论**（GPU 轴与 NPU Vector 轴不对应）。

**轴质量评分维度**（逐候选）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | 全 workload P=64/128（bp=64 → P/bp ∈ {1,2} ✓ 尾轴 128B 对齐 ✓）；Q ∈ {64,128,256}（bl=64 整除 ✓）；N ∈ {32,64,128}（bn=64 时 N=32 → 1 块 K=32 贴分形下限 ✓）；bs=64 与 Q 整除 ✓。尾 lane 浪费率 = 0（全整除）；非整除契约 shape 由 R6 尾块路径覆盖 |
| 尾 lane 浪费率 | 0%（manifest/test 全部 shape 整除；通用契约下尾块 slice 裁剪） |
| 累加链形态 | K 维（bn/bs）进 Cube 分形（无跨步累加问题）；行缩放 [bl,bp]⊙[bl,1] 行广播（文档明示）；列因子 [bl,bs]⊙[1,bs]（探针 P-1，fallback vbrc 列展开）✓ |
| repack 代价 | GM 布局契约保留（无 host permute）；核内 cb/x/C/prev 直读（无转置链）——state 的"转置"由 gemm `b_transpose=True` 硬件处理（0 repack）；对角 diff 的行/列广播用 vbrc 展开（[bl,1]→[bl,bs] 与 [1,bs]→[bl,bs]）+ vmul 广播混合（见候选矩阵） |
| UB 容量影响 | 见 §4.5 预算（~100KB 级，含 1.7× 膨胀裕量评估） |

**候选矩阵**（Cube/MixCV 类：fractal/NZ 路径 × epilogue 归属；规约子问题：水平 vs 垂直）：

| # | 布局方案 | 向量化轴 / 关键结构 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|--------------------|------------|--------------|---------|
| 1 | **主选：契约布局直读 + gemm b_transpose + 方向化因子链**（§1.4，v1 修订） | Vector 轴 = 尾轴（P 轴 for [·,bp]、bs 轴 for [bl,bs]）；**行因子 [M,1]（vmul 文档明示）/ 列因子 [1,N]（探针 P-1，fallback vbrc 列展开）**；diag 掩码 = **算术惩罚主选**（PL-1.11 verified）；state 经 `b_transpose=True` 免转置；diag diff 用 vbrc 展开 | 无 repack（state 转置进 Cube 分形；vbrc 仅 [bl,bs] 对角链 2 次） | 尾轴连续 128B 对齐；op 数 §1.6.2 清单（~140/任务）；L1 双操作数 tile ~32KB | ✅（对齐 flash_attn_npuir_dev.py 先例结构；惩罚掩码规避 vcmp 标量化——PL-1.11 verified 4.4–14.2x） |
| 2 | diag diff 广播 vsub（若 `T.vsub` 支持 [bl,bs]−[1,bs] 广播） | 同 #1 但 diff = `vsub(vbrc(dA_l [bl,1]), dA_s_row [1,bs])`——列向量 [bl,1] 与矩阵 vsub（一次 vbrc 省 1 op） | 无 | 省 1 op/diag（4 diag/任务 → 4 op/任务）；**广播语义无文档佐证**（docs 数学操作目录 T.vsub.md 仅双 tensor 同 shape 语义描述） | ❌ 设计期（**列入 Stage 3 API 探针 P-2**：编译期验证 vsub 广播形态，可用则采纳——收益小但零风险） |
| 3 | vcmp/vselect 常量掩码链（v0 主选，v1 降级为 band 域专用） | mask bool 常量 + vselect 值替换 | int16 idx 常量核级预计算 | op 数：vselect（1）+ 核级 vcmp；**但 vcmp int16 全形态标量化**（PL-1.11 **verified**，非 stale——2026-09-16 a13585dc 重验维持：aiv_scalar 94–97%、对角块聚合 ≈ 壁钟 45%、惩罚掩码 4.4–14.2x） | ❌ 整除域主选、✅ **band-carrying 域必选**（R6 尾块 stale 列可能 NaN——加法掩码无法清除 NaN，vselect 值替换是唯一 NaN 免疫形态，PL-1.11 NaN 免疫边界） |
| 4 | bl=128 大 l-tile（Q=256 → 2 l-tile） | 同 #1，bl=128：diag 面积 [128,128]、s-block 数 10→4、L0C 128×64×4B=32KB ✓、L1 tile 翻倍 ~64KB ✓ UB ~150KB（×1.7 膨胀后接近 192KB 上限） | 无 | op 数 ~140→~80/任务（s-block 数与循环开销 ÷2）；UB 预算贴限（风险）；Q=64 workload 退化为 bl=64（shape 分派） | ❌ 主选、✅ **备选（实验裁决）**：UB 膨胀系数（1.10–1.12 vs 1.7 两版实测，CONST-capacity-910B2C 内两口径）未定 → bl=64 主选安全余量大 |
| 5 | host permute 重排输入（如 cb 预转置 / x 折叠 H） | 各种"便利布局" | 2–6 次 host permute | host permute 实测 ~百 µs 级（PL-1.3 stale 线索 + host 禁改输入约束 ascend-constraints.md §4）——净亏且违反约束 | ❌（违反 host 约束；证据：ascend-constraints.md §4 禁止 host 改输入内容，permute 产生副本引入 GM 往返） |
| 6 | H 维 fold 任务（同 group 的 head 共享 cb/C 的核内复用） | 任务 = (b,c,g,h_tile)，cb_tile/C_tile 在 L1 复用 heads_per_group 次 | 无 | cb/C 复读流量 ÷ heads_per_group（w2: cb 逻辑读 2.1MB×48→2.1MB）；但任务数 ÷48（w2: 768→16 < 24 核欠载）需再切 h_tile——任务结构复杂化 + L1 生命周期管理 | ❌ 首版（**Stage 4 候选**：L2 命中已缓解大部分复读——设计期估算 cb/C 复读占 HBM 流量主导项已按 L2 50% 命中折算；fold 的边际收益需 L2 命中率实测后重评） |
| 7 | 全 fp32 gemm（HF32 路径，操作数免 cast） | state/prev 已 fp32；cb/x cast 消除 | 无 | 省向量 cast op（~20/任务）；Cube 吞吐 HF32 ≈ fp16 一半（docs T.gemm.md §2.3 条 4"fp32×fp32 走 HF32 实测可用"，吞吐比未实测——未文档化假设：按 Ascend HF32 惯例 ~½ 估） | ❌（Cube 吞吐减半代价 > cast 节省；且偏离源数值路径〔dtype 操作数〕，精度基线漂移） |

**规约子问题（水平 vs 垂直）**：本算子唯一规约是 gemm 的 K 维（n 维 history / s 维 intra）——**全部进 Cube 分形**（K 维累加由 mmadL1 完成，docs T.gemm.md §3），无 Vector 侧水平/垂直归约决策点（`T.reduce` 不出现在主路径）。核内无独立规约轴选择问题，豁免该子项。

**实验裁决计划（v1 修订：掩码主选已由 PL-1.11 verified 实测定局，裁决项收敛为 bl 与形态微调）**：
- **主选方案**：§1.4 全部（bl=bs=bp=64、**算术惩罚掩码**〔整除域〕+ vselect〔band 域〕双 trace、契约布局直读、行因子 [M,1]/列因子 [1,N] 探针 P-1）——进入 Stage 3。
- **Stage 3 API 探针（P-1/P-2，编译期即可裁决）**：P-1 = `T.vmul` 列因子 `[M,N]⊙[1,N]` 广播形态（无文档/生产先例——T.vmul.md §2.2.2 仅列 [M,N]*[M,1] 行广播；失败 fallback = `vbrc([1,bs] → [bl,bs])` + 同 shape vmul，+1 op/处、UB +16KB/处，§4.5 有余量）；P-2 = `T.vsub` 广播 `[bl,bs]−[1,bs]`（省 1 op/diag，失败 fallback = vbrc×2 + vsub）。
- **备选方案 B（bl=128 变体）**：bl=128/bs=128/bp=64；L0C 32KB、L1 tile ~64KB、UB ~150KB（按 1.12 膨胀）/或压缩 bs=64 降 UB；s-block 数减半；Q=64/128 workload 由工厂分派回退 bl=64（lru_cache shape 特化天然支持，PL-1.4 工厂层回退分发先例）。
- **测量指标**：代表 shape = w2（mamba2-780m-b1-s4k）+ w4（b2-s32k 长序列）；latency（msprof）+ 惩罚掩码 vs vselect 的 A/B 对照（PL-1.11 结论在 Developer 模式的复证——GQA 实测为 Expert 形态，Developer 下 cv_split 行为待实测确认，若 Developer 下 vcmp 不标量化则两形等价回退）+ UB 实际占用（BishengIR 报错文本反推，CONST-capacity 条目方法）。
- **判定阈值**：备选 latency < 主选 × (1−10%) 才翻转；精度回归（atol 门禁）任一失败即弃。
- **回写路径**：实测数据触发设计修订 → 回写 §1.6.3 判定依据（附录补记）；shape 分派可行性（B 变体按 Q 分派）在裁决时评估。

**布局决策结论（v1）**：选定方案 #1：核内布局 = **契约布局直读**（x/cb/C/dt/dA/out 不重排）、向量化轴 = **尾轴（P 轴 / bs 轴）**、因子形态 = **行因子 [M,1]（vmul 文档明示）/ 列因子 [1,N]（探针 P-1 + vbrc fallback）**、对角掩码 = **算术惩罚（整除域主选，PL-1.11 verified）+ vselect（band 域）**、state 转置 = **gemm b_transpose 硬件路径**、repack = **无**（对角 diff 链 2 次 vbrc 是唯一展开开销）。该结论与 §3.3 伪代码、§4 内存规划（buffer 形状）、§6 循环结构三方一致（累加循环内层向量维 = 尾轴；buffer 形状 = [M,1]/[1,N] 方向化因子 + [bl,bs]/[bs,bp] tile）。弃选方案的量化理由见候选矩阵（#2 vsub 广播无文档佐证〔探针 P-2〕、#3 vcmp 标量化 verified 实证〔整除域弃、band 域必选〕、#4 UB 贴限实验裁决、#5 违反 host 约束、#6 Stage 4 候选、#7 吞吐减半）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（用户指定；`@tilelang.jit(target="npuir")` + `T.alloc_shared/T.alloc_fragment` + v-prefix API + `T.gemm`，不写 `T.Scope`/手动 flag 同步）

### 2.2 选型理由

1. **用户显式指定 developer**（迁移 prompt："用 developer 模式实现"）。
2. 算子特征匹配（decision-tree.md §2）：matmul（双 GEMM 路径）+ element-wise 前处理（因子链）/后处理（行缩放）→ **CV 融合算子 → Developer 模式（推荐）**，写法参考 `examples/flash_attention/flash_attn_npuir_dev.py`（online flash attention：`T.Kernel + alloc_shared + T.gemm + T.vmul/vexp/vcast/vmax/vsub + T.Pipelined` 与本算子结构同构）。
3. 无需跨核 workspace 协同（任务间零同步，§0.6 R1），Developer 自动同步足够；Expert 的 `T.Scope`/`sync_block_set/wait` 手段（GQA 形态）在本算子无必要收益——两 GEMM 路径的 CV 交替由编译器 `cv_split` 自动编排（AGENTS.md Developer MixCV 触发规则：T.gemm + v-prefix 同 kernel → mixcv 结构，Developer 模式可表达）。
4. 尾块/分形钳位等 Expert 手段在 Developer 模式有等价表达（size-form `T.gemm`、slice `T.copy` 尾块），flash_attn_npuir_dev.py 与 docs T.gemm.md 示例均为 Developer 形态。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配（v1 修订：向量侧显式 UB） | gemm 操作数 tile 用 `T.alloc_shared`（编译器映射 L1，flash_attn_npuir_dev.py L29–33 先例）；**向量 op 操作数 buffer 显式 `T.alloc_ub`**（T.vmul.md §2.3 条 1「操作数 buffer 必须分配在 UB 上」；T.vmul/T.vexp 文档示例即 Developer 简单 jit 形态用 alloc_ub）+ `T.alloc_fragment`（gemm dst/向量链中转，flash_attn_dev L34 `scores` fragment 作 vmul 操作数先例）；不使用 `T.alloc_L1/L0C`（Expert 手段） |
| 计算方式 | Cube：`T.gemm`（含 `b_transpose`/`initC`/`size`）；Vector：v-prefix 链（`T.vmul/vexp/vcast/vsub/vbrc/vcmp/vselect`）+ `T.arange`；块内并行由编译器向量化（无 `T.Parallel` 逐元素循环） |
| 同步 | 自动同步（编译器插入依赖；无 `T.sync_threads`/手动 flag）——§7 |
| Kernel 形态 | 一维 `T.Kernel(24, is_npu=True)` persistent + `T.serial` 任务循环（§0.6 R1） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（OPT-2/4/6 定型形态）；每个步骤优先映射 v-prefix 向量 API，与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `dA_ub = dA[b,h,c,:]`；`dt_ub = fp32(dt[b,h,c,:])` | 任务头 [Q,1] 因子源（GM→UB） |
| 2 | `anchor = dA_ub[l0]`；`exp_l = exp(dA_ub[l0:l0+bl] − anchor)`；`exp_anchor = exp(anchor)` | l-tile 行因子（≤ 1）+ history 标量因子 |
| 3 | `exp_s_all = exp(anchor − dA_ub) ⊙ dt_ub`（[Q,1]） | full-lower 列因子统一预计算（OPT-4） |
| 4 | `hist = Σ_n c_tile[i,n]·state_t[p,n]`（gemm，b_transpose） | history 路径 fp32 累加 |
| 5 | `acc = hist · exp_anchor` | OPT-2 第一步（标量乘） |
| 6 | `lcb_fl = cast(fp32(cb[i,j]) · exp_s_all[s0+j])`；`acc += lcb_fl @ x[s0+j, p]` | full-lower gemm 链（exp_l 已提出） |
| 7 | `acc = acc ⊙ exp_l`（行广播） | OPT-2 第二步（统一行缩放） |
| 8 | `lcb_diag = cast( mask ⊙ fp32(cb[i,j]) · exp(dA[i]−dA[j]) · dt[j] )`；`acc += lcb_diag @ x[j, p]` | 对角块（直接差分 + 常量掩码，OPT-6） |
| 9 | `out[b, l0+i, h, p] = acc[i,p]` | fp32 写回 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | dA/dt 载入 | `T.copy(dA_cumsum[bz, bh, bc_idx, 0:Q], dA_ub[0:Q, 0:1])`；`T.copy(dt[bz, bh, bc_idx, 0:Q], dt_ub[0:Q, 0:1])`（跨 dtype 自动 VCast） | src 4D 混合 slice；dst [Q,1] fp32 | Developer |
| 2 | exp_l | `T.vsub(dA_ub[l0:l0+bl, 0:1], anchor_v, tmp[0:bl, 0:1])` → `T.vexp(tmp, exp_l)` | anchor_v 为 let-bound 标量广播（vbrc 到 [bl,1] 后 vsub，或 host 侧常量折入——trace-time Python 标量直接相减更简：`T.copy(dA_ub[l0:l0+bl], exp_l)` 后减法仍需 op；采用 vbrc(anchor, anchor_row[bl,1]) + T.vsub） | Developer |
| 3 | exp_s_all（(1,Q) 行形态，v1 修订） | `T.vsub(anchor_row (1,Q), dA_ub_row (1,Q), tmp (1,Q))` → `T.vexp(tmp, tmp)` → `T.vmul(tmp, dt_ub_row (1,Q), exp_s_all (1,Q))`（全程同 shape [1,N] 运算，文档支持） | vbrc(anchor, anchor_row [1,Q]) 先行；切片 `[0:1, s0:s0+bs]` 即 [1,bs] 列因子 | Developer |
| 4 | history gemm | `T.copy(C_mat[bz, cs+l0 : cs+l0+tl, bg, n0 : n0+tn], c_tile[0:tl, 0:tn])`；`T.copy(prev_states[bz, bc_idx, bh, p0 : p0+tp, n0 : n0+tn], state_t[0:tp, 0:tn])`；`T.gemm(c_tile, state_t, acc, initC=(n_blk==0), b_transpose=True, size=[tmc, tk, tnp])` | slice 尾块 + size-form；state 直装载 [bp,bn]；acc 首块覆写初始化 | Developer |
| 5 | 标量乘 | `T.vmul(acc, exp_anchor, acc)`（tensor·scalar，GQA L519/L654 生产先例） | exp_anchor 为 let-bound 标量（T.Kernel 内、T.Scope 外定义，T.vmul.md §2.3 条 2） | Developer |
| 6 | full-lower 链（v1 修订：cb 直拷 UB + 列因子探针） | `T.copy(cb[bz, bc_idx, bg, l0 : l0+tl, s0 : s0+ts], cb_f32[0:tl, 0:ts])`（GM→UB 跨 dtype 自动 VCast，docs T.copy.md §2.3 条 4）→ `T.vmul(cb_f32, exp_s_all[0:1, s0:s0+ts]（[1,bs] 列因子）, lcb_f32)`（**[M,N]⊙[1,N] 探针 P-1**；fallback：`T.vbrc(exp_s_col [1,bs] → exp_s_mat [bl,bs])` + 同 shape `T.vmul`）→ `T.vcast(lcb_f32, lcb, round_mode="rint")` → `T.copy(x[bz, cs+s0 : cs+s0+ts, bh, p0 : p0+tp], x_tile[0:ts, 0:tp])` → `T.gemm(lcb, x_tile, acc, initC=False, size=[tmc, tks, tnp])` | 列因子方向 [1,N]（沿 M 广播）；行因子见步 7 | Developer |
| 7 | 行缩放 | `T.vmul(acc, exp_l, acc)`（[bl,bp]⊙[bl,1] **行广播**，vmul 文档明示形态；GQA L645 Expert 先例） | — | Developer |
| 8 | 对角链（v1 修订：算术惩罚主选） | `T.vbrc(dA_ub[l0:l0+bl, 0:1] → dA_l_mat[bl,bs])`；`T.vbrc(dA_ub_row[0:1, l0:l0+bl] → dA_s_mat[bl,bs])` → `T.vsub(dA_l_mat, dA_s_mat, dA_l_mat)`（in-place diff）→ `T.vadd(dA_l_mat, pen_const, dA_l_mat)`（**惩罚融进指数**，pen_const 为核级常量 0/−PEN 矩阵）→ `T.vexp(dA_l_mat, dA_l_mat)` → `T.vcast(cb slice, cb_f32, round_mode="rint")`（同步 6 直拷）→ `T.vmul(cb_f32, dA_l_mat, lcb_f32)` → `T.vmul(lcb_f32, dt_ub_row[0:1, s0:s0+bs]（[1,bs] 列因子·探针 P-1）, lcb_f32)` → `T.vcast(lcb_f32, lcb, round_mode="rint")` → `T.gemm(lcb, x_tile, acc, initC=False, size=[...])`。**band 域变体**（Q%bs≠0）：dt 乘后加 `T.vselect(mask_const, lcb_f32, zero_f32, lcb_f32)`（NaN 免疫）+ 列 OOB 守卫；常量：整除域 `pen_const`（`T.arange` 行列差 fp32 → vmin/clamp 派生 0/−PEN）核级一次，band 域另备 `T.arange` int16 + `T.vcmp` mask | 惩罚在 vexp 指数域（exp(−PEN) 精确 +0.0）；vcast 上行 f16→f32 补 `round_mode="rint"`（vcast.md dtype 矩阵：f16→f32 仅 rint） | Developer |
| 9 | 写回 | `T.copy(acc[0:tl, 0:tp], out[bz, cs+l0 : cs+l0+tl, bh, p0 : p0+tp])` | fp32 直写 | Developer |

### 3.3 计算伪代码

> dtype 位置参数化（迁移模板规则：`T.Tensor(shape, dtype)` 第二位置参数，不用 `dtype=` 关键字）。

```python
# 工厂层（trace-time，lru_cache 特化）
#   shape 校验：S == C*Q、H % G == 0；NUM_KERNELS = 24（NPUUtils 实查，§5.5）
#   num_logical = B * C * H；bn = min(64, N)；bl = bs = bp = 64
#   trace 分派（R4）：band_free = (Q % bs == 0)——主 trace（惩罚掩码）；
#                     band 域（Q % bs != 0）追加 mask/zero 常量与 vselect 链

@tilelang.jit(out_idx=[-1], target="npuir")
def kernel(x, cb, dA_cumsum, C_mat, prev_states, dt, out):
    with T.Kernel(24, is_npu=True) as (kernel_id, subid):
        num_local_tasks = T.ceildiv(num_logical - kernel_id, 24)
        # ---- 核级常量（任务循环外，R4）：对角掩码（惩罚主选，band 域另备 vselect）----
        idx = T.alloc_ub((bl, bs), "float32")                        # arange 行列差（f32 精确整数）
        T.arange(idx, [1, 0], 0); T.vsub(idx, idx_j_col, idx)        # d = i − j（PL-1.11 同款负步长形态）
        T.vmin(idx, 0, pen_const)                                    # vmin(d,0) → 0/负
        T.vmul(pen_const, PEN, pen_const)                            # 0/−PEN 常量矩阵（PEN=1e30）
        # band 域 trace 追加：mask_const = vcmp(arange 差, 0, "le")（bool）+ zero_f32 = vbrc(0)

        for task_id in T.serial(num_local_tasks):
            cid = T.min(task_id * 24 + kernel_id, num_logical - 1)   # ghost 钳位
            bz, bc_idx, bh = decode(cid)        # (b, c, h) 标量索引
            # ---- 任务头（R3）：因子源（行/列双形态，方向显式）----
            T.copy(dA_cumsum[bz, bh, bc_idx, 0:Q], dA_ub[0:Q, 0:1])      # (Q,1) 列形
            T.copy(dA_cumsum[bz, bh, bc_idx, 0:Q], dA_ub_row[0:1, 0:Q])  # (1,Q) 行形
            T.copy(dt[bz, bh, bc_idx, 0:Q], dt_ub[0:Q, 0:1])             # 自动 VCast→fp32
            T.copy(dt[bz, bh, bc_idx, 0:Q], dt_ub_row[0:1, 0:Q])
            for pp in T.serial(P_tiles):                              # bp=64
                p0 = pp * bp
                for lt in T.serial(L_tiles):                          # bl=64
                    l0 = lt * bl; tl = T.min(bl, Q - l0)
                    # ---- 因子（R3/OPT-4，v1 方向化）----
                    vbrc(anchor, anchor_col[bl,1]); T.vsub(dA_ub[l0:l0+bl,0:1], anchor_col, tmp_l)
                    T.vexp(tmp_l, exp_l[bl,1])                        # 行因子 [M,1]（≤1）
                    vbrc(anchor, anchor_row[1,Q]); T.vsub(anchor_row, dA_ub_row, tmp_s)
                    T.vexp(tmp_s, tmp_s); T.vmul(tmp_s, dt_ub_row, exp_s_all[1,Q])  # (1,Q) 行形态
                    # ---- history（R2）：n-loop 流水（acc 首块 initC=True 覆写初始化）----
                    for n_blk in T.Pipelined(N_tiles, num_stages=2):
                        T.copy(C_mat[bz, cs+l0 : cs+l0+tl, bg, n0 : n0+tn], c_tile)
                        T.copy(prev_states[bz, bc_idx, bh, p0 : p0+tp, n0 : n0+tn],
                               state_t[0:tp, 0:tn])                   # 直装载 [bp,bn]
                        T.gemm(c_tile, state_t, acc, initC=(n_blk == 0),   # ← acc 此处初始化
                               b_transpose=True, size=[tmc, tk, tnp])
                    T.vmul(acc, exp_anchor, acc)                      # OPT-2 步骤 1（tensor·scalar）
                    # ---- full-lower s-loop（对角块移出；T.serial 主选，§6.1）----
                    for s_blk in T.serial(l0 // bs):
                        s0 = s_blk * bs; ts = T.min(bs, Q - s0)
                        T.copy(cb[bz, bc_idx, bg, l0 : l0+tl, s0 : s0+ts],
                               cb_f32[0:tl, 0:ts])                    # GM→UB 直拷 + 自动 VCast
                        T.copy(x[bz, cs+s0 : cs+s0+ts, bh, p0 : p0+tp], x_tile[0:ts, 0:tp])
                        T.vmul(cb_f32, exp_s_all[0:1, s0 : s0+ts], lcb_f32)  # [M,N]⊙[1,N] 探针 P-1
                        #   （fallback：vbrc(exp_s_col[1,bs] → [bl,bs]) + 同 shape vmul）
                        T.vcast(lcb_f32, lcb, round_mode="rint")
                        T.gemm(lcb, x_tile, acc, initC=False, size=[tmc, tks, tnp])
                    # ---- 行缩放（OPT-2 步骤 2，行广播 [M,1]）----
                    T.vmul(acc, exp_l, acc)
                    # ---- 对角块（R4/OPT-6，s-loop 后单独处理；惩罚主选）----
                    s0 = (l0 // bs) * bs; ts = T.min(bs, Q - s0)
                    T.copy(cb[bz, bc_idx, bg, l0 : l0+tl, s0 : s0+ts], cb_f32[0:tl, 0:ts])
                    T.copy(x[bz, cs+s0 : cs+s0+ts, bh, p0 : p0+tp], x_tile[0:ts, 0:tp])
                    vbrc(dA_ub[l0:l0+bl, 0:1] → dA_l_mat[bl,bs])      # 行向展开 [M,1]→[M,N]
                    vbrc(dA_ub_row[0:1, l0:l0+bl] → dA_s_mat[bl,bs])  # 列向展开 [1,N]→[M,N]
                    T.vsub(dA_l_mat, dA_s_mat, dA_l_mat)              # in-place diff
                    T.vadd(dA_l_mat, pen_const, dA_l_mat)             # 惩罚融进指数
                    T.vexp(dA_l_mat, dA_l_mat)
                    T.vmul(cb_f32, dA_l_mat, lcb_f32)
                    T.vmul(lcb_f32, dt_ub_row[0:1, s0 : s0+ts], lcb_f32)  # [1,bs] 列因子·探针 P-1
                    # band 域此处改：T.vselect(mask_const, lcb_f32, zero_f32, lcb_f32) + 列 OOB 守卫
                    T.vcast(lcb_f32, lcb, round_mode="rint")
                    T.gemm(lcb, x_tile, acc, initC=False, size=[tmc, tks, tnp])
                    # ---- 写回 ----
                    T.copy(acc[0:tl, 0:tp], out[bz, cs+l0 : cs+l0+tl, bh, p0 : p0+tp])
```

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|----------|------|
| `T.Kernel(n, is_npu=True)` 一维 persistent | docs/开发指南.md §3.3 官方模板；GQA 生产代码同款 | ✅ 已验证（模板级） |
| `T.gemm(src1, src2, dst, size, initC, b_transpose)` | docs/Tilelang.language/线性代数操作/T.gemm.md（Developer Op；fp16/bf16 × fp32 dst；size/initC/transpose 参数 §2.1） | ✅ 文档确认 |
| `T.copy` slice 尾块形态 | docs/Tilelang.language/内存操作/T.copy.md（§2.2.2 尾块 extents 借用；§2.3 跨 dtype VCast 条 4；GM→UB/L1→L0C 方向表） | ✅ 文档确认 |
| `T.vmul`（tensor·scalar / 行广播 [M,N]⊙[M,1] / 列广播 [M,N]⊙[1,N]） | docs/Tilelang.language/数学操作/T.vmul.md §2.2.2（明示 **行广播 `[M,N]*[M,1]`** 与标量广播；**无 `[M,N]*[1,N]` 列因子形态**——仅 `[1,N]*[1,1]` 向量×标量）；生产先例：tensor·scalar = GQA L519/L654；行广播 [half,dim]⊙[half,1] = GQA L645（Expert）；**列广播无先例 → Stage 3 探针 P-1**，fallback = vbrc 列展开 + 同 shape vmul | ✅ 文档+先例确认（行/标量）；⚠️ 列因子形态探针（fallback 就绪） |
| `T.vmin` / `T.vadd`（惩罚矩阵派生） | docs/Tilelang.language/比较操作/T.vmin.md；数学操作 T.vadd（惩罚链 PL-1.11 verified 同款算术） | ✅ 文档确认 |
| `T.vexp`（fp32） | docs/Tilelang.language/数学操作/T.vexp.md（fp16/fp32 ✓，bf16 ×——本设计 exp 全程 fp32 域，不受限） | ✅ 文档确认 |
| `T.vcast`（f32↔f16/bf16, round_mode） | docs/Tilelang.language/数据类型转换操作/T.vcast.md（f32→f16/bf16 全 round mode；f16→f32 rint） | ✅ 文档确认 |
| `T.vbrc`（标量/向量广播，[·,1]→[·,n]） | docs/Tilelang.language/shape操作/T.vbrc.md（§2.2.2 向量广播 rank 一致 + 尺寸 1 规则） | ✅ 文档确认 |
| `T.vcmp` + bool 片上使用 / `T.vselect` | docs/Tilelang.language/比较操作/T.vcmp.md（§2.3 条 4：bool 不可写回 GM，配合 vselect）；条件操作/T.vselect.md | ✅ 文档确认 |
| `T.arange`（int16 索引矩阵） | docs/Tilelang.language/创建操作/T.arange.md（int16 ✓ strides 形态） | ✅ 文档确认 |
| `T.Pipelined(num_stages=2)` | flash_attn_npuir_dev.py L55 生产先例；docs/开发指南.md | ✅ 先例确认 |

**未实证形态（v1 修订：Stage 3 API 探针 P-1/P-2，非 API 存在性问题而是形态问题）**：**P-1** = `T.vmul` 广播——①行 `[M,N]⊙[M,1]`：文档明示（T.vmul.md §2.2.2）+ GQA L645 Expert 先例，Developer 编译待 L0 验证；②**列 `[M,N]⊙[1,N]`：无文档形态无生产先例**（v0 错向 `[bs,1]` 曾静默算错）——验证 = 编译通过 + L0-1 精度对照；fallback：vbrc 列展开 + 同 shape vmul（+1 op/处）。**P-2** = `T.vsub` 广播 `[bl,bs]−[1,bs]`（§1.6.3 候选 #2，失败 fallback：vbrc×2 + vsub）。两探针均为编译期可裁决，进入 Stage 3 首日清单。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测 5 项）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源为 `T.Kernel(l·p, B·C, H)`） | 一维乘积展开 + persistent 24 核核内串行（§0.6 R1；ascend-constraints.md 限制 1） |
| GPU 专用 API（threads/swizzle/sync_threads） | Yes（threads=128、make_swizzled_layout、T.sync_threads） | 全部舍弃/替换（§0.5 行 3/13；TRAP-threads-kwarg-noop 线索 + NPU 仓 K9 已移除 threads） |
| GEMM 非整除（M/N 不被 block 整除） | 契约上可能（任意 Q/P/N）；当前 workload 全整除 | R6 尾块路径（T.min tail + slice copy + size-form gemm + 分形钳位 tmc/tnc ≥ 16、K ≥ 32 + stale band 掩码中和）；§8.2 L0 边界用例覆盖 |
| L0C 溢出（bl·bp·4B > 128KB） | bl·bp = 64×64 = 4096 × 4B = 16KB ≤ 128KB | ✅（§5.3 复核；备选 bl=128 时 32KB 仍 ✓） |
| 分核策略三要素 | Yes（逻辑核数 8–16384 跨越三档） | §5.5（实查 24 + persistent 串行方案） |

#### 3.5.2 参考实现差异说明（影响 API 选型的关键差异汇总；完整差异分析见 §0.5/§0.6）

| 差异项 | 参考实现（GPU） | 本项目（Ascend Developer） | 转换方案 |
|--------|----------------|---------------------------|----------|
| Kernel 维度 | 三维 `T.Kernel(·,·,H)` + threads | 一维 persistent 24（`is_npu=True`） | §0.6 R1；GQA/开发指南模板 |
| GEMM | `T.gemm`（GPU mma，fragment 累加） | `T.gemm`（Cube，`b_transpose`/`size`/`initC` 参数化） | docs T.gemm.md；state 转置经 b_transpose（R2） |
| 内存分配 | `alloc_shared/alloc_fragment`（SMEM/寄存器） | 同名 API（自动映射 L1/UB/L0C） | flash_attn_npuir_dev.py 先例 |
| 标量 exp/分支 | `T.exp` fragment 标量 + `if_then_else` | `T.vexp/vselect/vcmp` 向量链 | §1.6.2；R3/R4 |
| 布局 | SMEM swizzle | 无（NZ 分形框架处理） | §0.5 舍弃行 3 |
| 流水 | `T.Pipelined(num_stages=3)` cp.async | `T.Pipelined(num_stages=2)` MTE 双缓冲 | §6.3（容量重算） |

#### 3.5.3 本项目同类实现参考

**必须列出**：本项目 examples/ 中最相似的实现：

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/flash_attention/flash_attn_npuir_dev.py` | **高度相似**（Developer 模式 MixCV：gemm + v-prefix + Pipelined 同构） | Kernel 骨架（alloc_shared/fragment、T.copy slice、T.gemm initC/b_transpose、vbrc 标量、reduce、Pipelined(2)）、CV 交替自动编排、 Developer 环境变量 `TILELANG_ASCEND_MODE=Developer` |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 高（同族 MixCV 迁移，Expert 形态） | persistent 24 核 + ghost 钳位（T.ceildiv 陷阱绕法）、尾块分形钳位（E7）、**算术惩罚掩码 + band 域 vselect 分域（E3/PL-1.11 verified）**、vmul 行广播先例（L645）/标量乘先例（L519/L654）、精度双门禁（D8） |
| `examples/mixcv/mixcv_mixkernel.py` | 中（MixCV 双 Scope 最小样例） | load_nd2nz/store_fixpipe 形态（本设计未用——slice T.copy 路线，GQA D1 实证跨步区 load_nd2nz 误读）、sync_block_set/wait 语义 |
| `examples/TileOPs/tileops/testing/mamba2_reference.py` | golden 基准 | §8.1 直接采用 |
---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| x | (B, S, H, P) | fp16/bf16 | 主输入；S = C·Q；行（s 维）stride = H·P 元素，行内 p 连续 |
| cb | (B, C, G, Q, Q) | 同 x | 预计算耦合；group-owned（同组 head 复读，L2 缓解）；tile 切片为最后两维 |
| dA_cumsum | (B, H, C, Q) | fp32 | 单调非增；[Q] 连续切片直拷 |
| C_mat | (B, S, G, N) | 同 x | readout；行 stride = G·N，行内 n 连续 |
| prev_states | (B, C, H, P, N) | fp32 | **P 在 N 前**：state_t 直装载 [bp,bn] 行 stride = N，行内 n 连续（R2 免转置） |
| dt | (B, H, C, Q) | 同 x | [Q] 连续切片直拷（自动 VCast→fp32） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| out | (B, S, H, P) | **fp32** | 与 x 同 layout（非转置）；写回 slice [cs+l0 : cs+l0+tl, p0 : p0+tp] |

### 4.3 中间缓冲区

> Developer 模式（v1 修订用词）：gemm 操作数 tile 用 `T.alloc_shared`（编译器映射 L1）；**向量 op 操作数 buffer 显式 `T.alloc_ub`**（T.vmul.md §2.3 条 1「操作数 buffer 必须分配在 UB 上」强制；T.vmul/T.vexp 文档示例即 Developer 简单 jit 形态用 alloc_ub）；`T.alloc_fragment`（gemm dst / 向量链中转，flash_attn_dev L34 `scores` fragment 作 vmul 操作数先例）。设计默认 bl=bs=bp=64、bn=64、Q=256（w2 标定）；**对角链采用 in-place 链接**（`vsub(a,b,a)`/`vadd(a,p,a)`/`vmul(a,c,a)`，flash_attn_npuir_dev.py L62 `T.vmul(scores, scales, scores)` in-place 先例）压缩 liveness 峰值。

| Buffer 名 | Shape | dtype | 存储层级（逻辑） | 用途 |
|-----------|-------|-------|------------------|------|
| c_tile | (bl, bn) | dtype | L1（`alloc_shared`，gemm 操作数） | history 的 C tile |
| state_t | (bp, bn) | dtype | L1（`alloc_shared`，gemm 操作数） | prev_states 直装载（R2） |
| x_tile | (bs, bp) | dtype | L1（`alloc_shared`，gemm 操作数） | intra 的 x tile |
| acc | (bl, bp) | fp32 | L0C（fragment，gemm dst + vmul 操作数） | 统一累加器（**history n-loop 首块 gemm `initC=True` 覆写初始化**——无需预清零；→ 标量乘 → full-lower 累加〔`initC=False`〕→ 行缩放 → diag 累加〔`initC=False`〕；hist_acc 并入单 buffer，省 16KB） |
| cb_f32 | (bl, bs) | fp32 | UB（`alloc_ub`） | cb 的 GM→UB 直拷 + 自动 VCast（跨 dtype T.copy，docs T.copy.md §2.3 条 4；无 dtype 中转 tile） |
| lcb | (bl, bs) | dtype | fragment→gemm 操作数 | 因子化后的 cb（scores_cast 同位，flash_attn_dev L74 先例） |
| dA_l_mat / dA_s_mat | (bl, bs) | fp32 | fragment（向量链，in-place） | diag 的 diff 链展开（行/列 vbrc；v1：dt 因子改 [1,bs] 直乘探针，dt_mat 全矩阵展开仅 fallback 需要） |
| pen_const | (bl, bs) | fp32 | UB（`alloc_ub`，核级常量） | 算术惩罚矩阵（0/−PEN，arange 行列差派生，任务循环外一次；整除域主选） |
| mask_const / zero_f32 | (bl, bs) | bool / f32 | UB（`alloc_ub`，核级常量；**仅 band 域 trace 分配**） | vselect 因果掩码与 else 分支（NaN 免疫，R4 band 域） |
| dA_ub / dA_ub_row | (Q,1) / (1,Q) | fp32 | UB（`alloc_ub`） | dA 列/行双形态（列形供 exp_l；行形供 exp_s_all 计算与 diag 列因子切片） |
| dt_ub / dt_ub_row | (Q,1) / (1,Q) | fp32 | UB（`alloc_ub`） | dt 双形态（行形供 exp_s_all 计算与 diag 列因子 [1,bs] 切片） |
| exp_s_all | (1, Q) | fp32 | UB（`alloc_ub`） | full-lower 列因子源（OPT-4；切片 [0:1, s0:s0+bs] 为 [1,bs]） |
| exp_l | (bl, 1) | fp32 | UB（`alloc_ub`） | 行因子（OPT-2；vmul 行广播文档明示形态） |
| anchor_col / anchor_row / tmp | (bl,1)/(1,Q)/(1,Q) | fp32 | UB（`alloc_ub`） | 因子计算中转（liveness 短） |
| idx（arange 中转） | (bl, bs) | fp32 | UB（`alloc_ub`，仅核初始化期存活） | pen_const 生成的 arange 索引差（init 后死，liveness 回收） |

### 4.4 内存搬运路径

```
GM[dA[b,h,c,:]] ──T.copy──▶ UB[dA_ub (Q,1) / dA_ub_row (1,Q)]（fp32 直拷）
GM[dt[b,h,c,:]] ──T.copy(auto VCast)──▶ UB[dt_ub / dt_ub_row]（f32）
GM[cb[b,c,g,l0:l0+tl, s0:s0+ts]] ──T.copy(auto VCast)──▶ UB[cb_f32 (bl,bs) f32]
GM[C[b,cs+l0..,g,n0..]] ──T.copy──▶ L1[c_tile (bl,bn) dtype]
GM[prev[b,c,h,p0..,n0..]] ──T.copy(auto VCast)──▶ L1[state_t (bp,bn) dtype]
GM[x[b,cs+s0..,h,p0..]] ──T.copy──▶ L1[x_tile (bs,bp) dtype]
L1[c_tile] × L1[state_t] ──T.gemm(b_transpose)──▶ L0C[acc]（n-loop，initC 首块覆写）
UB[cb_f32] ──vmul(exp_s_all [1,bs] 列因子·探针P-1)──▶ fragment[lcb dtype]（full-lower）
UB[cb_f32] ──diff 链(vbrc×2/vsub/vadd(pen)/vexp/vmul×2)──▶ fragment[lcb dtype]（diag 整除域）
UB[cb_f32] ──diff 链+vselect(mask)──▶ fragment[lcb dtype]（diag band 域）
fragment[lcb] × L1[x_tile] ──T.gemm(initC=False)──▶ L0C[acc] 累加
L0C[acc] ──vmul(exp_anchor / exp_l)──▶ L0C[acc]（标量乘/行缩放，Vector 侧）
L0C[acc] ──T.copy──▶ GM[out[b,cs+l0..,h,p0..]]（fp32 直写）
```

（`idx_i/idx_j → vsub → vcmp` 掩码常量链在核初始化期 UB 内完成，无 GM 流量。）

### 4.5 UB 内存预算

> 设计默认 bl=bs=bp=64（[64,64] fp32 = 16KB / dtype = 8KB）；**liveness 分配（v1 修订：惩罚主选形态）**：核级常量（pen 16KB；整除域不分配 mask/zero）与任务稳态期分开计；任务内 full-lower 链与 diag 链共享 cb_f32/lcb/dA_l_mat/dA_s_mat；diag 峰值 = dA_l_mat + dA_s_mat + cb_f32 + lcb（dt 列因子 [1,bs] 仅 256B，dt_mat 全展开仅 fallback 形态需要 +16KB）。

| Buffer（稳态峰值集合，整除域主 trace） | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| pen_const | (64,64) | f32 | 16,384 |
| dA_ub + dA_ub_row + dt_ub + dt_ub_row | (256,1)/(1,256) | f32 | 4 × 1,024 |
| exp_s_all (1,256) + exp_l (64,1) + anchor 双形态 + tmp | — | f32 | ~3,600 |
| acc | (64,64) | f32 | 16,384 |
| cb_f32 + lcb | (64,64) | f32 / dtype | 16,384 + 8,192 |
| dA_l_mat + dA_s_mat（diag 峰值，in-place 链） | (64,64) ×2 | f32 | 32,768 |
| **整除域稳态峰值合计** | | | **≈ 99 KB** |
| **× 1.12 膨胀系数**（CONST-capacity-910B2C 2026-09-15 重校口径，显式 alloc 结构） | | | **≈ 111 KB** |
| band 域 trace 增量（mask 4KB + zero 16KB + 探针 P-1 失败的 vbrc 展开 +16KB/处〔fallback 计 2 处〕） | | | ≈ +52KB → **~163KB（×1.12 后 ~151KB + 备用 vbrc 展开仍在限内）** |
| 对照：× 1.7 悲观口径（TRAP-UB-multibuffer-inflation 旧口径；PL-1.11 r9 实测显式 alloc 结构 ratio 0.981，1.7 已证过悲观） | | | ≈ 168 KB（整除域，仍 < 192KB） |

**结论（v1）**：整除域主 trace 按 1.12 口径 **111KB ≤ 192KB（余量 42%）**；band 域/探针 fallback 叠加最坏 ~163KB 仍限内；1.7 悲观口径 168KB 亦不再超限（v0 的 196KB 贴限源于 vselect 形态的 zero 16KB + mask/idx 常量，惩罚主选后已消除）——**Stage 3 首次编译仍以 BishengIR `ub overflow` 报错文本实测校准**（CONST-capacity 条目方法），超限时的消减序：① dt 列因子 fallback 的 dt_mat 并入 dA_s_mat（dt 预乘进 dA_ub_row 切片，数学等价需重验 §1.6.1 口径）；② bl=64/bs=32（diag 链 [64,32] 减半，K 分形 ✓）；③ band 域 mask/zero 降为 [bl,bs] 最小形。L1 侧（c_tile + state_t + x_tile = 24KB，Pipelined 双缓冲 48KB ≤ 512KB）与 L0C 侧（acc 16KB ≤ 128KB）均宽裕。

### 4.6 动态轴定义

**无**（工厂 `lru_cache` 按 (B, C, Q, H, P, N, G, dtype) 静态特化；契约 shape 由 host `forward` 校验——`S == C·Q`、`H % G == 0`，kernel 内全部循环边界为 trace-time 常量或 kernel_id/循环变量派生 PrimExpr）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[-1], target="npuir")
# out_idx=[-1]：仅 out 为输出（对齐源工厂 @tilelang.jit(out_idx=[-1])）
# Developer 模式经环境变量 TILELANG_ASCEND_MODE=Developer 声明
#（flash_attn_npuir_dev.py L94 先例；不设 pass_configs——自动流水/自动多缓冲交由编译器，
#  UB 预算按 §4.5 校准，超限时 Stage 3 再评估 NPUIR_ENABLE_AUTO_MULTI_BUFFER=False）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：混合（MixCV：Cube 双 GEMM 路径 + Vector 因子/掩码/缩放链）

**判定依据**：算子含 2 类 matmul（history `c_tile@state_t`、intra `lcb@x_tile`）+ 逐元素前处理（因子化链）与后处理（行缩放），同为 Developer 模式自动 CV 切分结构（AGENTS.md Developer MixCV 触发规则：T.gemm + v-prefix 同 kernel）。

### 5.2 Block 划分

```python
bl = 64   # l-tile（Q 维）：源默认同值；L0C acc 16KB；Q∈{64,128,256} 全整除
bs = 64   # s-tile（s 维）：源默认同值；diag 直接差分的掩码块
bp = 64   # p-tile（P 维）：源默认同值；P∈{64,128} 整除（P=128 时 2 块）
bn = min(64, N)   # n-tile（history K 维）：N∈{32,64,128}；bn=32 时 K 贴分形下限 32 ✓
NUM_KERNELS = 24  # persistent 物理核数（§5.5 实查）
# 逻辑任务（任务粒度 = (b, c, h)，R1）：num_logical = B * C * H
# 核内展开：p-loop × l-loop × (n-loop + full-lower s-loop + diag)
```

**取值理由**：64 对齐源 GPU 默认 config（block_l/p/n/s=64）——已验证的负载形态；分形 M/N=64 ≥ 16、K=64（或 32）≥ 32 ✓；尾轴 64×2B = 128B ≥ 32B 对齐 ✓；L0C acc [64,64]×4B = 16KB ≤ 128KB（约束 `bl·bp ≤ 16384` ✓ 余量 8×）。

### 5.3 约束分析

- **对齐约束**：bl/bp/bn/bs ∈ {32,64}，尾轴 32B 对齐 ✓（fp16 64 元素 = 128B）；分形限制 M/N ≥ 16、K ≥ 32 ✓（smoke1 N=32 → bn=32 贴 K 下限）。
- **UB 容量**：稳态峰值 ~115KB × 1.12 ≈ 129KB ≤ 192KB ✓（§4.5；1.7 悲观口径贴限 → Stage 3 实测校准）。
- **L0 容量**：acc [64,64] fp32 = 16KB ≤ 128KB ✓；L1：24KB×2 流水 ≤ 512KB ✓。
- **GEMM 非整除**：契约 shape 任意时 `Q % bl`、`P % bp`、`N % bn` 可能非零——R6 尾块路径（当前 manifest/test 全整除，尾块为契约完备性设计）。

### 5.4 注意事项

1. **T.ceildiv 负数陷阱**：`num_local_tasks = T.ceildiv(num_logical − kernel_id, 24)` 在 `kernel_id ≥ num_logical`（欠载域）时返回 1 而非 0（截断除法 + 1 的 lower 行为，GQA 2026-09-15 工具链实证）——ghost 任务以 `cid = T.min(·, num_logical − 1)` 钳位到合法任务重算（bit-identical 写回，安全）。w1（8 任务/24 核）即落此域。
2. **对角块数值边界（v1 修订：惩罚主选分域论证）**：diag 直接差分的下三角指数恒 ≤ 0（s ≤ l ⟹ dA_l ≤ dA_s）；上三角正指数真实分布 ~e^32、深衰减 stress 可至 fp32 上限——**整除域**（惩罚掩码）：上三角指数被 `−PEN`（PEN=1e30）吸收为 −PEN（大数吃小数）→ exp(−PEN) 下溢精确 +0.0，且 PEN 为有限哨兵规避 −inf−(−inf) NaN 路径（PL-1.11 −1e38 同理）；**band 域**（vselect）：上三角大值被选路丢弃。双域 cast 输入均有限 → 无 inf/NaN 进 gemm；深衰减角点由 verify_equiv deep_decay 案（dA→−256）+ L0 门禁覆盖。
3. **bn=32（N=32 workload）**：history gemm K=32 贴分形下限，单 n-block 无流水——正确性无碍，性能由 Stage 4 评估。
4. **bf16 路径**：gemm bf16×bf16→fp32 ✓（docs T.gemm.md §2.3 条 3"建议 fp32，实测可用"）；vexp 仅 fp32 ✓（本设计 exp 全程 fp32 域）；GQA bf16 先例为 f32 ws 载体（保守），本设计 gemm 操作数保持 bf16（对齐源数值路径），精度风险列 §9。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **物理核数（要素②，实查）**：**24**。查询代码与返回值记录：

```python
from tilelang.utils.npu_utils import NPUUtils
print(NPUUtils.get().get_aicore_num())   # 2026-09-17 03:58 实查输出：24（Ascend910B2C，npu-smi 确认 910B2C 在位）
```

（混合 Cube+Vector 算子直接使用返回值 24；与 GQA 设计 2026-09-15 实查记录（24）及 CONST-aicore-910B2C 条目一致。）

- **逻辑核数（要素①）**：任务粒度 = `(b, c, h)`（§0.6 R1），`num_logical = B × C × H`：
  - w1 smoke（1,2,·,4）：1×2×4 = **8**
  - w2 mamba2-780m-b1-s4k（1,16,·,48）：**768**
  - w3 mamba2-2p7b-b4-s2k（4,8,·,80）：**2560**
  - w4 mamba2-1p3b-b2-s32k（2,128,·,64）：**16384**
  - 测试 fixture：(1,2,·,4)=8、(1,2,·,4)=8、(2,4,·,8)=64、(2,2,·,4)=16
- **规模判定（要素③）**：w2–w4 逻辑核数 768–16384 ≫ 24（**极大规模**，无法通过调整 block 缩减——任务粒度由数据维度决定）；w1/部分测试 case（8–16 < 24，欠载域）。
- **分核方案**：**固定启动内核数 = 物理核数 24，核内 `T.serial` 串行处理多个逻辑任务**（极大规模标准方案，开发指南 §3.3 官方模板）：`T.Kernel(24, is_npu=True)`，每核 `num_local_tasks = T.ceildiv(num_logical − kernel_id, 24)`、任务解码 `cid = T.min(task_id·24 + kernel_id, num_logical − 1)`（轮转 + ghost 钳位，注意事项 1）。**循环边界静态性**：`num_local_tasks` 为 kernel_id 派生的 PrimExpr（进入循环前一次计算、循环内不变；开发指南 §3.3 官方模板同形，GQA E1-E7 D3 实证可 lower）——不含运行时动态 shape。**欠载域处置**（w1 等 num_logical < 24）：16 个核空载 + ghost 钳位重算（bit-identical 安全，正确性无损；smoke 域性能不敏感）；Stage 4 可按 shape 分派非 persistent 直发形态（`T.Kernel(num_logical)`，lru_cache 工厂天然支持，PL-1.4 工厂层回退分发先例），不在 Stage 1 特化。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环，无任何逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务级（(b,c,h) 分派） | persistent 核内串行 | `T.serial(num_local_tasks)` | 分核策略要素③（§5.5）；轮转解码 + ghost 钳位 |
| p 维（P=128 时 2 块） | 块级迭代 | `T.serial(P_tiles)` | 任务内 p-tile 展开（P=64 时单次） |
| l 维（Q/bl 个 l-tile） | 块级迭代 | `T.serial(L_tiles)` | 因子（anchor/exp_l/exp_s_all）per-l-tile 重算 |
| history n 维 | **分块流水** | `T.Pipelined(N_tiles, num_stages=2)` | 边界 `ceildiv(N, bn)` 为全局常量（trace-time 静态）✓ 安全；c_tile/state_t 装载与 gemm 重叠 |
| intra full-lower s 维 | 块级串行（主选） | `T.serial(l0 // bs)` | 边界依赖外层 l-loop 变量（循环携带 PrimExpr）——`T.Pipelined` 对循环携带边界的支持**未实证**（flash_attn_dev 先例仅全局常量边界），主选 `T.serial` 保正确性；`T.Pipelined` 升级列 Stage 4（§9 风险 R-3） |
| 对角块 | 单块（s-loop 外显式） | 顺序语句 | OPT-2 行缩放序约束（行缩放必须在 diag gemm 前） |
| 元素级 | 向量化 | v-prefix 链（§3.2） | §1.6.2 结论 |

### 6.2 循环伪代码

```python
with T.Kernel(24, is_npu=True) as (kernel_id, subid):
    # 核级常量（惩罚矩阵，一次；band 域另备 vselect mask/zero）
    T.arange(idx, [1, 0], 0)                    # f32 行列索引差
    ...vmin/vmul 派生 → pen_const (0/−PEN)...
    for task_id in T.serial(T.ceildiv(num_logical - kernel_id, 24)):   # 边界：kernel_id 派生 PrimExpr
        cid = T.min(task_id * 24 + kernel_id, num_logical - 1)          # ghost 钳位
        bz, bc_idx, bh = ...  # cid 解码（标量，§1.6.2 #10）
        T.copy(dA..., dA_ub / dA_ub_row); T.copy(dt..., dt_ub / dt_ub_row)  # 任务头（双形态）
        for pp in T.serial(P_tiles):
            for lt in T.serial(L_tiles):
                l0 = lt * bl
                # 因子（OPT-4 / R3，方向化：exp_l [bl,1] 行因子；exp_s_all (1,Q) 行形态）
                ...vbrc/vsub/vexp/vmul...
                for n_blk in T.Pipelined(N_tiles, num_stages=2):        # history（全局常量边界；
                    ...c_tile / state_t 装载 + T.gemm(b_transpose=True, initC=(n_blk==0))...  # 首块覆写初始化 acc
                T.vmul(acc, exp_anchor, acc)                            # OPT-2 步骤 1
                for s_blk in T.serial(l0 // bs):                        # full-lower（循环携带边界；与 §3.3 一致 T.serial）
                    ...cb_f32 直拷 / x_tile 装载 + 列因子链（探针 P-1）+ T.gemm(initC=False)...
                T.vmul(acc, exp_l, acc)                                 # OPT-2 步骤 2（行广播缩放）
                ...diag 块（vadd(pen) 惩罚链〔整除域〕/ vselect〔band 域〕+ T.gemm(initC=False)）...
                T.copy(acc[0:tl, 0:tp], out[...])                       # 写回
```

### 6.3 流水线优化

- **history n-loop**：`T.Pipelined(num_stages=2)`（c_tile/state_t 双缓冲，L1 24KB×2 ≤ 512KB 宽裕）；源 num_stages=3 在 NPU 的 UB/L1 容量与 buffer 数下取 2（flash_attn_dev L55 同值先例；三缓冲的边际收益交 Stage 4 实测）。
- **full-lower s-loop**：主选 `T.serial`（循环携带边界的 Pipelined 未实证，§6.1）；Developer 编译器的自动调度（cv_split + 依赖分析）仍可提供跨语句级重叠（copy → vcast/vmul → gemm 的三引擎交替）。
- **任务间流水**：persistent 结构下相邻任务的 dA/dt 装载与上一任务写回可由编译器重叠（自动调度）；显式任务级双缓冲列 Stage 4。

### 6.4 尾块处理

**当 Q/P/N 不被 bl/bp/bn 整除时**（契约完备性路径；当前 workload 全整除）：`tl = T.min(bl, Q − l0)` 等尾块尺寸 → slice-form `T.copy`（src/dst 双侧显式 slice，extents 借用规则 docs T.copy.md §2.2.2 条 2–4）+ size-form `T.gemm(size=[tmc, tk, tnp])` + 分形钳位 `tmc = max(16, ceil16(tl))`、`tnp = max(16, ceil16(tp))`、`tk ≥ 32`（GQA E7 形态）。stale band 的中和：装载仅拷真实行列；diag 掩码链的列上界（`j ≥ tl_s` 强制 0）；写回仅拷 `[0:tl, 0:tp]`。**host 侧不做 padding**（ascend-constraints.md §4）。
---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步**（Developer 模式；编译器按数据依赖插入 MTE/VEC/CUBE 管线同步）

### 7.2 同步点说明

Developer 模式无手动同步点（不使用 `T.sync_block_set/wait`/`T.set_flag/wait_flag`/`T.pipe_barrier`）。编译器需识别的依赖边界（供 Stage 3 验证检查点，非手写代码）：

| 依赖边界 | 语义 | 编译器处理（预期） |
|----------|------|--------------------|
| dA_ub/dt_ub 装载 → exp 因子链 | MTE2 → VEC | 自动依赖序 |
| cb_f32 装载（GM→UB 含 VCast）→ vmul 链 | MTE2+VCast → VEC | 自动 |
| 因子链（lcb fragment）→ T.gemm | VEC → CUBE（fragment 作 gemm A 操作数，flash_attn_dev L74→L81 同构） | 自动 cv 协同 |
| hist gemm（n-loop）→ vmul(exp_anchor) → full-lower gemm 链 → vmul(exp_l) → diag gemm | L0C ↔ VEC ↔ CUBE 交替 | 自动（acc 单 buffer 的 RAW 链） |
| acc → T.copy 写回 | L0C → MTE3 | 自动 |

**跨核**：任务间零数据依赖（每 (b,c,h) 任务写独立 out 切片）、无 workspace、无跨核归约——无需任何核间同步（与 GQA 的 flag 协议形成对照：本算子无两相位结构）。

### 7.3 pass_configs 配置

```python
# 主选：不设 pass_configs（Developer 默认全开：自动流水 / 自动多缓冲 / cv_split）
# 风险预案（§4.5 UB 超限时启用，Stage 3 裁定）：
#   pass_configs={
#       tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,   # 关自动多缓冲，按手工 UB 预算
#   }
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。**直接采用 NPU 仓已移植的参考实现**（`examples/TileOPs/tileops/testing/mamba2_reference.py` 中的 `ssd_chunk_scan_fwd_ref` 函数，与源仓同名文件逐行对应）：纯 fp32 PyTorch materialize（einsum 双路径 + tril 掩码，无因子化、无 dtype 量化、无向量化结构）。

```python
def golden_ssd_chunk_scan_fwd(x, cb, dA_cumsum, C, prev_states, dt, n_groups):
    """官方对齐参考（独立于 NPU 算法）：fp32 materialize。
    x:[B,S,H,P] cb:[B,C,G,Q,Q] dA:[B,H,C,Q] C:[B,S,G,N] prev:[B,C,H,P,N] dt:[B,H,C,Q]
    → out:[B,S,H,P] fp32
    """
    b, S, h, p = x.shape
    _, _, c, L = dA_cumsum.shape
    n = C.shape[-1]; g = n_groups; hpg = h // g
    x_chunked = x.float().reshape(b, c, L, h, p)
    C_chunked = C.float().reshape(b, c, L, g, n)
    C_heads = C_chunked[:, :, :, torch.arange(h) // hpg, :]           # [B,C,L,H,N]
    dA = dA_cumsum.float().permute(0, 2, 3, 1)                          # [B,C,L,H]
    y_off = torch.einsum("bclhn,bchpn->bclhp", C_heads, prev_states.float())
    y_off = y_off * torch.exp(dA).unsqueeze(-1)
    cb_heads = cb.float()[:, :, torch.arange(h) // hpg, :, :]
    decay = torch.exp(dA_cumsum.float().unsqueeze(-1) - dA_cumsum.float().unsqueeze(-2))
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
    decay = decay.masked_fill(~mask, 0.0).permute(0, 2, 1, 3, 4)       # [B,C,H,L,L]
    dt_s = dt.float().permute(0, 2, 1, 3).unsqueeze(-2)                 # [B,C,H,1,L]
    lcb = cb_heads * decay * dt_s
    y_diag = torch.einsum("bchls,bcshp->bclhp", lcb, x_chunked)
    return (y_off + y_diag).reshape(b, S, h, p)
```

（与仓内 `ssd_chunk_scan_fwd_ref` 等价；`tests/ops/test_mamba.py` 的 `ref_program` 即此函数——**Stage 3 直接复用测试基建**，勿另写。）

### 8.2 精度标准与 L0 门槛测试计划

| dtype | atol | rtol | 来源 |
|-------|------|------|------|
| float16 | 1e-3 | 1e-5 | tests/ops/test_mamba.py 既有门禁（对齐 GPU 基线） |
| bfloat16 | 2e-3 | 1e-5 | 同上 |

**L0 门槛测试计划**（Stage 3 阻塞项；完整 L1/L2/Boundary 套件交 tilelang-op-develop）：

| # | 用例 | 验证目标 |
|---|------|----------|
| L0-1 | smoke1 (1,2,64,4,64,32,1) fp16 | 主 trace 精度（含 ghost 钳位域 8 任务/24 核） |
| L0-2 | smoke2 (1,2,128,4,128,32,1) bf16 | bf16 路径 + P=128 双 p-tile + Q=128 双 l-tile（full-lower 路径激活） |
| L0-3 | full3 (2,4,64,8,64,64,2) fp16 | G=2 分组索引 + N=64 + 多 batch |
| L0-4 | full4 (2,2,64,4,64,32,2) bf16 | bf16 + G=2 + 尾块邻域（Q=64 单 l-tile 纯对角） |
| L0-5 | 边界 shape（非整除：Q=96/P=48/N=48 等契约内组合） | R6 尾块路径 + 分形钳位（契约完备性；当前 manifest 无此 shape，新增定向用例） |
| L0-6 | wrapper 契约（lru_cache 工厂、out shape/dtype = (B,C·Q,H,P) fp32、contiguous 输入） | 接口不变性（迁移 prompt 要求） |

**等价性验证已前置**：`verify_equiv.py`（§1.6.1 结果表，EQUIV_PASS）——设计期已机器拦截 K 侧因子移动的噪声放大（否决项回归记录在案，Stage 2 复核可重跑）。

---

## 9. 风险点与注意事项

### 9.1 已知约束（技术约束检测结论汇总）

| 约束 | 结论 | 处置 |
|------|------|------|
| 三维 Kernel 不支持 | **触发**（源 `T.Kernel(·, B·C, H)` 三维） | §0.6 R1 一维 persistent + 串行（§5.5） |
| GPU 专用 API（threads/swizzle/sync_threads） | **触发** | §0.5 舍弃/替换（threads kwarg 在 npuir 无效果——TRAP-threads-kwarg-noop，stale 线索 + NPU 仓 K9 移除佐证） |
| GEMM 非整除 | 契约可能 / 现役 shape 无 | R6 尾块 + L0-5 用例 |
| L0C 溢出 | 未触发（16KB ≤ 128KB） | §5.3 复核 |
| 分核适配 | **触发**（768–16384 逻辑核 ≫ 24） | §5.5 persistent 串行 + ghost 钳位 |
| UB 容量 | **贴限风险**（1.7 悲观口径 ≈196KB > 192KB；1.12 口径 129KB ✓） | §4.5 消减序 + Stage 3 BishengIR 报错实测校准 |

### 9.2 常见错误与风险登记

| # | 风险 | 触发场景 | 影响 | 缓解 |
|---|------|----------|------|------|
| R-1 | **`T.vmul` 广播形态分两类风险（v1 修订）**：① 行广播 `[M,N]⊙[M,1]`——文档明示（T.vmul.md §2.2.2）但生产先例 GQA L645 为 Expert，Developer 模式待 L0 验证；② **列广播 `[M,N]⊙[1,N]`——无文档形态亦无生产先例**（文档仅 [M,N]*[M,1] 行广播与 [1,N]*[1,1] 向量×标量；v0 曾以 `[bs,1]` 错向形态书写且因 bl==bs 形状兼容而**静默算错**——检视数值演示 max\|Δ\|=7.32） | 编译期/静默数值错误 | 行广播编译失败（低风险）；列广播误写错向形态则**静默算错**（高风险，v0 实际发生） | ① Stage 3 探针验证；② **探针 P-1** 显式验证 `[M,N]⊙[1,N]`（编译通过 + L0-1 精度对照），fallback：`vbrc([1,bs] → [bl,bs])` 列展开 + 同 shape vmul（+1 op/处，UB +16KB/处——预算 §4.5 band 行有余量）；形态约定已写入 §1.6.3/§3.2/§3.3/§4.3 四方一致 |
| R-2 | **跨 dtype `T.copy`（GM fp32 → UB fp16/bf16 自动 VCast）的数值路径**（state_t/dt 装载；round_mode 默认值未文档化于 T.copy 条目） | 运行时精度 | 与源 `T.cast`（默认 round）的舍入差异 | L0 精度门禁覆盖；超差时改为显式 vcast 链（fp32 UB 中转） |
| R-3 | **s-loop `T.Pipelined` 循环携带边界未实证**（flash_attn_dev 先例仅全局常量边界） | 编译期 | full-lower 流水降级 | 主选已定为 `T.serial`（§6.1）；Pipelined 升级为 Stage 4 项（若编译器支持则免费收益） |
| R-4 | **UB 膨胀系数不确定**（1.10–1.12 与 ~1.7 两版实测口径，CONST-capacity-910B2C 内部演进，条目 stale） | 编译期 | §4.5 贴限 | Stage 3 实测校准 + 消减序（§4.5 ①②③） |
| R-5 | **bf16 gemm 操作数直连**（GQA bf16 先例用 f32 ws 载体保守；本设计对齐源路径用 bf16 操作数） | 精度 | bf16 用例超差 | L0-2/L0-4 门禁；超差 fallback：lcb/cb 走 fp32→bf16 的 GQA 保守载体（gemm 仍 bf16，中转精度提升） |
| R-6 | **dA 非单调输入**（上游异常数据，违反 §0.1 假定） | 运行时数值 | anchor 因子化前提失效（exp_s 切片 > 1 区域扩大、diag 上三角大值仍被掩码——不崩溃但数值路径偏离） | 语义上源实现同样依赖该假定（§0.1）；不在 kernel 内加校验（与源一致）；风险登记 |
| R-7 | **向量链 op 数为第一疑似瓶颈**（§1.6.0 roofline 发射项） | 性能 | 首版延迟高于流量下界数倍 | Stage 4 清单：bl=128（§1.6.3 备选 B）、s-loop Pipelined（R-3）、vsub 广播（探针 P-2）、惩罚 vs vselect 的 Developer 模式 A/B 复证（PL-1.11 实测为 Expert 形态，Developer 下 cv_split 对 vcmp 的行为待实测——若 Developer 下 vcmp 不标量化则两形等价） |
| R-8 | **T.ceildiv 负数陷阱**（欠载域 ghost 任务） | 正确性（已缓解） | 越界任务 | ghost 钳位（§5.4 注意 1，GQA 实证绕法）；L0-1 覆盖 8 任务欠载域 |

### 9.3 特殊场景处理

- **非整除分块**：R6 尾块路径（slice + size gemm + 分形钳位 + stale band 中和）。
- **极小 shape**（Q=64 单 l-tile 纯对角、B·C·H=8 欠载）：ghost 钳位保正确性；性能不敏感域。
- **混合精度**：fp16/bf16 双 dtype 同构 trace（仅 dtype 字符串分派，无结构分支）；exp 全程 fp32（vexp 不支持 bf16 的约束已规避）。
- **深衰减 dA**（→ −256）：OPT-2 的 exp(anchor) 下溢行为与源 exp(dA_l) 下溢一致（verify_equiv deep_decay 案 PASS）；diag 上三角大指数被掩码丢弃（无 inf/NaN 路径，§0.6 R4 论证）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/
├── _ssd_chunk_scan_fwd_kernel.py   # 算子实现（Stage 3 产出：重写 tileops/kernels/mamba/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernels.py 为 target="npuir"）
├── DESIGN.md                       # 本设计文档
├── verify_equiv.py                 # §1.6.1 等价性机器验证脚本（已交付并执行，EQUIV_PASS）
├── RETROSPECTIVE.md                # Stage 1 复盘（自进化钩子）
└── history_version/                # 历史版本备份（revision 时使用）
```

> 注：本算子属于 TileOPs harness 迁移模式——kernel 实现最终落位于 `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernels.py`（重写为 npuir 版），wrapper/Kernel class（`ssd_chunk_scan.py`）与 op/tests/bench 基建已由 Stage 0 移植就位；独立目录 `examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/` 承载设计工件与验证脚本（DESIGN.md/verify_equiv.py/RETROSPECTIVE.md）。

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档（迁移任务含 §0） |
| `verify_equiv.py` | ✅ 已交付并执行 | 等价性机器验证（OVERALL: EQUIV_PASS，2026-09-17） |
| `_ssd_chunk_scan_fwd_kernel.py`（npuir 实现） | ⬜ 待实现（Stage 3） | 按 §3.3 伪代码重写迁移目标文件 |
| golden 函数 | ✅ 已就位（复用） | `examples/TileOPs/tileops/testing/mamba2_reference.py` 中的 `ssd_chunk_scan_fwd_ref`（§8.1） |
| 测试/性能基建 | ✅ 已就位（复用） | `tests/ops/test_mamba.py` / `benchmarks/ops/bench_mamba.py`（Stage 0 移植） |

### 10.3 命名规范

- 项目目录：`ssd_chunk_scan`（conductor 指定）；算子目录：`_ssd_chunk_scan_fwd_kernel`（snake_case，与迁移源函数同名）
- 实现文件：`_ssd_chunk_scan_fwd_kernel.py`；测试文件：`tests/ops/test_mamba.py`（TileOPs harness 既有路径）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md，含 §0 迁移分析 + §1.6 调研与优化 + 机器验证）
2. ✅ Golden 函数（复用仓内 `ssd_chunk_scan_fwd_ref`）
3. ⬜ 算子实现（Stage 3：`_ssd_chunk_scan_fwd_kernels.py` 重写为 npuir Developer 版）+ L0 门禁（§8.2 六用例）+ TileOPs 测试全量

## 11. 性能目标（Stage 4 调优输入，conductor 追加）

- 性能目标类型: best_effort（用户指令：尽力调优，以 plateau / budget 中止）
- 目标数值: 无（best_effort 不设硬性目标）
- Baseline: Stage 3 产物 `_ssd_chunk_scan_fwd_kernel.py`（本目录，L0/L1/L2/Boundary 全过 + 二次校验通过）
- 测试 shape: manifest `SSDChunkScanFwdOp` workloads（mamba2-780m-b1-s4k / mamba2-2p7b-b4-s2k / mamba2-1p3b-b2-s32k + smoke）
- 噪声阈值: 3%（默认）
- 最大迭代数: max_rounds=10、max_experiments=30（默认）
- 编程模式约束: **Expert**（用户指定；Stage 3 实现已为 Expert 模式——Developer + persistent 分核 + gemm 存在模式级不兼容，运行时崩溃 "unaligned UUB addresses"，见 RETROSPECTIVE 与 repro 探针）
- 主指标: `msprof op` Task Duration(us)（唯一 kernel 时延测量口径）
