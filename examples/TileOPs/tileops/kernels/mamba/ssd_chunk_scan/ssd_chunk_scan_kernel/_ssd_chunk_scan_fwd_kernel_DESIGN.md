# _ssd_chunk_scan_fwd_kernel 算子设计文档（SSDChunkScanFwdOp · GPU→NPU 迁移 · Expert 模式 MixCV）

> 项目：`ssd_chunk_scan`（TileOPs Mamba-2/SSD 族，NPU 仓 `examples/TileOPs`）；算子：`_ssd_chunk_scan_fwd_kernel`（迁移任务，harness 模式，**revision v2**，上一版 `history_version/design_v1.md`）。
> 源算子：`/home/tilelang/l00970450/TileOPs/tileops/kernels/mamba/ssd_chunk_scan.py`（GPU TileLang，三维 grid + SMEM swizzle + 双向 GEMM + anchor 因子化）。
> 迁移目标接口：`examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/_s_s_d_chunk_scan_fwd_kernels.py`（Stage 0 pattern B 提取；`threads` 参数为 GPU 后端实现参数，NPU 侧迁移时移除——接口其余部分不变）。
> 用户指定编程模式：**expert**（显式指定，不可改为 developer）。
> 物理核数实查（2026-09-20 12:31，本机 Ascend910B2C，当前工具链 `tilelang 0.1.2+4515de8`；Stage 2 两轮检视独立复查均 24）：`NPUUtils.get().get_aicore_num()` 返回 **24**（混合 Cube+Vector 算子直接使用）。

## 相对上一版（design_v1）的关键调整（revision v2）

> 触发：Stage 2 复检不通过（1 阻塞 + 1 建议，见 REVIEW.md 复检轮）。**已确认保留（v1 全部 7 项修复经复检核实合格）**：wait 任务头（令牌推演复核通过）、x 任务×pp 级装载、Vector pp 绑定（方案 a）、R3 per-task 口径（独立复算精确一致）、表述簇②③④、UB 数字 177.1/180.7KB（与检视逐项求和 181,376B 精确一致）、host cast 三件套论证——本轮仅修 **band 组装的搬运原语**（v1 在「写实」v0 占位式 band gemm 时引入了 L1→L1 拷贝这一当前工具链实测编译失败的非法方向）与 §3.2 同步遗漏。

| # | 调整 | 检视问题 | 为何不会再犯同一错误 |
|---|------|---------|---------------------|
| 1 | **band 组装恢复 HEAD 旧 verified 终版的单 buffer GM→L1 直载形态**（六处同步）：① §4.3 删除 `l1_lcb (bl,bs)` 中间 buffer 行，`l1_band = T.alloc_L1([bl, Q], dtype)` 为唯一 band buffer；② §1.4/§6.2 s_blk 循环改**单跳** `T.copy(ws_lcb[kernel_id, slot, pp, lt, s_blk, 0:bl, 0:ts], l1_band[0:bl, s0:s0+ts])`——src/dst 双侧按 `ts = T.min(bs, Q−s0)` 裁剪（TRAP-L1-band-dst-tail-overrun verified：未裁剪的 s0+bs 在 Q%bs≠0 尾块越界 [bl,Q] 缓冲、污染相邻 L1 分配，B-q96-bf16 回归 max_diff 7.8e-3 在案）；③ band gemm 单条 `T.gemm(l1_band, l1_x, l0_acc, initC=False, size=[tmc, l0+tl, tnp])`（K=l0+tl 连续因果带，保留）；④ §3.2 步骤 7 同步为单条口径（copy 行 src `0:ts` 裁剪 + gemm 行 `size=[tmc, l0+tl, tnp]`，删 `l1_x[s0:s0+ts]` 逐块切片与 ts size——v1 的混合形态字面实现每条 gemm 恒读带前 ts 列，结果错误）；⑤ §4.4 路径图改单跳；⑥ L1 合计 104→**96KB**（明细 16+16+32+32，删 l1_lcb 8KB）。**依据**：v1 的 `T.copy(l1_lcb, l1_band[...])` 为 L1→L1 拷贝——docs T.copy.md 搬运方向表无此方向、全仓 examples 零先例、**检视编译探针 1 实测当前工具链硬失败**（`'hivm.hir.copy' op Unsupported copy from cbuf to cbuf!`）；单 buffer 直载形态经**探针 2 实测当前工具链编译通过**（"AscendNPU IR compile success"），且为 HEAD 旧 verified 终版形态（段数墙/217.65µs 锚点即以此形态测得，恢复后与继承裁决口径自洽） | 复检问题 1（阻塞） | 每个新增搬运原语先过三关：T.copy.md 方向表在列 + 全仓 examples 先例 + 最小编译探针实证；「写实占位伪代码」时必须对照引用的 verified 实现逐行比对落地形态，不得自创未经证实的搬运方向 |
| 2 | **§2.3 残留措辞**：「x 任务级缓存」→「x 任务×pp 级缓存」；该行 buffer 清单删 l1_lcb 后改「gemm 操作数 + x 缓存 + band 组装区」 | 复检问题 2（建议） | 措辞统一类修订以全文模式检索收尾（本轮一致性检查含 §2.3） |

## 相对 design_v0 的关键调整（revision v1，复检已全部核实）

> 触发：Stage 2 首轮检视不通过（3 阻塞 + 4 建议）。**已确认保留**：§0.1–§0.4 源算子三问解读、§1.6.0 算法族选型（分块双路径）与 R1 候选表、OPT-A/OPT-B 采纳与 OPT-X/OPT-Y 否决（verify_equiv 27 cases 重跑双确认）、R1–R8 重设计决策、分核三要素、§1.6.3 布局决策与继承裁决——仅修**伪代码落地层与数字口径**，未推翻任何已通过决策。

| # | 调整 | 检视问题 | 为何不会再犯同一错误 |
|---|------|---------|---------------------|
| 1 | **Vector 域 slot-free wait 从任务尾移到任务头**（§1.4/§6.2/§7.2 行 #4 三处同步）：`T.sync_block_wait(CONS(slot))` 置于任务解码后、任何 ws[slot] 写之前；`set READY(slot)` 保持任务尾。修复 WAR 竞争——v0 尾置形态下 T=2 对 ws[0] 的写只被 prologue 令牌放行，不受 Cube 消费完任务 0 的 set 约束（违反 §0.6 R7「cons 先于覆写」）。修正后任务 W 的同槽写受 C 完成任务 W−2 的 set 约束（深度 2 节拍），与 PL-1.12 update 及 HEAD 旧 verified 终版（L356–362，注释 "from task 2 on it waits for Cube's consumption of task (T-2)"）逐位一致 | 问题 1（阻塞） | 伪代码的每个 flag set/wait 都做令牌推演（set 序/wait 序配对 + 写序位置核对），并与 HEAD 旧 verified 实现行级比对后才落笔 |
| 2 | **x 装载移入 `for pp` 循环体首部**（§1.4/§6.2），slice 改 `x[bz, cs:cs+Q, bh, p0:p0+tp] → l1_x[0:Q, 0:tp]`（尾块 `tp = T.min(bp, P−p0)`）：修复 v0 把 [Q,P] 全宽 copy 放 pp 外与 `l1_x=[Q,bp]` 的形状矛盾（P=128 即 L0-2 门禁域触发）；「任务级一次装载」改述为「**任务×pp 级一次装载**（每 (task,pp) 一次，消跨 lt 2.5× 重读）」（§0.6 R7.4/§4.3/§6.3 同步）。对齐 HEAD 旧 verified 终版 L233/L242–243 | 问题 2（阻塞） | 伪代码中每个 `T.copy` 的 src/dst 形状逐维核对（含 P_tiles≥2 / Q 非整除 / N 非整除契约域），不再依赖"当前 workload 全整除"的侥幸域 |
| 3 | **Vector 域补 `for pp in T.serial(P_tiles)` 循环**（lt 蛇形分片循环之外，§1.4/§6.2）：修复 v0 的 ws 写索引 `pp` 未绑定（P=128 时漏产 pp=1 因子）。**选方案 (a)（逐 pp 重复生产）**，依据：① HEAD 旧 verified 终版即此形态（L364），生产实证（旧任务 Stage 4 final + Stage 5 集成全绿）；② 改动最小（ws 形状/Cube 侧/§4.1 不动，P=64 全域零差异）；③ 方案 (b)（因子 pp-不变性去 ws 的 P_tiles 维，P=128 域 ws 流量减半）属结构性变更——双域流水平衡点与 ws 读索引语义变化须实测裁决，修订轮不引入，**列为 §1.6.3 Stage 4 优化候选** | 问题 3（阻塞） | Vector/Cube 双域的循环嵌套与 ws 索引维度逐项对照（哪个循环变量绑定哪个 ws 维），未绑定变量在自检中显式检出 |
| 4 | **§1.6.0 R3 ws 流量口径修正为 per-task**：ws_c 单程 64KB（P_tiles·L_tiles·bl·N·2B@w2，v0 误用 per-l-tile 16KB）+ ws_lcb 单程 80KB（Σ(lt+1)·bl·bs·2B，v0 误用 per-AIV 40KB），写+读 ≈ **216MB@w2**（v0 87MB，低估 ~2.5×）；D-2 流量项区间同步更新 **~128–214µs** 并注明「仍低于段数项、下界结论不变（段数+scalar 170–280µs 主导）」 | 问题 4（建议） | R3 每个流量数字标注口径（per-task / per-l-tile / per-AIV），并按 ws 形状逐维复算 |
| 5 | **表述级修正簇**：① §1.4 alloc 清单补 `l1_band = T.alloc_L1([bl, Q], dtype)`、band gemm 操作数写实（`l1_band[0:bl, 0:l0+tl]` slice）；② §1.6.2 「平均 5 s-block/l-tile」→「10 s-block/任务 ÷ 2 AIV = 5 s-block/AIV」；③ §1.6.3 「2016-09-17」→「2026-09-17」；④ §0.7/§2.2 旧集成版在位性表述改为「HEAD 提交 4515de8 在位（工作树已由本任务 Stage 0 重建删除，git show 可查）」 | 问题 5（建议） | 占位式命名（`...` 后缀）与无出处的"在位"断言列入自检禁则 |
| 6 | **UB/L1 数字统一**：§4.3 Vector 稳态 176.4→**177.1KB**（逐项和 181,376B 复算）、band 域 180.4→**180.7KB**、L1 统一 **104KB**（§1.6.0 R4 行 ~90KB 与 §4.5 104KB 的不一致消除；§4.5/§5.3/§1.6.3 同步）〔v2 注：本轮删 l1_lcb 后 L1 更新为 96KB，见 v2 调整表 #1〕 | 问题 6（建议） | 容量数字以逐项求和表为准（§4.3 明细 → §4.5 汇总 → 各处引用同一组数） |
| 7 | **prev_states host cast 合规补强**（§0.6 R2/§0.5 coalesced 行）：① 引用 HEAD 旧集成版 host cast 先例（L345–348 注释 "source casts prev_states (fp32) to the input dtype before the gemm"，经 PR #188 合并、旧任务 Stage 5 集成全绿）作为项目内既判实践；② 对「核内 Vector vcast + ws_prev 中继」替代给量化否决（段数墙 +~10%、UB 贴限 +~12KB、每任务多一次握手相位内生产——与 c_scaled 链同构但净亏）。处置结论（host cast）维持 | 问题 7（建议） | host 侧操作的合规论证 = 约束条款 + 同族先例 + 替代方案量化否决三件套，不再只引 API 约束单证 |

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
- NPU 版必须保持同构数值路径（fp16/bf16 操作数 + fp32 累加 + 相近的量化点），否则对 golden（纯 fp32 PyTorch）的偏差水平会偏离源实现的已验证基线。本设计 Expert 形态的一处**已论证偏移**：history 行缩放 `exp(dA_l)` 从「gemm 后 fp32 域缩放」移到「A 操作数装载前缩放后 cast」（§1.6.1 OPT-A，机器验证噪声比 0.63–1.07，容差内等价）。

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

**典型 workload**（manifest `mamba.yaml` SSDChunkScanFwdOp 条目，决定设计标定域）：

| label | B | S | H | P | C | Q | G | N | dtype | 逻辑任务 B·C·H |
|-------|---|---|---|---|---|---|---|---|-------|--------------|
| ssd-chunk-scan-smoke | 1 | 128 | 4 | 64 | 2 | 64 | 1 | 32 | fp16/bf16 | **8** |
| mamba2-780m-b1-s4k（w2） | 1 | 4096 | 48 | 64 | 16 | 256 | 1 | 128 | fp16/bf16 | **768** |
| mamba2-2p7b-b4-s2k（w3） | 4 | 2048 | 80 | 64 | 8 | 256 | 1 | 128 | bf16 | **2560** |
| mamba2-1p3b-b2-s32k（w4） | 2 | 32768 | 64 | 64 | 128 | 256 | 1 | 128 | fp16 | **16384** |

测试精度门禁（tests/ops/test_mamba.py）：atol = 1e-3（fp16）/ 2e-3（bf16），rtol = 1e-5。
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
| 9 | **L-side anchor 因子化**：`exp(dA_l−dA_s) = exp(dA_l−anchor)·exp(anchor−dA_s)`，anchor = dA[l0] | MUFU（exp 单元）计数从 bl·bs 降为 bl+bs；**数值稳定**（full-lower 域两因子指数均 ≤ 0，无溢出；深衰减 chunk 下不产生下溢为 0 的伪影） | full-lower 块因子分离 | — | **可移植**（纯算法层，依赖 dA 单调性这一语义属性） |
| 10 | 上三角 s-block 完全跳过（loop bound `ceildiv(l0+bl, bs)`） | 省 ~50% 对角 GEMM 计算 | causal 结构裁剪 | — | **可移植**（纯算法层） |
| 11 | 微块 2×2 因子化（M=32，仅 block_l=block_s=64 启用） | 对角块 MUFU 削减（~50%）+ (0,1) 微块零填充跳过 | diag 块拆微块，(1,0) 严格下三角因子化 | GPU 标量 MUFU 吞吐瓶颈 | 算法层可移植，**但收益机制硬件相关**（NPU 向量整块发射，微块化反而增加 op 数） |
| 12 | register 累加（fragment acc 跨循环驻留） | 消中间写回 | 寄存器文件容量 | 大寄存器堆 | 硬件强相关（NPU 用 L0C/UB 等价） |
| 13 | threads=128 / autotune 配置集 | 占用率与 ILP 平衡 | warp 调度 | warp 结构 | 硬件强相关（NPU 无 threads 概念） |
| 14 | `lru_cache` 工厂（shape 特化编译缓存） | 消重复编译 | Python 层缓存 | — | 可移植 |

> 识别不出来 ≠ 不存在。以上 14 项覆盖源码全部显式优化注释与结构（swizzle/流水/hoist/因子化/微块/负载均衡）。迁移中 #1–#7、#12–#13 需按 §0.5 处置；#8–#10、#14 为算法层/工程层直接保留；#11 的**意图**（对角块代价削减）须由 NPU 结构承接（见 §0.5 行 11）。
### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定依据：migration-analysis.md §5.3 映射表 / examples 佐证 / docs 条目 / capability-gaps 登记；Phase R 调研结论〔§1.6.0〕作为判定输入——调研确认分块双路径即该数学语义的并行最优结构族，源算法未被替换，重设计方向为「结构保留 + NPU 双引擎重构」。）

**编程模式的前置约束（决定全部重设计形态）**：本算子为 persistent 分核 + Cube gemm + Vector 因子链的混排结构。**CG-2026-0010**（capability-gaps，2026-09-17 登记，本算子族同结构实证）：Developer 模式对 persistent 分核 + `T.gemm` + v-prefix 混排**运行时崩溃**（"Illegal instruction, which is usually caused by unaligned UUB addresses"，vector core exception retCode=0x31），无编译期诊断；**Expert 模式（显式双 Scope）同结构完全正常**（PL-1.16 两任务实证：GQA 首证 + 本算子族第二证）。用户指定 expert 模式与该实证一致——本设计全程 Expert 双 Scope 形态（`T.Scope("Cube")` + `T.Scope("Vector")`），**不存在 Developer 回退路径**。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（§0.1 全部） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 以此为基线 |
| 双路径分解（history + intra-causal）+ causal 上三角跳过（#10） | 算法 | 无（纯数学结构） | — | **保留** | 调研（§1.6.0 R3/R4）确认分块双路径为最优结构；块循环边界 `ceildiv(l0+bl, bs)` 语义照搬 |
| L-side anchor 因子化（#9，full-lower 域）+ exp hoist（#8） | 算法 | 无 | — | **保留（限 full-lower 域）** | 数值稳定结构保留；Expert 形态下 Vector 因子链与 Cube gemm 双域流水重叠、AIV 非关键路径（PL-1.18 实证 AIV vec 62% 非关键），因子化形态简化为**直接差分链**（与 golden 数值路径一致、无条件安全；对角域因子化经机器验证否决，见 §1.6.1 OPT-Y）——**意图承接**：#9 的 MUFU 削减意图由「Vector 整块 vexp 发射 + 双域重叠」承接（NPU 向量 op 整块处理 [64,64]，无逐元素 MUFU 瓶颈） |
| dA/dt 的 Q 标量缓存（#7） | 算法/优化边界 | shared memory | 有（UB） | **等价替换（受限形态）** | 理想形态「task 级 [1,Q] 行 buffer 驻留 UB」被 **CG-2026-0012** 阻塞：动态偏移 UB subview 进嵌套循环时 auto-multi-buffer pass 产出非支配 IR（"operand #2 does not dominate this use"，Q≥128 触发，v3_hoist 实证）——退回 **per-l-tile/per-s-block GM 直载**（[bl,1]/[1,bs] 小片 copy），复读流量由 L2 吸收（cube read_hit 91% / AIV 99% 实测） |
| SMEM tiling（#1） | 优化 | shared memory | 有（L1 512KB / UB 192KB） | **等价替换** | gemm 操作数 tile 走 `T.alloc_L1`（Expert 显式；GQA 生产先例）；向量因子 buffer 走 `T.alloc_ub`（T.vmul.md §2.3 条 1「操作数必须 UB」） |
| Tensor Core mma（#5） | 优化 | Tensor Core | 有（Cube） | **等价替换** | `T.gemm`（Expert 下直接调用，GQA/ssd 两生产实证；docs 线性代数操作/T.gemm.md 另提供 `T.npuir_dot` Expert 别名）；`b_transpose=True` 参数承接 state 转置 |
| coalesced 装载（#4） | 优化 | 线程-内存合并 | 有（MTE 引擎） | **等价替换** | `T.copy` slice 形态（2D strided tile 搬运由 MTE 完成；GQA D1 实证 slice-form 优于 load_nd2nz 处理跨步区）；**Expert 约束**：通用路径 T.copy 要求 src/dst dtype 一致（docs T.copy.md §2.2.1）且 GM→L1 ND2NZ 不支持 fp32（docs T.load_nd2nz.md §2.2.1/§2.3 条 1）——prev_states fp32→dtype 的 cast 移至 host 侧 `.to(dtype)`（cast 语义 rint 与源 `T.cast` 一致，等价性见 §1.6.1；合规论证与替代否决见 §0.6 R2——项目内既判实践 + 核内替代量化否决） |
| `T.Pipelined` 异步流水（#6） | 优化 | cp.async | 有（MTE + 双缓冲） | **等价替换（形态改变）** | Expert kernel 内**无 fragment 抽象，`T.Pipelined`/`T.Parallel` 不可用**（PL-1.16 Expert 硬边界）——流水意图由**任务级深度 2 双域流水**承接（Vector(T+1) 产因子与 Cube(T) 消费全重叠，PL-1.12 形态，§7）；块内装载-gemm 交替由 `T.rs` 管线区域标注静态排序 |
| register 累加（#12） | 优化 | 寄存器堆 | 有（L0C 128KB） | **等价替换** | `acc` 用 `T.alloc_L0C([bl,bp], fp32)` 作 gemm dst；bl·bp=4096 × 4B = 16KB ≤ 128KB ✓；**约束**：L0C 无 Vector 侧 rescale 原语（CG-2026-0011：gemm dst 不支持 L0C region 写、无「两 L0C acc 相加」原语）→ history 行缩放**不可**在 L0C 域做 → 折进 A 操作数（§0.6 R5/OPT-A） |
| 三维 grid + H 独立维（#2） | 优化 | CUDA 三维 grid | **无**（本项目 `T.Kernel` 仅一维） | **重新设计** | → §0.6 R1：三维乘积展开为一维逻辑任务 + persistent 24 核核内串行 |
| SMEM swizzle（#3） | 优化 | SMEM bank 结构 | **无此问题** | **舍弃** | NPU 无 bank conflict；NZ 分形布局由框架处理。**舍弃理由**：其目的（bank conflict 消除）在 NPU 不存在，等价收益由 MTE 搬运与 Cube 分形布局原生获得 |
| threads=128 / autotune（#13） | 优化 | warp 调度 | **无** threads 概念 | **舍弃** | `T.Kernel(threads=)` 在 npuir 无效果；autotune 已被 NPU 仓 K8 移除。**舍弃理由**：调度粒度由核数（24 核 × 2 AIV 分片）与向量化轴替代表达 |
| 微块 2×2 因子化 M=32（#11） | 优化 | GPU 标量 MUFU 吞吐 | 无同款瓶颈 | **舍弃（意图承接）** | **舍弃理由**：其目的是削减标量 exp 计数（GPU MUFU 稀贵）；NPU 向量 `T.vexp` 整块发射（一次 op 处理 [64,64] 全块），微块化将 7 op 拆为 28 op，在发射开销口径下为负优化。**意图承接**：对角块代价削减由「算术惩罚掩码核级常量 + 直接差分整块向量链」承担（§0.6 R4），(0,1) 微块置零跳过的意图由惩罚掩码（整除域）/vselect+列守卫（band 域）承接 |
| 对角块逐元素 `if_then_else` valid 掩码 | 实现 | 标量分支 | 有（向量算术惩罚/选择） | **重新设计** | → §0.6 R4：**算术惩罚掩码为主选**（整除域，PL-1.11 verified 实测 4.4–14.2x vs vcmp 标量化）；band-carrying 域（尾块 stale）保留 vselect + 列 OOB 守卫（NaN 免疫，§1.6.1 机器验证） |
| 尾块 guard `if_then_else` + safe 索引装载 | 实现 | 标量分支 | 有（slice 尾块拷贝） | **重新设计** | → §0.6 R6：`T.min` 尾块尺寸 + slice-form `T.copy` + size-form `T.gemm`（GQA E7 形态） |

**四态处置统计**：保留 4 项（语义/双路径+跳过/anchor+hoist〔受限形态〕/lru_cache）｜等价替换 6 项（dA/dt 缓存〔受限〕/SMEM tiling/mma/coalesced/Pipelined〔形态改变〕/register 累加）｜重新设计 4 项（三维 grid/对角掩码/尾块 guard + 新增跨引擎通路与 AIV 分片见 §0.6 R7/R8）｜舍弃 2 项（swizzle/threads，均附理由与意图承接）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（对照 Phase R 调研结论：源算法族保留，重设计聚焦 NPU 双引擎结构落地；源算法优化手段的意图承接已在 §0.5 逐项标注）。**本节形态大量承接本算子族旧任务的 Stage 3/4 实证结构**（知识预注入 E-1：CASE-ssd-chunkscan-migration / PL-1.12 / PL-1.13 / PL-1.14 / PL-1.15 / PL-1.16 / PL-1.18，条目戳 `tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / 2026-09-17`，被 kb_stale_check 标记 stale 系版本戳差异——当前工具链 `0.1.2+4515de8` 为旧任务收尾提交后的增量提交，**结构性结论（双 Scope 形态/ws 中继/深度 2 流水/AIV 分片/块连续布局）作为主选结构继承**，硬边界类结论（UB 容量/编译器支配性/gemm dst 形态）在当前工具链下标注「待重验」并列入 §9 风险与 Stage 3 首日验证清单）：

**重设计项 R1: 三维 grid → 一维 persistent 分核**

- **源方案**：`T.Kernel(l_tiles·p_tiles, B·C, H, threads=128)` 三维并行，block 数 = 4·16·48 = 3072（w2）等；意图 = 负载均衡 + 充足并行度。
- **NPU 新算法**：一维 `T.Kernel(NUM_KERNELS=24, is_npu=True) as (kernel_id, subid)`（persistent），逻辑任务 = `(b, c, h)`（**任务粒度取 chunk-head 而非 l×p tile**——因子预计算在任务内一次完成，ws 中继以任务为槽位）。任务解码 `cid = task_id·24 + kernel_id`（轮转），`num_local_tasks = T.ceildiv(num_logical − kernel_id, 24)`，**ghost 任务钳位** `cid = T.min(·, num_logical − 1)`（T.ceildiv 对负数返回 1 的陷阱，GQA 2026-09-15 工具链实证 + 绕法）。核内循环边界为静态推导（task_id/kernel_id 派生 PrimExpr，循环前计算一次）。
- **语义保持论证**：任务重排只改变 (b,c,h) 的计算位置与顺序，每个任务的计算语义与累加顺序（n_blk/s_blk 升序）不变；ghost 钳位任务重算已有任务的相同输出（相同输入 + 相同指令序 = bit-identical 写回，GQA 同款论证）；fp32 累加分组与源一致（块内由 Cube 分形完成，块间升序）——数值上块内 mma 顺序差异属 fp32 ulp 级（2^-24），远小于 fp16 操作数噪声（2^-11），容差内等价。边界：B·C·H=1 退化（单任务）时 persistent 24 核仅 1 核有效 + 23 ghost 重算——bit-identical 安全，性能由 Stage 4 shape 分派优化（不在 Stage 1 特化）。

**重设计项 R2: state_tile 转置装载 → 直装载 + `b_transpose=True`**

- **源方案**：`state_tile[nn,pp] = prev_states[b,c,h,p,n]`（**转置**装载，[bn,bp] SMEM），`T.gemm(c_tile[bl,bn], state_tile[bn,bp], hist_acc)`；注释表明迭代序 (pp,nn) 换取 nn 连续的 coalesced 128B 装载。
- **NPU 新算法**：`state_t[pp,nn] = prev_states[bz,bc_idx,bh,p0+pp,n0+nn]` **直装载**（[bp,bn]，内存布局原生方向：相邻 pp 行 stride=N、行内 nn 连续），`T.gemm(c_tile, state_t, acc, initC=(n_blk==0), b_transpose=True)`——转置由 Cube gemm 的 B 转置参数在分形内完成（docs T.gemm.md §2.1 `b_transpose`；GQA 生产先例）。gemm 输出直写统一累加器 acc。**Expert 约束**：T.copy 通用路径不跨 dtype（docs T.copy.md §2.2.1）且 GM→L1 ND2NZ 不支持 fp32（docs T.load_nd2nz.md §2.2.1/§2.3 条 1）——prev_states 的 fp32→dtype cast 在 host 侧 wrapper 完成（`prev_states.to(torch_dtype)`，rint 舍入语义与源 `T.cast` 一致）。
- **host cast 合规论证（v1 补强，问题 7）**：① **项目内既判实践**——HEAD 旧集成版 wrapper 即采用同款 host cast（`git show 4515de8` 提交内 `ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py`〔集成版，位于 §0.7 wrapper 同目录 `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/` 下；工作树已由 Stage 0 重建删除〕L345–348：`prev_states_c = prev_states.to(torch_dtype)`，注释原文 "source casts prev_states (fp32) to the input dtype before the gemm; replicate that cast here (Expert T.copy cannot cast across dtype)"——经 PR #188 合并、旧任务 Stage 5 集成全绿），ascend-constraints.md §4 禁止的是 host 改动**输入张量真实内容**，而 `.to(dtype)` 作用于副本（返回新张量、输入不变），与 `.contiguous()` 同属允许的 host 侧张量操作类别；② **核内替代（Vector vcast + ws_prev 中继）量化否决**——结构上可行（与本设计 c_scaled 链同构：copy fp32 GM→UB → vcast(→dtype) → copy UB→GM ws_prev → Cube L1），但代价为：ws_prev 中继 +16KB/task 写读流量（Cube mte2 段数墙 +~10%——prev 直载 128 段/task 变为中继读 128 段 + Vector 侧 MTE3 写段）、UB 贴限再 +~12KB（分块中转形式，§4.5 已 92.3% 贴限）、每任务多一次握手相位内的生产（Vector 临界路径 +~13 op/task）——三项合计净亏，且数值路径与 host cast 完全等价（同一 rint 舍入点）。**处置结论维持 host cast**。
- **语义保持论证**：`hist[ll,pp] = Σ_nn c[ll,nn]·state_t[pp,nn]` 与源 `Σ_nn c[ll,nn]·state[nn,pp]` 逐元素恒等（转置仅改变操作数布局，K 维求和序由 Cube 分形决定，fp32 累加容差内等价）；装载 guard 语义不变（§0.6 R6 统一处理）。

**重设计项 R3: 因子计算的标量/fragment 形态 → Vector 域向量链（直接差分）**

- **源方案**：`exp_dA_l/exp_l/exp_s` 为 `[bl]/[bs]` 标量序列，`T.Parallel` 循环逐元素 `T.exp`（GPU 标量 MUFU）；lcb 缩放为 `T.Parallel` 二重循环 `cb·exp_l[ll]·exp_s[ss]`。
- **NPU 新算法（Expert 双域分工）**：全部因子与 lcb 缩放计算移入 `T.Scope("Vector")`（UB buffer + v-prefix 链）；Cube 侧只做 gemm 消费。因子链采用**直接差分形态**（与 golden 的 `exp(dA_l−dA_s)` 数值路径一致）：
  1. per-l-tile：`dA_l_col = copy(dA_cumsum[b,h,c,l0:l0+bl])`（[bl,1] fp32）→ `vexp` → `exp_dA_l_col`（history 缩放因子源）。
  2. per-s-block：`vbrc(dA_l_col → dA_l_mat [bl,bs])`；`dA_s_row = copy(dA_cumsum[b,h,c,s0:s0+ts])`（[1,bs]）→ `vbrc(→ dA_s_mat [bl,bs])` → `vsub(dA_l_mat, dA_s_mat, diff)`（in-place）→ 对角块 `vadd(diff, pen_const)`（惩罚，R4）→ `vexp` → `vmul(cb_f32, ·)` → `vmul(·, vbrc(dt_s_row → dt_mat))` → `vcast(→dtype)` → `lcb_16` 写 ws_lcb。
  3. history：`c_f32 = vcast(copy(C[b,cs+l0..,bg,0:N]) →f32)` → `vmul(c_f32, exp_dA_l_col)`（**行广播 [M,N]⊙[M,1]，vmul 文档明示形态**）→ `vcast(→dtype)` → `c_scaled_16` 写 ws_c（R5）。
  - **vbrc 展开替代列广播**：exp_s/dt 的列因子方向（[1,N] 沿 M 广播）在旧任务 Developer 版曾是探针 P-1；Stage 4 v4_p2 实测**未文档化广播形态可编译可算对但有性能税 +12~16%**（PL-1.15，w2/w3 实测）——本设计直接采用文档化两步形态（`vbrc([1,bs]→[bl,bs])` + 同 shape vmul），不再设探针。
- **语义保持论证**：直接差分 `exp(dA_l−dA_s)` 是语义公式的原式（无因子化重排）；dA/dt 装载为同 dtype 路径（dA fp32 直拷；dt dtype 拷贝后 `vcast` rint→fp32，与源 `dt_smem = cast(dt, accum)` 一致）；块内 exp 参数域：下三角 s ≤ l ⟹ 指数 ≤ 0（dA 单调非增）无溢出，上三角由惩罚掩码（整除域）或 vselect+列守卫（band 域）中和——**无条件数值安全**（§1.6.1 机器验证 decay>88 角点 PASS）。
- **意图承接**：源 #7（dA/dt 缓存）→ per-l-tile/per-s-block GM 直载 + L2 吸收（CG-2026-0012 阻塞 task 级行驻留，§0.5）；源 #8/#9（exp hoist/anchor 因子化）→ **双域流水重叠承接**（Vector 因子链与 Cube gemm 并行执行，AIV 非关键路径——PL-1.18 实证 AIV vec 62% 稳态非关键；因子链发射数已非瓶颈，数值安全优先于发射削减）。

**重设计项 R4: 对角块因果掩码 → 算术惩罚掩码主选 + band 域 vselect + 列守卫**

- **源方案**：对角块逐元素 `valid = (s_abs <= l_abs)` 的 `if_then_else` 标量分支（+ 微块变体的分区掩码）。
- **NPU 新算法（整除域主选：算术惩罚掩码）**：惩罚矩阵 `(i < j ? −PEN : 0)` 是**核级常量**（与数据无关）：Vector 域核初始化一次 `T.arange`（[1,0] 行向 + [0,1] 列向，fp32 精确整数）→ `vsub`（i−j）→ `vmin(·, 0)` → `vmul(·, PEN)`（PEN=1e30 有限哨兵——GQA PL-1.11 用 −1e38 同理：有限大负数规避 −inf−(−inf) NaN 路径）→ `pen_const [bl,bs]`。对角块 diff 链中 `vadd(diff, pen_const)` 把惩罚融进指数：s > l 位指数 + (−PEN) → 大负 → `vexp` 下溢为**精确 +0.0**；s ≤ l 位惩罚恒 0，指数不变——**数学严格等价**（§1.6.1 机器验证：下三角逐位原值、上三角精确 +0）。**主选依据（PL-1.11 verified，2026-09-16 a13585dc 重验维持；本任务引用时核对 front-matter status 字段）**：int16 `vcmp` 全形态标量化（aiv_scalar 94–97%、对角块聚合 ≈ 壁钟 45%），算术惩罚掩码实测 **4.4–14.2x（几何 8.53x）**。
- **band-carrying 域（R6 尾块，Q%bs≠0 或 ts<bs）：vselect + 列 OOB 守卫双保险**——PL-1.11 NaN 免疫边界 + **本设计机器验证新证据**（verify_equiv.py stale-NaN 案）：stale 列（[ts,bs) 未初始化 UB 位）可能为 NaN 且**可跨越下三角**（i ≥ j ≥ ts 位）；① 加法掩码无法清除 NaN（NaN+x=NaN 污染下三角）；② **单一因果 vselect 也不足**（stale 列与下三角交集处的 NaN 被选路保留）；③ 完整形态 `(j ≤ i) & (j < ts)` 才干净——列守卫必须与 vselect 并存（装载侧 `T.copy` slice 只拷 [0:ts] 真实列 + 惩罚/vselect 域判据含 `j < ts`，或装载后清零 stale 列）。**当前 manifest/test 全部 Q∈{64,128,256} 对 bs=64 整除——主 trace 走惩罚形态**；band 域为契约完备性路径（L0 边界用例覆盖）。
- **NaN/inf 安全（双域）**：直接差分的下三角指数恒 ≤ 0；上三角正指数真实分布 ~e^32、深衰减可至 fp32 上限——惩罚域被 −PEN 吸收（大数吃小数）→ exp 下溢 0；vselect 域被选路丢弃；两域 cast 输入均有限。**对角块 L-side 因子化维持否决**（§1.6.1 OPT-Y 机器验证：decay>88 角点 NaN 污染）。
- **意图承接**：源 #11 微块 (0,1) 置零与逐元素 valid 分支的意图（无效面积零贡献 + 对角 exp 代价削减）由惩罚掩码一次性向量化 + 直接差分整块 vexp 承接。

**重设计项 R5: history 行缩放折进 A 操作数（Expert 特有，源于 §1.6.1 采纳项 OPT-A）**

- **源方案**：`acc += hist_acc · exp_dA_l[ll]`（history gemm 后的 [bl,bp] 行广播缩放，fragment 域）。
- **NPU 新算法**：Expert 形态下 acc 驻留 L0C（`T.alloc_L0C`），而 **L0C 无 Vector 侧 rescale 原语**（CG-2026-0011：gemm dst 不支持 L0C region 写、无「两 L0C acc 相加」原语；CONST-store-fixpipe-gm-only：L0C 跨引擎仅 GM 往返）——行缩放若走「L0C→GM→UB→vmul→GM→L1」中继，引入等量 GM 流量 + 第三握手相位，净亏（旧任务 opt_log §5 blocked 清单实证）。**方案**：利用行缩放的秩-1 结构，把 `exp(dA_l)` 折进 A 操作数：`c_scaled = cast(fp32(C) ⊙ e^{dA_l} → dtype)`（Vector 域，R3 步骤 3），`T.gemm(c_scaled, state_t, acc, initC=True, b_transpose=True)`——acc 从首块起即为最终 history 贡献，**全程纯 Cube 域驻留，零 rescale**。
- **语义保持论证**：数学恒等（标量行因子乘进 gemm 的 A 操作数 = gemm 后行缩放，分配律）；cast 位置从「C 本身」移到「C·e^{dA_l}」——被 cast 量随衰减缩小（dA_l ≤ 0 ⟹ e^{dA_l} ≤ 1），cast 绝对噪声随之缩小，与源码「cb 侧因子化」同向（衰减因子乘在值随因子缩小的操作数一侧，VP-2026-0083 pending 提案与本任务 §1.6.1 OPT-X 否决行同结论）；深衰减下 `e^{dA_l}` 下溢为 0 时 c_scaled 全行 0，history 贡献 0——与源 `exp_dA_l` 直接下溢为 0 的行为一致。**机器验证**：§1.6.1 FULL 行（OPT-A 含在内）fp16/bf16 × 4 角点 ratio 0.63–1.07 全 PASS。
- **收益**：每 l-tile 省 1 次 [bl,bp] L0C rescale 中继（GM 往返 ~32KB + 2 次握手相位）；ws_c 载体 dtype（[bl,N]，16KB/槽）。

**重设计项 R6: 尾块/非整除 guard 装载 → slice 尾块拷贝 + size-form gemm**

- **源方案**：所有 tile 装载用 `T.Parallel` 逐元素 + `if_then_else(guard)` + `T.min` safe 索引（标量分支填 0）。
- **NPU 新算法**：`tl = T.min(bl, Q − l0)` 等尾块尺寸 → `T.copy(GM[l0 : l0+tl, …], tile[0:tl, …])`（slice-form，尾块裁剪拷贝，docs T.copy.md §2.2.2 条 2–4 尾块借用规则）+ `T.gemm(…, size=[tmc, tk, tnp], initC=…)`（size-form 指定有效范围）。**分形钳位分层**：M/N 维 `tmc = max(16, ceil16(tl))`、`tnp = max(16, ceil16(tp))`——stale 行/列带 [tail, tmc) 参与 gemm 但只污染 stale 行/列自身的输出（写回裁剪 [0:tl] 丢弃）；**K 维优先用真实值 `tk = tn`**（tn ≥ 32 分形下限时 gemm K 维无 stale band，装载 slice 只拷 [0:tn] 真实列）；仅当 tn < 32 时钳位 `tk = 32` 引入 stale 列带——此时 stale 列会经 gemm K 维污染有效输出，必须中和（装载后清零 stale 列带，或 bn=32 单块路径）。intra 路径 K 维（s 维）同理：`tks = ts`（ts ≥ 32 时无 stale）；ts < 32 时对角块 stale 列走 R4 band 域（vselect + 列守卫）。**当前 manifest/test 全部 shape 整除**（Q∈{64,128,256}%64=0、P∈{64,128}%64=0、N∈{32,64,128}%32=0——最小 N=32 恰好 ≥ 分形 K 下限），尾块路径为契约完备性设计，正确性由 L0 边界用例覆盖。
- **语义保持论证**：slice 拷贝的越界语义 = 不拷贝（T.copy 区域语义规则 1：前向补 1 的 extents 借用），与源 guard 置 0 的差异在于 tile 内 stale 区不被清零——M/N 维 stale 行/列的贡献随写回裁剪丢弃（逐元素等价）；K 维在 tn ≥ 32 域由 size=tn 严格排除（逐元素等价）、tn < 32 域由 stale 列清零中和（清零后贡献 = 0 = 源 guard 语义）；**host 侧不做 padding**（ascend-constraints.md §4 禁止 host 改输入内容；prev_states 的 dtype cast 是 wrapper 对副本的操作、不改输入张量真实内容，属允许的 dtype 适配）。

**重设计项 R7: 跨引擎数据通路 → GM workspace 多槽中继 + 深度 2 任务流水（Expert 双 Scope 结构核心）**

- **源方案**：单 Scope 内 fragment/SMEM 直接互通（lcb_cast fragment 作 gemm A 操作数，无跨引擎边界）。
- **NPU 新算法**：Expert 双 Scope 下 Vector 与 Cube 是两条独立指令流，跨引擎数据经 **GM workspace 中继**（Vector 产因子写 ws → flag 握手 → Cube 读 ws 消费）：
  1. **ws 布局（块连续，PL-1.14 铁律）**：`ws_c [24核, 2槽, P_tiles, L_tiles, bl, N]`（dtype，c_scaled 因子）、`ws_lcb [24核, 2槽, P_tiles, L_tiles, L_tiles, bl, bs]`（dtype，lcb 因子）——每 [64,64]/[64,N] 块占连续 8/16KB；**band 化布局实测 +2.3% 回退**（v2_band：MTE3 跨步写 3× 慢，行 stride 512B）。
  2. **深度 2 任务流水（PL-1.12 形态 + Cube 前导 set）**：ws 双槽（slot = task_id % 2）+ 4 flag（ready/cons × slot0/slot1，≤ 15 预算内 ✓）；**Cube prologue 前导 set 两个槽的 cons flag**（初始"空闲"，Vector T=0/1 的 slot-free wait 立即通过，免除运行时分支）；Vector(T+1) 产因子与 Cube(T) 消费全重叠——**实测 −39.6%（w2 608→367µs）**，为全链最大单点。**深度 3 实测否决**（w2 +0.1%/w3 −2.7%/w4 +1.4% 平区；3 任务 in-flight 的 ws 工作集 9.2→13.8MB 劣化 L2 局部性，mte2 每条 256→346ns——深度 2 是 L2 甜点，PL-1.18）。
  3. **TASKDONE 类全 drain 屏障不设**（PL-1.12 冗余审计：ws 的跨 task WAR 由「Cube 单指令流程序顺序 + ready/cons 消费链 + AIV 串行」闭合，逐对象论证见 §7.2）。
  4. **x 任务×pp 级 L1 缓存 + Cube band 组装（v5_xband，实测 −1.8~−4.3% 全域一致）**：x tile [Q,bp] 在 `for pp` 循环体内、`for lt` 循环外一次装载 L1 驻留（每 (task,pp) 一次，消跨 lt 的 2.5× 重读；slice `x[bz, cs:cs+Q, bh, p0:p0+tp] → l1_x[0:Q, 0:tp]`，尾块 `tp = T.min(bp, P−p0)`——v1 修订：v0 的"任务级一次、0:P 全宽"在 P_tiles≥2 域与 [Q,bp] 形状矛盾，问题 2）；Cube 侧 band 组装——每 lt 的 s-block 序列在 L1 内组装为 band 形态后单条 band gemm（gemm 10→4/任务，ws_lcb 仍块连续读入 L1 列偏移区）。
  5. **pass_configs（PL-1.16 Expert 硬边界）**：`TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False` + `NPUIR_ENABLE_AUTO_MULTI_BUFFER: False`（不关闭时 planner 重排/重作用域 UB buffer → codegen "cannot find variable" 崩溃，ssd 第二证）。
- **语义保持论证**：ws 中继只改变数据的位置与时刻（Vector 计算的因子值经 GM 传递给 Cube gemm），数值无变换（dtype 一致直拷）；flag 握手保证生产-消费序（ready 先于读、cons 先于覆写——**精确索引口径（v1 修订，问题 1）**：Vector 任务 W 对 ws[slot=W%2] 的写受 Cube 完成任务 W−2 的 `set CONS(slot)` 约束〔wait CONS 前置于任务头任何 ws 写，深度 2 节拍；任务 0/1 消费 prologue 前导令牌立即通过〕）；流水深度不改变任何任务内的累加顺序。**带宽代价**：ws 中继引入因子流量（ws_lcb 640 段 + ws_c 128 段/任务的 Cube mte2 段数，PL-1.18 段数墙构成）——这是「Vector 产因子 → GM ws 中继 → Cube 消费」Expert 结构的物理流量，五个削减方向已实测穷尽否决（§1.6.0 roofline 行与 §9）。
- **意图承接**：源 #12（fragment 直通 gemm）的零中继意图无法在 Expert 双 Scope 复现（跨引擎仅 GM 往返，CONST-store-fixpipe-gm-only）——由深度 2 流水把中继延迟隐藏在双域重叠中承接。

**重设计项 R8: AIV 双子核分片（subid 蛇形均衡）**

- **源方案**：无对应（GPU warp 调度由硬件处理）。
- **NPU 新算法**：`T.Kernel(24, is_npu=True)` 的 Mix Block Dim = 48（每 block 2 AIV）——**Vector 程序默认在 2 个 AIV 上重复执行**（判据：两 AIV 子块 vec/mte ratio 完全相同而非各半；未分片时每 AIV 承担 100% 向量工作，产能 2× 浪费，PL-1.13）。分片形态（无 if 守卫，TRAP-tvm-parser-rules 运行时分支风险规避）：`for i in T.serial(lt_count): lt = i + subid + (i%2)·(L_tiles − 2i − 2·subid)` + 双向 clamp——**蛇形均衡**（L=4 → {0,3}/{1,2}，s-block 权重 5/5）；退化 L 折叠为重复 lt=0（bit-identical 良性）；分片维度（lt）与 ws 写区域（按 lt 索引）正交；双 AIV 仍执行相同 set/wait 序列（one-set-multi-wait 已证）。**实测 −28~−36%**（w2 323→212µs）。
- **语义保持论证**：分片只改变 Vector 因子计算的 (lt → AIV) 映射；ws 按 (kernel, slot, pp, lt) 索引，两 AIV 写区域不相交；退化域重复计算同一 lt 的因子 = bit-identical 覆写，安全。

**重设计项汇总**：R1（一维 persistent 分核）、R2（gemm b_transpose 直装载 + host cast）、R3（Vector 域直接差分因子链）、R4（惩罚掩码 + band 域 vselect/列守卫）、R5（history 缩放折进 A 操作数）、R6（slice 尾块 + size gemm）、R7（GM ws 中继 + 深度 2 流水 + band 组装 + pass_configs）、R8（AIV 蛇形分片）。§0.5 其余条目为保留（4 项）/ 等价替换（6 项）/ 舍弃（2 项，均附理由与意图承接）。

### 0.7 标杆实现

- **源算子路径（迁移对象）**：`/home/tilelang/l00970450/TileOPs/tileops/kernels/mamba/ssd_chunk_scan.py`（工厂 `_ssd_chunk_scan_fwd_kernel` + custom_op wrapper + `SSDChunkScanFwdKernel` 类）。
- **NPU 仓已提取的 kernel 骨架（迁移目标文件）**：`examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/_s_s_d_chunk_scan_fwd_kernels.py`（pattern B 提取，本设计 §3.3 伪代码的落地位置——**Stage 3 将其重写为 target="npuir" Expert 实现**）。
- **NPU 仓 wrapper/Kernel class（已移植，接口契约）**：`examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan.py`（K5/K7/K8/K9 适配完成：`npub::ssd_chunk_scan_fwd`、无 autotune、无 threads；default_config = {block_l:64, block_p:64, block_n:min(64,N), block_s:64, num_stages:3}——Stage 3 按本设计更新 default 为 {block_l:64, block_p:64, block_n:min(128,N), block_s:64, num_stages:2}，bn 提升依据见 §5.2）。
- **golden 参考实现（测试基准，源仓移植）**：`examples/TileOPs/tileops/testing/mamba2_reference.py`（`ssd_chunk_scan_fwd_ref` 函数）——纯 fp32 PyTorch materialize（einsum 双路径 + tril 掩码），**独立于本设计 NPU 算法**（无因子化、无 dtype 量化、无向量化结构），满足 golden 独立性要求（§8.1 直接采用）。
- **测试/性能入口**：`examples/TileOPs/tests/ops/test_mamba.py`（atol=1e-3 fp16 / 2e-3 bf16，rtol=1e-5）；`examples/TileOPs/benchmarks/ops/bench_mamba.py`。
- **同族 Expert MixCV 生产先例**：`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`（GQA prefill fwd：persistent 24 核 + 双 Scope + ws 中继 + flag 握手 + 尾块钳位 + 惩罚掩码——本设计 Expert 形态的主要结构参照，当前工作树在位）。本算子族旧任务集成版（集成 kernel + perf_opt 终版 + opt_log，目录 `ssd_chunk_scan_kernel/`——本节第三条 wrapper 目录 `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/` 下的子目录）**HEAD 提交 4515de8 在位（工作树已由本任务 Stage 0 重建删除，`git show 4515de8` 可查）**——引用其行号时一律经 git show 只读核对。
---

## 1. 概述

### 1.1 算子名称

`_ssd_chunk_scan_fwd_kernel`（TileOPs 项目 `SSDChunkScanFwdOp` 的 kernel 实现；custom_op 名 `npub::ssd_chunk_scan_fwd`）

### 1.2 功能描述

Mamba-2 SSD 分块扫描输出前向：融合「历史状态贡献（exp(dA)·C@prev_states）」与「分块内因果衰减贡献（cb·exp(ΔdA)·dt 因果卷积 x）」两条 GEMM 路径，单 kernel 产出 `[B, S, H, P]` fp32 输出。迁移后为 **NPU Expert 模式 MixCV 算子**（显式 `T.Scope("Cube")` 双 GEMM 消费 + `T.Scope("Vector")` 因子生产，GM workspace 多槽中继 + 深度 2 任务流水 + AIV 蛇形分片）。

### 1.3 数学公式

$$
\text{out}[b, cQ{+}l, h, p] = \underbrace{e^{\text{dA}[b,h,c,l]} \cdot \sum_{n} C[b, cQ{+}l, g(h), n] \cdot S_{\text{prev}}[b,c,h,p,n]}_{Y_{\text{off}}\ \text{(history path)}} + \underbrace{\sum_{s \le l} \text{cb}[b,c,g(h),l,s] \cdot e^{\text{dA}_l - \text{dA}_s} \cdot \text{dt}[b,h,c,s] \cdot x[b, cQ{+}s, h, p]}_{Y_{\text{diag}}\ \text{(intra-chunk causal path)}}
$$

（完整语义与 dtype/边界语义见 §0.1；`g(h) = h // (H/G)`。）

### 1.4 算法描述（迁移决策后的 NPU 侧 Expert 模式算法）

本节描述 §0.5/§0.6 决策与 §1.6.1 机器验证定型后的 NPU 算法。**注**：§1.6.1 的机器验证否决了「K 侧因子移动（衰减因子乘 x 侧）」候选（fp16 NaN / bf16 噪声放大 1e26 量级）与「对角块 L-side 因子化」（decay>88 角点 NaN）——衰减因子**保持在 cb/C 侧**进入 cast（值随衰减缩小 ⟹ cast 噪声随之缩小）；对角块采用直接差分形式（无条件数值安全）。与源算法的结构差异逐条标注来源。

**Kernel 骨架**：

```python
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

@tilelang.jit(out_idx=[-1], target="npuir", pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,   # PL-1.16 Expert 硬边界
})
def main(x, cb, dA_cumsum, C_mat, prev_states, dt, ws_c, ws_lcb, out):
    with T.Kernel(24, is_npu=True) as (kernel_id, subid):   # persistent（§5.5）
```

（`ws_c/ws_lcb` 为 per-core GM workspace 显式参数——workspace 需求不构成改 wrapper 接口的理由，PL-1.16 wrapper 契约兼容先例；host 侧 `prev_states.to(dtype)` 完成唯一 dtype 适配。）

**任务结构**：一维 persistent `T.Kernel(24)`（§0.6 R1），逻辑任务 = `(b, c, h)`（chunk-head 粒度；源为 (l×p tile, b·c, h) 三维 grid——**差异来源 R1**）。**深度 2 任务流水**（R7）：ws 双槽（slot = task_id % 2）、4 flag、Cube 前导 set。

**Cube 域（`T.Scope("Cube")`）**——纯 gemm 消费，任务循环 `T.serial(num_local_tasks)`：

```python
with T.Scope("Cube"):
    l1_c    = T.alloc_L1([bl, bn], dtype)     # c_scaled 操作数（bn=min(128,N)，§5.2）
    l1_state= T.alloc_L1([bp, bn], dtype)     # prev_states 直装载（R2）
    l1_x    = T.alloc_L1([Q, bp], dtype)      # x 缓存（R7.4 v5_xband；每 (task,pp) 装载一次）
    l1_band = T.alloc_L1([bl, Q], dtype)      # band 组装区（唯一 band buffer，v2：GM→L1 直载列偏移；
                                               #   每 lt 的因果带 [bl, l0+tl]；w2 最大 32KB）
    l0_acc  = T.alloc_L0C([bl, bp], "float32")
    # prologue：前导 set 两槽 cons flag（PL-1.12——Vector T=0/1 免等通过）
    with T.rs("PIPE_FIX"): T.sync_block_set(CONS0); T.sync_block_set(CONS1)
    for task_id in T.serial(num_local_tasks):
        cid = T.min(task_id * 24 + kernel_id, num_logical - 1)     # ghost 钳位
        bz, bc_idx, bh = decode(cid); bg = bh // HPG; cs = bc_idx * Q
        slot = task_id % 2
        with T.rs("PIPE_MTE2"): T.sync_block_wait(READY(slot))     # 因子就绪
        for pp in T.serial(P_tiles):
            p0 = pp * bp; tp = T.min(bp, P - p0)                   # 尾块（R6）
            tnp = T.max(16, T.min(bp, T.ceildiv(tp, 16) * 16))
            # x 任务×pp 级装载（pp 循环内、lt 循环外一次，消跨 lt 2.5× 重读；R7.4）
            T.copy(x[bz, cs:cs+Q, bh, p0:p0+tp], l1_x[0:Q, 0:tp])
            for lt in T.serial(L_tiles):
                l0 = lt * bl; tl = T.min(bl, Q - l0)
                tmc = T.max(16, T.min(bl, T.ceildiv(tl, 16) * 16))
                # ---- history（n-loop；bn≥N 时单块）----
                for n_blk in T.serial(N_tiles):
                    n0 = n_blk * bn; tn = T.min(bn, N - n0)
                    T.copy(ws_c[kernel_id, slot, pp, lt, 0:tl, n0:n0+tn], l1_c[0:tl, 0:tn])
                    T.copy(prev_states[bz, bc_idx, bh, p0:p0+tp, n0:n0+tn], l1_state[0:tp, 0:tn])
                    T.gemm(l1_c, l1_state, l0_acc, initC=(n_blk == 0),
                           b_transpose=True, size=[tmc, tn, tnp])
                # ---- intra：full-lower s-blocks（s_blk ∈ [0, lt)）+ 对角块（s_blk == lt）----
                # band 组装形态（R7.4，v2 修订·复检问题 1：单 buffer GM→L1 直载——
                # L1→L1 拷贝实测编译失败 'Unsupported copy from cbuf to cbuf'，探针 1；
                # 直载形态探针 2 编译通过 = HEAD 旧 verified 终版形态）：
                # ws_lcb 逐块 GM→L1 直载进 band 列偏移区（src/dst 双侧 ts 裁剪——
                # TRAP-L1-band-dst-tail-overrun：未裁剪的 s0+bs 越界 [bl,Q] 污染相邻 L1），
                # 每 lt 单条 band gemm（K = l0+tl 连续因果带；gemm 10→4/任务）
                for s_blk in T.serial(lt + 1):
                    s0 = s_blk * bs; ts = T.min(bs, Q - s0)
                    T.copy(ws_lcb[kernel_id, slot, pp, lt, s_blk, 0:bl, 0:ts],
                           l1_band[0:bl, s0:s0+ts])         # 单跳直载（列偏移 dst）
                T.gemm(l1_band, l1_x, l0_acc,
                       initC=False, size=[tmc, l0+tl, tnp])         # band gemm（K 连续因果带）
                # ---- 写回 ----
                T.copy(l0_acc[0:tl, 0:tp], out[bz, cs+l0:cs+l0+tl, bh, p0:p0+tp])
        with T.rs("PIPE_FIX"): T.sync_block_set(CONS(slot))        # 槽位已消费
```

**Vector 域（`T.Scope("Vector")`）**——因子生产，任务循环（slot-free wait 任务头）+ pp 重复生产 + **AIV 蛇形分片**（R8）：

```python
with T.Scope("Vector"):
    # 核级常量（R4，一次）：pen_const = (i<j ? −1e30 : 0)（arange 行列差 → vsub → vmin → vmul）
    pen_const = T.alloc_ub((bl, bs), "float32"); ...  # T.arange(pen_const,[1,0],0) 等
    dA_l_col/exp_dA_l_col (bl,1) f32; dA_s_row/dt_s_row (1,bs) f32; dt_16 (1,bs) dtype
    cb_16 (bl,bs) dtype; cb_f32/lcb_f32 (bl,bs) f32; lcb_16 (bl,bs) dtype
    c_16 (bl,N) dtype; c_f32 (bl,N) f32; c_scaled_16 (bl,N) dtype
    dA_l_mat/dA_s_mat/dt_mat (bl,bs) f32
    for task_id in T.serial(num_local_tasks):
        cid = ...; slot = task_id % 2
        # slot-free 握手（v1 修订，问题 1——wait 前置于任何 ws 写）：任务 0/1 消费
        # prologue 前导令牌立即通过；任务 W≥2 等待 Cube 消费完任务 W−2（同槽上一使用者）
        with T.rs("PIPE_MTE2"): T.sync_block_wait(CONS(slot))     # 等槽位空闲（深度 2 节拍）
        for pp in T.serial(P_tiles):          # v1 修订（问题 3 方案 a）：因子按 pp 重复生产
            # AIV 蛇形分片（R8/PL-1.13）：lt = i + subid + (i%2)·(L_tiles − 2i − 2·subid) + clamp
            for i in T.serial(lt_count):
                lt = snake(i, subid, L_tiles); l0 = lt * bl; tl = T.min(bl, Q - l0)
                # ---- history 因子（R5/OPT-A）：c_scaled = cast(C ⊙ e^{dA_l}) ----
                T.copy(dA_cumsum[bz, bh, bc_idx, l0:l0+tl], dA_l_col[0:tl, 0:1])
                T.vexp(dA_l_col, exp_dA_l_col)                          # [bl,1]
                T.copy(C_mat[bz, cs+l0:cs+l0+tl, bg, 0:N], c_16[0:tl, 0:N])
                T.vcast(c_16, c_f32, round_mode="rint")
                T.vmul(c_f32, exp_dA_l_col, c_f32)                       # 行广播 [M,N]⊙[M,1]
                T.vcast(c_f32, c_scaled_16, round_mode="rint")
                T.copy(c_scaled_16[0:tl, 0:N], ws_c[kernel_id, slot, pp, lt, 0:tl, 0:N])
                # ---- intra 因子：直接差分链（R3）----
                for s_blk in T.serial(lt + 1):
                    s0 = s_blk * bs; ts = T.min(bs, Q - s0)
                    T.copy(cb[bz, bc_idx, bg, l0:l0+tl, s0:s0+ts], cb_16, size=[tl, ts])
                    T.vcast(cb_16, cb_f32, round_mode="rint")
                    T.copy(dA_cumsum[bz, bh, bc_idx, s0:s0+ts], dA_s_row[0:1, 0:ts])
                    T.vbrc(dA_l_col, dA_l_mat)                           # [bl,1]→[bl,bs]
                    T.vbrc(dA_s_row, dA_s_mat)                           # [1,bs]→[bl,bs]
                    T.vsub(dA_l_mat, dA_s_mat, dA_l_mat)                 # diff = dA_l − dA_s
                    if s_blk == lt:                                      # 对角块（trace 静态比较）
                        T.vadd(dA_l_mat, pen_const, dA_l_mat)            # 惩罚融进指数（OPT-B）
                    T.vexp(dA_l_mat, dA_l_mat)
                    T.vmul(cb_f32, dA_l_mat, lcb_f32)
                    T.copy(dt[bz, bh, bc_idx, s0:s0+ts], dt_16[0:1, 0:ts])
                    T.vcast(dt_16, dt_s_row, round_mode="rint")
                    T.vbrc(dt_s_row, dt_mat)                             # [1,bs]→[bl,bs]
                    T.vmul(lcb_f32, dt_mat, lcb_f32)
                    T.vcast(lcb_f32, lcb_16, round_mode="rint")          # band 域此处改 vselect+列守卫（R4）
                    T.copy(lcb_16[0:tl, 0:bs], ws_lcb[kernel_id, slot, pp, lt, s_blk, 0:tl, 0:bs])
        with T.rs("PIPE_MTE3"): T.sync_block_set(READY(slot))        # 因子就绪
```

**执行序要点**：① 深度 2 流水下 Vector(T+1) 因子生产与 Cube(T) gemm 消费全重叠（R7.2）；② 对角块（s_blk == lt）恰为 s 序最后一块，惩罚掩码在 vexp 前融入；③ band 组装域的 gemm K 维为连续因果带（数学上 = 逐块 gemm 之和，块边界即 K 分段，累加序不变）；④ **Vector 任务头 `wait CONS(slot)` 前置于任何 ws[slot] 写**（v1 修订，问题 1——任务 W 等待 Cube 完成任务 W−2 同槽消费，深度 2 节拍下 Vector 最多领先 Cube 1 个任务）；⑤ Vector 域因子按 pp 重复生产（v1 修订，问题 3 方案 (a)——P=64 时单次，P=128 双次，对齐 HEAD 旧 verified 终版 L364）。

**与源算法的结构差异清单**：任务粒度与 persistent 分核（R1）、state 直装载 + host cast（R2）、Vector 域直接差分因子链（R3）、常量惩罚掩码（R4）、history 缩放折进 A 操作数（R5+OPT-A）、尾块 slice/size（R6）、GM ws 中继 + 深度 2 流水 + band 组装 + pass_configs（R7）、AIV 蛇形分片（R8）。**保持不变**（机器验证或安全分析裁定）：衰减因子乘 cb/C 侧后 cast（源数值路径）、对角块直接差分（源 fallback 路径）、上三角跳过（源 #10）、因果掩码 pad-0 语义（源边界语义）。

### 1.5 数据流图

```
【Vector 域（每 AIV，蛇形分片 lt；任务头 wait CONS(slot)——深度 2 节拍，任务 W 等任务 W−2 同槽消费完）】
GM[dA[b,h,c,l0..]] ─copy──▶ UB[dA_l_col (bl,1) f32] ─vexp─▶ UB[exp_dA_l_col (bl,1)]──┐(行因子)
GM[C[b,cs+l0..,g,0:N]] ─copy──▶ UB[c_16 (bl,N)] ─vcast(rint)─▶ UB[c_f32] ─vmul(行广播)┘│
                                                                    │vcast(rint)        │
                              GM ws ◀─copy── UB[c_scaled_16 (bl,N) dtype]（history 因子，R5）│
GM[cb[b,c,g,l0..,s0..]] ─copy──▶ UB[cb_16] ─vcast(rint)─▶ UB[cb_f32 (bl,bs) f32]        │
GM[dA[..,s0..]] ─copy─▶ UB[dA_s_row (1,bs)] ─vbrc─▶ UB[dA_s_mat]─┐                      │
UB[dA_l_col] ──vbrc──▶ UB[dA_l_mat] ──vsub(in-place)◀─────────────┘ diff = dA_l − dA_s  │
UB[pen_const（核级常量 0/−1e30）] ──vadd（仅对角块）──▶ diff ──vexp──▶ UB[decay]          │
GM[dt[b,h,c,s0..]] ─copy─▶ UB[dt_16] ─vcast(rint)─▶ UB[dt_s_row] ─vbrc─▶ UB[dt_mat]     │
UB[cb_f32] ─vmul(decay)─▶ UB[lcb_f32] ─vmul(dt_mat)─▶ UB[lcb_f32] ─vcast(rint)─▶ UB[lcb_16]
                              GM ws ◀─copy── UB[lcb_16]（intra 因子；块连续布局，PL-1.14；按 pp 重复生产）
                                    │ READY(slot) flag（PIPE_MTE3，任务尾）
                                    ▼（深度 2 流水：Vector(T+1) ∥ Cube(T)）
【Cube 域】
GM ws_c[slot] ─copy──▶ L1[l1_c (bl,bn) dtype] ────────┐
GM[prev[b,c,h,p..,n..]] ─copy(host 预 cast dtype)──▶ L1[l1_state (bp,bn)] ─┤T.gemm(b_transpose=True,
GM[x[b,cs:cs+Q,h,p0:p0+tp]] ─copy(每 (task,pp) 一次)──▶ L1[l1_x (Q,bp)] ────────┤  initC=(n_blk==0))
GM ws_lcb[slot] ─copy(块连续, 列偏移 dst, ts 裁剪)─▶ L1[l1_band (bl,Q) band 形] ──┤T.gemm(initC=False)
                                                       ▼
                              L0C[l0_acc (bl,bp) f32]（纯 Cube 驻留，零 rescale——R5）
                                                       │T.copy（L0C→GM 自动 Fixpipe）
                                                       ▼
                                GM[out (B,S,H,P) f32]（写回裁剪 [0:tl,0:tp]）
```

（Expert 双 Scope：Vector 侧全部 `T.alloc_ub` + v-prefix 链；Cube 侧 `T.alloc_L1/L0C` + `T.copy/T.gemm`；跨引擎仅经 GM ws + flag 握手。）
### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**——对同一数学语义的算法族回答调研四问（§1.6.0；迁移任务在源算子解读〔§0〕后、耦合性判定与重设计〔§0.5/§0.6〕前完成，调研结论已作为 M1 判定输入）；再在保证数学等价的前提下优化公式（§1.6.1，**含机器验证**）；再把保留下来的计算全部交给向量单元（§1.6.2）；最后回答"在哪个轴上向量化"（§1.6.3，阻塞级）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（§0.1 公式）的算法族。源算法（SSD 分块双路径）只是基线候选之一。调研深度：**完整调研**（矩阵 × 多步 × 融合类算子）。信息源：algorithm-candidates.md 参考表（ALG-attention/ALG-gemv/ALG-elementwise 命中行评估）、pattern-library（kb_search "ssd chunk scan mamba MixCV" 命中 8 条：CASE-ssd-chunkscan-migration / PL-1.18 / CG-2026-0011 / CG-2026-0010 / PL-1.16 / VP-2026-0083 / VP-2026-0002 / VP-2026-0088——已全部消费：CASE 为本算子族同任务先例档案、PL-1.16/1.18 为 Expert 结构与稳态地板、CG 为硬边界、VP 为 cast 侧选择与 AIV 分片教训）、本仓 examples/（GQA Expert MixCV 生产版、flash_attention、mixcv 最小样例）、源算子（§0.3/§0.4 为"该算子族存在什么算法"的直接证据）、外部已知算法（Mamba-2 SSD 论文族：Transformers are SSMs（Dao & Gu 2024）；mamba_ssm 官方 Triton `_chunk_scan_fwd`/`_chunk_cumsum` 知识——只取算法思路）、知识预注入 E-1 全部条目（消费方式见 §0.6 引言：结构性结论继承为主选、硬边界标注待重验）。**未使用互联网检索**（本地信息源已覆盖：源码 + 官方对齐注释 + 同族 examples + 同任务先例档案 + 模型知识一致收敛于分块双路径族；记录为"已评估无需联网补充"）。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：SSD 分块双路径**（源算法 = mamba_ssm 官方 `_chunk_scan_fwd` 对齐） | Y_off（rank-N off-diagonal 显式给态）+ Y_diag（causal 对角块） | — | — | ✅（基线） |
| 1 | 全序列 materialize（非分块） | `out = (L(cb ⊙ exp(ΔdA) ⊙ dt) · x) + off`，L 为 [S,S] 全 causal 矩阵 | 数学恒等（分块 = 对同一矩阵的分块求值） | 无收益：中间 [S,S] 矩阵 S=32k 时 4G 元素/chunk/head | ❌（访存爆炸，结构上不可行） |
| 2 | 逐 token 递推扫描（RNN 式在线） | `o_t = a_t·o_{t−1} + B_t·x_t`（a = exp(dA_t−dA_{t−1})） | 数学恒等（半可分矩阵的连乘形式） | O(S·H·P) FLOPs 低于分块 | ❌（见 R2：串行依赖不可并行；且本算子输入契约是 cb/prev_states 预处理量，递推形偏离输入契约） |
| 3 | semiseparable 分块分解的抽象重述 | off-diagonal 秩 ≤ N 分解 + 对角因果块 | 与基线同构（数学抽象层） | — | ❌（与 #0 重复） |
| 4 | history 行缩放折进 A 操作数（Expert 形态） | `e^{dA_l}·(C@S) = (C⊙e^{dA_l})@S`（行因子乘进 gemm A 操作数） | 数学恒等（分配律）；cast 位置移动需容差评估 | 消 L0C 上的 Vector rescale（CG-2026-0011 下唯一可行方向）；每 l-tile 省 1 次 [bl,bp] GM 中继往返 | ✅（→ §1.6.1 OPT-A） |
| 5 | exp_s_all 统一预计算（per-l-tile [Q] 向量替代 per-s-block [bs]） | 同式 `exp(anchor−dA_s)·dt_s` 求值位置上移 + 切片复用 | 严格等价（同式同 dtype 路径） | 省 s-block 级重复计算；**但依赖 anchor 因子化形态**（OPT-Y 同族数值边界）且 task 级行驻留被 CG-2026-0012 阻塞 | ❌（Expert 形态下不采纳：① anchor 因子化对角域数值边界否决（§1.6.1 OPT-Y）；② task 级行 buffer 驻留被 CG-2026-0012 阻塞（动态偏移 subview 非支配 IR）；③ 双域流水下 AIV 非关键路径（PL-1.18 实证 vec 62% 非关键），发射削减无墙钟收益——v3_hoist 家族双重 blocked 实证） |
| 6 | K 侧因子吸收（cb 侧 → x 侧） | `Σ_s cb·r[l]·c[s]·x = r[l]·Σ_s cb·(c[s]·x)` | 数学恒等；**舍入位置移动需容差评估** | cb_tile 免 fp32 中转直读 gemm；dt 全局一次吸收 | ✅（→ §1.6.1 评估，**机器验证否决**） |
| 7 | 掩码向量化（算术惩罚 / vselect 值替换） | `exp(指数+pen)`（pen 为 0/−PEN 常量矩阵）/ `s≤l ? v : 0` 的向量选择 | 双形均精确等价（惩罚形下溢精确 0；vselect 形选路替换） | 替代逐元素标量分支；惩罚形规避 vcmp 标量化（PL-1.11 verified 4.4–14.2x） | ✅（→ §1.6.1 OPT-B，惩罚主选 + band 域 vselect+列守卫分域） |
| 8 | exp2 域换算（exp → exp2·log2e） | `e^x = 2^{x·log2 e}` | 数学恒等；vexp2 vs vexp 吞吐差未实测 | 潜在发射/吞吐收益 | ❌（收益方向不成立：docs 数学操作目录确认 `T.vexp`（fp32 ✓）与 `T.vexp2` 并存，但无实测吞吐差证据；双域流水下 Vector 因子链非关键路径（PL-1.18），exp 发射数已非瓶颈——列为 Stage 4 可选微优化，不作设计期采纳） |
| 9 | 对角块 L-side 因子化（exp_l_col × exp_as_col 替代直接差分） | `exp(dA_l−dA_s) = exp(dA_l−anchor)·exp(anchor−dA_s)`（anchor = dA[l0]） | 数学恒等 | 对角块省 vbrc×2+vsub（diff 链） | ❌（**数值边界淘汰 + 机器验证否决**（§1.6.1 OPT-Y）：对角块内 s ≥ l0 ⟹ anchor−dA_s ≥ 0 ⟹ 列因子 exp_as ≥ 1，块内衰减 > ~88 时 fp32 溢出 inf 并污染下三角；verify_equiv.py block_decay_gt88 案 NaN 实证。直接差分形式（主选）下三角指数恒 ≤ 0、上三角正指数被掩码丢弃——无条件安全且与源 fallback 路径一致） |
| 10 | H 维 fold 任务（同 group 的 head 共享 cb/C 的核内复用） | 任务 = (b,c,g,h_tile)，cb_tile/C_tile 在 L1 复用 heads_per_group 次 | 数学恒等（G<H 时 cb/C 被同组 head 复读） | cb/C 复读流量 ÷ heads_per_group | ❌ 首版（**Stage 4 候选**：实测 GQA 类读放大由 L2 吸收——cube read_hit 91% / AIV 99%（PL-1.18）；当前 workload G=1（H/G=48~80），fold 使任务数 ÷48~80 导致 24 核严重欠载（w2: 768→16 任务），需再切 h_tile——任务结构复杂化，L2 已吸收的边际收益存疑；设计期估算按 L2 命中折算，Stage 4 实测后重评） |

**R2 在线算法**：**结论：无适用于本算子契约的在线变体**（结构依据如下，非无据断言）：
1. **本算子的输入已是"在线化的中间产物"**：Mamba-2 SSD 将序列按 chunk 切分后，跨块状态传递（`prev_states` 的链式更新 `S_c = A_c·S_{c−1} + B_c·x_c`）由上游算子 **SSDStatePassing/SSDChunkState** 完成（NPU manifest 注释明确这些算子未迁移）；`cb = C@B` 耦合与 `dA_cumsum` 亦为上游预处理。本算子是**块内双 GEMM 融合输出算子**，输入契约（6 张量）不含跨块 running 状态更新需求——在线性已在流水线上游完成，算子内无 running 统计量可提取（输出对每个 (l,p) 是**有限窗口因果和**，窗口 = 所在 chunk，非前缀依赖）。
2. 逐 token 递推（R1 #2）是数学等价的"流式"形式，但 S 长串行依赖链不可并行（每 token 依赖前一 token 的完整 [H,P] 状态），且需要 `A_t/B_t` 原始序列而非本算子的 `cb/prev_states` 预处理输入——**偏离输入契约**，属于算法族替换而非本算子的变体。
3. chunk 级 prefix-scan（log-depth 扫描传递状态）会改变上游算子分工（state passing 与 scan 融合），超出本算子迁移范围（迁移 prompt 明确接口不可变）。
> 收益口径核对：本算子的"扫描遍数"= 每 chunk 单遍（x 每 (b,c,h) 任务内 1× L1 驻留复用、cb/C 单遍逻辑读、中间缓冲 O(tile) + ws 中继 O(task)）——分块结构已给出在线级的遍数与缓冲量纲，无进一步单遍化空间。

**R3 复杂度对比**（四口径；以 workload w2 `mamba2-780m-b1-s4k` 标定：B=1, S=4096, H=48, P=64, C=16, Q=256, G=1, N=128，tokens = B·S·H = 196,608；口径注明含/不含）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| **基线：分块双路径**（causal 跳过后） | history `2·tokens·N·P` = 2·196,608·128·64 = **3.22G** + intra `2·(Q(Q+1)/2)·P·C·H` = 2·32,896·64·16·48 = **3.23G** ≈ **6.46G**（含全部 MAC=2；causal 面积取 Q(Q+1)/2 下三角；不含向量因子链 ~0.15G elem-op；与 manifest mamba.yaml roofline 公式代入 w2 = 6.44G 交叉一致，差值为 Q+1 vs Q 的整化） | 逻辑读：x 25.2M + cb 2.1M + C 1.05M + prev 25.2M + dA 0.79M + dt 0.39M + 写 out 50.3M ≈ **105MB**（HBM 口径；cb/C 的 G<H 复读依赖 L2 命中缓解，L2 命中后 HBM ~53MB）**+ ws 中继流量**（Expert 结构特有，**per-task 口径**〔v1 修订，问题 4〕）：ws_c 单程 64KB（= P_tiles·L_tiles·bl·N·2B = 1×4×64×128×2B@w2）+ ws_lcb 单程 80KB（= P_tiles·Σ_{lt}(lt+1)·bl·bs·2B = 10×64×64×2B），写（Vector MTE3）+读（Cube MTE2）≈ 2×144KB×768 ≈ **+216MB**（v0 误按 per-l-tile/per-AIV 口径估 87MB，低估 ~2.5×；ws 的 L2 命中率未实测，L2 命中 0–50% 折算后 HBM 增量 ~108–216MB） | x 任务内 L1 驻留 1×；cb/C/prev/dA/dt 1×；ws 因子 1×（生产→消费）；out 1× | tile 级（[64,64]×~6 UB buffer + [Q,bp] L1 + ws 任务槽 ~9.2MB/24 核 in-flight 2 任务） | 任务 (b,c,h) = 768，无跨核归约（GEMM 累加核内完成）；跨核零同步（flag 为引擎内握手，非核间） |
| 候选 1：全 materialize | intra ≈ 19.3G（无 causal 跳过）+ 中间 [S,S,H] 写读 ≈ **5.4TB** 访存 | 爆炸（S=4k 时 [4096,4096,48] fp32 中间 ~3.2TB） | 多遍（写+读中间矩阵） | [S,S] 级（GM 都放不下） | 高并行但访存主导不可行 |
| 候选 2：逐 token 递推 | 2·S·H·P·B ≈ 0.1G（最低） | 输入重释（契约偏离）~ O(输入) ≈ 80MB | 单遍 | [H,P] running 状态（O(tile)） | **串行 S=4096 步依赖链**——并行度 1，时延不可行 |
| 基线 + §1.6.1 采纳优化（选定方案） | 同基线 6.46G（MAC 不变）；向量 op 数 ~40/任务·AIV（蛇形分片后口径，§1.6.2 清单） | 同基线 + ws（中继结构不变；OPT-A 折叠消 rescale 往返 ~32KB/l-tile） | 同基线 | 同基线 | 同基线 |

**R4 硬件亲和性评估**（逐候选对照检查清单）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（分块双路径，Expert 双 Scope 落地） | 双 GEMM → Cube（fp16/bf16 mma）；exp/掩码/缩放因子链 → Vector ✓ MixCV 天然分工 | [64,64] tile：L0C 16KB ≤ 128KB ✓；L1（l1_c 16K + l1_state 16K + l1_x 32K + l1_band 32K）**96KB** ≤ 512KB ✓；UB 向量缓冲 177.1KB ≤ 192KB（膨胀评估见 §4.5——Expert 显式 alloc 结构实测 ≈ 手工核算 ±2%） | bl/bp/bs = 64、bn=min(128,N)：分形 M/N≥16、K≥32 ✓（bn=128 时 N=128 单块 K=128 ✓；N=32 → bn=32 贴 K 下限 ✓）；尾轴 64×2B = 128B ≥ 32B ✓ | 循环边界 Q/N/P 派生 PrimExpr（静态 shape，工厂特化）；s_blk 上界 lt+1 为循环携带 PrimExpr（进循环前计算，静态可 lower）✓ | 深度 2 任务流水（双域重叠，PL-1.12）；块内装载-gemm 交替（T.rs 管线区域静态排序）；CV 经 ws 中继 + flag 握手 | ✅（选定） |
| 全 materialize | GEMM 匹配 | 中间 [S,S] 超 GM 容量 | ✓ | ✓ | — | ❌（容量淘汰，结构不可行） |
| 逐 token 递推 | 逐元素 → Vector | ✓ | ✓ | ✗（循环携带依赖阻断流水） | 串行链不可并行 | ❌（并行度淘汰） |
| 优化后基线（本设计） | 同基线 + OPT-A 消 rescale 中继、OPT-B 掩码向量化（PL-1.11 verified） | 同基线 | 同基线 | 同基线 | 同基线 | ✅（主选） |

**调研结论**：**选定算法族 = SSD 分块双路径（基线族，保留源算法结构）**，关键依据：R3 表中基线四口径全面占优（6.46G FLOPs / 105MB+216MB ws 逻辑访存〔per-task 口径，v1 修订〕 / tile 级缓冲 / 768 任务无跨核同步），候选 1 因 [S,S] 中间容量爆炸淘汰、候选 2 因 S 长串行依赖链淘汰（均为结构性依据，非容差权衡）；R4 确认基线结构与 NPU Cube+Vector 双引擎高度匹配（Expert MixCV 有 GQA 生产先例 + 本算子族旧任务先例档案）。与基线的结构差异：**族不变，块内实现重构**——采纳 R1 #4/#7 两项等价变形（§1.6.1 完成四要素论证与机器验证），#5（exp_s_all 预计算）因 Expert 形态三重理由不采纳、#6（K 侧因子）经机器验证**否决**、#9（对角因子化）经机器验证**否决**、#10（H fold）列 Stage 4；源实现的标量因子计算/逐元素分支重构为 NPU Vector 域向量链 + ws 中继。调研范围（"无更优算法族替代"的依据）：algorithm-candidates.md 全表（ALG-attention/ALG-gemv/ALG-elementwise/ALG-transpose 等命中行已评估）、kb_search "ssd chunk scan mamba MixCV"（8 条命中全消费）、examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py（GQA Expert MixCV）+ examples/flash_attention/flash_attn_npuir_dev.py + examples/mixcv/mixcv_mixkernel.py、源算子注释（官方对齐声明）、外部模型知识（mamba_ssm 官方实现族：`_chunk_scan_fwd` 同构双路径、`_chunk_cumsum` 上游分工）。**源算法优化手段的意图承接**：§0.4 的 14 项在 §0.5 逐项标注（#9 anchor 因子化与 #10 上三角跳过直接保留进选定方案〔#9 限 full-lower 域、Expert 直接差分下由双域重叠承接意图〕；#7/#8 由 R3 因子链 + L2 吸收承接；#11 由 R4+OPT-B 算术惩罚掩码承接；#1–#6/#12 由等价替换承接；#3/#13 舍弃附理由）。

**设计期 roofline（D-2，估算下界行）**（标定 w2；常数引自 pattern-library/constants.md 并标注版本戳；**stale 提示**：kb_stale_check 当前戳 `tilelang 0.1.2+4515de8`，下列常数条目戳为 1990aa9fe4 及更早（版本戳差异标 stale）——按"设计期估算口径"使用，Stage 3/4 须在当前工具链重验；其中同任务旧档 final 实测值作为锚点回填）：
- **容量项**：L0C 16KB/核（[64,64] fp32 acc）≤ 128KB（CONST-capacity-910B2C）——非地板；L1 **96KB**（l1_c 16K + l1_state 16K + l1_x 32K + l1_band 32K，§4.3）≤ 512KB——非地板；UB **177.1KB** ≤ 192KB（贴限 92.3%，Expert 显式 alloc 结构实测 ≈ 手工核算 ±2%，CONST-capacity-910B2C ssd 第三数据点）——**容量约束锁定 bl=bs=bp=64**（bl=128 实测需 357–582KB，容量否决）。
- **流量项**：HBM 逻辑 105MB ÷ 混合流量地板 ~1.26TB/s（CONST-mte2-degradation 256M 档）≈ **83µs**；+ ws 中继 216MB（per-task 口径，v1 修订；L2 命中 0–50% 折算 ~108–216MB）——含 ws 的 HBM 口径 ~161–321MB ÷ 1.26TB/s ≈ **128–255µs**（估算区间；ws 的 L2 命中率未实测——未实测假设：按 0–50% 命中估）。**仍低于段数项、下界结论不变**（见下）。
- **发射项**：向量链 ~40 op/任务·AIV（蛇形分片后口径，§1.6.2 清单）× 768 任务 ÷ 48 AIV × ~0.5µs/op（CONST-vector-launch-overhead 串行发射口径）≈ **~320µs 串行发射上界**；双域流水重叠后的有效值实测 ~110µs 量级（旧档 final 画像：AIV vec 43.5% × ~218µs 壁钟）——**发射项在深度 2 流水下非关键路径**（PL-1.18 实证）。
- **Cube 项**：6.46G ÷ 24 核 ÷ ~13 TFLOPS/核（910B2C fp16 Cube 量级；未实测假设：按公开规格 ~313T/24 估）≈ **20.7µs/核**——非地板。
- **段数项（Expert ws 中继结构特有地板，CONST-mte2-degradation 指令维度口径）**：Cube mte2 ≈ 1216 段/任务（ws_lcb band 重组 640 + x 256 + ws_c 128 + prev 128）× ~4ns/128B 段 × 768 任务 ÷ 24 核 ≈ **~156µs@w2**——**旧档实测锚点**：final Cube mte2 ~156µs（16 nd2nz 指令/任务 × 32 任务/核 × ~256ns）+ scalar ~120µs，为第一主项；w4 稳态画像 mte2 83.5% 忙比（段数墙，PL-1.18 五方向实测穷尽否决削减可能）。
- **估算下界 = max(流量 128–255µs〔v1 修正口径〕, 段数+scalar ~170–280µs, 发射(重叠后) ~110µs, Cube 20.7µs)** ≈ **~170–280µs 区间**（流量项修正后仍低于段数项，下界结论不变）；**旧档 final 实测 217.65µs@w2（几何 2.913× vs baseline 608µs）落在区间内**——结构地板由 Cube mte2 段数墙主导，Stage 4 目标 = 逼近段数墙下界（已实测穷尽五个削减方向，见 §9）；新工具链下以 Stage 4 首轮 profile 复核。结论供 §5 tiling 与 Stage 4 调优参照。
#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（分块双路径）经 §0.6 重设计落地的 NPU Expert 形态。逐项四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）；**机器验证（D-1）**：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/verify_equiv.py`（fp64 oracle + 目标 dtype 舍入路径对照，随机张量 + 角点：深衰减 dA→−128 量级 / dt 上界 / 全零输入 / 块内衰减 >88 / stale 列 NaN，fp16 与 bf16 双 dtype；2026-09-20 执行，torch CPU），结果表内嵌于下。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 (OPT-A) | **history 行缩放折进 A 操作数（Expert 形态）** | `acc += hist_gemm ⊙ exp(dA_l)`（gemm 后行广播缩放；源 fragment 域形态） | `c_scaled = cast( fp32(C) ⊙ e^{dA_l} → dtype )`；`acc = c_scaled @ state_t`（gemm，initC=True 覆写初始化）——行因子乘进 A 操作数 | 数学恒等（分配律：`e^{dA_l}·Σ_n C·S = Σ_n (C·e^{dA_l})·S`）。cast 位置从 C 移到 C⊙e^{dA_l}：被 cast 量随衰减缩小（dA_l ≤ 0 ⟹ e^{dA_l} ≤ 1）⟹ cast 绝对噪声随之缩小，与源码 cb 侧因子化同向（VP-2026-0083 教训的正向侧）；深衰减下 e^{dA_l} 下溢 0 时 c_scaled 全行 0、贡献 0——与源 exp_dA_l 直接下溢行为一致。**机器验证**：FULL 行（OPT-A 含在内）fp16/bf16 × 4 角点 ratio 0.63–1.07，容差内等价（见下表） | 每 l-tile 省 1 次 [bl,bp] L0C rescale 中继（GM 往返 ~32KB + 潜在第三握手相位 ~2 flag）——CG-2026-0011（L0C 无 rescale 原语）下的唯一可行行缩放路径；acc 全程纯 Cube 域驻留 |
| 2 (OPT-B) | **对角块因果掩码向量化（算术惩罚主选）** | 逐元素 `if_then_else(s ≤ l, cast(value), 0)` 标量分支（源 fallback 路径） | **主选（整除域）：算术惩罚掩码**——核级常量 `pen`（arange 行列差派生 0/−PEN 矩阵，PEN=1e30 有限哨兵）`vadd` 融进 exp 指数，s>l 位 exp 下溢为**精确 +0.0**、s≤l 位惩罚恒 0；**band-carrying 域（R6 尾块）：vselect + 列 OOB 守卫双保险**（`(j ≤ i) & (j < ts)` 值替换——NaN 免疫） | **精确等价（双域）**：惩罚形下三角逐位原值、上三角 fp32 下溢精确 +0.0（机器验证 penalty-matrix exactness 行：双 dtype 精确）；vselect 形选路无算术副作用。**band 域双保险的机器证明（本设计新证据，verify_equiv.py stale-NaN 案）**：stale 列 NaN 可跨越下三角（i ≥ j ≥ ts 位）——① 加法掩码 NaN+x=NaN 污染下三角（不可用）；② 单一因果 vselect 也不足（stale 列与下三角交集 NaN 被选路保留）；③ `(j≤i)&(j<ts)` 组合形态干净。**主选依据（PL-1.11 verified，2026-09-16 a13585dc 重验维持）**：int16 vcmp 全形态标量化（aiv_scalar 94–97%、对角块聚合 ≈ 壁钟 45%），算术惩罚掩码实测 **4.4–14.2x（几何 8.53x）** | 消对角块逐元素标量分支；惩罚矩阵从每任务一次降为每核一次（w2: 768 任务 → 24 核，÷32）；掩码 op 链 = vadd（1 op/对角块）+ 核级常量，规避 vcmp 标量化税 |
| 3（否决） | ~~K 侧因子吸收（衰减因子乘 x 侧 + dt 全局吸收）~~ | `lcb = cast(cb·exp_l·exp_s)`，`gemm(lcb, x)` | ~~`x_scaled = cast(exp_s·dt·x)`，`gemm(cb_dtype, x_scaled)`~~ | ~~数学恒等（乘法交换律）~~；**机器验证否决（本设计回归 + 旧档双证）**：cast 位置移到 x 侧后，被舍入操作数（x·exp_s·dt）不随衰减缩小，且 exp_s 在 s < anchor 段为 exp(正大数) → inf 路径——本设计实测 fp16 random 案直接 **NaN**、bf16 random 案绝对误差 **3.4e+22**（ratio 1.4e+26）；旧档同族实测噪声放大 38–135× 超 atol。**违反容差内等价，禁止采纳** | （不采纳）源码将衰减因子乘在 cb/C 侧再 cast 是**有数值必要性的结构**：衰减因子使被 cast 量缩小 ⟹ cast 绝对噪声随之缩小 |
| 4（否决） | ~~对角块 L-side 因子化（exp_l_col × exp_as_col）~~ | 对角块逐元素 `exp(dA_l−dA_s)` | ~~`exp_l[i]·exp_as[j]`（anchor = dA[l0] 的行列因子积）~~ | 数学恒等；**数值边界否决 + 机器验证否决**：对角块内 s ≥ l0 ⟹ anchor−dA_s ≥ 0 ⟹ 列因子 exp_as ≥ 1，块内衰减 > ~88（fp32 exp 上限）时溢出 inf，且 `inf·exp_l` 污染下三角。**本设计机器验证（block_decay_gt88 案）**：OPT-Y 形 fp16/bf16 双 dtype 输出 **NaN**（exp_as 溢出路径实证）；同角点下主选 FULL 形（直接差分 + 惩罚）ratio 1.00–1.01 PASS——**直接差分无条件安全**（下三角指数恒 ≤ 0、上三角被 −PEN 吸收） | （不采纳；连同「块内衰减 < 88」护栏记入 Stage 4 备选——若实测对角块为热点且分布受控可重评） |

**等价性机器验证结果表**（`verify_equiv.py`，2026-09-20 执行，torch CPU，fp64 oracle；Q=128 双 l-tile 覆盖 full-lower + diag 两路径；BL=BS=64；判定口径：噪声比 ratio = max\|cand−oracle\| / max\|base−oracle\| ≤ 3 且绝对误差 ≤ 1e-2；base = §0.1 源数值路径逐点复刻，cand = §1.4 Expert 定型算法数值路径）：

| 验证项 | dtype | case | max\|cand−oracle\| | max\|base−oracle\| | ratio | 结论 |
|--------|-------|------|-------------------|-------------------|-------|------|
| FULL（OPT-A+OPT-B，§1.4 定型算法） | fp16 | random | 4.557e-05 | 4.246e-05 | 1.073 | **EQUIV_PASS** |
| FULL | fp16 | deep_decay | 4.749e-05 | 6.541e-05 | 0.726 | **EQUIV_PASS** |
| FULL | fp16 | dt_upper_bound | 2.708e-05 | 4.314e-05 | 0.628 | **EQUIV_PASS** |
| FULL | fp16 | zero_input | 0.000e+00 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| FULL | bf16 | random | 2.493e-04 | 2.575e-04 | 0.968 | **EQUIV_PASS** |
| FULL | bf16 | deep_decay | 1.936e-04 | 2.356e-04 | 0.822 | **EQUIV_PASS** |
| FULL | bf16 | dt_upper_bound | 4.699e-04 | 5.918e-04 | 0.794 | **EQUIV_PASS** |
| FULL | bf16 | zero_input | 0.000e+00 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| OPT-B2 vselect 形（整除域数值等价） | fp16/bf16 | 4 角点 ×2 | 同 FULL | 同 FULL | 0.63–1.07 | **EQUIV_PASS**（8/8） |
| FULL（decay>88 角点） | fp16 | block_decay_gt88 | 6.234e-05 | 6.156e-05 | 1.013 | **EQUIV_PASS**（直接差分无条件安全实证） |
| FULL（decay>88 角点） | bf16 | block_decay_gt88 | 6.108e-04 | 6.108e-04 | 1.000 | **EQUIV_PASS** |
| OPT-B 惩罚矩阵精确性（下三角原值 / 上三角精确 +0） | fp16/bf16 | matrix | 0（精确） | — | 1.000 | **EQUIV_PASS**（2/2） |
| OPT-B2 band 域完整形态（vselect+列守卫，stale-NaN 注入） | fp32 | stale-NaN | 0（干净） | — | 1.000 | **EQUIV_PASS**；机器证明：加法掩码污染下三角（True）、单一因果 vselect 泄漏 stale-NaN（True，列守卫必须）、组合形态干净（True） |
| ~~OPT-X K 侧因子（否决项回归记录）~~ | fp16 | random | **NaN** | 2.836e-05 | — | **EQUIV_FAIL（已否决，不采纳；维持否决）** |
| ~~OPT-X~~ | fp16 | dt_upper_bound | **NaN** | 3.633e-05 | — | **EQUIV_FAIL（已否决）** |
| ~~OPT-X~~ | bf16 | random | 3.419e+22 | 2.390e-04 | 1.4e+26 | **EQUIV_FAIL（已否决）** |
| ~~OPT-X~~ | bf16 | dt_upper_bound | 2.876e+24 | 1.904e-04 | 1.5e+26 | **EQUIV_FAIL（已否决）** |
| ~~OPT-Y 对角 L-side 因子化（否决项回归记录）~~ | fp16 | block_decay_gt88 | **NaN** | 6.156e-05 | — | **EQUIV_FAIL（已否决，不采纳；维持否决）** |
| ~~OPT-Y~~ | bf16 | block_decay_gt88 | **NaN** | 6.108e-04 | — | **EQUIV_FAIL（已否决）** |

**OVERALL: EQUIV_PASS**（27 cases；采纳项失败 0）。**优化结论**：采纳 **2 项**（OPT-A history 缩放折进 A 操作数、OPT-B 对角算术惩罚掩码〔band 域 vselect+列守卫分域〕），全部机器验证 EQUIV_PASS（组合噪声比 0.63–1.07，fp16/bf16 双 dtype × 4 角点 + decay>88 专项 + 掩码矩阵精确性 + stale-NaN 专项）；否决 2 项（OPT-X K 侧因子移动——NaN/1e26 量级噪声；OPT-Y 对角 L-side 因子化——decay>88 NaN 污染）。**优化后公式（§3.1 拆解的唯一输入）**：

$$
\text{out}[l_0{+}i, p] = \underbrace{\textstyle\sum_n \text{cast}\big(C_i[n] \cdot e^{\text{dA}_i}\big) \cdot S_p[n]}_{\text{history gemm (OPT-A)}} + \underbrace{\textstyle\sum_{s_0}\textstyle\sum_j \text{cast}\big(\text{cb}[i,j] \cdot e^{\text{dA}_i - \text{dA}_j + \text{pen}(i,j)} \cdot \text{dt}[j]\big) \cdot x[s_0{+}j, p]}_{\text{intra gemm chain (OPT-B)}}
$$

（第一项 underbrace 为 history gemm 路径——缩放折进 A 操作数，OPT-A；第二项 underbrace 为 intra gemm 链——直接差分 + 惩罚，OPT-B。）

其中 `pen(i,j) = (i < j ? −PEN : 0)`（核级常量，PEN=1e30——整除域主选；band 域等价换为 `(j ≤ i) & (j < ts)` 的 vselect 值替换 + 列守卫）。无优化空间残余说明：gemm 的 MAC 计算量已由算法族选型（causal 跳过）与源一致；向量链 op 数经双域流水重叠后非关键路径（PL-1.18），进一步压缩（exp_s_all 预计算、vsub 广播、微块化）经三重理由/实测税否决（§1.6.0 R1 #5 / PL-1.15 / §0.5 行 11）；结构层压缩（band 段数、深度 3、L1 双缓冲等）经旧档 Stage 4 五方向实测穷尽否决（PL-1.18——段数是中继结构的物理流量）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点 §1.4 实现方案中全部循环与标量计算点（§0.3 步骤表 S1–S11 的 Expert 形态）；替代方案 API 均有 `docs/` 与 `examples/` 目录佐证。**Expert 形态约束**：kernel 内无 fragment 抽象（`T.alloc_fragment`/`T.Pipelined`/`T.Parallel` 不可用，PL-1.16 硬边界）——逐元素并行全部以 UB buffer + v-prefix 整块 op 表达。

| # | 计算点 | 原实现形态（源/GPU 式） | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|------------------------|-------------|---------|------------------------------|
| 1 | dA/dt 因子源装载（S4） | `T.Parallel(Q)` 逐元素写 SMEM | `T.copy`（GM→UB [bl,1]/[1,bs] 小片，per-l-tile/per-s-block；dtype 一致直拷） | ✅ | —（task 级 [1,Q] 行驻留形态被 CG-2026-0012 阻塞，退回小片直载——§0.5 已论证，复读由 L2 吸收） |
| 2 | exp_dA_l 计算（S5） | `T.Parallel(bl)` 逐元素 `T.exp` | `T.vexp`（fp32 ✓，docs 数学操作/T.vexp.md）on [bl,1] UB buffer | ✅ | — |
| 3 | c_scaled 因子链（S1+S6 合并，OPT-A） | `T.Parallel` 逐元素 + guard | `T.copy`（GM→UB）→ `T.vcast(c_16→c_f32, rint)` → `T.vmul(c_f32, exp_dA_l_col)`（**行广播 [M,N]⊙[M,1]，vmul 文档明示形态**）→ `T.vcast(→dtype, rint)` → `T.copy`（UB→GM ws） | ✅ | — |
| 4 | intra diff 链（S7+S9） | `T.Parallel(bl,bs)` 逐元素三乘 + exp + cast | `T.vbrc`×2（[bl,1]/[1,bs]→[bl,bs] 展开）→ `T.vsub`（in-place diff）→ 对角块 `T.vadd(pen)` → `T.vexp` → `T.vmul(cb_f32, ·)` → `T.vmul(·, dt_mat)` → `T.vcast(→dtype, rint)` —— 全 v-prefix 链 | ✅ | — |
| 5 | cb/x/C/prev 装载（S1/S2/S8） | `T.Parallel` 逐元素 + `if_then_else` guard + safe 索引 | `T.copy` slice 形态（尾块 `T.min` 裁剪；docs T.copy.md §2.2.2 尾块规则；GQA 生产先例；**Expert 约束**：dtype 一致直拷——prev 的 cast 在 host 侧） | ✅ | — |
| 6 | 对角块因果掩码（S9'） | 逐元素 `if_then_else(valid)` 标量分支 | 整除域：核级 `T.arange`/`vsub`/`vmin`/`vmul` 惩罚常量 + 任务内 `vadd` 融进指数（OPT-B 主选，PL-1.11 verified）；band 域：`T.vcmp`/`T.vselect` + 列守卫（docs 比较操作/T.vcmp.md §2.3 条 4 片上使用 + 条件操作/T.vselect.md） | ✅ | — |
| 7 | gemm 累加（S3/S10） | Tensor Core mma | `T.gemm`（Expert 下直接调用，GQA/ssd 生产实证；docs 线性代数操作/T.gemm.md 另有 `T.npuir_dot` Expert 别名） | ✅ | — |
| 8 | out 写回（S11） | `T.Parallel` + guard 写 | `T.copy(l0_acc[0:tl, 0:tp], out[...])` slice 形态（L0C→GM 自动 Fixpipe，docs T.copy.md 搬运方向表） | ✅ | — |
| 9 | 任务解码 `cid → (b,c,h)`、`l0/p0/s0` 块索引、snake 分片仿射式 | 标量整除/取余 | 无（每任务 O(1) 次整数标量，非逐元素热点） | ❌ | **block 级索引/任务映射计算**：每任务 ~10 次整数运算，执行一次，无逐元素等价 API（T.ceildiv/div 域为标量 PrimExpr）；GQA 同款（L285–291） |
| 10 | persistent 任务循环 `for task_id in T.serial(num_local_tasks)`（双域各一条） | 块级串行 | 无 | ❌ | **tile 级顺序依赖/任务映射结构**：深度 2 流水的槽位管理（slot = task_id % 2）与 flag 交替要求任务序严格递增（core-split-strategy.md 要素③ 极大规模方案）；`T.serial` 是分核与流水标准结构，非逐元素计算 |
| 11 | 尾块尺寸 `T.min(bl, Q−l0)`、分形钳位 `tmc/tnp`、slot 号等边界标量 | 标量 min/max | 无 | ❌ | **依赖动态边界的块级元数据**：每 tile O(1) 次，进 slice extents/gemm size（T.copy 尾块借用规则与 T.gemm size-form 的输入）；非逐元素热点 |
| 12 | host 侧 shape 校验/工厂参数/ws 分配/prev cast | Python 标量 | 无 | ❌ | **host 元数据计算**（不在 kernel 内；ascend-constraints.md §4 允许的视图/元数据操作——prev 的 `.to(dtype)` 作用于副本、不改输入张量真实内容） |

**向量化结论**：逐元素计算点（#1–#8）**全部向量化**（替代 API 均有 docs/Tilelang.language/ 条目与 examples 生产佐证：T.copy/T.vexp/T.vmul/T.vcast/T.vsub/T.vadd/T.vmin/T.vbrc/T.arange/T.vcmp/T.vselect/T.gemm）；**保留 4 类**标量/循环（表 #9–#12），理由类别：block 索引/任务映射（#9）、分核与流水结构串行（#10）、动态边界元数据（#11，含 gemm 尾块 size 标量）、host 元数据（#12）——均非逐元素热点，与 §6 循环结构一致（§6 无任何逐元素标量循环）。**向量链 op 数清单（per 任务·per AIV，蛇形分片后 w2 口径）**：任务头解码 ~4 标量；per-l-tile：c 链 5 op（copy×2+vcast×2+vmul）+ exp 1 + ws 写 1 ≈ 7；per-s-block：diff 链 10 op（copy×3+vbrc×3+vsub+〔diag: vadd〕+vexp+vmul×2+〔vselect band〕+vcast+ws 写）；w2 每 (b,c,h) 任务 Σ(lt+1) = 10 s-block ÷ 2 AIV = 5 s-block/AIV + 4 l-tile/2 ≈ 每 AIV 每 (b,c,h) 任务 ~40 向量 op + ~10 copy——与 §1.6.0 发射项口径一致。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（x/cb/C/dA/prev/dt/out 的 GM 布局固定，见 §0.2），但**核内布局、ws 中继布局与 AIV-lane 映射是自由变量**。本算子为 Cube/Vector 混合类（§1.6.3 类别清单的 Cube/MixCV 类）；候选矩阵按类别枚举。GPU 源码的并行轴选择（thread→(ll,nn) 映射、三维 grid）是**输入而非结论**。**本节大部分判定已由本算子族旧任务 Stage 4 同 shape 实测裁决（msprof op，median of 20，2026-09-17 工具链 1990aa9fe4）——按「继承裁决」记录，当前工具链下由 Stage 4 首轮 profile 复核**。

**轴质量评分维度**（逐候选）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | 全 workload P=64/128（bp=64 → P/bp ∈ {1,2} ✓ 尾轴 128B 对齐 ✓）；Q ∈ {64,128,256}（bl=64 整除 ✓）；N ∈ {32,64,128}（bn=min(128,N)：N=128 单块 K=128 ✓、N=64 单块 K=64 ✓、N=32 单块 K=32 贴分形下限 ✓）；bs=64 与 Q 整除 ✓。尾 lane 浪费率 = 0（全整除）；非整除契约 shape 由 R6 尾块路径覆盖 |
| 尾 lane 浪费率 | 0%（manifest/test 全部 shape 整除；通用契约下尾块 slice 裁剪） |
| 累加链形态 | K 维（bn/bs）进 Cube 分形（无跨步累加问题）；行广播 [M,N]⊙[M,1]（vmul 文档明示）；列因子经 vbrc 两步展开（文档化形态，PL-1.15 实测列广播直乘有 +12~16% 税——弃） |
| repack 代价 | GM 布局契约保留（无 host permute）；核内 cb/x/C/prev 直读（无转置链）——state 的"转置"由 gemm `b_transpose=True` 硬件处理（0 repack）；diff 链行/列广播用 vbrc 展开（2 次/块）；**ws 中继块连续布局**（PL-1.14：band 化实测 +2.3% 回退——MTE3 跨步写 3× 慢） |
| UB 容量影响 | 见 §4.5 预算（177.1KB，贴限 92.3%——Expert 显式 alloc 结构实测 ≈ 手工核算 ±2%） |

**候选矩阵**（Cube/MixCV 类：fractal/NZ 路径 × epilogue 归属 × ws 中继布局 × AIV 映射）：

| # | 布局方案 | 向量化轴 / 关键结构 | repack 路径 | 实测/估算依据 | 是否采纳 |
|---|---------|--------------------|------------|--------------|---------|
| 1 | **主选：契约布局直读 + gemm b_transpose + vbrc 两步展开因子链 + ws 块连续 + AIV 蛇形分片 + 深度 2 流水**（§1.4） | Vector 轴 = 尾轴（N 轴 for [·,N]、bs 轴 for [bl,bs]）；行因子 [M,1] vmul 直乘；列因子 [1,N]→vbrc→同 shape vmul；diag 掩码 = 算术惩罚主选；state 经 `b_transpose=True` 免转置；ws 每块连续 8/16KB；lt 蛇形映射 AIV | 无 repack（state 转置进 Cube 分形；vbrc 仅 diff 链 2 次/块） | 旧档 final 实测 **217.65µs@w2（几何 2.913×）**：深度 2 流水 −39.6% + band 组装 −1.8~4.3% + bn=128 −10.5% + AIV 分片 −34.4% + 蛇形 tie 采纳（ab_test 交错协议）——**结构演化链终点**（opt_log §4）；PL-1.11/1.12/1.13/1.14/1.16 verified 多条目支撑 | ✅（继承裁决：全部组成项均经同 shape 实测胜出） |
| 2 | vsub/vmul 列广播直乘（省 vbrc 展开 1 op/块） | diff = `vsub(vbrc(dA_l), dA_s_row [1,bs])` 列广播 | 无 | **实测否决（v4_p2）**：可编译可算对但 w2 +12.2% / w3 +16.0%（PL-1.15 形态一）；v11 的 vsub alias 形态再证 +17~20% 税（形态二） | ❌（性能税实证；文档化两步形态为主选——T.vsub.md §2.2.2 未列列广播形态） |
| 3 | vcmp/vselect 常量掩码链主选 | int16 idx 常量 + vselect 值替换 | int16 idx 核级预计算 | **实测否决（PL-1.11 verified）**：vcmp int16 全形态标量化（aiv_scalar 94–97%、壁钟 45%），惩罚掩码 4.4–14.2x | ❌ 整除域主选、✅ **band-carrying 域必选**（stale 列 NaN 免疫 + 列守卫，§1.6.1 机器证明） |
| 4 | bl=128 大 l-tile（Q=256 → 2 l-tile；s-block 数 10→6） | 同 #1，bl=128：diag 面积 [128,128] | 无 | **实测否决（容量）**：编译实测需 582KB（bs=128）/ 357KB（bs=64）vs UB 192KB（CONST-capacity-910B2C ssd 第三数据点，logs/bl128_probe.log）——容量否决 | ❌（UB 容量硬边界；Stage 4 若工具链 UB 记账变化可重验） |
| 5 | ws band 化布局（每 l-tile 因果带连续，换 Cube 单条 band 读） | ws_lcb 行 stride 512B | band 化 | **实测否决（v2_band）**：w2 +2.3% 回退——Vector MTE3 跨步写 3× 慢（抵消 Cube 收益）；**块连续 ≫ band 化（PL-1.14 铁律）** | ❌（写侧跨步税） |
| 6 | Cube band 组装（ws 块连续读入 L1 列偏移区 + 每 lt 单条 band gemm + x 任务×pp 级 L1 缓存） | Cube 侧 band 形态；gemm 10→4/任务 | L1 内组装 | **实测采纳（v5_xband）**：w2 −1.8% / w3 −2.2% / w4 −4.3% 全域一致（小幅但非必测 dispatch 无回退） | ✅（并入主选 #1；band 增量组装变体数学不可行——band 行绑定 lt，L0 实测 L_tiles≥2 全挂，PL-1.18 OP1） |
| 7 | host permute 重排输入（cb 预转置 / x 折叠 H） | 各种"便利布局" | 2–6 次 host permute | host permute ~百 µs 级 + host 禁改输入约束（ascend-constraints.md §4）——净亏且违反约束 | ❌（违反 host 约束；permute 产生副本引入 GM 往返） |
| 8 | H 维 fold 任务（同 group head 共享 cb/C 复用） | 任务 = (b,c,g,h_tile) | 无 | cb/C 复读流量 ÷ HPG；但任务数 ÷ HPG（w2: 768→16 < 24 核欠载）；**L2 已吸收复读**（cube read_hit 91%，PL-1.18） | ❌ 首版（Stage 4 候选：L2 命中率实测后重评——R1 #10） |
| 12 | ws 去 P_tiles 维（因子 pp-不变性共享，v1 新增——问题 3 方案 (b)） | c_scaled/lcb 因子与 p 无关（C/cb/dA/dt 均不含 p 维）→ ws_c [24,2,L_tiles,bl,N]、ws_lcb [24,2,L_tiles,L_tiles,bl,bs]，Vector 每任务只产一份、Cube pp 循环共享读 | 无（ws 读索引去 pp 维） | P=128 域 ws 流量与 Vector 产量 ÷P_tiles 减半；但双域流水平衡点与 ws 读语义变化须实测裁决（P=64 全域两形态等价） | ❌ 修订轮（**Stage 4 优化候选**：P=128 workload 实测 ws 流量减半收益后裁决；当前主选 = 方案 (a) 逐 pp 重复生产，对齐 HEAD 旧 verified 终版 L364） |
| 9 | 全 fp32 gemm（HF32 路径，操作数免 cast） | state/prev 已 fp32；cb/x cast 消除 | 无 | 省向量 cast op；Cube 吞吐 HF32 ≈ fp16 一半（docs T.gemm.md §2.3 条 4"fp32×fp32 走 HF32 实测可用"，吞吐比未实测——未文档化假设：按 Ascend HF32 惯例 ~½ 估） | ❌（Cube 吞吐减半代价 > cast 节省；且偏离源数值路径〔dtype 操作数〕，精度基线漂移） |
| 10 | c_scaled 链 fp16 化中转（省 [64,128] vcast/vmul ×2 + UB 32KB） | c_f32 省略，dtype 域直乘 | 无 | **实测否决（v7_cf16）**：`T.vmul` bf16 ×（RETROSPECTIVE 已知约束）需 dtype 分派；w2 −2.4% / w4 +1.4% 收益不稳未过 T-2 协议 | ❌（VP-2026-0002 教训：dtype 特化变体在 bf16 workload 编译失败的先例；收益 <5% 噪声阈） |
| 11 | task 级 dA/dt 行装载（[1,Q] UB 驻留，省 12 次 tiny GM 读/任务） | 因子源任务级驻留 | 无 | **实测否决（CG-2026-0012）**：动态偏移 UB subview 进嵌套循环 → auto-multi-buffer pass 产出非支配 IR（"operand #2 does not dominate this use"，Q≥128 触发，v3_hoist L0 失败实录） | ❌（编译器硬边界；per-block 直载 + L2 吸收为主选） |

**规约子问题（水平 vs 垂直）**：本算子唯一规约是 gemm 的 K 维（n 维 history / s 维 intra）——**全部进 Cube 分形**（K 维累加由 mmadL1 完成，docs T.gemm.md §3），无 Vector 侧水平/垂直归约决策点（`T.reduce` 不出现在主路径）。核内无独立规约轴选择问题，豁免该子项。

**实验裁决计划（继承裁决 + 复核项）**：本算子族旧任务 Stage 4 已对同 shape 完成结构演化链实测裁决（baseline 608.15µs → final 217.65µs@w2，几何 2.913×，六轮 + 二轮取景框共 12 个实验分支、五方向穷尽否决）——**主选方案 #1 的全部组成项均经实测胜出，无「判定裕度依赖未实证常数」的悬空项**。当前工具链（4515de8）下的复核项（Stage 4 首轮 profile 对齐检查，非新裁决）：
- **代表 shape**：w2（mamba2-780m-b1-s4k）+ w4（b2-s32k 长序列稳态）；
- **测量指标**：msprof op Task Duration（median of 20）+ Cube mte2 忙比/段数（对照旧档 83.5%/1216 段/任务锚点）+ AIV 双子块 ratio（PL-1.13 分片生效判据：两 AIV 指标应不同）；
- **判定阈值**：final 结构在当前工具链复测落入旧档记录 ±10%（217.65µs@w2 ±10%）→ 结构继承确认；劣化 >10% → 触发结构复核（依次检查 pass_configs 行为 / flag 语义 / UB 记账——三项均为版本敏感点）；
- **回写路径**：实测数据触发设计修订 → 回写 §1.6.3 判定依据（附录补记）；shape 特化工厂（lru_cache）支持按 Q/N 分派（bn=min(128,N) 已是分派形态）。
- **设计期探针（D-3）**：**不需要**——裁决依赖的全部常数（段数墙/发射税/容量/流水深度）已有旧档实测值（stale 仅版本戳差异，结构性结论不依赖未实证常数；新工具链行为差异由 Stage 4 复核项覆盖，设计期探针无法替代运行期 profile）。

**布局决策结论**：选定方案 #1：核内布局 = **契约布局直读**（x/cb/C/dt/dA/out 不重排）、向量化轴 = **尾轴（N 轴 / bs 轴）**、因子形态 = **行因子 [M,1] vmul 直乘（文档明示）/ 列因子 [1,N] vbrc 两步展开（文档化形态，PL-1.15 税规避）**、对角掩码 = **算术惩罚（整除域主选，PL-1.11 verified）+ vselect+列守卫（band 域）**、state 转置 = **gemm b_transpose 硬件路径**、ws 中继 = **块连续布局（PL-1.14）+ 双槽**、AIV 映射 = **lt 蛇形分片（PL-1.13）**、任务流水 = **深度 2（PL-1.12/1.18 L2 甜点）**、Cube 消费 = **band 组装 + x 任务×pp 级 L1 缓存（v5_xband）**。该结论与 §3.3 伪代码、§4 内存规划（buffer 形状/UB 预算）、§6 循环结构三方一致（累加循环内层向量维 = 尾轴；buffer 形状 = [M,1]/[1,N] 方向化因子 + [bl,bs]/[bs,bp] tile + ws 块连续槽位）。弃选方案的量化理由见候选矩阵（#2 列广播税 +12~16% 实证、#3 vcmp 标量化 verified 实证〔整除域弃、band 域必选〕、#4/#11 容量/编译器硬边界实证、#5 MTE3 跨步税实证、#7 违反 host 约束、#8/#9/#10 收益不稳或代价超收益）。
---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Expert**（用户显式指定「expert模式」；`TILELANG_ASCEND_MODE=Expert` + 显式 `T.Scope("Cube")`/`T.Scope("Vector")` + `T.alloc_L1/L0C/alloc_ub` + 手动 flag 同步 + pass_configs 关闭两个 pass）

### 2.2 选型理由

1. **用户显式指定 expert**（迁移公共规则以用户为准；不可改为 developer）。
2. **结构级实证支持**（本算子族同结构先例）：persistent 分核 + `T.gemm` + v-prefix 混排在 Developer 模式**运行时崩溃**（CG-2026-0010："Illegal instruction … unaligned UUB addresses"，vector core exception retCode=0x31，无编译期诊断；PL-1.16 判据③——user_requirement 指定 Developer 时以该实测为仲裁依据切换，本次用户直接指定 Expert 与实证一致）。本算子的逻辑核数 768–16384 ≫ 24 物理核（§5.5），persistent 分核是极大规模的标准方案，**Developer 无可行路径**。
3. **算子特征匹配**（decision-tree.md §2）：matmul（双 GEMM 路径）+ element-wise 前处理（因子链）→ CV 融合算子；Expert 双 Scope 给出确定性的 CV 切分（Vector 产因子 → ws 中继 → Cube 消费），避免 Developer auto cv_split 在 persistent+gemm 形态的崩溃域（CG-2026-0010）与谓词标量化域（PL-1.16 判据①：aiv_scalar >50%）。
4. **生产先例**：GQA prefill fwd（`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`，当前工作树在位）与本算子族旧任务集成版（**HEAD 提交 4515de8 在位——工作树已由本任务 Stage 0 重建删除，`git show 4515de8` 提交内 `ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py` 可查，目录定位见 §0.7**）均为 Expert 双 Scope persistent 形态，结构同构可参照。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | Cube 侧：`T.alloc_L1`（gemm 操作数 + x 任务×pp 级缓存 + band 组装区）、`T.alloc_L0C`（acc）；Vector 侧：`T.alloc_ub`（全部因子链 buffer + 核级常量）；跨引擎：GM workspace（ws_c/ws_lcb 双槽，显式参数）；**不使用** `T.alloc_fragment`/`T.alloc_shared`（Expert 无 fragment 抽象，PL-1.16 硬边界） |
| 计算方式 | Cube：`T.gemm`（`b_transpose`/`initC`/`size` 参数化）；Vector：v-prefix 链（`T.vmul/vexp/vcast/vsub/vadd/vmin/vbrc/vcmp/vselect`）+ `T.arange`；块内并行由 v-prefix 整块发射表达（无 `T.Parallel`） |
| 同步 | **手动**：`T.rs("PIPE_MTE2"/"PIPE_MTE3"/"PIPE_FIX")` 管线区域 + `T.sync_block_set/wait(id)` 4-flag 握手（深度 2 任务流水，§7） |
| Kernel 形态 | 一维 `T.Kernel(24, is_npu=True)` persistent + 双域 `T.serial` 任务循环（§0.6 R1/R7） |
| 编译配置 | `pass_configs = {TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False, NPUIR_ENABLE_AUTO_MULTI_BUFFER: False}`（PL-1.16：不关闭时 planner 重排/重作用域 UB buffer → codegen "cannot find variable" 崩溃，ssd 第二证）；环境变量 `TILELANG_ASCEND_MODE=Expert` |
| wrapper 契约 | 工厂签名保留源参数（batch…dtype，`threads` 移除——GPU 后端实现参数）；内层工厂 `(block_l, block_p, block_n, block_s, num_stages)`；workspace 以显式参数加入 kernel 签名（PL-1.16：workspace 需求不构成改 wrapper 的理由）；host 侧 `prev_states.to(dtype)` 唯一 dtype 适配 |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（OPT-A/OPT-B 定型形态）；每个步骤优先映射 v-prefix 向量 API，与 §1.6.2 的向量化结论一致。V = Vector 域，C = Cube 域。

| 步骤 | 数学表达 | 域 | 说明 |
|------|----------|-----|------|
| 1 | `dA_l_col = dA[b,h,c,l0:l0+bl]`；`exp_dA_l_col = e^{dA_l_col}` | V | l-tile 行因子源（GM→UB [bl,1] + vexp） |
| 2 | `c_scaled = cast( fp32(C[b,cs+l0..,g,:]) ⊙ exp_dA_l_col → dtype )` | V | history 缩放折进 A 操作数（OPT-A）→ 写 ws_c |
| 3 | `hist = Σ_n c_scaled[i,n]·state_t[p,n]` | C | history gemm（b_transpose，initC 首块覆写） |
| 4 | `diff = vbrc(dA_l_col) − vbrc(dA_s_row)` | V | intra 直接差分（s-block 级） |
| 5 | 对角块：`diff += pen(i,j)`（0/−PEN 核级常量） | V | OPT-B 惩罚融进指数（band 域改 vselect+列守卫） |
| 6 | `lcb = cast( fp32(cb[i,j]) ⊙ e^{diff} ⊙ fp32(dt[j]) → dtype )` | V | intra 因子积 → 写 ws_lcb |
| 7 | `acc += Σ_j lcb[i,j]·x[s0+j, p]`（band 组装形态） | C | intra gemm 链（initC=False 累加） |
| 8 | `out[b, cs+l0+i, h, p] = acc[i,p]` | C | L0C→GM 写回（自动 Fixpipe） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | dA 载入 + exp | `T.copy(dA_cumsum[bz, bh, bc_idx, l0:l0+tl], dA_l_col[0:tl, 0:1])`；`T.vexp(dA_l_col, exp_dA_l_col)` | src 4D 混合 slice；dst [bl,1] fp32 | Expert |
| 2 | c_scaled 链 | `T.copy(C_mat[bz, cs+l0:cs+l0+tl, bg, 0:N], c_16[0:tl, 0:N])` → `T.vcast(c_16, c_f32, round_mode="rint")` → `T.vmul(c_f32, exp_dA_l_col, c_f32)`（**行广播 [M,N]⊙[M,1]**）→ `T.vcast(c_f32, c_scaled_16, round_mode="rint")` → `T.copy(c_scaled_16[0:tl, 0:N], ws_c[kid, slot, pp, lt, 0:tl, 0:N])` | GM→UB dtype 直拷；f16→f32 上行 rint（vcast.md dtype 矩阵：f16→f32 仅 rint） | Expert |
| 3 | history gemm | `T.copy(ws_c[kid, slot, pp, lt, 0:tl, n0:n0+tn], l1_c[0:tl, 0:tn])`；`T.copy(prev_states[bz, bc_idx, bh, p0:p0+tp, n0:n0+tn], l1_state[0:tp, 0:tn])`（host 预 cast dtype）；`T.gemm(l1_c, l1_state, l0_acc, initC=(n_blk == 0), b_transpose=True, size=[tmc, tn, tnp])` | slice 尾块 + size-form；state 直装载 [bp,bn]；acc 首块覆写初始化 | Expert |
| 4 | diff 链 | `T.copy(cb[bz, bc_idx, bg, l0:l0+tl, s0:s0+ts], cb_16, size=[tl, ts])` → `T.vcast(cb_16, cb_f32, round_mode="rint")`；`T.copy(dA_cumsum[bz, bh, bc_idx, s0:s0+ts], dA_s_row[0:1, 0:ts])`；`T.vbrc(dA_l_col, dA_l_mat)`；`T.vbrc(dA_s_row, dA_s_mat)`；`T.vsub(dA_l_mat, dA_s_mat, dA_l_mat)`（in-place） | vbrc 两步展开（[bl,1]/[1,bs]→[bl,bs]，文档化形态，PL-1.15 税规避） | Expert |
| 5 | 对角惩罚 | `T.vadd(dA_l_mat, pen_const, dA_l_mat)`（仅 s_blk == lt，trace 静态比较） | pen_const 核级常量（步骤 0 生成） | Expert |
| 5' | band 域变体 | `... vmul(·, dt_mat) → T.vselect(band_mask, lcb_f32, zero_f32, lcb_f32) → T.vcast`（+ 列守卫：装载 slice 只拷 [0:ts] + band_mask 含 `j < ts` 列判据或装载后清零 stale 列） | NaN 免疫（§1.6.1 机器证明：列守卫必须） | Expert |
| 6 | lcb 因子积 | `T.vexp(dA_l_mat, dA_l_mat)` → `T.vmul(cb_f32, dA_l_mat, lcb_f32)`；`T.copy(dt[bz, bh, bc_idx, s0:s0+ts], dt_16[0:1, 0:ts])` → `T.vcast(dt_16, dt_s_row, round_mode="rint")` → `T.vbrc(dt_s_row, dt_mat)` → `T.vmul(lcb_f32, dt_mat, lcb_f32)` → `T.vcast(lcb_f32, lcb_16, round_mode="rint")` → `T.copy(lcb_16[0:tl, 0:bs], ws_lcb[kid, slot, pp, lt, s_blk, 0:tl, 0:bs])` | 全 v-prefix in-place 链 | Expert |
| 7 | intra gemm（band 组装，单 buffer 直载——v2 与 §1.4 同步） | `T.copy(ws_lcb[kid, slot, pp, lt, s_blk, 0:bl, 0:ts], l1_band[0:bl, s0:s0+ts])`（GM→L1 直载列偏移 dst，src/dst 双侧 ts 裁剪）×(lt+1) 块 → `T.gemm(l1_band, l1_x, l0_acc, initC=False, size=[tmc, l0+tl, tnp])`（band 形态：K = l0+tl 连续带，单条/任务·lt） | x 任务×pp 级 L1 缓存 [Q,bp]（R7.4） | Expert |
| 8 | 写回 | `T.copy(l0_acc[0:tl, 0:tp], out[bz, cs+l0:cs+l0+tl, bh, p0:p0+tp])` | L0C→GM 自动 Fixpipe（fp32 直写） | Expert |
| 0 | 核级常量 | `T.arange(pen_i, [1, 0], 0)`；`T.arange(pen_j, [0, 1], 0)`（fp32 行/列索引）→ `T.vsub(pen_i, pen_j, d)` → `T.vmin(d, zero, d)` → `T.vmul(d, pen_val, pen_const)`（pen_val = 1e30） | 任务循环外一次（Vector 域） | Expert |

### 3.3 计算伪代码

> dtype 位置参数化（迁移模板规则：`T.Tensor(shape, dtype)` 第二位置参数）。完整结构见 §1.4（Cube 域 / Vector 域两段已展开）；此处补齐 trace 分派与工厂层。

```python
# 工厂层（trace-time，lru_cache 特化）
#   shape 校验：S == C*Q、H % G == 0；NUM_KERNELS = 24（NPUUtils 实查，§5.5）
#   num_logical = B * C * H；bl = bs = bp = 64；bn = min(128, N)
#   L_tiles = ceildiv(Q, bl)；P_tiles = ceildiv(P, bp)；N_tiles = ceildiv(N, bn)
#   trace 分派（R4）：band_free = (Q % bs == 0)——主 trace（惩罚掩码）；
#                     band 域（Q % bs != 0）追加 band_mask/zero 常量与 vselect+列守卫链
#   ws 形状：ws_c [24, 2, P_tiles, L_tiles, bl, N]（dtype）
#            ws_lcb [24, 2, P_tiles, L_tiles, L_tiles, bl, bs]（dtype）

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

@tilelang.jit(out_idx=[-1], target="npuir", pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
})
def kernel(x, cb, dA_cumsum, C_mat, prev_states, dt, ws_c, ws_lcb, out):
    with T.Kernel(24, is_npu=True) as (kernel_id, subid):
        num_local_tasks = T.ceildiv(num_logical - kernel_id, 24)
        # ---- Cube 域：见 §1.4（prologue 前导 set ×2 → 任务循环 [wait READY →
        #      pp 循环 [x 装载（每 (task,pp) 一次）→ lt 循环 [history n-loop gemm
        #      + intra band 组装与 band gemm + 写回]] → set CONS]）----
        with T.Scope("Cube"): ...
        # ---- Vector 域：核级常量 pen_const（arange/vsub/vmin/vmul，一次）
        #      → 任务循环 [任务头 wait CONS（先于任何 ws 写，深度 2 节拍：任务 W
        #      等任务 W−2 同槽消费完）→ pp 循环 [蛇形 lt 分片 [c_scaled 链 → ws_c；
        #      diff 链 → ws_lcb]] → set READY（任务尾）] ----
        with T.Scope("Vector"): ...

# host wrapper（内层闭包）：
#   wrapped(x, cb, dA_cumsum, C_mat, prev_states, dt):
#     ws_c   = torch.empty((24, 2, P_tiles, L_tiles, bl, N), dtype, device)
#     ws_lcb = torch.empty((24, 2, P_tiles, L_tiles, L_tiles, bl, bs), dtype, device)
#     prev_c = prev_states.to(dtype)          # Expert T.copy 不跨 dtype（docs T.copy.md §2.2.1）
#     return kernel(x, cb, dA_cumsum, C_mat, prev_c, dt, ws_c, ws_lcb)
```

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|----------|------|
| `T.Kernel(24, is_npu=True)` 一维 persistent + (kernel_id, subid) | docs/开发指南.md §3.3 官方模板；GQA/ssd 生产代码同款（当前工作树） | ✅ 生产实证 |
| `T.Scope("Cube")` / `T.Scope("Vector")` 双域 | docs 同步管道操作示例（T.sync_block_set.md §2.4）；GQA/ssd 生产 | ✅ 生产实证 |
| `T.gemm(src1, src2, dst, size, initC, b_transpose)`（Expert 直调） | docs/Tilelang.language/线性代数操作/T.gemm.md（`size`/`initC`/`a_transpose`/`b_transpose` 参数 §2.1；fp16/bf16 × fp32 dst §2.3）；GQA L334/L376、ssd 集成版生产（Expert 下 T.gemm 与 T.npuir_dot 别名等价，以 T.gemm 为主） | ✅ 文档 + 生产实证 |
| `T.copy` slice / size / 尾块形态 | docs/Tilelang.language/内存操作/T.copy.md（§2.1 extent 推断规则 1–6；§2.2.1 **Expert 通用路径 dtype 一致约束** + L0C→GM 自动 Fixpipe；GM→L1 自动 ND2NZ） | ✅ 文档确认 |
| `T.alloc_L1` / `T.alloc_L0C` / `T.alloc_ub` | docs/Tilelang.language/内存操作/T.alloc_L1.md / T.alloc_L0C.md / T.alloc_ub.md | ✅ 文档确认 |
| `T.rs("PIPE_MTE2"/"PIPE_MTE3"/"PIPE_FIX")` + `T.sync_block_set/wait(id)` | docs/Tilelang.language/同步管道操作/T.sync_block_set.md / T.sync_block_wait.md（§2.4 双 Scope 示例；set/wait 须在 T.rs 上下文）；GQA/ssd 生产 | ✅ 文档 + 生产实证 |
| `T.vmul`（tensor·tensor 同 shape / 行广播 [M,N]⊙[M,1]） | docs/Tilelang.language/数学操作/T.vmul.md §2.2.2（**明示行广播 [M,N]*[M,1]** 与标量广播；**无 [M,N]*[1,N] 列因子形态**）；GQA L519/L654 标量乘、L645 行广播生产先例 | ✅ 文档 + 生产实证（列因子形态弃用——PL-1.15 性能税，vbrc 两步展开替代） |
| `T.vsub` / `T.vadd` / `T.vmin`（diff 链/惩罚链） | docs 数学操作目录（同 shape 语义）；PL-1.11 惩罚链 verified 同款算术；**alias 形态（vsub dst=src2）有 +17~20% 税**（PL-1.15 形态二）——设计中 diff 链为 dst=src1 in-place 形态 | ✅ 文档 + 生产实证 |
| `T.vexp`（fp32） | docs/Tilelang.language/数学操作/T.vexp.md（fp16/fp32 ✓，bf16 ×——本设计 exp 全程 fp32 域，不受限） | ✅ 文档确认 |
| `T.vcast`（f32↔f16/bf16, round_mode） | docs/Tilelang.language/数据类型转换操作/T.vcast.md（f32→f16/bf16 全 round mode；**f16→f32 仅 rint**——全部上行 cast 显式 `round_mode="rint"`） | ✅ 文档确认 |
| `T.vbrc`（向量/标量广播） | docs/Tilelang.language/shape操作/T.vbrc.md（§2.2.2 rank 一致 + 尺寸 1 广播规则；[bl,1]→[bl,bs] 与 [1,bs]→[bl,bs] 均合法） | ✅ 文档确认 |
| `T.arange`（strides 形态，fp32 精确整数） | docs/Tilelang.language/创建操作/T.arange.md；PL-1.11 同款负步长形态 | ✅ 文档 + verified 实证 |
| `T.vcmp`（bool 片上）+ `T.vselect` | docs/Tilelang.language/比较操作/T.vcmp.md（§2.3 条 4：bool 不可写回 GM，配合 vselect）；条件操作/T.vselect.md | ✅ 文档确认（band 域专用） |
| `T.ceildiv` / `T.min` / `T.max`（标量 PrimExpr） | GQA L272/L291/L293 生产（含 ceildiv 负数陷阱注释）；docs/开发指南.md | ✅ 生产实证 |
| `pass_configs`（PassConfigKey 两项关闭） | PL-1.16（GQA L231–237 / ssd 集成版生产同款） | ✅ 生产实证 |

**未实证形态**：无（本设计所有 API 形态均有文档条目 + 当前工作树生产先例双重佐证；旧版设计的探针 P-1/P-2〔vmul/vsub 列广播〕已被 PL-1.15 实测定局为弃用形态，不再设探针）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测 5 项）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源为 `T.Kernel(l·p, B·C, H)`） | 一维乘积展开 + persistent 24 核核内串行（§0.6 R1；ascend-constraints.md 限制 1） |
| GPU 专用 API（threads/swizzle/sync_threads） | Yes（threads=128、make_swizzled_layout、T.sync_threads） | 全部舍弃/替换（§0.5 行 3/13）；`threads` 参数在 NPU 侧 wrapper 移除（GPU 后端实现参数） |
| GEMM 非整除（M/N 不被 block 整除） | 契约上可能（任意 Q/P/N）；当前 workload 全整除 | R6 尾块路径（T.min tail + slice copy + size-form gemm + 分形钳位 tmc/tnc ≥ 16、K ≥ 32 + stale 带中和）；§8.2 L0 边界用例覆盖 |
| L0C 溢出（bl·bp·4B > 128KB） | bl·bp = 64×64 = 4096 × 4B = 16KB ≤ 128KB | ✅（§5.3 复核；bl=128 变体已被 UB 容量否决——§1.6.3 #4） |
| 分核策略三要素 | Yes（逻辑核数 8–16384 跨越三档） | §5.5（实查 24 + persistent 串行方案） |

#### 3.5.2 参考实现差异说明（影响 API 选型的关键差异汇总；完整差异分析见 §0.5/§0.6）

| 差异项 | 参考实现（GPU） | 本项目（Ascend Expert） | 转换方案 |
|--------|----------------|------------------------|---------|
| Kernel 维度 | 三维 `T.Kernel(·,·,H)` + threads | 一维 persistent 24（`is_npu=True`）+ 双域双 AIV | §0.6 R1/R8 |
| GEMM | `T.gemm`（GPU mma，fragment 累加） | `T.gemm`（Cube，`b_transpose`/`size`/`initC` 参数化，L0C dst） | docs T.gemm.md；state 转置经 b_transpose（R2） |
| 内存分配 | `alloc_shared/alloc_fragment`（SMEM/寄存器） | `alloc_L1/alloc_L0C`（Cube）+ `alloc_ub`（Vector）+ GM ws | PL-1.16 结构；§4 |
| 标量 exp/分支 | `T.exp` fragment 标量 + `if_then_else` | `T.vexp/vselect/vcmp` 向量链（Vector 域） | R3/R4 |
| 布局 | SMEM swizzle | 无（块连续 ws + ND2NZ 自动分形） | §0.5 舍弃行 3；PL-1.14 |
| 流水 | `T.Pipelined(num_stages=3)` cp.async | 深度 2 任务级双域流水（ws 双槽 + 4 flag + 前导 set） | R7（Expert 无 Pipelined 抽象，PL-1.16） |
| prev_states dtype | kernel 内 `T.cast`（装载时） | host 侧 `.to(dtype)`（Expert T.copy 不跨 dtype） | R2；cast 语义一致（rint） |

#### 3.5.3 本项目同类实现参考

**必须列出**：本项目 examples/ 中最相似的实现：

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | **高度相似**（Expert MixCV persistent：双 Scope + alloc_L1/L0C/ub + T.gemm + sync_block_set/wait + ghost 钳位 + 尾块分形钳位 + ws 中继） | Kernel 骨架（persistent 24 + 双域任务循环 + flag 握手 + pass_configs 双关闭）、惩罚掩码先例（PL-1.11）、蛇形分片边界表达式（PL-1.13 v11 先例）、T.ceildiv 陷阱注释（L258） |
| `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/_s_s_d_chunk_scan_fwd_kernels.py` | 迁移目标（GPU 源提取，Stage 3 重写对象） | 接口契约与源语义基准 |
| `examples/mixcv/mixcv_mixkernel.py` | 中（MixCV 双 Scope 最小样例） | load_nd2nz/store_fixpipe 形态（本设计未用——slice T.copy 路线，GQA D1 实证跨步区 load_nd2nz 误读）、sync_block_set/wait 语义 |
| `examples/flash_attention/flash_attn_npuir_dev.py` | 中（Developer 模式 MixCV 参照——本设计不采用，仅对照） | Developer 形态边界（本算子结构在 Developer 崩溃域，CG-2026-0010） |
| `examples/TileOPs/tileops/testing/mamba2_reference.py` | golden 基准 | §8.1 直接采用 |
---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| x | (B, S, H, P) | fp16/bf16 | 主输入；S = C·Q；行（s 维）stride = H·P 元素，行内 p 连续 |
| cb | (B, C, G, Q, Q) | 同 x | 预计算耦合；group-owned（同组 head 复读，L2 缓解——cube read_hit 91% 实测）；tile 切片为最后两维 |
| dA_cumsum | (B, H, C, Q) | fp32 | 单调非增；[Q] 连续切片直拷 |
| C_mat | (B, S, G, N) | 同 x | readout；行 stride = G·N，行内 n 连续 |
| prev_states | (B, C, H, P, N) | fp32 | **P 在 N 前**：state_t 直装载 [bp,bn] 行 stride = N，行内 n 连续（R2 免转置）；**host 侧预 cast → dtype**（Expert T.copy 不跨 dtype） |
| dt | (B, H, C, Q) | 同 x | [Q] 连续切片直拷（dtype 一致拷贝，vcast rint 上行在 UB 内） |
| ws_c（workspace） | (24, 2, P_tiles, L_tiles, bl, N) | 同 x | c_scaled 因子中继（Vector→Cube）；块连续布局 |
| ws_lcb（workspace） | (24, 2, P_tiles, L_tiles, L_tiles, bl, bs) | 同 x | lcb 因子中继；块连续布局（PL-1.14） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| out | (B, S, H, P) | **fp32** | 与 x 同 layout（非转置）；写回 slice [cs+l0 : cs+l0+tl, p0 : p0+tp]（L0C→GM 自动 Fixpipe） |

### 4.3 中间缓冲区

> Expert 显式分配（无 fragment 抽象）；**ws GM 工作集**（w2：ws_c 3.1MB + ws_lcb 6.1MB ≈ 9.2MB / 全核 2 任务 in-flight——PL-1.18 深度 3 的 L2 劣化论证即以此为工作集基数）。设计默认 bl=bs=bp=64、bn=min(128,N)、Q=256、N=128（w2 标定）。

**Cube 域（L1 / L0C）**：

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| l1_c | (bl, bn) | dtype | L1（`alloc_L1`） | c_scaled 操作数（history gemm A） |
| l1_state | (bp, bn) | dtype | L1 | prev_states 直装载（history gemm B，b_transpose） |
| l1_x | (Q, bp) | dtype | L1 | **x 任务×pp 级缓存**（R7.4：每 (task,pp) 一次装载、pp 内跨 lt 复用，消跨 lt 2.5× 重读；Q=256 时 32KB） |
| l1_band | (bl, Q) | dtype | L1 | **band 组装区（唯一 band buffer，v2：GM→L1 直载列偏移，ts 裁剪——TRAP-L1-band-dst-tail-overrun）**；每 lt 的因果带 [bl, l0+tl]；w2 最大 32KB |
| l0_acc | (bl, bp) | fp32 | L0C（`alloc_L0C`） | 统一累加器（history 首块 initC=True 覆写初始化 → intra initC=False 累加；**纯 Cube 驻留零 rescale**，R5） |

**Vector 域（UB）**：

| Buffer 名 | Shape | dtype | 大小 (Bytes) | 用途 |
|-----------|-------|-------|-------------|------|
| pen_const | (bl, bs) | f32 | 16,384 | 核级惩罚常量（0/−1e30；arange/vsub/vmin/vmul 一次生成；任务循环外） |
| dA_l_col / exp_dA_l_col | (bl, 1) | f32 | 2×256 | 行因子源 / history 缩放因子 |
| dA_s_row / dt_s_row / dt_16 | (1, bs) | f32 / dtype | 2×256 + 128 | s-block 列因子源（直拷 + vcast rint） |
| cb_16 / cb_f32 | (bl, bs) | dtype / f32 | 8,192 + 16,384 | cb 装载（dtype 拷贝）+ 上行 |
| lcb_f32 / lcb_16 | (bl, bs) | f32 / dtype | 16,384 + 8,192 | intra 因子积（in-place 链）+ 下行 |
| c_16 / c_f32 / c_scaled_16 | (bl, N) | dtype / f32 / dtype | 16,384 + 32,768 + 16,384 | history 因子链（N=128 口径） |
| dA_l_mat / dA_s_mat / dt_mat | (bl, bs) | f32 | 3×16,384 | diff 链展开（vbrc 两步） |
| **稳态峰值合计** | | | **≈ 177.1 KB（181,376B）** | （初始化期峰值 = pen 链中转 48KB < 稳态，非瓶颈） |

### 4.4 内存搬运路径

```
【Vector 域】GM[dA] ─copy─▶ UB[dA_l_col (bl,1) f32] ─vexp─▶ UB[exp_dA_l_col]
GM[C] ─copy─▶ UB[c_16] ─vcast(rint)─▶ UB[c_f32] ─vmul(exp_dA_l_col 行广播)─▶ UB[c_f32] ─vcast(rint)─▶ UB[c_scaled_16] ─copy─▶ GM ws_c[slot]
GM[cb] ─copy(size)─▶ UB[cb_16] ─vcast(rint)─▶ UB[cb_f32]
GM[dA/dt] ─copy─▶ UB[dA_s_row/dt_16] ─vcast(rint)─▶ UB[dt_s_row]
UB[dA_l_col/dA_s_row] ─vbrc×2─▶ UB[dA_l_mat/dA_s_mat] ─vsub(in-place)─▶ diff
UB[pen_const] ─vadd（仅对角块）─▶ diff ─vexp─▶ decay ─vmul(cb_f32)─▶ ─vmul(vbrc(dt_mat))─▶ lcb_f32 ─vcast(rint)─▶ lcb_16 ─copy─▶ GM ws_lcb[slot]
                                    （READY(slot) flag：PIPE_MTE3）
【Cube 域】GM ws_c[slot] ─copy─▶ L1[l1_c]；GM[prev(host cast)] ─copy─▶ L1[l1_state]
GM[x[b,cs:cs+Q,h,p0:p0+tp]] ─copy(每 (task,pp) 一次)─▶ L1[l1_x (Q,bp)]
GM ws_lcb[slot] ─copy(块连续, 列偏移 dst, ts 裁剪)─▶ L1[l1_band]
L1[l1_c]×L1[l1_state] ─T.gemm(b_transpose=True, initC=首块)─▶ L0C[l0_acc]
L1[l1_band]×L1[l1_x] ─T.gemm(initC=False)─▶ L0C[l0_acc]（累加）
L0C[l0_acc] ─T.copy（自动 Fixpipe）─▶ GM[out]（写回裁剪 [0:tl,0:tp]）
```

### 4.5 UB 内存预算

> 设计默认 bl=bs=bp=64、N=128（[64,64] fp32 = 16KB / dtype = 8KB；[64,128] dtype = 16KB / fp32 = 32KB）。**Expert 显式 alloc 结构的记账特性**（CONST-capacity-910B2C ssd 第三数据点，1990aa9fe4 实测，stale 待重验）：BishengIR 实际需求 ≈ 手工核算 ±2%（v3 实测 +16KB 手工 → 195.1KB 报错，偏差 <2%）——**Expert 显式结构不适用 1.7× auto-multi-buffer 膨胀口径**（该系数属 Developer/auto-multi-buffer 域），也不必按 1.10–1.12 上偏（那是 auto-multi-buffer=false 下 Developer 形态的口径）。

| 域 | 峰值集合 | 手工核算 | 对照 192KB |
|----|---------|---------|-----------|
| 整除域主 trace（稳态） | §4.3 Vector 全清单（逐项和 181,376B，v1 复算修正） | **≈ 177.1 KB（92.3%）** | ✓ 贴限通过（±2% 记账 → ≤ 181KB） |
| band 域 trace（Q%bs≠0 分派） | 稳态 − pen_const(16KB) + band_mask(4KB bool) + zero_f32(16KB) | **≈ 180.7 KB（94.1%）** | ✓ 贴限通过（消减序备用，见下） |
| 初始化期 | pen_const + arange 中转 ×2 | 48 KB | ✓ 非峰值 |

**贴限消减序（band 域或 Stage 3 实测超限时）**：① zero_f32 与 lcb_16 复用（vselect else 值恒 0，可先 `T.copy(lcb_16, zero)` 占位生成——省 16KB）；② c_f32 与 lcb_f32 生命周期错峰复用（c 链完成于 intra 链开始前——省 16KB）；③ bs=32 band 域变体（diag 链 [64,32] 减半）。**Stage 3 首次编译以 BishengIR `ub overflow` 报错文本实测校准**（CONST-capacity 条目方法）。**L1 侧**：l1_c 16K + l1_state 16K + l1_x 32K + l1_band 32K ≈ **96KB ≤ 512KB**（余量 5.3×；Q 更大时 l1_x/l1_band 按 [·,Q] 线性增长，Q=512 → 144KB 仍宽裕）。**L0C 侧**：l0_acc 16KB ≤ 128KB（余量 8×）。

### 4.6 动态轴定义

**无**（工厂 `lru_cache` 按 (B, C, Q, H, P, N, G, dtype) 静态特化；契约 shape 由 host `forward` 校验——`S == C·Q`、`H % G == 0`，kernel 内全部循环边界为 trace-time 常量或 kernel_id/task_id/循环变量派生 PrimExpr）。

### 4.7 JIT 配置

```python
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")   # Expert 模式声明（GQA/ssd 生产同款）

@tilelang.jit(out_idx=[-1], target="npuir", pass_configs={
    # PL-1.16 Expert 硬边界（两项必关，否则 codegen "cannot find variable" 崩溃）：
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
})
# out_idx=[-1]：仅 out 为输出（对齐源工厂 @tilelang.jit(out_idx=[-1])）；
# ws_c/ws_lcb 为显式 workspace 参数（host 分配，非编译输出）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：混合（MixCV：Cube 双 GEMM 消费 + Vector 因子生产，GM ws 中继 + 深度 2 任务流水）

**判定依据**：算子含 2 类 matmul（history `c_scaled@state_t`、intra `lcb@x`）+ 逐元素前处理（因子链），Expert 显式双 Scope 切分（§2）。

### 5.2 Block 划分

```python
bl = 64             # l-tile（Q 维）：源默认同值；UB 容量锁定（bl=128 实测 357–582KB > 192KB 否决）
bs = 64             # s-tile（s 维）：源默认同值；diag 直接差分的掩码块
bp = 64             # p-tile（P 维）：源默认同值；P∈{64,128} 整除（P=128 时 2 块）
bn = min(128, N)    # n-tile（history K 维）：**N=128 单块**（v6_bn128 实测 −10.5%，n-loop 的
                    #   mte2+gemm 减半）；N<128 自然回退单块（N∈{32,64} 全域单块）；
                    #   工厂层钳位防分形 stale 带（opt_log §6 修复记录）
NUM_KERNELS = 24    # persistent 物理核数（§5.5 实查）
# 逻辑任务（任务粒度 = (b, c, h)，R1）：num_logical = B * C * H
# 核内展开：AIV 蛇形 lt 分片 × pp-loop × [history n-loop + intra band s-loop + diag]
```

**取值理由**：64 对齐源 GPU 默认 config（block_l/p/s=64）——已验证的负载形态；分形 M/N=64 ≥ 16、K=64/128（或 32）≥ 32 ✓；尾轴 64×2B = 128B ≥ 32B 对齐 ✓；L0C acc [64,64]×4B = 16KB ≤ 128KB；**bn=128 为本设计相对源默认（64）的唯一 tiling 提升**——依据 v6_bn128 实测（w2 −10.5% / w3 −5.0% / w4 −15.7%：N=128 时 n-loop 2 块 → 1 块，Cube 省 4 mte2 + 4 gemm/任务），N<128 时 `min` 钳位自然回退且无分形 stale 带（bn ≤ N ⟹ K 维全真实）。

### 5.3 约束分析

- **对齐约束**：bl/bp/bs ∈ {64}、bn ∈ {32,64,128}，尾轴 32B 对齐 ✓（fp16 64 元素 = 128B）；分形限制 M/N ≥ 16、K ≥ 32 ✓（smoke N=32 → bn=32 贴 K 下限）。
- **UB 容量**：稳态峰值 ~177.1KB ≤ 192KB（贴限 92.3%，Expert 显式 alloc 记账 ±2%，§4.5）；band 域 ~180.7KB（94.1%，消减序备用）。
- **L0 容量**：l0_acc [64,64] fp32 = 16KB ≤ 128KB ✓。
- **L1 容量**：~96KB（w2，v2：删 l1_lcb 中间 buffer 后 16+16+32+32）≤ 512KB ✓。
- **GEMM 非整除**：契约 shape 任意时 `Q % bl`、`P % bp`、`N % bn` 可能非零——R6 尾块路径（当前 manifest/test 全整除，尾块为契约完备性设计）。
- **ws GM 工作集**：9.2MB（w2，2 任务 in-flight）——深度 3 会膨胀到 13.8MB 劣化 L2（PL-1.18 实测否决），深度 2 为 L2 甜点。

### 5.4 注意事项

1. **T.ceildiv 负数陷阱**：`num_local_tasks = T.ceildiv(num_logical − kernel_id, 24)` 在 `kernel_id ≥ num_logical`（欠载域）时返回 1 而非 0（截断除法 + 1 的 lower 行为，GQA 2026-09-15 工具链实证）——ghost 任务以 `cid = T.min(·, num_logical − 1)` 钳位到合法任务重算（bit-identical 写回，安全）。w1（8 任务/24 核）即落此域。
2. **对角块数值边界（惩罚主选分域论证）**：diag 直接差分的下三角指数恒 ≤ 0（s ≤ l ⟹ dA_l ≤ dA_s）；上三角正指数真实分布 ~e^32、深衰减可至 fp32 上限——**整除域**（惩罚掩码）：上三角指数被 `−PEN`（PEN=1e30）吸收为 −PEN（大数吃小数）→ exp(−PEN) 下溢精确 +0.0，且 PEN 为有限哨兵规避 −inf−(−inf) NaN 路径（PL-1.11 −1e38 同理）；**band 域**（vselect+列守卫）：上三角大值被选路丢弃、stale 列 NaN 不进算术。双域 cast 输入均有限 → 无 inf/NaN 进 gemm；深衰减角点由 verify_equiv deep_decay（dA→−128 量级）与 block_decay_gt88（块内衰减 >88）案 PASS 覆盖。
3. **bn=32（N=32 workload）**：history gemm K=32 贴分形下限，单 n-block 无流水——正确性无碍，性能由 Stage 4 评估。
4. **bf16 路径**：gemm bf16×bf16→fp32 ✓（docs T.gemm.md §2.3 条 3"建议 fp32，实测可用"）；vexp 仅 fp32 ✓（本设计 exp 全程 fp32 域）；**`T.vmul` bf16 ×**（v7_cf16 教训：c_scaled 链 dtype 化中转在 bf16 workload 编译失败，VP-2026-0002）——因子链中转保持 fp32（c_f32/lcb_f32），仅首尾 vcast 进出 dtype；GQA bf16 先例为 f32 ws 载体（保守），本设计 gemm 操作数保持 bf16（对齐源数值路径），精度风险列 §9。
5. **AIV 分片退化域**：L_tiles=1（Q=64）时蛇形式折叠为两 AIV 重复 lt=0——bit-identical 良性（PL-1.13）；smoke（Q=64）即此域。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **物理核数（要素②，实查）**：**24**。查询代码与返回值记录：

```python
from tilelang.utils.npu_utils import NPUUtils
print(NPUUtils.get().get_aicore_num())
# 2026-09-20 12:31 实查输出：24（本机 Ascend910B2C；当前工具链 tilelang 0.1.2+4515de8 dev root build）
# 混合 Cube+Vector 算子直接使用返回值 24；与 GQA 设计 2026-09-15 实查记录（24）、
# ssd 旧任务 2026-09-17 实查记录（24）及 CONST-aicore-910B2C 条目一致。
```

- **逻辑核数（要素①）**：任务粒度 = `(b, c, h)`（§0.6 R1），`num_logical = B × C × H`：
  - w1 smoke（1,2,·,4）：1×2×4 = **8**
  - w2 mamba2-780m-b1-s4k（1,16,·,48）：**768**
  - w3 mamba2-2p7b-b4-s2k（4,8,·,80）：**2560**
  - w4 mamba2-1p3b-b2-s32k（2,128,·,64）：**16384**
  - 测试 fixture：(1,2,·,4)=8、(2,4,·,8)=64、(2,2,·,4)=16
- **规模判定（要素③）**：w2–w4 逻辑核数 768–16384 ≫ 24（**极大规模**，无法通过调整 block 缩减——任务粒度由数据维度 (b,c,h) 决定，bl/bp/bn 的调整不改变任务数）；w1/部分测试 case（8–16 < 24，欠载域）。
- **分核方案**：**固定启动内核数 = 物理核数 24，核内 `T.serial` 串行处理多个逻辑任务**（极大规模标准方案，开发指南 §3.3 官方模板）：`T.Kernel(24, is_npu=True)`，每核 `num_local_tasks = T.ceildiv(num_logical − kernel_id, 24)`、任务解码 `cid = T.min(task_id·24 + kernel_id, num_logical − 1)`（轮转 + ghost 钳位，注意事项 1）。**循环边界静态性**：`num_local_tasks` 为 kernel_id 派生的 PrimExpr（进入循环前一次计算、循环内不变；开发指南 §3.3 官方模板同形，GQA E1–E7 D3 实证可 lower）——不含运行时动态 shape。**欠载域处置**（w1 等 num_logical < 24）：16 个核空载 + ghost 钳位重算（bit-identical 安全，正确性无损；smoke 域性能不敏感——旧档实测 smoke 跨轮 30.75–32.17µs 平区摆动）；Stage 4 可按 shape 分派非 persistent 直发形态（`T.Kernel(num_logical)`，lru_cache 工厂天然支持），不在 Stage 1 特化。
---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环，无任何逐元素标量循环。**Expert 形态**：kernel 内无 `T.Pipelined`/`T.Parallel`（PL-1.16 硬边界）——块内流水意图由 `T.rs` 管线区域静态排序 + 深度 2 任务级双域流水承接。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务级（(b,c,h) 分派，双域各一条） | persistent 核内串行 | `T.serial(num_local_tasks)` | 分核策略要素③（§5.5）；深度 2 流水的槽位/flag 管理要求任务序严格递增 |
| p 维（P=128 时 2 块） | 块级迭代 | `T.serial(P_tiles)`（**双域各一条**：Cube 域 pp→lt、Vector 域 pp→lt〔问题 3 方案 (a)〕） | 任务内 p-tile 展开（P=64 时单次）；Vector 侧因子按 pp 重复生产写 ws |
| AIV 分片（lt → AIV 映射） | 边界表达式循环 | `T.serial(lt_count)` + 蛇形仿射式 `lt = i + subid + (i%2)·(L_tiles−2i−2·subid)` + clamp | PL-1.13（消双 AIV 重复执行，−28~−36% 实测）；无 if 守卫（TRAP-tvm-parser-rules 运行时分支风险）；置于 pp 循环之内（§1.4） |
| l 维（Q/bl 个 l-tile） | 块级迭代 | `T.serial(L_tiles)`（Cube 域）/ 蛇形分片循环（Vector 域） | 因子与 band gemm per-l-tile |
| history n 维 | 块级迭代 | `T.serial(N_tiles)` | bn=min(128,N) ⟹ 当前 workload 全域单块（N ∈ {32,64,128} ≤ bn）；通用契约多块时升序累加 |
| intra s 维（full-lower + diag） | 块级迭代 | `T.serial(lt + 1)` | s_blk ∈ [0, lt]——full-lower（s_blk < lt，无惩罚）+ 对角（s_blk == lt，惩罚/trace 静态比较）；上三角由循环边界完全跳过（源 #10 保留） |
| 元素级 | 向量化 | v-prefix 链（§3.2） | §1.6.2 结论 |

### 6.2 循环伪代码

```python
with T.Kernel(24, is_npu=True) as (kernel_id, subid):
    num_local_tasks = T.ceildiv(num_logical - kernel_id, 24)
    with T.Scope("Cube"):
        ...alloc L1/L0C（l1_c/l1_state/l1_x/l1_band/l0_acc）...
        with T.rs("PIPE_FIX"):                                   # prologue：前导 set（PL-1.12）
            T.sync_block_set(CONS0); T.sync_block_set(CONS1)     # 两槽初始"空闲"
        for task_id in T.serial(num_local_tasks):
            cid = T.min(task_id * 24 + kernel_id, num_logical - 1)   # ghost 钳位
            bz, bc_idx, bh = decode(cid); slot = task_id % 2
            with T.rs("PIPE_MTE2"): T.sync_block_wait(READY(slot))  # 因子就绪
            for pp in T.serial(P_tiles):
                p0 = pp * bp; tp = T.min(bp, P - p0)
                T.copy(x[bz, cs:cs+Q, bh, p0:p0+tp], l1_x[0:Q, 0:tp])  # x 每 (task,pp) 一次（pp 内 lt 外）
                for lt in T.serial(L_tiles):
                    l0 = lt * bl; tl = T.min(bl, Q - l0); tmc = ...
                    for n_blk in T.serial(N_tiles):                  # history（单块为主）
                        T.copy(ws_c[kernel_id, slot, pp, lt, ...], l1_c)
                        T.copy(prev_states[bz, bc_idx, bh, p0:p0+tp, ...], l1_state)
                        T.gemm(l1_c, l1_state, l0_acc, initC=(n_blk == 0),
                               b_transpose=True, size=[tmc, tn, tnp])
                    for s_blk in T.serial(lt + 1):                   # intra band 组装（含对角，单跳直载）
                        s0 = s_blk * bs; ts = T.min(bs, Q - s0)
                        T.copy(ws_lcb[kernel_id, slot, pp, lt, s_blk, 0:bl, 0:ts],
                               l1_band[0:bl, s0:s0+ts])              # GM→L1 直载（列偏移 dst，ts 裁剪）
                    T.gemm(l1_band, l1_x, l0_acc, initC=False,
                           size=[tmc, l0+tl, tnp])                  # band gemm（K=l0+tl 连续带）
                    T.copy(l0_acc[0:tl, 0:tp], out[bz, cs+l0:..., bh, p0:p0+tp])  # 写回
            with T.rs("PIPE_FIX"): T.sync_block_set(CONS(slot))      # 槽位已消费
    with T.Scope("Vector"):
        ...alloc UB + 核级 pen_const（arange/vsub/vmin/vmul 一次）...
        for task_id in T.serial(num_local_tasks):
            cid = ...; slot = task_id % 2
            with T.rs("PIPE_MTE2"): T.sync_block_wait(CONS(slot))  # 等槽位空闲（任务头！深度 2 节拍：
                                                                  # 任务 W 等任务 W−2 同槽消费完；0/1 走 prologue）
            for pp in T.serial(P_tiles):                           # 因子按 pp 重复生产（问题 3 方案 a）
                for i in T.serial(lt_count):                       # AIV 蛇形分片（R8）
                    lt = snake(i, subid, L_tiles)
                    ...c_scaled 链（copy/vcast/vmul 行广播/vcast/copy → ws_c[kernel_id, slot, pp, lt, ...]）...
                    for s_blk in T.serial(lt + 1):
                        ...diff 链（copy×3/vbrc×3/vsub/〔s_blk==lt: vadd(pen)〕/vexp/vmul×2/vcast/copy → ws_lcb[kernel_id, slot, pp, lt, s_blk, ...]）...
            with T.rs("PIPE_MTE3"): T.sync_block_set(READY(slot))  # 因子就绪（任务尾）
```

### 6.3 流水线优化

- **任务级深度 2 双域流水（主流水，R7.2）**：ws 双槽 + 4 flag + **Cube 前导 set**——Vector(T+1) 因子生产与 Cube(T) gemm 消费全重叠（**实测 −39.6%**，全链最大单点）。**深度 3 实测否决**（L2 局部性劣化：ws 工作集 9.2→13.8MB，mte2 每条 256→346ns——深度 2 是 L2 甜点，PL-1.18）。
- **x 任务×pp 级 L1 缓存 + Cube band 组装（R7.4）**：x 每 (task,pp) 一次装载 [Q,bp]（pp 循环内、lt 循环外）消跨 lt 2.5× 重读；band 组装使 gemm 10→4/任务（实测 −1.8~−4.3% 全域一致）。
- **块内交替**：Cube 域内 copy（MTE2）→ gemm（CUBE）→ copy（FIX）交替由 `T.rs` 管线区域静态排序，硬件按依赖并行发射；**L1 双缓冲软件流水实测否决**（v14：prefetch 的 MTE2 写与 gemm 的 MTE1 读在 L1 端口互拖，+12~19% 回退——910B2C 此 BiSheng 调度下 family blocked，PL-1.18 OP5）。
- **AIV 蛇形分片（R8）**：两 AIV 各承担 ~50% l-tile 因子计算（消重复执行，−28~−36%）；蛇形 vs 交错 ab_test tie，按负载均衡采纳（PL-1.13）。

### 6.4 尾块处理

**当 Q/P/N 不被 bl/bp/bn 整除时**（契约完备性路径；当前 workload 全整除）：`tl = T.min(bl, Q − l0)` 等尾块尺寸 → slice-form `T.copy`（src/dst 双侧显式 slice，extents 借用规则 docs T.copy.md §2.2.2 条 2–4）+ size-form `T.gemm(size=[tmc, tk, tnp])` + 分形钳位 `tmc = max(16, ceil16(tl))`、`tnp = max(16, ceil16(tp))`、K 维优先真实值（tn ≥ 32 时无 stale band；< 32 时钳位 32 + stale 列清零/单块路径）（GQA E7 形态）。diag 掩码的列上界（`j ≥ ts` 列守卫，band 域）；写回仅拷 `[0:tl, 0:tp]`。**host 侧不做 padding**（ascend-constraints.md §4）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**手动同步**（Expert 模式；`T.rs(pipe)` 管线区域 + `T.sync_block_set/wait(id)` flag 握手——与编程模式选型 §2.3 匹配）

### 7.2 同步点说明

**flag 预算**：4 个（READY0/READY1/CONS0/CONS1）≤ 15/核预算（CONST-flag-id-budget）✓。

**深度 2 任务流水协议（PL-1.12 形态 + Cube 前导 set）**：

| # | 同步点 | 位置（域/管线） | 语义 |
|---|--------|----------------|------|
| 0 | `T.sync_block_set(CONS0); T.sync_block_set(CONS1)` | Cube 域 prologue / `PIPE_FIX` | **前导 set**：两槽初始"空闲"——Vector T=0/1 的 slot-free wait 立即通过，免除运行时分支（PL-1.12） |
| 1 | `T.sync_block_wait(READY(slot))` | Cube 域任务头 / `PIPE_MTE2` | 等待 Vector 的本槽因子就绪 |
| 2 | `T.sync_block_set(CONS(slot))` | Cube 域任务尾 / `PIPE_FIX` | 本槽已消费，Vector 可覆写 |
| 3 | `T.sync_block_set(READY(slot))` | Vector 域任务尾 / `PIPE_MTE3` | 因子已写 ws，Cube 可读 |
| 4 | `T.sync_block_wait(CONS(slot))` | **Vector 域任务头** / `PIPE_MTE2`（v1 修订，问题 1——v0 误置于任务尾构成 WAR 竞争） | **等槽位空闲，前置于任务内任何 ws[slot] 写**：消费 prologue（任务 0/1）或 Cube 上一次同槽 set（任务 W 等待 Cube 完成任务 W−2 的 `set CONS(slot)`）后方才写 ws——深度 2 节拍：Vector 最多领先 Cube 1 个任务（领先者正在写的槽位是 Cube 两任务前已消费完的）。与 HEAD 旧 verified 终版 L356–362 逐位一致 |

**WAR/RAW 闭合论证（TASKDONE 类全 drain 屏障不设——PL-1.12 冗余审计方法）**：跨 task 的数据竞争逐对象闭合——① `ws[slot]` 的 Vector 写者（任务 W，任务头）晚于 `wait CONS(slot)`（蕴含 Cube 已完成任务 W−2 的读取——**同槽双任务间隔 = 深度 2**；任务 0/1 消费 prologue 前导令牌）；② `ws[slot]` 的 Cube 读序（读后 set CONS）先于 Vector 下一轮同 slot 覆写（同 flag 令牌序）；③ `out` 每任务写独立 (b,c,h) 切片（任务间零数据依赖）；④ 双 AIV 执行相同 set/wait 序列（one-set-multi-wait 已证，PL-1.13）——删除任务边界全 drain 屏障后，task T+1 的 Cube 消费与 task T 的 Vector 生产全重叠（实测 −39.6% 的来源之一）。

**引擎内依赖**：Cube 域内 copy→gemm→copy 的序由 `T.rs` 管线区域 + 单指令流程序顺序保证；Vector 域内 v-prefix 链的 RAW 由 AIV 串行指令流保证（UB in-place 链安全）。

**跨核**：任务间零数据依赖（每 (b,c,h) 任务写独立 out 切片）、无跨核归约、ws 按 (kernel_id, slot) 索引核间不相交——**无需任何核间同步**（flag 为引擎内握手，非核间）。

### 7.3 pass_configs 配置

```python
pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,  # PL-1.16（必关）
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,              # PL-1.16（必关）
}
# 依据：Expert 手动 CV split + 多 UB buffer 显式结构下，不关闭会使 planner 重排/重作用域
# buffer → codegen "cannot find variable" 崩溃（GQA 首证 + ssd 第二证，PL-1.16）。
# 附带效应：auto-multi-buffer 关闭后 UB 记账 ≈ 手工核算 ±2%（CONST-capacity ssd 数据点）。
```
---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。**直接采用 NPU 仓已移植的参考实现**（`examples/TileOPs/tileops/testing/mamba2_reference.py` 中的 `ssd_chunk_scan_fwd_ref` 函数，与 GPU 源仓同名参考文件逐行对应）：纯 fp32 PyTorch materialize（einsum 双路径 + tril 掩码，无因子化、无 dtype 量化、无向量化结构）。

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
| L0-1 | smoke (1,2,64,4,64,32,1) fp16 | 主 trace 精度（含 ghost 钳位域 8 任务/24 核 + Q=64 单 l-tile 纯对角 + AIV 分片退化域 + bn=32 贴分形下限） |
| L0-2 | (1,2,128,4,128,32,1) bf16 | bf16 路径 + P=128 双 p-tile + Q=128 双 l-tile（full-lower 路径激活） |
| L0-3 | full (2,4,64,8,64,64,2) fp16 | G=2 分组索引（g(h) = h//HPG）+ N=64 + 多 batch |
| L0-4 | full (2,2,64,4,64,32,2) bf16 | bf16 + G=2 + 欠载域（16 任务/24 核） |
| L0-5 | 边界 shape（非整除：Q=96/P=48/N=48 等契约内组合） | R6 尾块路径 + 分形钳位 + band 域 vselect+列守卫（契约完备性；当前 manifest 无此 shape，新增定向用例） |
| L0-6 | wrapper 契约（lru_cache 工厂、out shape/dtype = (B,C·Q,H,P) fp32、contiguous 输入、ws 分配、prev host cast） | 接口不变性（迁移 prompt 要求：接口参数不变，仅 threads 移除） |
| L0-7 | w2 (1,16,256,48,64,128,1) fp16 + bf16 | 代表 workload 全路径（深度 2 流水稳态 + bn=128 单块 + 蛇形分片非退化域） |

**等价性验证已前置**：`verify_equiv.py`（§1.6.1 结果表，2026-09-20 执行 OVERALL EQUIV_PASS）——设计期已机器拦截 K 侧因子移动（NaN/1e26 噪声）与对角 L-side 因子化（decay>88 NaN），并机器证明 band 域「vselect+列守卫」双保险必要性（单一因果 vselect 泄漏 stale NaN）。Stage 2 检视可重跑复核。

---

## 9. 风险点与注意事项

### 9.1 已知约束（技术约束检测结论汇总）

| 约束 | 结论 | 处置 |
|------|------|------|
| 三维 Kernel 不支持 | **触发**（源 `T.Kernel(·, B·C, H)` 三维） | §0.6 R1 一维 persistent + 串行（§5.5） |
| GPU 专用 API（threads/swizzle/sync_threads） | **触发** | §0.5 舍弃/替换（threads 在 wrapper 移除） |
| GEMM 非整除 | 契约可能 / 现役 shape 无 | R6 尾块 + L0-5 用例 |
| L0C 溢出 | 未触发（16KB ≤ 128KB） | §5.3 复核 |
| 分核适配 | **触发**（768–16384 逻辑核 ≫ 24） | §5.5 persistent 串行 + ghost 钳位 |
| UB 容量 | **贴限**（整除域 92% / band 域 94%，Expert 显式 alloc 记账 ±2%） | §4.5 消减序 + Stage 3 BishengIR 报错实测校准 |
| Developer 模式不可用 | **触发**（CG-2026-0010：persistent+gemm 混排运行时崩溃） | 本设计全程 Expert（用户指定 + 实证一致）；无 Developer 回退路径 |

### 9.2 常见错误与风险登记

| # | 风险 | 触发场景 | 影响 | 缓解 |
|---|------|----------|------|------|
| R-1 | **知识预注入 stale 条目的工具链重验风险**（E-1：CASE/PL-1.16/1.18/CG-0010/0011/0012 等条目戳为 1990aa9fe4，当前 4515de8） | Stage 3 编译/运行 | 结构性结论（双 Scope/ws 中继/深度 2/分片/块连续）大概率存活（生产代码在当前 HEAD 已集成）；硬边界类（UB 记账/支配性/gemm dst）可能随编译器变化 | Stage 3 首日清单：①pass_configs 行为验证（不关是否仍崩溃）；②UB 记账校准（§4.5）；③T.ceildiv 负数陷阱复验（GQA 注释同款）；④深度 2 流水 flag 语义；L0 全过 + w2 基线 profile 对照旧档 217.65µs ±10% |
| R-2 | **UB 贴限**（整除域 177.1KB/92.3%，band 域 180.7KB/94.1%） | Stage 3 编译 | `ub overflow` 编译失败 | §4.5 消减序（zero/lcb_16 复用 −16KB、c_f32/lcb_f32 错峰 −16KB、bs=32 band 变体）；BishengIR 报错文本实测校准（CONST-capacity 方法） |
| R-3 | **bf16 gemm 操作数直连**（GQA bf16 先例用 f32 ws 载体保守；本设计对齐源路径用 bf16 操作数） | bf16 用例精度 | bf16 用例超差 | L0-2/L0-4 门禁；超差 fallback：lcb/c_scaled 走 fp32→bf16 的 GQA 保守载体（gemm 仍 bf16，中转精度提升）；**禁止 c_scaled 链 bf16 化中转**（`T.vmul` bf16 ×，v7_cf16 编译失败先例，VP-2026-0002） |
| R-4 | **dA 非单调输入**（上游异常数据，违反 §0.1 假定） | 运行时数值 | 直接差分的上三角正指数真实分布 ~e^32、极端可至 fp32 上限（被惩罚/vselect 中和，不崩溃）；下三角 dA_l−dA_s 可为正（语义偏离设计假定） | 语义上源实现同样依赖该假定（§0.1）；不在 kernel 内加校验（与源一致）；风险登记 |
| R-5 | **结构地板（Cube mte2 段数墙）** | Stage 4 性能 | 旧档实测 1216 段/任务 × ~4ns 为 Expert ws 中继结构的物理流量（w4 稳态 83.5% 忙比）；五个削减方向实测穷尽否决（PL-1.18） | 接受为设计地板（final 217.65µs@w2 已逼近）；Stage 4 首轮 profile 复核当前工具链段数墙数值；如需突破须结构族更换（超出本算子迁移范围，列入 perf_feedback 天花板证据） |
| R-6 | **T.ceildiv 负数陷阱**（欠载域 ghost 任务） | 正确性（已缓解） | 越界任务 | ghost 钳位（§5.4 注意 1，GQA 实证绕法）；L0-1/L0-4 覆盖欠载域 |
| R-7 | **band 域列守卫的实现完备性** | Q%64≠0 契约 shape | stale 列 NaN 污染（机器证明：单一因果 vselect 不足） | R4 完整形态 `(j≤i)&(j<ts)` 双判据 + 装载 slice 只拷真实列；L0-5 定向用例（Q=96 等）；当前 workload 全整除不触发 |
| R-8 | **ws GM 工作集与 L2 交互** | 深度调优 | 深度 3+ 会劣化 L2（9.2→13.8MB 实测）；ws 分配失败（GM 碎片） | 深度锁定 2（§6.3）；ws 由 torch.empty 每次分配（host 侧，无碎片累积） |
| R-9 | **`T.gemm` vs `T.npuir_dot` 的 Expert 别名一致性** | Stage 3 编译 | docs 头注标注 T.gemm 为 [Developer mode]、T.npuir_dot 为 [Expert mode]，但 GQA/ssd 生产代码在 Expert 下直接用 T.gemm 正常 | 以生产实证为准（T.gemm 主选）；若当前工具链出现告警/异常，切换 T.npuir_dot 别名（参数同构，docs §1） |
| R-10 | **蛇形分片与 ws 槽位的正交性** | AIV 并发写 ws | 两 AIV 写区域冲突 | 分片维度（lt）与 ws 索引（slot, pp, lt）正交——同任务内两 AIV 处理不同 lt，ws 写区间不相交（PL-1.13）；退化域（L_tiles=1）重复计算 = bit-identical 覆写 |

### 9.3 特殊场景处理

- **非整除分块**：R6 尾块路径（slice + size gemm + 分形钳位 + band 域 vselect+列守卫）。
- **极小 shape**（Q=64 单 l-tile 纯对角、B·C·H=8 欠载）：ghost 钳位 + AIV 退化域重复计算均 bit-identical 安全；性能不敏感域（smoke 平区 30.75–32.17µs）。
- **混合精度**：fp16/bf16 双 dtype 同构 trace（仅 dtype 字符串分派，无结构分支）；exp 全程 fp32（vexp 不支持 bf16 的约束已规避）；因子链中转全程 fp32（v7_cf16 教训）。
- **深衰减 dA**（→ −128 量级 / 块内衰减 >88）：OPT-A 的 c_scaled 下溢行为与源 exp_dA_l 下溢一致（verify_equiv deep_decay PASS）；diag 上三角大指数被惩罚/vselect 中和（block_decay_gt88 案 FULL PASS / OPT-Y NaN 否决实证）。
- **G=1 与 G>1**：g(h) = h // HPG 索引计算覆盖两域（G=1 时 HPG=H，bg 恒 0；L0-3/L0-4 覆盖 G=2）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/
├── _ssd_chunk_scan_fwd_kernel.py   # 算子实现（Stage 3 产出：重写 tileops/kernels/mamba/ssd_chunk_scan/_s_s_d_chunk_scan_fwd_kernels.py 为 target="npuir" Expert 版）
├── DESIGN.md                       # 本设计文档
├── verify_equiv.py                 # §1.6.1 等价性机器验证脚本（已交付并执行，OVERALL EQUIV_PASS）
├── RETROSPECTIVE.md                # Stage 1 复盘（自进化钩子）
└── history_version/                # 历史版本备份（revision 时使用）
```

> 注：本算子属于 TileOPs harness 迁移模式——kernel 实现最终落位于 `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/_s_s_d_chunk_scan_fwd_kernels.py`（重写为 npuir Expert 版；文件名 `_s_s_d_chunk_scan_fwd_kernels` 源自 Stage 0 提取脚本对 "SSD" 的逐大写字母 snake_case 转换，**函数名仍为 `_ssd_chunk_scan_fwd_kernel`**，wrapper import 形如 `from ._s_s_d_chunk_scan_fwd_kernels import _ssd_chunk_scan_fwd_kernel`），wrapper/Kernel class（`ssd_chunk_scan.py`）与 op/tests/bench 基建已由 Stage 0 移植就位（default_config 待 Stage 3 按本设计 §5.2 更新 bn=min(128,N)）；独立目录 `examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/` 承载设计工件与验证脚本。

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档（迁移任务含 §0；Expert 模式 MixCV） |
| `verify_equiv.py` | ✅ 已交付并执行 | 等价性机器验证（OVERALL: EQUIV_PASS，2026-09-20，27 cases） |
| `_ssd_chunk_scan_fwd_kernel.py`（npuir Expert 实现） | ⬜ 待实现（Stage 3） | 按 §1.4/§3.3 伪代码重写迁移目标文件 |
| golden 函数 | ✅ 已就位（复用） | `examples/TileOPs/tileops/testing/mamba2_reference.py` 中的 `ssd_chunk_scan_fwd_ref`（§8.1） |
| 测试/性能基建 | ✅ 已就位（复用） | `tests/ops/test_mamba.py` / `benchmarks/ops/bench_mamba.py`（Stage 0 移植） |

### 10.3 命名规范

- 项目目录：`ssd_chunk_scan`（conductor 指定）；算子目录：`_ssd_chunk_scan_fwd_kernel`（snake_case，与迁移源函数同名）
- 实现文件：`_ssd_chunk_scan_fwd_kernel.py`；测试文件：`tests/ops/test_mamba.py`（TileOPs harness 既有路径）
- workspace 参数：`ws_c` / `ws_lcb`（显式 kernel 参数，host 分配）；flag 常量：`READY0/READY1/CONS0/CONS1`
