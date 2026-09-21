# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""PL-1.18-floor2-wins.py -- SSD chunk scan 二轮调优三项胜出结构（delta 形态）

条目: pattern-library/attention.md PL-1.18-ssd-steady-structure-floor
  〔2026-09-21 二轮调优 update〕
现象/优化点: 旧段数墙记账漏算 prev_states 的 lt 循环重读（×4）；且 gemm 链
  经单 L0C acc 串行化、Vector 链存在 lt 不变广播的逐 s-block 重做。三项胜出
  （慢→快）：①prevhoist（prev 载装提升出 lt 循环，mte2 字节 −20%）②l0c2x
  （L0C acc 乒乓配对循环，gemm 链 2 深重叠）③vbrchoist（vbrc hoist 干净形
  态，净零 UB）。实测累计 w2 −9.0% / w3 −6.2% / w4 −6.9%（msprof op Task
  Duration，median of 20 + ab_test 交错协议）。
断言: delta 形态（效应只在完整 kernel 规模显现）——本文件只做 py_compile 语
  法校验；关键代码更改以 before/after 骨架表达。
首次验证: 2026-09-21 ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-4515de8 重跑
  任务 Stage 4（tilelang 0.1.2+15ad002b3d〔与 4515de8 同源，git diff 零改动〕
  / CANN 8.5.0 / Ascend910B2C；perf_records round1/2/4 + final：w2
  223.53→215.88→210.70→203.43→204.33µs）

优化见解摘要（机制归因）:
  1) prevhoist: T.copy(prev_states[...]) 的索引只含 task/pp/n_blk 维——对 lt
     循环不变，却写在 lt 循环体内 = 每任务 L_tiles 次重读（Q=256 时 4×64KB
     中 48KB 冗余 = Cube mte2 的 20%）。发现方法 = aic_mte2_instructions
     (19.0/任务) 对照 kernel 拷贝语句清单逐项互证（指令数是装载冗余的机械
     化探测器）。注意节省的 mte2 字节仅 ~20% 兑现为墙钟——忙比受 L1 端口
     轮转约束，须同时解除 gemm 链串行（见 2）才能部分兑现。
  2) l0c2x: 单 l0_acc 下每 lt 的 [hist initC 覆写 → band 累加 → out fixpipe]
     经同一 L0C buffer 形成 WAR/WAW 串行链（hist(lt+1) 等 out(lt)）。配对循
     环 for jp in T.serial(L_pairs) 内 lt=2j 用 l0_acc_a、lt=2j+1 用
     l0_acc_b（相邻 lt 交替），gemm/mte1/fixpipe 链 2 深重叠（cube_wait
     0.897→0.836）。buffer 选择必须在 trace 期完成（lt 为运行时循环变量，
     三元式被 TVM 解析器拒收——TRAP-tvm-parser-rules；运行时 if 则是调度
     毒，见 4）。lt 间本就是独立计算（l0_acc 仅作 per-lt 暂存），配对不改数
     值。奇数尾块用纯名 Python bool 无-else if（if L_TILES_ODD:）在 trace
     期折叠。
  3) vbrchoist: vbrc(dA_l_col→[bl,bs]) 是 lt 不变量却在 s_blk 循环内逐块重
     做（每 AIV 每任务 5→2 次）。旧任务 v11 的 hoist 失败归因 vsub dst=src2
     alias 形态税（+17~20%），非 hoist 本身——干净形态：hoist 后 vsub 写
     fresh diff_mat（dst≠src1/src2），diff_mat 在最后一次读
     vmul(cb_f32,diff_mat,lcb_f32) 之后被复用为 dt 广播目标
     vbrc(dt_s_row→diff_mat)——buffer 改名净增量为零（dA_l_mat/dt_mat →
     dA_l_hoist/diff_mat），UB 不变。
  4) 反向教训（同轮实测）: 运行时 if（哪怕 once-true 守卫）进入 Cube 热循环
     破坏 BiSheng 跨迭代调度（w4 +15.65%）；x 的 32KB 大块拷贝在 mte2 流头
     是局部最优（静态后移也 +3.1~+10.5%）；L1 操作数双缓冲在任一结构状态
     下均因 MTE2 写/MTE1 读端口互拖回退（二次独立实证）。
"""

# ---- 慢形态骨架（baseline: prev 逐 lt 重读 + 单 l0_acc 串行 + vbrc 逐 s-block）----
SLOW_SKELETON = """
# with T.Scope("Cube"):
#     l0_acc = T.alloc_L0C([bl, bp], accum_dtype)          # 单 acc
#     for task_id in T.serial(num_local_tasks):
#         for pp in T.serial(P_tiles):
#             T.copy(x[...], l1_x[...])                     # x 在流头（局部最优，勿动）
#             for lt in T.serial(L_tiles):
#                 for n_blk in T.serial(N_tiles):
#                     T.copy(ws_c[k, slot, pp, lt, ...], l1_c[...])
#                     T.copy(prev_states[bz, bc, bh, p0, n0], l1_state[...])   # ← lt 不变却每 lt 重读
#                     T.gemm(l1_c, l1_state, l0_acc, initC=(n_blk==0), b_transpose=True, ...)
#                 for s_blk in T.serial(lt + 1):
#                     T.copy(ws_lcb[...], l1_band[...])
#                 T.gemm(l1_band, l1_x, l0_acc, initC=False, ...)             # ← 与 hist 经同一 l0_acc 串行
#                 T.copy(l0_acc[...], out[...])
"""

# ---- 快形态骨架（final: prevhoist + L0C acc 乒乓 + 干净 vbrc hoist）----
FAST_SKELETON = """
# with T.Scope("Cube"):
#     l0_acc_a = T.alloc_L0C([bl, bp], accum_dtype)        # acc 乒乓（2x16KB <= L0C 128KB）
#     l0_acc_b = T.alloc_L0C([bl, bp], accum_dtype)
#     for task_id in T.serial(num_local_tasks):
#         for pp in T.serial(P_tiles):
#             T.copy(x[...], l1_x[...])                     # x 保持流头
#             if N_tiles == 1:                              # trace-time Python 分支（非运行时 if）
#                 T.copy(prev_states[bz, bc, bh, p0, 0:bn], l1_state[0:tp, 0:bn])   # ← hoist: 每 (task,pp) 一次
#                 for jp in T.serial(L_pairs):              # 配对循环：lt=2j→A / lt=2j+1→B
#                     lt0 = jp * 2                          # even body: ws_c(lt0) copy ->
#                     ...                                   #   hist(l0_acc_a) -> band(l0_acc_a) -> out
#                     lt1 = jp * 2 + 1                      # odd body: 同构，用 l0_acc_b
#                     ...
#                 if L_TILES_ODD:                           # 纯名 bool 无-else if，trace 折叠
#                     ...                                   # tail body on l0_acc_a
#
# with T.Scope("Vector"):
#     dA_l_hoist = T.alloc_ub((bl, bs), accum_dtype)       # 改名自 dA_l_mat（净零 UB）
#     diff_mat = T.alloc_ub((bl, bs), accum_dtype)         # 改名自 dt_mat
#     for i in T.serial(lt_count):
#         lt = snake(i, subid, L_tiles)
#         T.vbrc(dA_l_col, dA_l_hoist)                     # ← hoist: 每 lt 一次（原每 s-block）
#         for s_blk in T.serial(lt + 1):
#             T.vbrc(dA_s_row, dA_s_mat)
#             T.vsub(dA_l_hoist, dA_s_mat, diff_mat)        # fresh dst，无 alias（旧 v11 的税源）
#             if s_blk == lt:
#                 T.vadd(diff_mat, pen_const, diff_mat)
#             T.vexp(diff_mat, diff_mat)
#             T.vmul(cb_f32, diff_mat, lcb_f32)
#             T.vbrc(dt_s_row, diff_mat)                    # ← 死后复用：diff_mat 即原 dt_mat
#             T.vmul(lcb_f32, diff_mat, lcb_f32)
#             T.vcast(lcb_f32, lcb_16, round_mode="rint")
"""
