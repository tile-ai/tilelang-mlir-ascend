# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""PL-1.13-aiv-dup-subid-split.py -- Mix kernel 双 AIV 重复执行与 subid 分片（delta 形态）

条目: pattern-library/attention.md PL-1.13-aiv-dup-subid-split
关联条目（同任务结构族，共用本骨架）: elementwise.md PL-1.14-mte3-strided-ws
  （ws 块连续 vs band 布局）；layout.md PL-1.15-broadcast-perf-tax（vsub 列
  广播性能税）；traps-compiler.md TRAP-UB-dynsubview-dominance（task 级行
  buffer 动态 subview）；traps-runtime.md TRAP-L1-band-dst-tail-overrun
  （L1 band 组装尾块裁剪）。
现象/优化点: T.Kernel(n, is_npu=True) 的 T.Scope("Vector") 程序默认在 block 的
  两个 AIV 上**重复执行**（Mix Block Dim = 2 x Block Dim；profile 判据：两 AIV
  子块的 vec/mte ratio 完全相同而非减半）——AIV 产能 2x 浪费。用 subid 边界
  表达式把工作按 l-tile 交错/蛇形分给两个 AIV，实测 -28%~-36%（msprof op Task
  Duration，ssd_chunk_scan w2-w4）。
断言: delta 形态（效应只在完整 kernel 规模显现）——本文件只做 py_compile 语法
  校验；关键代码更改（慢: 双 AIV 全量重复 vs 快: subid 分片）以骨架表达。
首次验证: 2026-09-17 ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel Stage 4
  (tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C; perf_records
  round4 v8_aivsplit w2 323.01->211.85us -34.4%, round5 v9_snake tie)
重验记录: (追加式)

优化见解摘要（机制归因）:
  1) Bisheng mix kernel 的 Vector scope 在每个 block 的全部 AIV 上运行同一指
     令流；不按 subid 区分时两 AIV 各自计算并写入相同 ws（值相同故良性竞
     态，GQA dual-producer 同 flag 语义即源于此），但每 AIV 都承担 100% 向
     量工作。AIV 分片后每 AIV 只处理自己的 l-tile 子集（ws 按 lt 天然不相
     交），per-AIV vec/mte2/mte3 时间减半、AIV 侧 GM 流量减半（CONST-
     mte2-degradation 的流量维度），Vector critical path 从主导项降为次临
     界项。
  2) 分片必须用**边界表达式**而非 if 守卫（TRAP-tvm-parser-rules 的运行时分
     支风险）: for i in T.serial(lt_count): lt = <affine expr of i, subid>。
     交错式 lt = min(i*2+subid, L-1) 简单但负载不均（w2 的 4/6 s-block）；
     蛇形式 lt = i + subid + (i%2)*(L-2i-2*subid) 再双向 clamp 对 L_tiles=4
     恰为均衡 {0,3}/{1,2}（5/5），w3/w4 -1.8~-2.4%（w2 ab_test tie，
     run-state 双态内）。退化 L_tiles<=1 折叠为重复 lt=0（bit-identical 良
     性，与未分片行为一致）。
  3) 配套约束: 双 AIV 仍执行相同 sync_block_set/wait 序列（one-set-multi-wait
     已证安全）；分片维度必须与 ws 写区域正交（按 lt 分片 x ws 按 lt 索引）。
"""

# ---- 慢形态骨架（未分片：两 AIV 重复执行同一 Vector 程序）----
SLOW_SKELETON = """
import tilelang.language as T

def vector_scope_unsplit(L_tiles):
    with T.Scope("Vector"):
        for task_id in T.serial(num_local_tasks):
            for pp in T.serial(P_tiles):
                for lt in T.serial(L_tiles):          # 两 AIV 都跑满 L_tiles
                    produce_c_scaled(lt)              # -> ws_c[k, slot, pp, lt]
                    for s_blk in T.serial(lt + 1):
                        produce_lcb(lt, s_blk)        # -> ws_lcb[k, slot, pp, lt, s_blk]
            T.sync_block_set(ready_flag)              # 双 AIV 同 set（dual-producer）
"""

# ---- 快形态骨架（subid 蛇形分片：每 AIV 只处理自己的 l-tile 子集）----
FAST_SKELETON = """
import tilelang.language as T

def vector_scope_subid_split(L_tiles):
    with T.Scope("Vector"):
        lt_count = (L_tiles + 1) // 2                  # 每 AIV 的 l-tile 数
        for task_id in T.serial(num_local_tasks):
            for pp in T.serial(P_tiles):
                for i in T.serial(lt_count):
                    # 蛇形均衡: AIV0={0,L-1,2,...}, AIV1={1,L-2,3,...}
                    # (L_tiles=4 时为 {0,3}/{1,2}, s-block 权重 5/5 均分;
                    #  退化 L 折叠为重复 lt=0 -- bit-identical 良性)
                    lt = i + subid + (i % 2) * (L_tiles - 2 * i - 2 * subid)
                    lt = T.min(T.max(lt, 0), L_tiles - 1)
                    produce_c_scaled(lt)              # ws 按 lt 天然不相交
                    for s_blk in T.serial(lt + 1):
                        produce_lcb(lt, s_blk)
            T.sync_block_set(ready_flag)              # 仍双 AIV 同 set
"""

# py_compile 自检（delta 形态：语法级）
if __name__ == "__main__":
    import py_compile
    import tempfile
    import os

    for name, skeleton in (("slow", SLOW_SKELETON), ("fast", FAST_SKELETON)):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("def _stub():\n    pass\n" if name == "slow" else "")
            f.write(skeleton)
            path = f.name
        try:
            py_compile.compile(path, doraise=True)
            print(f"[PL-1.13] {name} skeleton: py_compile PASS")
        except py_compile.PyCompileError as e:
            raise SystemExit(f"[PL-1.13] {name} skeleton FAILED: {e}") from e
        finally:
            os.unlink(path)
