---
id: pattern/decode-split-kv-fused-barrier
layer: pattern
arch: [Ascend950]
kind: hw-pattern
status: active
verified_scope: tilelang-ascend
---

# Decode Split-KV 单核融合屏障、物理核对齐与 DFU 流水优化

## 适用症状与触发条件

- 高长上下文 Decode（典型如 FP8 MLA / GQA Decode，长 `Sk`、小 `Sq`、大 `D` + `RoPE`）呈现以下特征：
  - 单请求并行度不足，无法填满全卡物理核，或分两段 kernel（`partial` + `merge`）引入额外启动与同步空泡；
  - 网格块数与物理核数不匹配导致多轮尾波（例如 28 核卡跑 32 block，第二轮严重空载，耗时接近翻倍）；
  - 单核内 `PV` 吞吐、Vector Softmax 与输出更新未充分重叠，纯 Cube DFU 偏低。

## 核心物理机理与组合手法

1. **Split-KV + 单次 MIX 全核屏障融合（`SyncAll`）**
   - 将长 `Sk` 切分为 `K` 个 split，每个 split 产出 FP32 部分结果写入 workspace，随后在**同一次 MIX kernel 启动内**通过全核屏障（如 `SyncAll`）同步并直接由 AIV 完成归并写出，消除两段 kernel 之间的 launch 间隙（实测可将官方级实现耗时压降 >35%）。
2. **严格对齐物理核数，消除尾波并杜绝死锁**
   - **死锁铁律**：只要 kernel 内使用了全核屏障，**`grid` 绝对不能超过硬件可并发调度的物理核数**（例如 28 核卡上发射 `grid32` 会因等待未被调度的 block 而死锁）。
   - **整除分配**：选择 split 数 `K` 使总任务数 `B × K` 尽可能整除物理核数（例如 `B=8` 在 28 核卡上选 `K=7 → 56` 任务恰好 2 满轮，显著快于 `K=4 → 32` 任务产生 4 个残余任务的第二轮尾波）。
3. **非 2 的幂 Head 精确打包，不盲目补零**
   - 例如 `Hq=24` 直接映射为 Cube `M=48` 与 Vector `24` 行（双 AIV 各 12 行融合归并），不补齐到 32 head，直接省去 25% 无效计算与搬运。
4. **Decode 核内高 DFU 流水组合**
   - **大列宽 PV + 双缓冲槽**：将 `PV` 切分为 `N128` 级列块，配合双 `L0C` / 双 `UB` 槽隐藏写回与转换延迟；
   - **Softmax 与输出更新交错 + 两行展开**：Softmax 按两 head 展开并插入输出更新的上下半部之间，SIMD 输出更新每次展开 2 个独立行，隐藏指令延迟；
   - **匹配 `MMAD/FIX` UnitFlag + N-first 顺序**：减少冗余显式同步；
   - **下一页 KV 提前预取**：在当前页 `QK` 启动前发起下一页 `GM → L1` KV 装载，保持 4 槽生命周期握手。

## 不适用边界

- 若物理核数在运行时未知或无法保证所有 block 同时驻留硬件，**不得使用全核屏障融合**，应回退为双 kernel（`partial` + `merge`）避免死锁。
- 若 `Sk` 很短、`B` 已经足够填满所有物理核，强行 Split-KV 只会增加 workspace 读写和归并开销。

## 验证清单

- 核查不同物理核数卡（如 28 核 vs 32 核）各自显式匹配 `grid`，禁止跨核数混跑融合版；不同物理核数的实卡性能与精度各自独立验收，不得相互外推。
- 覆盖空闲核、页尾、ragged 变长、多任务回环与满载主形状；核对所有被消费的 partial 行、workspace 写入与前后 guard 字节。
- 注意单次 launch 计时与稳态多轮计时分开报告，混合核 AIC/AIV 记录取 `max(start+dur) - min(start)`，绝不将重叠时长相加。
