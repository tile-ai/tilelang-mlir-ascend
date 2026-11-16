# Counts：FLOPS运算量与分母口径

本页回答两个问题：一个算子该记多少运算量，以及该除以哪个峰值。
建模语义与实测方法见 [roofline.md](roofline.md)，峰值数值见 [peaks.md](peaks.md)，MFU 的分子范围见 [mfu.md](mfu.md)。

## 计数规则

| 操作 | 语义计数 | 计入 | 可用的分母 |
|---|---|---|---|
| `a+b`、`a-b`、`-a` | 1 /元素 | useful 与 executed | Vector 标称峰值（dtype 匹配） |
| `a*b` | 1 /元素 | useful 与 executed | Vector 标称峰值 |
| `a*b+c`（FMA） | 2 /元素 | useful 与 executed | Vector 标称峰值 |
| `max/min/abs/cmp/select` | 1 次比较或选择 | 单列 `op`，不并入浮点 MFU | 无逐指令峰值，报 op/s |
| dtype 转换（fp32↔bf16 等） | 1 /元素 | 单列 | 不并入 MFU |
| `a/b` | 1 次语义除法 | 单列 | 无官方峰值：报 GOP/s，或用下方实测分母 |
| `exp/log/sqrt/rsqrt/tanh/sigmoid/reciprocal` | 1 次语义特殊操作 | 单列 | 无官方峰值：报 GOP/s，或用下方实测分母 |
| 整数 MAC（INT8） | 2 ops /MAC | useful 与 executed | Cube INT8 峰值（TOPS），称整数利用率 |
| 纯搬运、copy | 0 | 只计 bytes | 无 |

- **FMA 记 2 与 mul+add 各记 1 之和相同**，所以 FLOPs 不因是否融合而变化；差异只体现在峰值口径（FMA 峰值 vs 标量流水）与指令数上。不能靠"融合更多 FMA"提高分子。
- 同一份工作量只用一种计数器：把 FMA 记 2 与把它记 1 MAC 混用会整体差 2 倍。
- 归约 `sum(x[0..N-1])` 为 `N-1` 次加法，`max` 归约为 `N-1` 次比较。
- `F_useful` 扣除 padding、越界 tile、被跳过的 mask 与重计算；`F_executed` 按实际执行统计。两者用同一张表，只是计数对象不同。
- epilogue/FIXPIPE 的 bias、激活、量化/反量化属于 Vector 或搬运口径，不加入 Cube 分子。
- 工具给出的 expanded/weighted OPS 先按版本核验 FMA 权重与特殊函数权重，核验后才可作为 executed 计数。

## 常见场景运算量

`B` 为 batch（或等效的多 head/多实例数），方阵序列长度 `S`，还原维度 `K`。

| 场景 | 运算量 | 分母 | 备注 |
|---|---|---|---|
| 无掩码 GEMM `M×N×K` | `2·B·M·N·K` | Cube(dtype) | M/N/K 用语义真实尺寸，排除 padding |
| 输出掩码 GEMM | `2·K·N_valid`，`N_valid=Σ n_i`（含 batch）；等价于 `2·B·P·K`（`P` 为有效输出位置数） | Cube | 前提是每个有效输出仍做满 `K` 次归约 |
| 归约维掩码（PV、变长 KV、稀疏 attention） | `2·Σ` 有效 `(i,j,k)` 三元组，即 `2·valid_macs` | Cube | **不能写成 `2·K·N_valid`** |
| causal `QK^T`，`S×d`（`j≤i`） | `2·d·P = d·S(S+1)`；大 S 约 `S²d` | Cube | `P=S(S+1)/2`；"全量的一半"只是 `(S+1)/(2S)` 近似 |
| causal `PV`，`S×d` | `d·S(S+1)` | Cube | mask 在归约维，走上一条 |
| 滑窗 attention（`i-w ≤ j ≤ i`） | `≈2·d·S·(w+1)`，精确按窗口计数 | Cube | 不能复用 `S(S+1)/2` |
| Attention 前向（`H` 个 head） | `2·B·H·P·(Dqk+Dv)` | Cube | 不含 softmax 与缩放 |
| softmax 每行 `N` | `≈4N`（max、sub、exp、div）；用倒数乘约 `3N` | Vector 特殊操作 | 无匹配峰值，报 GOP/s |
| LayerNorm/RMSNorm 每元素 | `≈5–8N`，随实现 | Vector | 含均值、方差、rsqrt、缩放 |
| GELU/SiLU 每元素 | `≈5–10N`，随近似路径 | Vector | 分母须匹配实际指令路径 |
| Conv2d | `2·N·K·C·R·S·Hout·Wout` | Cube | 隐式 GEMM 口径 |
| 逐元素融合 `y=a*x+b` | `2N` | Vector | |
| 纯 copy | 0 | — | 只算 BWU |
| N 卡（同构同精度） | 单卡工作量 × N | 各卡峰值同步求和 | 必须同一时间窗；不累加并行卡墙钟 |

**causal 的计数约定**：按 `j≤i+offset` 精确计数，方阵默认 `P=L(L+1)/2`，不要直接把全量除以 2；
非方阵通常按右下对齐取 `offset=N-M`，先核对语义。各 batch/head 实例的有效数不同时拆条目。

### 两个反例

- **归约维掩码套输出掩码公式**：PV，`S=1024`、`d=64`。正确 `d·S(S+1)=6.717e7`；
  若按 `2·K·N_valid`（`K=S`、`N_valid=S(S+1)/2`）得 `1.075e9`，高估 `S/d=16` 倍。
- **把 causal 直接除以 2**：`S=8`、`d=64`。精确 `d·S(S+1)=4608`，`S²d=4096`，已偏差 12.5%；
  S 越大才越接近一半。

三个可复算的检查：`S=4,d=2` 的单头 causal `QK^T` 为 `2·10·2=40`；`PV` 取 `valid_macs=30`、`d=3` 为 `2·30·3=60`；
合计 `100` 次有效 Cube 运算（与 mfu.md 的示例一致）。

## 算子 → 分母

| 指令类别 | 分母 | 现状 |
|---|---|---|
| Cube fp16/bf16、tf32、fp8/mxfp8/hif8/mxfp4、int8 | 对应档位 Cube 峰值 | [peaks.json](../assets/peaks.json) 已收录 |
| Cube **fp32** | 未定义 | 白皮书无 FP32 Cube 行；走 TF32、分解还是 SIMT 须按实现声明，未核实留空 |
| Cube int4/int16/int32 | 未核实 | 不以倍乘补齐 |
| Vector fp16/bf16/fp32/int8/int16/int32/int64 | Vector 标称峰值 | 已有；SIMD/SIMT 与特殊指令未拆分 |
| Vector 特殊函数（exp 等） | 官方无；可用实测 | 见下节“实例”；没有实测时只能报 GOP/s |
| FIXPIPE/epilogue、量化 | 归 Vector 或搬运 | 不并入 Cube |

未证实的组合不通过倍乘补齐；缺少分母时保留符号表达式与实际吞吐，不输出百分比。

## 实例：SIMT 实测分母（950DT）

用 `bench/` 的探针在本机测得（Ascend950DT_9592，36 AIC / 72 AIV，1650 MHz，CANN 9.3.0，TileLang `de44ac0a`）。
方法、三轮 rounds 与几何交叉校验见 [bench/RESULTS-950dt-simt-fp32.md](../bench/RESULTS-950dt-simt-fp32.md)。

| 操作 | 路径 | cycles / 128 元素 | op/cycle/AIV | 实测吞吐 | 相对 FMA 成本 |
|---|---|---:|---:|---:|---:|
| FMA（基线） | SIMT VF | 1.16 | 110.6 | 26.28 TFLOPS | 1.0 |
| `sqrt` | SIMT VF + `sqrtf` | 5.60 | 22.8 | 2.71 Tops/s | 4.8 |
| `1/x` | SIMT VF + 除法 | 5.62 | 22.8 | 2.70 Tops/s | 4.9 |
| `exp` | SIMT VF + `expf` | 5.99 | 21.4 | 2.54 Tops/s | 5.2 |
| `vexpdif`（等价 `exp(x-max)`） | SIMT VF + `expf` | 3.42 | 37.4 | 4.45 Tops/s | 3.0 |
| `rsqrt` | SIMT VF + `rsqrtf` | 12.75 | 10.0 | 1.19 Tops/s | 11.0 |
| `log` | SIMT VF + `logf` | 21.90 | 5.9 | 0.70 Tops/s | 18.9 |
| `sigmoid` | 复合 `exp→add→div` | 25.36 | 5.1 | 0.60 Tops/s | 21.9 |

- 利用率 = 同路径的实测 op/s ÷ 上表吞吐；**不要**拿 TFLOPS 名义峰值去除 exp 的 op/s，两者不同单位。
- 这是 `basis=measured` 的**本机**值，不是官方峰值；换设备、CANN/TileLang 或频率都要重测，并在报告里写清范围与日期。
- 覆盖的是 **SIMT VF + libm 序列**：`aiv_vec_sfu_ratio` 全程 0，没有用上 SIMD 的 `vexp/vln` 指令；SIMD 路径需另测，不能把这里的数当 SFU 硬件上限。
- 不确定度约 ±10%：重复运行与第二几何相差 <1%，但 rounds 扫描仍单调上升（18.7 / 21.4 / 22.8 @ 64/128/256）；`sigmoid` 的串行链未打满，当前值是下界。
- **表达式形式会改变调度**：同一语义的 exp 三种写法在本探针里是 21.3 / 37.4 / 52.8 op/cycle
  （`acc += exp(v)`、`exp(v-0.5)`、`exp(v)-0.5`）。建模时按目标 kernel 的实际写法取，跨写法套用会差 2 倍；
  上表 `vexpdif` 取 `exp(v-max)` 形式，重复与 `exp(v-1)` 一致（37.4 / 37.3）。
- 融合 SIMD 指令 `vexpdif` 未能单独隔离：SIMD 探针的 control/`vexp`/`vexpdif` 三个变体为
  455.10 / 455.11 / 455.13 us，被每 chunk 的同步与搬运开销主导，差值在噪声内；
  上表 `vexpdif` 行是 **SIMT 等价实现**的值，不是 SIMD 指令峰值。
- 对照“只做一次 exp 的朴素 kernel”：其 AI 仅 0.125 op/B，远低于 exp 拐点（≈0.64 op/B），实测 3.41 TB/s 且判 `latency bound: memory caused`，测到的是主存带宽而非 exp。

## 与工具对账

| 需要的量 | 优先来源 | 核对点 |
|---|---|---|
| `F_executed` | 工具 op summary 的 OPS 或指令计数 | FMA 是否按 2、特殊函数是否加权、是否含 padding；按版本核验 |
| `F_useful` | 算法与源码结构推导 | 不由工具反推；`q=F_useful/F_executed` 只在同口径下有意义 |
| 有效位置 `P` 或 `valid_macs` | 索引/掩码语义 | causal 的 `offset`；非方阵右下对齐；各实例是否拆分 |
| 分母 | [hardware.md](hardware.md) 的档位 + 本页映射 | 引擎、dtype、指令类别、参与核数一致 |

工具计数与源码计数不能映射到同一种操作定义时，分别报告，不强行求 `q`。

## 最小计算单元全量清单（算子库盘点）

盘点范围：`tilelang/examples/ascend`、`tilelang/testing/ascend`、`TileKernels/tile_kernels`（`*_asc.py`）。
状态列：✅=已有计数规则；🔴=FLOPS 折算待实测；⬜=不计入浮点 FLOPS（报 op/s 或 bytes）。

| 类别 | 最小计算单元 | API/指令 | dtype | 计数口径 | 状态 |
|---|---|---|---|---|---|
| Cube | 普通 GEMM | `T.gemm` | bf16/fp16/fp32(hf32/tf32)/fp8_e4m3fn/int8 | `2·M·N·K` | ✅ |
| Cube | 块缩放 GEMM | `T.gemm_blockscaled` | fp8_e4m3fn/fp4_e2m1fn + int16 scale | `2·M·N·K` | ✅ |
| Vector | 加/减/取负 | `vadd/vsub/vadds/vsubs/vneg` | fp32/bf16/fp16/int* | 1/元素 | ✅ |
| Vector | 乘 | `vmul/vmuls` | 同上 | 1/元素 | ✅ |
| Vector | 除 | `vdiv` | 同上 | 1 语义除法 | ✅(GOP/s) |
| Vector | 乘加 FMA | `vmula/vmadd/vaxpy` | 同上 | 2/元素 | ✅ |
| Vector | 指数 | `vexp/T.exp` | fp32/bf16 | 1/元素 | ✅(T.exp 实测) |
| Vector | 指数(带 max 偏置) | `vexpdif` | fp32 | 1 次融合操作 = 1 sub + 1 exp | ✅(SIMT 等价实测；SIMD 指令未隔离) |
| Vector | 对数 | `vln` | fp32 | 1/元素 | ✅(T.log 实测) |
| Vector | 平方根 | `vsqrt/T.sqrt` | fp32/bf16 | 1/元素 | ✅(T.sqrt 实测) |
| Vector | 平方根倒数 | `T.rsqrt` / `1÷vsqrt` | fp32/bf16 | 1/元素 | ✅(T.rsqrt 实测) |
| Vector | 倒数 | `vdiv(1,x)` | fp32/bf16 | 1 语义除法 | ✅(1/x 实测) |
| Vector | sigmoid | `T.sigmoid` / `1/(1+exp(-x))` | fp32/bf16 | 原生 1；复合 exp+add+div | ✅(复合实测·下界) |
| Vector | tanh / erf | 当前未出现 | — | 预留 | 🔴 |
| Vector | (L)ReLU/PReLU | `vrelu/vlrelu/vprelu` | fp32/bf16 | 1 选择 | ✅ |
| 比较 | max/min | `vmax/vmin/vmaxs/vmins` | 同 dtype | 1 比较 | ✅(op/s) |
| 比较 | abs/绝对差 | `vabs` / `vabsdif` | 同 dtype | 1 | ✅(op/s) |
| 比较 | 比较谓词 | `vcmp/vcmps`(gt/ge/lt/le/eq/ne) | 同 dtype | 1 比较 | ✅(op/s) |
| 比较 | 选择/clamp | `vsel/vselr/T.Select/T.if_then_else` | 同 dtype | 1 选择 | ✅(op/s) |
| 转换 | 类型转换 | `vcvt/T.cast` | bf16↔fp32、fp32↔fp8_e4m3/e5m2、fp4_e2m1、fp↔int、int32↔int64 | 1/元素 | ✅(单列) |
| 转换 | 位级重解释 | `T.reinterpret` | 任意 | BOP | ⬜ |
| 转换 | 随机舍入 cast | 整数 hash+进位（fp32→bf16） | 整数 | BOP | ⬜ |
| 归约 | sum 归约 | `vcadd/vcgadd/T.reduce_sum/reducer(sum)/warp_reduce_sum` | 同 dtype | N−1 次加 | ✅ |
| 归约 | max/min 归约 | `vcmax/vcgmax/vcmin/vcgmin/reduce_max/warp_reduce_max` | 同 dtype | N−1 次比较 | ✅ |
| 位运算 | 与/或/异或/非 | `vand/vor/vxor/vnot` | int*/f16(转位) | BOP | ⬜ |
| 位运算 | 移位 | `vshrs/vshls` | 整数 | BOP | ⬜ |
| 位运算 | 打包/交织/解交织 | `vpack/vintlv/vdintlv` | 各种 | 搬运/BOP | ⬜ |
| 数据搬 | gather/scatter/copy | `vgather2/vscatter/T.copy/T.dual_copy` | 各种 | 0(只计 bytes) | ⬜ |
| 整数 | hash(乘+异或+取模) | `vmul/vxor/%` | int32/int64 | 整数 op | ⬜ |
| 整数 | 带进位加/跨对加 | `vaddc` / `vcpadd` | int* | 整数 op | ⬜ |
| 随机 | RNG 生成 | `rng_init/rng_rand_float`(normal/uniform) | fp32 | 非常规，需测 | 🔴 |
| 特殊 | 直方图分箱 | `dhistv2/chistv2` | u8/u16 | 非常规，需测 | 🔴 |
| 特殊 | 全局原子 | `atomic add/max/min` | 同 dtype | 单列 | ⬜ |
| 特殊 | 量化 scale | round_sf 位操作 / `amax+vdiv` | — | 位运算或 1 除 | ⬜ |
| 特殊 | topk 迭代 | `vmax+vcmax+vcmin+mask` | — | 复合 | ⬜ |

### 需实测的特殊函数折算

`exp`、`ln`(log)、`sqrt`、`rsqrt`、`reciprocal`、`sigmoid` 的 **fp32 SIMT 路径**已实测，数值见上文“实例”。
仍待补的项：bf16 各组、`vexpdif`、`tanh/erf`、SIMD（`vexp/vln`）指令路径。
无 FLOPS 定义的非常规原语（`rng`、`dhistv2/chistv2`、整数 hash、`vaddc/vcpadd`、原子）建议单独报 op/s，不并入 MFU 分子（除非占据某算子的主导项）。
