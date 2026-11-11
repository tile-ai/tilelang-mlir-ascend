# torch_npu 性能采集方法

## 执行位置与输入

输入由 TASK 固定：case/input、候选和构建身份、完整调用边界、warmup/repeats、缓存策略、频率规则、统计量、baseline。不得在测量后改变口径。配套脚本支持单 NPU、设备执行区间并集的指标；解析器支持 128 字节 ffts 记录布局。使用前须核对实际工具链的记录格式与字段含义；不把该布局当作稳定公开 ABI，不据此指定芯片或 CANN 版本。

整个设备进程在约定的 npu-guard 下运行，包含 torch_npu 导入、编译、输入搬运、采集和同步。Docker 保持 host 完整锁目录、观察入口及进程可见性。排队与取得锁后的运行分别计时，本地控制端时限覆盖两者及收尾，保留可查询的作业和退出记录。

`torch_npu.profiler.profile` 必须在实际执行 kernel 的进程内启动和停止。DeepJIT 模式将采集逻辑放到受 guard 保护的 worker，或调用已确认在 worker 内开启 profiler 的接口；在本地仅包围 SSH/RPC 调用不等于采到了远端 NPU。允许在私有目录写 worker 适配，不修改共享环境或候选实现。

## 完整调用适配

准备一个无参数 `call()`，每次执行完整算子：所有必需初始化、各阶段与后处理按正确 stream 顺序运行。编译、输入上传及可复用分配放采样窗口外；若绑定协议要求这些计入则另按协议实现，不能使用本脚本的默认边界冒充。保持缓冲区直到最终同步；有原地/累加状态时按协议重置。记录阶段到实际入口/二进制/launch 的映射。

从生成函数属性和编译器 wrapper 提取完整 launch 契约，包括 grid、参数顺序、mix mode、stream、动态共享/local memory 字节数及所需 launch API。目标 kernel 和 flush kernel 都要核对。仅把源码编成二进制再按入口名启动可能遗漏运行参数；不能用“同一个源码”代替这个检查。TileLang 的 `dyn_shared_memory_buf` 需确认如何映射到所选 runtime：若使用 DeepJIT，核对 legacy/V2 分支及 `local_memory_size` 是否真正生效，不能无依据地默认 legacy/0。先保存该映射，再执行独立 flush 校验。

不得只选择最快的一个 kernel，也不得把多阶段 duration 简单求和。相同入口名的多个阶段需用 task 数量、launch 顺序、stream/关联标识和构建清单核对，不能用同名筛选证明完整性。debug canary 上传、输出下载、CPU 比较不放性能窗口中；目标自身的必要初始化不能移出。

## 采样步骤

1. 核对候选身份与已通过精度证据。在本次执行适配上先做完整输出预检；改变 launch/缓冲区/调用序列须重新验证受影响精度。
2. 明确 `flush()`、`synchronize()`、`frequency()` 和 guard 所有权检查。flush 按协议处理真实大小的缓冲区；不能用空函数表示已清缓存。频率返回协议指定组件的实际读数；频率查询失败或缺失不得填默认值。
3. 预编译所有目标和 flush kernel。调用 flush 并同步，再做一次完整调用并同步，重复 warmup 次。预热在 profiler 外，不计入正式样本。
4. 每个正式样本：flush → 同步 → 读频率 → 新建独立 profiler session → 完整 `call()` → 同步 → `prof.step()` → 关闭 session → 再读频率。采样配置为 NPU activity、schedule(wait=0,warmup=0,active=1,repeat=1)、Level1/PipeUtilization、data_simplification=False、analyse_flag=False，保存原始记录。
5. 解析本 session 的原始 ffts 记录，核对每个必要阶段/初始化均在窗口内、flush 不在窗口内；验证原始记录属于同一设备/次调用。对设备时间区间排序并合并重叠，得到区间并集长度；host 间隙、分配墙钟和 profiler 总运行时间不是这个指标。
6. 保存每个样本及两次频率观测，按冻结规则计算均值、median、范围、样本标准差。频率不符、记录缺失、覆盖不足时保留失败证据并停止该组正式比较，不事后挑快样本。
7. 所有正式采样结束后回收 guard 私有 receipt，并检查 guard/worker/SSH 退出与冲突。只有采集覆盖、协议、频率、receipt 都合格时才比较冻结标杆和逐case加速比。脚本返回 `pending_guard_receipt` 不是验收成功。

flush大小、预热次数、采样数量和频率规则由绑定协议提供；脚本不默认采用某次实验的参数。旧标杆只有在环境和协议可比时才能复用。

## 配套脚本的使用

把 `scripts/torch_npu_measure.py` 与 `scripts/profile_parser.py` 一起放入私有执行目录。在已受 guard 保护的设备进程中导入，调用 `measure()`；具体签名、回调职责见函数 docstring。`verify_profile(path, parsed)` 必须检查原始 trace 的实际阶段覆盖，脚本不知算子结构，不提供始终返回成功的默认实现。

先选一个已通过精度的 case，在同一有界性能作业中验证采集范围及与协议匹配的频率/原始文件，再顺序执行全部必测 case。小 case 失败保留证据，不扩成无界重试。频率/flush回调与调用适配是当前节点需补的具体代码，不因缺少现成方法名直接返回环境准备。

使用完整 profiler 设备数据就是执行性能测试；只有另采流水诊断指标时才是附加诊断任务。工具适配、排队超时、连接中断、候选失败和测量冲突分别报告。

## 面向下一轮 analysis 的详细采集

本文件及 measure() 负责固定协议性能样本；详细诊断使用 [上板流水与指标](onboard-diagnostics.md) 的原生 torch_npu 分指标采集或 msprof op + MSTX 路径，配套 torch_npu_diagnostics.py。生成 ANALYSIS_DATA，记录实际 kernel、原始数据与范围；不要将 PipeUtilization 单项称为 PipeTimeline，也不要用诊断重放时间替换本协议样本。
