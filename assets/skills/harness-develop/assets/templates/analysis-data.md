# 上板分析数据交付

- contract: operator-harness/analysis-data@1
- project_id: {{项目身份}}
- run_id: {{develop run}}
- candidate_id: {{实测候选；多次尝试分别生成本文件}}
- status: {{ready / partial / unavailable / not_run / not_applicable}}
- purpose: analysis_only
- task: {{冻结 TASK 路径与摘要，含诊断计划、必需/可选项和预算}}
- correctness: {{全部必选精度证据；仅诊断例外须标明，不能据此接受精度失败}}
- performance: {{独立的正式性能证据，不使用详细 profiling 的耗时替换}}

## 身份与采集范围

{{case/输入/seed、源码/补丁、二进制或 JIT 缓存、构建参数与 launch 身份；真实 kernel 名及其来源，完整算子中的阶段、stream/task 对应；环境修订、芯片、核心限制、torch/torch_npu、CANN、msprof/msopprof 版本；重放模式、缓存/重置和同步范围。记录实际采集的 core/pipe，不能由运行核数推断覆盖。}}

## 计划与实际覆盖

| case/阶段/真实 kernel | 采集后端与指标 | 必需性 | 状态 | 产物入口与缺口原因 |
| --- | --- | --- | --- | --- |
| {{代表性或退化 case}} | {{主采集 msprof op/msopprof 的完整指标组合；按需补充的 torch_npu 枚举另列}} | {{TASK 固定}} | {{collected / unsupported / failed / not_run}} | {{路径；不支持的版本/硬件依据}} |

默认性能优化任务选择代表性 case、退化 case 与不同实现分支；无需每轮对所有 case 做全指标重采。复用时提供确切候选/输入/构建一致性依据，不将旧数据冒充新采集。ready 表示 TASK 要求的范围已核对，不表示硬件全部 core 或全部指标均被采到。

## 重放与观测边界

- 采集分组：{{从日志/元数据确认的 PMU 分组与专项流水/插桩 pass；实际执行次数和依据，不可见时注明未知，不按指标数或 launch-count 推算}}
- 状态恢复：{{工具 Backup/Restore 适用的设备内存范围及失败日志；应用 prepare 的作用；未覆盖的 host/通信/外部状态，选择 kernel/application 模式的依据}}
- 运行差异：{{L2 清理、注入/插桩、任务队列配置、同步与正式执行的差异；依照实际版本记录}}
- 关联限制：{{哪些原始事件在同次采集中，哪些统计来自不同重放；不能跨 pass 拼成同时发生的流水}}

## 原始文件和可复现入口

| 文件/目录 | 摘要或内容清单路径 | 格式与解析工具/版本 | 能回答的问题及覆盖限制 |
| --- | --- | --- | --- |
| {{原始 profile、trace.json/trace_view.json、CSV/DB、visualize_data.bin、日志}} | {{SHA256 清单；不只保留截图}} | {{实际字段名/单位；二进制用官方工具}} | {{调用映射 / MTE2-V-MTE3 重叠 / 核间负载 / 带宽与计算边界}} |

{{记录在设备进程执行的命令/argv、脚本与参数、工作目录、激活入口、查询/回收方式；worker/msprof/runner 退出与所需资源 receipt。无匹配 kernel、空数据、权限/工具不支持与采集失败分别记录，不能仅凭退出码为0标 ready。}}

## 给 analysis 的观察与问题

| 观察（含原始文件行/事件/core/pipe） | 推断及不确定性 | 可检验的下一问题 |
| --- | --- | --- |
| {{数据事实}} | {{与实际源码/IR/生成代码的联系；避免把利用率直接当重叠}} | {{调整后预期哪些计数/流水变化}} |

{{性能退化归因与本文件交叉链接。实现问题仍在 develop 内修复；思路问题连同精度通过、正式性能证据、实际实现核对和本分析数据交给下一 analysis。}}

## 未完成项与接收

{{必需项未完成则保持未验收检查点；可选项缺失、能力不支持、复用旧证据分别说明。没有执行时明确原因、影响与补齐入口，不伪造指标，不为采集自行加额或改变核心限制。}}
