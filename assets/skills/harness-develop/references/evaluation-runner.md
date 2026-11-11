# 从执行到验收

`evaluate.py` 在独立、带超时的本地 Python 子进程中执行冻结适配器、参考和比较器，生成候选清单、逐 case 报告、执行回执、validation、ANALYSIS_DATA 和 RESULT 草稿。启用 control 时，runctl 记录预留、job 观察和结算。评测证据由真实执行生成，agent 负责实现与解释。

## 固定评测输入

注册 TASK 前固定 evaluation-plan@1（见 [模板](../../harness-resume/assets/templates/evaluation-plan.json)）：

- 六项 inputs 引用需求、独立参考、比较器、case、协议和环境，路径相对 workspace_root。用 searchctl reference 生成摘要，多文件依赖使用 artifact-index。
- cases 指定必选输出、输入参数及适用的样本数/耗时目标；仅精度任务使用 performance_kind=none。
- execution 是必填对象，包含 backend（cpu 或 torch_npu）、冻结 adapter 引用和正数 timeout_seconds。缺失 execution 在准入时拒绝，legacy 记录按其明确兼容路径处理。
- metric 可省略，由 backend 确定；显式填写时 cpu 对应 cpu_wall_ms，torch_npu 对应 mean_device_busy_union_ms。比较身份总包含 backend 与实际 metric，performance 报告中的 metric 也须匹配；跨口径结果不能替换同一 best。
- TASK 引用 evaluation_plan。collect-only 也需执行绑定；code-only 可以没有 plan，有 plan 时仍遵守其完整契约。

Python 适配接口如下。adapter、reference 和 comparator 各自固定身份；项目源码或协议变化后建立新任务。

| 模块 | 接口 | 用途 |
| --- | --- | --- |
| adapter | prepare(context, case) → state | 编译/加载 context.artifacts 的确切候选，准备本 case 输入 |
| adapter | run(state) → outputs | 执行完整候选调用，输出存活至同步结束 |
| adapter | synchronize(state) | 等待本次调用涉及的全部执行流 |
| adapter（可选） | teardown(state) | 在同步后释放本 case 的设备/外部资源；prepare 内部失败的部分资源由 prepare 自行清理 |
| reference | run(state) → expected | 独立计算参考输出 |
| comparator | compare(actual, expected, case) → {输出名: bool} | 按固定容限比较全部必选输出，异常传播为失败 |
| adapter（torch_npu 性能） | measurement(state, case, output) → kwargs | 返回 torch_npu_measure.measure 参数，含设备预留、flush、同步、频率及 profile 验证回调 |
| adapter（需要诊断） | verify_diagnostics(capture, case) | 检查真实导出内容与 kernel/指标覆盖，不满足时抛异常 |
| adapter（需要诊断） | diagnostics(state, case, output, context) → {evidence: ref} | 调用 capture_diagnostics(..., evidence_context=context) 或 emission.diagnostics 组织实际采集产物 |

context 提供 workspace_root、project/run/candidate/request 身份、源码/构建/launch 引用、plan 和 case_id。精度阶段逐 case 准备、比较、保存输出、同步和清理；全部必选精度通过后，性能/诊断阶段逐 case 重新 prepare，再执行并清理。因此 prepare 必须可重复调用，不能依赖上个 case 的存活 state。原地更新、输入重置与重放按冻结协议实现。

CPU 路径测同步后的完整调用 wall time；torch_npu 路径复用设备测量脚本。多指标诊断仍按 [上板诊断](onboard-diagnostics.md) 使用实际工具。

## 启动与恢复

先登记 TASK，再用源码快照、构建清单与 launch 清单生成候选；初版省略 --parent：

```bash
python3 .agents/skills/harness-develop/scripts/evaluate.py \
  --workspace . --project example --owner SESSION candidate \
  --run d1 --id C1 --parent C0 \
  --source .workspace/example/runs/d1/source.py \
  --build .workspace/example/runs/d1/build.json \
  --launch .workspace/example/runs/d1/launch.json

python3 .agents/skills/harness-develop/scripts/evaluate.py \
  --workspace . --project example --owner SESSION run \
  --run d1 --candidate .workspace/example/runs/d1/candidates/C1/CANDIDATE.md \
  --request d1-attempt1 --pool device --amount 60
```

control 未启用时省略 pool/amount；启用后池单位为 seconds 或 executions。request 在项目内唯一。同一请求的恢复使用原 run、候选、请求及预留参数，只查询或整理原结果；新尝试使用新 ID。精度失败返回非零、保留反例并跳过正式计时；失联先查询原 job。

search@2 的 candidate 命令先将 source/build/launch 及显式索引复制到内容快照。多文件索引须列出重建所需源码、补丁和未跟踪文件；适配器从 context 读取冻结引用。plan 和 initial_design 在队列准入前用 searchctl freeze 固定，见 [依赖队列](../../harness-resume/references/task-queue.md)。

## 设备结果的最终确认

成功的 torch_npu worker，包括仅精度和 collect-only，先返回 `pending_guard_receipt`，其中提供 context 路径和 worker_result 引用。此时保存原始数据，尚不生成可接收的 validation/RESULT。torch_npu 请求设置 defer_settlement，control 中的设备预留在本地 worker 结束后继续保留，直到原 guard 的最终回执核验完成。外层执行入口在原 guard 结束后回收真实退出与冲突观察，生成 device-receipt@1：

```json
{
  "schema": "operator-harness/device-receipt@1",
  "project_id": "example",
  "run_id": "d1",
  "candidate_id": "C1",
  "request_id": "d1-attempt1",
  "artifacts": {
    "source": {"path": "{{source path from context}}", "sha256": "{{sha256}}"},
    "build": {"path": "{{build path from context}}", "sha256": "{{sha256}}"},
    "launch": {"path": "{{launch path from context}}", "sha256": "{{sha256}}"}
  },
  "evaluation_plan": {"path": "{{plan path from context}}", "sha256": "{{sha256}}"},
  "worker_result": {"path": "{{path from pending output}}", "sha256": "{{sha256}}"},
  "status": "released",
  "exit_codes": {"guard": 0, "worker": 0},
  "conflicts": [],
  "resource_usage": {"pool": "{{original pool}}", "unit": "{{seconds / executions}}", "used": null},
  "raw_files": [{"path": "原 guard 观察或日志的 workspace 相对路径", "sha256": "实际摘要"}]
}
```

artifacts/evaluation_plan 引用须与 context 一致，worker_result 使用 pending 输出给出的确切引用。回执由外层入口依据原作业生成，不能由 agent 猜测成功。exit_codes 的层名为非空字符串，guard 必需；实际存在的 worker/SSH/容器等退出层全部记录，值须为整数 0。raw_files 非空，绑定真实 guard/退出观察文件；本地 process-receipt 由 evaluate 另外关联。control 启用时 resource_usage 必填，pool/unit 匹配原预留。executions 的 used 必须为整数 1，表示已启动的一次设备请求；seconds 的 used 是整个 guard 生命周期的实际正有限用量，必须不小于原 process-receipt.elapsed_seconds。纯设备 busy 秒数不代表该预算池用量，本地进程 elapsed_seconds 也不能直接冒充 guard 的完整占用；示意中的 null 须替换为实际值。无 control 时可省略 resource_usage。

取得回执后，以相同请求和原预留参数再次调用：

```bash
python3 .agents/skills/harness-develop/scripts/evaluate.py \
  --workspace . --project example --owner SESSION run \
  --run d1 --candidate .workspace/example/runs/d1/candidates/C1/CANDIDATE.md \
  --request d1-attempt1 --pool device --amount 60 \
  --device-receipt .workspace/example/runs/d1/device-receipt.json
```

此操作验证身份、worker_result、原始观察、退出和冲突，并按实际 resource_usage 幂等结算原预留，再整理原结果；不重跑 worker。错误或 pending 回执拒绝且保留原样本；项目进入 freezing 时可沿已有 context 完成原结果确认，仍禁止启动新的评测。成功后同回执重试幂等，也可省略 --device-receipt 复用已固定的 device-finalization.json；另一份回执不能覆盖已完成确认。accept 再次核验最终 device_receipt。精度失败可先生成失败证据以保存反例，仍不能接收或晋升。失败或无法核验的 guard 保留预留，宿主确认原作业停止及实际用量后，按既有 control-job/control-settle 收尾；不能填零消耗或重投来清除占用。

## 结果与接收

产物在 `runs/<run>/evaluations/<request>/`，进程与账本观察在 `runs/<run>/jobs/<request>/`。RESULT.generated.md 填写机器字段，正文供补充实现说明。常规优化性能未达标时先保留 unknown 归因，真实分析后提交独立 RESULT；移植另需 PORTING。随后按 [执行控制](../../harness-resume/references/execution-control.md) accept，合格候选可独立 promote-candidate；best 的选择理由仍由协调者记录。

[validation 模板](../../harness-resume/assets/templates/validation.json) 展示生成证据的结构，使用时以工具输出为准。CPU execution-receipt 省略 device_receipt/worker_result；torch_npu 成功回执绑定两者。精度任务没有 performance 记录，collect-only 必须实际采集，即使旧 TASK diagnostics_required=false 也执行本操作的诊断，输出 diagnostic-evidence 并保留 not_run/not_applicable 状态；成功的 torch_npu 诊断还绑定 device_receipt、worker_result 和 process。采集器旧 collection/pass 文件只服务工具内部诊断。

## 支持范围

独立安装和真实 CPU 子进程覆盖好/坏候选、仅采集、仅精度移植、幂等与控制闭环；torch_npu 接线和设备最终回执通过 CPU mock 验证。具体芯片/工具链仍需真实设备验收。当前是本机 Python 适配入口，远端/容器执行、设备锁和编译器 ABI 自动导出继续按 M2/M4 建设；任意同账号代码的 OS 权限隔离属于独立后端能力。
