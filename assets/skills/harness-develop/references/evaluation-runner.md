# 从执行到验收

`evaluate.py` 在独立、带超时的本地 Python 子进程中执行冻结适配器、参考和比较器，生成候选清单、逐 case 报告、执行回执、validation、ANALYSIS_DATA 和 RESULT 草稿。启用控制账本时，预留、job 观察和结算由 runctl 写入。Agent 负责实现与解释，无需手写通过布尔值或拼接证据摘要。

## 准备一次，复用每轮

在注册 TASK 前固定 evaluation-plan@1（见 [模板](../../harness-resume/assets/templates/evaluation-plan.json)）：

- 六项 inputs 指向实际需求、独立参考、比较器、case、协议和环境。路径统一相对 workspace_root；用 searchctl reference 生成引用。
- cases 包含输出名、输入参数以及适用的样本数/耗时目标；仅精度任务 performance_kind=none。
- execution 包含 backend（cpu 或 torch_npu）、adapter 引用和 timeout_seconds。backend 固定测量类型，CPU 样本只能用于 CPU 契约。
- TASK 写 evaluation_plan；collect-only 使用同样的执行绑定，但仅执行诊断回调。code-only 无需执行 plan。

Python 适配接口如下。adapter、reference 和 comparator 是独立文件，均在 run 登记前固定。项目源码或协议变化后建立新任务。

| 模块 | 接口 | 用途 |
| --- | --- | --- |
| adapter | prepare(context, case) → state | 编译/加载 context.artifacts 的确切候选，按 case 准备输入 |
| adapter | run(state) → outputs | 执行完整候选调用，输出保持存活至同步结束 |
| adapter | synchronize(state) | 等待本次调用涉及的全部执行流 |
| reference | run(state) → expected | 独立计算参考输出 |
| comparator | compare(actual, expected, case) → {输出名: bool} | 按固定容限比较全部必选输出；异常传播为失败 |
| adapter（torch_npu 性能） | measurement(state, case, output) → kwargs | 返回 torch_npu_measure.measure 的参数，含真实设备预留检查、flush、同步、频率及 profile 验证回调 |
| adapter（需要诊断） | verify_diagnostics(capture, case) | 检查实际导出内容、kernel/指标覆盖；不满足要求时抛异常 |
| adapter（需要诊断） | diagnostics(state, case, output, context) → {evidence: ref} | 调用 capture_diagnostics(..., evidence_context=context)，或用 emission.diagnostics 组织实际外部采集产物 |

context 提供 workspace_root、project/run/candidate/request 身份、源码/构建/launch 引用、plan 和 case_id。执行器先完成所有必选 case 的参考比较，保存实际与参考输出；全部精度通过后才计时。CPU 路径测同步后的完整调用 wall time；torch_npu 路径复用现有设备测量脚本。多指标诊断仍按 [上板诊断](onboard-diagnostics.md) 使用实际工具。

## 执行

先登记 TASK，再用源码快照、构建清单与 launch 清单生成候选（初版省略 --parent）：

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

控制账本未启用时省略 pool/amount；已启用时预算池单位使用 seconds 或 executions。工具每次返回生成的结果路径。已有同 request 只恢复和整理原结果；精度失败返回非零，保留反例并跳过计时；失联查询原 job。

search@2 的 candidate 命令先把 source/build/launch 及显式 artifact-index 复制到项目内容快照，再生成清单；传入的多文件索引须包含重建所需源码、补丁和未跟踪文件。适配器必须使用 context 中冻结的引用，不能绕回可变 worktree。评测计划与 initial_design 在队列准入前用 searchctl freeze 固定，完整流程见 [依赖队列](../../harness-resume/references/task-queue.md)。

产物位于 `runs/<run>/evaluations/<request>/`，进程与账本观察位于 `runs/<run>/jobs/<request>/`。RESULT.generated.md 的头部已填写机器字段，正文用于补充实现说明。性能未达标的常规优化保留 unknown 归因，需要真实分析后提交独立 RESULT；移植还需补 PORTING。然后按 [执行控制](../../harness-resume/references/execution-control.md) accept，符合目标的候选再独立 promote-candidate。

collect-only 自动生成 diagnostic-evidence，并保留 not_run/not_applicable 精度性能状态。采集器通过 evidence_context 可直接输出同一 schema，内部旧 collection.json/pass 文件列表仅用于具体工具诊断。

## 支持范围

已以独立安装、真实 CPU 子进程验证正确/错误候选、只采集、只精度移植、幂等和控制账本闭环。torch_npu 采集接线由 CPU mock 测试检查，具体设备适配与硬件结果仍待实测。当前是本机 Python 适配入口；远端/容器执行、设备锁与编译器 ABI 自动导出仍按 M2/M4 实施。评测代码与候选分开绑定、子进程运行；恶意同账号代码的 OS 权限隔离不属于当前实现。
