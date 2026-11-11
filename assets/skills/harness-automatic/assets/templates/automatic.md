# Automatic 控制记录

- contract: operator-harness/automatic@2
- batch_id: {{项目内唯一批次}}
- predecessor: {{上一批次归档路径；首次为 not_applicable}}
- task_scope: {{当前任务/需求路径与版本、用户请求来源；不默认为整个路线图}}
- controller: {{实际协调 agent/会话身份及最近更新时间；无可查询ID就说明}}
- execution_strategy: {{subagents / inline，实际可用模式}}
- status: {{running / needs_input / blocked / paused / budget_exhausted / completed}}
- max_steps: {{正整数，用户/已有约定优先，否则8}}
- steps_used: {{启动前递增；失败/中断不退回}}
- resource_ledger: {{各环境资源账本入口；控制步数不提供设备许可}}
- max_parallel_develop: {{默认2，受宿主、用户与资源上限约束；inline为1}}
- in_flight_steps: {{全部 prepared/dispatched/等待接收步骤列表，含 step/run/batch/request/agent/job 与状态；无则 []}}
- updated_at: {{日期时间与时区}}

模式切换时在下节记录待生效请求及其绑定步骤/决定 ID；旧记录没有该字段时查 DECISIONS，不猜已恢复自动运行。

## 目标与停止条件

{{本批次具体目标、验收证据、停止条件、禁止扩大的范围；授权与环境来源}}

## 步进日志

| 步骤 | 动作/原因与输入身份 | run/agent/job 或记录入口 | 开始/结束及状态 | 接收证据、预算变化与下一判断 |
| --- | --- | --- | --- | --- |
| {{编号}} | {{选择}} | {{身份}} | {{状态}} | {{依据}} |

## 等待、接管与恢复

{{待回答问题 ID、决定水位、停止原因；最新模式请求 automatic/next/paused、绑定 run/步骤、是否已通知节点/到达停止边界；原协调者与节点各自状态、接管依据；派发意图/请求标识、响应是否确认；查询/回收/取消入口。原步骤恢复不重复占步，unknown额度仍占用。无缺口写none}}

## 本批次结论

{{目标是否达成、已用/剩余步骤、当前/最佳候选与证据、未运行/未达标/在途项、推荐下一步}}

旧 automatic@1 的 current_step 恢复为单元素 in_flight_steps（none 为 []），保留旧记录与原步数，不能清零或重投。每个并行成员独立计步；批次字段不替代成员状态。
