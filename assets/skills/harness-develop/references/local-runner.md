# 本地前台进程 runner

需要完整评测时使用 [evaluate.py](evaluation-runner.md)。已有独立命令可用 runctl.py，获得持久 request、进程组超时、stdout/stderr 和 process-receipt。

```bash
python3 .agents/skills/harness-develop/scripts/runctl.py \
  --workspace . --project example --owner SESSION run request.json
python3 .agents/skills/harness-develop/scripts/runctl.py \
  --workspace . --project example status --run d1 --request attempt1
```

新请求使用 process-request@2：project_id/run_id/request_id、argv 数组、cwd、timeout_seconds、inputs 引用列表及可选 env。cwd 和 inputs.path 都相对 workspace_root，与 searchctl reference 一致。旧 @1 明确按项目记录目录解析以兼容历史；projectctl 的内部仓库路径仍使用其原项目目录接口。

控制账本启用时增加 reservation={pool, amount}，池单位为 seconds 或 executions。runner 自动预留、登记 intent/running/exited 观察并结算实际用量。进程失败仍保留非零退出，不能通过重复 request 再执行；同请求重复调用可补齐终态记账。

status 区分 not_found、running、unknown 和已有终态；监督进程退出但工作进程的 PID/starttime 仍匹配时报告 orphaned/running。unknown 保留原 ID并检查进程，not_found 表示尚无提交意图。超时回收本地前台进程组，脱离进程组的 daemon 不在该接口范围。

process-receipt 描述实际本地进程结果。设备评测由冻结适配器和 evaluate 的执行回执进一步关联；SSH/容器/NPU 的退出与锁策略需由对应后端实现。
