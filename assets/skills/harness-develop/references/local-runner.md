# 本地前台进程 runner

需要完整评测时使用 [evaluate.py](evaluation-runner.md)。已有独立命令可用 runctl.py，获得持久 request、进程组超时、stdout/stderr 和 process-receipt。

```bash
python3 .agents/skills/harness-develop/scripts/runctl.py \
  --workspace . --project example --owner SESSION run request.json
python3 .agents/skills/harness-develop/scripts/runctl.py \
  --workspace . --project example status --run d1 --request attempt1
```

新请求使用 process-request@2：project_id/run_id/request_id、argv 数组、cwd、timeout_seconds、inputs 引用列表及可选 env。request_id 在项目内唯一，不同 run 使用不同 ID。cwd 和 inputs.path 都相对 workspace_root，与 searchctl reference 一致。旧 @1 明确按项目记录目录解析以兼容历史；projectctl 的内部仓库路径仍使用其原项目目录接口。

控制账本启用时增加 reservation={pool, amount}，池单位为 seconds 或 executions。runner 在启动前原子核验 run、完整 request_sha256 与 pool/amount，并预留额度、登记 intent；已有记录必须匹配，冲突记 not_submitted 且不启动。随后记录实际进程状态并结算。旧账本恢复需与原 request.json 完全一致。进程失败保留实际退出；同请求重复调用只恢复原状态或补齐终态记账。启动后记账异常标 interrupted，确认退出才记取消/结算，无法确认则保留 unknown 与预留。

process-request 可设置 defer_settlement=true：已经启动的终态进程只记录本地结果，预留继续保留并显示 settlement_required，等待外层资源终态与真实用量证据。torch_npu evaluate 使用此模式，在 device-receipt 核验后结算；未启动的失败可按零用量收尾。

退出未确认的进程后来已停止时，恢复可将原 job 记为 cancelled；executions 池按已启动的一次结算，seconds 池保留预留并记录 settlement_required，等待宿主补齐未观测时段实际用量证据后 control-settle。旧 receipt 的 elapsed_seconds 不能代表完整耗时。

status 区分 not_found、running、unknown 和已有终态；监督进程退出但工作进程的 PID/starttime 仍匹配时报告 orphaned/running。unknown 保留原 ID并检查进程，not_found 表示尚无提交意图。超时回收本地前台进程组，脱离进程组的 daemon 不在该接口范围。

项目进入 freezing 后，已有 request 可继续查询、恢复原终态和结算；新 request 创建和实际 Popen 仍要求 active，冻结过程不会启动新工作。

process-receipt 描述实际本地进程结果。设备评测由冻结适配器和 evaluate 的执行回执进一步关联；SSH/容器/NPU 的退出与锁策略需由对应后端实现。
