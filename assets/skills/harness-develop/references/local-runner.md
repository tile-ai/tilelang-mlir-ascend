# 本地前台进程 runner

`runctl.py` 是随包提供的 Linux 本地进程执行原语，用于已有、明确的命令入口。它保存提交意图、日志、进程身份、超时及退出回执；不是远端/NPU 执行适配器，不导出编译器 ABI，不替代设备锁、控制账本或验收器。先按 TASK 核对环境、预算与排他约定；需要设备锁时 argv 调用环境中已核实的前台 wrapper，不能以本工具绕过。

request JSON 使用 process-request@1：project_id、run_id、request_id、argv 字符串数组、cwd（相对项目记录目录）、timeout_seconds（正数，上限一天）、inputs（项目记录目录相对 path/sha256 列表）、可选 env 字符串字典。inputs 明确声明实际依赖，至少绑定执行脚本；未列入的依赖不受摘要检查。源码 cwd 通常为 repos/<repo>，不通过 shell 拼接命令。

```bash
python3 .agents/skills/harness-develop/scripts/runctl.py \
  --workspace . --project example --owner SESSION run request.json
python3 .agents/skills/harness-develop/scripts/runctl.py \
  --workspace . --project example status --run r1 --request request1
```

产物在项目 runs/<run>/jobs/<request>/。同一 request 的内容冲突会拒绝；内容一致且有终态回执则复用，缺回执则只报告 running/unknown，不再次启动。实际非零退出向调用端传播，超时退出码为 124；stdout/stderr 和 process-receipt.json 保留。首次准备/创建子进程之间中断也不能自动重投，应核查原请求。

只支持前台进程及同一进程组的子进程，结束/超时后清理本组残留；不支持 double-fork、setsid 逃离进程组或外部 daemon 提交。supervisor 被强制杀死后无法保证清理，状态保持 unknown，按原 PID/启动身份查询恢复，不能仅凭 PID 或旧 running 文本判断当前仍活跃。status 对已结束请求读取终态回执；无回执时检查原 supervisor 的 Linux starttime，避免 PID 重用。

process-receipt@1 明确 validation_scope=local_process_only、device_validation=not_evaluated。设备适配器必须另行核验资源回执、实际构建/launch、各执行层退出、产物身份和计时有效性，才能产生 execution-receipt@1。禁止把进程退出 0 直接改成设备 valid=true。通用 SSH/container、取消/回收适配及真实 NPU 链路仍按 M2 建设。
