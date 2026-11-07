# 候选早期检查

- contract: operator-harness/precheck@1
- project_id: {{与 TASK/project.json 一致}}
- run_id: {{所属 develop run}}
- project_contract: {{PROJECT 修订、快照路径与 SHA-256}}
- candidate: {{候选 ID、实际文件清单与摘要；未形成完整候选时说明范围}}
- rules: {{目标仓库规范快照/摘要，及实际检查工具版本}}

## 实际检查

| 检查 ID | 具体输入范围 | 机器/环境、cwd 与实际命令 | 判定 | 日志/退出码与依据 | 对下一步的影响 |
| --- | --- | --- | --- | --- | --- |
| {{PROJECT 中的检查 ID}} | {{文件/配置/依赖身份}} | {{已审查入口；执行或 not_run}} | {{passed / failed / not_run / invalid / not_applicable}} | {{日志引用；不适用/未运行的原因}} | {{可交接/先修复/仅允许指定诊断}} |

检查有自动修改时保存前后文件差异，重新固定候选身份。工具缺失、bootstrap 失败或未运行不能写 passed；使用替代工具时说明它覆盖和不覆盖哪些规则。

## 结论

{{按当前 TASK 和契约判断是否可交给 device-test；尚缺哪些发布检查；复用旧检查时说明输入仍一致的依据。不将 CPU 检查通过写成设备验证通过。}}
