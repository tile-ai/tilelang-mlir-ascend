# Insight 兼容与验证

## 事件与连线

- 使用 [Ascend/msinsight](https://github.com/Ascend/msinsight) 官方发布版本，检查其对
  原名 intra-block B/E 的支持；不支持时报告版本限制，不改写事件或解析器。
- 对未将 GET/RLS X 的顶层 id 关联到 slice.flag_id 的解析器，增强脚本保留硬件 PIPE
  上的原名事件，另加带 `viewer_compatibility_anchor=true` 和来源信息的 B/E
  SET_FLAG/WAIT_FLAG 锚点。这些是辅助显示事件，不是官方采集的真实指令。
- GET/RLS 与锚点都从 raw completion 起显示 1 tick，语义耗时为零。锚点分别位于
  `PIPE / GET_BUF (compat)` 与 `PIPE / RLS_BUF (compat)`；s/t/f 端点使用对应锚点
  的 pid/tid 和起点。连线表示 acquire/release 范围，不代表指令延迟或跨 PIPE 依赖。
- 分离两端泳道是为了避免已观察到的 `(flag_id, track_id)` 首条 slice depth 查询
  忽略端点时间，以及同轨长区间遮挡。仅修改 thread_name 不能改变归属，须核对 pid/tid。
- 色块和连线使用 sunsetOrange（调色板索引 9）。色块按 cname 哈希选色，线按 cat
  哈希选色；脚本的 `GET/RLS buffer scopes` 对应同一橙色，仅改 flow.cname 无效。

## 实际验证

使用新目录导入，避免旧解析数据库缓存；区分以下验证层级：

1. JSON/raw：源记录覆盖、PC/tick/core/PIPE、配对 ID、真实指令与锚点分别计数。
2. 解析库：GET/RLS 原名、intra-block 原生区间、锚点 flag_id 与 flow 的关联。
3. 流水接口：`unit/threadTraces` 与 `unit/flows` 逐端点比较时间、泳道和 depth；
   `flow/categoryEvents` 另验全量连线。单节点成功不证明全量视图正确。
4. GUI：实际点击色块与检查连线。未测试时明确说明，不将接口验证称为 GUI 验收。

1 tick 的纳秒宽度为 `1 / frequency_ghz`，整数纳秒解析可能量化为 0 ns，影响可见性/命中。
保留 1 tick 并报告实际限制，不静默扩大宽度。空查询结果需检查版本、请求参数和数据，
不能直接判定无事件，也不能将其他版本或算子的验证结果当作本次验证。

点击 GET/RLS 的兼容 SET/WAIT 锚点查连线，不能宣称原名 X 已支持直接关联。
兼容性以官方解析器的实际结果为准，解析限制随 trace 一并说明。
