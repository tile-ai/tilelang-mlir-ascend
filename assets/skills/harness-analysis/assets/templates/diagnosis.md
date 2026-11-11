# 当前候选诊断与优化提案

本文件作为 node-result@4 的附件，也可将对应内容写入 RESULT 正文；它不替代 RESULT 的 metadata 或验收证据。内容按实际问题填写，没有新采集时标 not_run，缺少某类证据时标 unknown/partial 及影响。

## 输入与覆盖

- TASK / run / search_node_id / candidate / build / baseline / case / 环境及摘要：
- 前序 ANALYSIS_DATA、正式性能与精度、PORTING（若有）：
- 原始 trace/表格、解析命令及 profile-input/profile-summary（使用脚本时）：
- 实际 kernel/阶段/launch、stream/task/core/pipe、工具版本、PMU pass、单位与覆盖限制：
- 输入一致性、必需数据缺口、当前分析可回答的范围：

## 当前问题与缺陷

| finding_id | case/阶段/源码位置 | 原始观测与定位 | 缺陷/瓶颈解释及强度 | 竞争解释/反证/缺口 | 处置节点 |
| --- | --- | --- | --- | --- | --- |

缺陷给出预期与实际行为、复现输入和影响范围；区分精度/实现、测量/采集、性能机制和未知。已接收候选发现反例时说明对当前/最佳候选证据的影响及重验建议。

## 算子结构与工作量

阶段/数据依赖、TileLang→生成代码映射、分派与融合边界；逻辑/实现/实测物理流量分别核算，Cube/Vector/SIMT/特殊函数成本分别解释。理论界限写来源、单位、并行/重叠假设及局限。

## 可执行方案与探索树

| plan_id / mechanism_id | 输入候选与 case | 依据 finding_id | 最小改动及 API/lowering 依据 | 优先级/准入状态 | 同节点或子节点建议 / prior_nodes |
| --- | --- | --- | --- | --- | --- |

每个方案补充 expected、falsifier、keep、discard、simplification；精度/性能验证 case、预期 trace 变化、风险/回退条件、依赖/互斥关系、预算需求。较大实验链接 search-plan@1。重试同机制写新证据或 changed_conditions；design-only 单列缺口。

## 主推荐与知识提案

一个下一动作、输入与返回/停止条件；需要补采时精确到 case/kernel/launch、指标/覆盖、环境和额度。供协调者接收后写入 hot 的 observation/hypothesis/conclusion、适用条件、反例和原始证据引用。
