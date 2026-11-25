---
id: pattern/<short-kebab-name>
layer: pattern
arch: [Ascend950]
kind: hw-pattern
status: active
verified_scope: tilelang-ascend
---

# <模式标题>

## 适用症状与触发条件

- 列出 `msprof` / 仿真 / Roofline 中可观测的具体瓶颈信号。

## 核心物理机理

1. 用物理原因解释为什么该手法有效（减少什么搬运、隐藏什么延迟、反转什么瓶颈）。

## 实现要点与约束

- 关键布局、同步、精度与片上容量注意事项。
- 如有仓内可运行样例，用反引号注明相对路径（如 `` `example/...` ``）。

## 不适用边界

- 明确在哪些维度、架构或瓶颈类型下该模式无效甚至负收益。

## 验证清单

- 生成代码/ASC 核查点、仿真/真机观测指标与必测边界 case。
