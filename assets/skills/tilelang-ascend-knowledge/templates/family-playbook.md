---
id: family/<family-name>
layer: family
arch: [Ascend950]
kind: playbook
status: active
verified_scope: tilelang-ascend
---

# <算子族名称> 算子族 Playbook

简述本算子族覆盖的子算子与主要变体。

## 1. 架构级选型速查

| 子类 | 推荐主循环组织 | 典型瓶颈演进路线 | 核心参考模式 |
| --- | --- | --- | --- |
| ... | ... | ... | ... |

## 2. 常见陷阱与全矩阵泛化要点

- 列出 dtype / shape / 尾块 / 动态性 / 缓存键等关键工程防坑点。
- 引用已证伪反例卡，防止重复试错。
