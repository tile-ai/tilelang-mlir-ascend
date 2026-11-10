# TileLang 原生 Skill 跟踪

`tilelang/` 保存官方 [tile-ai/tilelang](https://github.com/tile-ai/tilelang/tree/main/.agents/skills) 的 `.agents/skills/tilelang-*` 原文及配套文件，保留文件权限和 LICENSE。它是可比较的源码快照，不是 harness 节点 Skill，也不由默认安装器激活。可安装的 Harness Skill 位于 [`assets/skills`](../skills/)。

快照以官方 `main` 为来源，不混入个人 fork 或工作树的未合并修改。

| Skill | 用途 |
| --- | --- |
| tilelang-ascend | Ascend 算子、SIMD/SIMT、流水、同步与 profiling |
| tilelang-build | 来源仓库的构建、安装与测试 |
| tilelang-dev-guide | C++ 规范与回归范围 |
| tilelang-backend | 后端与执行模式集成 |
| tilelang-layout | Layout、Fragment 与布局推导 |
| tilelang-semantic | 语义规则和合法性检查 |
| tilelang-simplification | 编译器和运行时简化分析 |
| tilelang-submit-pr | 来源仓库的提交、测试与 PR 流程 |
| tilelang-tvm-ir | TVM IR 对象与 C++ 使用约定 |

## 使用与路径

优先读取任务环境绑定的 TileLang checkout 中的同名 Skill，并记录实际版本。这里的原文用于离线参考和更新比较，不证明与目标编译器兼容。原文中的仓库路径和构建/发布命令始终相对于 **来源 TileLang checkout**，不是本 harness；例如 `../../../docs/` 应从原 `.agents/skills/<name>/SKILL.md` 位置解析。Skill 内和相邻 Skill 之间的链接可在本目录阅读，外部源码、示例、文档从所选 checkout 获取。不要在本 harness 目录执行原生构建或发布命令。

## 更新快照

从官方 TileLang `main` 的干净 checkout 复制 `.agents/skills/tilelang-*` 到 `tilelang/`，保留配套文件、文件权限和 LICENSE，跳过 Python 缓存。官方已删除的文件在快照中一并移除，用 Git 审阅后提交：

```bash
git status --short -- assets/native-skills/
git diff -- assets/native-skills/
```

实际来源路径、revision 和同步记录保存在 `.workspace/`，运行任务自行选择工具链版本。
