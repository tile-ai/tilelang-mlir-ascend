# 流程来源

这两个 Operator Harness skills 的阶段顺序与产物设计参考 Open GSD 的 GSD Core，按昇腾 TileLang 目标与 Codex 技能机制重新实现。它们不是官方 GSD 命令，不依赖 GSD runtime，也不保留 GSD 命令别名。

来源：[Open GSD / GSD Core](https://github.com/open-gsd/gsd-core)。参考入口包括 `skills/gsd-new-project/`、`skills/gsd-onboard/` 和 `gsd-core/` 下的流程及追问资料，具体路径随上游版本变化；不要求安装固定 commit。

本实现使用项目级运行记录目录，保留需求澄清、证据映射、需求到阶段的覆盖和恢复机制；去掉自动提交及对其他 GSD 命令和专用子 agent 的依赖。

追问保留沿回答深入、具体化歧义、自由表达切换及“继续探索 / 进入规划”的收敛过程；明确直接生成的用户指令可跳过重复确认。onboard 保留先理解代码和吸收文档、处理残缺状态再进入规划的顺序，在同一会话完成，不要求用户反复跳转到其他命令。两者共享追问规则，并按 Codex 的实际提问能力执行。

上游 MIT 许可保留于 [gsd-license.txt](gsd-license.txt)，不代表本项目整体许可证已确定。
