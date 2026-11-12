# 官方依据与辅助工具

本 Skill 以官方项目为技术依据：

- [tile-ai/tilelang](https://github.com/tile-ai/tilelang/tree/main/.agents/skills/tilelang-ascend)：TileLang API、编译与 profiling 方法。
- [Ascend/msopprof 仿真指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md)：采集命令、SoC 选择及输出格式。
- [Ascend/msinsight](https://github.com/Ascend/msinsight)：原版 Insight 的导入和流水查看功能。

随包的两个 Python 脚本是 harness 维护的离线辅助工具，不属于上述官方项目。它们处理匹配格式的已有采集，输出附带来源和兼容说明的 trace。使用官方发布的工具链和 Insight，无需修改官方源码。许可保留在 [LICENSE](../LICENSE)，实际环境与验证结果记录在任务 `.workspace/` 中。

原生 TileLang 编程 Skill、API 和仓库路径以任务绑定的 TileLang checkout 为依据。
