# 带 NPUIR 工具链的 wheel

此方式把公开构建的 NPUIR 编译器、设备 bitcode、Python 绑定和许可证装入
TileLang wheel。CANN、驱动、PyTorch 与 torch_npu 仍是外部依赖；不将私有
构建脚本或 CANN 编译器复制进 wheel。

## 构建

先完成 TileLang 和配套 NPUIR 的编译，再指定同一份 NPUIR 安装和源码：

```bash
export TILELANG_BUNDLE_NPUIR_INSTALL=/path/to/AscendNPU-IR/build/install
export TILELANG_BUNDLE_NPUIR_SOURCE=/path/to/AscendNPU-IR
export TILELANG_SKIP_BUILD=1
export USE_NPUIR=true
export TILELANG_BUNDLE_ZSTD_LIBRARY=/path/to/libzstd.so.1
export TILELANG_BUNDLE_ZSTD_LICENSE=/path/to/zstd/LICENSE
python setup.py bdist_wheel
```

使用新的打包暂存目录，避免混入旧版 Python 绑定。缺少必需文件时构建失败，
不会悄悄改用 PATH 中另一份编译器。`lib/npuir/manifest.json` 保存随包文件的
SHA-256。发布前还需核查动态库依赖、平台/Python/CANN 兼容性及独立安装回归；
生成 wheel 本身不代表这些检查通过。

如果构建链接了共享 zstd，上述两个 ZSTD 变量选择要随包分发的库和对应
许可证。打包时用带哈希的私有 SONAME 重写包内依赖，避免与 Python
`zstandard` 的不同版本冲突；需要 `patchelf`。只改暂存副本，不改系统库。
修复后的哈希记录在 `lib/dependency-manifest.json`，NPUIR 清单同步更新。

## 安装与查找

```bash
source /path/to/cann/set_env.sh
python -m pip install /path/to/tilelang.whl
```

默认先找包内 `tilelang/lib/npuir/bin`，再找 PATH。开发者可设置
`TILELANG_NPU_COMPILER_PATH` 指定工具目录或安装根目录；显式指定但无效时
直接报错，不回落到其他版本。用户无需访问构建者的源码工作树。

## 本轮验证范围

- 目标：Ascend910B、CANN 9.0.0、Linux x86_64、CPython 3.11。
- 包内工具链通过 packed sort、2/3/4 路归并、MixCV 和标量条件共 7 项设备测试。
- MlpLightningIndexer 重新导出 18 个 AOT：16 个 H16 shape，以及 H32 的
  `(Tq,Tk,B)=(3504,7223,1)`、`(4144,20528,2)`；完整输出与 Graph 回归通过，
  H32 另做 CPU 参考抽检。这不代表所有模型 shape 或整网验收。
- 安装目标目录独立，Python 的 torch/torch_npu 依赖复用验证环境；尚未验证
  全新系统安装或其他 CANN/Python 版本。候选 wheel 保留 `linux_x86_64` 标签，
  所用构建包含 `GLIBCXX_3.4.30` 依赖，不承诺旧版系统兼容。
