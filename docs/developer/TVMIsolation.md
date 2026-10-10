# TileLang 与 CANN TVM 隔离

CANN 的 `te` / `tbe` 会加载自己的公共 `tvm` 包。TileLang 使用另一版本的 TVM，
两者共用 Python 模块名或本地符号时，会出现 `IRBuilderFrameNode` 未定义符号、
注册函数缺失等错误。仅调整 `sys.path` 无法处理已经加载的 TVM。

## 使用方式

TileLang 的 TVM 通过以下接口访问：

```python
import te
import importlib

tilelang = importlib.import_module("tilelang")
from tilelang import tvm
from tilelang.tvm import tir
```

也可以先导入 TileLang，再导入 `te`。公共 `import tvm` 留给 CANN 或调用方；
TileLang 不再设置公共 `TVM_LIBRARY_PATH` 或添加公共 TVM 的 Python 搜索路径。
依赖旧行为的代码需要把 TileLang 相关的 `import tvm` 改为 `from tilelang import tvm`。
不要在两个 TVM 之间传递 TIR、NDArray 或 PackedFunc 对象。旧版本 TVM 对象的 pickle
不保证兼容；新对象使用 `tilelang.tvm` 模块名序列化。

## 构建与打包

当前隔离加载器面向 Linux/glibc。实现包含三个部分：

- CMake 和 setup 在构建目录生成私有 Python 包，不修改 TVM 子模块源码，保留其许可证。
  私有 TVM 使用 ctypes FFI，不使用公共 TVM 的 Cython 扩展。
- 本地库使用 `libtilelang_tvm.so` 和 `libtilelang_tvm_runtime.so` 的文件名及 SONAME。
  加载采用 `RTLD_LOCAL | RTLD_DEEPBIND`，限制本地符号和注册表的交叉影响。
- GCC 编译 TVM 和 TileLang 时使用 `-fno-gnu-unique`，避免类型索引等静态数据跨库合并。
  打包会检查动态符号表，拒绝仍含 GNU-unique 符号的产物。

普通源码构建继续使用 `install_npuir.sh`，已有编译产物使用 `build_wheel.sh` 打包。
升级时必须重新编译 TVM 和 TileLang；不能直接把旧 `libtvm.so` 改名后使用。
CI 的 TVM cache 使用新的 `tvm-prebuild-isolated-v1` 前缀。

独立编译 TVM 并供 TileLang 复用时，在原有 CMake 配置基础上增加编译选项，然后生成私有库：

```bash
cmake -S 3rdparty/tvm -B 3rdparty/tvm/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fno-gnu-unique
cmake --build 3rdparty/tvm/build -j16
python3 tools/prepare_tvm.py libraries 3rdparty/tvm/build tvm-prebuild
# TileLang 的 CMake 配置传入 -DTVM_PREBUILD_PATH="$PWD/tvm-prebuild"
```

此处还需要原有 NPUIR/CANN 构建配置。`tools/prepare_tvm.py libraries` 依赖
`readelf` 和 `patchelf`。wheel 内同时包含私有 Python 包、私有本地库和原有 NPUIR 编译器资源。

## 验证

在已加载 CANN 环境变量、安装匹配的 CPU Torch / torch_npu 的环境中运行：

```bash
python3 -m pytest -q testing/python/test_tvm_vendor.py
python3 -m pytest -q testing/npuir/import_ops/test_tvm_isolation.py
```

前者验证 Python 导入转换；后者为两种导入顺序分别启动新进程，检查公共模块身份、
私有库路径、注册表、回调、线程并发、缺失属性访问、pickle 和 spawn 子进程。
未安装 CANN TBE 时，共存测试会跳过。验证 wheel 时，应从安装环境运行并避开源码目录，
例如把共存测试复制到临时目录后执行。

还需运行现有 NPU 算子数值测试；仅成功 import 不能证明隔离正确。
本方案解决同一进程内这两套受控 TVM 的共存问题，不是任意本地插件的安全隔离边界。
