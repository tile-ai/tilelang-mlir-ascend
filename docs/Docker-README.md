# TileLang Ascend Docker 使用指南

本指南介绍如何使用 TileLang Ascend 项目的 Docker 环境进行开发和部署。

## 概述

此 Dockerfile 基于 Ubuntu 22.04，预装了 TileLang 项目及其依赖，包括 Ascend NPU 支持。容器已预编译 AscendNPU-IR 组件，并优化了镜像大小以便导出为 tar 文件。

## 组件依赖

容器基于 `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11` 构建，预装以下组件：

### 系统工具

`bash` `ca-certificates` `curl` `git` `gnupg2` `make` `sudo` `unzip` `vim` `wget`

### 编译工具链

| 组件 | 版本 / 说明 |
|------|-------------|
| Clang | 15（默认编译器） |
| LLD | 15（默认链接器） |
| CMake | latest（通过 Kitware 源安装） |
| ccache | 编译缓存加速 |
| zlib1g-dev | 压缩库 |
| libzstd-dev | Zstandard 压缩库 |
| Ninja | ≥1.12.0 |

### Python 环境

| 组件 | 版本 |
|------|------|
| Python | 3.11 |
| PyTorch | 2.7.1 (CPU) |
| torch_npu | 2.7.1 |
| TileLang | 源码编译安装 |
| AscendNPU-IR | 预编译（`/build/AscendNPU-IR/build/`） |
| 其他依赖 | 来自 `requirements.txt` + `requirements-dev.txt` |

### NPU 运行时

- **CANN**: 9.0.0（基础镜像内置）
- **AscendNPU-IR**: 预编译二进制

## 构建镜像

### 默认构建（中国大陆用户）

```bash
sudo docker build -f docker/Dockerfile -t tilelang-ascend .
```

此构建会：

- 使用清华镜像加速 Git 子模块下载
- 预编译 AscendNPU-IR 组件
- 清理源码以减小镜像体积

### 其他地区构建

如果您在中国大陆以外地区，可以指定地区参数跳过 URL 替换：

```bash
docker build --build-arg REGION=other -f docker/Dockerfile -t tilelang-ascend .
```

## 运行容器

### 交互式运行

```bash
docker run -it --rm tilelang-ascend
```

进入容器后，您会看到欢迎消息：

```shell
Welcome to TileLang AscendNPU IR Docker container!

TileLang URL: https://github.com/tile-ai/tilelang-mlir-ascend

Pre-compiled AscendNPU-IR binaries are available at /build/AscendNPU-IR/build/
To reinstall or update NPU IR with pre-compiled binaries (faster), use:
  bash install_npuir.sh --bishengir-path=/build/AscendNPU-IR/build/install
```

如需进入已运行容器，也可执行：

```bash
docker exec -it <container_name_or_id> bash
```

## 容器特性

### 预编译组件

- **AscendNPU-IR**: 已预编译并存储在 `/build/AscendNPU-IR/build/`
- **TileLang**: 已在`~/.bashrc`通过`PYTHONPATH` 配置，可直接 `import tilelang`

### 加速优化

- **Git 镜像**: 中国大陆默认使用清华镜像加速子模块下载
- **编译缓存**: 预编译二进制文件可复用，避免重复编译

## 使用预构建镜像（推荐）

TileLang Ascend 在 Ascend 官方镜像仓库中提供预构建的 Docker 镜像，用户可直接拉取使用，无需本地编译：

- **镜像仓库地址**：[https://quay.io/repository/ascend/tilelang](https://quay.io/repository/ascend/tilelang)

```bash
# 拉取最新镜像
docker pull quay.io/ascend/tilelang:latest

# 运行容器
docker run -it --rm quay.io/ascend/tilelang:latest
```

您可以在仓库页面选择所需的 tag（如特定 CPU 型号、Ascend 芯片型号等）进行拉取。

## 使用示例

### 1. 验证环境

```bash
# 进入容器后
python -c "import tilelang; print('TileLang imported successfully')"
```

### 2. 开发者源码开发

```bash
cd tilelang-ascend
# 修改代码后重新编译
bash install_npuir.sh --bishengir-path=/build/AscendNPU-IR/build/install
```

## 构建参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHIP_TYPE` | `A3` | Ascend 芯片类型 |
| `CANN_VERSION` | `9.0.0` | CANN 版本 |
| `REGION` | `ChinaMainland` | 地区设置（影响 Git 镜像） |

## 故障排除

### 构建失败

- 检查网络连接（特别是 Git 子模块下载）
- 确认系统支持 Docker 环境

### 运行时问题

- 确保主机有 Ascend 设备访问权限
- 检查 Python 版本兼容性

## 贡献

如果您发现问题或有改进建议，请提交 Issue 或 Pull Request。
