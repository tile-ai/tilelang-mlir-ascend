# ============================================================
# Build arguments for FROM tag construction.
# NOTE: Docker uses separate ARG scopes before and after FROM.
#       Pre-FROM ARGs are only visible in the FROM line itself.
#       To use them in ENV below, they must be redeclared
#       after FROM (lines 14-17). This is a Docker design
#       constraint, not a duplication bug.
# ============================================================
ARG CANN_VERSION=8.5.0
ARG CHIP_TYPE=a3
ARG OS=ubuntu
ARG OS_VERSION=22.04
ARG PY_VERSION=3.11

FROM quay.io/ascend/cann:${CANN_VERSION}-${CHIP_TYPE}-${OS}${OS_VERSION}-py${PY_VERSION} AS base

# Enable pipefail for robust error handling in pipelines
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG PY_VERSION=3.11
ARG REGION=ChinaMainland
ENV DEBIAN_FRONTEND=noninteractive \
    TZ="Asia/Shanghai" \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY}

# ============================================================
# Layer 1: System-level dependencies (stable, cached permanently)
# ============================================================
RUN apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    bash \
    ca-certificates \
    curl \
    git \
    gnupg2 \
    make \
    sudo \
    unzip \
    vim \
    wget && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

# Dead-snakes PPA for multi-Python version support (e.g. 3.9, 3.10 on top of base)
RUN curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xBA6932366A755776" \
    | gpg --dearmor > /usr/share/keyrings/deadsnakes-ppa.gpg  && \
    chmod 644 /usr/share/keyrings/deadsnakes-ppa.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/deadsnakes-ppa.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ppa.list

RUN apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    ccache \
    clang-15 \
    clang-format \
    cmake \
    lld-15 \
    zlib1g-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 \
    --slave /usr/bin/clang++ clang++ /usr/bin/clang++-15 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 100 \
    --slave /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-15

# Update cmake to latest via Kitware repository
RUN apt remove --purge --auto-remove --yes cmake && \
    apt-get update && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" | tee /etc/apt/sources.list.d/kitware.list && \
    apt-get update && \
    apt-get install -y cmake && \
    apt-get install -y libzstd-dev && \
    apt-get install -y patch && \
    rm -rf /var/lib/apt/lists/* && \
    echo "cmake version: $(cmake --version | head -n 1)"

WORKDIR /root

COPY . tilelang-ascend/
# ============================================================
# Layer 2: Python dependencies (cache until requirements files change)
#           AscendNPU-IR build depends on some Python packages
#           (e.g. pybind11, build), so this layer must come BEFORE
#           the submodule build layer.
# ============================================================
RUN python${PY_VERSION} -m pip install "ninja>=1.12.0" && \
    python${PY_VERSION} -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu && \
    python${PY_VERSION} -m pip install "torch_npu==2.7.1" && \
    python${PY_VERSION} -m pip install -r tilelang-ascend/requirements-dev.txt && \
    python${PY_VERSION} -m pip install -r tilelang-ascend/requirements.txt && \
    rm -rf /root/.cache/pip

# ============================================================
# Layer 3: Build AscendNPU-IR submodule (cached until .gitmodules changes).
#           Avoids re-downloading and recompiling AscendNPU-IR
#           on every source code change.
# ============================================================

# Step 3a: Initialize and checkout the AscendNPU-IR submodule only
RUN cd tilelang-ascend && \
    git config --global --add safe.directory /root/tilelang-ascend && \
    git submodule init 3rdparty/AscendNPU-IR && \
    git submodule update --init 3rdparty/AscendNPU-IR

# Step 3b: Pass 1 — Replace llvm URL in AscendNPU-IR/.gitmodules
#          (which lists llvm-project and torch-mlir as submodules)
RUN cd tilelang-ascend/3rdparty/AscendNPU-IR && \
    if [ "$REGION" = "ChinaMainland" ]; then \
    find . -name ".gitmodules" -type f -exec sed -i 's|https://github.com/llvm/llvm-project.git|https://mirrors.tuna.tsinghua.edu.cn/git/llvm-project.git|g' {} + ; \
    fi

# Step 3c: Sync and download llvm-project + torch-mlir submodules with retry
RUN cd tilelang-ascend/3rdparty/AscendNPU-IR && \
    git submodule sync && \
    git config --global http.postBuffer 524288000 && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 999999 && \
    for i in $(seq 1 5); do \
    git submodule update --init --jobs 1 && break || \
    (echo "Submodule clone attempt $i failed, retrying in 10s..." && sleep 10); \
    done && \
    git config --global --unset http.postBuffer || true && \
    git config --global --unset http.lowSpeedLimit || true && \
    git config --global --unset http.lowSpeedTime || true

# Step 3d: Pass 2 — Now torch-mlir/.gitmodules is available,
#          replace its nested llvm-project URL too
RUN cd tilelang-ascend/3rdparty/AscendNPU-IR && \
    if [ "$REGION" = "ChinaMainland" ]; then \
    find . -name ".gitmodules" -type f -exec sed -i 's|https://github.com/llvm/llvm-project.git|https://mirrors.tuna.tsinghua.edu.cn/git/llvm-project.git|g' {} + ; \
    fi

# Step 3e: Build AscendNPU-IR
#          Suppress compiler warnings to avoid hitting GitHub Actions 2MiB log limit.
#          LLVM/MLIR produces massive -Wsign-compare / -Wformat-nonliteral noise
#          that fills the log buffer and makes the build appear hung.
RUN cd tilelang-ascend/3rdparty/AscendNPU-IR && \
    bash ./build-tools/apply_patches.sh && \
    rm -rf ./build && \
    ./build-tools/build.sh -o ./build --python-binding --c-compiler=clang --cxx-compiler=clang++ \
    --add-cmake-options="-DCMAKE_LINKER=lld -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_RTTI=ON -DCMAKE_C_FLAGS=-w -DCMAKE_CXX_FLAGS=-w" \
    --apply-patches --bishengir-publish=off

# Step 3f: Save build artifacts and clean source to reduce final image size
RUN mkdir -p /build/AscendNPU-IR/build && \
    cp -r tilelang-ascend/3rdparty/AscendNPU-IR/build/* /build/AscendNPU-IR/build/ && \
    rm -rf tilelang-ascend/3rdparty/AscendNPU-IR

# ============================================================
# Layer 4: Full source code + TileLang installation (rebuilt on source change)
#           install_npuir.sh --bishengir-path skips AscendNPU-IR build
#           and reuses /build/AscendNPU-IR/build/install artifacts.
# ============================================================
RUN cd tilelang-ascend && \
    bash install_npuir.sh --bishengir-path=/build/AscendNPU-IR/build/install && \
    rm -rf 3rdparty/AscendNPU-IR

# ============================================================
# Layer 5: Final container configuration
# ============================================================
# Set CANN environment variables explicitly (ENV is more reliable than .bashrc sourcing)
ENV ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
RUN echo "source /usr/local/Ascend/cann/set_env.sh" >> /root/.bashrc

# Add welcome message for container usage
RUN echo "Welcome to TileLang AscendNPU IR Docker container!" > /etc/motd && \
    echo "" >> /etc/motd && \
    echo "TileLang URL: https://github.com/tile-ai/tilelang-mlir-ascend" >> /etc/motd && \
    echo "TileLang URL: https://github.com/tile-ai/tilelang-ascend/tree/npuir" >> /etc/motd && \
    echo "" >> /etc/motd && \
    echo "Pre-compiled AscendNPU-IR binaries are available at /build/AscendNPU-IR/build/" >> /etc/motd && \
    echo "To reinstall or update NPU IR with pre-compiled binaries (faster), use:" >> /etc/motd && \
    echo "  bash install_npuir.sh --bishengir-path=/build/AscendNPU-IR/build/install" >> /etc/motd && \
    echo "" >> /etc/motd

# Show welcome message when entering an interactive bash shell via docker exec.
RUN grep -q "TileLang MOTD" /etc/bash.bashrc || \
    printf '\n# TileLang MOTD\nif [[ $- == *i* ]] && [[ -f /etc/motd ]]; then\n  cat /etc/motd\nfi\n' >> /etc/bash.bashrc
