# TileLang Ascend 950 image

The release image contains TileLang **0.1.15**, CANN **9.3.0**
(`9.3.0~weekly.20260916.01`), Torch **2.10.0+cpu**, Torch-NPU **2.10.0**,
Python **3.11** and Ubuntu **22.04** for **linux/amd64**:

```text
quay.io/ascend/tilelang:0.1.15-950-ubuntu22.04-py3.11-x86_64
```

## Release configuration

The build is not ready to publish until the CANN checksums and installation paths
in [tilelang/release.json](tilelang/release.json) have been verified. Empty fields
are intentional: preflight fails before downloading the source or building an
image. There are no release-time version or download URL inputs.

Before publishing, commit the verified values for:

- SHA256 checksums for the pinned CANN x86_64 Toolkit and 950 ops downloads.
  The fixed URLs select batch `20260916000323606`; confirm they are reachable
  from the runner. The preflight rejects aarch64 packages and other weekly builds.
- Any additional required CANN `.run` packages, appended to `cann_packages` in
  installation order. Toolkit is installed first, then 950 ops.
- The CANN home, environment script, compiler home (containing `bin/bisheng`
  and the CCE-capable `bin/ld.lld`) and installed version metadata file.
  The version file must identify `9.3.0` or `9.3.0~weekly.20260916.01` with a `version=`, `Version=` or
  `CANN_VERSION=` line (a colon separator is also accepted). Verify these
  paths and the metadata format against the supplied installation bundle.

Torch `2.10.0+cpu` and Torch-NPU `2.10.0` Python 3.11 x86_64 wheel URLs and
SHA256 checksums are fixed from the PyTorch CPU index and PyPI metadata. The
Torch-NPU wheel requires `torch==2.10.0`, which accepts the CPU local version.

The source is pinned to `v0.1.15`, commit
`a35f8ddf45eba16c21211ec8822d56ce5363036f`, with its recursive submodules.
The image tag version is read from that checkout's `VERSION` file.

Validate the fixed inputs locally:

```bash
python3 docker/tilelang/release.py validate
```

## Build and publish

1. Merge the independent **Build TileLang Docker** workflow into the default
   branch so that GitHub exposes its manual dispatch entry point.
2. Ensure `tilelang-docker-image-release` contains the reviewed workflow,
   Dockerfile and completed release configuration.
3. Ensure repository Actions secrets `QUAY_USERNAME` and `QUAY_PASSWORD` can
   publish to `quay.io/ascend/tilelang`.
4. In Actions, select **Build TileLang Docker**, choose **Run workflow**, select
   **tilelang-docker-image-release**, and start the run. Other branches fail the
   release-branch check. Pushes, pull requests and releases do not trigger it.

The workflow builds and loads one amd64 image, validates it without an NPU,
then pushes that same image. Publications to the fixed tag are serialized.
Before replacing an existing tag, the workflow records its previous digest.
It pulls the published tag and checks that its image ID and architecture still
match the tested image. The Actions summary records the new digest and versions;
the `tilelang-950-image-validation` artifact contains verification logs.

The image builds the Ascend backend with CUDA, ROCm and LLVM disabled. Source
and examples are retained in `/opt/tilelang`; TileLang is installed from a wheel
in `/opt/venv`, not as an editable package. CANN compilers, headers and runtime
libraries remain available for JIT compilation. Compilation uses two CPU jobs.

## Validation boundary

The required checks cover OS/Python/architecture, installed package versions,
the compiled Ascend backend, CANN installation metadata, and a `dav-3510` device
kernel compiled and linked into an executable ELF. They run outside the source
directory through both the container entrypoint and `docker exec`.

CI disables Torch's automatic device-plugin loading only for its no-device
validation containers. It verifies Torch-NPU package metadata without importing
or initializing the NPU runtime. Normal containers retain Torch's default device
autoload behavior. No NPU availability, execution, accuracy or performance claim
is made by these checks.

## Use the image

```bash
docker pull quay.io/ascend/tilelang:0.1.15-950-ubuntu22.04-py3.11-x86_64

# Inspect the installation on a machine without an NPU.
docker run --rm -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  quay.io/ascend/tilelang:0.1.15-950-ubuntu22.04-py3.11-x86_64 \
  python /opt/tilelang-image/smoke.py
```

For NPU execution, use an x86_64 Ascend 950 host with a CANN-compatible driver.
Expose the host's NPU devices and driver libraries using its Ascend container
runtime configuration. The image supplies the toolkit; the host supplies the
driver. Then check `import torch; import torch_npu; torch.npu.is_available()`
and run the examples under `/opt/tilelang/examples/ascend`.

To restore a previously recorded image, retag and push its digest:

```bash
IMAGE=quay.io/ascend/tilelang:0.1.15-950-ubuntu22.04-py3.11-x86_64
PREVIOUS_DIGEST=sha256:REPLACE_WITH_RECORDED_DIGEST
docker pull "quay.io/ascend/tilelang@$PREVIOUS_DIGEST"
docker tag "quay.io/ascend/tilelang@$PREVIOUS_DIGEST" "$IMAGE"
docker push "$IMAGE"
```

# GPU development images

To ease the process of installing all the dependencies, we provide a Dockerfile and a simple guideline to build a Docker image with all of above installed. The Docker image is built on top of Ubuntu 20.04, and it contains all the dependencies required to run the experiments. We only provide the Dockerfile for NVIDIA GPU, and the Dockerfile for AMD GPU will be provided upon request.

```bash
git clone --recursive https://github.com/tile-ai/tilelang TileLang
cd TileLang/docker
# build the image, this may take a while (around 10+ minutes on our test machine)
# replace the version number cu124 with the one you want to use
# replace .cu** with .rocm for AMD GPU
docker build -t tilelang_workspace -f Dockerfile.cu124 .
# run the container
# if it's nvidia
docker run -it --cap-add=SYS_ADMIN --network=host --gpus all --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name tilelang_test tilelang_workspace bash
# if it's amd
docker run -it --cap-add=SYS_ADMIN --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name tilelang_test tilelang_workspace bash
```
