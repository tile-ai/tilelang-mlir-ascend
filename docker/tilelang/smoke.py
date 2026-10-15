"""Check the installed image and compile a 950 device binary without a device."""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from release import CONFIG, check_cann, validate


def main():
    config = json.loads(CONFIG.read_text())
    validate(config)
    check_cann(config)
    assert sys.version_info[:2] == (3, 11), sys.version
    assert platform.machine() == "x86_64", platform.machine()
    os_release = platform.freedesktop_os_release()
    assert (os_release["ID"], os_release["VERSION_ID"]) == ("ubuntu", "22.04")
    assert (
        Path("/opt/tilelang/.git_commit.txt").read_text().strip()
        == config["tilelang_commit"]
    )

    import tilelang
    import torch
    from tilelang.contrib import bisheng
    from tilelang.transform import PassConfigKey
    from tvm.target import Target
    from tvm.transform import PassContext

    assert not Path(tilelang.__file__).resolve().is_relative_to("/opt/tilelang"), (
        tilelang.__file__
    )
    assert tilelang.__version__ == config["tilelang_version"], tilelang.__version__
    assert importlib.metadata.version("tilelang") == config["tilelang_version"]
    for name in ("torch", "torch_npu"):
        assert importlib.metadata.version(name) == config[name]["version"], name
    assert torch.version.cuda is None, (
        "Expected a CPU Torch wheel paired with torch_npu"
    )
    # This option is registered only when the Ascend backend is compiled.
    with PassContext(config={PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: True}):
        pass
    assert bisheng.get_target_npu_arch(Target("ascend")) == "dav-3510"
    for executable in (bisheng.find_bisheng_path(), bisheng.find_ld_lld_path()):
        assert (
            Path(executable)
            .resolve()
            .is_relative_to(Path(config["bisheng_home"]).resolve())
        ), executable
        subprocess.run([executable, "--version"], check=True)

    # This uses the same device compiler/linker entry point as TileLang's JIT.
    # No ACL initialization, device allocation or kernel launch is performed.
    binary = bisheng.compile_ascend(
        'extern "C" __global__ __vector__ void tilelang_image_smoke() {}',
        target_format="aibin",
        npu_arch="dav-3510",
        verbose=True,
    )
    assert binary[:4] == b"\x7fELF", "Expected a linked CCE ELF"
    # ELF e_type == ET_EXEC (2), rather than an unlinked ET_REL object (1).
    byteorder = "little" if binary[5] == 1 else "big"
    assert int.from_bytes(binary[16:18], byteorder) == 2, (
        "Expected an executable CCE ELF"
    )
    print(
        json.dumps(
            {
                "tilelang": tilelang.__version__,
                "commit": config["tilelang_commit"],
                "cann": config["cann_version"],
                "cann_package": config["cann_package_version"],
                "torch": importlib.metadata.version("torch"),
                "torch_npu": importlib.metadata.version("torch_npu"),
                "arch": os.environ["ASCEND_NPU_ARCH"],
                "validation": "CPU-only installation and device compilation; no NPU execution",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
