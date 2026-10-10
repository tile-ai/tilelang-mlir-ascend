#include "tile_npu_profiler.h"

#include <cstdlib>
#include <cstring>

namespace tvm {
namespace tile {
namespace {

extern "C" {

void AscendProfRegister();

bool GetAscendProfStatus();

void StartAscendProf(const char *name, uint64_t *start_time);

void ReportAscendProf(const char *name, uint32_t block_dim, uint32_t task_type, uint64_t start_time);

uint32_t AscendCGetProfkTypeImpl(void *function);

} // extern "C"

bool IsNpuProfilerEnabled() {
  const char *value = std::getenv("TILELANG_NPU_PROFILE");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

// CANN 官方生成代码在注册算子二进制时调用 AscendProfRegister
// Tilelang 的二进制加载流程不同，因此在运行库加载时注册一次
struct NpuProfilerRegistration {
  NpuProfilerRegistration() {
    AscendProfRegister();
  }
};

NpuProfilerRegistration registration;

} // namespace

NpuProfilerProbe::NpuProfilerProbe(const char *kernel_name, void *function, uint32_t block_dim) {
  if (!IsNpuProfilerEnabled()) {
    return;
  }

  kernel_name_ = kernel_name;
  function_ = function;
  block_dim_ = block_dim;

  StartAscendProf(kernel_name_, &start_time_);
  active_ = true;
}

NpuProfilerProbe::~NpuProfilerProbe() {
  if (!active_ || !GetAscendProfStatus()) {
    return;
  }

  uint32_t task_type = AscendCGetProfkTypeImpl(function_);
  ReportAscendProf(kernel_name_, block_dim_, task_type, start_time_);
}

} // namespace tile
} // namespace tvm
