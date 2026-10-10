#ifndef TVM_TL_TILE_RUNTIME_NPU_PROFILER_H_
#define TVM_TL_TILE_RUNTIME_NPU_PROFILER_H_

#include <cstdint>

namespace tvm {
namespace tile {

class NpuProfilerProbe {
 public:
  NpuProfilerProbe(const char *kernel_name, void *function,
                   uint32_t block_dim);
  ~NpuProfilerProbe();

  NpuProfilerProbe(const NpuProfilerProbe &) = delete;
  NpuProfilerProbe &operator=(const NpuProfilerProbe &) = delete;

 private:
  const char *kernel_name_{nullptr};
  void *function_{nullptr};
  uint32_t block_dim_{0};
  uint64_t start_time_{0};
  bool active_{false};
};

}  // namespace tile
}  // namespace tvm

#endif  // TVM_TL_TILE_RUNTIME_NPU_PROFILER_H_
