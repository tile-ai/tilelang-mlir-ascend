/*!
 * \file tile_obj_launcher.cc
 * \brief Bind and launch compiler-owned Tile backend object bytes.
 *
 * Device compilation is owned by the Tile JIT. This file implements the
 * runtime half:
 *
 *   1. Lazily dlopen("libascendcl.so") and dlsym the ACL entry points,
 *      so TileLang keeps no CANN build-time dependency.
 *   2. Bind object bytes and their ABI once, then lazily load and resolve the
 *      kernel function per device.
 *   3. Pack arguments (int64-encoded on the Python side) into a contiguous
 *      aligned buffer.  The pack layout is computed once per kernel and
 *      cached; small kernels use a stack-backed buffer so steady-state
 *      launches do not allocate.
 *   4. Submit with aclrtLaunchKernelWithHostArgs. The block count and dynamic
 *      UBUF size are supplied by the compiler-produced launch specification.
 *
 * Python-side entry points:
 *   - tl.tile.BindKernel(object_bytes, name, arg_types) returns a token.
 *   - tl.tile.LaunchKernel(token, ..., args) submits a bound kernel.
 *
 *   - arg_types[i] is one of:
 *       "handle", "int8", "int16", "int32", "int64",
 *       "uint8", "uint16", "uint32", "uint64", "float32", "float64"
 *   - args[i] is one int64 per argument:
 *       handle  -> raw device pointer value
 *       integer -> the value
 *       float32 -> IEEE-754 bit pattern in the low 32 bits
 *       float64 -> IEEE-754 bit pattern
 *   - stream is the raw NPU stream handle (0 = ACL default stream).
 */
#ifdef TILELANG_CANN_PROFILER_AVAILABLE
#include "tile_npu_profiler.h"
#endif

#include <dlfcn.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tile {
namespace {

using AclError = int32_t;
using AclBinHandle = void *;
using AclFuncHandle = void *;
using AclStream = void *;

constexpr AclError kAclSuccess = 0;
constexpr int32_t kAclLaunchKernelAttrDynUbufSize = 2;
// Mirrors tilelang-ascend-cce's runtime: every scalar kernel argument is
// aligned to at least 4 bytes, and the whole packed buffer to 8 bytes.
constexpr size_t kAclArgMinAlignment = 4;
constexpr size_t kAclArgBufferAlignment = 8;

size_t AlignUp(size_t value, size_t alignment) {
  TVM_FFI_ICHECK_NE(alignment, 0U);
  TVM_FFI_ICHECK_EQ(alignment & (alignment - 1), 0U)
      << "Alignment must be a power of two, got " << alignment;
  return (value + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// Argument kinds and packing
// ---------------------------------------------------------------------------

enum class AclArgKind {
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kUInt8,
  kUInt16,
  kUInt32,
  kUInt64,
  kFloat32,
  kFloat64,
  kHandle,
};

AclArgKind ParseArgKind(const std::string &type, size_t index) {
  if (type == "handle") return AclArgKind::kHandle;
  if (type == "int8") return AclArgKind::kInt8;
  if (type == "int16") return AclArgKind::kInt16;
  if (type == "int32") return AclArgKind::kInt32;
  if (type == "int64") return AclArgKind::kInt64;
  if (type == "uint8") return AclArgKind::kUInt8;
  if (type == "uint16") return AclArgKind::kUInt16;
  if (type == "uint32") return AclArgKind::kUInt32;
  if (type == "uint64") return AclArgKind::kUInt64;
  if (type == "float32") return AclArgKind::kFloat32;
  if (type == "float64") return AclArgKind::kFloat64;
  TVM_FFI_THROW(RuntimeError) << "Unsupported tile arg type `" << type
                              << "` at argument " << index;
  TVM_FFI_UNREACHABLE();
}

size_t GetAclArgSize(AclArgKind kind) {
  switch (kind) {
  case AclArgKind::kInt8:
  case AclArgKind::kUInt8:
    return 1;
  case AclArgKind::kInt16:
  case AclArgKind::kUInt16:
    return 2;
  case AclArgKind::kInt32:
  case AclArgKind::kUInt32:
  case AclArgKind::kFloat32:
    return 4;
  case AclArgKind::kInt64:
  case AclArgKind::kUInt64:
  case AclArgKind::kFloat64:
  case AclArgKind::kHandle:
    return 8;
  }
  TVM_FFI_UNREACHABLE();
}

size_t GetAclArgAlignment(AclArgKind kind) {
  size_t alignment = 0;
  switch (kind) {
  case AclArgKind::kInt8:
    alignment = alignof(int8_t);
    break;
  case AclArgKind::kInt16:
    alignment = alignof(int16_t);
    break;
  case AclArgKind::kInt32:
    alignment = alignof(int32_t);
    break;
  case AclArgKind::kInt64:
    alignment = alignof(int64_t);
    break;
  case AclArgKind::kUInt8:
    alignment = alignof(uint8_t);
    break;
  case AclArgKind::kUInt16:
    alignment = alignof(uint16_t);
    break;
  case AclArgKind::kUInt32:
    alignment = alignof(uint32_t);
    break;
  case AclArgKind::kUInt64:
    alignment = alignof(uint64_t);
    break;
  case AclArgKind::kFloat32:
    alignment = alignof(float);
    break;
  case AclArgKind::kFloat64:
    alignment = alignof(double);
    break;
  case AclArgKind::kHandle:
    alignment = alignof(void *);
    break;
  }
  return std::max(alignment, kAclArgMinAlignment);
}

struct AclArgLayout {
  AclArgKind kind;
  size_t offset;
};

struct AclArgPackPlan {
  std::vector<AclArgLayout> args;
  size_t buffer_size{0};
};

// Computed once when object bytes are bound; each launch only fills the buffer.
// Mirrors MakeAclArgPackPlan in tilelang-ascend-cce's
// ascend_module.cc.
AclArgPackPlan MakeAclArgPackPlan(const ffi::Array<ffi::String> &arg_types) {
  AclArgPackPlan plan;
  plan.args.reserve(arg_types.size());
  size_t offset = 0;
  for (size_t i = 0; i < arg_types.size(); ++i) {
    AclArgKind kind = ParseArgKind(arg_types[i].operator std::string(), i);
    offset = AlignUp(offset, GetAclArgAlignment(kind));
    plan.args.push_back({kind, offset});
    offset += GetAclArgSize(kind);
  }
  plan.buffer_size = AlignUp(offset, kAclArgBufferAlignment);
  return plan;
}

void PackArg(AclArgKind kind, int64_t value, uint8_t *dst) {
  switch (kind) {
  case AclArgKind::kInt8:
  case AclArgKind::kUInt8: {
    uint8_t v = static_cast<uint8_t>(value);
    std::memcpy(dst, &v, sizeof(v));
    break;
  }
  case AclArgKind::kInt16:
  case AclArgKind::kUInt16: {
    uint16_t v = static_cast<uint16_t>(value);
    std::memcpy(dst, &v, sizeof(v));
    break;
  }
  case AclArgKind::kInt32:
  case AclArgKind::kUInt32:
  case AclArgKind::kFloat32: {
    uint32_t v = static_cast<uint32_t>(value);
    std::memcpy(dst, &v, sizeof(v));
    break;
  }
  case AclArgKind::kInt64:
  case AclArgKind::kUInt64:
  case AclArgKind::kFloat64:
  case AclArgKind::kHandle: {
    std::memcpy(dst, &value, sizeof(value));
    break;
  }
  }
}

// Stack-backed packing buffer for small kernels, heap-backed beyond
// kNumWords == 0.  Mirrors AclArgBuffer in tilelang-ascend-cce so the common
// case (<= 16 words, i.e. <= 128 bytes of arguments) never allocates.
template <size_t kNumWords> class AclArgBuffer {
public:
  explicit AclArgBuffer(size_t) {}

  uint8_t *data() { return reinterpret_cast<uint8_t *>(storage_.data()); }

private:
  alignas(kAclArgBufferAlignment) std::array<uint64_t, kNumWords> storage_{};
};

template <> class AclArgBuffer<0> {
public:
  explicit AclArgBuffer(size_t num_words) : storage_(num_words, 0) {}

  uint8_t *data() { return reinterpret_cast<uint8_t *>(storage_.data()); }

private:
  std::vector<uint64_t> storage_;
};

// ---------------------------------------------------------------------------
// ACL driver: lazy dlopen + dlsym
// ---------------------------------------------------------------------------

union AclLaunchKernelAttrValue {
  uint8_t schem_mode;
  uint32_t dyn_ubuf_size;
  uint32_t engine_type;
  uint32_t block_dim_offset;
  uint8_t is_block_task_prefetch;
  uint8_t is_data_dump;
  uint16_t timeout;
  uint32_t reserved[4];
};

struct AclLaunchKernelAttr {
  int32_t id;
  AclLaunchKernelAttrValue value;
};

struct AclLaunchKernelCfg {
  AclLaunchKernelAttr *attrs;
  size_t num_attrs;
};

class AclDriver {
public:
  static AclDriver *Global() {
    static auto *driver = new AclDriver();
    return driver;
  }

  AclError BinaryLoadFromData(const void *data, size_t size,
                              AclBinHandle *handle) const {
    return binary_load_from_data_(data, size, nullptr, handle);
  }

  AclError BinaryGetFunction(AclBinHandle binary, const char *name,
                             AclFuncHandle *function) const {
    return binary_get_function_(binary, name, function);
  }

  AclError GetDevice(int32_t *device_id) const { return get_device_(device_id); }

  AclError LaunchKernelWithHostArgs(AclFuncHandle function, uint32_t num_blocks,
                                    AclStream stream,
                                    AclLaunchKernelCfg *config, void *args,
                                    size_t args_size) const {
    return launch_kernel_with_host_args_(function, num_blocks, stream, config,
                                         args, args_size, nullptr, 0);
  }

  const char *GetRecentErrorMessage() const {
    return get_recent_error_message_ == nullptr
               ? nullptr
               : get_recent_error_message_();
  }

private:
  using BinaryLoadFromDataFn = AclError (*)(const void *, size_t, const void *,
                                            AclBinHandle *);
  using BinaryGetFunctionFn = AclError (*)(AclBinHandle, const char *,
                                           AclFuncHandle *);
  using GetDeviceFn = AclError (*)(int32_t *);
  using LaunchKernelWithHostArgsFn = AclError (*)(AclFuncHandle, uint32_t,
                                                  AclStream,
                                                  AclLaunchKernelCfg *, void *,
                                                  size_t, void *, size_t);
  using GetRecentErrorMessageFn = const char *(*)();

  AclDriver() {
    library_ = dlopen("libascendcl.so", RTLD_LAZY | RTLD_LOCAL);
    TVM_FFI_CHECK(library_ != nullptr, RuntimeError)
        << "Tile runtime could not load libascendcl.so: " << dlerror();
    binary_load_from_data_ =
        LoadSymbol<BinaryLoadFromDataFn>("aclrtBinaryLoadFromData");
    binary_get_function_ =
        LoadSymbol<BinaryGetFunctionFn>("aclrtBinaryGetFunction");
    get_device_ = LoadSymbol<GetDeviceFn>("aclrtGetDevice");
    launch_kernel_with_host_args_ =
        LoadSymbol<LaunchKernelWithHostArgsFn>("aclrtLaunchKernelWithHostArgs");
    get_recent_error_message_ =
        LoadSymbol<GetRecentErrorMessageFn>("aclGetRecentErrMsg");
  }

  template <typename FunctionType> FunctionType LoadSymbol(const char *name) {
    dlerror();
    void *symbol = dlsym(library_, name);
    const char *error = dlerror();
    TVM_FFI_CHECK(symbol != nullptr && error == nullptr, RuntimeError)
        << "Tile runtime could not resolve " << name
        << " from libascendcl.so: "
        << (error == nullptr ? "symbol not found" : error);
    return reinterpret_cast<FunctionType>(symbol);
  }

  void *library_{nullptr};
  BinaryLoadFromDataFn binary_load_from_data_{nullptr};
  BinaryGetFunctionFn binary_get_function_{nullptr};
  GetDeviceFn get_device_{nullptr};
  LaunchKernelWithHostArgsFn launch_kernel_with_host_args_{nullptr};
  GetRecentErrorMessageFn get_recent_error_message_{nullptr};
};

void CheckAcl(AclError result, const char *operation) {
  if (result == kAclSuccess) return;
  const char *message = AclDriver::Global()->GetRecentErrorMessage();
  TVM_FFI_THROW(RuntimeError)
      << operation << " failed with ACL error " << result
      << (message == nullptr ? "" : std::string(": ") + message);
}

// ---------------------------------------------------------------------------
// Bound kernel registry (process lifetime)
// ---------------------------------------------------------------------------

using BoundKernelToken = int64_t;

struct BoundKernel {
  std::string object_bytes;
  std::string kernel_name;
  AclArgPackPlan pack_plan;
  std::unordered_map<int32_t, AclBinHandle> binaries;
  std::unordered_map<int32_t, AclFuncHandle> functions;
};

struct BoundLaunch {
  const std::string *kernel_name;
  const AclArgPackPlan *pack_plan;
  AclFuncHandle function;
};

class AclKernelRegistry {
public:
  static AclKernelRegistry *Global() {
    static auto *registry = new AclKernelRegistry();
    return registry;
  }

  BoundKernelToken Bind(ffi::Bytes object_bytes, ffi::String kernel_name,
                        const ffi::Array<ffi::String> &arg_types) {
    std::string symbol = kernel_name.operator std::string();

    auto kernel = std::make_unique<BoundKernel>();
    kernel->object_bytes.assign(object_bytes.data(), object_bytes.size());
    kernel->kernel_name = std::move(symbol);
    kernel->pack_plan = MakeAclArgPackPlan(arg_types);

    std::lock_guard<std::mutex> lock(mutex_);
    TVM_FFI_ICHECK_LT(next_token_, std::numeric_limits<BoundKernelToken>::max())
        << "Tile bound-kernel token space exhausted";
    BoundKernelToken token = next_token_++;
    kernels_.emplace(token, std::move(kernel));
    return token;
  }

  BoundLaunch GetLaunch(BoundKernelToken token, int32_t device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = kernels_.find(token);
    TVM_FFI_ICHECK(found != kernels_.end())
        << "Unknown Tile bound-kernel token " << token;

    BoundKernel *kernel = found->second.get();
    AclDriver *driver = AclDriver::Global();
    AclBinHandle &binary = kernel->binaries[device_id];
    if (binary == nullptr) {
      CheckAcl(driver->BinaryLoadFromData(kernel->object_bytes.data(),
                                          kernel->object_bytes.size(), &binary),
               "aclrtBinaryLoadFromData");
    }
    AclFuncHandle &function = kernel->functions[device_id];
    if (function == nullptr) {
      CheckAcl(driver->BinaryGetFunction(binary, kernel->kernel_name.c_str(),
                                          &function),
               "aclrtBinaryGetFunction");
    }
    return {&kernel->kernel_name, &kernel->pack_plan, function};
  }

private:
  std::mutex mutex_;
  BoundKernelToken next_token_{1};
  std::unordered_map<BoundKernelToken, std::unique_ptr<BoundKernel>> kernels_;
};

// ---------------------------------------------------------------------------
// The launch entry point
// ---------------------------------------------------------------------------

// Fills a stack- or heap-backed argument buffer and submits the kernel.
// Returns the raw ACL error code; error reporting stays in the launch entry
// point so block-count and dynamic-UBUF context are attached in one place.
template <size_t kNumWords>
AclError PackAndLaunch(AclDriver *driver, const char *kernel_name,
                       AclFuncHandle function, uint32_t num_blocks,
                       AclStream stream, AclLaunchKernelCfg *config,
                       const AclArgPackPlan &plan,
                       const ffi::Array<int64_t> &args) {
  AclArgBuffer<kNumWords> buffer(plan.buffer_size / sizeof(uint64_t));
  uint8_t *base = buffer.data();
  for (size_t i = 0; i < plan.args.size(); ++i) {
    PackArg(plan.args[i].kind, args[i], base + plan.args[i].offset);
  }

#ifdef TILELANG_CANN_PROFILER_AVAILABLE
  NpuProfilerProbe profiler_probe(kernel_name, function, num_blocks);
#endif

  return driver->LaunchKernelWithHostArgs(function, num_blocks, stream,
                                          config, base, plan.buffer_size);
}

BoundKernelToken BindTileObjectKernel(
    ffi::Bytes object_bytes, ffi::String kernel_name,
    ffi::Array<ffi::String> arg_types) {
  return AclKernelRegistry::Global()->Bind(object_bytes, kernel_name, arg_types);
}

void LaunchTileObjectKernel(BoundKernelToken kernel_token, int64_t block_count,
                            int64_t dynamic_ubuf_bytes, uint64_t stream,
                            ffi::Array<int64_t> args) {
  // Static launch values were validated when TileLaunchSpec was built. Keep
  // only the structural checks needed to pack this FFI call safely.
  AclDriver *driver = AclDriver::Global();
  int32_t device_id = 0;
  CheckAcl(driver->GetDevice(&device_id), "aclrtGetDevice");
  BoundLaunch bound =
      AclKernelRegistry::Global()->GetLaunch(kernel_token, device_id);
  TVM_FFI_ICHECK_EQ(args.size(), bound.pack_plan->args.size())
      << "Tile kernel `" << *bound.kernel_name << "` expects "
      << bound.pack_plan->args.size() << " arguments but got " << args.size();

  AclLaunchKernelAttr attribute{};
  AclLaunchKernelCfg config{};
  AclLaunchKernelCfg *config_ptr = nullptr;
  if (dynamic_ubuf_bytes > 0) {
    attribute.id = kAclLaunchKernelAttrDynUbufSize;
    attribute.value.dyn_ubuf_size =
        static_cast<uint32_t>(dynamic_ubuf_bytes);
    config.attrs = &attribute;
    config.num_attrs = 1;
    config_ptr = &config;
  }

  AclError result;
  size_t num_words = bound.pack_plan->buffer_size / sizeof(uint64_t);
  if (num_words <= 4) {
    result = PackAndLaunch<4>(
        driver, bound.kernel_name->c_str(), bound.function,
        static_cast<uint32_t>(block_count),
        reinterpret_cast<AclStream>(stream), config_ptr, *bound.pack_plan,
        args);
  } else if (num_words <= 8) {
    result = PackAndLaunch<8>(
        driver, bound.kernel_name->c_str(), bound.function,
        static_cast<uint32_t>(block_count),
        reinterpret_cast<AclStream>(stream), config_ptr, *bound.pack_plan,
        args);
  } else if (num_words <= 16) {
    result = PackAndLaunch<16>(
        driver, bound.kernel_name->c_str(), bound.function,
        static_cast<uint32_t>(block_count),
        reinterpret_cast<AclStream>(stream), config_ptr, *bound.pack_plan,
        args);
  } else {
    result = PackAndLaunch<0>(
        driver, bound.kernel_name->c_str(), bound.function,
        static_cast<uint32_t>(block_count),
        reinterpret_cast<AclStream>(stream), config_ptr, *bound.pack_plan,
        args);
  }

  if (result != kAclSuccess) {
    const char *message = driver->GetRecentErrorMessage();
    std::ostringstream error;
    error << "aclrtLaunchKernelWithHostArgs failed for " << *bound.kernel_name
          << " with ACL error " << result
          << ", block_count=" << block_count
          << ", dynamic_ubuf_bytes=" << dynamic_ubuf_bytes;
    if (message != nullptr) {
      error << ": " << message;
    }
    TVM_FFI_THROW(RuntimeError) << error.str();
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.tile.BindKernel", &BindTileObjectKernel)
      .def("tl.tile.LaunchKernel", &LaunchTileObjectKernel);
}

} // namespace
} // namespace tile
} // namespace tvm
