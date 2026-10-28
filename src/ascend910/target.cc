/*!
 * \file src/ascend910/target.cc
 * \brief Ascend 910 (A2) target kind registration.
 *
 * Kept deliberately separate from src/ascend/target.cc: A2 and A5 are distinct
 * code-generation backends and must never share target-kind state.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include "dlpack/dlpack.h"

namespace tvm {

namespace refl = tvm::ffi::reflection;

TVM_REGISTER_TARGET_KIND("ascend910", kDLExtDev)
    .add_attr_option<ffi::String>("mcpu")
    .add_attr_option<ffi::String>("arch")
    .add_attr_option<bool>("supports_kernel_status_return",
                           refl::DefaultValue(false))
    .add_attr_option<bool>("supports_vector_predicate",
                           refl::DefaultValue(true))
    .add_attr_option<int64_t>("max_vector_bits", refl::DefaultValue(64))
    .add_attr_option<bool>("launch_grid_only", refl::DefaultValue(true))
    .set_default_keys({"ascend910"});

// Pass configuration options consumed by the Ascend 910 JIT (they map onto
// bishengir-compile flags).  Registering them keeps ``pass_configs`` usable
// from ``tilelang.compile``.
TVM_REGISTER_PASS_CONFIG_OPTION("tl.npuir_enable_auto_multi_buffer", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tl.npuir_disable_hivm_auto_inject_sync", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tl.enable_plan_and_update_buffer_allocation", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("npuir.enable_auto_multi_buffer", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("npuir.disable_hivm_auto_inject_sync", Bool);

} // namespace tvm
