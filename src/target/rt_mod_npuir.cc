// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "codegen_npuir.h"
#ifdef TILELANG_ENABLE_NPUIR_A5
#include "codegen_npuir_api_a5.h"
#include "codegen_npuir_dev_a5.h"
#else
#include "codegen_npuir_api.h"
#include "codegen_npuir_dev.h"
#endif

namespace tvm {
namespace codegen {

#ifndef TILELANG_ENABLE_NPUIR_A5
runtime::Module BuildTileLangNPUIR(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenTileLangNPUIR cg;
  cg.Init(output_ssa);

  Array<String> function_names;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangNPUIR: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetFunctionName(gvar));
  }

  std::string code = cg.Finish();

  return CSourceModuleCreate(code, "c", function_names);
}

/**
 * @brief Builds a runtime module containing TileLang NPU IR MLIR code for
 * Expert mode.
 *
 * This function takes an IRModule and target specification, generates MLIR code
 * using the CodeGenTileLangNPUIRAPI code generator, and creates a CSourceModule
 * suitable for deployment in TileLang's Expert mode. The Expert mode provides
 * low-level, performance-oriented APIs for advanced users who require
 * fine-grained control over NPU operations and memory management.
 *
 * @param mod The input IRModule containing PrimFuncs to be compiled.
 * @param target Not used yet.
 * @return runtime::Module A runtime module containing the generated MLIR code.
 */
runtime::Module BuildTileLangNPUIRMLIRAPIs(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  CodeGenTileLangNPUIRAPI cg;
  Array<String> function_names;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangNPUIRAPI: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetCurrentFunctionName());
  }
  std::string mlirCode = cg.Finish();
  return CSourceModuleCreate(mlirCode, "c", function_names);
}

/**
 * @brief Builds a runtime module containing TileLang NPU IR MLIR code for
 * Developer mode.
 *
 * This function takes an IRModule and target specification, generates MLIR code
 * using the CodeGenTileLangNPUIRDEV code generator, and creates a CSourceModule
 * suitable for use in TileLang's Developer mode. The Developer mode provides
 * higher-level abstractions and developer-friendly APIs that simplify NPU
 * programming while maintaining reasonable performance for application
 * development.
 *
 * @param mod The input IRModule containing PrimFuncs to be compiled.
 * @param target Not used yet.
 * @return runtime::Module A runtime module containing the generated MLIR code.
 */
runtime::Module BuildTileLangNPUIRMLIRDEV(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  CodeGenTileLangNPUIRDEV cg;
  Array<String> function_names;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangNPUIRDEV: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetCurrentFunctionName());
  }
  std::string mlirCode = cg.Finish();
  return CSourceModuleCreate(mlirCode, "c", function_names);
}
#endif // !TILELANG_ENABLE_NPUIR_A5

#ifdef TILELANG_ENABLE_NPUIR_A5
runtime::Module BuildTileLangNPUIRMLIRAPIsA5(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  CodeGenTileLangNPUIRAPIA5 cg;
  Array<String> function_names;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangNPUIRAPIA5: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetCurrentFunctionName());
  }
  std::string mlirCode = cg.Finish();
  return CSourceModuleCreate(mlirCode, "c", function_names);
}

/**
 * @brief Builds a runtime module containing TileLang NPU IR MLIR code for
 * Developer mode.
 *
 * This function takes an IRModule and target specification, generates MLIR code
 * using the CodeGenTileLangNPUIRDEV code generator, and creates a CSourceModule
 * suitable for use in TileLang's Developer mode. The Developer mode provides
 * higher-level abstractions and developer-friendly APIs that simplify NPU
 * programming while maintaining reasonable performance for application
 * development.
 *
 * @param mod The input IRModule containing PrimFuncs to be compiled.
 * @param target Not used yet.
 * @return runtime::Module A runtime module containing the generated MLIR code.
 */
runtime::Module BuildTileLangNPUIRMLIRDEVA5(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  CodeGenTileLangNPUIRDEVA5 cg;
  Array<String> function_names;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangNPUIRDEVA5: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetCurrentFunctionName());
  }
  std::string mlirCode = cg.Finish();
  return CSourceModuleCreate(mlirCode, "c", function_names);
}
#endif // TILELANG_ENABLE_NPUIR_A5

TVM_REGISTER_TARGET_KIND("npuir", kDLExtDev);

#ifdef TILELANG_ENABLE_NPUIR_A5
TVM_REGISTER_GLOBAL("target.build.tilelang_npuir_apis")
    .set_body_typed(BuildTileLangNPUIRMLIRAPIsA5);

TVM_REGISTER_GLOBAL("target.build.tilelang_npuir_dev")
    .set_body_typed(BuildTileLangNPUIRMLIRDEVA5);
#else
TVM_REGISTER_GLOBAL("target.build.tilelang_npuir")
    .set_body_typed(BuildTileLangNPUIR);

TVM_REGISTER_GLOBAL("target.build.tilelang_npuir_apis")
    .set_body_typed(BuildTileLangNPUIRMLIRAPIs);

TVM_REGISTER_GLOBAL("target.build.tilelang_npuir_dev")
    .set_body_typed(BuildTileLangNPUIRMLIRDEV);
#endif

} // namespace codegen
} // namespace tvm
