// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "codegen_npuir.h"
#include "tir_to_tlir.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <unistd.h>
#ifdef TILELANG_ENABLE_NPUIR_A5
#include "codegen_npuir_api_a5.h"
#include "codegen_npuir_dev_a5.h"
#else
#include "codegen_npuir_api.h"
#include "codegen_npuir_dev.h"
#endif

namespace tvm {
namespace codegen {

// ===========================================================================
// Baseline (existing) builders -- UNCHANGED. Kept so the TLIR pipeline can be
// compared against them.
// ===========================================================================

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

// Expert mode: low-level APIs (CodeGenTileLangNPUIRAPI).
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

// Developer mode: higher-level APIs (CodeGenTileLangNPUIRDEV).
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

// ===========================================================================
// TLIR pipeline (new): TIR -> TLIR (upstream memref/linalg) -> HIVM
// ===========================================================================

namespace {

// Runs the TIR->TLIR codegen over every PrimFunc; returns the TLIR text.
std::string GenerateTLIR(const IRModule &mod, Array<String> *function_names) {
  CodeGenTIRToTLIR cg;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTIRToTLIR: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names->push_back(static_cast<std::string>(
        f->GetAttr<String>(tvm::attr::kGlobalSymbol).value()));
  }
  return cg.Finish();
}

std::string MakeTempFile(const char *tag) {
  std::string tmpl = std::string("/tmp/tilelang_") + tag + "_XXXXXX";
  int fd = mkstemp(&tmpl[0]);
  if (fd < 0) {
    throw std::runtime_error("BuildTileLangHIVM: mkstemp failed in /tmp");
  }
  close(fd);
  return tmpl;
}

std::string ReadFile(const std::string &path) {
  std::string text;
  std::array<char, 4096> buf;
  if (FILE *f = std::fopen(path.c_str(), "r")) {
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0)
      text.append(buf.data(), n);
    std::fclose(f);
  }
  return text;
}

// Runs bishengir-opt on the TLIR text and returns the resulting HIVM MLIR.
//
// Overridable without rebuilding:
//   TILELANG_BISHENGIR_OPT     path to the bishengir-opt binary
//   TILELANG_BISHENGIR_PASSES  space-separated pass flags
// Default: NO passes (bishengir-opt only parses/verifies/reprints the TLIR).
// The real TLIR -> HIVM -> LLVM lowering is done by `bishengir-compile`, which
// runs all its passes (including hfusion->hivm) internally, so no pass is
// invoked explicitly here. Set TILELANG_BISHENGIR_PASSES only for experiments.
std::string RunBishengirOptToHIVM(const std::string &tlir_text) {
  const char *env_opt = std::getenv("TILELANG_BISHENGIR_OPT");
  std::string opt_path =
      env_opt && *env_opt ? std::string(env_opt)
                          : "/workspace/AscendNPU-IR/build/bin/bishengir-opt";
  const char *env_passes = std::getenv("TILELANG_BISHENGIR_PASSES");
  std::string passes = env_passes ? std::string(env_passes) : "";

  std::string in_path = MakeTempFile("in");
  std::string err_path = MakeTempFile("err");

  if (FILE *f = std::fopen(in_path.c_str(), "w")) {
    std::fwrite(tlir_text.data(), 1, tlir_text.size(), f);
    std::fclose(f);
  } else {
    std::remove(in_path.c_str());
    std::remove(err_path.c_str());
    throw std::runtime_error(
        "BuildTileLangHIVM: failed to write temp input file");
  }

  std::string cmd = opt_path + " " + passes + " " + in_path + " 2>" + err_path;

  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    std::remove(in_path.c_str());
    std::remove(err_path.c_str());
    throw std::runtime_error(
        "BuildTileLangHIVM: failed to launch bishengir-opt at '" + opt_path +
        "'. Set TILELANG_BISHENGIR_OPT to override the path.");
  }
  std::string output;
  std::array<char, 4096> buffer;
  size_t n;
  while ((n = std::fread(buffer.data(), 1, buffer.size(), pipe)) > 0)
    output.append(buffer.data(), n);
  int status = pclose(pipe);

  std::string stderr_text = ReadFile(err_path);
  std::remove(in_path.c_str());
  std::remove(err_path.c_str());

  if (status != 0) {
    throw std::runtime_error(
        "BuildTileLangHIVM: bishengir-opt failed (exit status " +
        std::to_string(status) + ").\n--- command ---\n" + opt_path + " " +
        passes + "\n--- stderr ---\n" + stderr_text +
        "\n--- TLIR input that failed ---\n" + tlir_text);
  }
  return output;
}

} // namespace

// Stage 1 only: TIR -> TLIR text (used to inspect/test the importer).
runtime::Module BuildTileLangTLIR(IRModule mod, Target target) {
  Array<String> function_names;
  std::string tlirCode = GenerateTLIR(mod, &function_names);
  return CSourceModuleCreate(tlirCode, "c", function_names);
}

// Stages 1+2: TIR -> TLIR -> (bishengir-opt) -> HIVM text.
runtime::Module BuildTileLangHIVM(IRModule mod, Target target) {
  Array<String> function_names;
  std::string tlirCode = GenerateTLIR(mod, &function_names);
  std::string hivmCode = RunBishengirOptToHIVM(tlirCode);
  return CSourceModuleCreate(hivmCode, "c", function_names);
}

// ===========================================================================
// Registration: makes these visible to TVM as target.build.<name>
// ===========================================================================

TVM_REGISTER_TARGET_KIND("npuir", kDLExtDev);

TVM_REGISTER_GLOBAL("target.build.tilelang_tlir")
    .set_body_typed(BuildTileLangTLIR);

TVM_REGISTER_GLOBAL("target.build.tilelang_hivm")
    .set_body_typed(BuildTileLangHIVM);

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
