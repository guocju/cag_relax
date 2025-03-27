#include <llvm/Support/FileSystem.h>
#include <llvm/Target/TargetMachine.h>
#include <tvm/ir/module.h>
#include <tvm/relay/runtime.h>
#include <tvm/runtime/container/string.h>
#include <tvm/support/with.h>
#include <tvm/target/target.h>

#include "../../runtime/fpga/fpga_module.h"
#include "../build_common.h"
#include "codegen_llvm.h"
#include "llvm_instance.h"
namespace tvm {
namespace codegen {

namespace {
#if TVM_LLVM_VERSION <= 70
constexpr auto llvm_open_output_flag = llvm::sys::fs::F_None;
#else
constexpr auto llvm_open_output_flag = llvm::sys::fs::OF_None;
#endif
}  // namespace

runtime::Module BuildFPGA(IRModule mod, Target target) {
  std::string file_addr = "./zlocalTest/fpga_compile_result/result.ll";
  LLVMInstance llvm_instance;
  With<LLVMTarget> llvm_target(llvm_instance, target);
  llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
  std::unique_ptr<CodeGenLLVM> cg = CodeGenLLVM::Create(llvm_target.get());

  std::string entry_func;
  relay::Runtime runtime =
      mod->GetAttr<relay::Runtime>(tvm::attr::kRuntime).value_or(relay::Runtime::Create("cpp"));

  Optional<String> system_lib_prefix = mod->GetAttr<String>(tvm::attr::kSystemLibPrefix);
  if (!system_lib_prefix && runtime->GetAttr<Bool>("system-lib").value_or(Bool(false))) {
    system_lib_prefix = "";
  }

  bool target_c_runtime = runtime->name == "crt";

  for (auto kv : mod->functions) {
    if (!kv.second->IsInstance<PrimFuncNode>()) {
      // (@jroesch): we relax constraints here, Relay functions will just be ignored.
      DLOG(INFO) << "Can only lower IR Module with PrimFuncs, but got " << kv.second->GetTypeKey();
      continue;
    }
    auto f = Downcast<PrimFunc>(kv.second);
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    bool is_entry_func = f->HasNonzeroAttr(tir::attr::kIsEntryFunc);

    ICHECK(global_symbol || !is_entry_func) << "The entry func must be exposed externally.";

    if (is_entry_func) {
      entry_func = global_symbol.value();
    }
  }
  // TODO(@jroesch): follow up on this condition.
  // ICHECK(funcs.size() > 0);
  // TODO(tqchen): remove the entry function behavior as it does not
  // makes sense when we start to use multiple modules.
  cg->Init("TVMMod", llvm_target.get(), system_lib_prefix, system_lib_prefix.defined(),
           target_c_runtime);
  cg->SetFastMathFlags(llvm_target->GetFastMathFlags());

  cg->AddFunctionsOrdered(mod->functions.begin(), mod->functions.end());
  if (entry_func.length() != 0) {
    cg->AddMainFunction(entry_func);
  }

  std::unique_ptr<llvm::Module> module_owning_ptr = cg->Finish();
  llvm::Module* llvm_module = module_owning_ptr.get();
  llvm_target->SetTargetMetadata(llvm_module);
  llvm_module->addModuleFlag(llvm::Module::Override, "Debug Info Version",
                             llvm::DEBUG_METADATA_VERSION);

  if (system_lib_prefix) {
    std::string str_val = system_lib_prefix.value();
    llvm_module->addModuleFlag(llvm::Module::Warning, "tvm_system_lib_prefix",
                               llvm::MDString::get(*(llvm_target->GetContext()), str_val));
  }

  llvm_module->addModuleFlag(llvm::Module::Override, "Dwarf Version",
                             tm->getTargetTriple().isOSDarwin() ? 2 : 4);
  std::error_code ecode;
  llvm::raw_fd_ostream dest(file_addr, ecode, llvm_open_output_flag);
  ICHECK_EQ(ecode.value(), 0) << "Cannot open file: " << file_addr << " " << ecode.message();
  llvm_module->print(dest, nullptr);  // save llvm IR
  dest.close();
  return FPGAModuleCreate(file_addr, "ll", ExtractFuncInfo(mod));
}

TVM_REGISTER_GLOBAL("target.build.fpga").set_body_typed(BuildFPGA);

}  // namespace codegen
}  // namespace tvm