#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO.h>
#include <tvm/ir/module.h>
#include <tvm/relay/runtime.h>
#include <tvm/runtime/container/string.h>
#include <tvm/support/with.h>
#include <tvm/target/target.h>

#include "../../runtime/fpga/fpga_module.h"
#include "../build_common.h"
#include "codegen_cpu.h"
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

class CodeGenFPGA : public CodeGenCPU {
 public:
  CodeGenFPGA() = default;
  virtual ~CodeGenFPGA() = default;

 protected:
  void BufferAccessHelper(
      Buffer buffer, Array<PrimExpr> indices, Optional<PrimExpr> predicate, DataType value_dtype,
      std::function<llvm::Instruction*(TypedPointer buffer_ptr, int subelement_i,
                                       llvm::Value* predicate, int alignment, bool is_volatile)>
          make_instruction) override {
    DataType buffer_element_dtype = buffer->dtype;

    ICHECK_GE(indices.size(), 1)
        << "Buffer " << buffer->name << " is accessed with no indices.  "
        << "0-d scalar buffers are expected to be flattened to 1-d buffers prior to codegen.";

    // Only the last index is allowed to be multi-lane.  All earlier
    // indices must be scalar.  This only matters for subclasses of
    // CodeGenLLVM, because the default implementation of GetBufferPtr
    // requires 1-d indices.
    std::vector<llvm::Value*> earlier_index_values;
    for (size_t i = 0; i < indices.size() - 1; i++) {
      ICHECK_EQ(indices[i].dtype().lanes(), 1)
          << "Buffer " << buffer->name << " is accessed with a multi-lane index at position " << i
          << ".  Multi-lane indices are only supported as the last index.";
      earlier_index_values.push_back(MakeValue(indices[i]));
    }

    PrimExpr last_index = indices[indices.size() - 1];
    ICHECK_EQ(value_dtype.get_lanes_or_vscale_factor(),
              last_index.dtype().get_lanes_or_vscale_factor() * buffer_element_dtype.lanes());

    // Record index and elemtype in original form used for alias info
    PrimExpr last_index_origin = last_index;
    DataType buffer_element_dtype_origin = buffer_element_dtype;

    bool is_volatile = volatile_buf_.count(buffer->data.get());

    // If the buffer index is a contiguous ramp node, we only need to
    // access the first element, then cast to the value type.
    if (const RampNode* ramp_index = last_index.as<RampNode>()) {
      if (is_one(ramp_index->stride)) {
        last_index = ramp_index->base;
      }
    }

    // All TVM arrays are densely packed.  If the vectorized LLVM type
    // contains padding for alignment, we need to index based on the
    // size of the scalar type to avoid introducing that padding.
    if (last_index.dtype().lanes() == 1 && HasAlignmentPadding(buffer_element_dtype)) {
      last_index = buffer_element_dtype.lanes() * last_index;
      buffer_element_dtype = buffer_element_dtype.element_of();
    }

    int alignment;
    if (last_index.dtype().lanes() == 1) {
      // If we are accessing with a single index, then the vectorized
      // element being accessed may require more alignment than the
      // underlying data type.
      int native_bits;
      GetAlignment(value_dtype, buffer->data.get(), last_index, &alignment, &native_bits);
    } else {
      // Otherwise, alignment is based on the return value's scalar
      // type.
      ICHECK_GE(value_dtype.bits(), 8);
      alignment = value_dtype.bits() / 8;
    }

    llvm::Value* cached_vector_index = nullptr;
    for (int i = 0; i < last_index.dtype().lanes(); ++i) {
      llvm::Value* last_index_value;
      int subelement_i = i;
      if (const RampNode* ramp = last_index.as<RampNode>()) {
        PrimExpr offset = ramp->base + (ramp->stride * i);
        last_index_value = MakeValue(offset);
      } else if (last_index.dtype().is_vector()) {
        if (i == 0) {
          cached_vector_index = MakeValue(last_index);
        }
        last_index_value = builder_->CreateExtractElement(cached_vector_index, i);
      } else {
        last_index_value = MakeValue(last_index);
        subelement_i = -1;
      }

      std::vector<llvm::Value*> all_index_values = earlier_index_values;
      all_index_values.push_back(last_index_value);

      llvm::Value* predicate_value = nullptr;
      if (predicate.defined()) {
        predicate_value = MakeValue(predicate.value());
      }

      TypedPointer buffer_ptr =
          value_dtype.is_scalable_vector()
              ? CreateBufferPtr(MakeValue(buffer->data), buffer_element_dtype, all_index_values,
                                value_dtype.with_scalable_vscale_factor(
                                    value_dtype.vscale_factor() / last_index.dtype().lanes()))
              : CreateBufferPtr(
                    MakeValue(buffer->data), buffer_element_dtype, all_index_values,
                    value_dtype.with_lanes(value_dtype.lanes() / last_index.dtype().lanes()));
      auto instruction =
          make_instruction(buffer_ptr, subelement_i, predicate_value, alignment, is_volatile);
      // AddAliasInfo(instruction, buffer->data.get(), last_index_origin,
      // buffer_element_dtype_origin);
    }
  }
};

runtime::Module BuildFPGA(IRModule mod, Target target) {
  std::string file_addr = "./zlocalTest/fpga_compile_result/result.ll";
  LLVMInstance llvm_instance;
  With<LLVMTarget> llvm_target(llvm_instance, target);
  llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
  std::unique_ptr<CodeGenLLVM> cg = std::make_unique<CodeGenFPGA>();

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
      // (@jroesch): we relax constraints here, Relay functions will just be
      // ignored.
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
  cg->Init("TVMMod", llvm_target.get(), system_lib_prefix, system_lib_prefix.defined(),
           target_c_runtime);
  cg->SetFastMathFlags(llvm_target->GetFastMathFlags());

  cg->AddFunctionsOrdered(mod->functions.begin(), mod->functions.end());
  if (entry_func.length() != 0) {
    cg->AddMainFunction(entry_func);
  }

  std::unique_ptr<llvm::Module> module_owning_ptr = cg->Finish();
  llvm::Module* llvm_module = module_owning_ptr.get();
  // llvm_target->SetTargetMetadata(llvm_module);
  // llvm_module->addModuleFlag(llvm::Module::Override, "Debug Info Version",
  //                            llvm::DEBUG_METADATA_VERSION);

  // if (system_lib_prefix) {
  //   std::string str_val = system_lib_prefix.value();
  //   llvm_module->addModuleFlag(llvm::Module::Warning, "tvm_system_lib_prefix",
  //                              llvm::MDString::get(*(llvm_target->GetContext()), str_val));
  // }

  // llvm_module->addModuleFlag(llvm::Module::Override, "Dwarf Version",
  //                            tm->getTargetTriple().isOSDarwin() ? 2 : 4);
  llvm::legacy::PassManager pass;
  pass.add(llvm::createStripSymbolsPass());
  pass.add(llvm::createStripDebugDeclarePass());
  pass.add(llvm::createStripDeadDebugInfoPass());
  pass.add(llvm::createGlobalDCEPass());
  pass.run(*llvm_module);
  std::error_code ecode;
  llvm::raw_fd_ostream dest(file_addr, ecode, llvm_open_output_flag);
  ICHECK_EQ(ecode.value(), 0) << "Cannot open file: " << file_addr << " " << ecode.message();
  llvm_module->print(dest, nullptr, false, false);
  dest.close();
  return FPGAModuleCreate(file_addr, "ll", ExtractFuncInfo(mod));
}

TVM_REGISTER_GLOBAL("target.build.fpga").set_body_typed(BuildFPGA);
}  // namespace codegen
}  // namespace tvm