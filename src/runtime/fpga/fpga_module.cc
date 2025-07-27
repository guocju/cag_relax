/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "fpga_module.h"

#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "fpga_common.h"

namespace tvm {
namespace runtime {

class FPGAModuleNode final : public runtime::ModuleNode {
 public:
  ~FPGAModuleNode() = default;
  explicit FPGAModuleNode(std::string addr, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap)
      : addr_(addr), fmt_(fmt), fmap_(fmap) {}

  const char* type_key() const final { return "fpga"; }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;

  /*! \brief Get the property of the runtime module .*/
  // TODO(tvm-team): Make it serializable
  int GetPropertyMask() const override {
    return runtime::ModulePropertyMask::kRunnable | runtime::ModulePropertyMask::kDSOExportable;
  }

  // void SaveToFile(const String& file_name, const String& format) final;
  // void SaveToBinary(dmlc::Stream* stream) final;
  // String GetSource(const String& format) final;
 private:
  std::mutex mutex_;
  std::string addr_;
  // The format
  std::string fmt_;
  //   std::array<hipModule_t, kMaxNumGPUs> module_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
};

class FPGAWrappedFunc {
 public:
  void Init(FPGAModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_void_args, const std::vector<DLDataType> arg_types,
            const std::vector<std::vector<int>>& buffer_sizes,
            const std::vector<int> buffer_types) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    buffer_sizes_ = buffer_sizes;
    arg_types_ = arg_types;
    buffer_kinds_ = buffer_types;
    args_num_ = num_void_args;
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    std::vector<int> arg_sizes;

    for (auto& row : buffer_sizes_) {
      int arg_size = 1;
      for (auto element : row) {
        if (element <= 0) {
          int idx = -element;
          DLDataType t = arg_types_[idx];
          int bits = t.bits;
          void* data = void_args[idx];
          switch (bits) {
            case 8:
              element = *reinterpret_cast<int8_t*>(data);
              break;
            case 16:
              element = *reinterpret_cast<int16_t*>(data);
              break;
            case 32:
              element = *reinterpret_cast<int32_t*>(data);
              break;
            case 64:
              element = *reinterpret_cast<int64_t*>(data);
              break;
            default:
              throw std::runtime_error("Unsupported bits width");
          }
        }
        arg_size *= element;
      }
      arg_sizes.push_back(arg_size);
    }

    FPGA_CALL(fpgaModuleLaunchKernel(arg_sizes.data(), buffer_kinds_.data(), void_args, args_num_));
  }

 private:
  // internal module
  FPGAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
  std::vector<DLDataType> arg_types_;
  std::vector<std::vector<int>> buffer_sizes_;
  mutable std::vector<int> buffer_kinds_;
  int args_num_;
};

PackedFunc FPGAModuleNode::GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  FPGAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.arg_types, info.buffer_sizes,
         info.buffer_types);
  return PackFuncVoidAddr(f, info.arg_types);
}

Module FPGAModuleCreate(std::string addr, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap) {
  auto n = make_object<FPGAModuleNode>(addr, fmt, fmap);
  return Module(n);
}

}  // namespace runtime
}  // namespace tvm
