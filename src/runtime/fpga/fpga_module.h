#ifndef TVM_RUNTIME_FPGA_FPGA_MODULE_H_
#define TVM_RUNTIME_FPGA_FPGA_MODULE_H_

#include <tvm/runtime/module.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../meta_data.h"

namespace tvm {
namespace runtime {

Module FPGAModuleCreate(std::string addr, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_FPGA_FPGA_MODULE_H_