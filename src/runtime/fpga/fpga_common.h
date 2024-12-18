#ifndef TVM_RUNTIME_FPGA_FPGA_COMMON_H_
#define TVM_RUNTIME_FPGA_FPGA_COMMON_H_

#include <tvm/runtime/packed_func.h>

#include "fpga_utils.h"

namespace tvm {
namespace runtime {

#define FPGA_CALL(func) \
  { (func); }

void fpgaSetDevice(int id) { return; }

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_FPGA_FPGA_COMMON_H_