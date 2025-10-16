#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include "../../fpga/fpga_utils.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

using tvm::runtime::NDArray;

TVM_REGISTER_GLOBAL("vm.builtin.p2p.descriptor_transfer")
    .set_body_typed([](NDArray head, NDArray tail, NDArray ptr_array, size_t gpu_bank_size) {
      // ShapeTuple shape = slice_num.Shape();
      // ICHECK_EQ(shape.size(), 1) << "NDArray slice_num only has 1 dimension";
      // ICHECK_EQ(shape[0], 1) << "Shape of NDArray slice_num should be (1,)";
      // ICHECK_EQ(slice_num->dtype.code, 0) << "NDArray slice_num must be
      // int32"; ICHECK_EQ(slice_num->dtype.bits, 32) << "NDArray slice_num must
      // be int32";
      void* head_ptr = head.operator->()->data;
      void* tail_ptr = tail.operator->()->data;
      ShapeTuple shape = ptr_array.Shape();
      int bank_num = shape[0];
      uint64_t* data_ptr = static_cast<uint64_t*>(ptr_array->data);
      void* void_ptr_array[bank_num];
      for (size_t i = 0; i < bank_num; ++i) {
        void_ptr_array[i] = reinterpret_cast<void*>(data_ptr[i]);
      }

      transfer_descriptor(head_ptr, tail_ptr, void_ptr_array, bank_num, gpu_bank_size);
    });

TVM_REGISTER_GLOBAL("vm.builtin.p2p.unmap_gpu_memory")
    .set_body_typed([](NDArray head, NDArray tail, NDArray bank_array) { memory_unmap(); });

static std::unordered_map<std::string, int> dtype_map = {
    {"uint8", 0}, {"uint16", 1}, {"uint32", 2},  {"uint64", 3},  {"int8", 4},     {"int16", 5},
    {"int32", 6}, {"int64", 7},  {"float16", 8}, {"float32", 9}, {"float64", 10}, {"handle", 11},
};

TVM_REGISTER_GLOBAL("vm.builtin.fpga.launch_kernel").set_body([](TVMArgs args, TVMRetValue* rv) {
  int n = args.num_args;
  auto buffer_sizes = std::make_unique<int[]>(n);
  auto buffer_kinds = std::make_unique<int[]>(n);
  auto ptrs = std::make_unique<void*[]>(n);
  std::vector<int> int_values;
  std::vector<float> float_values;
  int_values.reserve(n);
  float_values.reserve(n);

  for (int i = 0; i < n; i++) {
    TVMArgValue arg = args[i];
    if (arg.IsObjectRef<NDArray>()) {
      buffer_kinds[i] = 11;
      const TVMValue& value = arg.value();
      ptrs[i] = value.v_handle;
    } else if (arg.TryAsInt()) {
      int v = arg.value().v_int64;
      int_values.push_back(v);
      buffer_kinds[i] = 6;
      ptrs[i] = &int_values.back();
    } else if (arg.TryAsFloat()) {
      float v = static_cast<float>(arg.value().v_float64);
      float_values.push_back(v);
      buffer_kinds[i] = 9;
      ptrs[i] = &float_values.back();
    }
  }
  fpgaModuleLaunchKernel(buffer_sizes.get(), buffer_kinds.get(), ptrs.get(), n);
  fpga_sync();
});

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm