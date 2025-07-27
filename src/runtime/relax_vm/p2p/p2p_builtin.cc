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

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm