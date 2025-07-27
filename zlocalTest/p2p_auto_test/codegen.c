// tvm target: c -keys=cpu 
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t slice_add_one_fpga(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL void slice_add_one_fpga_kernel_1(int32_t* A, int8_t* bank_buffer, uint64_t* ptr_array_buffer, int32_t n, int32_t slicelen, int32_t x, int32_t y);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t launch_slice_kernel(int32_t*, int8_t*, uint64_t*, int32_t, int32_t, int32_t, int32_t);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t slice_add_one_fpga(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  TVMValue stack[4];
  void* stack_tcode = stack;
  TVMValue stack_1[8];
  void* stack_value = stack_1;
  int32_t a_handle_code = arg_type_ids[0];
  int32_t x_code = arg_type_ids[1];
  int32_t y_code = arg_type_ids[2];
  int32_t ptr_array_code = arg_type_ids[3];
  int32_t slicelen_code = arg_type_ids[4];
  void* a_handle = (((TVMValue*)args)[0].v_handle);
  int32_t x = ((int32_t)(((TVMValue*)args)[1].v_int64));
  int32_t y = ((int32_t)(((TVMValue*)args)[2].v_int64));
  void* ptr_array = (((TVMValue*)args)[3].v_handle);
  int32_t slicelen = ((int32_t)(((TVMValue*)args)[4].v_int64));
  void* slice_add_one_fpga_a_handle_shape = (((DLTensor*)a_handle)[0].shape);
  int32_t n = ((int32_t)((int64_t*)slice_add_one_fpga_a_handle_shape)[0]);
  void* slice_add_one_fpga_a_handle_strides = (((DLTensor*)a_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)a_handle)[0].device.device_id);
  void* A = (((DLTensor*)a_handle)[0].data);
  void* slice_add_one_fpga_ptr_array_shape = (((DLTensor*)ptr_array)[0].shape);
  void* slice_add_one_fpga_ptr_array_strides = (((DLTensor*)ptr_array)[0].strides);
  void* ptr_array_buffer = (((DLTensor*)ptr_array)[0].data);
  if (!(slice_add_one_fpga_a_handle_strides == NULL)) {
  }
  if (!(slice_add_one_fpga_ptr_array_strides == NULL)) {
  }
  void* bank_buffer = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)524288, 0, 8);
  if (bank_buffer == NULL) {
    return -1;
  }
  (((TVMValue*)stack_value)[0].v_handle) = A;
  if (A == NULL) {
    ((int32_t*)stack_tcode)[0] = 4;
  } else {
    ((int32_t*)stack_tcode)[0] = 3;
  }
  (((TVMValue*)stack_value)[1].v_handle) = bank_buffer;
  if (bank_buffer == NULL) {
    ((int32_t*)stack_tcode)[1] = 4;
  } else {
    ((int32_t*)stack_tcode)[1] = 3;
  }
  (((TVMValue*)stack_value)[2].v_handle) = ptr_array_buffer;
  if (ptr_array_buffer == NULL) {
    ((int32_t*)stack_tcode)[2] = 4;
  } else {
    ((int32_t*)stack_tcode)[2] = 3;
  }
  (((TVMValue*)stack_value)[3].v_int64) = ((int64_t)n);
  ((int32_t*)stack_tcode)[3] = 0;
  (((TVMValue*)stack_value)[4].v_int64) = ((int64_t)slicelen);
  ((int32_t*)stack_tcode)[4] = 0;
  (((TVMValue*)stack_value)[5].v_int64) = ((int64_t)x);
  ((int32_t*)stack_tcode)[5] = 0;
  (((TVMValue*)stack_value)[6].v_int64) = ((int64_t)y);
  ((int32_t*)stack_tcode)[6] = 0;
  TVMValue ret_val;
  int ret_type_code;
  if (slice_add_one_fpga_kernel_1( (TVMValue*) stack_value , (int*) stack_tcode, 6, &ret_val, &ret_type_code, NULL) != 0){
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, bank_buffer) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL void slice_add_one_fpga_kernel_1(int32_t* A, int8_t* bank_buffer, uint64_t* ptr_array_buffer, int32_t n, int32_t slicelen, int32_t x, int32_t y) {
  launch_slice_kernel(A, bank_buffer, ptr_array_buffer, n, slicelen, x, y);
}

