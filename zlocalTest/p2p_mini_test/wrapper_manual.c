#include <stdint.h>

void slice_add_one_fpga_kernel_1(uint64_t inBuf, uint64_t bankBuf,
                                 uint64_t ptrBuf, int32_t total_len,
                                 int32_t Slicelen, int32_t PlusNum,
                                 int32_t PlusNumGPU);

void wrapper_slice_add_one_fpga_kernel_1() {
  uint8_t* param_buffer = (uint8_t*)0x30;

  // 解析
  uint64_t inBuf_addr = *(uint64_t*)(param_buffer + 0);    // handle
  uint64_t bankBuf_addr = *(uint64_t*)(param_buffer + 8);  // handle
  uint64_t ptrBuf_addr = *(uint64_t*)(param_buffer + 16);  // handle
  int32_t total_len = *(int32_t*)(param_buffer + 24);      // int32
  int32_t Slicelen = *(int32_t*)(param_buffer + 28);       // int32
  int32_t PlusNum = *(int32_t*)(param_buffer + 32);        // int32
  int32_t PlusNumGPU = *(int32_t*)(param_buffer + 36);     // int32

  // 调用
  slice_add_one_fpga_kernel_1(inBuf_addr, bankBuf_addr, ptrBuf_addr, total_len,
                              Slicelen, PlusNum, PlusNumGPU);
}