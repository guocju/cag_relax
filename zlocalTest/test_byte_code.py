import os
import tvm
import numpy as np
import tvm.testing
from tvm import relax, tir
from tvm.script import relax as R, tir as T, ir as I
from tvm.ir.global_info import VDevice
from typing import List
from tvm._ffi.runtime_ctypes import Device
from tvm.ir.module import IRModule
from get_buffer_size import get_buffer_shapes

def compile(
    mod: IRModule,
    device: List[Device] = [
        tvm.cpu()
    ],
) -> relax.VirtualMachine:
    # compile the model
    mod = relax.transform.RealizeVDevice()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    # no need to feed target argument for mult-target compilation
    ex = relax.build(mod)
    return relax.VirtualMachine(ex, device, buffer_num = 2)


llvm_target = tvm.target.Target("llvm")
fpga_target = tvm.target.Target("fpga")
cuda_target = tvm.target.Target("cuda")
slice_number = 46
n=16 
const_x = 8
const_y = 3

@I.ir_module
class TestP2P:
    I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                    I.vdevice("cuda"),
                ]
            }
        )
    @T.prim_func
    def init_ptr_array(gv6: T.handle, gv7: T.handle, gv8: T.handle, gv10: T.handle, gv11: T.handle, gv12: T.handle, gv6_arr: T.handle, gv7_arr: T.handle, gv8_arr: T.handle, gv10_arr: T.handle, gv11_arr: T.handle, gv12_arr: T.handle):
        T.func_attr({"target": T.target("cuda")})
        p_gv6 = T.match_buffer(gv6, (8,), "uint64")
        p_gv7 = T.match_buffer(gv7, (8,), "uint64")
        p_gv8 = T.match_buffer(gv8, (8,), "uint64")
        p_gv10 = T.match_buffer(gv10, (8,), "uint64")
        p_gv11 = T.match_buffer(gv11, (8,), "uint64")
        p_gv12 = T.match_buffer(gv12, (8,), "uint64")
        gv6_arr_val = T.match_buffer(gv6_arr, (8, 100, 100), "float32")
        gv7_arr_val = T.match_buffer(gv7_arr, (8, 100, 100), "int32")
        gv8_arr_val = T.match_buffer(gv8_arr, (8, 100), "int32")
        gv10_arr_val = T.match_buffer(gv10_arr, (8, 100, 64), "float32")
        gv11_arr_val = T.match_buffer(gv11_arr, (8, 100, 64), "int32")
        gv12_arr_val = T.match_buffer(gv12_arr, (8, 100), "int32")
        with T.block("init"):
            T.reads()
            T.writes(p_gv6[0:8], p_gv7[0:8], p_gv8[0:8], p_gv10[0:8], p_gv11[0:8], p_gv12[0:8])
            for threadIdx_x in T.thread_binding(1, thread="threadIdx.x"):
                for i in range(8):
                    p_gv6[i] = T.reinterpret("uint64", T.address_of(gv6_arr_val[i, 0, 0]))
                    p_gv7[i] = T.reinterpret("uint64",T.address_of(gv7_arr_val[i, 0, 0]))
                    p_gv8[i] = T.reinterpret("uint64",T.address_of(gv8_arr_val[i, 0]))
                    p_gv10[i] = T.reinterpret("uint64",T.address_of(gv10_arr_val[i, 0, 0]))
                    p_gv11[i] = T.reinterpret("uint64",T.address_of(gv11_arr_val[i, 0, 0]))
                    p_gv12[i] = T.reinterpret("uint64",T.address_of(gv12_arr_val[i, 0]))
       
    
    @R.function(pure=False)
    def foo()-> R.Tensor:
        R.func_attr({"global_symbol": "foo"})
        # p_gv = R.builtin.alloc_tensor(R.shape([6, 8]), dtype="uint64", runtime_device_index=1)
        p_gv6 = R.builtin.alloc_tensor(R.shape([8]), dtype="uint64", runtime_device_index=1)
        p_gv7 = R.builtin.alloc_tensor(R.shape([8]), dtype="uint64", runtime_device_index=1)
        p_gv8 = R.builtin.alloc_tensor(R.shape([8]), dtype="uint64", runtime_device_index=1)
        p_gv10 = R.builtin.alloc_tensor(R.shape([8]), dtype="uint64", runtime_device_index=1)
        p_gv11 = R.builtin.alloc_tensor(R.shape([8]), dtype="uint64", runtime_device_index=1)
        p_gv12 = R.builtin.alloc_tensor(R.shape([8]), dtype="uint64", runtime_device_index=1)
        
        gv6_arr = R.builtin.alloc_tensor(R.shape([8, 100,100]), dtype="float32", runtime_device_index=1)
        gv7_arr = R.builtin.alloc_tensor(R.shape([8,100,100]), dtype="int32", runtime_device_index=1)
        gv8_arr = R.builtin.alloc_tensor(R.shape([8,100]), dtype="int32", runtime_device_index=1)
        gv10_arr = R.builtin.alloc_tensor(R.shape([8,100,64]), dtype="float32", runtime_device_index=1)
        gv11_arr = R.builtin.alloc_tensor(R.shape([8,100,64]), dtype="int32", runtime_device_index=1)
        gv12_arr = R.builtin.alloc_tensor(R.shape([8,100]), dtype="int32", runtime_device_index=1)
        gv_ptr_array = R.call_tir_inplace(TestP2P.init_ptr_array, (p_gv6, p_gv7, p_gv8, p_gv10, p_gv11, p_gv12, gv6_arr, gv7_arr, gv8_arr, 
                                                               gv10_arr, gv11_arr, gv12_arr), inplace_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
                                          out_sinfo=[R.Tensor((8,), dtype="uint64", vdevice="cuda:0"), R.Tensor((8,), dtype="uint64", vdevice="cuda:0"), R.Tensor((8,), dtype="uint64", vdevice="cuda:0"), R.Tensor((8,), dtype="uint64", vdevice="cuda:0"), R.Tensor((8,), dtype="uint64", vdevice="cuda:0"), R.Tensor((8,), dtype="uint64", vdevice="cuda:0"), R.Tensor((8, 100, 100), dtype="float32", vdevice="cuda:0"), R.Tensor((8, 100, 100), dtype="int32", vdevice="cuda:0"), R.Tensor((8, 100), dtype="int32", vdevice="cuda:0"), R.Tensor((8, 100, 64), dtype="float32", vdevice="cuda:0"), R.Tensor((8, 100, 64), dtype="int32", vdevice="cuda:0"), R.Tensor((8, 100), dtype="int32", vdevice="cuda:0")])
                
        return gv_ptr_array[0]



mod = TestP2P
target = [tvm.cpu(0), tvm.cuda(0)]
vm = compile(mod, target)
inp = tvm.nd.array(np.zeros((n,), dtype="int32"), target[0])
res = vm["foo"](inp)
print(res)