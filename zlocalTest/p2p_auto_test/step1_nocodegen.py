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
from OutputAnalyze import get_buffer_shapes

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
    return relax.VirtualMachine(ex, device)


llvm_target = tvm.target.Target("llvm")
fpga_target = tvm.target.Target("fpga")
gpu_target = tvm.target.Target("cuda")
slice_length = 8
# length of input vector
n=260
slice_number = (n + slice_length - 1) // slice_length
buffer_num = 2
const_x = 8
const_y = 3
gpu_bank_size = 65536

@I.ir_module
class TestP2P_2:
    I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                    I.vdevice("fpga"),
                    I.vdevice("cuda"),
                ]
            }
        )
    
    @T.prim_func
    def slice_add_two_fpga(
        a_handle: T.handle,
        b_handle: T.handle,
        x: T.int32,
        y: T.int32,
        ptr_array: T.handle,
        out_handle: T.handle 
    ) -> None:
        T.func_attr({
            "global_symbol": "slice_add_two_fpga",
            "tir.noalias": True,
            "target": fpga_target
        })
        n = T.int32()
        A = T.match_buffer(a_handle, (n,), dtype="int32")
        B = T.match_buffer(b_handle, (n,), dtype="int32")
        Out = T.match_buffer(out_handle, (n,), dtype="int32")
        ptr_array_buffer = T.match_buffer(ptr_array, (8,), dtype="uint64")
        
        T.attr(T.target("fpga"), "target", 0)
        with T.block():
            bank = T.allocate([8*512], "int32", "global")
            bank_buffer = T.Buffer((8, 512,), dtype="int32", data=bank, strides=[512, 1], scope="global") # 8*2KB
            with T.block("launch"):
                T.reads(A[0:n], B[0:n], ptr_array_buffer[0:8])
                T.writes(bank_buffer[0:8, 0:512])
                T.call_extern(
                    "int32", "launch_slice_kernel",
                    A.data,
                    B.data,
                    x, y,
                    bank_buffer.data,
                    ptr_array_buffer.data
                )

    @T.prim_func
    def add_two_fpga(
        a_handle: T.handle,
        b_handle: T.handle,
        x: T.int32,
        y: T.int32,
        out_handle: T.handle
    ) -> None:
        T.func_attr({
            "global_symbol": "add_two_fpga",
            "tir.noalias": True,
            "target": fpga_target,
            "slice": [a_handle, b_handle],
            "slice_num": slice_number
        })

        n = T.int32()
        A = T.match_buffer(a_handle, (n,), dtype="int32")
        B = T.match_buffer(b_handle, (n,), dtype="int32")
        Out = T.match_buffer(out_handle, (n,), dtype="int32")

        T.attr(T.target("fpga"), "target", 0)

        for i in T.serial(0, n):
            Out[i] = A[i] * B[i] + x + y
       
    @T.prim_func
    def add_one_gpu(a_handle: T.handle, x:T.int32, b_handle: T.handle) -> None:
        T.func_attr({"global_symbol": "add_one_gpu", "tir.noalias": True, "target":gpu_target})

        n = T.int32()
        A = T.match_buffer(a_handle, (n,), dtype="int32")
        B = T.match_buffer(b_handle, (n,), dtype="int32")
        
        for i in T.thread_binding(n, thread="threadIdx.x"):
            B[i] = A[i] + x
    
    @T.prim_func
    def init_ptr_array(ptr_handle: T.handle) -> None:
        T.func_attr({"global_symbol": "init_ptr_array", "tir.noalias": True, "target":gpu_target})
        ptr_num = T.int32()
        ptr_buffer = T.match_buffer(ptr_handle, (ptr_num,), dtype="uint64")
        
        for i in T.thread_binding(ptr_num, thread="threadIdx.x"):
            ptr_buffer[i] = i
    
    @R.function(pure=False)
    def foo(A: R.Tensor(("len",), dtype="int32"),
            B: R.Tensor(("len",), dtype="int32"),) -> R.Tensor(("len",), dtype="int32"):
        R.func_attr({"global_symbol": "foo"})
        len = T.int64()
        
        data_a = R.to_vdevice(A, "fpga")
        data_b = R.to_vdevice(B, "fpga")

        inp_x = R.prim_value(T.IntImm("int32", const_x))
        inp_y = R.prim_value(T.IntImm("int32", const_y))
        
        head = R.builtin.alloc_tensor(R.shape([16384]), dtype="int32", runtime_device_index=2)
        tail = R.builtin.alloc_tensor(R.shape([16384]), dtype="int32", runtime_device_index=2)
        ptr_array = R.builtin.alloc_tensor(R.shape([8]), dtype="uint64", runtime_device_index=2)
        ptr_array = R.call_tir_inplace(TestP2P_2.init_ptr_array, ptr_array, 0, out_sinfo=R.Tensor((8,), "uint64", vdevice="cuda"))
        ptr_array_1 = R.to_vdevice(ptr_array, "llvm")
        
        # _ = R.call_packed("vm.builtin.p2p.descriptor_transfer", head, tail, ptr_array_1, gpu_bank_size)
        out = R.call_tir(
            TestP2P_2.slice_add_two_fpga,               
            (data_a, data_b, inp_x, inp_y, ptr_array),
            out_sinfo=R.Tensor((len,), "int32", vdevice="fpga")
        )
        gpu_inp = R.to_vdevice(out, "cuda")
        
        res = R.call_tir(
            TestP2P_2.add_one_gpu, 
            (gpu_inp, inp_x),
            out_sinfo=R.Tensor((len,), "int32", vdevice="cuda")
        )
        return res


mod = TestP2P_2
# mod = get_buffer_shapes(mod, "foo")
target = [tvm.cpu(0), tvm.fpga(0), tvm.cuda(0)]
vm = compile(mod, target)
inp = tvm.nd.array(np.zeros((n,), dtype="int32"), target[0])
inp_2 = tvm.nd.array(np.zeros((n,), dtype="int32"), target[0])
res = vm["foo"](inp, inp_2)
print(res)