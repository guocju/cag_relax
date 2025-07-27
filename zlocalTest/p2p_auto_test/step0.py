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
gpu_target = tvm.target.Target("cuda")
slice_number = 46
n=16
const_x = 8
const_y = 3

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
    
    
    @R.function(pure=False)
    def foo(A: R.Tensor(("len",), dtype="int32"),
            B: R.Tensor(("len",), dtype="int32")) -> R.Tensor(("len",), dtype="int32"):
        R.func_attr({"global_symbol": "foo"})
        len = T.int64()
        
        data_a = R.to_vdevice(A, "fpga")
        data_b = R.to_vdevice(B, "fpga")

        inp_x = R.prim_value(T.IntImm("int32", const_x))
        inp_y = R.prim_value(T.IntImm("int32", const_y))

        out = R.call_tir(
            TestP2P_2.add_two_fpga,                 
            (data_a, data_b, inp_x, inp_y),
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
mod = get_buffer_shapes(mod, global_symbol="foo")
target = [tvm.cpu(0), tvm.fpga(0), tvm.cuda(0)]
vm = compile(mod, target)
inp = tvm.nd.array(np.zeros((n,), dtype="int32"), target[0])
inp_2 = tvm.nd.array(np.zeros((n,), dtype="int32"), target[0])
res = vm["foo"](inp, inp_2)
print(res)