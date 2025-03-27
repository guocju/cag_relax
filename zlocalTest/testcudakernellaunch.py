import os
import tvm
import numpy as np
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script import relax as R, tir as T, ir as I
from tvm.ir.global_info import VDevice
from typing import List
from tvm._ffi.runtime_ctypes import Device
from tvm.ir.module import IRModule

def compile(
    mod: IRModule,
    device: List[Device]
) -> relax.VirtualMachine:
    # compile the model
    mod = relax.transform.RealizeVDevice()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    # no need to feed target argument for mult-target compilation
    ex = relax.build(mod)
    return relax.VirtualMachine(ex, device)


llvm_target = tvm.target.Target("llvm")
fpga_target = tvm.target.Target("fpga")
cuda_target = tvm.target.Target("cuda")
slice_number = 46

@I.ir_module
class TestP2P:
    I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                    I.vdevice("cuda", 0),
                ]
            }
        )
    
    @T.prim_func
    def p2p_gpu_process(
        A: T.Buffer((1,), "int32"),B: T.Buffer((1, 256, 256), "float32"),
        C: T.Buffer((1, 256, 256), "float32"),D: T.Buffer((1, 256, 256), "float32"),
        E: T.Buffer((1, 256, 256), "float32"),F: T.Buffer((1, 256, 256), "float32"),
        G: T.Buffer((1, 256, 256), "float32"),
        out: T.Buffer((8, 256, 256), "float32"),
    ):
        T.func_attr({"global_symbol": "p2p_gpu_process", "target": cuda_target})

        bx = T.thread_binding(8, "blockIdx.x")
        by = T.thread_binding(256, "blockIdx.y")
        tx = T.thread_binding(256, "threadIdx.x")
        with T.block("compute"):
            vi = T.axis.spatial(8, bx)
            vj = T.axis.spatial(256, by)
            vk = T.axis.spatial(256, tx)
            out[vi, vj, vk] = (
                B[0, vj, vk] + C[0, vj, vk] + 
                D[0, vj, vk] + E[0, vj, vk] + 
                F[0, vj, vk] + G[0, vj, vk] + A[0]
            )
    
    
    @R.function(pure=False)
    def foo(A: R.Tensor((2,), "int8"), B: R.Tensor((1, 256, 256), "float32"),
            C: R.Tensor((1, 256, 256), "float32"),D: R.Tensor((1, 256, 256), "float32"),
            E: R.Tensor((1, 256, 256), "float32"),F: R.Tensor((1, 256, 256), "float32"),
            G: R.Tensor((1, 256, 256), "float32"),
            ) -> R.Tensor:
        R.func_attr({"global_symbol": "foo"})
        
        A1 = R.to_vdevice(A, "cuda")
        B1 = R.to_vdevice(B, "cuda")
        C1 = R.to_vdevice(C, "cuda")
        D1 = R.to_vdevice(D, "cuda")
        E1 = R.to_vdevice(E, "cuda")
        F1 = R.to_vdevice(F, "cuda")
        G1 = R.to_vdevice(G, "cuda")
        slice_num = T.int32(slice_number)
        slice_num = R.call_packed("vm.builtin.p2p_descriptor_transfer", slice_num, sinfo_args=None)
        res = R.call_tir(
            TestP2P.p2p_gpu_process, 
            (A1, B1, C1, D1, E1, F1, G1, ),
            out_sinfo=R.Tensor((8, 256, 256), "float32", vdevice="cuda") 
        )
        return res

mod = TestP2P
target = [tvm.cpu(0), tvm.cuda(0)]
vm = compile(mod, target)
        
A_np = tvm.nd.array(np.random.randint(0, 128, (1,), dtype=np.int32))
B_np = tvm.nd.array(np.random.rand(1, 256, 256).astype(np.float32))
C_np = tvm.nd.array(np.random.rand(1, 256, 256).astype(np.float32))
D_np = tvm.nd.array(np.random.rand(1, 256, 256).astype(np.float32))
E_np = tvm.nd.array(np.random.rand(1, 256, 256).astype(np.float32))
F_np = tvm.nd.array(np.random.rand(1, 256, 256).astype(np.float32))
G_np = tvm.nd.array(np.random.rand(1, 256, 256).astype(np.float32))

res = vm["foo"](A_np, B_np, C_np, D_np, E_np, F_np, G_np)

# 5. 查看输出结果
print(res.shape)