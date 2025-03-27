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
from get_buffer_size import get_buffer_sizes

def compile(
    mod: IRModule,
    device: List[Device] = [
        tvm.cpu(), tvm.fpga(), tvm.cuda()
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
cuda_target = tvm.target.Target("cuda")
slice_number = 46

@I.ir_module
class TestP2P:
    I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                    I.vdevice("cuda", 0),
                    I.vdevice("fpga", 0),
                ]
            }
        )
    # data, params, output_buffers, other
    @T.prim_func
    def fpga_process(
        A: T.Buffer((8, 256, 256), "float32"), params: T.Buffer((16, 16), "float32"),
        B: T.Buffer((2,), "int8"), C: T.Buffer((2, 256, 256), "float32"),
        D: T.Buffer((2, 256, 256), "float32"),E: T.Buffer((2, 256, 256), "float32"),
        F: T.Buffer((2, 256, 256), "float32"),G: T.Buffer((2, 256, 256), "float32"),
        H: T.Buffer((2, 256, 256), "float32"),
    ):
        T.func_attr({"global_symbol": "fpga_process", "target": fpga_target, 
                     "slice_num" : slice_number, "param_num" : 2})
        T.attr(T.target("fpga"), "target", 0) #necessary for compiler
        for i in T.serial(1):
            for j, k in T.grid(256, 256):
                with T.block("compute_slice"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    C[0, vj, vk] = A[vi, vj, vk] * 3.0
                    D[0, vj, vk] = A[vi, vj, vk] * 4.0
                    E[0, vj, vk] = A[vi, vj, vk] * 5.0
                    F[0, vj, vk] = A[vi, vj, vk] * 6.0
                    G[0, vj, vk] = A[vi, vj, vk] * 7.0
                    H[0, vj, vk] = A[vi, vj, vk] * 8.0
    
    @T.prim_func
    def p2p_gpu_process(
        A: T.Buffer((2,), "int8"),B: T.Buffer((2, 256, 256), "float32"),
        C: T.Buffer((2, 256, 256), "float32"),D: T.Buffer((2, 256, 256), "float32"),
        E: T.Buffer((2, 256, 256), "float32"),F: T.Buffer((2, 256, 256), "float32"),
        G: T.Buffer((2, 256, 256), "float32"),
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
                F[0, vj, vk] + G[0, vj, vk]
            )
    
    
    @R.function(pure=False)
    def foo(inp: R.Tensor((8, 256, 256), "float32"), params: R.Tensor((16, 16), "float32"), slice_num: R.Tensor((1,), "int32")) -> R.Tensor:
        R.func_attr({"global_symbol": "foo"})
        data = R.to_vdevice(inp, "fpga")
        params = R.to_vdevice(params, "fpga")

        out= R.call_tir(TestP2P.fpga_process, (data, params), out_sinfo=[
                R.Tensor((2,), "int8"),R.Tensor((2, 256, 256), "float32"),
                R.Tensor((2, 256, 256), "float32"),R.Tensor((2, 256, 256), "float32"),
                R.Tensor((2, 256, 256), "float32"),R.Tensor((2, 256, 256), "float32"),
                R.Tensor((2, 256, 256), "float32")]
        ) # 双倍的buffer
        
        slice_index = R.to_vdevice(R.TupleGetItem(out, 0), "cuda")
        c1 = R.to_vdevice(R.TupleGetItem(out, 1), "cuda")
        c2 = R.to_vdevice(R.TupleGetItem(out, 2), "cuda")
        c3 = R.to_vdevice(R.TupleGetItem(out, 3), "cuda")
        c4 = R.to_vdevice(R.TupleGetItem(out, 4), "cuda")
        c5 = R.to_vdevice(R.TupleGetItem(out, 5), "cuda")
        c6 = R.to_vdevice(R.TupleGetItem(out, 6), "cuda")
        
        slice_num = R.call_packed("vm.builtin.p2p_descriptor_transfer", slice_num,  sinfo_args=(R.Tensor((1,), "int32")))
		# 一次kernel的launch计算一个slice, kernel名前缀为"p2p_"
        res = R.call_tir(
            TestP2P.p2p_gpu_process, 
            (slice_index, c1, c2, c3, c4, c5, c6, ),
            out_sinfo=R.Tensor((8, 256, 256), "float32") 
        )

        return res


mod = TestP2P
mod = get_buffer_sizes(mod)
target = [tvm.cpu(0), tvm.fpga(0), tvm.cuda(0)]
vm = compile(mod, target)
inp = tvm.nd.array(np.random.rand(8, 256, 256).astype(np.float32))
inp1 = tvm.nd.array(np.random.rand(16, 16).astype(np.float32))
inp2 = tvm.nd.array(np.random.rand(1,).astype(np.int32))
res = vm["foo"](inp, inp1, inp2)
tvm.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)
# check the resulting tensor is on cpu:0
assert str(res.device) == "fpga(0)"
assert res.device.device_type == 17
assert res.device.device_id == 0