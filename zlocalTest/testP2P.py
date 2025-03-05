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
    device: List[Device] = [
        tvm.cpu(),
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


@tvm.script.ir_module
class TestVMToDevice:
    @T.prim_func
    def fpga_kernel(
        A: T.Buffer((3, 4), "float32"),
        B: T.Buffer((3, 4), "float32"),
        C: T.Buffer((3, 4), "float32"),
        D: T.Buffer((3, 4), "float32"),
        E: T.Buffer((3, 4), "float32"),
        F: T.Buffer((3, 4), "float32"),
        G: T.Buffer((3, 4), "float32")
    ):
        T.func_attr({"global_symbol": "fpga_kernel", "target": fpga_target})
        
        for i, j in T.grid(3, 4):
            with T.block("compute"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 1.0  # 复制 A
                C[vi, vj] = A[vi, vj] + 1.0  # 加 1
                D[vi, vj] = A[vi, vj] - 1.0  # 减 1
                E[vi, vj] = A[vi, vj] * 2.0  # 乘 2
                F[vi, vj] = A[vi, vj] / 2.0  # 除 2
                G[vi, vj] = -A[vi, vj]       # 取负数
    
    I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                    I.vdevice("fpga", 0),
                ]
            }
        )
    
    @R.function(pure=False)
    def foo(x: R.Tensor((3, 4), "float32"), 
            ) -> (R.Tensor((3, 4), "float32"),
                R.Tensor((3, 4), "float32"),
                R.Tensor((3, 4), "float32"),
                R.Tensor((3, 4), "float32"),
                R.Tensor((3, 4), "float32"),
                R.Tensor((3, 4), "float32")):
            R.func_attr({"global_symbol": "foo"})
            cls = TestVMToDevice

            # 将输入数据传输到 FPGA 设备
            x_fpga = R.to_vdevice(x, "fpga")

            # 调用 TIR 内核，返回 6 个张量
            y1_fpga, y2_fpga, y3_fpga, y4_fpga, y5_fpga, y6_fpga = R.call_tir(
                cls.fpga_kernel, 
                (x_fpga,), 
                out_sinfo=(
                    R.Tensor((3, 4), "float32"),
                    R.Tensor((3, 4), "float32"),
                    R.Tensor((3, 4), "float32"),
                    R.Tensor((3, 4), "float32"),
                    R.Tensor((3, 4), "float32"),
                    R.Tensor((3, 4), "float32")
                )
            )

            # 将所有输出张量转换回 CPU
            y1_cpu = R.to_vdevice(y1_fpga, "llvm")
            y2_cpu = R.to_vdevice(y2_fpga, "llvm")
            y3_cpu = R.to_vdevice(y3_fpga, "llvm")
            y4_cpu = R.to_vdevice(y4_fpga, "llvm")
            y5_cpu = R.to_vdevice(y5_fpga, "llvm")
            y6_cpu = R.to_vdevice(y6_fpga, "llvm")

            return y1_cpu, y2_cpu, y3_cpu, y4_cpu, y5_cpu, y6_cpu
    

mod = TestVMToDevice
target = [tvm.cpu(0), tvm.fpga(0)]
vm = compile(mod, target)
inp = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
res = vm["foo"](inp)
tvm.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)
# check the resulting tensor is on cpu:0
assert str(res.device) == "fpga(0)"
assert res.device.device_type == 17
assert res.device.device_id == 0