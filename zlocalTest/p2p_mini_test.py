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
from get_buffer_size import get_buffer_sizes

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
slice_number = 46
n=16 

@I.ir_module
class TestP2P:
    I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                    I.vdevice("fpga", 0),
                ]
            }
        )
    # data, params, output_buffers, other
    @T.prim_func
    def fpga_process(a_handle: T.handle, b_handle: T.handle) -> None:
        T.func_attr({"global_symbol": "add_one", "tir.noalias": True})

        n = T.int32()
        A = T.match_buffer(a_handle, (n,), dtype="int32")
        B = T.match_buffer(b_handle, (n,), dtype="int32")

        for i in T.serial(0, n):
            B[i] = A[i] + 1
    
    
    
    @R.function(pure=False)
    def foo(inp: R.Tensor((n,), "int32")) -> R.Tensor:
        R.func_attr({"global_symbol": "foo"})
        data = R.to_vdevice(inp, "fpga")

        out= R.call_tir(TestP2P.fpga_process, (data), out_sinfo=
                R.Tensor((n,), "int32")
        )
        
        res = R.to_vdevice(out, "fpga")

        return res



mod = TestP2P
# mod = get_buffer_sizes(mod, global_symbol="foo")
target = [tvm.cpu(0), tvm.fpga(0)]
vm = compile(mod, target)
inp = tvm.nd.array(np.zeros((n,), dtype="int32"))
res = vm["foo"](inp)
tvm.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)
# check the resulting tensor is on cpu:0
assert str(res.device) == "fpga(0)"
assert res.device.device_type == 17
assert res.device.device_id == 0