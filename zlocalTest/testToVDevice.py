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

@tvm.script.ir_module
class TestVMToDevice:
    I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                    I.vdevice("fpga", 0),
                ]
            }
        )
    @R.function(pure=False)
    def foo(x: R.Tensor((3, 4), "float32")):
        R.func_attr({"global_symbol": "foo"})
        z = R.to_vdevice(x, "fpga")
        z = R.to_vdevice(x, "fpga")
        return z

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