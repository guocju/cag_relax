import tvm
import numpy as np
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script import relax as R, tir as T, ir as I
from tvm.relax.testing.vm import check_saved_func


def codegen(mod, target, exec_mode="bytecode"):
    builder = relax.ExecBuilder()
    tir_mod = relax.vm_build._vmcodegen(builder, mod, exec_mode=exec_mode)
    return relax.vm_build._vmlink(builder, target, tir_mod)


@tvm.script.ir_module
class TestVMToDevice:
    @R.function(pure=False)
    def foo(x: R.Tensor((3, 4), "float32")):
        R.func_attr({"global_symbol": "foo"})
        # Copy x to the first cpu: device_type=1 and device_id=0.
        # More device info. please take a look at python/tvm/_ffi/runtime_ctypes.py
        z = R.call_packed(
            "vm.builtin.to_device", x, 17, 0, sinfo_args=(R.Tensor((3, 4), dtype="float32"))
        )
        return z

mod = TestVMToDevice
target = tvm.target.Target("llvm", host="llvm")
ex = codegen(mod, target)
inp = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
vm = relax.VirtualMachine(ex, tvm.fpga())
res = vm["foo"](inp)
tvm.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)
# check the resulting tensor is on cpu:0
assert str(res.device) == "fpga(0)"
assert res.device.device_type == 17
assert res.device.device_id == 0