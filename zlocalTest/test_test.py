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

n = 8

# --------------------
# Static Shape Version
# --------------------
@I.ir_module
class TestP2P_Static:
    @R.function(pure=False)
    def foo(x: R.Tensor((1, 3, 8, 8), dtype="float32"), w: R.Tensor((3, 3, 3, 3), dtype="float32")) -> R.Tensor((1, 3, 6, 6), dtype="float32"):
        R.func_attr({"global_symbol": "foo"})
        return R.nn.conv2d(x, w, padding=(0, 0), strides=(1, 1))

# ----------------------
# Dynamic Shape Version
# ----------------------
@I.ir_module
class TestP2P_Dynamic:
    @R.function(pure=False)
    def foo(x: R.Tensor((1, 3, "high", "width"), dtype="float32"), w: R.Tensor((3, 3, 3, 3), dtype="float32")) -> R.Tensor:
        R.func_attr({"global_symbol": "foo"})
        return R.nn.conv2d(x, w, padding=(0, 0), strides=(1, 1))

# ------------------------
# Compilation and Execution
# ------------------------
def compile(mod: IRModule, device: List[Device]) -> relax.VirtualMachine:
    mod = relax.transform.RealizeVDevice()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    ex = relax.build(mod)
    return relax.VirtualMachine(ex, device)

# Setup
target = [tvm.cpu(0)]
inp = tvm.nd.array(np.random.randn(1, 3, 8, 8).astype("float32"), target[0])
kernel = tvm.nd.array(np.random.randn(3, 3, 3, 3).astype("float32"), target[0])

# Run Static
print("\n==== Static Shape ====")
vm_static = compile(TestP2P_Static, target)
res_static = vm_static["foo"](inp, kernel)
print(res_static.shape)

# Run Dynamic
print("\n==== Dynamic Shape ====")
vm_dynamic = compile(TestP2P_Dynamic, target)
res_dynamic = vm_dynamic["foo"](inp, kernel)
print(res_dynamic.shape)
