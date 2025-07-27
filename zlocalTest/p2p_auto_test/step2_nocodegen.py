# test c codegen
import tvm
from tvm import tir
from tvm.script import tir as T
from tvm.relay.backend import Runtime

slice_length = 8
target = tvm.target.Target("c")
host = tvm.target.Target("c")
runtime = Runtime("crt", {"system-lib": True})

@tvm.script.ir_module
class MyModule:
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
            "target": target
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


def run_passes(mod):
    mod = tvm.driver.build_module.lower(mod)
    mod = tvm.tir.transform.BindTarget(host)(mod)
    mod = tvm.tir.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tir.transform.SplitHostDevice()(mod)
    return mod

mod = MyModule
mod = run_passes(mod)

ex = tvm.build(mod, target=target, runtime=runtime)

imported_module = ex.imported_modules[0]
with open("zlocalTest/p2p_auto_test/codegen.c", "w") as f:
    f.write(imported_module.get_source("c"))