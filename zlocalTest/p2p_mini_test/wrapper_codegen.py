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
    def slice_add_one_fpga(
        a_handle: T.handle,
        x: T.int32, # for fpga
        y: T.int32, # for gpu
        ptr_array: T.handle,
        slicelen: T.int32,
    ) -> None:
        T.func_attr({
            "global_symbol": "slice_add_one_fpga",
            "tir.noalias": True,
            "target": target
        })
        n = T.int32()
        A = T.match_buffer(a_handle, (n,), dtype="int32")
        ptr_array_buffer = T.match_buffer(ptr_array, (8,), dtype="uint64")
        bank = T.allocate([8*65536], "int8", "global")
        bank_buffer = T.Buffer((8, 65536,), dtype="int8", data=bank, scope="global") # 8*64KB DDR bank
        
        # (A, bank_buffer, ptr_array, n, slicelen, x, y)
        T.attr(T.target("fpga"), "target", 0)
        with T.block("launch"):
            T.reads(A[0:n], ptr_array_buffer[0:8])
            T.writes(bank_buffer[0:8, 0:65536])
            T.call_extern(
                "int32", "launch_slice_kernel",
                A.data,
                bank_buffer.data,
                ptr_array_buffer.data,
                n, slicelen, 
                x, y
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
with open("zlocalTest/p2p_mini_test/codegen.c", "w") as f:
    f.write(imported_module.get_source("c"))