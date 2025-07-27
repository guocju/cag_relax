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
        n_slices = (n + slice_length) // slice_length  # 向上取整
        with T.block():
            bank = T.allocate([8*512], "int32", "global")
            bank_buffer = T.Buffer((8, 512,), dtype="int32", data=bank, strides=[512, 1], scope="global") # 8*2KB
            for i in T.serial(0, n_slices):
                T.call_extern("int32", "is_buffer_ready")
                base_addr = ptr_array_buffer[i % 8]
                bank_idx = i % 8
                _ = bank_buffer[bank_idx, 511]
                
                start = i * slice_length
                remain = T.min(slice_length, n - start)
                a_offset = 2 + 1 + 2 + 1
                a_handle_gpu = base_addr + a_offset
                
                bank_buffer[bank_idx, 0] = T.Cast("int32", a_handle_gpu & T.uint64(0xFFFFFFFF))
                bank_buffer[bank_idx, 1] = T.Cast("int32", a_handle_gpu >> T.uint64(32))
                bank_buffer[bank_idx, 2] = x
                
                b_offset = a_offset + remain * 4
                b_handle_gpu = base_addr + b_offset
                bank_buffer[bank_idx, 3] = T.Cast("int32", b_handle_gpu & T.uint64(0xFFFFFFFF))
                bank_buffer[bank_idx, 4] = T.Cast("int32", b_handle_gpu >> T.uint64(32))
                bank_buffer[bank_idx, 5] = remain
                
                with T.block("launch"):
                    slice_start = i * slice_length
                    slice_remain = T.min(slice_length, n - i * slice_length)
                    T.reads(A[i * slice_length : i * slice_length + T.min(slice_length, n - i * slice_length)], 
                        B[i * slice_length : i * slice_length + T.min(slice_length, n - i * slice_length)])
                    T.writes(bank_buffer[bank_idx, 0:512])
                    T.call_extern(
                        "int32", "launch_kernel",
                        T.address_of(A[slice_start]),
                        T.address_of(B[slice_start]),
                        x, y,
                        T.address_of(bank_buffer[bank_idx, 6]),
                        slice_remain
                    )
                
                with T.block("enqueue"):
                    slice_remain = T.min(slice_length, n - i * slice_length)
                    T.reads(bank_buffer[bank_idx, 0 : 6 + T.min(slice_length, n - i * slice_length)])
                    T.writes([])
                    total_data_size = (6 + slice_remain) * 4
                    T.call_extern("int32", "p2p_enqueue", T.address_of(bank_buffer[bank_idx, 0]), total_data_size)


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