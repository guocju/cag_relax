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
            bank_buffer = T.Buffer((8, 512,), dtype="int32", data=bank, scope="global") # 8*2KB
            n_slices = (n + slice_length) // slice_length  # 向上取整
            for i in T.serial(0, n_slices):
                # T.call_extern("is_buffer_ready")
                base_addr = ptr_array_buffer[i%8]

                one_bank = T.match_buffer(bank_buffer[i % 8, :], (512,), dtype="int32")
                R1 = T.match_buffer(one_bank[0:2], (2,), dtype="int32")
                R2 = T.match_buffer(one_bank[2:3], (1,), dtype="int32")
                R3 = T.match_buffer(one_bank[3:5], (2,), dtype="int32")
                R4 = T.match_buffer(one_bank[5:6], (1,), dtype="int32")
                R5 = T.match_buffer(one_bank[6:], (506,), dtype="int32")

                start = i * 4
                remain = T.min(slice_length, n - start)
                a_offset = 2+1+2+1
                a_handle_gpu = base_addr + a_offset

                R1[0] = tir.Cast("int32", a_handle_gpu & T.uint64(0xFFFFFFFF))
                R1[1] = tir.Cast("int32", a_handle_gpu >> T.uint64(32))

                R2[0] = x
                b_offset = a_offset+remain*4
                b_handle_gpu = base_addr + b_offset
                R3[0] = tir.Cast("int32", b_handle_gpu & T.uint64(0xFFFFFFFF))
                R3[1] = tir.Cast("int32", b_handle_gpu >> T.uint64(32))
                R4[0] = remain
                A_slice = T.match_buffer(A[start : start + remain], (remain,), dtype="int32")
                B_slice = T.match_buffer(B[start : start + remain], (remain,), dtype="int32")
                with T.block("launch"):
                    slice_start = i * 4
                    slice_remain = T.min(slice_length, n - slice_start)
                    T.reads(A[slice_start : slice_start + slice_remain], B[slice_start : slice_start + slice_remain])
                    T.writes(R5[0 : slice_remain])
                    T.call_extern(
                        "int32", "launch_kernel",
                        T.address_of(A_slice[0]),
                        T.address_of(B_slice[0]),
                        x, y,
                        T.address_of(R5[0]),
                        slice_remain
                    )
                with T.block("enqueue"):
                    slice_start = i * 4
                    slice_remain = T.min(slice_length, n - slice_start)
                    total_data_size = (2 + 1 + 2 + 1 + slice_remain) * 4
                    T.call_extern("int32", "p2p_enqueue", T.address_of(one_bank[0]), total_data_size)


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