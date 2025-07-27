import tvm
from tvm.script import tir as T, ir as I


def run_passes(mod):
    mod = tvm.driver.build_module.lower(mod)
    cuda_target = tvm.target.Target("cuda", host="llvm")
    assert cuda_target.thread_warp_size == 32
    mod = tvm.tir.transform.BindTarget(cuda_target)(mod)
    mod = tvm.tir.transform.ThreadSync("shared")(mod)
    mod = tvm.tir.transform.ThreadSync("shared.dyn")(mod)
    mod = tvm.tir.transform.ThreadSync("warp")(mod)
    mod = tvm.tir.transform.InferFragment()(mod)
    mod = tvm.tir.transform.LowerThreadAllreduce()(mod)
    mod = tvm.tir.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tir.transform.SplitHostDevice()(mod)
    mod = tvm.tir.transform.LowerWarpMemory()(mod)
    mod = tvm.tir.transform.ExtractBufferShape()(mod)
    # mod = tvm.tir.transform.ReplaceMatchBuffer()(mod)
    return mod

@I.ir_module
class Module:
    @T.prim_func
    def parent_kernel(value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle):
        T.func_attr({"global_symbol": "parent_kernel", "target": T.target("cuda")})
        row_num_split, col_num_split = T.int32(), T.int32()
        buffer_1_val = T.match_buffer(value_1, (row_num_split, col_num_split))
        col_1_idx = T.match_buffer(c_idx_1, (row_num_split, col_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        col_num_real = T.int32()
        vec_val = T.match_buffer(vector_in, (col_num_real,))
        row_num_real = T.int32()
        res_val = T.match_buffer(res_t, (row_num_real,))
        for i in T.thread_binding(1, thread="blockIdx.x"):
            for j in T.thread_binding(1, thread="threadIdx.x"):
                for k in range(3):
                        Module.long_lines(buffer_1_val.data, col_1_idx.data, buffer_1_idx.data, vec_val.data, res_val.data,
                                                col_num_split, row_num_real, row_num_split, col_num_real)
                        
    @T.prim_func
    def long_lines(value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle, col_num_real: T.int32, col_num_split: T.int32, row_num_split: T.int32, row_num_real: T.int32):
        T.func_attr({"target": T.target({"arch": "sm_86", "host": {"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-pc-linux-gnu", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        buffer_1_val = T.match_buffer(value_1, (row_num_split, col_num_split))
        col_1_idx = T.match_buffer(c_idx_1, (row_num_split, col_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        vec_val = T.match_buffer(vector_in, (col_num_real,))
        res_val = T.match_buffer(res_t, (row_num_real,))
        for ii in T.thread_binding(1, thread="blockIdx.x"):
            for jj in T.thread_binding(1, thread="threadIdx.x"):
                if 0 < row_num_split:
                    Module.long_lines_kernel_3(res_val.data, row_num_real)
                else:
                    Module.long_lines_kernel_3_2(res_val.data, row_num_real)

    @T.prim_func(private=True)
    def long_lines_kernel_3(res_val: T.handle("float32", "global"), row_num_real: T.int32):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_global_func": T.bool(True), "tir.noalias": T.bool(True)})
        res_val_1 = T.decl_buffer((row_num_real,), data=res_val)
        blockIdx_x = T.launch_thread("blockIdx.x", (row_num_real + 255) // 256)
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        if blockIdx_x * 256 + threadIdx_x < row_num_real:
            res_val_1[blockIdx_x * 256 + threadIdx_x] = T.float32(0.0)

    @T.prim_func(private=True)
    def long_lines_kernel_3_2(res_val: T.handle("float32", "global"), row_num_real: T.int32):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_global_func": T.bool(True), "tir.noalias": T.bool(True)})
        res_val_1 = T.decl_buffer((row_num_real,), data=res_val)
        blockIdx_x = T.launch_thread("blockIdx.x", (row_num_real + 255) // 256)
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        if blockIdx_x * 256 + threadIdx_x < row_num_real:
            res_val_1[blockIdx_x * 256 + threadIdx_x] = T.float32(0.0)
mod = Module
mod = mod.with_attr("entry_func", "parent_kernel")
mod = run_passes(mod)

lowered_mod = tvm.build(mod, target="cuda")
row_num_split = 4
col_num_split = 4
col_num_real = 4
row_num_real = 4

import numpy as np
from tvm import nd
dev = tvm.cuda(0)
value = nd.array(np.random.rand(row_num_split, col_num_split).astype("float32"), dev)
c_idx = nd.array(np.random.randint(0, 10, size=(row_num_split, col_num_split)).astype("int32"), dev)
idx = nd.array(np.arange(row_num_split).astype("int32"), dev)
vec = nd.array(np.random.rand(col_num_real).astype("float32"), dev)
res = nd.array(np.zeros(row_num_real, dtype="float32"), dev)

# 调用 kernel
f = lowered_mod["parent_kernel"]
f(value, c_idx, idx, vec, res)

# 查看输出
print("res:", res.numpy())
