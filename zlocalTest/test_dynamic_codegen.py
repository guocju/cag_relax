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
    # mod = tvm.tir.transform.LowerWarpMemory()(mod)
    mod = tvm.tir.transform.ExtractBufferShape()(mod)
    mod = tvm.tir.transform.ReplaceMatchBuffer()(mod)
    mod = tvm.tir.transform.AddThreadBinding()(mod)
    mod = tvm.tir.transform.MarkChildFunctions()(mod)
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
    def long_lines(value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle):
        row_num_split, col_num_split = T.int32(), T.int32()
        buffer_1_val = T.match_buffer(value_1, (row_num_split, col_num_split))
        col_1_idx = T.match_buffer(c_idx_1, (row_num_split, col_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        col_num_real = T.int32()
        vec_val = T.match_buffer(vector_in, (col_num_real,))
        row_num_real = T.int32()
        res_val = T.match_buffer(res_t, (row_num_real,))
        # with T.block("root"):
        if row_num_split > 0:
            with T.block("outer"):
                T.reads(buffer_1_val[0:row_num_split, 0:col_num_split], vec_val[0:col_num_real], col_1_idx[0:row_num_split, 0:col_num_split], res_val[0:row_num_real], buffer_1_idx[0:row_num_split])
                T.writes(res_val[0:row_num_real])
                for real_row_idx_0 in T.thread_binding((row_num_real + 255) // 256, thread="blockIdx.x"):
                    for real_row_idx_1 in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("spmv_init"):
                            v_real_row_idx = T.axis.spatial(row_num_real, real_row_idx_0 * 256 + real_row_idx_1)
                            T.where(real_row_idx_0 * 256 + real_row_idx_1 < row_num_real)
                            T.reads()
                            T.writes(res_val[v_real_row_idx])
                            res_val[v_real_row_idx] = T.float32(0.0)
                for row_idx in T.thread_binding(row_num_split, thread="blockIdx.x"):
                    with T.block("spmv_long_o"):
                        v_row_idx = T.axis.spatial(row_num_split, row_idx)
                        T.reads(buffer_1_val[v_row_idx, 0:col_num_split], vec_val[0:col_num_real], col_1_idx[v_row_idx, 0:col_num_split], res_val[buffer_1_idx[v_row_idx]], buffer_1_idx[v_row_idx])
                        T.writes()
                        res_local = T.alloc_buffer((1,), scope="shared")
                        for col_idx_0 in range((col_num_split + 127) // 128):
                            for col_idx_1 in T.thread_binding(128, thread="threadIdx.x"):
                                with T.block("spmv_long"):
                                    v_col_idx = T.axis.reduce(col_num_split, col_idx_0 * 128 + col_idx_1)
                                    T.where(col_idx_0 * 128 + col_idx_1 < col_num_split)
                                    T.reads(buffer_1_val[v_row_idx, v_col_idx], vec_val[col_1_idx[v_row_idx, v_col_idx]], col_1_idx[v_row_idx, v_col_idx])
                                    T.writes(res_local[0])
                                    with T.init():
                                        res_local[0] = T.float32(0.0)
                                    res_local[0] = res_local[0] + buffer_1_val[v_row_idx, v_col_idx] * vec_val[col_1_idx[v_row_idx, v_col_idx]]
                        tx = T.launch_thread("threadIdx.x", 128)
                        with T.block("spmv_long_2"):
                            T.reads(res_val[buffer_1_idx[v_row_idx]], buffer_1_idx[v_row_idx], res_local[0])
                            T.writes()
                            T.tvm_storage_sync("shared")
                            if tx == 0:
                                T.call_extern("float32", "atomicAdd", T.address_of(res_val[buffer_1_idx[v_row_idx]]), res_local[0])
        else:
            with T.block("outer"):
                T.reads()
                T.writes(res_val[0:row_num_real])
                for real_row_idx_0 in T.thread_binding((row_num_real + 255) // 256, thread="blockIdx.x"):
                    for real_row_idx_1 in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("spmv_init_2"):
                            v_real_row_idx = T.axis.spatial(row_num_real, real_row_idx_0 * 256 + real_row_idx_1)
                            T.where(real_row_idx_0 * 256 + real_row_idx_1 < row_num_real)
                            T.reads()
                            T.writes(res_val[v_real_row_idx])
                            res_val[v_real_row_idx] = T.float32(0.0)
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
