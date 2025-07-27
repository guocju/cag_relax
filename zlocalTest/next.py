from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    I.module_attrs({"entry_func": "parent_kernel"})
    @T.prim_func
    def long_lines(value_1: T.handle("float32", "global"), c_idx_1: T.handle("int32", "global"), idx_1: T.handle("int32", "global"), vector_in: T.handle("float32", "global"), res_t: T.handle("float32", "global"), row_num_real: T.int32, col_num_split: T.int32, row_num_split: T.int32, col_num_real: T.int32):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        for threadIdx_x in T.thread_binding(1, thread="threadIdx.x"):
            buffer_1_val = T.decl_buffer((row_num_split, col_num_split), data=value_1)
            col_1_idx = T.decl_buffer((row_num_split, col_num_split), "int32", data=c_idx_1)
            buffer_1_idx = T.decl_buffer((row_num_split,), "int32", data=idx_1)
            vec_val = T.decl_buffer((col_num_real,), data=vector_in)
            res_val = T.decl_buffer((row_num_real,), data=res_t)
            if 0 < row_num_split:
                Module.long_lines_kernel_3(res_t, row_num_real)
                Module.long_lines_kernel_3_1(idx_1, value_1, c_idx_1, res_t, vector_in, col_num_real, col_num_split, row_num_real, row_num_split)
            else:
                Module.long_lines_kernel_3_2(res_t, row_num_real)

    @T.prim_func
    def long_lines_kernel_3(res_val: T.handle("float32", "global"), row_num_real: T.int32):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_global_func": T.bool(True), "tir.noalias": T.bool(True)})
        res_val_1 = T.decl_buffer((row_num_real,), data=res_val)
        blockIdx_x = T.launch_thread("blockIdx.x", (row_num_real + 255) // 256)
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        if blockIdx_x * 256 + threadIdx_x < row_num_real:
            res_val_1[blockIdx_x * 256 + threadIdx_x] = T.float32(0.0)

    @T.prim_func
    def long_lines_kernel_3_1(buffer_1_idx: T.handle("int32", "global"), buffer_1_val: T.handle("float32", "global"), col_1_idx: T.handle("int32", "global"), res_val: T.handle("float32", "global"), vec_val: T.handle("float32", "global"), col_num_real: T.int32, col_num_split: T.int32, row_num_real: T.int32, row_num_split: T.int32):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_global_func": T.bool(True), "tir.noalias": T.bool(True)})
        buffer_1_idx_1 = T.decl_buffer((row_num_split,), "int32", data=buffer_1_idx)
        res_val_1 = T.decl_buffer((row_num_real,), data=res_val)
        res_local = T.handle("float32", "shared")
        res_local_1 = T.decl_buffer((1,), data=res_local, scope="shared")
        red_result = T.handle("float32", "shared")
        red_result_1 = T.decl_buffer((1,), data=red_result, scope="shared")
        col_1_idx_1 = T.decl_buffer((row_num_split * col_num_split,), "int32", data=col_1_idx)
        vec_val_1 = T.decl_buffer((col_num_real,), data=vec_val)
        buffer_1_val_1 = T.decl_buffer((row_num_split * col_num_split,), data=buffer_1_val)
        in_thread_res_local = T.handle("float32", "local")
        in_thread_res_local_1 = T.decl_buffer((1,), data=in_thread_res_local, scope="local")
        blockIdx_x = T.launch_thread("blockIdx.x", row_num_split)
        in_thread_res_local = T.allocate([1], "float32", "local")
        red_result = T.allocate([1], "float32", "shared")
        T.attr(red_result, "volatile_scope", 1)
        res_local = T.allocate([1], "float32", "shared")
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        in_thread_res_local_1[0] = T.float32(0.0)
        for col_idx_0 in range((col_num_split + 127) // 128):
            if col_idx_0 * 128 + threadIdx_x < col_num_split:
                in_thread_res_local_1[0] = in_thread_res_local_1[0] + buffer_1_val_1[col_idx_0 * 128 + blockIdx_x * col_num_split + threadIdx_x] * vec_val_1[col_1_idx_1[col_idx_0 * 128 + blockIdx_x * col_num_split + threadIdx_x]]
        with T.attr(T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0.0)]), "reduce_scope", T.reinterpret("handle", T.uint64(0))):
            red_buf0 = T.decl_buffer((1,), scope="local")
            mask = T.decl_buffer((1,), "uint32", scope="local")
            t0 = T.decl_buffer((1,), scope="local")
            red_buf0_1 = T.decl_buffer((1,), scope="local")
            mask_1 = T.decl_buffer((1,), "uint32", scope="local")
            t0_1 = T.decl_buffer((1,), scope="local")
            red_buf_staging = T.decl_buffer((4,), scope="shared")
            red_buf0_1[0] = in_thread_res_local_1[0]
            mask_1[0] = T.tvm_warp_activemask()
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 16, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 8, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 4, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 2, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 1, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            if threadIdx_x % 32 == 0:
                red_buf_staging[threadIdx_x // 32] = red_buf0_1[0]
            T.tvm_storage_sync("shared")
            if threadIdx_x < 4:
                red_buf0[0] = red_buf_staging[threadIdx_x]for threadIdx_x in T.thread_binding(1, thread="threadIdx.x"):
            mask[0] = T.tvm_warp_activemask()
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 2, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 1, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            if threadIdx_x == 0:
                red_result_1[0] = red_buf0[0]
            T.tvm_storage_sync("shared")
        if threadIdx_x == 0:
            res_local_1[0] = red_result_1[0]
        if threadIdx_x == 0:
            T.tvm_storage_sync("shared")
            T.call_extern("float32", "atomicAdd", T.address_of(res_val_1[buffer_1_idx_1[blockIdx_x]]), res_local_1[0])

    @T.prim_func
    def long_lines_kernel_3_2(res_val: T.handle("float32", "global"), row_num_real: T.int32):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_global_func": T.bool(True), "tir.noalias": T.bool(True)})
        res_val_1 = T.decl_buffer((row_num_real,), data=res_val)
        blockIdx_x = T.launch_thread("blockIdx.x", (row_num_real + 255) // 256)
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        if blockIdx_x * 256 + threadIdx_x < row_num_real:
            res_val_1[blockIdx_x * 256 + threadIdx_x] = T.float32(0.0)

    @T.prim_func
    def parent_kernel(value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle, row_num_real: T.int32, col_num_split: T.int32, col_num_real: T.int32, row_num_split: T.int32):
        T.func_attr({"target": T.target({"arch": "sm_86", "host": {"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-pc-linux-gnu", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        buffer_1_val = T.match_buffer(value_1, (row_num_split, col_num_split))
        col_1_idx = T.match_buffer(c_idx_1, (row_num_split, col_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        vec_val = T.match_buffer(vector_in, (col_num_real,))
        res_val = T.match_buffer(res_t, (row_num_real,))
        Module.parent_kernel_kernel_1(buffer_1_idx.data, buffer_1_val.data, col_1_idx.data, res_val.data, vec_val.data, col_num_real, col_num_split, row_num_real, row_num_split)

    @T.prim_func
    def parent_kernel_kernel_1(buffer_1_idx: T.handle("int32", "global"), buffer_1_val: T.handle("float32", "global"), col_1_idx: T.handle("int32", "global"), res_val: T.handle("float32", "global"), vec_val: T.handle("float32", "global"), col_num_real: T.int32, col_num_split: T.int32, row_num_real: T.int32, row_num_split: T.int32):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_global_func": T.bool(True), "tir.noalias": T.bool(True)})
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 1)
        for k in range(3):
            Module.long_lines(buffer_1_val, col_1_idx, buffer_1_idx, vec_val, res_val, col_num_split, row_num_real, row_num_split, col_num_real)