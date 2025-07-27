from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_global_infos({"vdevice": [I.vdevice({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-unknown-linux-gnu", "tag": ""}, 0, "global"), I.vdevice({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}, 0, "global")]})
    @T.prim_func
    def calculate_buffer_size_gpu(indptr: T.handle, output_buffer: T.Buffer((2,), "int32")):
        T.func_attr({"func_type": "preprocess", "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        row_add_one = T.int32()
        indptr_buffer = T.match_buffer(indptr, (row_add_one,), "int32")
        # with T.block("root"):
        row: T.int32 = row_add_one - 1
        for idx in T.thread_binding(2, thread="threadIdx.x"):
            with T.block("initialize"):
                T.reads()
                T.writes(output_buffer[idx])
                output_buffer[idx] = 0
        for row_idx_0 in T.thread_binding((row + 127) // 128, thread="blockIdx.x"):
            for row_idx_1 in T.thread_binding(128, thread="threadIdx.x"):
                with T.block("compute"):
                    T.where(row_idx_0 * 128 + row_idx_1 < row)
                    T.reads(indptr_buffer[0:row + 1])
                    T.writes(output_buffer[0:2])
                    nnz: T.int32 = indptr_buffer[row_idx_0 * 128 + row_idx_1 + 1] - indptr_buffer[row_idx_0 * 128 + row_idx_1]
                    if nnz <= 128:
                        T.call_extern("int32", "atomicAdd", T.address_of(output_buffer[0]), (nnz + 16 - 1) // 16)
                    else:
                        T.call_extern("int32", "atomicAdd", T.address_of(output_buffer[1]), (nnz + 64 - 1) // 64)

    @T.prim_func
    def long_lines_gpu(value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle):
        T.func_attr({"func_type": "compute", "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
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
                T.writes()
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
                            if tx == 0:
                                T.call_extern("float32", "atomicAdd", T.address_of(res_val[buffer_1_idx[v_row_idx]]), res_local[0])

    @T.prim_func
    def preprocess_gpu(indptr: T.handle, indices: T.handle, data: T.handle, row_idx_s: T.handle, col_idx_s: T.handle, val_s: T.handle, row_idx_l: T.handle, col_idx_l: T.handle, val_l: T.handle):
        T.func_attr({"func_type": "preprocess", "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        row_add_one = T.int32()
        indptr_buffer = T.match_buffer(indptr, (row_add_one,), "int32")
        nnz4all = T.int32()
        indices_buffer = T.match_buffer(indices, (nnz4all,), "int32")
        data_buffer = T.match_buffer(data, (nnz4all,))
        row_s = T.int64()
        row_idx_s_buffer = T.match_buffer(row_idx_s, (row_s,), "int32")
        col_idx_s_buffer = T.match_buffer(col_idx_s, (16, row_s), "int32")
        val_s_buffer = T.match_buffer(val_s, (16, row_s))
        row_l = T.int64()
        row_idx_l_buffer = T.match_buffer(row_idx_l, (row_l,), "int32")
        col_idx_l_buffer = T.match_buffer(col_idx_l, (row_l, 64), "int32")
        val_l_buffer = T.match_buffer(val_l, (row_l, 64))
        # with T.block("root"):
        bias_for_thread = T.alloc_buffer((2,), "int32", scope="local")
        row: T.int32 = row_add_one - 1
        for block_idx in T.thread_binding((row + 127) // 128, thread="blockIdx.x"):
            for thread_idx in T.thread_binding(128, thread="threadIdx.x"):
                row_idx: T.int32 = block_idx * 128 + thread_idx
                if row_idx < row:
                    with T.block("compute"):
                        T.reads(indptr_buffer[T.min(row_idx, 0):T.min(row_idx, 0) + (T.max(0, row_idx) + 2)], bias_for_thread[0:2], indices_buffer[indptr_buffer[row_idx]:indptr_buffer[row_idx] + (indptr_buffer[row_idx + 1] - indptr_buffer[row_idx])], data_buffer[indptr_buffer[row_idx]:indptr_buffer[row_idx] + (indptr_buffer[row_idx + 1] - indptr_buffer[row_idx])])
                        T.writes(bias_for_thread[0:2], row_idx_s_buffer[bias_for_thread[0]], col_idx_s_buffer[bias_for_thread[1] % 16, bias_for_thread[1] // 16], val_s_buffer[bias_for_thread[1] % 16, bias_for_thread[1] // 16], row_idx_l_buffer[bias_for_thread[0]], col_idx_l_buffer[bias_for_thread[1] // 64, bias_for_thread[1] % 64], val_l_buffer[bias_for_thread[1] // 64, bias_for_thread[1] % 64])
                        bias_for_thread[0] = 0
                        bias_for_thread[1] = 0
                        if indptr_buffer[row_idx + 1] - indptr_buffer[row_idx] <= 128:
                            for i in range(row_idx):
                                if indptr_buffer[i + 1] - indptr_buffer[i] <= 128:
                                    bias_for_thread[0] = bias_for_thread[0] + (indptr_buffer[i + 1] - indptr_buffer[i] + 16 - 1) // 16
                            bias_for_thread[1] = bias_for_thread[0] * 16
                            for idx in range(indptr_buffer[row_idx + 1] - indptr_buffer[row_idx]):
                                from_idx: T.int32 = indptr_buffer[row_idx] + idx
                                if idx % 16 == 0:
                                    row_idx_s_buffer[bias_for_thread[0]] = row_idx
                                    bias_for_thread[0] = bias_for_thread[0] + 1
                                x: T.int32 = bias_for_thread[1] % 16
                                y: T.int32 = bias_for_thread[1] // 16
                                col_idx_s_buffer[x, y] = indices_buffer[from_idx]
                                val_s_buffer[x, y] = data_buffer[from_idx]
                                bias_for_thread[1] = bias_for_thread[1] + 1
                            for i in range(bias_for_thread[1] % -16 * -1):
                                x: T.int32 = bias_for_thread[1] % 16
                                y: T.int32 = bias_for_thread[1] // 16
                                col_idx_s_buffer[x, y] = 0
                                val_s_buffer[x, y] = T.float32(0.0)
                                bias_for_thread[1] = bias_for_thread[1] + 1
                        else:
                            for i in range(row_idx):
                                if indptr_buffer[i + 1] - indptr_buffer[i] > 128:
                                    bias_for_thread[0] = bias_for_thread[0] + (indptr_buffer[i + 1] - indptr_buffer[i] + 64 - 1) // 64
                            bias_for_thread[1] = bias_for_thread[0] * 64
                            for idx in range(indptr_buffer[row_idx + 1] - indptr_buffer[row_idx]):
                                from_idx: T.int32 = indptr_buffer[row_idx] + idx
                                if idx % 64 == 0:
                                    row_idx_l_buffer[bias_for_thread[0]] = row_idx
                                    bias_for_thread[0] = bias_for_thread[0] + 1
                                x: T.int32 = bias_for_thread[1] // 64
                                y: T.int32 = bias_for_thread[1] % 64
                                col_idx_l_buffer[x, y] = indices_buffer[from_idx]
                                val_l_buffer[x, y] = data_buffer[from_idx]
                                bias_for_thread[1] = bias_for_thread[1] + 1
                            for i in range(bias_for_thread[1] % -64 * -1):
                                x: T.int32 = bias_for_thread[1] // 64
                                y: T.int32 = bias_for_thread[1] % 64
                                col_idx_l_buffer[x, y] = 0
                                val_l_buffer[x, y] = T.float32(0.0)
                                bias_for_thread[1] = bias_for_thread[1] + 1

    @T.prim_func
    def short_lines_easier_gpu(value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle):
        T.func_attr({"func_type": "compute", "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        col_num_split, row_num_split = T.int32(), T.int32()
        buffer_1_val = T.match_buffer(value_1, (col_num_split, row_num_split))
        col_1_idx = T.match_buffer(c_idx_1, (col_num_split, row_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        col_num_real = T.int32()
        vec_val = T.match_buffer(vector_in, (col_num_real,))
        row_num_real = T.int32()
        res_val = T.match_buffer(res_t, (row_num_real,))
        # with T.block("root"):
        if row_num_split > 0:
            with T.block("outer"):
                T.reads(buffer_1_val[0:col_num_split, 0:(row_num_split + 127) // 128 * 128], vec_val[0:col_num_real], col_1_idx[0:col_num_split, 0:(row_num_split + 127) // 128 * 128], res_val[0:row_num_real], buffer_1_idx[0:(row_num_split + 127) // 128 * 128])
                T.writes(res_val[0:row_num_real])
                for real_row_idx_0 in T.thread_binding((row_num_real + 127) // 128, thread="blockIdx.x"):
                    for real_row_idx_1 in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("spmv_init"):
                            v_real_row_idx = T.axis.spatial(row_num_real, real_row_idx_0 * 128 + real_row_idx_1)
                            T.where(real_row_idx_0 * 128 + real_row_idx_1 < row_num_real)
                            T.reads()
                            T.writes(res_val[v_real_row_idx])
                            res_val[v_real_row_idx] = T.float32(0.0)
                for row_idx_0 in T.thread_binding((row_num_split + 127) // 128, thread="blockIdx.x"):
                    for row_idx_1 in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("spmv_short_o"):
                            T.reads(buffer_1_val[0:col_num_split, row_idx_0 * 128 + row_idx_1], vec_val[0:col_num_real], col_1_idx[0:col_num_split, row_idx_0 * 128 + row_idx_1], res_val[buffer_1_idx[row_idx_0 * 128 + row_idx_1]], buffer_1_idx[row_idx_0 * 128 + row_idx_1])
                            T.writes()
                            res_local = T.alloc_buffer((1,), scope="local")
                            res_local[0] = T.float32(0.0)
                            if row_idx_0 * 128 + row_idx_1 < row_num_split:
                                for col_idx in range(col_num_split):
                                    with T.block("spmv_short"):
                                        T.reads(res_local[0], buffer_1_val[col_idx, row_idx_0 * 128 + row_idx_1], vec_val[col_1_idx[col_idx, row_idx_0 * 128 + row_idx_1]], col_1_idx[col_idx, row_idx_0 * 128 + row_idx_1])
                                        T.writes(res_local[0])
                                        res_local[0] = res_local[0] + buffer_1_val[col_idx, row_idx_0 * 128 + row_idx_1] * vec_val[col_1_idx[col_idx, row_idx_0 * 128 + row_idx_1]]
                                with T.block("spmv_short_write_back"):
                                    T.reads(res_val[buffer_1_idx[row_idx_0 * 128 + row_idx_1]], buffer_1_idx[row_idx_0 * 128 + row_idx_1], res_local[0])
                                    T.writes()
                                    T.call_extern("float32", "atomicAdd", T.address_of(res_val[buffer_1_idx[row_idx_0 * 128 + row_idx_1]]), res_local[0])
        else:
            with T.block("outer"):
                T.reads()
                T.writes(res_val[0:row_num_real])
                for real_row_idx_0 in T.thread_binding((row_num_real + 127) // 128, thread="blockIdx.x"):
                    for real_row_idx_1 in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("spmv_init_2"):
                            v_real_row_idx = T.axis.spatial(row_num_real, real_row_idx_0 * 128 + real_row_idx_1)
                            T.where(real_row_idx_0 * 128 + real_row_idx_1 < row_num_real)
                            T.reads()
                            T.writes(res_val[v_real_row_idx])
                            res_val[v_real_row_idx] = T.float32(0.0)

    @R.function(private=True)
    def fused_short_lines_easier_gpu_long_lines_gpu(gv6: R.Tensor((16, "splited_short_row_num"), dtype="float32", vdevice="cuda:0"), gv7: R.Tensor((16, "splited_short_row_num"), dtype="int32", vdevice="cuda:0"), gv8: R.Tensor(("splited_short_row_num",), dtype="int32", vdevice="cuda:0"), vin: R.Tensor(("origin_row_",), dtype="float32", vdevice="cuda:0"), gv10: R.Tensor(("splited_long_row_num", 64), dtype="float32", vdevice="cuda:0"), gv11: R.Tensor(("splited_long_row_num", 64), dtype="int32", vdevice="cuda:0"), gv12: R.Tensor(("splited_long_row_num",), dtype="int32", vdevice="cuda:0")) -> R.Tensor(("origin_row_",), dtype="float32", vdevice="cuda:0"):
        origin_row_ = T.int64()
        splited_short_row_num = T.int64()
        splited_long_row_num = T.int64()
        R.func_attr({"Composite": "spmv.custom", "Primitive": True})
        cls = Module
        with R.dataflow():
            output = R.call_tir(cls.short_lines_easier_gpu, (gv6, gv7, gv8, vin), out_sinfo=R.Tensor((origin_row_,), dtype="float32", vdevice="cuda:0"))
            gv: R.Tensor((origin_row_,), dtype="float32", vdevice="cuda:0") = R.call_tir_inplace(cls.long_lines_gpu, (gv10, gv11, gv12, vin, output), out_sinfo=R.Tensor((origin_row_,), dtype="float32", vdevice="cuda:0"), inplace_indices=[4])
            R.output(gv)
        return gv
    
    @T.prim_func
    def parent_fused_short_lines_easier_gpu_long_lines_gpu(head: T.handle, tail: T.handle, gv6: T.handle, gv7: T.handle, gv8: T.handle, vin: T.handle, gv10: T.handle, gv11: T.handle, gv12: T.handle):
        # TODO
        pass
    
    @T.prim_func
    def init_queue(head: T.handle, tail: T.handle):
        T.func_attr({"target": T.target("cuda")})
        head_val = T.match_buffer(head, (1,), "int32")
        tail_val = T.match_buffer(tail, (1,), "int32")
        with T.block("init"):
            T.reads()
            T.writes(head_val[0], tail_val[0])
            head_val[0] = 0
            tail_val[0] = 0
            
    @T.prim_func
    def init_ptr_array(gv6: T.handle, gv7: T.handle, gv8: T.handle, gv10: T.handle, gv11: T.handle, gv12: T.handle, gv6_arr: T.handle, gv7_arr: T.handle, gv8_arr: T.handle, gv10_arr: T.handle, gv11_arr: T.handle, gv12_arr: T.handle):
        T.func_attr({"target": T.target("cuda")})
        p_gv6 = T.match_buffer(gv6, (8,), "handle")
        p_gv7 = T.match_buffer(gv7, (8,), "handle")
        p_gv8 = T.match_buffer(gv8, (8,), "handle")
        p_gv10 = T.match_buffer(gv10, (8,), "handle")
        p_gv11 = T.match_buffer(gv11, (8,), "handle")
        p_gv12 = T.match_buffer(gv12, (8,), "handle")
        gv6_arr_val = T.match_buffer(gv6_arr, (8, 100, 100), "float32")
        gv7_arr_val = T.match_buffer(gv7_arr, (8, 100, 100), "int32")
        gv8_arr_val = T.match_buffer(gv8_arr, (8, 100), "int32")
        gv10_arr_val = T.match_buffer(gv10_arr, (8, 100, 64), "float32")
        gv11_arr_val = T.match_buffer(gv11_arr, (8, 100, 64), "int32")
        gv12_arr_val = T.match_buffer(gv12_arr, (8, 100), "int32")
        with T.block("init"):
            T.reads()
            T.writes(p_gv6[0:8], p_gv7[0:8], p_gv8[0:8], p_gv10[0:8], p_gv11[0:8], p_gv12[0:8])
            for i in range(8):
                p_gv6[i] = T.address_of(gv6_arr_val[i, 0, 0])
                p_gv7[i] = T.address_of(gv7_arr_val[i, 0, 0])
                p_gv8[i] = T.address_of(gv8_arr_val[i, 0])
                p_gv10[i] = T.address_of(gv10_arr_val[i, 0, 0])
                p_gv11[i] = T.address_of(gv11_arr_val[i, 0, 0])
                p_gv12[i] = T.address_of(gv12_arr_val[i, 0])


    @R.function
    def while_loop(alpha_tensor: R.Tensor((1,), dtype="float32", vdevice="cuda:0"), beta_tensor: R.Tensor((1,), dtype="float32", vdevice="cuda:0"), indptr: R.Tensor(("origin_row_add_1_",), dtype="int32", vdevice="cuda:0"), indices: R.Tensor(("origin_nnz_",), dtype="int32", vdevice="cuda:0"), data: R.Tensor(("origin_nnz_",), dtype="float32", vdevice="cuda:0"), vin: R.Tensor(("origin_row_",), dtype="float32", vdevice="cuda:0"), i: R.Tensor((), dtype="int32", vdevice="llvm:0")) -> R.Tensor(("origin_row_",), dtype="float32"):
        origin_row_ = T.int64()
        origin_row_add_1_ = T.int64()
        origin_nnz_ = T.int64()
        cls = Module
        splited_short_row_num = T.int64()
        splited_long_row_num = T.int64()
        with R.dataflow():
            gv1: R.Tensor((), dtype="int32", vdevice="llvm:0") = R.to_vdevice(R.const(1, "int32"), dst_vdevice="llvm:0")
            cond1: R.Tensor((), dtype="bool", vdevice="llvm:0") = R.less(i, gv1)
            R.output(gv1, cond1)
        if cond1:
            with R.dataflow():
                lv1 = R.call_tir(cls.calculate_buffer_size_gpu, (indptr,), out_sinfo=R.Tensor((2,), dtype="int32", vdevice="cuda:0"))
                splited_num: R.Shape(ndim=2) = R.tensor_to_shape(lv1)
                _: R.Shape([splited_short_row_num, splited_long_row_num]) = R.match_cast(splited_num, R.Shape([splited_short_row_num, splited_long_row_num]))
                lv2 = R.call_tir(cls.preprocess_gpu, (indptr, indices, data), out_sinfo=[R.Tensor((splited_short_row_num,), dtype="int32", vdevice="cuda:0"), R.Tensor((16, splited_short_row_num), dtype="int32", vdevice="cuda:0"), R.Tensor((16, splited_short_row_num), dtype="float32", vdevice="cuda:0"), R.Tensor((splited_long_row_num,), dtype="int32", vdevice="cuda:0"), R.Tensor((splited_long_row_num, 64), dtype="int32", vdevice="cuda:0"), R.Tensor((splited_long_row_num, 64), dtype="float32", vdevice="cuda:0")])
                gv6: R.Tensor((16, splited_short_row_num), dtype="float32", vdevice="cuda:0") = lv2[2]
                gv7: R.Tensor((16, splited_short_row_num), dtype="int32", vdevice="cuda:0") = lv2[1]
                gv8: R.Tensor((splited_short_row_num,), dtype="int32", vdevice="cuda:0") = lv2[0]
                gv10: R.Tensor((splited_long_row_num, 64), dtype="float32", vdevice="cuda:0") = lv2[5]
                gv11: R.Tensor((splited_long_row_num, 64), dtype="int32", vdevice="cuda:0") = lv2[4]
                gv12: R.Tensor((splited_long_row_num,), dtype="int32", vdevice="cuda:0") = lv2[3]
                head = R.builtin.alloc_tensor(R.shape([1]), dtype="int32", runtime_device_index=1)
                tail = R.builtin.alloc_tensor(R.shape([1]), dtype="int32", runtime_device_index=1)
                gv_queue = R.call_tir_inplace(cls.init_queue, (head, tail), inplace_indices=[0, 1], out_sinfo=[R.Tensor((1,), dtype="int32", vdevice="cuda:0"), R.Tensor((1,), dtype="int32", vdevice="cuda:0")])
                p_gv6 = R.builtin.alloc_tensor(R.shape([8]), dtype="handle", runtime_device_index=1)
                p_gv7 = R.builtin.alloc_tensor(R.shape([8]), dtype="handle", runtime_device_index=1)
                p_gv8 = R.builtin.alloc_tensor(R.shape([8]), dtype="handle", runtime_device_index=1)
                p_gv10 = R.builtin.alloc_tensor(R.shape([8]), dtype="handle", runtime_device_index=1)
                p_gv11 = R.builtin.alloc_tensor(R.shape([8]), dtype="handle", runtime_device_index=1)
                p_gv12 = R.builtin.alloc_tensor(R.shape([8]), dtype="handle", runtime_device_index=1)
                gv6_arr = R.builtin.alloc_tensor(R.shape([8, 100,100]), dtype="float32", runtime_device_index=1)
                gv7_arr = R.builtin.alloc_tensor(R.shape([8,100,100]), dtype="int32", runtime_device_index=1)
                gv8_arr = R.builtin.alloc_tensor(R.shape([8,100]), dtype="int32", runtime_device_index=1)
                gv10_arr = R.builtin.alloc_tensor(R.shape([8,100,64]), dtype="float32", runtime_device_index=1)
                gv11_arr = R.builtin.alloc_tensor(R.shape([8,100,64]), dtype="int32", runtime_device_index=1)
                gv12_arr = R.builtin.alloc_tensor(R.shape([8,100]), dtype="int32", runtime_device_index=1)
                gv_ptr_array = R.call_tir_inplace(cls.init_ptr_array, (p_gv6, p_gv7, p_gv8, p_gv10, p_gv11, p_gv12, gv6_arr, gv7_arr, gv8_arr, gv10_arr, gv11_arr, gv12_arr), inplace_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], out_sinfo=[R.Tensor((8,), dtype="handle", vdevice="cuda:0"), R.Tensor((8,), dtype="handle", vdevice="cuda:0"), R.Tensor((8,), dtype="handle", vdevice="cuda:0"), R.Tensor((8,), dtype="handle", vdevice="cuda:0"), R.Tensor((8,), dtype="handle", vdevice="cuda:0"), R.Tensor((8,), dtype="handle", vdevice="cuda:0"), R.Tensor((8, 100, 100), dtype="float32", vdevice="cuda:0"), R.Tensor((8, 100, 100), dtype="int32", vdevice="cuda:0"), R.Tensor((8, 100), dtype="int32", vdevice="cuda:0"), R.Tensor((8, 100, 64), dtype="float32", vdevice="cuda:0"), R.Tensor((8, 100, 64), dtype="int32", vdevice="cuda:0"), R.Tensor((8, 100), dtype="int32", vdevice="cuda:0")])
                _ = R.call_tir_inplace(cls.parent_fused_short_lines_easier_gpu_long_lines_gpu, (gv_queue[0], gv_queue[1], gv_ptr_array[0], gv_ptr_array[1], gv_ptr_array[2], vin, gv_ptr_array[3], gv_ptr_array[4], gv_ptr_array[5]),inplace_indices=[0, 1], out_sinfo=[R.Tensor((1,), dtype="int32", vdevice="cuda:0"), R.Tensor((1,), dtype="int32", vdevice="cuda:0")])
                
                lv: R.Tensor((origin_row_,), dtype="float32", vdevice="cuda:0") = R.builtin.alloc_tensor(R.shape([origin_row_]), dtype="float32", runtime_device_index=1)
                # TODO: get lv from FPGA
                
                i_: R.Tensor((), dtype="int32", vdevice="llvm:0") = R.add(i, gv1)
                lv1_1: R.Tensor((origin_row_,), dtype="float32") = cls.while_loop(alpha_tensor, beta_tensor, indptr, indices, data, lv, i_)
                then_branch_with_dyn: R.Tensor(dtype="float32", ndim=1) = R.match_cast(lv1_1, R.Tensor(dtype="float32", ndim=1))
                R.output(then_branch_with_dyn)
            ret: R.Tensor(dtype="float32", ndim=1) = then_branch_with_dyn
        else:
            else_branch_with_dyn: R.Tensor(dtype="float32", ndim=1) = R.match_cast(vin, R.Tensor(dtype="float32", ndim=1))
            ret: R.Tensor(dtype="float32", ndim=1) = else_branch_with_dyn
        return ret

    @R.function
    def main(alpha_tensor: R.Tensor((1,), dtype="float32"), beta_tensor: R.Tensor((1,), dtype="float32"), indptr: R.Tensor(("origin_row_add_1",), dtype="int32"), indices: R.Tensor(("origin_nnz",), dtype="int32"), data: R.Tensor(("origin_nnz",), dtype="float32"), vin: R.Tensor(("origin_col",), dtype="float32"), iter_num: R.Tensor((), dtype="int32")) -> R.Tensor(("origin_col",), dtype="float32"):
        origin_col = T.int64()
        origin_row_add_1 = T.int64()
        origin_nnz = T.int64()
        cls = Module
        with R.dataflow():
            gv: R.Tensor((1,), dtype="float32", vdevice="cuda:0") = R.to_vdevice(alpha_tensor, dst_vdevice="cuda:0")
            gv1: R.Tensor((1,), dtype="float32", vdevice="cuda:0") = R.to_vdevice(beta_tensor, dst_vdevice="cuda:0")
            gv2: R.Tensor((origin_row_add_1,), dtype="int32", vdevice="cuda:0") = R.to_vdevice(indptr, dst_vdevice="cuda:0")
            gv3: R.Tensor((origin_nnz,), dtype="int32", vdevice="cuda:0") = R.to_vdevice(indices, dst_vdevice="cuda:0")
            gv4: R.Tensor((origin_nnz,), dtype="float32", vdevice="cuda:0") = R.to_vdevice(data, dst_vdevice="cuda:0")
            gv5: R.Tensor((origin_col,), dtype="float32", vdevice="cuda:0") = R.to_vdevice(vin, dst_vdevice="cuda:0")
            gv6: R.Tensor((), dtype="int32", vdevice="llvm:0") = R.to_vdevice(iter_num, dst_vdevice="llvm:0")
            gv_1: R.Tensor((origin_col,), dtype="float32") = cls.while_loop(gv, gv1, gv2, gv3, gv4, gv5, gv6)
            R.output(gv_1)
        return gv_1

if __name__ == "__main__":
    pass
