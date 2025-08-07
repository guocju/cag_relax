import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.script import tir as T, ir as I, relax as R
from config import val_dtype, thread_extent_for_short, thread_extent_for_long, length_threshold, long_align_val, short_align_val

class ScheduleFn:
    def prepreprocess_sch(input_func):
        mod = tvm.IRModule.from_expr(input_func)
        sch = tvm.tir.Schedule(mod)
        sch.work_on("calculate_buffer_size_gpu")
        block = sch.get_block("compute")
        loop = sch.get_loops(block)[0]
        block_loop, thread_loop = sch.split(loop, [None, thread_extent_for_short])
        sch.bind(block_loop, "blockIdx.x")
        sch.bind(thread_loop, "threadIdx.x")
        return sch.mod["calculate_buffer_size_gpu"]
    
    def long_lines_sch(input_func):
        mod = tvm.IRModule.from_expr(input_func)
        sch = tvm.tir.Schedule(mod)
        sch.work_on("long_lines_gpu")

        block_b = sch.get_block("spmv_long")
        [col] = sch.get_loops(block_b)
        _, col_oi = sch.split(col, [None, thread_extent_for_long])
        sch.bind(col_oi, "threadIdx.x")

        return sch.mod["long_lines_gpu"]

    def short_lines_easier_sch(input_func):
        mod = tvm.IRModule.from_expr(input_func)
        sch = tvm.tir.Schedule(mod)
        sch.work_on("short_lines_easier_gpu")

        block_b = sch.get_block("spmv_init")
        [row] = sch.get_loops(block_b)
        row_oi, row_ii = sch.split(row, [None, thread_extent_for_short])
        sch.bind(row_oi, "blockIdx.x")
        sch.bind(row_ii, "threadIdx.x")

        block_b = sch.get_block("spmv_init_2")
        [row] = sch.get_loops(block_b)
        row_oi, row_ii = sch.split(row, [None, thread_extent_for_short])
        sch.bind(row_oi, "blockIdx.x")
        sch.bind(row_ii, "threadIdx.x")

        return sch.mod["short_lines_easier_gpu"]


@I.ir_module(post_proc_hook={"calculate_buffer_size_gpu": ScheduleFn.prepreprocess_sch, "long_lines_gpu": ScheduleFn.long_lines_sch, "short_lines_easier_gpu": ScheduleFn.short_lines_easier_sch})
class OptimizedSpmv:
    @T.prim_func
    def calculate_buffer_size_cpu(
        indptr: T.handle,
        output: T.handle,
    ) -> None:
        T.func_attr({"target": T.target("llvm"), "func_type": "preprocess"})
        row_add_one = T.int32()
        indptr_buffer = T.match_buffer(indptr, (row_add_one,), "int32")
        output_buffer = T.match_buffer(output, (2,), "int32")
        # [0] : thread_kernel_size
        # [1] : block_kernel_size

        row: T.int32 = row_add_one - 1
        with T.block("root"):
            T.reads(indptr_buffer[0 : row + 1])
            T.writes(output_buffer[0:2])
            for idx in T.serial(2):
                output_buffer[idx] = T.int32(0)
            for row_idx in T.serial(row):
                nnz: T.int32 = indptr_buffer[row_idx + 1] - indptr_buffer[row_idx]
                if nnz <= T.int32(length_threshold):
                    output_buffer[0] += T.floordiv(
                        nnz + T.int32(short_align_val) - 1, T.int32(short_align_val)
                    )
                else:
                    output_buffer[1] += T.floordiv(
                        nnz + T.int32(long_align_val) - 1, T.int32(long_align_val)
                    )


    @T.prim_func
    def calculate_buffer_size_gpu(
        indptr: T.handle,
        output: T.handle,
    ) -> None:
        T.func_attr({"target": T.target("cuda"), "func_type": "preprocess"})
        row_add_one = T.int32()
        indptr_buffer = T.match_buffer(indptr, (row_add_one,), "int32")
        output_buffer = T.match_buffer(output, (2,), "int32")

        row: T.int32 = row_add_one - 1
        for row_idx in range(row):
            with T.block("compute"):
                T.reads(indptr_buffer[0 : row + 1])
                T.writes(output_buffer[0:2])
                nnz: T.int32 = indptr_buffer[row_idx + 1] - indptr_buffer[row_idx]
                if nnz <= T.cast(length_threshold, "int32"):
                    T.call_extern(
                        "int32",
                        "atomicAdd",
                        T.call_intrin("handle", "tir.address_of", output_buffer[0]),
                        T.floordiv(
                            nnz + T.cast(short_align_val, "int32") - 1,
                            T.cast(short_align_val, "int32"),
                        ),
                    )
                else:
                    T.call_extern(
                        "int32",
                        "atomicAdd",
                        T.call_intrin("handle", "tir.address_of", output_buffer[1]),
                        T.floordiv(
                            nnz + T.cast(long_align_val, "int32") - 1,
                            T.cast(long_align_val, "int32"),
                        ),
                    )


    @T.prim_func
    def preprocess_cpu(
        indptr: T.handle,
        indices: T.handle,
        data: T.handle,
        row_idx_s: T.handle,
        col_idx_s: T.handle,
        val_s: T.handle,
        row_idx_l: T.handle,
        col_idx_l: T.handle,
        val_l: T.handle,
    ) -> None:
        T.func_attr({"target": T.target("llvm"), "func_type": "preprocess"})
        row_add_one = T.int32()
        nnz4all = T.int32()
        indptr_buffer = T.match_buffer(indptr, (row_add_one,), "int32")
        indices_buffer = T.match_buffer(indices, (nnz4all,), "int32")
        data_buffer = T.match_buffer(data, (nnz4all,), val_dtype)

        row_s = T.int64()
        row_l = T.int64()

        row_idx_s_buffer = T.match_buffer(row_idx_s, (row_s,), "int32")
        col_idx_s_buffer = T.match_buffer(col_idx_s, (short_align_val, row_s), "int32")
        val_s_buffer = T.match_buffer(val_s, (short_align_val, row_s), val_dtype)
        status_s = T.alloc_buffer((2,), "int32")

        row_idx_l_buffer = T.match_buffer(row_idx_l, (row_l,), "int32")
        col_idx_l_buffer = T.match_buffer(col_idx_l, (row_l, long_align_val), "int32")
        val_l_buffer = T.match_buffer(val_l, (row_l, long_align_val), val_dtype)
        status_l = T.alloc_buffer((2,), "int32")

        row: T.int32 = row_add_one - 1
        with T.block("initialize"):
            status_s[0] = 0
            status_s[1] = 0
            status_l[0] = 0
            status_l[1] = 0

        with T.block("compute"):
            for row_idx in T.serial(row):
                nnz = indptr_buffer[row_idx + 1] - indptr_buffer[row_idx]
                if nnz <= T.int32(length_threshold):
                    for idx in T.serial(nnz):
                        from_idx: T.int32 = indptr_buffer[row_idx] + idx
                        if idx % T.int32(short_align_val) == 0:
                            row_idx_s_buffer[status_s[0]] = row_idx
                            status_s[0] += 1

                        x: T.int32 = T.floormod(status_s[1], T.int32(short_align_val))
                        y: T.int32 = T.floordiv(status_s[1], T.int32(short_align_val))
                        col_idx_s_buffer[x, y] = indices_buffer[from_idx]
                        val_s_buffer[x, y] = data_buffer[from_idx]
                        status_s[1] += 1

                    if status_s[1] % T.int32(short_align_val) != 0:
                        for i in T.serial(
                            T.int32(short_align_val)
                            - T.floormod(status_s[1], T.int32(short_align_val))
                        ):
                            x: T.int32 = T.floormod(status_s[1], T.int32(short_align_val))
                            y: T.int32 = T.floordiv(status_s[1], T.int32(short_align_val))
                            col_idx_s_buffer[x, y] = 0
                            val_s_buffer[x, y] = 0
                            status_s[1] += 1

                else:
                    for idx in T.serial(nnz):
                        from_idx: T.int32 = indptr_buffer[row_idx] + idx
                        if idx % T.int32(long_align_val) == 0:
                            row_idx_l_buffer[status_l[0]] = row_idx
                            status_l[0] += 1

                        x: T.int32 = T.floordiv(status_l[1], T.int32(long_align_val))
                        y: T.int32 = T.floormod(status_l[1], T.int32(long_align_val))
                        col_idx_l_buffer[x, y] = indices_buffer[from_idx]
                        val_l_buffer[x, y] = data_buffer[from_idx]
                        status_l[1] += 1

                    if status_l[1] % T.int32(long_align_val) != 0:
                        for i in T.serial(
                            T.int32(long_align_val)
                            - T.floormod(status_l[1], T.int32(long_align_val))
                        ):
                            x: T.int32 = T.floordiv(status_l[1], T.int32(long_align_val))
                            y: T.int32 = T.floormod(status_l[1], T.int32(long_align_val))
                            col_idx_l_buffer[x, y] = 0
                            val_l_buffer[x, y] = 0
                            status_l[1] += 1


    @T.prim_func
    def preprocess_gpu(
        indptr: T.handle,
        indices: T.handle,
        data: T.handle,
        row_idx_s: T.handle,
        col_idx_s: T.handle,
        val_s: T.handle,
        row_idx_l: T.handle,
        col_idx_l: T.handle,
        val_l: T.handle,
    ) -> None:
        T.func_attr({"target": T.target("cuda"), "func_type": "preprocess"})
        row_add_one = T.int32()
        nnz4all = T.int32()
        indptr_buffer = T.match_buffer(indptr, (row_add_one,), "int32")
        indices_buffer = T.match_buffer(indices, (nnz4all,), "int32")
        data_buffer = T.match_buffer(data, (nnz4all,), "float32")

        row_s = T.int64()
        row_l = T.int64()

        row_idx_s_buffer = T.match_buffer(row_idx_s, (row_s,), "int32")
        col_idx_s_buffer = T.match_buffer(col_idx_s, (short_align_val, row_s), "int32")
        val_s_buffer = T.match_buffer(val_s, (short_align_val, row_s), "float32")

        row_idx_l_buffer = T.match_buffer(row_idx_l, (row_l,), "int32")
        col_idx_l_buffer = T.match_buffer(col_idx_l, (row_l, long_align_val), "int32")
        val_l_buffer = T.match_buffer(val_l, (row_l, long_align_val), "float32")
        bias_for_thread = T.alloc_buffer(
            [
                2,
            ],
            "int32",
            scope="local",
        )

        row: T.int32 = row_add_one - 1

        for block_idx in T.thread_binding(T.ceildiv(row, thread_extent_for_short), "blockIdx.x"):
            for thread_idx in T.thread_binding(thread_extent_for_short, "threadIdx.x"):
                row_idx: T.int32 = block_idx * thread_extent_for_short + thread_idx
                if row_idx < row:
                    with T.block("compute"):
                        bias_for_thread[0] = 0  # row bias
                        bias_for_thread[1] = 0  # nnz bias
                        if indptr_buffer[row_idx + 1] - indptr_buffer[row_idx] <= T.cast(
                            length_threshold, "int32"
                        ):
                            for i in T.serial(row_idx):
                                if indptr_buffer[i + 1] - indptr_buffer[i] <= T.cast(
                                    length_threshold, "int32"
                                ):
                                    bias_for_thread[0] += T.ceildiv(
                                        indptr_buffer[i + 1] - indptr_buffer[i],
                                        T.cast(short_align_val, "int32"),
                                    )
                            bias_for_thread[1] = bias_for_thread[0] * T.cast(
                                short_align_val, "int32"
                            )
                            for idx in T.serial(
                                indptr_buffer[row_idx + 1] - indptr_buffer[row_idx]
                            ):
                                from_idx: T.int32 = indptr_buffer[row_idx] + idx
                                if idx % T.cast(short_align_val, "int32") == 0:
                                    row_idx_s_buffer[bias_for_thread[0]] = row_idx
                                    bias_for_thread[0] += 1

                                x: T.int32 = T.floormod(
                                    bias_for_thread[1], T.cast(short_align_val, "int32")
                                )
                                y: T.int32 = T.floordiv(
                                    bias_for_thread[1], T.cast(short_align_val, "int32")
                                )
                                col_idx_s_buffer[x, y] = indices_buffer[from_idx]
                                val_s_buffer[x, y] = data_buffer[from_idx]
                                bias_for_thread[1] += 1

                            if bias_for_thread[1] % T.cast(short_align_val, "int32") != 0:
                                for i in T.serial(
                                    T.cast(short_align_val, "int32")
                                    - T.floormod(
                                        bias_for_thread[1], T.cast(short_align_val, "int32")
                                    )
                                ):
                                    x: T.int32 = T.floormod(
                                        bias_for_thread[1], T.cast(short_align_val, "int32")
                                    )
                                    y: T.int32 = T.floordiv(
                                        bias_for_thread[1], T.cast(short_align_val, "int32")
                                    )
                                    col_idx_s_buffer[x, y] = 0
                                    val_s_buffer[x, y] = 0
                                    bias_for_thread[1] += 1

                        else:
                            for i in T.serial(row_idx):
                                if indptr_buffer[i + 1] - indptr_buffer[i] > T.cast(
                                    length_threshold, "int32"
                                ):
                                    bias_for_thread[0] += T.ceildiv(
                                        indptr_buffer[i + 1] - indptr_buffer[i],
                                        T.cast(long_align_val, "int32"),
                                    )
                            bias_for_thread[1] = bias_for_thread[0] * T.cast(
                                long_align_val, "int32"
                            )
                            for idx in T.serial(
                                indptr_buffer[row_idx + 1] - indptr_buffer[row_idx]
                            ):
                                from_idx: T.int32 = indptr_buffer[row_idx] + idx
                                if idx % T.cast(long_align_val, "int32") == 0:
                                    row_idx_l_buffer[bias_for_thread[0]] = row_idx
                                    bias_for_thread[0] += 1

                                x: T.int32 = T.floordiv(
                                    bias_for_thread[1], T.cast(long_align_val, "int32")
                                )
                                y: T.int32 = T.floormod(
                                    bias_for_thread[1], T.cast(long_align_val, "int32")
                                )
                                col_idx_l_buffer[x, y] = indices_buffer[from_idx]
                                val_l_buffer[x, y] = data_buffer[from_idx]
                                bias_for_thread[1] += 1

                            if bias_for_thread[1] % T.cast(long_align_val, "int32") != 0:
                                for i in T.serial(
                                    T.cast(long_align_val, "int32")
                                    - T.floormod(
                                        bias_for_thread[1], T.cast(long_align_val, "int32")
                                    )
                                ):
                                    x: T.int32 = T.floordiv(
                                        bias_for_thread[1], T.cast(long_align_val, "int32")
                                    )
                                    y: T.int32 = T.floormod(
                                        bias_for_thread[1], T.cast(long_align_val, "int32")
                                    )
                                    col_idx_l_buffer[x, y] = 0
                                    val_l_buffer[x, y] = 0
                                    bias_for_thread[1] += 1


    @T.prim_func
    def long_lines_cpu(
        value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle
    ) -> None:
        T.func_attr({"tir.noalias": True, "target": T.target("llvm"), "func_type": "compute"})

        row_num_split = T.int32()
        col_num_split = T.int32()
        row_num_real = T.int32()
        col_num_real = T.int32()

        buffer_1_val = T.match_buffer(value_1, (row_num_split, col_num_split), val_dtype)
        col_1_idx = T.match_buffer(c_idx_1, (row_num_split, col_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        vec_val = T.match_buffer(vector_in, (col_num_real,), val_dtype)
        res_val = T.match_buffer(res_t, (row_num_real,), val_dtype)

        if row_num_split > 0:
            with T.block("outer"):

                for row_idx in range(row_num_split):
                    with T.block("spmv_long_o"):
                        v_row_idx = T.axis.remap("S", [row_idx])
                        res_local = T.alloc_buffer(
                            [
                                1,
                            ],
                            dtype=val_dtype,
                        )
                        for col_idx in range(col_num_split):
                            with T.block("spmv_long"):
                                v_col_idx = T.axis.remap("R", [col_idx])
                                with T.init():
                                    res_local[0] = 0.0
                                res_local[0] += (
                                    buffer_1_val[v_row_idx, v_col_idx]
                                    * vec_val[col_1_idx[v_row_idx, v_col_idx]]
                                )

                        with T.block("spmv_long_2"):
                            res_val[buffer_1_idx[v_row_idx]] += res_local[0]


    @T.prim_func
    def long_lines_gpu(
        value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle
    ) -> None:
        T.func_attr({"target": T.target("cuda"), "func_type": "compute"})
        row_num_split = T.int32()
        col_num_split = T.int32()
        row_num_real = T.int32()
        col_num_real = T.int32()
        tx = T.env_thread("threadIdx.x")

        buffer_1_val = T.match_buffer(value_1, (row_num_split, col_num_split), val_dtype)
        col_1_idx = T.match_buffer(c_idx_1, (row_num_split, col_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        vec_val = T.match_buffer(vector_in, (col_num_real,), val_dtype)
        res_val = T.match_buffer(res_t, (row_num_real,), val_dtype)

        if row_num_split > 0:
            with T.block("outer"):
                for row_idx in T.thread_binding(row_num_split, thread="blockIdx.x"):
                    with T.block("spmv_long_o"):
                        v_row_idx = T.axis.remap("S", [row_idx])
                        res_local = T.alloc_buffer(
                            [
                                1,
                            ],
                            dtype=val_dtype,
                            scope="shared",
                        )
                        for col_idx in range(col_num_split):
                            with T.block("spmv_long"):
                                v_col_idx = T.axis.remap("R", [col_idx])
                                with T.init():
                                    res_local[0] = T.float32(0)
                                res_local[0] += (
                                    buffer_1_val[v_row_idx, v_col_idx]
                                    * vec_val[col_1_idx[v_row_idx, v_col_idx]]
                                )

                        T.launch_thread(tx, thread_extent_for_long)
                        with T.block("spmv_long_2"):
                            if tx == 0:
                                T.call_extern(
                                    val_dtype,
                                    "atomicAdd",
                                    T.call_intrin(
                                        "handle", "tir.address_of", res_val[buffer_1_idx[v_row_idx]]
                                    ),
                                    res_local[0],
                                )


    @T.prim_func
    def short_lines_easier_cpu(
        value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle
    ) -> None:
        T.func_attr({"tir.noalias": True, "target": T.target("llvm"), "func_type": "compute"})

        row_num_split = T.int32()
        col_num_split = T.int32()
        row_num_real = T.int32()
        col_num_real = T.int32()

        buffer_1_val = T.match_buffer(value_1, (col_num_split, row_num_split), val_dtype)
        col_1_idx = T.match_buffer(c_idx_1, (col_num_split, row_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        vec_val = T.match_buffer(vector_in, (col_num_real,), val_dtype)
        res_val = T.match_buffer(res_t, (row_num_real,), val_dtype)

        if row_num_split > 0:
            with T.block("outer"):
                for real_row_idx in range(row_num_real):
                    with T.block("spmv_init"):
                        v_real_row_idx = T.axis.remap("S", [real_row_idx])
                        res_val[v_real_row_idx] = 0.0

                for row_idx_0 in range(T.ceildiv(row_num_split, thread_extent_for_short)):
                    for row_idx_1 in range(thread_extent_for_short):
                        with T.block("spmv_short_o"):
                            res_local = T.alloc_buffer(
                                [
                                    1,
                                ],
                                dtype=val_dtype,
                                scope="local",
                            )
                            res_local[0] = 0.0
                            if row_idx_0 * thread_extent_for_short + row_idx_1 < row_num_split:
                                for col_idx in range(col_num_split):
                                    with T.block("spmv_short"):
                                        res_local[0] += (
                                            buffer_1_val[
                                                col_idx, row_idx_0 * thread_extent_for_short + row_idx_1
                                            ]
                                            * vec_val[
                                                col_1_idx[
                                                    col_idx, row_idx_0 * thread_extent_for_short + row_idx_1
                                                ]
                                            ]
                                        )

                                with T.block("spmv_short_write_back"):
                                    res_val[
                                        buffer_1_idx[row_idx_0 * thread_extent_for_short + row_idx_1]
                                    ] += res_local[0]

        else:
            with T.block("outer"):
                for real_row_idx in range(row_num_real):
                    with T.block("spmv_init_2"):
                        v_real_row_idx = T.axis.remap("S", [real_row_idx])
                        res_val[v_real_row_idx] = 0.0


    @T.prim_func
    def short_lines_easier_gpu(
        value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle
    ) -> None:
        T.func_attr({"target": T.target("cuda"), "func_type": "compute"})
        row_num_split = T.int32()
        col_num_split = T.int32()
        row_num_real = T.int32()
        col_num_real = T.int32()

        buffer_1_val = T.match_buffer(value_1, (col_num_split, row_num_split), val_dtype)
        col_1_idx = T.match_buffer(c_idx_1, (col_num_split, row_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        vec_val = T.match_buffer(vector_in, (col_num_real,), val_dtype)
        res_val = T.match_buffer(res_t, (row_num_real,), val_dtype)

        if row_num_split > 0:
            with T.block("outer"):
                for real_row_idx in range(row_num_real):
                    with T.block("spmv_init"):
                        v_real_row_idx = T.axis.remap("S", [real_row_idx])
                        res_val[v_real_row_idx] = T.float32(0)

                for row_idx_0 in T.thread_binding(
                    T.ceildiv(row_num_split, thread_extent_for_short), thread="blockIdx.x"
                ):
                    for row_idx_1 in T.thread_binding(thread_extent_for_short, thread="threadIdx.x"):
                        with T.block("spmv_short_o"):
                            res_local = T.alloc_buffer(
                                [
                                    1,
                                ],
                                dtype=val_dtype,
                                scope="local",
                            )
                            res_local[0] = T.float32(0)
                            if row_idx_0 * thread_extent_for_short + row_idx_1 < row_num_split:
                                for col_idx in range(col_num_split):
                                    with T.block("spmv_short"):
                                        res_local[0] += (
                                            buffer_1_val[
                                                col_idx, row_idx_0 * thread_extent_for_short + row_idx_1
                                            ]
                                            * vec_val[
                                                col_1_idx[
                                                    col_idx, row_idx_0 * thread_extent_for_short + row_idx_1
                                                ]
                                            ]
                                        )

                                with T.block("spmv_short_write_back"):
                                    T.call_extern(
                                        val_dtype,
                                        "atomicAdd",
                                        T.call_intrin(
                                            "handle",
                                            "tir.address_of",
                                            res_val[
                                                buffer_1_idx[row_idx_0 * thread_extent_for_short + row_idx_1]
                                            ],
                                        ),
                                        res_local[0],
                                    )

        else:
            with T.block("outer"):
                for real_row_idx in range(row_num_real):
                    with T.block("spmv_init_2"):
                        v_real_row_idx = T.axis.remap("S", [real_row_idx])
                        res_val[v_real_row_idx] = T.float32(0)


    @R.function
    def optimized_spmv(
        indptr: R.Tensor(("origin_row_add_1_",), dtype="int32"),
        indices: R.Tensor(("origin_nnz_",), dtype="int32"),
        data: R.Tensor(("origin_nnz_",), dtype=val_dtype),
        vin: R.Tensor(("origin_row_",), dtype=val_dtype)
    ) -> R.Tensor(ndim=1, dtype=val_dtype):
        splited_long_row_num = T.int64()
        splited_short_row_num = T.int64()
        origin_row_ = T.int64()

        lv1: R.Tensor([2], "int32") = R.call_tir(
            OptimizedSpmv.calculate_buffer_size_cpu,
            (indptr,),
            R.Tensor([2], "int32"),
        )

        splited_num: R.Shape(ndim=2) = R.tensor_to_shape(lv1)
        _: R.Shape([splited_short_row_num, splited_long_row_num]) = R.match_cast(
            splited_num, R.Shape([splited_short_row_num, splited_long_row_num])
        )

        lv2 = R.call_tir(
            OptimizedSpmv.preprocess_cpu,
            (
                indptr,
                indices,
                data,
            ),
            [
                R.Tensor((splited_short_row_num,), "int32"),
                R.Tensor((short_align_val, splited_short_row_num), "int32"),
                R.Tensor((short_align_val, splited_short_row_num), val_dtype),
                R.Tensor((splited_long_row_num,), "int32"),
                R.Tensor((splited_long_row_num, long_align_val), "int32"),
                R.Tensor((splited_long_row_num, long_align_val), val_dtype),
            ],
        )

        output = R.call_tir(
            OptimizedSpmv.short_lines_easier_cpu,
            (lv2[2], lv2[1], lv2[0], vin),
            R.Tensor((origin_row_,), val_dtype),
        )
        output2 = R.call_tir_inplace(
            OptimizedSpmv.long_lines_cpu,
            (lv2[5], lv2[4], lv2[3], vin, output),
            [4],
            R.Tensor((origin_row_,), val_dtype),
        )

        return output2