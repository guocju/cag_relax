import os

# print(os.getpid())
import tvm
from tvm import relay
from tvm.script import tir as T
from tvm.relay import vm
import numpy as np
from tvm import runtime
from tvm.contrib import utils
import tvm.testing

from dataclasses import dataclass


@dataclass
class _config:
    t_thread_extent = 128
    b_thread_extent = 128


DUMP_FLIE = True
val_dtype = "float32"


def spmv_gpu_long_lines(thread_extent=_config.b_thread_extent):
    @T.prim_func
    def long_lines(
        value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle
    ) -> None:
        row_num_split = T.var("int32")
        col_num_split = T.var("int32")
        row_num_real = T.var("int32")
        col_num_real = T.var("int32")
        tx = T.env_thread("threadIdx.x")

        buffer_1_val = T.match_buffer(value_1, (row_num_split, col_num_split), val_dtype)
        col_1_idx = T.match_buffer(c_idx_1, (row_num_split, col_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        vec_val = T.match_buffer(vector_in, (col_num_real,), val_dtype)
        res_val = T.match_buffer(res_t, (row_num_real,), val_dtype)

        if row_num_split > 0:
            with T.block("outer"):
                for real_row_idx in range(row_num_real):
                    with T.block("spmv_init"):
                        v_real_row_idx = T.axis.remap("S", [real_row_idx])
                        res_val[v_real_row_idx] = T.float32(0)

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

                        T.launch_thread(tx, thread_extent)
                        with T.block("spmv_long_2"):
                            T.evaluate(T.tvm_storage_sync("shared"))
                            if tx == 0:
                                T.call_extern(
                                    val_dtype,
                                    "atomicAdd",
                                    T.call_intrin(
                                        "handle", "tir.address_of", res_val[buffer_1_idx[v_row_idx]]
                                    ),
                                    res_local[0],
                                )
        else:
            with T.block("outer"):
                for real_row_idx in range(row_num_real):
                    with T.block("spmv_init_2"):
                        v_real_row_idx = T.axis.remap("S", [real_row_idx])
                        res_val[v_real_row_idx] = T.float32(0)

    mod = tvm.IRModule.from_expr(long_lines)
    sch = tvm.tir.Schedule(mod)
    sch.work_on("long_lines")

    block_b = sch.get_block("spmv_init")
    [row] = sch.get_loops(block_b)
    row_oi, row_ii = sch.split(row, [None, 256])
    sch.bind(row_oi, "blockIdx.x")
    sch.bind(row_ii, "threadIdx.x")

    block_b = sch.get_block("spmv_init_2")
    [row] = sch.get_loops(block_b)
    row_oi, row_ii = sch.split(row, [None, 256])
    sch.bind(row_oi, "blockIdx.x")
    sch.bind(row_ii, "threadIdx.x")

    block_b = sch.get_block("spmv_long")
    [col] = sch.get_loops(block_b)
    _, col_oi = sch.split(col, [None, thread_extent])
    sch.bind(col_oi, "threadIdx.x")

    return sch.mod


def spmv_gpu_short_lines_easier(thread_extent=_config.t_thread_extent):
    @T.prim_func
    def short_lines_easier(
        value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle
    ) -> None:
        row_num_split = T.var("int32")
        col_num_split = T.var("int32")
        row_num_real = T.var("int32")
        col_num_real = T.var("int32")

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
                    T.ceildiv(row_num_split, thread_extent), thread="blockIdx.x"
                ):
                    for row_idx_1 in T.thread_binding(thread_extent, thread="threadIdx.x"):
                        with T.block("spmv_short_o"):
                            res_local = T.alloc_buffer(
                                [
                                    1,
                                ],
                                dtype=val_dtype,
                                scope="local",
                            )
                            res_local[0] = T.float32(0)
                            if row_idx_0 * thread_extent + row_idx_1 < row_num_split:
                                for col_idx in range(col_num_split):
                                    with T.block("spmv_short"):
                                        res_local[0] += (
                                            buffer_1_val[
                                                col_idx, row_idx_0 * thread_extent + row_idx_1
                                            ]
                                            * vec_val[
                                                col_1_idx[
                                                    col_idx, row_idx_0 * thread_extent + row_idx_1
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
                                                buffer_1_idx[row_idx_0 * thread_extent + row_idx_1]
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

    mod = tvm.IRModule.from_expr(short_lines_easier)
    sch = tvm.tir.Schedule(mod)
    sch.work_on("short_lines_easier")

    block_b = sch.get_block("spmv_init")
    [row] = sch.get_loops(block_b)
    row_oi, row_ii = sch.split(row, [None, 256])
    sch.bind(row_oi, "blockIdx.x")
    sch.bind(row_ii, "threadIdx.x")

    block_b = sch.get_block("spmv_init_2")
    [row] = sch.get_loops(block_b)
    row_oi, row_ii = sch.split(row, [None, 256])
    sch.bind(row_oi, "blockIdx.x")
    sch.bind(row_ii, "threadIdx.x")

    return sch.mod["short_lines_easier"]


def run_passes(mod):
    mod = tvm.driver.build_module.lower(mod)
    cuda_target = tvm.target.Target("cuda", host="llvm")
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
    return mod

def test_spmv_gpu_long_lines():
    mod = spmv_gpu_long_lines(128)
    mod = run_passes(mod)
    f = tvm.build(mod, target="cuda")
    ROW = 80
    COL = 80
    dev = tvm.cuda(0)
    value1 = tvm.nd.array(np.random.rand(ROW, COL).astype(val_dtype), dev)
    col_idx1 = tvm.nd.array(np.random.randint(0, ROW, (ROW, COL)).astype("int32"), dev)
    row_idx1 = tvm.nd.array(np.random.randint(0, ROW, (ROW,)).astype("int32"), dev)
    output = tvm.nd.array(np.random.rand(ROW).astype(val_dtype), dev)
    input = tvm.nd.array(np.random.rand(ROW).astype(val_dtype), dev)
    f(value1, col_idx1, row_idx1, input, output)
    out_ref = np.zeros((ROW), dtype=val_dtype)
    for row in range(ROW):
        for col in range(COL):
            out_ref[row_idx1.numpy()[row]] += (
                value1.numpy()[row, col] * input.numpy()[col_idx1.numpy()[row, col]]
            )
    tvm.testing.assert_allclose(output.numpy(), out_ref, rtol=1e-4, atol=1e-4)
    print("tir test pass!")


def test_spmv_gpu_short_easier_lines():
    with tvm.transform.PassContext(
        opt_level=0,
        config={"tir.disable_storage_rewrite": True},
        disabled_pass=["tir.LowerAutoCopy"],
    ):
        f = tvm.build(spmv_gpu_short_lines_easier(128), target="cuda")

    ROW = 80
    COL = 80
    dev = tvm.cuda(0)
    col_idx_np = np.random.randint(0, ROW, (ROW, COL)).astype("int32")
    col_idx_np_transpose = np.ascontiguousarray(col_idx_np.T)
    col_idx1 = tvm.nd.array(col_idx_np_transpose, dev)
    row_idx1 = tvm.nd.array(np.random.randint(0, ROW, (ROW,)).astype("int32"), dev)
    output = tvm.nd.array(np.random.rand(ROW).astype(val_dtype), dev)
    input = tvm.nd.array(np.random.rand(ROW).astype(val_dtype), dev)
    value_np = np.random.rand(ROW, COL).astype(val_dtype)
    value_np_transpose = np.ascontiguousarray(value_np.T)
    value1 = tvm.nd.array(value_np_transpose, dev)

    f(value1, col_idx1, row_idx1, input, output)
    out_ref = np.zeros((ROW), dtype=val_dtype)
    for row in range(ROW):
        for col in range(COL):
            out_ref[row_idx1.numpy()[row]] += (
                value_np[row, col] * input.numpy()[col_idx_np[row, col]]
            )
    tvm.testing.assert_allclose(output.numpy(), out_ref, rtol=1e-4, atol=1e-4)
    print("tir test pass!")


if __name__ == "__main__":
    test_spmv_gpu_long_lines()
    # test_spmv_gpu_short_easier_lines()
